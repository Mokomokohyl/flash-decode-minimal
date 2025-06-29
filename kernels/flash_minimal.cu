#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__
void forward_kernel(const c10::Half* Q, const c10::Half* K, const c10::Half* V, const int NQ, const int NKV, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, c10::Half* O, const c10::Half* mask) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index


    // Offset into Q,K,V,O,l,m - different for each batch and head
    int q_offset = (bx * gridDim.y * NQ * d) + (by * NQ * d);  // gridDim.y = nh
    int kv_offset = (bx * gridDim.y * NKV * d) + (by * NKV * d);
    int lm_offset = (bx * gridDim.y * NQ) + (by * NQ);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            if (Bc * j + tx < NKV) {
                Kj[(tx * d) + x] = static_cast<float>(K[kv_offset + (tile_size * j) + (tx * d) + x]);
                Vj[(tx * d) + x] = static_cast<float>(V[kv_offset + (tile_size * j) + (tx * d) + x]);
            }
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {
            int q_idx = i * Br + tx;
            if (q_idx >= NQ) continue;

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = static_cast<float>(Q[q_offset + q_idx * d + x]);
            }
            float row_m_prev = m[lm_offset + q_idx];
            float row_l_prev = l[lm_offset + q_idx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                int k_idx = j * Bc + y;
                float sum = -INFINITY; 
                
                if (k_idx < NKV) {
                    sum = 0;
                    for (int x = 0; x < d; x++) {
                        sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                    }
                    sum *= softmax_scale;
                    
                    if (mask != nullptr) {
                        int mask_idx = q_idx * NKV + k_idx;
                        sum += static_cast<float>(mask[mask_idx]);
                    }
                }
                S[(Bc * tx) + y] = sum;
                row_m = max(row_m, sum);
            }


            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                if (S[(Bc * tx) + y] == -INFINITY) {
                    S[(Bc * tx) + y] = 0.0f;
                }
                else {
                    S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                }
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                float o_val = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * static_cast<float>(O[q_offset + (tile_size * i) + (tx * d) + x])) \
                    + (__expf(row_m - row_m_new) * pv));
                // Convert float back to half for output
                O[q_offset + (tile_size * i) + (tx * d) + x] = static_cast<c10::Half>(o_val);
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 16; const int Br = 16;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int NQ = Q.size(2); const int d = Q.size(3);
    const int NKV = K.size(2);

    const int Tc = ceil((float) NKV / Bc); const int Tr = ceil((float) NQ / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, NQ}, torch::dtype(torch::kFloat32).device(Q.device()));
    auto m = torch::full({B, nh, NQ}, -INFINITY, torch::dtype(torch::kFloat32).device(Q.device()));

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    const bool has_mask = mask.numel() > 0;
    c10::Half* mask_ptr = has_mask ? mask.data_ptr<c10::Half>() : nullptr;

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<c10::Half>(), K.data_ptr<c10::Half>(), V.data_ptr<c10::Half>(),
        NQ, NKV, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<c10::Half>(), mask_ptr
    );
    return O;
}