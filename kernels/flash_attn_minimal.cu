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
    float* Qi = sram;
    float* Kj = &sram[Br * d];
    float* Vj = &sram[Br * d + Bc * d];
    float* S = &sram[Br * d + 2 * Bc * d];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        // Each thread loads a row of K, V ([N, d])
        int k_idx = Bc * j + tx;
        for (int x = 0; x < d; x++) {
            if (Bc * j + tx < NKV) {
                Kj[(tx * d) + x] = static_cast<float>(K[kv_offset + k_idx * d + x]);
                Vj[(tx * d) + x] = static_cast<float>(V[kv_offset + k_idx * d + x]);
            } else {
                Kj[(tx * d) + x] = 0.0f;
                Vj[(tx * d) + x] = 0.0f;
            }
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {
            // Each thread assigned a row of Q. Br <= Bc, there might be more threads than Br.
            int q_idx = i * Br + tx;
            if (tx < Br && q_idx < NQ) {
                // Load Qi to SRAM, l and m to registers
                for (int x = 0; x < d; x++) {
                    Qi[(tx * d) + x] = static_cast<float>(Q[q_offset + q_idx * d + x]);
                }

                float row_m_prev = m[lm_offset + q_idx];
                float row_l_prev = l[lm_offset + q_idx];

                // S = QK^T, row_m = rowmax(S)
                float row_m = -INFINITY;
                for (int y = 0; y < Bc; y++) {

                    float sum = -INFINITY; 
                    if ((j * Bc + y) < NKV) {
                        sum = 0;
                        for (int x = 0; x < d; x++) {
                            sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                        }
                        sum *= softmax_scale;
                        
                        if (mask != nullptr) {
                            int mask_idx = q_idx * NKV + (j * Bc + y);
                            sum += static_cast<float>(mask[mask_idx]);
                        }
                    }
                    S[(Bc * tx) + y] = sum;
                    row_m = fmaxf(row_m, sum);
                }


                // P = exp(S - row_m), row_l = rowsum(P)
                float row_l = 0;
                for (int y = 0; y < Bc; y++) {
                    if ((j * Bc + y) < NKV) {
                        S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                        row_l += S[(Bc * tx) + y];
                    } else {
                        S[(Bc * tx) + y] = 0.0f;
                    }
                }

                // Compute new m and l
                float row_m_new = max(row_m_prev, row_m);
                float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

                // Write O, l, m to HBM
                for (int x = 0; x < d; x++) {
                    float pv = 0;  // Pij * Vj
                    for (int y = 0; y < Bc; y++) {
                        if ((j * Bc + y < NKV)) {
                            pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                        }
                    }
                    float o_val = (1 / row_l_new) \
                        * ((row_l_prev * __expf(row_m_prev - row_m_new) * static_cast<float>(O[q_offset + q_idx * d + x])) \
                        + (__expf(row_m - row_m_new) * pv));

                    // Convert float back to half for output
                    O[q_offset + q_idx * d + x] = static_cast<c10::Half>(o_val);
                }
                m[lm_offset + q_idx] = row_m_new;
                l[lm_offset + q_idx] = row_l_new;
            }
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask) {
    const int B = Q.size(0); const int nh = Q.size(1);
    const int NQ = Q.size(2); const int d = Q.size(3);
    const int NKV = K.size(2);

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

#ifdef DEBUG
    printf("Q.contiguous() time: %.3f ms\n", q_contiguous_time);
    printf("K.contiguous() time: %.3f ms\n", k_contiguous_time);
    printf("V.contiguous() time: %.3f ms\n", v_contiguous_time);
    printf("Total contiguous time: %.3f ms\n", q_contiguous_time + k_contiguous_time + v_contiguous_time);
#endif

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    float M = (float)max_sram_size / sizeof(float);

    // Calculate SRAM size needed per block
    const int Bc = ceil(M / (4 * d)); const int Br = min(Bc, d);
    const int sram_size = (Br * d * sizeof(float)) + (2 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    const int Tr = ceil((float) NQ / Br);
    const int Tc = ceil((float) NKV / Bc);
    const float softmax_scale = 1.0 / sqrt(d);


    // Initialize O, l, m to HBM
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, NQ}, options);
    auto m = torch::full({B, nh, NQ}, -INFINITY, options);


    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    const bool has_mask = mask.numel() > 0;
    c10::Half* mask_ptr = has_mask ? mask.data_ptr<c10::Half>() : nullptr;

#ifdef DEBUG
    if (has_mask) {
        printf("with mask\n");
    } else {
        printf("no mask\n");
    }
    printf("B: %d, nh: %d, NQ: %d, d: %d, NKV: %d", B, nh, NQ, d, NKV);
    printf("Max sram size: %d\n", max_sram_size);
    printf("Bc: %d, Br: %d, NQ: %d, NKV: %d, Tr: %d, Tc: %d\n", Bc, Br, NQ, NKV, Tr, Tc);
    printf("Required Sram size: %d\n", sram_size);
#endif

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<c10::Half>(), K.data_ptr<c10::Half>(), V.data_ptr<c10::Half>(),
        NQ, NKV, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<c10::Half>(), mask_ptr
    );
    return O;
}