#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// add log2e to accelerate softmax
constexpr float log2e = 1.44269504088896340736f;

__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

__global__
void forward_kernel(const c10::Half* Q, const c10::Half* K, const c10::Half* V, const int NQ, const int NKV, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, c10::Half* O, const c10::Half* mask) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
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
        int k_idx = Bc * j + ty;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (Bc * j + ty < NKV) {
                Kj[(ty * d) + tx * 4 + k] = static_cast<float>(K[kv_offset + k_idx * d + tx * 4 + k]);
                Vj[(ty * d) + tx * 4 + k] = static_cast<float>(V[kv_offset + k_idx * d + tx * 4 + k]);
            } else {
                Kj[(ty * d) + tx * 4 + k] = 0.0f;
                Vj[(ty * d) + tx * 4 + k] = 0.0f;
            }
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {
            // Each warp assigned a row of Q. Br <= Bc.
            int q_idx = i * Br + ty;
            if (ty < Br && q_idx < NQ) {
                // Load Qi to SRAM, l and m to registers
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    Qi[(ty * d) + tx * 4 + k] = static_cast<float>(Q[q_offset + q_idx * d + tx * 4 + k]);
                }

                float row_m_prev = m[lm_offset + q_idx];
                float row_l_prev = l[lm_offset + q_idx];

                // S = QK^T, row_m = rowmax(S)
                float row_m = -INFINITY;
                for (int y = 0; y < Bc; y++) {

                    float sum = -INFINITY; 
                    sum = 0;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        sum += Qi[(ty * d) + tx * 4 + k] * Kj[(y * d) + tx * 4 + k];
                    }
                    for (int offset = 16; offset > 0; offset /= 2) {
                        sum += shfl_xor_sync(sum, offset);
                    }
                    sum *= softmax_scale;
                    
                    if (mask != nullptr) {
                        int mask_idx = q_idx * NKV + (j * Bc + y);
                        sum += static_cast<float>(mask[mask_idx]);
                    }
                    S[(Bc * ty) + y] = (j * Bc + y) < NKV ? sum : -INFINITY;
                    row_m = fmaxf(row_m, sum);
                }


                // P = exp(S - row_m), row_l = rowsum(P)
                float row_l = 0;
                for (int y = 0; y < Bc; y++) {
                    if ((j * Bc + y) < NKV) {
                        S[(Bc * ty) + y] = ptx_exp2(S[(Bc * ty) + y] - row_m);
                        row_l += S[(Bc * ty) + y];
                    } else {
                        S[(Bc * ty) + y] = 0.0f;
                    }
                }

                // Compute new m and l
                float row_m_new = max(row_m_prev, row_m);
                float row_l_new = (ptx_exp2(row_m_prev - row_m_new) * row_l_prev) + (ptx_exp2(row_m - row_m_new) * row_l);

                // Write O, l, m to HBM
                for (int x = 0; x < d; x++) {
                    float pv = 0;  // Pij * Vj
                    for (int y = 0; y < Bc; y++) {
                        if ((j * Bc + y < NKV)) {
                            pv += S[(Bc * ty) + y] * Vj[(y * d) + x];
                        }
                    }
                    O[q_offset + q_idx * d + x] = static_cast<c10::Half>((1 / row_l_new) \
                        * ((row_l_prev * ptx_exp2(row_m_prev - row_m_new) * O[q_offset + q_idx * d + x]) \
                        + (ptx_exp2(row_m - row_m_new) * pv)));
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

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    float M = (float)max_sram_size / sizeof(float);

    // Calculate SRAM size needed per block
    const int Bc = ceil(M / (4 * d)); const int Br = min(Bc, d);
    const int sram_size = (Br * d * sizeof(float)) + (2 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    const int Tr = ceil((float) NQ / Br);
    const int Tc = ceil((float) NKV / Bc);
    const float softmax_scale = log2e * 1.0 / sqrt(d);


    // Initialize O, l, m to HBM
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, NQ}, options);
    auto m = torch::full({B, nh, NQ}, -INFINITY, options);


    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(32, Bc);  // Bc threads per block

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