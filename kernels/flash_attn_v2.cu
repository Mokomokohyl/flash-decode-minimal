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
                    c10::Half* O, const c10::Half* mask) {
    int tx = threadIdx.x; // head_dim
    int ty = threadIdx.y; // a token of Q
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index


    // Offset into Q,K,V,O,l,m - different for each batch and head
    int q_offset = (bx * gridDim.y * NQ * d) + (by * NQ * d);  // gridDim.y = nh
    int kv_offset = (bx * gridDim.y * NKV * d) + (by * NKV * d);

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    float* Qi = sram;
    float* Oi = &sram[Br * d];
    float* Kj = &sram[2 * Br * d];
    float* Vj = &sram[2 * Br * d + Bc * d];
    float* S = &sram[2 * Br * d + 2 * Bc * d];

    for (int i = 0; i < Tr; i++) {
        // Load Qi from HBM, init Oi[Br, d] = 0, l = 0, m = -INF
        int q_idx = i * Br + ty;
        if (q_idx < NQ && ty < Br) {
#pragma unroll
            for (int k = 0; k < 8; k++) {
                Qi[ty * d + tx * 8 + k] = static_cast<float>(Q[q_offset + q_idx * d + tx * 8 + k]);
                Oi[ty * d + tx * 8 + k] = 0.0f;
            }
        }
        float l_local = 0.0f;
        float m_local = -INFINITY;

        for (int j = 0; j < Tc; j++) {
            // Load Kj, Vj from HBM
            int kv_idx = j * Bc + ty;
            if (kv_idx < NKV) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    Kj[ty * d + tx * 8 + k] = static_cast<float>(K[kv_offset + kv_idx * d + tx * 8 + k]);
                    Vj[ty * d + tx * 8 + k] = static_cast<float>(V[kv_offset + kv_idx * d + tx * 8 + k]);
                }
            } else {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    Kj[ty * d + tx * 8 + k] = 0.0f;
                    Vj[ty * d + tx * 8 + k] = 0.0f;
                }
            }
            __syncthreads();

            float m_prev = m_local;
            float sum = 0.0f;
            // Compute Sij = q @ kT ([Br, Bc])
            for (int y = 0; y < Bc; y++) {
                sum = 0;
                bool pos_valid = (q_idx < NQ) && (ty < Br) && ((j * Bc + y) < NKV);
                if (pos_valid) {
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        sum += Qi[ty * d + tx * 8 + k] * Kj[y * d + tx * 8 + k];
                    }
                }
#pragma unroll
                for (int offset = 8; offset > 0; offset /= 2) {
                    sum += shfl_xor_sync(sum, offset);
                }
                sum *= softmax_scale;
                if (pos_valid) {
                    if (mask != nullptr) {
                        int mask_idx = q_idx * NKV + j * Bc + y;
                        S[ty * Bc + y] = sum + static_cast<float>(mask[mask_idx]);
                    } else {
                        S[ty * Bc + y] = sum;
                    }
                    m_local = fmaxf(S[ty * Bc + y], m_local);
                } 
            }


            float row_l = 0.0f; // row_l = rowsum(P)
            for (int y = 0; y < Bc; y++) {
                if ((q_idx < NQ) && (ty < Br) && ((j * Bc + y) < NKV)) {
                    S[ty * Bc + y] = ptx_exp2(S[ty * Bc + y] - m_local);
                    row_l += S[ty * Bc + y];
                }
            }
            float o_scale = ptx_exp2(m_prev - m_local);
            l_local = o_scale * l_local + row_l;

            // scale Oi, then add PV
            if (q_idx < NQ && ty < Br) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    Oi[ty * d + tx * 8 + k] = o_scale * Oi[ty * d + tx * 8 + k];
                }
            }

            // the outer loop iterate over KV tokens(seq_len + cache_len dim)
            // the inner loop can be parallelized over head_dim
            for (int y = 0; y < Bc; y++) {
                if (ty < Br && q_idx < NQ) {
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        Oi[ty * d + tx * 8 + k] += S[ty * Bc + y] * Vj[y * d + tx * 8 + k];
                    }
                }
            }
            __syncthreads();
        }

        if (q_idx < NQ && ty < Br) {
            // Write Oi to O in HBM
#pragma unroll
            for (int k = 0; k < 8; k++) {
                O[q_offset + q_idx * d + tx * 8 + k] = static_cast<c10::Half>((1 / l_local) * Oi[ty * d + tx * 8 + k]);
            }
        }
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
    const int Bc = (NQ == 1) ? ceil(M / (3 * d)) : ceil(M / (5 * d)); const int Br = (NQ == 1) ? 1 : min(Bc, d);
    const int sram_size = (2 * Br * d * sizeof(float)) + (2 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    const int Tr = ceil((float) NQ / Br);
    const int Tc = ceil((float) NKV / Bc);
    const float softmax_scale = log2e * 1.0 / sqrt(d);


    // Initialize O to HBM
    auto O = torch::zeros_like(Q);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(16, Bc);  // Bc threads per block

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
        O.data_ptr<c10::Half>(), mask_ptr
    );
    return O;
}