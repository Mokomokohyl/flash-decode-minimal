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

// vec_t type for accelerating dtype cast and load to registers
template <size_t vec_size>
struct float_vec_t {
    union {
        float data[vec_size];
        struct { int4 i4_0, i4_1; };
    };
    __device__ __forceinline__ float& operator[](size_t i) { return ((float*)(data))[i]; }
    __device__ __forceinline__ const float& operator[](size_t i) const { return ((const float*)(data))[i]; }

    __device__ __forceinline__ void load_from_half8(c10::Half* smem_ptr) {
        static_assert(vec_size % 2 == 0, "vec_size must be even");
#pragma unroll
        for (int i = 0; i < vec_size; i += 2) {
            __half2 h2 = *reinterpret_cast<__half2*>(&smem_ptr[i]);
            float2 f2 = __half22float2(h2);
            data[i] = f2.x;
            data[i + 1] = f2.y;
        }
    }

    __device__ __forceinline__ void load_from_float8(float* smem_ptr) {
        static_assert(vec_size == 8, "vec_size must be 8 for the method");
        i4_0 = *reinterpret_cast<const int4*>(&smem_ptr[0]);
        i4_1 = *reinterpret_cast<const int4*>(&smem_ptr[4]);
    }

    __device__ __forceinline__ void store_to_half(c10::Half* mem_ptr) {
        static_assert(vec_size % 2 == 0, "vec_size must be even");
    #pragma unroll
        for (int i = 0; i < vec_size; i += 2) {
            float2 f2 = make_float2(data[i], data[i+1]);
            __half2 h2 = __float22half2_rn(f2);
            *reinterpret_cast<__half2*>(&mem_ptr[i]) = h2;
        }
    }
};

// wrapper of cp.async
template <bool FillZero>
__device__ __forceinline__ void cp_async_pred_load_128b(c10::Half* smem_ptr, const c10::Half* gmem_ptr, bool predicate) {
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    if constexpr (FillZero == true) {
        int src_in_bytes = predicate ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                    "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    } else {
        asm volatile(
            "{\n"
            " .reg .pred p;\n"
            " setp.ne.b32 p, %0, 0;\n"
            " @p cp.async.cg.shared.global [%1], [%2], %3;\n"
            "}\n" ::"r"((int)predicate),
            "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    }
}

// wrapper of cp.async.commit_group
__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// wrapper of cp.async.wait_group
template <size_t n>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

template <size_t vec_size>
__global__
void forward_kernel(const c10::Half* Q, const c10::Half* K, const c10::Half* V, const int NQ, const int NKV, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    c10::Half* O, const c10::Half* mask, const size_t num_stages_smem) {
    int tx = threadIdx.x; // head_dim
    int ty = threadIdx.y; // a token of Q
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index


    // Offset into Q,K,V,O,l,m - different for each batch and head
    int q_offset = (bx * gridDim.y * NQ * d) + (by * NQ * d);  // gridDim.y = nh
    int kv_offset = (bx * gridDim.y * NKV * d) + (by * NKV * d);

    // Register for q, k, v vec
    float_vec_t<vec_size> q_vec, k_vec, v_vec, o_vec;
    // Define SRAM for Q,K,V,S
    extern __shared__ uint32_t sram[];
    float* Qi = reinterpret_cast<float*>(sram);
    c10::Half* Kj = reinterpret_cast<c10::Half*>(sram + Br * d);
    c10::Half* Vj = reinterpret_cast<c10::Half*>(sram + Br * d + Bc * d);
    float* S = reinterpret_cast<float*>(sram + Br * d + 2 * Bc * d);

    for (int i = 0; i < Tr; i++) {
        // Load Qi from HBM, init Oi[Br, vec_size] = 0, l = 0, m = -INF
        int q_idx = i * Br + ty;
        if (q_idx < NQ && ty < Br) {
#pragma unroll
            for (int k = 0; k < vec_size; k++) {
                Qi[ty * d + tx * vec_size + k] = static_cast<float>(Q[q_offset + q_idx * d + tx * vec_size + k]);
                o_vec[k] = 0.0f;
            }
        }
        float l_local = 0.0f;
        float m_local = -INFINITY;

        int stage_idx = 0;
        int producer_kv_idx = 0;
        int consumer_kv_idx = 0;
        //prelogue for pipeline. fill data for K1, V1, K2, V2
        for (int j = 0; j < num_stages_smem; j++) {
            // load k
            cp_async_pred_load_128b<true>( /* fill zero for k, v */
                Kj + stage_idx * Bc * d + ty * d + tx * vec_size, 
                K + kv_offset + (producer_kv_idx + ty) * d + tx * vec_size,
                (producer_kv_idx + ty) < NKV
            );
            cp_async_commit_group();
            cp_async_pred_load_128b<true>(
                Vj + stage_idx * Bc * d + ty * d + tx * vec_size,
                V + kv_offset + (producer_kv_idx + ty) * d + tx * vec_size,
                (producer_kv_idx + ty) < NKV
            );
            cp_async_commit_group();
            producer_kv_idx += Bc;
            stage_idx ^= 1;
        }

        // pipeline
#pragma unroll 2
        for (int j = 0; j < Tc; j++) {
            float m_prev = m_local;
            float sum = 0.0f;
            cp_async_wait_group<3>(); // wait for last k load
            __syncthreads();

            // Compute Sij = q @ kT ([Br, Bc]) (use k)
            for (int y = 0; y < Bc; y++) {
                sum = 0;
                bool pos_valid = (q_idx < NQ) && (ty < Br) && ((consumer_kv_idx + y) < NKV);
                if (pos_valid) {
                    q_vec.load_from_float8(Qi + ty * d + tx * vec_size);
                    k_vec.load_from_half8(Kj + stage_idx * Bc * d + y * d + tx * vec_size);
#pragma unroll
                    for (int x = 0; x < vec_size; x++) {
                        sum += q_vec[x] * k_vec[x];
                    }
                }
#pragma unroll
                for (int offset = 8; offset > 0; offset /= 2) {
                    sum += shfl_xor_sync(sum, offset);
                }
                sum *= softmax_scale;
                if (pos_valid) {
                    if (mask != nullptr) {
                        int mask_idx = q_idx * NKV + consumer_kv_idx * Bc + y;
                        S[ty * Bc + y] = sum + static_cast<float>(mask[mask_idx]);
                    } else {
                        S[ty * Bc + y] = sum;
                    }
                    m_local = fmaxf(S[ty * Bc + y], m_local);
                } 
            }

            __syncthreads();

            // load k
            cp_async_pred_load_128b<true>( /* fill zero for k, v */
                Kj + stage_idx * Bc * d + ty * d + tx * vec_size,
                K + kv_offset + (producer_kv_idx + ty) * d + tx * vec_size,
                (producer_kv_idx + ty) < NKV
            );
            cp_async_commit_group();

            float row_l = 0.0f; // row_l = rowsum(P)
            for (int y = 0; y < Bc; y++) {
                if ((q_idx < NQ) && (ty < Br) && ((consumer_kv_idx + y) < NKV)) {
                    S[ty * Bc + y] = ptx_exp2(S[ty * Bc + y] - m_local);
                    row_l += S[ty * Bc + y];
                }
            }
            float o_scale = ptx_exp2(m_prev - m_local);
            l_local = o_scale * l_local + row_l;

            // scale Oi, then add PV
            if (q_idx < NQ && ty < Br) {
#pragma unroll
                for (int k = 0; k < vec_size; k++) {
                    o_vec[k] = o_scale * o_vec[k];
                }
            }

            cp_async_wait_group<3>(); // wait for last v load
            __syncthreads();

            // use v
            // the outer loop iterate over KV tokens(seq_len + cache_len dim)
            // the inner loop can be parallelized over head_dim
            for (int y = 0; y < Bc; y++) {
                if (ty < Br && q_idx < NQ) {
                    v_vec.load_from_half8(Vj + stage_idx * Bc * d + y * d + tx * vec_size);
#pragma unroll
                    for (int k = 0; k < vec_size; k++) {
                        o_vec[k] += S[ty * Bc + y] * v_vec[k];
                    }
                }
            }
            __syncthreads();

            // load v
            cp_async_pred_load_128b<true>(
                Vj + stage_idx * Bc * d + ty * d + tx * vec_size,
                V + kv_offset + (producer_kv_idx + ty) * d + tx * vec_size,
                (producer_kv_idx + ty) < NKV
            );
            cp_async_commit_group();

            // shift stage_idx to use correct buffer
            stage_idx ^= 1;
            producer_kv_idx += Bc;
            consumer_kv_idx += Bc;
        }

        cp_async_wait_group<0>();
        __syncthreads();
        if (q_idx < NQ && ty < Br) {
            // Write Oi to O in HBM
#pragma unroll
            for (int k = 0; k < vec_size; k++) {
                o_vec[k] = (1 / l_local) * o_vec[k];
            }
            o_vec.store_to_half(O + q_offset + q_idx * d + tx * vec_size);
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
    const size_t num_stages_smem = 2; // double buffer for pipeline
    // fix Bc to be 16 make llama talk more.
    const int Bc = (NQ == 1) ? ceil(M / (3 * d)) : ceil(M / (4 * d)); const int Br = (NQ == 1) ? 1 : min(Bc, d);
    const int sram_size = (Br * d * sizeof(float)) + (2 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    const int Tr = ceil((float) NQ / Br);
    const int Tc = ceil((float) NKV / Bc);
    const float softmax_scale = log2e * 1.0 / sqrt(d);
    const size_t vec_size = 8;

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

    forward_kernel<vec_size><<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<c10::Half>(), K.data_ptr<c10::Half>(), V.data_ptr<c10::Half>(),
        NQ, NKV, d, Tc, Tr, Bc, Br, softmax_scale,
        O.data_ptr<c10::Half>(), mask_ptr, num_stages_smem
    );
    return O;
}