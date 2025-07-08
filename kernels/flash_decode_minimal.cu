#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// add log2e to accelerate softmax exp
constexpr float log2e = 1.44269504088896340736f;

// calculate 2^x. better than __expf()
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// wrapper of shfl.sync.bfly.b32 ptx inst for reduction in a warp
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

/***** cp_async wrappers to build pipeline *****/

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

/***** vec_t type for accelerating dtype cast and load to registers *****/
template <size_t vec_size>
struct float_vec_t {
    union {
        float data[vec_size];
        struct { int4 i4_0, i4_1; };
    };
    __device__ __forceinline__ float& operator[](size_t i) { return ((float*)(data))[i]; }
    __device__ __forceinline__ const float& operator[](size_t i) const { return ((const float*)(data))[i]; }

    __device__ __forceinline__ void load_from_half8(const c10::Half* smem_ptr) {
        static_assert(vec_size % 2 == 0, "vec_size must be even");
#pragma unroll
        for (int i = 0; i < vec_size; i += 2) {
            __half2 h2 = *reinterpret_cast<const __half2*>(&smem_ptr[i]);
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

    __device__ __forceinline__ void store_to_float(float* memptr) {
        static_assert(vec_size == 8, "vec_size must be 8 for the method");
        *reinterpret_cast<int4*>(&memptr[0]) = i4_0;
        *reinterpret_cast<int4*>(&memptr[4]) = i4_1;
    }
};


/***** state_t struct to store flash attn states *****/
template<size_t vec_size>
struct state_t {
    float l; // the rowsum of exp(S - m)
    float m; // current max
    float_vec_t<vec_size> o_vec; // a piece of output O

    // init st values
    __device__ __forceinline__ void init() {
        l = 1.0f;
        m = -INFINITY;
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
            o_vec[i] = 0.0f;
        }
    }

    // the core of flash attention algorithm
    __device__ __forceinline__ void merge(const float_vec_t<vec_size>& o_vec_other, float l_other, float m_other) {
        float m_prev = m;
        m = fmaxf(m_prev, m_other);
        l = l_other * ptx_exp2(m_other - m) + l * ptx_exp2(m_prev - m);
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
            o_vec[i] = o_vec[i] * ptx_exp2(m_prev - m) + o_vec_other[i] * ptx_exp2(m_other - m);
        }
    }

    __device__ __forceinline__ void normalize() {
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
            o_vec[i] = __fdividef(o_vec[i], l);
        }
    }
};

__host__ __device__
struct TensorStrides {
    long b, h, n, d; // batch, head, sequence, dimension
};

// same as flash_attn_v2.cu. removed mask_ptr check
template <size_t vec_size>
__global__
void prefill_kernel(const c10::Half* Q, const c10::Half* K, const c10::Half* V, const int NQ, const int NKV, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    c10::Half* O, const c10::Half* mask, const size_t num_stages_smem, TensorStrides q_stride, TensorStrides kv_stride) {
    int tx = threadIdx.x; // head_dim
    int ty = threadIdx.y; // a token of Q
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index


    // Offset into Q,K,V,O,l,m - different for each batch and head
    int q_offset = (bx * q_stride.b) + (by * q_stride.h);  // gridDim.y = nh
    int kv_offset = (bx * kv_stride.b) + (by * kv_stride.h);

    // Register for q, k, v vec, state_t
    float_vec_t<vec_size> q_vec, k_vec, v_vec;
    state_t<vec_size> st;
    // Define SRAM for Q,K,V,S
    extern __shared__ uint32_t sram[];
    float* Qi = reinterpret_cast<float*>(sram);
    c10::Half* Kj = reinterpret_cast<c10::Half*>(sram + Br * d);
    c10::Half* Vj = reinterpret_cast<c10::Half*>(sram + Br * d + Bc * d);
    float* S = reinterpret_cast<float*>(sram + Br * d + 2 * Bc * d);

    for (int i = 0; i < Tr; i++) {
        // Load Qi from HBM, init Oi[Br, d] = 0, l = 0, m = -INF
        int q_idx = i * Br + ty;
        if (q_idx < NQ && ty < Br) {
#pragma unroll
            for (int k = 0; k < vec_size; k++) {
                Qi[ty * d + tx * vec_size + k] = static_cast<float>(Q[q_offset + q_idx * q_stride.n + tx * vec_size + k]);
            }
        }
        st.init();

        int stage_idx = 0;
        int producer_kv_idx = 0;
        int consumer_kv_idx = 0;
        //prelogue for pipeline
        for (int j = 0; j < num_stages_smem; j++) {
            // load k
            cp_async_pred_load_128b<true>( /* fill zero for k, v */
                Kj + stage_idx * Bc * d + ty * d + tx * vec_size, 
                K + kv_offset + (producer_kv_idx + ty) * kv_stride.n + tx * vec_size,
                (producer_kv_idx + ty) < NKV
            );
            cp_async_commit_group();
            cp_async_pred_load_128b<true>(
                Vj + stage_idx * Bc * d + ty * d + tx * vec_size,
                V + kv_offset + (producer_kv_idx + ty) * kv_stride.n + tx * vec_size,
                (producer_kv_idx + ty) < NKV
            );
            cp_async_commit_group();
            producer_kv_idx += Bc;
            stage_idx ^= 1;
        }

        // pipeline
        for (int j = 0; j < Tc; j++) {
            float m_prev = st.m;
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
                    int mask_idx = q_idx * NKV + consumer_kv_idx + y;
                    S[ty * Bc + y] = sum + static_cast<float>(mask[mask_idx]);
                    st.m = fmaxf(S[ty * Bc + y], st.m);
                } 
            }

            __syncthreads();

            // load k
            cp_async_pred_load_128b<true>( /* fill zero for k, v */
                Kj + stage_idx * Bc * d + ty * d + tx * vec_size,
                K + kv_offset + (producer_kv_idx + ty) * kv_stride.n + tx * vec_size,
                (producer_kv_idx + ty) < NKV
            );
            cp_async_commit_group();

            float row_l = 0.0f; // row_l = rowsum(P)
            for (int y = 0; y < Bc; y++) {
                if ((q_idx < NQ) && (ty < Br) && ((consumer_kv_idx + y) < NKV)) {
                    S[ty * Bc + y] = ptx_exp2(S[ty * Bc + y] - st.m);
                    row_l += S[ty * Bc + y];
                }
            }
            float o_scale = ptx_exp2(m_prev - st.m);
            st.l = o_scale * st.l + row_l;

            // scale Oi, then add PV
            if (q_idx < NQ && ty < Br) {
#pragma unroll
                for (int k = 0; k < vec_size; k++) {
                    st.o_vec[k] = o_scale * st.o_vec[k];
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
                        st.o_vec[k] += S[ty * Bc + y] * v_vec[k];
                    }
                }
            }
            __syncthreads();

            // load v
            cp_async_pred_load_128b<true>(
                Vj + stage_idx * Bc * d + ty * d + tx * vec_size,
                V + kv_offset + (producer_kv_idx + ty) * kv_stride.n + tx * vec_size,
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
                st.o_vec[k] = (1 / st.l) * st.o_vec[k];
            }
            st.o_vec.store_to_half(O + ((bx * gridDim.y * NQ * d) + (by * NQ * d)) + q_idx * d + tx * vec_size);
        }
    }

}

// flash-decode minimal kernel
template <size_t vec_size, size_t tile_size_per_bdx, size_t num_stages_smem>
__global__
void decode_kernel(const c10::Half* Q, const c10::Half* K, const c10::Half* V, const int NQ, const int NKV, const int d,
                    const int Tc, const int Bc, const float softmax_scale,
                    c10::Half* O, const int bdx, const int bdy, TensorStrides q_stride, TensorStrides kv_stride) {
    int tx = threadIdx.x; // head_dim
    int ty = threadIdx.y; // in decode kernel, use ty to devide Bc into bdy * tile_size_per_bdx.
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int q_offset = (bx * q_stride.b) + (by * q_stride.h);  // gridDim.y = nh
    int kv_offset = (bx * kv_stride.b) + (by * kv_stride.h);

    // Register for q, k, v vec
    float_vec_t<vec_size> q_vec, k_vec, v_vec;
    state_t<vec_size> st;
    // Define SRAM for Q,K,V,S
    extern __shared__ uint32_t sram[];
    float* Oi = reinterpret_cast<float*>(sram); // in decode we can directly store Q in q_vec, so the space is for O
    c10::Half* Kj = reinterpret_cast<c10::Half*>(sram + bdy * d);
    c10::Half* Vj = reinterpret_cast<c10::Half*>(sram + bdy * d + Bc * d);
    float* S = reinterpret_cast<float*>(sram + bdy * d + 2 * Bc * d);

    // Load Qi from HBM, init o_vec = 0, l = 0, m = -INF
    q_vec.load_from_half8(Q + q_offset + tx * vec_size);
    st.init();

    int stage_idx = 0;
    int producer_kv_idx = 0;
    int consumer_kv_idx = 0;
    //prelogue for pipeline
#pragma unroll
    for (int i = 0; i < num_stages_smem; i++) {
#pragma unroll
        for (int j = 0; j < tile_size_per_bdx; j++) {
            int cur_kv_token = producer_kv_idx + ty * tile_size_per_bdx + j;
            // load k
            cp_async_pred_load_128b<true>( /* fill zero for k, v */
                Kj + stage_idx * Bc * d + (ty * tile_size_per_bdx + j) * d + tx * vec_size, 
                K + kv_offset + cur_kv_token * kv_stride.n + tx * vec_size,
                cur_kv_token < NKV
            );
        }
        cp_async_commit_group();
        for (int j = 0; j < tile_size_per_bdx; j++) {
            int cur_kv_token = producer_kv_idx + ty * tile_size_per_bdx + j;
            cp_async_pred_load_128b<true>(
                Vj + stage_idx * Bc * d + (ty * tile_size_per_bdx + j) * d + tx * vec_size,
                V + kv_offset + cur_kv_token * kv_stride.n + tx * vec_size,
                cur_kv_token < NKV
            );
        }
        cp_async_commit_group();
        producer_kv_idx += Bc; // Bc = bdy * tile_size_per_bdx
        stage_idx ^= 1;
    }

    // pipeline
#pragma unroll 2
    for (int j = 0; j < Tc; j++) {
        float m_prev = st.m;
        float sum = 0.0f;
        cp_async_wait_group<2 * num_stages_smem - 1>(); // wait for last k load
        __syncthreads();

        // use k. Compute Sij = q @ kT ([Br, Bc]) and update m
#pragma unroll
        for (int y = 0; y < tile_size_per_bdx; y++) {
            sum = 0;
            bool pos_valid = ((consumer_kv_idx + ty * tile_size_per_bdx + y) < NKV);
            if (pos_valid) {
                k_vec.load_from_half8(Kj + stage_idx * Bc * d + (ty * tile_size_per_bdx + y) * d + tx * vec_size);
#pragma unroll
                for (int x = 0; x < vec_size; x++) {
                    sum += q_vec[x] * k_vec[x];
                }
            }
#pragma unroll
            for (int offset = bdx / 2; offset > 0; offset /= 2) {
                sum += shfl_xor_sync(sum, offset);
            }
            sum *= softmax_scale;
            if (pos_valid) {
                S[ty * tile_size_per_bdx + y] = sum;
                st.m = fmaxf(sum, st.m);
            } 
        }

        __syncthreads();

        // load k
#pragma unroll
        for (int j = 0; j < tile_size_per_bdx; j++) {
            int cur_kv_token = producer_kv_idx + ty * tile_size_per_bdx + j;
            cp_async_pred_load_128b<true>( /* fill zero for k, v */
                Kj + stage_idx * Bc * d + (ty * tile_size_per_bdx + j) * d + tx * vec_size,
                K + kv_offset + cur_kv_token * kv_stride.n + tx * vec_size,
                cur_kv_token < NKV
            );
        }
        cp_async_commit_group();

        // update l
        float row_l = 0.0f; // row_l = rowsum(P)
#pragma unroll
        for (int y = 0; y < tile_size_per_bdx; y++) {
            if ((consumer_kv_idx + ty * tile_size_per_bdx + y) < NKV) {
                S[ty * tile_size_per_bdx + y] = ptx_exp2(S[ty * tile_size_per_bdx + y] - st.m);
                row_l += S[ty * tile_size_per_bdx + y];
            }
        }
        float o_scale = ptx_exp2(m_prev - st.m);
        st.l = o_scale * st.l + row_l;

        // scale Oi, then add PV
#pragma unroll
        for (int k = 0; k < vec_size; k++) {
            st.o_vec[k] = o_scale * st.o_vec[k];
        }

        cp_async_wait_group<2 * num_stages_smem - 1>(); // wait for last v load
        __syncthreads();

        // use v
        // the outer loop iterate over tile_size_per_bdx KV tokens
        // the inner loop can be parallelized over head_dim
#pragma unroll
        for (int y = 0; y < tile_size_per_bdx; y++) {
            v_vec.load_from_half8(Vj + stage_idx * Bc * d + (ty * tile_size_per_bdx + y) * d + tx * vec_size);
#pragma unroll
            for (int k = 0; k < vec_size; k++) {
                st.o_vec[k] += S[ty * tile_size_per_bdx + y] * v_vec[k];
            }
        }
        __syncthreads();

        // load v
#pragma unroll
        for (int y = 0; y < tile_size_per_bdx; y++) {
            int cur_kv_token = producer_kv_idx + ty * tile_size_per_bdx + y;
            cp_async_pred_load_128b<true>(
                Vj + stage_idx * Bc * d + (ty * tile_size_per_bdx + y) * d + tx * vec_size,
                V + kv_offset + cur_kv_token * kv_stride.n + tx * vec_size,
                cur_kv_token < NKV
            );
        }
        cp_async_commit_group();

        // shift stage_idx to use correct buffer
        stage_idx ^= 1;
        producer_kv_idx += Bc;
        consumer_kv_idx += Bc;
    }

    cp_async_wait_group<0>();
    __syncthreads();

    // Merge states across ty (0,..., bdy - 1) to get final Oi
    // write current state_t register contents to shared memroy
    float *smem_lm = Oi + bdy * d;
    smem_lm[ty * 2] = st.l;
    smem_lm[ty * 2 + 1] = st.m;
    st.o_vec.store_to_float(Oi + ty * d + tx * vec_size);
    __syncthreads();

    st.init();
    float_vec_t<vec_size> cur_o_vec;
    float cur_m, cur_l;
#pragma unroll
    for (int i = 0; i < bdy; i++) {
        cur_m = smem_lm[i * 2 + 1];
        if (cur_m != -INFINITY) { // important check
            cur_l = smem_lm[i * 2];
            cur_o_vec.load_from_float8(Oi + i * d + tx * vec_size);
            st.merge(cur_o_vec, cur_l, cur_m);
        }
    }

    // final scale
    st.normalize();
    // Write Oi to O in HBM
    st.o_vec.store_to_half(O + ((bx * gridDim.y * NQ * d) + (by * NQ * d)) + tx * vec_size);

}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask) {
    const int B = Q.size(0); const int nh = Q.size(1);
    const int NQ = Q.size(2); const int d = Q.size(3);
    const int NKV = K.size(2);

    TensorStrides q_stride = {Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3)};
    TensorStrides kv_stride = {K.stride(0), K.stride(1), K.stride(2), K.stride(3)};
    TORCH_CHECK(kv_stride.d == 1, "kv_stride in head_dim must be 1");
    TORCH_CHECK(q_stride.d == 1, "q_stride in head_dim must be 1");

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    float M = (float)max_sram_size / sizeof(float);

    // Calculate SRAM size needed per block
    const size_t num_stages_smem = 2; // double buffer for pipeline

    // for decode kernel
    const size_t tile_size_per_bdx = 8U; // 8 KV tokens per bdx.
    const int bdx = 16, bdy = 8;

    const int Bc = (NQ == 1) ? bdy * tile_size_per_bdx: ceil(M / (4 * d));
    const int Br = min(Bc, d);
    const int sram_size = (NQ == 1) ? 8 * d * sizeof(float) + 2 * num_stages_smem * Bc * d * sizeof(c10::Half) + Bc * sizeof(float)
    : (Br * d * sizeof(float)) + (2 * num_stages_smem * Bc * d * sizeof(c10::Half)) + (Bc * Br * sizeof(float));
    const int Tr = ceil((float) NQ / Br);
    const int Tc = ceil((float) NKV / Bc);
    const float softmax_scale = log2e * 1.0 / sqrt(d);
    const size_t vec_size = 8;

    // Set sram_size, otherwise the launch would fail.
    cudaFuncSetAttribute(
        (const void*)decode_kernel<vec_size, tile_size_per_bdx, num_stages_smem>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sram_size
    );
    cudaFuncSetAttribute(
        (const void*)prefill_kernel<vec_size>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sram_size
    );

    // Initialize O to HBM
    auto O = torch::zeros({B, nh, NQ, d}, Q.options());


    const bool has_mask = mask.numel() > 0;
    c10::Half* mask_ptr = has_mask ? mask.data_ptr<c10::Half>() : nullptr;

#ifdef DEBUG
    if (has_mask) {
        printf("with mask\n");
    } else {
        printf("no mask\n");
    }
    printf("B: %d, nh: %d, NQ: %d, d: %d, NKV: %d", B, nh, NQ, d, NKV);
    printf("Default sram size: %d\n", max_sram_size);
    printf("Bc: %d, Br: %d, NQ: %d, NKV: %d, Tr: %d, Tc: %d\n", Bc, Br, NQ, NKV, Tr, Tc);
    printf("Required Sram size: %d\n", sram_size);
#endif

    if (mask_ptr != nullptr) {
        dim3 grid_dim(B, nh);  // batch_size x num_heads
        dim3 block_dim(16, Bc);  // Bc x 16 threads per block
        prefill_kernel<vec_size><<<grid_dim, block_dim, sram_size>>>(
            Q.data_ptr<c10::Half>(), K.data_ptr<c10::Half>(), V.data_ptr<c10::Half>(),
            NQ, NKV, d, Tc, Tr, Bc, Br, softmax_scale,
            O.data_ptr<c10::Half>(), mask_ptr, num_stages_smem, q_stride, kv_stride
        );
    } else {
        dim3 grid_dim(B, nh);  // batch_size x num_heads
        dim3 block_dim(bdx, bdy);  // 128 threads per block according to flashinfer
        decode_kernel<vec_size, tile_size_per_bdx, num_stages_smem><<<grid_dim, block_dim, sram_size>>>(
            Q.data_ptr<c10::Half>(), K.data_ptr<c10::Half>(), V.data_ptr<c10::Half>(),
            NQ, NKV, d, Tc, Bc, softmax_scale,
            O.data_ptr<c10::Half>(), bdx, bdy, q_stride, kv_stride
        );
    }
    return O;
}