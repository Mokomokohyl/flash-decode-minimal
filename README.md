# flash-decode-minimal
The repo records my experience learning flash attention/flash decode.
Currently the kernels only support llama-2-7b-chat(head_dim=128, MHA) model inference.

## Environment setup
- CUDA 12.6
- Create a new conda virtual env, run 
```bash
pip install -e .
```
## Build and run
Update the `LLAMA_CHAT_MODEL_PATH` to your Llama-2-7B-chat model path in `Makefile` or pass it as a command-line argument (e.g. `make bench LLAMA_MODEL_PATH=/path/to/model`).
Type `make compile` to compile all kernels. It will take 2~3 minutes to generate llama/kernels/*.so for llama/model.py to import and call.
- `make ref`: Run the original pytorch llama-7b chat completion.
- `make bench-minimal`: Runs flash attention minimal kernel from https://github.com/tspeterkim/flash-attention-minimal with some revision.
    + Revised the `Bc` and `Br` calculation and add mask for prefill. (kernels/flash_attn_minimal.cu)
    + Naive causal mask
- `make bench-v1`: Run a slightly optimized Flash Attention v1 kernel based on flash-attention-minimal (simply add head_dim parallelism, each (ty, tx) handles [1, vec_size](vec_size = 8) elements). (kernels/flash_attn_v1.cu)
- `make bench-minimal-v2`: Run my implementation of Flash Attention V2 of flash-attention-minimal style. (kernels/flash_attn_minimal_v2.cu)
- `make bench-v2`: Run my implementation of Flash Attention V2. Add double buffer pipeline. (kernels/flash_attn_v2.cu)
- `make bench-fdm`: Run flash-decode-minimal kernel, referring to https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/decode.cuh (kernels/flash_decode_minimal.cu)
- `make bench-fdm-fixkv`: Run flash-decode-minimal kernel that avoid Q/K/V.contiguous() call (use Q.stride() to get stride for B/nh/N/d and calculate index correspondingly). (kernels/flash_decode_fixed_len_kv_cache.cu)

The terminal outputs are redirected to logs/bench_*.log.

Replace `bench` with `debug` will compile kernels with `-DDEBUG` and output some debug messages in the logs.

## Notes

You may find that the kernels `minimal`, `v1`, `minimal-v2`, `v2` run slower than the original PyTorch implementation.

The custom kernels replace the following calculation in forward() method of class Attention in `llama/model.py`. 
```python
class Attention(nn.module):
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
    # ... 
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
    # ...
```
The `keys` and `values` tensors are slices of a pre-allocated fixed-length KV cache, making them non-contiguous in memory. In the kernels before `fdm-fixkv`, Q/K/V.contiguous() functions are called to make Q/K/V contiguous in memory, which introduces non-negligible overhead. To verify this, you can run
```bash
git checkout profile_contiguous
make bench-contiguous LLAMA_CHAT_MODEL_PATH=/path/to/model
```
and go to /logs/profile_contiguous.log to check the profile results.

When I revise flash-attn-minimal so that it receives q_stride and kv_stride struct for indexing, it performance got worse than calling Q/K/V.contiguous()(Approximately 9.43 tokens/s on A100). Then I restore the Q/K/V.contiguous() calling. I didn't figure out why.

### TODOs

- [ ] Replace KV cache with Paged KV cache
- [ ] Support GQA (Grouped-Query Attention)
- [ ] Add better profiling methods