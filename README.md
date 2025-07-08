# flash-decode-minimal
The repo records my experience learning flash attention/flash decode.
Currently the kernels only support llama-7b-chat(head_dim=128, MHA) model inference.

## Environment
- CUDA 12.6
- Create a new conda virtual env, run 
```bash
pip install -e .
```
## Build and run
Update the `LLAMA_CHAT_MODEL_PATH` to your Llama-7B-chat model path in `Makefile` or pass it as a command-line argument (e.g. `make bench LLAMA_MODEL_PATH=/path/to/model`).
- `make ref`: Runs the original pytorch llama-7b chat completion.
- `make bench-minimal`: Runs flash attention minimal kernel from https://github.com/tspeterkim/flash-attention-minimal. Only revised the `Bc` and `Br` calculation and add mask for prefill. (kernels/flash_attn_minimal.cu)
- `make bench-v1`: Runs a slightly optimized Flash Attention v1 kernel based on flash-attention-minimal (simply add head_dim parallelism, each (ty, tx) handles [1, vec_size](vec_size = 8) elements). (kernels/flash_attn_v1.cu)
- `make bench-v2`: Runs my implementation of Flash Attention V2. (kernels/flash_attn_v2.cu)
- `make bench-fdm`: Runs flash-decode-minimal kernel, based on https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/decode.cuh (kernels/flash_decode_minimal.cu)

## Notes

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
The `keys` and `values` tensors are slices of a pre-allocated fixed-length KV cache, making them non-contiguous in memory. I had to call K.contiguous() and V.contiguous() in the kernels, which introduces non-negligible overhead.

### TODOs

- [ ] Replace KV cache with Paged KV cache
- [ ] Support GQA (Grouped-Query Attention)
- [ ] Add better profiling methods