# flash-decode-minimal
The repo records my experience learning flash attention/flash decode.
Currently the kernels only supports llama-7b-chat model.

## Environment
- CUDA 12.6
- Create a new conda virtual env, run 
```bash
pip install -e .
```
## Build and run
Revise the `LLAMA_CHAT_MODEL_PATH` to the llama-7b-chat model path in `Makefile`.
- Run `make ref` to run original pytorch llama chat completion.
- Run `make bench-minimal` to run flash attention minimal kernel from https://github.com/tspeterkim/flash-attention-minimal.
- Run `make bench-v1` to run a slightly optimized flash attn v1 kernel (simply add head_dim parallelism, each (ty, tx) handles [1, vec_size](vec_size = 8) elements).
- Run `make bench-v2` to run my impl of flash attention v2.
- Run `make bench-fdm` to run flash-decode-minimal kernel, which I wrote in reference to https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/decode.cuh

## Notes

The above kernels replace the following calculation in class Attention in `llama/model.py`. 
```python
class Attention(nn.module):
    # ... 
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
    # ...
```
The two tensors keys and values are not contiguous in memory as they refer to a slice of pre-allocated fixed-length KV cache. I had to call K.contiguous() and V.contiguous() in the kernels, which are not negligible overheads.