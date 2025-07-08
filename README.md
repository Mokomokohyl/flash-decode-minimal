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
Revise the `LLAMA_CHAT_MODEL_PATH` to the llama-7b-chat model path in `Makefile` to avoid type in terminal every time.
- Run `make ref LLAMA_CHAT_MODEL_PATH = </path/to/llama-7b-chat>` to run original pytorch llama chat completion.
- Run `make bench-minimal LLAMA_CHAT_MODEL_PATH = </path/to/llama-7b-chat>` to run flash attention minimal kernel from https://github.com/tspeterkim/flash-attention-minimal.
- Run `make bench-v1 LLAMA_CHAT_MODEL_PATH = </path/to/llama-7b-chat>` to run a slightly optimized flash attn v1 kernel (simply add head_dim parallelism, each (ty, tx) handles [1, vec_size](vec_size = 8) elements).
- Run `make bench-v2 LLAMA_CHAT_MODEL_PATH = </path/to/llama-7b-chat>` to run my impl of flash attention v2.
- Run `make bench-fdm LLAMA_CHAT_MODEL_PATH = </path/to/llama-7b-chat>` to run flash-decode-minimal kernel.