
bench_v1:
	USE_FLASH_V1=true python3 compile_kernels.py build_ext --inplace > ./logs/bench_v1.log 2>&1 && \
	USE_FLASH_V1=true torchrun --nproc_per_node 1 bench.py \
		--ckpt_dir /home/ylhuang/llama-2-7b-chat \
		--tokenizer_path ./tokenizer.model \
		--max_seq_len 512 --max_batch_size 6 >> ./logs/bench_v1.log 2>&1

debug_v1:
	USE_FLASH_V1=true python3 compile_debug_kernels.py build_ext --inplace > ./logs/debug_v1.log 2>&1 && \
	USE_FLASH_V1=true torchrun --nproc_per_node 1 bench.py \
		--ckpt_dir /home/ylhuang/llama-2-7b-chat \
		--tokenizer_path ./tokenizer.model \
		--max_seq_len 512 --max_batch_size 6 >> ./logs/debug_v1.log 2>&1

bench_minimal:
	USE_FLASH_MINIMAL=true python3 compile_kernels.py build_ext --inplace > ./logs/bench_minimal.log 2>&1 && \
	USE_FLASH_MINIMAL=true torchrun --nproc_per_node 1 bench.py \
		--ckpt_dir /home/ylhuang/llama-2-7b-chat \
		--tokenizer_path ./tokenizer.model \
		--max_seq_len 512 --max_batch_size 6 >> ./logs/bench_minimal.log 2>&1

debug_minimal:
	USE_FLASH_MINIMAL=true python3 compile_debug_kernels.py build_ext --inplace > ./logs/debug_minimal.log 2>&1 && \
	USE_FLASH_MINIMAL=true torchrun --nproc_per_node 1 bench.py \
		--ckpt_dir /home/ylhuang/llama-2-7b-chat \
		--tokenizer_path ./tokenizer.model \
		--max_seq_len 512 --max_batch_size 6 >> ./logs/debug_minimal.log 2>&1

original:
	torchrun --nproc_per_node 1 bench.py \
		--ckpt_dir /home/ylhuang/llama-2-7b-chat \
		--tokenizer_path ./tokenizer.model \
		--max_seq_len 512 --max_batch_size 6 > ./logs/bench_ref.log 2>&1