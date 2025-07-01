bench:
	export USE_CUDA_KERNEL=true
	python3 compile_kernels.py build_ext --inplace > bench.log 2>&1
	torchrun --nproc_per_node 1 bench.py \
		--ckpt_dir /home/ylhuang/llama-2-7b-chat \
		--tokenizer_path ./tokenizer.model \
		--max_seq_len 512 --max_batch_size 6 >> bench.log 2>&1


original:
	export USE_CUDA_KERNEL=false
	torchrun --nproc_per_node 1 bench.py \
		--ckpt_dir /home/ylhuang/llama-2-7b-chat \
		--tokenizer_path ./tokenizer.model \
		--max_seq_len 512 --max_batch_size 6 > bench_ref.log 2>&1