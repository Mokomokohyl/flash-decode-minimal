LLAMA_CHAT_MODEL_PATH ?= /path/to/model # revise to run chat completion and bench
MAX_SEQ_LEN = 1024

VERSION ?= v2
MODE ?= bench

LOG_FILE_NAME ?= $(MODE)_$(VERSION)
LOG_PATH = ./logs/$(LOG_FILE_NAME).log

ENV_PREFIX = MODE=$(MODE)

ifeq ($(VERSION),v1)
	ENV_PREFIX += USE_FLASH_V1=true
	KERNELS=v1
else ifeq ($(VERSION),v2)
	ENV_PREFIX += USE_FLASH_V2=true
	KERNELS=v2
else ifeq ($(VERSION),minimal)
	ENV_PREFIX += USE_FLASH_MINIMAL=true
	KERNELS=minimal
else ifeq ($(VERSION),minimal_v2)
	ENV_PREFIX += USE_FLASH_MINIMAL_V2=true
	KERNELS=minimal_v2
else ifeq ($(VERSION),fdm)
	ENV_PREFIX += USE_FLASH_DECODE_MINIMAL=true
	KERNELS=fdm
else ifeq ($(VERSION),ref)
	ENV_PREFIX = 
else
	$(error Unsupported VERSION: $(VERSION). Use v1, v2, minimal, minimal-v2, fdm, fdm-fixkv or ref)
endif

COMPILE_SCRIPT = compile_kernels.py

BENCH_CMD = torchrun --nproc_per_node 1 bench.py \
	--ckpt_dir $(LLAMA_CHAT_MODEL_PATH) \
	--tokenizer_path ./tokenizer.model \
	--max_seq_len $(MAX_SEQ_LEN) --max_batch_size 6

run:
ifeq ($(MODE), debug)
	mkdir -p logs/
	KERNELS=$(KERNELS) $(ENV_PREFIX) python3 $(COMPILE_SCRIPT) build_ext --inplace > $(LOG_PATH) 2>&1 && \
	$(ENV_PREFIX) $(BENCH_CMD) >> $(LOG_PATH) 2>&1
else
	mkdir -p logs/
	$(ENV_PREFIX) $(BENCH_CMD) > $(LOG_PATH) 2>&1
endif

KERNELS ?= all
compile:
	mkdir -p llama/kernels && touch llama/kernels/__init__.py
	KERNELS=$(KERNELS) python3 $(COMPILE_SCRIPT) build_ext --inplace

bench:
	$(MAKE) run MODE=bench VERSION=$(VERSION)

debug:
	$(MAKE) run MODE=debug VERSION=$(VERSION)

clean:
	@echo "Cleaning up build artifacts..."
	@rm -rf build/ dist/ *.egg-info
	@rm -rf llama/kernels/* 
	@find . -name "__pycache__" -type d -exec rm -r {} +

clean_logs:
	@rm -rf logs/

bench-v1: ; $(MAKE) bench VERSION=v1
bench-v2: ; $(MAKE) bench VERSION=v2
bench-fdm: ; $(MAKE) bench VERSION=fdm
bench-minimal: ; $(MAKE) bench VERSION=minimal
bench-minimal-v2: ; $(MAKE) bench VERSION=minimal_v2
debug-v1: ; $(MAKE) debug VERSION=v1
debug-v2: ; $(MAKE) debug VERSION=v2
debug-fdm: ; $(MAKE) debug VERSION=fdm
debug-minimal: ; $(MAKE) debug VERSION=minimal
debug-minimal-v2: ; $(MAKE) debug VERSION=minimal_v2
ref: ; $(MAKE) run VERSION=ref
bench-contiguous: ; $(MAKE) bench VERSION=fdm LOG_FILE_NAME=profile_contiguous

.PHONY: run bench debug help bench-v1 bench-v2 bench-minimal debug-v1 debug-v2 debug-minimal ref