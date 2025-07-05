LLAMA_CHAT_MODEL_PATH = /state/partition/ylhuang/llama-2-7b-chat
MAX_SEQ_LEN = 1024

VERSION ?= v2
MODE ?= bench

ifeq ($(VERSION),v1)
    ENV_PREFIX = USE_FLASH_V1=true
else ifeq ($(VERSION),v2)
    ENV_PREFIX = USE_FLASH_V2=true
else ifeq ($(VERSION),minimal)
    ENV_PREFIX = USE_FLASH_MINIMAL=true
else ifeq ($(VERSION),ref)
    ENV_PREFIX = 
else
    $(error Unsupported VERSION: $(VERSION). Use v1, v2, minimal, or ref)
endif

ifeq ($(MODE),debug)
    COMPILE_SCRIPT = compile_debug_kernels.py
else
    COMPILE_SCRIPT = compile_kernels.py
endif

BENCH_CMD = torchrun --nproc_per_node 1 bench.py \
    --ckpt_dir $(LLAMA_CHAT_MODEL_PATH) \
    --tokenizer_path ./tokenizer.model \
    --max_seq_len $(MAX_SEQ_LEN) --max_batch_size 6

run:
ifeq ($(VERSION),ref)
    $(BENCH_CMD) > ./logs/$(MODE)_$(VERSION).log 2>&1
else
    $(ENV_PREFIX) python3 $(COMPILE_SCRIPT) build_ext --inplace > ./logs/$(MODE)_$(VERSION).log 2>&1 && \
    $(ENV_PREFIX) $(BENCH_CMD) >> ./logs/$(MODE)_$(VERSION).log 2>&1
endif

bench:
    $(MAKE) run MODE=bench VERSION=$(VERSION)

debug:
    $(MAKE) run MODE=debug VERSION=$(VERSION)

bench-v1: ; $(MAKE) bench VERSION=v1
bench-v2: ; $(MAKE) bench VERSION=v2
bench-minimal: ; $(MAKE) bench VERSION=minimal
debug-v1: ; $(MAKE) debug VERSION=v1
debug-v2: ; $(MAKE) debug VERSION=v2
debug-minimal: ; $(MAKE) debug VERSION=minimal
ref: ; $(MAKE) run VERSION=ref

help:
    @echo "Usage:"
    @echo "  make run VERSION=<version> MODE=<mode>"
    @echo "  make bench [VERSION=<version>]        # default: v2"
    @echo "  make debug [VERSION=<version>]        # default: v2"
    @echo ""
    @echo "Versions: v1, v2, minimal, ref"
    @echo "Modes: bench, debug"
    @echo ""
    @echo "Examples:"
    @echo "  make bench                  # bench v2"
    @echo "  make debug VERSION=v1      # debug v1"
    @echo "  make bench-minimal         # bench minimal"

.PHONY: run bench debug help bench-v1 bench-v2 bench-minimal debug-v1 debug-v2 debug-minimal ref