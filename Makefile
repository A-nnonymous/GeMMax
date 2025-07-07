GPU_IDS := 0 1 2 3 4 5 6 7
ENV_DIR := /root/paddlejob/workspace/env_run/panzhaowu/envs/pd_kit
LOG_DIR := logs
DRY_RUN ?= true
PYTHON := ${ENV_DIR}/bin/python

ifeq ($(DRY_RUN),true)
    RUN := echo "Would run:"
else
    RUN := @
endif

LOG_FILE = $(LOG_DIR)/$(1)_gpu$(2).log

setup:
	$(RUN) $(PYTHON) setup.py install && cd third_party/deep_gemm && $(PYTHON) setup.py install

$(LOG_DIR):
	$(RUN) mkdir -p $(LOG_DIR)

define run_benchmark
	@echo "Running $(1) benchmark on $(NUM_GPUS) GPUs..."
	@selected_gpus=$$(echo $(GPU_IDS) | tr ' ' '\n' | $(2)); \
	for gpu in $$selected_gpus; do \
		cmd="$(PYTHON) tests/test_$(1).py > $(call LOG_FILE,$(1),$$gpu) 2>&1 &"; \
		$(RUN) $$cmd; \
	done; \
	wait
endef

run_deepgemm_benchmark: NUM_GPUS ?= 8
run_deepgemm_benchmark: $(LOG_DIR)
	$(call run_benchmark,deepgemm,head -n $(NUM_GPUS))

run_gemmax_benchmark: NUM_GPUS ?= 8
run_gemmax_benchmark: $(LOG_DIR)
	$(call run_benchmark,gemmax,tail -n $(NUM_GPUS))

run_all_benchmarks: $(LOG_DIR)
	make run_deepgemm_benchmark NUM_GPUS=4 & \
	make run_gemmax_benchmark NUM_GPUS=4
	wait

clean_logs:
	$(RUN) rm -rf $(LOG_DIR)
