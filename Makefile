GPU_IDS := 0 1 2 3 4 5 6 7
ENV_DIR := /root/paddlejob/workspace/env_run/panzhaowu/envs/gemm_dev
LOG_DIR := logs
DRY_RUN ?= false
PYTHON := ${ENV_DIR}/bin/python
DEEPGEMM_DIR := third_party/DeepGEMM_annotation
.SILENT:

ifeq ($(DRY_RUN),true)
    RUN := echo "-- [DRY RUN]:"
else
    RUN :=
endif

LOG_FILE = $(LOG_DIR)/$(1)_gpu$(2).log

#---------------------- Setup ----------------------
setup_gemmax:
	${RUN} $(PYTHON) setup.py install

setup_deepgemm:
	${RUN} cd ${DEEPGEMM_DIR} && \
	${RUN} ${PYTHON} setup.py develop
	${RUN} cd ${DEEPGEMM_DIR} && \
	${RUN} $(PYTHON) setup.py install

setup:
	make setup_gemmax
	make setup_deepgemm


#----------------------- Clean -----------------------
clean_gemmax:
	${RUN} rm -rf ./build && \
	${RUN} rm -rf ./dist && \
	${RUN} rm -rf ./*.egg-info && \
	${RUN} ${PYTHON} -m pip uninstall -y gemmax

clean_deepgemm:
	${RUN} cd ${DEEPGEMM_DIR} && \
	${RUN} rm -rf ./build && \
	${RUN} rm -rf ./dist && \
	${RUN} rm -rf ./*.egg-info && \
	${RUN} ${PYTHON} -m pip uninstall -y deep_gemm

clean_whl:
	make clean_gemmax
	make clean_deepgemm

clean_logs:
	$(RUN) rm -rf $(LOG_DIR)

clean_all:
	make clean_whl
	make clean_logs

clean_init:
	make clean_all
	make setup

#---------------------- Benchmark ----------------------
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

# -------------------- Unit tests -------------------
run_deepgemm_unittest:
	$(RUN) $(PYTHON) tests/unittests/test_core.py
# -------------------- Visualize ---------------------
# TODO: implement these
combine_logs:
	$(RUN) $(PYTHON) visualizations/scripts/processing_logs.py

visualize_csv:
	$(RUN) $(PYTHON) visualizations/scripts/plot_gemmax.py
	$(RUN) $(PYTHON) visualizations/scripts/plot_deepgemm.py

# --------------------- Cleaning ----------------------
