PYTHON ?= python

# 1 enbale, 0 disable
BACKWARD ?= 1
SPLIT ?= 0
PAGEDKV ?= 0
APPENDKV ?= 0
LOCAL ?= 0
SOFTCAP ?= 0
PACKGQA ?= 0
FP16 ?= 0
FP8 ?= 0
VARLEN ?= 0
CLUSTER ?= 0
HDIM64 ?= 0
HDIM96 ?= 0
HDIM128 ?= 1
HDIM192 ?= 0
HDIM256 ?= 0
SM8X ?= 1
VCOLMAJOR ?= 0
HDIMDIFF64 ?= 0
HDIMDIFF192 ?= 0
FORCE_UNSTABLE_API ?= 1
# Arbitrary mask support
# ARBITRARY: 0=disable, 1=enable
# NUM_FUNC: comma-separated list of func_num values to compile (odd numbers 1-11)
#   e.g., NUM_FUNC=1 or NUM_FUNC=1,3,5
#   More values = more kernel templates = slower compile time
ARBITRARY ?= 1
NUM_FUNC ?= 3

ARBITRARY_TEST ?= 1

ON_VALUES := 1 true TRUE yes YES on ON
bool_false_if_on = $(if $(filter $(ON_VALUES),$($(1))),FALSE,TRUE)
bool_true_if_on  = $(if $(filter $(ON_VALUES),$($(1))),TRUE,FALSE)

export FLASH_ATTENTION_DISABLE_BACKWARD := $(call bool_false_if_on,BACKWARD)
export FLASH_ATTENTION_DISABLE_SPLIT := $(call bool_false_if_on,SPLIT)
export FLASH_ATTENTION_DISABLE_PAGEDKV := $(call bool_false_if_on,PAGEDKV)
export FLASH_ATTENTION_DISABLE_APPENDKV := $(call bool_false_if_on,APPENDKV)
export FLASH_ATTENTION_DISABLE_LOCAL := $(call bool_false_if_on,LOCAL)
export FLASH_ATTENTION_DISABLE_SOFTCAP := $(call bool_false_if_on,SOFTCAP)
export FLASH_ATTENTION_DISABLE_PACKGQA := $(call bool_false_if_on,PACKGQA)
export FLASH_ATTENTION_DISABLE_FP16 := $(call bool_false_if_on,FP16)
export FLASH_ATTENTION_DISABLE_FP8 := $(call bool_false_if_on,FP8)
export FLASH_ATTENTION_DISABLE_VARLEN := $(call bool_false_if_on,VARLEN)
export FLASH_ATTENTION_DISABLE_CLUSTER := $(call bool_false_if_on,CLUSTER)
export FLASH_ATTENTION_DISABLE_HDIM64 := $(call bool_false_if_on,HDIM64)
export FLASH_ATTENTION_DISABLE_HDIM96 := $(call bool_false_if_on,HDIM96)
export FLASH_ATTENTION_DISABLE_HDIM128 := $(call bool_false_if_on,HDIM128)
export FLASH_ATTENTION_DISABLE_HDIM192 := $(call bool_false_if_on,HDIM192)
export FLASH_ATTENTION_DISABLE_HDIM256 := $(call bool_false_if_on,HDIM256)
export FLASH_ATTENTION_DISABLE_SM80 := $(call bool_false_if_on,SM8X)
export FLASH_ATTENTION_ENABLE_VCOLMAJOR := $(call bool_true_if_on,VCOLMAJOR)
export FLASH_ATTENTION_DISABLE_HDIMDIFF64 := $(call bool_false_if_on,HDIMDIFF64)
export FLASH_ATTENTION_DISABLE_HDIMDIFF192 := $(call bool_false_if_on,HDIMDIFF192)
export FLASH_ATTENTION_FORCE_UNSTABLE_API := $(call bool_true_if_on,FORCE_UNSTABLE_API)
export FLASH_ATTENTION_DISABLE_ARBITRARY := $(call bool_false_if_on,ARBITRARY)
export FLASH_ATTENTION_NUM_FUNC := $(NUM_FUNC)

.PHONY: install show_flags help clean fast generate tt vt fm

# Generate kernel instantiation files
generate:
	@echo "Generating kernel instantiation files..."
	$(PYTHON) generate_kernels.py -o instantiations

help:
	@echo "Flash Attention Build System"
	@echo ""
	@echo "Targets:"
	@echo "  install      - Standard installation"
	@echo "  fast         - Fast build with NVCC_THREADS=4"
	@echo "  clean        - Remove build artifacts"
	@echo "  show_flags   - Display current build flags"
	@echo "  tt           - Run tests (default: test_flash_attn.py)"
	@echo "  tt TEST_FILE=test_arbitrary_mask.py  - Run arbitrary mask tests"
	@echo ""
	@echo "Speed Tips:"
	@echo "  1. Disable backward:     make install BACKWARD=0"
	@echo "  2. Single HDIM:          make install HDIM128=1 HDIM64=0 HDIM256=0"
	@echo "  3. More threads:         NVCC_THREADS=4 make install"
	@echo "  4. Use ccache:           PYTORCH_NVCC='ccache nvcc' make install"
	@echo ""
	@echo "Example (fastest dev build):"
	@echo "  make install BACKWARD=0"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf flash_attn_3.egg-info/
	rm -f *.so
	rm -f flash_attn_3/_C*.so
	rm -rf __pycache__/
	rm -rf .eggs/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

install: generate show_flags
	$(PYTHON) setup.py install

# Use test_arbitrary_mask.py only when both ARBITRARY=1 and ARBITRARY_TEST=1
ifneq ($(and $(filter 1,$(ARBITRARY)),$(filter 1,$(ARBITRARY_TEST))),)
    TEST_FILE = test_arbitrary_mask.py
else
    TEST_FILE = test_flash_attn.py
endif

tt:
	PYTHONPATH=${PWD} python $(TEST_FILE)

vt:
	PYTHONPATH=${PWD} pytest $(TEST_FILE)::test_arbitrary_mask -v

fm:
	PYTHONPATH=${PWD} ncu --set full --nvtx --nvtx-include "flash_attn_bwd_kernel/" --nvtx-include "flash_attn_fwd_kernel/"  -f -o flash_attn.%p  python $(TEST_FILE)

show_flags:
	@echo "Build flags -> env："
	@echo " BACKWARD=$(BACKWARD) -> FLASH_ATTENTION_DISABLE_BACKWARD=$(FLASH_ATTENTION_DISABLE_BACKWARD)"
	@echo " SPLIT=$(SPLIT) -> FLASH_ATTENTION_DISABLE_SPLIT=$(FLASH_ATTENTION_DISABLE_SPLIT)"
	@echo " PAGEDKV=$(PAGEDKV) -> FLASH_ATTENTION_DISABLE_PAGEDKV=$(FLASH_ATTENTION_DISABLE_PAGEDKV)"
	@echo " APPENDKV=$(APPENDKV) -> FLASH_ATTENTION_DISABLE_APPENDKV=$(FLASH_ATTENTION_DISABLE_APPENDKV)"
	@echo " LOCAL=$(LOCAL) -> FLASH_ATTENTION_DISABLE_LOCAL=$(FLASH_ATTENTION_DISABLE_LOCAL)"
	@echo " SOFTCAP=$(SOFTCAP) -> FLASH_ATTENTION_DISABLE_SOFTCAP=$(FLASH_ATTENTION_DISABLE_SOFTCAP)"
	@echo " PACKGQA=$(PACKGQA) -> FLASH_ATTENTION_DISABLE_PACKGQA=$(FLASH_ATTENTION_DISABLE_PACKGQA)"
	@echo " FP16=$(FP16) -> FLASH_ATTENTION_DISABLE_FP16=$(FLASH_ATTENTION_DISABLE_FP16)"
	@echo " FP8=$(FP8) -> FLASH_ATTENTION_DISABLE_FP8=$(FLASH_ATTENTION_DISABLE_FP8)"
	@echo " VARLEN=$(VARLEN) -> FLASH_ATTENTION_DISABLE_VARLEN=$(FLASH_ATTENTION_DISABLE_VARLEN)"
	@echo " CLUSTER=$(CLUSTER) -> FLASH_ATTENTION_DISABLE_CLUSTER=$(FLASH_ATTENTION_DISABLE_CLUSTER)"
	@echo " HDIM64=$(HDIM64) -> FLASH_ATTENTION_DISABLE_HDIM64=$(FLASH_ATTENTION_DISABLE_HDIM64)"
	@echo " HDIM96=$(HDIM96) -> FLASH_ATTENTION_DISABLE_HDIM96=$(FLASH_ATTENTION_DISABLE_HDIM96)"
	@echo " HDIM128=$(HDIM128) -> FLASH_ATTENTION_DISABLE_HDIM128=$(FLASH_ATTENTION_DISABLE_HDIM128)"
	@echo " HDIM192=$(HDIM192) -> FLASH_ATTENTION_DISABLE_HDIM192=$(FLASH_ATTENTION_DISABLE_HDIM192)"
	@echo " HDIM256=$(HDIM256) -> FLASH_ATTENTION_DISABLE_HDIM256=$(FLASH_ATTENTION_DISABLE_HDIM256)"
	@echo " SM8X=$(SM8X) -> FLASH_ATTENTION_DISABLE_SM80=$(FLASH_ATTENTION_DISABLE_SM80)"
	@echo " VCOLMAJOR=$(VCOLMAJOR) -> FLASH_ATTENTION_ENABLE_VCOLMAJOR=$(FLASH_ATTENTION_ENABLE_VCOLMAJOR)"
	@echo " HDIMDIFF64=$(HDIMDIFF64) -> FLASH_ATTENTION_DISABLE_HDIMDIFF64=$(FLASH_ATTENTION_DISABLE_HDIMDIFF64)"
	@echo " HDIMDIFF192=$(HDIMDIFF192) -> FLASH_ATTENTION_DISABLE_HDIMDIFF192=$(FLASH_ATTENTION_DISABLE_HDIMDIFF192)"
	@echo " FORCE_UNSTABLE_API=$(FORCE_UNSTABLE_API) -> FLASH_ATTENTION_FORCE_UNSTABLE_API=$(FLASH_ATTENTION_FORCE_UNSTABLE_API)"
	@echo " ARBITRARY=$(ARBITRARY) -> FLASH_ATTENTION_DISABLE_ARBITRARY=$(FLASH_ATTENTION_DISABLE_ARBITRARY)"
	@echo " NUM_FUNC=$(NUM_FUNC) -> FLASH_ATTENTION_NUM_FUNC=$(FLASH_ATTENTION_NUM_FUNC)"



