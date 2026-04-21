# Prefer a newer CUDA toolkit if installed under /usr/local; fall back to PATH.
NVCC     ?= $(firstword $(wildcard /usr/local/cuda/bin/nvcc /usr/local/cuda-13*/bin/nvcc /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-12.8/bin/nvcc) nvcc)
CXXSTD   ?= c++17
OPT      ?= -O3

# Detect the toolkit's major version to decide which arches and features to enable.
NVCC_MAJOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p')
NVCC_MINOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release [0-9][0-9]*\.\([0-9][0-9]*\).*/\1/p')

# sm_120a is Blackwell + arch-specific features (FP4 tensor cores); available in CUDA 12.8+.
HAS_SM120A := $(shell [ "$(NVCC_MAJOR)" -gt 12 ] 2>/dev/null && echo 1 || ( [ "$(NVCC_MAJOR)" = 12 ] && [ "$(NVCC_MINOR)" -ge 8 ] 2>/dev/null && echo 1 ))

ARCH     ?=
ifeq ($(ARCH),)
  # SASS for a spread of modern arches; PTX from the newest for forward JIT.
  GENCODE := -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90
  ifeq ($(HAS_SM120A),1)
    GENCODE += -gencode arch=compute_120,code=sm_120 \
               -gencode arch=compute_120a,code=sm_120a \
               -gencode arch=compute_120,code=compute_120
  else
    GENCODE += -gencode arch=compute_90,code=compute_90
  endif
else
  GENCODE := -arch=$(ARCH)
endif

DEFINES :=
ifeq ($(HAS_SM120A),1)
  DEFINES += -DGPUBENCH_ENABLE_FP4=1
endif

NVCCFLAGS := $(OPT) -std=$(CXXSTD) $(GENCODE) $(DEFINES) --use_fast_math -Xcompiler -Wall,-Wextra
LDLIBS    := -lcublas -lcublasLt

TARGET := gpubench
SRC    := gpubench.cu

.PHONY: all clean run native info

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ $(LDLIBS)

# Faster build: compile only for the host GPU.
native: NVCCFLAGS := $(OPT) -std=$(CXXSTD) $(DEFINES) --use_fast_math -Xcompiler -Wall,-Wextra
native:
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

info:
	@echo "NVCC        : $(NVCC)"
	@echo "NVCC version: $(NVCC_MAJOR).$(NVCC_MINOR)"
	@echo "HAS_SM120A  : $(HAS_SM120A)"
	@echo "GENCODE     : $(GENCODE)"
	@echo "DEFINES     : $(DEFINES)"

clean:
	rm -f $(TARGET)
