# Makefile

# CUDA code generation flags
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80
GENCODE_FLAGS   ?= $(GENCODE_SM60) $(GENCODE_SM70) $(GENCODE_SM80)

# Compilers
CC              := g++
NVCC            := nvcc

# Targets
GPU_TARGET      := transformacion-gpu
CPU_TARGET      := transformacion-cpu

# Source files
GPU_SRC         := transformacion-gpu.cu
CPU_SRC         := transformacion-cpu.cc

# Compiler flags
CFLAGS          := -O3
NVCCFLAGS       := -I./includes -O3 -m64 $(GENCODE_FLAGS)

all: $(GPU_TARGET) $(CPU_TARGET)

$(GPU_TARGET):
	$(NVCC) $(NVCCFLAGS) $(GPU_SRC) -o $(GPU_TARGET)

$(CPU_TARGET):
	$(CC) $(CFLAGS) $(CPU_SRC) -o $(CPU_TARGET)

clean:
	rm -f $(GPU_TARGET) $(CPU_TARGET) *.o *~
