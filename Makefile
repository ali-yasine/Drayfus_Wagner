
NVCC        = nvcc
NVCC_FLAGS  = -O3
OBJ         = main.o kernel.o cudaUtil.o
EXE         = dw


default: 	$(EXE)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)