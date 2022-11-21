
NVCC        = nvcc
NVCC_FLAGS  = -O0 -g -G
OBJ         = main.o kernel.o kernelo1.o cudaUtil.o
EXE         = dw


default: 	$(EXE)

%.o: %.cu
	$(NVCC) $(INCLUDE_DIR) $(NVCC_FLAGS) -dc -rdc=true -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)
	
