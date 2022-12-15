
NVCC        = nvcc
NVCC_FLAGS  = -O3
OBJ         = main.o kernel.o kernelo1.o kernelo2.o kernelo3.o kernelo4.o kernelo5.o cudaUtil.o
EXE         = dw


default: 	$(EXE)

%.o: %.cu
	$(NVCC) $(INCLUDE_DIR) $(NVCC_FLAGS) -dc -rdc=true -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)
	
