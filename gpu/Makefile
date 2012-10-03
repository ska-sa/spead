CFLAGS += -DDEBUG
CFLAGS += --compiler-options '-fPIC' --shared -x 'c'
#CFLAGS += -Wall
#CFLAGS += -ggdb
#CFLAGS += -O3
LIB = -lspead

IDIR = -I../src
LDIR = -L../src

#-arch=sm_20
#USECUFFT = 1
NCC = nvcc
CC  = gcc

#gcc -shared -Wl,-x in.o -o out.so

UTILS = cudafft.cu dedisperse.cu

OBJ = $(patsubst %.cu,%.so,$(UTILS))

all: $(OBJ)

%.so: %.cu
	$(NCC) $(CFLAGS) $(LIB) $(IDIR) $(LDIR) $< -o $@

clean:
	$(RM) -f $(OBJ) core vgcore* ocl

ocl: ocldd.c
	$(CC) $< -o $@ -ggdb -DDEBUG -lOpenCL

#install: all
#	install $(EXE) /usr/local/bin

