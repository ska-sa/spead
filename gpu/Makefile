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

all: $(OBJ) ocl

%.so: %.cu
	$(NCC) $(CFLAGS) $(LIB) $(IDIR) $(LDIR) $< -o $@

clean:
	$(RM) -f $(OBJ) core vgcore* ocl *.o

shared.o: shared.c
	$(CC) -c $< -o $@ -ggdb -DDEBUG

ocl: ocldd.c shared.o
	$(CC) $^ -o $@ -ggdb -DDEBUG -lOpenCL -lm

#install: all
#	install $(EXE) /usr/local/bin

