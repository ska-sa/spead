CFLAGS += -DDEBUG
#NFLAGS += --compiler-options '-fPIC' --shared -x 'c'
CFLAGS += -Wall
CFLAGS += -ggdb
CFLAGS += -O3
LIB = -lspead
OCLLIB = -lOpenCL -lm
OFLAGS = -shared -Wl,-x
KDIR = -DKERNELDIR=\"$(shell pwd)\"

IDIR = -I../src
LDIR = -L../src

#.SECONDARY:
#.PRECIOUS: %.o

#-arch=sm_20
#USECUFFT = 1
#NCC = nvcc
#CC  = gcc

#gcc -shared -Wl,-x in.o -o out.so

#UTILS = cudafft.cu dedisperse.cu

#OBJ = $(patsubst %.cu,%.so,$(UTILS))
HDR= shared.h
SRC= shared.c dummy.c ocl.c 
OBJ= $(patsubst %.c,%.o,$(SRC))


all: ocl.so dummy.so 

#%.so: %.cu
#	$(NCC) -DDEBUG $(NFLAGS) $(LIB) $(IDIR) $(LDIR) $< -o $@

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) $(KDIR) $(IDIR) -fPIC -c $< -o $@

%.so: %.o shared.o
	$(CC) $(OFLAGS) $^ -o $@ $(OCLLIB) $(LIB) $(LDIR) 

clean:
	$(RM) $(OBJ) core vgcore* *.o *.so

ocl: ocl.o shared.o
	$(CC) $^ -o $@ $(CFLAGS) -DSTANDALONE $(OCLLIB) $(LIB) $(LDIR)

#dummy.so: dummy.o
#	$(CC) $(CFLAGS) $^ -o $@  $(LIB) $(LDIR) $(OFLAGS)

#ocl.so: ocldd.o shared.o
#	$(CC) $(CFLAGS) $^ -o $@ $(OCLLIB) $(LIB) $(LDIR) $(OFLAGS)

#install: all
#	install $(EXE) /usr/local/bin

