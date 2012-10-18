#CFLAGS += -DDEBUG
NFLAGS += --compiler-options '-fPIC' --shared -x 'c'
CFLAGS += -Wall
CFLAGS += -ggdb
CFLAGS += -O3
LIB = -lspead
OCLLIB = -lOpenCL -lm
OFLAGS = -shared -Wl,-x
KDIR = -DKERNELDIR=\"$(shell pwd)\"

IDIR = -I../src
LDIR = -L../src

#-arch=sm_20
#USECUFFT = 1
NCC = nvcc
CC  = gcc

#gcc -shared -Wl,-x in.o -o out.so

UTILS = cudafft.cu dedisperse.cu

OBJ = $(patsubst %.cu,%.so,$(UTILS))

all: ocl.so $(OBJ)

%.so: %.cu
	$(NCC) -DDEBUG $(NFLAGS) $(LIB) $(IDIR) $(LDIR) $< -o $@

clean:
	$(RM) -f $(OBJ) core vgcore* *.o ocl.so

%.o: %.c
	$(CC) -c $< -o $@ $(KDIR) $(CFLAGS) $(IDIR) -fPIC

#ocl: ocldd.o shared.o
#	$(CC) $^ -o $@ $(CFLAGS) -DSTANDALONE $(OCLLIB) $(LIB) $(LDIR)

ocl.so: ocldd.o shared.o
	$(CC) $^ -o $@ $(CFLAGS) $(OCLLIB) $(LIB) $(LDIR) $(OFLAGS)

#install: all
#	install $(EXE) /usr/local/bin

