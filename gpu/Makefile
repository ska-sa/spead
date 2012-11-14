CFLAGS += -DIKATCP
CFLAGS += -DDEBUG
CFLAGS += -Wall
CFLAGS += -ggdb
CFLAGS += -O3

LIB = -lspead

OCLLIB = -lOpenCL -lm
OFLAGS = -shared -Wl,-x
KDIR = -DKERNELDIR=\"$(shell pwd)\"

NFLAGS += --compiler-options '-fPIC' --shared -x 'c'
NLIB = -lcufft
NCC = nvcc

IDIR = -I../src
LDIR = -L../src

.SECONDARY:
#.PRECIOUS: %.o

#-arch=sm_20
#USECUFFT = 1
#CC  = gcc

#UTILS = cudafft.cu dedisperse.cu

#OBJ = $(patsubst %.cu,%.so,$(UTILS))
HDR= shared.h
SRC= shared.c dummy.c ocl.c 
OBJ= $(patsubst %.c,%.o,$(SRC))
EXE= demo


all: ocl.so dummy.so fft.so $(EXE)

%.so: %.cu
	$(NCC) -DDEBUG $(NFLAGS) $(NLIB) $(LIB) $(LDIR) $(IDIR) $< -o $@

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) $(KDIR) $(IDIR) -fPIC -c $< -o $@

%.so: %.o shared.o
	$(CC) $(OFLAGS) $^ -o $@ $(OCLLIB) $(LIB) $(LDIR) 

clean:
	$(RM) $(OBJ) $(EXE) core vgcore* *.o *.so

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $< -o $@


#ocl: ocl.c shared.c
#	$(CC) $(CFLAGS) -DSTANDALONE $(KDIR) $^ -o $@ $(OCLLIB) $(LIB) $(IDIR) $(LDIR) 

#dummy.so: dummy.o
#	$(CC) $(CFLAGS) $^ -o $@  $(LIB) $(LDIR) $(OFLAGS)

#ocl.so: ocldd.o shared.o
#	$(CC) $(CFLAGS) $^ -o $@ $(OCLLIB) $(LIB) $(LDIR) $(OFLAGS)

#install: all
#	install $(EXE) /usr/local/bin

