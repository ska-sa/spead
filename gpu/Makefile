CFLAGS += -DIKATCP
#CFLAGS += -DDEBUG
CFLAGS += -O3

CC = gcc -ggdb -Wall

LIB = -lspead

OCLLIB = -lOpenCL -lm
OFLAGS = -shared -Wl,-x
KDIR = -DKERNELDIR=\"$(shell pwd)\"

NFLAGS += --compiler-options '-fPIC' --shared -x 'c'
NLIB = -lcufft
NCC = nvcc

IDIR = -I../src
LDIR = -L../src

#.PHONY: all
#.SUFFIXES: 
#.SUFFIXES: .c .o .so

###############
#   VERSION   #
GITVER = $(shell git describe --tags 2> /dev/null || echo unknown)
CFLAGS += -DVERSION=\"$(GITVER)\"

################

.SECONDARY:

#-arch=sm_20
#USECUFFT = 1
#CC  = gcc

#UTILS = cudafft.cu dedisperse.cu

#OBJ = $(patsubst %.cu,%.so,$(UTILS))
EXE= demo
MODS=         dummy.so ocl.so dumpdata.so writefile.so snapshot.so csv.so basic_ocl.so
MODS+= fft.so
SRC= shared.c dummy.c  ocl.c  dumpdata.c  writefile.c snapshot.c csv.c basic_ocl.c

HDR= shared.h
OBJ= $(patsubst %.c,%.o,$(SRC))

all: $(MODS) $(EXE)

%.so: %.cu
	$(NCC) $(CFLAGS) $(NFLAGS) $(NLIB) $(LIB) $(LDIR) $(IDIR) $< -o $@

%.o:: %.c $(HDR)
	$(CC) $(CFLAGS) $(KDIR) $(IDIR) -fPIC -c $< -o $@

%.so: %.o shared.o
	$(CC) $(OFLAGS) $^ -o $@ $(OCLLIB) $(LIB) $(LDIR) 

clean:
	$(RM) $(OBJ) $(EXE) core vgcore* *.o *.so

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) -Wl,-rpath,\"$(shell pwd)/\" $< -o $@ $(LIB) $(IDIR) $(LDIR)


#ocl: ocl.c shared.c
#	$(CC) $(CFLAGS) -DSTANDALONE $(KDIR) $^ -o $@ $(OCLLIB) $(LIB) $(IDIR) $(LDIR) 

#dummy.so: dummy.o
#	$(CC) $(CFLAGS) $^ -o $@  $(LIB) $(LDIR) $(OFLAGS)

#ocl.so: ocldd.o shared.o
#	$(CC) $(CFLAGS) $^ -o $@ $(OCLLIB) $(LIB) $(LDIR) $(OFLAGS)

#install: all
#	install $(EXE) /usr/local/bin

