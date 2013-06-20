CFLAGS += -DIKATCP
CFLAGS += -DDEBUG
#CFLAGS += -O3

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

###############
#   VERSION   #
GITVER = $(shell git describe --tags 2> /dev/null || echo unknown)
CFLAGS += -DVERSION=\"$(GITVER)\"

################

.SECONDARY:
#.PRECIOUS: %.o

#-arch=sm_20
#USECUFFT = 1
#CC  = gcc

#UTILS = cudafft.cu dedisperse.cu

#OBJ = $(patsubst %.cu,%.so,$(UTILS))
EXE= demo wave_gen device
MODS=         dummy.so ocl.so dumpdata.so writefile.so snapshot.so csv.so basic_ocl.so pipeline_0.so pipeline_1.so radix2_po2_fft.so power.so data_transfer.so
MODS+= fft.so
SRC= shared.c dummy.c  ocl.c  dumpdata.c  writefile.c snapshot.c csv.c basic_ocl.c pipeline_0.c pipeline_1.c radix2_po2_fft.c power.c data_transfer.c

HDR= shared.h
OBJ= $(patsubst %.c,%.o,$(SRC))
EXELIST= $(patsubst %,%.elf,$(EXE))

all: $(MODS) $(EXELIST)

%.so: %.cu
	$(NCC) $(CFLAGS) $(NFLAGS) $(NLIB) $(LIB) $(LDIR) $(IDIR) $< -o $@

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) $(KDIR) $(IDIR) -fPIC -c $< -o $@

%.so: %.o shared.o
	$(CC) $(OFLAGS) $^ -o $@ $(OCLLIB) $(LIB) $(LDIR) 

clean:
	$(RM) $(OBJ) $(EXE) core vgcore* *.o *.so

%.elf: %.c shared.o
	$(CC) $(CFLAGS) -Wl,-rpath,\"$(shell pwd)/\" $^ -o $@ $(OCLLIB) $(LIB) $(IDIR) $(LDIR) -ldl -lm
	mv $@ $(patsubst %.elf,%,$@)  

#ocl: ocl.c shared.c
#	$(CC) $(CFLAGS) -DSTANDALONE $(KDIR) $^ -o $@ $(OCLLIB) $(LIB) $(IDIR) $(LDIR) 

#dummy.so: dummy.o
#	$(CC) $(CFLAGS) $^ -o $@  $(LIB) $(LDIR) $(OFLAGS)

#ocl.so: ocldd.o shared.o
#	$(CC) $(CFLAGS) $^ -o $@ $(OCLLIB) $(LIB) $(LDIR) $(OFLAGS)

#install: all
#	install $(EXE) /usr/local/bin

