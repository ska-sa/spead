CFLAGS += -DDEBUG
CFLAGS += --compiler-options '-fPIC' --shared -x 'c'
#CFLAGS += -Wall
#CFLAGS += -ggdb
#CFLAGS += -O3
LIB = -lcufft
#SOF = -shared -Wl,-x
IDIR = -I../src

#-arch=sm_20
#USECUFFT = 1
NCC = nvcc


#gcc -shared -Wl,-x in.o -o out.so

UTILS = cudafft.cu dedisperse.cu

#EXE = $(patsubst %.cu,%,$(UTILS))
OBJ = $(patsubst %.cu,%.so,$(UTILS))
#SOS = $(patsubst %.o,%.so,$(OBJ))

all: $(OBJ)

%.so: %.cu
	$(NCC) $(CFLAGS) $(LIB) $(IDIR) $< -o $@

clean:
	$(RM) -f $(OBJ) core vgcore*

#install: all
#	install $(EXE) /usr/local/bin

