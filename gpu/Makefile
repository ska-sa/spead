CFLAGS += -DDEBUG
#CFLAGS += -Wall
#CFLAGS += -ggdb
#CFLAGS += -O3
CFLAGS  += -lcuda -lcudart
USECUFFT = 1
CC = nvcc

UTILS = cudafft.cu

EXE = $(patsubst %.cu,%,$(UTILS))

all: $(EXE)

$(EXE): $(UTILS)
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) -f $(EXE) core vgcore*

install: all
	install $(EXE) /usr/local/bin
