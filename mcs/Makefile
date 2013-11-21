CFLAGS += -DDEBUG
CFLAGS += -DDATA
CFLAGS += -Wall
CFLAGS += -ggdb
CFLAGS += -O3
CC = gcc

LIB= -lspead
LDIR= -L../src
IDIR= -I../src


HDR = mcs.h
UTILS = mcs.c packet.c
EXE = mcs



###############
#   VERSION   #
GITVER = $(shell git describe --tags 2> /dev/null || echo unknown)
CFLAGS += -DVERSION=\"$(GITVER)\"



OBJ = $(patsubst %.c,%.o,$(UTILS))

all: $(EXE)

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) $(IDIR) -c $< -o $@

$(EXE): $(OBJ)
	$(CC) $^ -o $@ $(LDIR) $(LIB)

clean:
	$(RM) $(OBJ) $(EXE) core vgcore*

install: all
	install $(EXE) /usr/local/bin
