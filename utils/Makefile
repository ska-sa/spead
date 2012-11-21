#CFLAGS += -DDEBUG
CFLAGS += -Wall
CFLAGS += -ggdb
CFLAGS += -O3

LIB= -lspead
LDIR= -L../src
IDIR= -I../src


UTILS = pipespead.c speadtx.c

EXE = $(patsubst %.c,%,$(UTILS))
OBJ = $(patsubst %.c,%.o,$(UTILS))

all: $(EXE)

$(EXE): 
	$(CC) $(CFLAGS) $< -o $@

%.o: %.c
	$(CC) $^ -o $@


clean:
	$(RM) $(EXE) core vgcore*

install: all
	install $(EXE) /usr/local/bin
