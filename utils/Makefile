CFLAGS += -DDEBUG
CFLAGS += -Wall
CFLAGS += -ggdb
CFLAGS += -O3

#LIB= -lspead
#LDIR= -L../src
#IDIR= -I../src


UTILS = pipespead.c

EXE = $(patsubst %.c,%,$(UTILS))
OBJ = $(patsubst %.c,%.o,$(UTILS))

all: $(EXE)

$(EXE): $(OBJ) 
	$(CC)  $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@


clean:
	$(RM) $(OBJ) $(EXE) core vgcore*

install: all
	install $(EXE) /usr/local/bin
