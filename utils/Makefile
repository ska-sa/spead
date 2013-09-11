CFLAGS += -DDEBUG
CFLAGS += -DIKATCP
CFLAGS += -Wall
CFLAGS += -ggdb
CFLAGS += -O3

LIB= -lspead
LDIR= -L../src
IDIR= -I../src


UTILS = pipespead.c send.c raw_spead.c
EXE = $(patsubst %.c,%.elf,$(UTILS))

OBJ = $(patsubst %.c,%.o,$(UTILS))

all: $(EXE)

%.o: %.c
	$(CC) $(CFLAGS) $(IDIR) -c $^ -o $@

%.elf: %.o 
	$(CC) $< -o $@ $(LDIR) $(LIB)
	mv $@ $(patsubst %.elf,%,$@)  


clean:
	$(RM) $(OBJ) $(EXE) core vgcore*

install: all
	install $(EXE) /usr/local/bin
