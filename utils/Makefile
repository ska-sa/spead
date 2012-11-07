#CFLAGS += -DDEBUG
CFLAGS += -Wall
CFLAGS += -ggdb
CFLAGS += -O3

UTILS = pipespead.c

EXE = $(patsubst %.c,%,$(UTILS))

all: $(EXE)

$(EXE): $(UTILS)
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) $(EXE) core vgcore*

install: all
	install $(EXE) /usr/local/bin
