




UTILS = pipespead.c

CFLAGS += -Wall
CFLAGS += -ggdb
CFLAGS += -O3

EXE = $(patsubst %.c,%,$(UTILS))

all: $(EXE)

$(EXE): $(UTILS)
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) -f $(EXE) core vgcore*

