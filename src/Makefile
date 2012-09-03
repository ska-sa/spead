
#CFLAGS += -DDEBUG=3
#CFLAGS += -DPROCESS
#CFLAGS += -DDEBUG
#CFLAGS += -DRATE         #RATE will only receive raw packets and discard them
CFLAGS += -DDATA
CFLAGS += -ggdb
CFLAGS += -O3
#CFLAGS += -O2
CFLAGS += -Wall

LIB = -ldl -lspead -L.

################

EXE = spead

SRC = module.c mutex.c sharedmem.c hash.c subprocess.c spead_packet.c server.c
HDR = mutex.h sharedmem.h hash.h spead_packet.h spead_api.h server.h

OBJ  = $(patsubst %.c,%.o,$(SRC))

SSRC = spead.c
SLIB = libspead.so

SOBJ = $(patsubst %.c,%.o,$(SSRC))

all: $(EXE)

$(SLIB): $(SOBJ)
	$(CC) -o $@ $^ -shared -Wl,-x 

$(SOBJ): $(SSRC)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC) -fPIC

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(LIB)

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC)

install: all
	$(INSTALL) $(EXE) $(PREFIX)/sbin

clean: 
	$(RM) $(EXE) $(SLIB) *.o core vgcore* test-*

test-hash: sharedmem.c hash.c
	$(CC) $(CFLAGS) -DTEST_HASH -o $@ $^

test-shm: sharedmem.c
	$(CC) $(CFLAGS) -DTEST_SHARED_MEM -o $@ $^

test-mutex: mutex.c
	$(CC) $(CFLAGS) $(INC) -DTEST_MUTEX -o $@ $^



