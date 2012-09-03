
#CFLAGS += -DDEBUG=3
#CFLAGS += -DPROCESS
CFLAGS += -DDEBUG
#CFLAGS += -DRATE         #RATE will only receive raw packets and discard them
CFLAGS += -DDATA
CFLAGS += -ggdb
CFLAGS += -O3
#CFLAGS += -O2
CFLAGS += -Wall

LIB = -ldl -lspead -L.

################

EXE = spead

SHDR = spead_api.h spead_packet.h
HDR = mutex.h sharedmem.h hash.h $(SHDR) server.h

SRC = module.c mutex.c sharedmem.c hash.c subprocess.c server.c

SSRC = spead.c spead_packet.c
SLIB = libspead.so


OBJ  = $(patsubst %.c,%.o,$(SRC))
SOBJ = $(patsubst %.c,%.lo,$(SSRC))


all: $(EXE)


$(SLIB): $(SOBJ)
	$(CC) -o $@ $^ -shared -Wl,-x 

%.lo: %.c $(SHDR)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC) -fPIC


$(EXE): $(OBJ) $(SLIB)
	$(CC) -o $@ $(OBJ) $(LIB)

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC)



install: all
	$(INSTALL) $(EXE) $(PREFIX)/sbin

clean: 
	$(RM) $(EXE) $(SLIB) *.o *.lo core vgcore* test-*

test-hash: sharedmem.c hash.c
	$(CC) $(CFLAGS) -DTEST_HASH -o $@ $^

test-shm: sharedmem.c
	$(CC) $(CFLAGS) -DTEST_SHARED_MEM -o $@ $^

test-mutex: mutex.c
	$(CC) $(CFLAGS) $(INC) -DTEST_MUTEX -o $@ $^


