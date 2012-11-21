#CFLAGS += -DDEBUG=3
CFLAGS += -DPROCESS
#CFLAGS += -DIKATCP
CFLAGS += -DDISCARD
CFLAGS += -DDEBUG
#CFLAGS += -DRATE         #RATE will only receive raw packets and discard them
CFLAGS += -DDATA
CFLAGS += -ggdb
CFLAGS += -O3
#CFLAGS += -O2
CFLAGS += -Wall

LIB = -lspead -L.

ifeq (,$(findstring -DIKATCP,$(CFLAGS)))
	LIB += -lkatcp
endif

###############
#   VERSION   #
GITVER = $(shell git describe --tags 2> /dev/null || echo unknown)
CFLAGS += -DVERSION=\"$(GITVER)\"

################

RXEXE = speadrx
TXEXE = speadtx

RXSRC = server.c
TXSRC = tx.c

SHDR = mutex.h hash.h spead_api.h spead_packet.h
HDR  = $(SHDR) server.h

SSRC = spead.c spead_packet.c mutex.c sharedmem.c hash.c module.c datafile.c net.c subprocess.c
SLIB = libspead.so

SOBJ = $(patsubst %.c,%.lo,$(SSRC))

TXOBJ  = $(patsubst %.c,%.o,$(TXSRC))
RXOBJ  = $(patsubst %.c,%.o,$(RXSRC))


all: $(RXEXE) $(TXEXE)


$(SLIB): $(SOBJ)
	$(CC) -o $@ $^ -shared -Wl,-x -ldl 

%.lo: %.c $(SHDR)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC) -fPIC


$(RXEXE): $(RXOBJ) $(SLIB)
	$(CC) -o $@ $< $(LIB)

$(TXEXE): $(TXOBJ) $(SLIB)
	$(CC) -o $@ $< $(LIB)

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC)



install: all
	ln -s $(PWD)/$(SLIB) $(PREFIX)/lib/$(SLIB)
#	$(INSTALL) $(EXE) $(PREFIX)/sbin

clean: 
	$(RM) $(TXEXE) $(RXEXE) $(SLIB) *.o *.lo core vgcore* test-*

test-hash: sharedmem.c hash.c
	$(CC) $(CFLAGS) -DTEST_HASH -o $@ $^

test-shm: sharedmem.c
	$(CC) $(CFLAGS) -DTEST_SHARED_MEM -o $@ $^

test-mutex: mutex.c
	$(CC) $(CFLAGS) $(INC) -DTEST_MUTEX -o $@ $^


