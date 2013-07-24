#install
PREFIX=/usr/local
INSTALL=install

#CFLAGS += -DDEBUG=2
#CFLAGS += -DPROCESS
#CFLAGS += -DPROCESSDESC
CFLAGS += -DDEBUG
CFLAGS += -DIKATCP
#CFLAGS += -DDISCARD
CFLAGS += -DDATA
#CFLAGS += -ggdb
CFLAGS += -O3
#CFLAGS += -O2
CFLAGS += -Wall
#CFLAGS += -m32
CC = gcc -ggdb

LIB = -lspead -L. -lm

ifeq (,$(findstring -DIKATCP,$(CFLAGS)))
	LIB += -lkatcp
endif

PWD = $(shell pwd)

###############
#   VERSION   #
GITVER = $(shell git describe --tags 2> /dev/null || echo unknown)
CFLAGS += -DVERSION=\"$(GITVER)\"

################

RXEXE = speadrx
TXEXE = speadtx

RXSRC = server.c
TXSRC = tx.c

SHDR = avltree.h mutex.h hash.h spead_api.h spead_packet.h stack.h
HDR  = $(SHDR) server.h tx.h

SSRC = avltree.c itemgroup.c spead.c spead_packet.c mutex.c sharedmem.c hash.c module.c datafile.c net.c subprocess.c stack.c pipeline.c
DLIB = libspead.so
SLIB = libspead.a


SOBJ = $(patsubst %.c,%.lo,$(SSRC))

TXOBJ  = $(patsubst %.c,%.o,$(TXSRC))
RXOBJ  = $(patsubst %.c,%.o,$(RXSRC))


all: $(RXEXE) $(TXEXE)
static: $(RXEXE)-static $(TXEXE)-static

$(DLIB): $(SOBJ)
	$(CC) -o $@ $^ -shared -Wl,-x -ldl 

$(SLIB): $(SOBJ)
	$(AR) rcs $@ $^

%.lo: %.c $(SHDR)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC) -fPIC

$(RXEXE): $(RXOBJ) $(DLIB)
	$(CC) -o $@ $< $(LIB)

$(TXEXE): $(TXOBJ) $(DLIB)
	$(CC) -o $@ $< $(LIB)

$(RXEXE)-static: $(RXOBJ) $(SLIB)
	$(CC) -o $@ $^ -ldl -lm

$(TXEXE)-static: $(TXOBJ) $(SLIB)
	$(CC) -o $@ $^ -ldl -lm

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@ $(INC)



install: all
#	ln -s $(PWD)/$(DLIB) $(PREFIX)/lib/$(DLIB)
	$(INSTALL) $(RXEXE) $(PREFIX)/sbin
	$(INSTALL) $(TXEXE) $(PREFIX)/sbin
	$(INSTALL) $(DLIB) $(PREFIX)/lib
	/sbin/ldconfig  

clean: 
	$(RM) $(TXEXE) $(RXEXE) $(SLIB) $(RXEXE)-static $(TXEXE)-static $(DLIB) *.o *.lo core vgcore* test-*

fresh: 
	make clean && make

test-hash: sharedmem.c hash.c
	$(CC) $(CFLAGS) -DTEST_HASH -o $@ $^

test-sm: sharedmem.c avltree.c stack.c mutex.c
	$(CC) $(CFLAGS) -DTEST_SHARED_MEM -o $@ $^

test-mutex: mutex.c
	$(CC) $(CFLAGS) $(INC) -DTEST_MUTEX -o $@ $^


