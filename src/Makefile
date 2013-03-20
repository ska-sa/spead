#install
PREFIX=/usr/local
INSTALL=install

#CFLAGS += -DDEBUG=2
#CFLAGS += -DPROCESS
#CFLAGS += -DPROCESSDESC
#CFLAGS += -DDEBUG
CFLAGS += -DIKATCP
#CFLAGS += -DDISCARD
CFLAGS += -DDATA
#CFLAGS += -ggdb
CFLAGS += -O3
#CFLAGS += -O2
CFLAGS += -Wall
CC = gcc -ggdb

LIB = -lspead -L.

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

SSRC = avltree.c itemgroup.c spead.c spead_packet.c mutex.c sharedmem.c hash.c module.c datafile.c net.c subprocess.c stack.c 
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
#	ln -s $(PWD)/$(SLIB) $(PREFIX)/lib/$(SLIB)
	$(INSTALL) $(RXEXE) $(PREFIX)/sbin
	$(INSTALL) $(TXEXE) $(PREFIX)/sbin
	$(INSTALL) $(SLIB) $(PREFIX)/lib
  

clean: 
	$(RM) $(TXEXE) $(RXEXE) $(SLIB) *.o *.lo core vgcore* test-*

fresh: 
	make clean && make

test-hash: sharedmem.c hash.c
	$(CC) $(CFLAGS) -DTEST_HASH -o $@ $^

test-sm: sharedmem.c avltree.c stack.c mutex.c
	$(CC) $(CFLAGS) -DTEST_SHARED_MEM -o $@ $^

test-mutex: mutex.c
	$(CC) $(CFLAGS) $(INC) -DTEST_MUTEX -o $@ $^


