#INC = -I$(SSLBUILD)/include,$(KATCP)
#LIB = -lssl 

#KERN = /usr/src/linux-headers-3.2.0-26-generic

#INC = -I$(KERN)/include


EXE = spead 
SRC = mutex.c sharedmem.c hash.c subprocess.c spead_packet.c spead_api.c server.c
HDR = mutex.h sharedmem.h hash.h spead_packet.h spead_api.h server.h

# CFLAGS += -DSTANDALONE
#CFLAGS += -DDEBUG=3
#CFLAGS += -DPROCESS
#CFLAGS += -DDEBUG
#CFLAGS += -DRATE         #RATE will only receive raw packets and discard them
CFLAGS += -DDATA
CFLAGS += -ggdb
CFLAGS += -O3
#CFLAGS += -O2
CFLAGS += -Wall


OBJ = $(patsubst %.c,%.o,$(SRC))
all: $(EXE)

#$(EXE): $(OBJ) $(KATCP)/libkatcp.a
$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(LIB)

clean: 
	$(RM) -f $(EXE) *.o core vgcore* test-*

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(INC)

install: all
	$(INSTALL) $(EXE) $(PREFIX)/sbin

test-hash: sharedmem.c hash.c
	$(CC) $(CFLAGS) -DTEST_HASH -o $@ $^

test-shm: sharedmem.c
	$(CC) $(CFLAGS) -DTEST_SHARED_MEM -o $@ $^

test-mutex: mutex.c
	$(CC) $(CFLAGS) $(INC) -DTEST_MUTEX -o $@ $^
