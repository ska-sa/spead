#INC = -I$(SSLBUILD)/include,$(KATCP)
#LIB = -lssl 

#KERN = /usr/src/linux-headers-3.2.0-26-generic

#INC = -I$(KERN)/include


EXE = spead 
SRC = sharedmem.c hash.c avltree.c subprocess.c spead_packet.c spead_api.c server.c
HDR = sharedmem.h hash.h avltree.h spead_packet.h spead_api.h server.h

# CFLAGS += -DSTANDALONE
#CFLAGS += -DDEBUG=2
CFLAGS += -DPROCESS
CFLAGS += -DDEBUG
CFLAGS += -DDATA
CFLAGS += -ggdb
CFLAGS += -O3
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
	$(CC) $(CFLAGS) $(INC) -o $@ $^
