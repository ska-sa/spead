#INC = -I$(SSLBUILD)/include,$(KATCP)
#LIB = -lssl 

EXE = spead 
SRC = avltree.c sp.c spead_packet.c spead_api.c server.c
HDR = avltree.h spead_packet.h spead_api.h server.h

# CFLAGS += -DSTANDALONE
#CFLAGS += -DDEBUG
CFLAGS += -DDATA
CFLAGS += -ggdb
CFLAGS += -O3


OBJ = $(patsubst %.c,%.o,$(SRC))
all: $(EXE)

#$(EXE): $(OBJ) $(KATCP)/libkatcp.a
$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(LIB)

clean: 
	$(RM) -f $(EXE) *.o core vgcore*

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(INC)

install: all
	$(INSTALL) $(EXE) $(PREFIX)/sbin
