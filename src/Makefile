#INC = -I$(SSLBUILD)/include,$(KATCP)
#LIB = -lssl 

EXE = spead 
SRC = udpserver.c spead_packet.c
HDR = udpserver.h spead_packet.h

# CFLAGS += -DSTANDALONE
CFLAGS += -DDEBUG
CFLAGS += -ggdb


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
