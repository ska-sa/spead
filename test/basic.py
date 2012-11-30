#!/usr/bin/python

import spead, logging, numpy

PORT  = 8888
#SHAPE = (100,250)
#SHAPE = (50,50)
SHAPE = (100,10)
logging.basicConfig(level=logging.DEBUG)

def send():
  tx = spead.Transmitter(spead.TransportUDPtx('127.0.0.1', PORT))
  ig = spead.ItemGroup()
  ig.add_item(name="var1", description="this is a variable", init_val=333)
 # data0 = numpy.ones(SHAPE)
 # ig.add_item(name="data1", description="these are some ones", shape=SHAPE, fmt='i\x00\x00\x20', init_val=data0)
  #ig.add_item(name="data2", description="these are some zeros", shape=SHAPE, fmt='i\x00\x00\x20', init_val=data0)
  #ig.add_item(name="data3", description="these are some zeros", shape=SHAPE, fmt='i\x00\x00\x20', init_val=data0)
  #ig.add_item(name="data4", description="these are some zeros", shape=SHAPE, fmt='i\x00\x00\x20', init_val=data0)

  tx.send_heap(ig.get_heap())
  tx.end()

send()
