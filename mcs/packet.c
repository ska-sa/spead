/* MULTICORE UDP RECEIVER*/
/* Released under the GNU GPLv3 - see COPYING */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>


int mcs_rx_packet_callback(unsigned char *pkt, int size)
{
  
#ifdef DEBUG
  fprintf(stderr, "%s: PID [%d] packet <%p> size [%d]\n", __func__, getpid(), pkt, size);
#endif




  return 0;
}


