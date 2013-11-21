#ifndef MCS_H
#define MCS_H

#include "mutex.h"

struct mcs_rx {
  mutex                 s_m;
  long                  s_cpus;
  struct data_file      *s_f;
  struct spead_socket   *s_x;
  struct spead_workers  *s_w;
  uint64_t              s_pc;
  uint64_t              s_bc;
};

int mcs_rx_packet_callback(unsigned char *pkt, int size);

#endif
