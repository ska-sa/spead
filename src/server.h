/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#ifndef UDP_SERVER_H
#define UDP_SERVER_H

#include <stdint.h>

#ifndef IKATCP
#include <katcl.h>
#endif

#include "mutex.h"

#define PORT      "8888"
#define BUF       100

struct u_server {
  mutex s_m;
  long s_cpus;
  struct data_file *s_f;
#if 0
  struct u_child **s_cs;
  struct spead_api_module *s_mod;
#endif 
  struct spead_socket *s_x;
  struct spead_workers *s_w;

  int s_fd;
  int s_hpcount;
  int s_hdcount;
  uint64_t s_bc;
  uint64_t s_pc;
  struct spead_heap_store *s_hs;
  struct spead_pipeline *s_p;
#ifndef IKATCP
  struct katcl_line *s_kl;
#endif
};

void print_format_bitrate(struct u_server *s, char x, uint64_t bps);

#endif
