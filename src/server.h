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
  struct u_child **s_cs;
  int s_fd;
  int s_hpcount;
  int s_hdcount;
  uint64_t s_bc;
  struct spead_heap_store *s_hs;
  struct spead_api_module *s_mod;
#ifndef IKATCP
  struct katcl_line *s_kl;
#endif
};

struct u_child {
  pid_t c_pid;
  int c_fd;
};

void destroy_child_sp(struct u_child *c);
struct u_child *fork_child_sp(struct u_server *s, int (*call)(struct u_server *s, struct spead_api_module *m, int cfd));

void print_format_bitrate(char x, uint64_t bps);

#endif
