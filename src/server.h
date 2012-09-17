/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */
#ifndef UDP_SERVER_H
#define UDP_SERVER_H

#include <stdint.h>
#include <katcl.h>

#include "mutex.h"

#define PORT      "8888"

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
  struct katcl_line *s_kl;
};

struct u_child {
  pid_t c_pid;
  int c_fd;
};

void destroy_child_sp(struct u_child *c);
struct u_child *fork_child_sp(struct u_server *s, int (*call)(struct u_server *s, int cfd));

void print_format_bitrate(char x, uint64_t bps);

/*modules api*/

#define SAPI_CALLBACK "spead_api_callback"
#define SAPI_SETUP    "spead_api_setup"
#define SAPI_DESTROY  "spead_api_destroy"

struct spead_api_module {
  void *m_handle;
  int (*m_cdfn)();
  int (*m_destroy)(void *data);
  void *m_data;
};


struct spead_api_module *load_api_user_module(char *mod);
void unload_api_user_module(struct spead_api_module *m);

#endif
