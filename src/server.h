/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */
#ifndef UDP_SERVER_H
#define UDP_SERVER_H

#define PORT      "8888"

struct u_server {
  long s_cpus;
  struct u_child **s_cs;
  int s_fd;
  uint64_t s_bc;
};

struct u_child {
  pid_t c_pid;
  int c_fd;
};

void destroy_child_sp(struct u_child *c);
struct u_child *fork_child_sp(struct u_server *s, int (*call)(struct u_server *s));


#endif