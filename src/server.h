/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */
#ifndef UDP_SERVER_H
#define UDP_SERVER_H

#define PORT      "8888"

struct u_server {
  long s_cpus;
  pid_t *s_sps;
  int s_fd;
  uint64_t s_bc;
};

struct u_client {

};

pid_t fork_child_sp(struct u_server *s, int (*call)(struct u_server *s));


#endif
