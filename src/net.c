/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>

#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>

#include "spead_api.h"

#define BIG_BUF 1024*1024*1024;

void destroy_spead_socket(struct spead_socket *x)
{
  if (x){
    if(x->x_res) 
      freeaddrinfo(x->x_res);
    
    if (x->x_fd)
      close(x->x_fd);

    free(x);
  }
} 

struct spead_socket *create_spead_socket(char *host, char *port)
{
  struct spead_socket *x;

  struct addrinfo hints;
  uint64_t reuse_addr;

  x = malloc(sizeof(struct spead_socket));
  if (x == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: logic cannot malloc\n", __func__);
#endif
    return NULL;
  }

  x->x_host   = host;
  x->x_port   = port;
  x->x_res    = NULL;
  x->x_active = NULL;
  x->x_fd     = 0;
  x->x_mode   = XSOCK_NONE;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family     = AF_UNSPEC;
  hints.ai_socktype   = SOCK_DGRAM;
  hints.ai_flags      = AI_PASSIVE;
  hints.ai_protocol   = 0;
  hints.ai_canonname  = NULL;
  hints.ai_addr       = NULL;
  hints.ai_next       = NULL;
  
  if ((reuse_addr = getaddrinfo(host, port, &hints, &(x->x_res))) != 0) {
#ifdef DEBUG
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(reuse_addr));
#endif
    destroy_spead_socket(x);
    return NULL;
  }

  for (x->x_active = x->x_res; x->x_active != NULL; x->x_active = x->x_active->ai_next) {
#if DEBUG>1
    fprintf(stderr, "%s: res (%p) with: %d\n", __func__, x->x_active, x->x_active->ai_protocol);
#endif
    if (x->x_active->ai_family == AF_INET6)
      break;
  }

  x->x_active = (x->x_active == NULL) ? x->x_res : x->x_active;

  x->x_fd = socket(x->x_active->ai_family, x->x_active->ai_socktype, x->x_active->ai_protocol);
  if (x->x_fd < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error socket\n", __func__);
#endif
    destroy_spead_socket(x);
    return NULL;
  }
  
  
  reuse_addr   = 1;
  setsockopt(x->x_fd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr));

  
  return x;
}


int bind_spead_socket(struct spead_socket *x)
{
  uint64_t recvbuf;

  if (x == NULL || x->x_active == NULL)
    return -1;

  recvbuf = BIG_BUF;
  if (setsockopt(x->x_fd, SOL_SOCKET, SO_RCVBUF, &recvbuf, sizeof(recvbuf)) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error cannot increase recv buf setsockopt: %s\n", __func__, strerror(errno));
#endif
  }
  
  if (bind(x->x_fd, x->x_active->ai_addr, x->x_active->ai_addrlen) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error bind to %s\n", __func__, x->x_port);
#endif
    return -1;
  }

  x->x_mode = (x->x_mode == XSOCK_NONE) ? XSOCK_BOUND : XSOCK_BOTH;

  return 0;
}


int connect_spead_socket(struct spead_socket *x)
{
  uint64_t sendbuf;

  if (x == NULL || x->x_active == NULL)
    return -1;

  sendbuf = BIG_BUF;
  if (setsockopt(x->x_fd, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf)) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error cannot increase send buf setsockopt: %s\n", __func__, strerror(errno));
#endif
  }

  if (connect(x->x_fd, x->x_active->ai_addr, x->x_active->ai_addrlen) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error connect to %s:%s\n", __func__, x->x_host, x->x_port);
#endif
    return -1;
  }
  
  x->x_mode = (x->x_mode == XSOCK_NONE) ? XSOCK_CONNECTED : XSOCK_BOTH;
  
  return 0;
}

int set_broadcast_opt_spead_socket(struct spead_socket *x)
{
  int opt;

  if (x == NULL)
    return -1;
  
  opt = 1;
  if (setsockopt(x->x_fd, SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt)) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error set broadcast setsockopt: %s\n", __func__, strerror(errno));
#endif
    return -1;
  }
  return 0;
}

int get_fd_spead_socket(struct spead_socket *x)
{
  if (x == NULL)
    return -1;

  return x->x_fd;
}

struct addrinfo *get_addr_spead_socket(struct spead_socket *x)
{
  if (x == NULL)
    return NULL;

  return x->x_active;
}

int send_packet(void *data, struct spead_packet *p)
{
  int sb, sfd;
  struct addrinfo *dst;
  struct spead_socket *x;

  x = data;

  if (x == NULL || p == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }

  sfd = get_fd_spead_socket(x);
  dst = get_addr_spead_socket(x);

  if (sfd <=0 || dst == NULL){
    return -1;
  }

  sb = sendto(sfd, p->data, SPEAD_MAX_PACKET_LEN, 0, dst->ai_addr, dst->ai_addrlen);
  if (sb < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: sendto err (%s)\n", __func__, strerror(errno));
#endif
    return -1;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: packet (%p) sb [%d] bytes\n", __func__, p, sb);
#endif

  return 0;
}

