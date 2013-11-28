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
#include "tx.h"

#define BIG_BUF 1024*1024*1024;

void destroy_spead_socket(struct spead_socket *x)
{
  if (x){

    if (unset_multicast_receive_opts_spead_socket(x) == 0){
      free(x->x_grp);
    }

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
  x->x_grp    = NULL;

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

#ifdef DEBUG
  fprintf(stderr, "%s: socket at %s:%s\n", __func__, host, port);
#endif
  
  return x;
}

struct spead_socket *create_raw_ip_spead_socket(char *host)
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
  x->x_port   = NULL;
  x->x_res    = NULL;
  x->x_active = NULL;
  x->x_fd     = 0;
  x->x_mode   = XSOCK_NONE;
  x->x_grp    = NULL;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family     = AF_UNSPEC;
  hints.ai_socktype   = SOCK_RAW;
  hints.ai_flags      = AI_PASSIVE;
  hints.ai_protocol   = 155;
  hints.ai_canonname  = NULL;
  hints.ai_addr       = NULL;
  hints.ai_next       = NULL;

  if ((reuse_addr = getaddrinfo(host, NULL, &hints, &(x->x_res))) != 0) {
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
    fprintf(stderr,"%s: error socket (%s)\n", __func__, strerror(errno));
#endif
    destroy_spead_socket(x);
    return NULL;
  }
  
  reuse_addr   = 1;
  setsockopt(x->x_fd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr));

  return x;
}

struct spead_socket *create_tcp_socket(char *host, char *port)
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
  x->x_grp    = NULL;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family     = AF_UNSPEC;
  hints.ai_socktype   = SOCK_STREAM;
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

#ifdef DEBUG
  fprintf(stderr, "%s: tcp socket at %s:%s\n", __func__, host, port);
#endif
  
  return x;
}


int bind_spead_socket(struct spead_socket *x)
{
  int recvbuf;
  socklen_t size;

  if (x == NULL || x->x_active == NULL)
    return -1;

  recvbuf = BIG_BUF;
  if (setsockopt(x->x_fd, SOL_SOCKET, SO_RCVBUF, &recvbuf, sizeof(recvbuf)) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error cannot increase recv buf setsockopt: %s\n", __func__, strerror(errno));
#endif
  }
  
  recvbuf = 0;
  size = sizeof(socklen_t);

#ifdef DEBUG
  getsockopt(x->x_fd, SOL_SOCKET, SO_RCVBUF, &recvbuf, &size);
  fprintf(stderr, "%s: RCVBUF is %d\n", __func__, recvbuf);
#endif

  if (bind(x->x_fd, x->x_active->ai_addr, x->x_active->ai_addrlen) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error bind to %s\n", __func__, x->x_port);
#endif
    return -1;
  }

  x->x_mode = (x->x_mode == XSOCK_NONE) ? XSOCK_BOUND : XSOCK_BOTH;

  return 0;
}

int listen_spead_socket(struct spead_socket *x)
{
  if (x == NULL || x->x_active == NULL)
    return -1;

  if (listen(x->x_fd, XSOCK_LISTEN_BACKLOG) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: Unable to listen on sock (%s)\n", __func__, strerror(errno));
#endif
    return -1;
  }
  
  return 0;
}

struct spead_client *accept_spead_socket(struct spead_socket *x)
{
  struct spead_client *c;

  if (x == NULL || x->x_active == NULL)
    return NULL;
  
  c = malloc(sizeof(struct spead_client));
  if (c == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: logic error cannot allocate memory\n", __func__);
#endif
    return NULL;
  }
  
  memset(c, 0, sizeof(struct spead_client));
  
  c->c_len = sizeof(struct sockaddr_in);

  c->c_fd = accept(x->x_fd, (struct sockaddr *) &(c->c_addr), &(c->c_len));
  if (c->c_fd < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: accept error (%s)\n", __func__, strerror(errno));
#endif
#if 0
    shared_free(c, sizeof(struct spead_client));
#endif
    free(c);
    return NULL;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: client connected from [%s:%d]\n", __func__, inet_ntoa(c->c_addr.sin_addr), ntohs(c->c_addr.sin_port));
#endif

  return c;
}

void destroy_spead_client(void *data)
{
  struct spead_client *c;

  c = data;
  if (c){
    shutdown(c->c_fd, SHUT_RDWR);
    close(c->c_fd);
#if 0
    shared_free(c, sizeof(struct spead_client));
#endif
    free(c);
#ifdef DEBUG
    fprintf(stderr, "%s: client destroyed\n", __func__);
#endif
  }
}

int compare_spead_clients(const void *v1, const void *v2)
{
  if (*(int *)v1 < *(int *)v2)
    return -1;
  else if (*(int *)v1 > *(int *)v2)
    return 1;
  return 0;
}


int connect_spead_socket(struct spead_socket *x)
{
  int sendbuf;
  socklen_t size;

  if (x == NULL || x->x_active == NULL)
    return -1;

  sendbuf = BIG_BUF;
  if (setsockopt(x->x_fd, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf)) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error cannot increase send buf setsockopt: %s\n", __func__, strerror(errno));
#endif
  }
  
  sendbuf = 0;
  size = sizeof(socklen_t);
  
#ifdef DEBUG
  getsockopt(x->x_fd, SOL_SOCKET, SO_SNDBUF, &sendbuf, &size);
  fprintf(stderr, "%s: SNDBUF is %d\n", __func__, sendbuf);
#endif

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

int set_multicast_send_opts_spead_socket(struct spead_socket *x, char *host)
{ 
  int loop;
  struct in_addr loco;

  if (x == NULL || host == NULL)
    return -1;

  loop = 0;

  if (setsockopt(x->x_fd, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop)) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error disable multicast loop setsockopt: %s\n", __func__, strerror(errno));
#endif
    return -1;
  }

  /*TODO: use inet_pton for ipv6 */
  loco.s_addr = inet_addr(host);
  if (setsockopt(x->x_fd, IPPROTO_IP, IP_MULTICAST_IF, (char*) &loco, sizeof(loco)) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: error setting mcast iface: %s\n", __func__, strerror(errno));
#endif
    return -1;
  }

  return 0;
}

int set_multicast_receive_opts_spead_socket(struct spead_socket *x, char *grp, char *interface)
{
  struct ip_mreq *group;

  if (grp == NULL || interface == NULL || x == NULL)
    return -1;
  
  x->x_grp = malloc(sizeof(struct ip_mreq));
  if (x->x_grp == NULL)
    return -1;
  
  group = x->x_grp;

  /*TODO: use inet_pton for ipv6 */
  group->imr_multiaddr.s_addr = inet_addr(grp);
  group->imr_interface.s_addr = inet_addr(interface);
  if (setsockopt(x->x_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, group, sizeof(struct ip_mreq)) < 0) {
#ifdef DEBUG
    fprintf(stderr, "%s: error adding mcast group membership: %s\n", __func__, strerror(errno));
#endif
    return -1;
  }

  return 0;
}

int unset_multicast_receive_opts_spead_socket(struct spead_socket *x)
{
  if (x == NULL || x->x_grp == NULL){
    return -1;
  }
  
  if (setsockopt(x->x_fd, IPPROTO_IP, IP_DROP_MEMBERSHIP, x->x_grp, sizeof(struct ip_mreq)) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: error adding mcast group membership: %s\n", __func__, strerror(errno));
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

int send_packet_spead_socket(void *data, struct spead_packet *p)
{
  int sb, sfd, mw;
  struct addrinfo *dst;
  struct spead_tx     *tx;
  struct spead_socket *x;

  tx = data;

  if (tx == NULL || tx->t_x == NULL || p == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }

  x = tx->t_x;

  sfd = get_fd_spead_socket(x);
  dst = get_addr_spead_socket(x);

  if (sfd <=0 || dst == NULL){
    return -1;
  }

  mw = SPEAD_HEADERLEN + p->n_items * SPEAD_ITEMLEN + p->payload_len;

#if 0 
def DEBUG
  print_data(p->data, mw);
#endif

  //mw = SPEAD_MAX_PACKET_LEN;

  sb = sendto(sfd, p->data, mw, 0, dst->ai_addr, dst->ai_addrlen);
  if (sb < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: packet (%p) size [%d] sb [%d] bytes\n", __func__, p, mw, sb);
#endif
    fprintf(stderr, "%s: sendto err (\033[31m%s\033[0m)\n", __func__, strerror(errno));
    return -1;
  }

  lock_mutex(&(tx->t_m));
  tx->t_pc++;
  unlock_mutex(&(tx->t_m));

#if DEBUG>1
  fprintf(stderr, "%s: packet (%p) size [%d] sb [%d] bytes\n", __func__, p, mw, sb);
#endif

  return 0;
}

int send_raw_data_spead_socket(void *obj, void *data, uint64_t len)
{
  int sb, sfd;
  struct addrinfo     *dst;
  struct spead_tx     *tx;
  struct spead_socket *x;

  tx = obj;

  if (tx == NULL || tx->t_x == NULL || data == NULL || len <= 0){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }

  x = tx->t_x;

  sfd = get_fd_spead_socket(x);
  dst = get_addr_spead_socket(x);

  if (sfd <=0 || dst == NULL){
    return -1;
  }

  sb = sendto(sfd, data, len, 0, dst->ai_addr, dst->ai_addrlen);
  if (sb < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: packet (%p) size [%ld] sb [%d] bytes\n", __func__, data, len, sb);
#endif
    fprintf(stderr, "%s: sendto err (\033[31m%s\033[0m)\n", __func__, strerror(errno));
    return -1;
  }

  lock_mutex(&(tx->t_m));
  tx->t_pc++;
  unlock_mutex(&(tx->t_m));

  return 0;
}
