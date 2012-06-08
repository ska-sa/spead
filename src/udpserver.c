/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sysexits.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <netdb.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/utsname.h>

#include <arpa/inet.h>
#include <netinet/in.h>

#include "udpserver.h"


static volatile int run = 1;

void handle_us(int signum) 
{
  run = 0;
}

int register_signals_us()
{
  struct sigaction sa;

  sigemptyset(&sa.sa_mask);
  sa.sa_handler   = handle_us;
  sa.sa_flags     = SA_RESTART;

  if (sigaction(SIGINT, &sa, NULL) < 0)
    return -1;

#if 0
  struct sigaction sa;
  sigset_t sigmask;
  int err;
  
  err           = 0;
  sa.sa_flags   = SA_RESTART;
  sa.sa_handler = u_handle;
  
  sigemptyset(&sa.sa_mask);
  err += sigaction(SIGINT, &sa, NULL);
  err += sigaction(SIGTERM, &sa, NULL);
  
  sa.sa_handler = SIG_IGN;

  err += sigaction(SIGPIPE, &sa, NULL);

  sigemptyset(&sigmask);
  sigaddset(&sigmask, SIGINT);
  sigaddset(&sigmask, SIGTERM);
  err += sigprocmask(SIG_BLOCK, &sigmask, NULL);

  if (err < 0)
    return -1;
#endif

  return 0;
}

struct u_server *create_server_us(int (*cdfn)(struct u_client *c))
{
  struct u_server *s;

#if 1
  if (cdfn == NULL)
    return NULL;
  
  s = malloc(sizeof(struct u_server));
  if (s == NULL)
    return NULL;
  
  s->s_fd = 0;

#if 0
  FD_ZERO(&s->s_in);
  FD_ZERO(&s->s_out);

  s->s_hi      = 0;
  s->s_c       = NULL;
  s->s_c_count = 0;
  s->s_br_count= 0;
  s->s_cdfn    = cdfn;
  s->s_tlsctx  = NULL;
  s->s_up_count= 0;
  s->s_sb      = NULL;
  s->s_sb_len  = 0;
#endif

#endif
  return s;
}

void destroy_server_us(struct u_server *s)
{
  if (s)
    free(s);

#ifdef DEBUG
  fprintf(stderr, "%s: destroyed server\n",  __func__);
#endif
}

int startup_server_us(struct u_server *s, char *port)
{
  struct addrinfo hints;
  struct addrinfo *res, *rp;
  int backlog, reuse_addr;

  if (s == NULL || port == NULL)
    return -1;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family     = AF_UNSPEC;
  hints.ai_socktype   = SOCK_DGRAM;
  hints.ai_flags      = AI_PASSIVE;
  hints.ai_protocol   = 0;
  hints.ai_canonname  = NULL;
  hints.ai_addr       = NULL;
  hints.ai_next       = NULL;
  
  if ((reuse_addr = getaddrinfo(NULL, port, &hints, &res)) != 0) {
#ifdef DEBUG
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(reuse_addr));
#endif
    return -1;
  }

  for (rp = res; rp != NULL; rp = rp->ai_next) {
#ifdef DEBUG
    fprintf(stderr, "%s: rp @%p\n", __func__, rp);
#endif

    if (rp->ai_family == AF_INET6)
      break;
  }

  rp = (rp == NULL) ? res : rp;

  s->s_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
  if (s->s_fd < 0){
#ifdef DEBUG
    fprintf(stderr,"wss: error socket\n");
#endif
    freeaddrinfo(res);
    return -1;
  }

  reuse_addr   = 1;

  setsockopt(s->s_fd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr));

  if (bind(s->s_fd, rp->ai_addr, rp->ai_addrlen) < 0){
#ifdef DEBUG
    fprintf(stderr,"wss: error bind on port: %s\n", port);
#endif
    freeaddrinfo(res);
    return -1;
  }

  freeaddrinfo(res);

  backlog      = 10;
#if 0
  memset(&sa, 0, sizeof(struct sockaddr_in));
  sa.sin_family       = AF_INET;
  sa.sin_port         = htons(port);
  sa.sin_addr.s_addr  = INADDR_ANY;
#endif

#if 0
  if (listen(s->s_fd, backlog) < 0) {
#ifdef DEBUG
    fprintf(stderr,"wss: error listen failed\n");  
#endif
    return -1;
  }

  s->s_hi = s->s_fd;
#endif

#ifdef DEBUG
  fprintf(stderr,"wss: server pid: %d running on port: %s\n", getpid(), port);
#endif

  return 0;
}

void shutdown_server_us(struct u_server *s)
{

#if 0
  struct u_client *c;

  if (s){
    while (s->s_c_count > 0){
      c = s->s_c[0];
      if (disconnect_client_ws(s, c) < 0){
#ifdef DEBUG
        fprintf(stderr, "wss: error server disconnect client\n");
#endif
      }
    }
    
    if (s->s_tlsctx){
      SSL_CTX_free(s->s_tlsctx);
    }

    if (shutdown(s->s_fd, SHUT_RDWR) < 0){
#ifdef DEBUG
      fprintf(stderr, "wss: error server shutdown: %s\n", strerror(errno));
#endif
    }

    if (close(s->s_fd) < 0){
#ifdef DEBUG
      fprintf(stderr, "wss: error server shutdown: %s\n", strerror(errno));
#endif
    }
    destroy_server_ws(s);
  }
  
#endif

#ifdef DEBUG
  fprintf(stderr, "%s: server shutdown complete\n", __func__);
#endif

} 

int socks_io_us(struct u_server *s) 
{
#if 0
  struct ws_client *c;
  int i;
  
  if (s == NULL)
    return -1;

  /*check the server listen fd for a new connection*/
  if (FD_ISSET(s->s_fd, &s->s_in)) {
#ifdef DEBUG
    fprintf(stderr,"wss: new incomming connection\n");
#endif
    if (handle_new_client_ws(s) < 0){
#ifdef DEBUG
      fprintf(stderr,"wss: error handle new clientn\n");
#endif
    }
  }

  for (i=0; i<s->s_c_count; i++){
    c = s->s_c[i];
    if (c != NULL){

      if (FD_ISSET(c->c_fd, &s->s_out)){
        if (send_client_data_ws(s, c) < 0){
#ifdef DEBUG
          fprintf(stderr, "wss: send client data error\n");
#endif
        }
      }

      if (FD_ISSET(c->c_fd, &s->s_in)) {
        if (get_client_data_ws(s, c) < 0){
#ifdef DEBUG
          fprintf(stderr, "wss: get client data error\n");
#endif
        }
      } 
        
    }
  }
#endif

  return 0;
}

void build_socket_set_us(struct u_server *s)
{
#if 0
  struct ws_client *c;
  int i, fd;

  if (s == NULL)
    return;

  FD_ZERO(&s->s_in);
  FD_ZERO(&s->s_out);
  FD_SET(s->s_fd, &s->s_in);
  
  //FD_SET(STDIN_FILENO, &s->insocks);
  for (i=0; i<s->s_c_count; i++){
    
    c = s->s_c[i];

    if (c != NULL){
      
      fd = c->c_fd;
      
      if (fd > 0) {
        FD_SET(fd, &s->s_in);

        if (fd > s->s_hi)
          s->s_hi = fd;
        
        if (c->c_sb_len > 0){
#ifdef DEBUG
          fprintf(stderr, "wss: must send to [%d] %dbytes\n", c->c_fd, c->c_sb_len);
#endif
          FD_SET(fd, &s->s_out);
        }

      }
    }
  }
#endif
}

int run_loop_us(struct u_server *s)
{
  struct sockaddr_storage peer_addr;
  socklen_t peer_addr_len;
#if 0
  sigset_t empty_mask;
#endif
  char buf[BUF_SIZE];
  ssize_t nread;
  int rtn;
  
  if (s == NULL)
    return -1;

#if 0
  sigemptyset(&empty_mask);
#endif

  while (run) {
    
    peer_addr_len = sizeof(struct sockaddr_storage); 
    nread = recvfrom(s->s_fd, buf, BUF_SIZE, 0, (struct sockaddr *) &peer_addr, &peer_addr_len);
    if (nread == -1)
      continue;               /* Ignore failed request */

    char host[NI_MAXHOST], service[NI_MAXSERV];

    rtn = getnameinfo((struct sockaddr *) &peer_addr, peer_addr_len, host, NI_MAXHOST, service, NI_MAXSERV, NI_NUMERICSERV);
#ifdef DEBUG
    if (rtn == 0)
      printf("Received %ld bytes from %s:%s\n", (long) nread, host, service);
    else
      fprintf(stderr, "getnameinfo: %s\n", gai_strerror(rtn));
#endif

    if (sendto(s->s_fd, buf, nread, 0, (struct sockaddr *) &peer_addr, peer_addr_len) != nread)
#ifdef DEBUG
      fprintf(stderr, "Error sending response\n");
#endif
    
#if 0
    build_socket_set_ws(s);  

    if (pselect(s->s_hi + 1, &s->s_in, &s->s_out, (fd_set *) NULL, NULL, &empty_mask) < 0) { 
      switch(errno){
        case EPIPE:
#ifdef DEBUG
          fprintf(stderr, "wss: EPIPE: %s\n", strerror(errno));
#endif
        case EINTR:
        case EAGAIN:
          break;
        default:
#ifdef DEBUG
          fprintf(stderr,"wss: select encountered an error: %s\n", strerror(errno)); 
#endif
          return -1;
      }
    }
    else {
      if (socks_io_ws(s) < 0) {
        //return -1; 
#ifdef DEBUG
        fprintf(stderr, "wss: error in read_socks_ws\n");
#endif
      }
    }
#endif

  }

  return 0;
}

int register_client_handler_server(int (*client_data_fn)(struct u_client *c), char *port)
{
  struct u_server *s;
  
  if (register_signals_us() < 0){
#ifdef DEBUG
    fprintf(stderr, "wss: error register signals\n");
#endif
    return -1;
  }

  s = create_server_us(client_data_fn);
  if (s == NULL){
#ifdef DEBUG
    fprintf(stderr, "wss: error could not create server\n");
#endif
    return -1;
  }

  if (startup_server_us(s, port) < 0){
#ifdef DEBUG
    fprintf(stderr,"wss: error in startup\n");
#endif
    shutdown_server_us(s);
    return -1;
  }

  if (run_loop_us(s) < 0){ 
#ifdef DEBUG
    fprintf(stderr,"wss: error during run\n");
#endif
    shutdown_server_us(s);
    return -1;
  }
  
  shutdown_server_us(s);

#ifdef DEBUG
  fprintf(stderr,"wss: server exiting\n");
#endif

  return 0;
}


int capture_client_data(struct u_client *c)
{

}

int main(int argc, char *argv[])
{
  
  return register_client_handler_server(&capture_client_data , PORT);
}
