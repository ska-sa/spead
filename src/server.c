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
#include <wait.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/utsname.h>
#include <sys/time.h>

#include <arpa/inet.h>
#include <netinet/in.h>

#include "spead_api.h"
#include "server.h"
#include "hash.h"


static volatile int run = 1;

int sub_time(struct timeval *delta, struct timeval *alpha, struct timeval *beta)
{
  if(alpha->tv_usec < beta->tv_usec){
    if(alpha->tv_sec <= beta->tv_sec){
      delta->tv_sec  = 0;
      delta->tv_usec = 0;
      return -1;
    }
    delta->tv_sec  = alpha->tv_sec - (beta->tv_sec + 1);
    delta->tv_usec = (1000000 + alpha->tv_usec) - beta->tv_usec;
  } else {
    if(alpha->tv_sec < beta->tv_sec){
      delta->tv_sec  = 0;
      delta->tv_usec = 0;
      return -1;
    }
    delta->tv_sec  = alpha->tv_sec  - beta->tv_sec;
    delta->tv_usec = alpha->tv_usec - beta->tv_usec;
  }
#ifdef DEBUG
  if(delta->tv_usec >= 1000000){
    fprintf(stderr, "major logic failure: %lu.%06lu-%lu.%06lu yields %lu.%06lu\n", alpha->tv_sec, alpha->tv_usec, beta->tv_sec, beta->tv_usec, delta->tv_sec, delta->tv_usec);
    abort();
  }
#endif
  return 0;
}

void print_time(struct timeval *result, int bytes)
{
  int64_t us;
  int64_t bpus;

  us = result->tv_sec*1000*1000 + result->tv_usec;
  //bpus = bytes / us * 1000 * 1000 / 1024 / 1024;
  bpus = (bytes / us) * 1000 * 1000;

#ifdef DATA
  fprintf(stderr, "[%d] component time: %lu.%06lds B/s: %ld\n", getpid(), result->tv_sec, result->tv_usec, bpus);
#endif
}


void handle_us(int signum) 
{
  run = 0;
}

int register_signals_us()
{
  struct sigaction sa;

  sigfillset(&sa.sa_mask);
  sa.sa_handler   = handle_us;
  sa.sa_flags     = 0;

  if (sigaction(SIGINT, &sa, NULL) < 0)
    return -1;

  if (sigaction(SIGTERM, &sa, NULL) < 0)
    return -1;

  return 0;
}

struct u_server *create_server_us(int (*cdfn)(), long cpus)
{
  struct u_server *s;

#if 1
  if (cdfn == NULL)
    return NULL;

  if (cpus < 1){
#ifdef DEBUG
    fprintf(stderr, "%s: must have at least 1 cpu\n", __func__);
#endif
    return NULL;
  }
  
  s = malloc(sizeof(struct u_server));
  if (s == NULL)
    return NULL;
  
  s->s_fd   = 0;
  s->s_bc   = 0;
  s->s_cpus = cpus;
  s->s_cs  = NULL;

#endif
  return s;
}

void destroy_server_us(struct u_server *s)
{
  int i;

  if (s){
    
    if (s->s_cs){

      /*WARNING: review code --ed should be ok now*/
      for (i=0; i < s->s_cpus; i++){
        destroy_child_sp(s->s_cs[i]);
      }

      free(s->s_cs);
    }

    destroy_store_hs(s->s_hs);

    free(s);
  }
#ifdef DEBUG
  fprintf(stderr, "%s: destroyed server\n",  __func__);
#endif
}

int startup_server_us(struct u_server *s, char *port)
{
  struct addrinfo hints;
  struct addrinfo *res, *rp;
  int reuse_addr;

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
    fprintf(stderr, "%s: res (%p) with: %d\n", __func__, rp, rp->ai_protocol);
#endif
    if (rp->ai_family == AF_INET6)
      break;
  }

  rp = (rp == NULL) ? res : rp;

  s->s_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
  if (s->s_fd < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error socket\n", __func__);
#endif
    freeaddrinfo(res);
    return -1;
  }

  reuse_addr   = 1;
  setsockopt(s->s_fd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr));

  reuse_addr = 700*1024*1024;
  if (setsockopt(s->s_fd, SOL_SOCKET, SO_RCVBUF, &reuse_addr, sizeof(reuse_addr)) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error setsockopt: %s\n", __func__, strerror(errno));
#endif
  }

  if (bind(s->s_fd, rp->ai_addr, rp->ai_addrlen) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error bind on port: %s\n", __func__, port);
#endif
    freeaddrinfo(res);
    return -1;
  }

  freeaddrinfo(res);

#ifdef DEBUG
  fprintf(stderr,"\tserver pid:\t%d\n\tport:\t\t%s\n", getpid(), port);
#endif

  sleep(10);

  return 0;
}

void shutdown_server_us(struct u_server *s)
{
  if (s){
    if (close(s->s_fd) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error server shutdown: %s\n", __func__, strerror(errno));
#endif
    }
    destroy_server_us(s);
  }
#ifdef DEBUG
  fprintf(stderr, "%s: server shutdown complete\n", __func__);
#endif
} 

int worker_task_us(struct u_server *s)
{
  struct spead_packet *p;
  struct spead_heap_store *hs;

  struct timeval prev, now, delta;
  struct sockaddr_storage peer_addr;
  socklen_t peer_addr_len;
  ssize_t nread;
  int rtn;
  uint64_t rcount, bcount;
  pid_t pid;

  rcount = 0;
  bcount = 0;
  p      = NULL;
  pid    = getpid();

  if (s == NULL || s->s_hs == NULL)
    return -1;

  hs = s->s_hs;

#ifdef DEBUG
  fprintf(stderr, "\t  CHILD\t\t[%d]\n", pid);
#endif

#if 0
  p = create_spead_packet();
  if (p == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot allocate memory for spead_packet\n", __func__);
#endif
    return -1;
  }
#endif

  while (run) {

#if 1
    p = get_data_hash_table(hs->s_hash, rcount);
    if (p == NULL){
      continue;
    }


    gettimeofday(&prev, NULL);
    rcount++;
    peer_addr_len = sizeof(struct sockaddr_storage); 

#if 0
    p = create_spead_packet();
    if (p == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: cannot allocate memory for spead_packet\n", __func__);
#endif
      return -1;
    }
#endif

    nread = recvfrom(s->s_fd, p->data, SPEAD_MAX_PACKET_LEN, 0, (struct sockaddr *) &peer_addr, &peer_addr_len);
    if (nread <= 0){
#ifdef DEBUG
      fprintf(stderr, "%s: rcount [%lu] unable to recvfrom: %s\n", __func__, rcount, strerror(errno));
#endif
      continue;
    }

    bcount += nread;

#if 0
    fwrite(p->data, 1, nread, stdout);
#endif

    if (process_packet_hs(s->s_hs, p) < 0){
#if 0
      destroy_spead_packet(p);
#endif
      continue; 
    }


    gettimeofday(&now, NULL);
//    sub_time(&delta, &now, &prev);
//    print_time(&delta, nread);
#endif

  }

#ifdef DEBUG
  fprintf(stderr, "%s:\tCHILD[%d]: exiting with bytes: %lu\n", __func__, getpid(), bcount);
#endif
#if 0
  destroy_spead_packet(p);
#endif
  return 0;
}

int add_child_us(struct u_server *s, struct u_child *c, int size)
{
  if (s == NULL || c == NULL)
    return -1;

  s->s_cs = realloc(s->s_cs, sizeof(struct u_child*) * (size+1));
  if (s->s_cs == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: logic error cannot realloc for worker list\n", __func__);
#endif 
    return -1;
  }

  s->s_cs[size] = c;
  
  return size + 1;
}

int spawn_workers_us(struct u_server *s)
{
  struct spead_heap_store *hs;
  struct u_child *c;
  int status, i, hi_fd;
  fd_set ins;
  pid_t sp;

  hs = NULL;
  
  if (s == NULL)
    return -1;
 
  hs = create_store_hs(1000);
  if (hs == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot create spead_heap_store\n", __func__);
#endif
    return -1;
  }

  s->s_hs = hs;

  i = 0;

#ifdef DEBUG
  fprintf(stderr, "\tworkers:\t%ld\n", s->s_cpus);
#endif
  
#if 1
  do {

    c = fork_child_sp(s, &worker_task_us);
    if (c == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: fork_child_sp fail\n", __func__);
#endif
      continue;
    }

    if (add_child_us(s, c, i) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: could not store worker pid [%d]\n", __func__, sp);
#endif
      destroy_child_sp(c);
    } else
      i++;

  } while (i < s->s_cpus);


#if 0
def DEBUG
  fprintf(stderr, "%s: PARENT about to loop\n", __func__);
#endif
  
  hi_fd = 0;

  while(run) {

#if 0
    FD_ZERO(&ins);
    
    for (i=0; i<s->s_cpus; i++){
      c = s->s_cs[i];
      if (c != NULL){
        if (c->c_fd > 0){
          FD_SET(c->c_fd, &ins);
          if (c->c_fd > hi_fd){
            hi_fd = c->c_fd;
          }
        }
      }
    }

    if (pselect(fd_hi + 1, &ins, (fd_set *) NULL, (fd_set *) NULL, NULL, &empty_mask) < 0){
      switch(errno){
        case EAGAIN:
        case EINTR:
          break;
        default:
#ifdef DEBUG
          fprintf(stderr, "%s: pselect error\n", __func__);
#endif    
          run = 0;
          break;
      }
    }
#endif
    
    sp = waitpid(-1, &status, 0);

#ifdef DEBUG
    fprintf(stderr,"%s: PARENT waitpid [%d]\n", __func__, sp);
#endif

    if (WIFEXITED(status)) {
#ifdef DEBUG
      fprintf(stderr, "exited, status=%d\n", WEXITSTATUS(status));
#endif
      run = 0;
    } else if (WIFSIGNALED(status)) {
#ifdef DEBUG
      fprintf(stderr, "killed by signal %d\n", WTERMSIG(status));
#endif
      run = 0;
    } else if (WIFSTOPPED(status)) {
#ifdef DEBUG
      fprintf(stderr, "stopped by signal %d\n", WSTOPSIG(status));
#endif
      run = 0;
    } else if (WIFCONTINUED(status)) {
#ifdef DEBUG
      fprintf(stderr, "continued\n");
#endif
      run = 0;
    }
      
  }

#endif

#ifdef DEBUG
  fprintf(stderr, "%s: final recv count: %ld bytes\n", __func__, s->s_bc);
#endif

#if 0 
  destroy_store_hs(hs);
#endif

  return 0;
}

int register_client_handler_server(int (*client_data_fn)(), char *port, long cpus)
{
  struct u_server *s;
  
  if (register_signals_us() < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: error register signals\n", __func__);
#endif
    return -1;
  }

  s = create_server_us(client_data_fn, cpus);
  if (s == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error could not create server\n", __func__);
#endif
    return -1;
  }

#if 1
  if (startup_server_us(s, port) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error in startup\n", __func__);
#endif
    shutdown_server_us(s);
    return -1;
  }
#endif

  if (spawn_workers_us(s) < 0){ 
#ifdef DEBUG
    fprintf(stderr,"%s: error during run\n", __func__);
#endif
    shutdown_server_us(s);
    return -1;
  }
  
  shutdown_server_us(s);

#ifdef DEBUG
  fprintf(stderr,"%s: server exiting\n", __func__);
#endif

  return 0;
}


int capture_client_data()
{

}

int main(int argc, char *argv[])
{
  long cpus;
  int i, j, c;
  char *port;

  i = 1;
  j = 1;

  port = PORT;
  cpus = sysconf(_SC_NPROCESSORS_ONLN);

#ifdef DEBUG
  fprintf(stderr, "\nS-P-E-A-D S.E.R.V.E.R\n\n");
#endif  

  while (i < argc){
    if (argv[i][0] == '-'){
      c = argv[i][j];

      switch(c){
        case '\0':
          j = 1;
          i++;
          break;
        case '-':
          j++;
          break;

        /*switches*/  
        case 'h':
          fprintf(stderr, "usage:\n\t%s -w [workers (d:%ld)] -p [port (d:%s)]\n\n", argv[0], cpus, port);
          return EX_OK;

        /*settings*/
        case 'p':
        case 'w':
          j++;
          if (argv[i][j] == '\0'){
            j = 0;
            i++;
          }
          if (i >= argc){
            fprintf(stderr, "%s: option -%c requires a parameter\n", argv[0], c);
            return EX_USAGE;
          }
          switch (c){
            case 'p':
              port = argv[i] + j;  
              break;
            case 'w':
              cpus = atol(argv[i] + j);
              break;
          }
          i++;
          j = 1;
          break;

        default:
          fprintf(stderr, "%s: unknown option -%c\n", argv[0], c);
          return EX_USAGE;
      }

    } else {

      fprintf(stderr, "%s: extra argument %s\n", argv[0], argv[i]);
      return EX_USAGE;
    }
    
  }



  return register_client_handler_server(&capture_client_data , port, cpus );
}
