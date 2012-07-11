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
#include <sys/mman.h>

#include <arpa/inet.h>
#include <netinet/in.h>


#include "spead_api.h"
#include "server.h"
#include "hash.h"
#include "sharedmem.h"
#include "mutex.h"


static volatile int run = 1;
static volatile int timer = 0;

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

void print_time(struct timeval *result, uint64_t bytes)
{
  int64_t us;
  int64_t bpus;

  us = result->tv_sec*1000*1000 + result->tv_usec;
  //bpus = bytes / us * 1000 * 1000 / 1024 / 1024;
  bpus = (bytes / us) * 1000 * 1000;
  print_format_bitrate('R', bpus);

#ifdef DATA
  fprintf(stderr, "RTIME\t[%d]:\t%3lu.%06ld seconds\n", getpid(), result->tv_sec, result->tv_usec);
#endif
}


void handle_us(int signum) 
{
  run = 0;
}

void timer_us(int signum) 
{
  timer = 1;
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
#if 0
  s = malloc(sizeof(struct u_server));
#endif
  s = mmap(NULL, sizeof(struct u_server), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, (-1), 0);
  if (s == NULL)
    return NULL;
  
  s->s_fd   = 0;
  s->s_bc   = 0;
  s->s_cpus = cpus;
  s->s_cs   = NULL;
  s->s_hs   = NULL;
  s->s_m    = 0;

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
    munmap(s, sizeof(struct u_server));
  }
#ifdef DEBUG
  fprintf(stderr, "%s: destroyed server\n",  __func__);
#endif
}

int startup_server_us(struct u_server *s, char *port)
{
  struct addrinfo hints;
  struct addrinfo *res, *rp;
  uint64_t reuse_addr;

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
#if DEBUG>1
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

  reuse_addr = 1024*1024*1024;
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
  fprintf(stderr,"\tSERVER:\t\t[%d]\n\tport:\t\t%s\n\tnice:\t\t%d\n", getpid(), port, nice(0));
#endif

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

void print_format_bitrate(char x, uint64_t bps)
{
  char *rates[] = {"B", "KB", "MB", "GB", "TB"};
  int i;
  double style;

#ifdef DATA
  if (bps > 0){
    
    for (i=0; (bps / 1024) > 0; i++, bps /= 1024){
      style = bps / 1024.0;
    }
    
    switch(x){

      case 'T':
        fprintf(stderr, "TOTAL\t[%d]:\t%10.6f %s\n", getpid(), style, rates[i]);
        break;

      case 'R':
        fprintf(stderr, "RATE\t[%d]:\t%10.9f %sps\n", getpid(), style, rates[i]);
        break;

      case 'D':
        fprintf(stderr, "DATA\t[%d]:\t%10.9f %s\n", getpid(), style, rates[i]);
        break;

    }
  }
#endif
}

int worker_task_us(struct u_server *s, int cfd)
{
  struct spead_packet *p;
  struct spead_heap_store *hs;
  struct hash_o *o;
#if 1
  struct timeval prev, now, delta;
#endif

  struct sockaddr_storage peer_addr;
  socklen_t peer_addr_len;

  ssize_t nread;
  uint64_t rcount, bcount;

  pid_t pid;

  rcount = 0;
  bcount = 0;
  p      = NULL;
  o      = NULL;
  pid    = getpid();

  if (s == NULL || s->s_hs == NULL)
    return -1;

  hs = s->s_hs;

#ifdef DEBUG
  fprintf(stderr, "\t  CHILD\t\t[%d]\n", pid);
#endif
  
#if 0
  p = malloc(sizeof(struct spead_packet));
  if (p == NULL)
    return -1;
#endif

  gettimeofday(&prev, NULL);

  while (run) {

#if 1
    o = pop_hash_o(hs->s_list);
    if (o == NULL){
      //run = 0;
      //break;
      //sleep(1);
      continue;
    }
#endif

#if 1
    p = get_data_hash_o(o);
    if (p == NULL){
      if (push_hash_o(hs->s_list, o) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: cannot push object!\n", __func__);
#endif
      }
      continue;
    }
#endif

    rcount++;
    peer_addr_len = sizeof(struct sockaddr_storage); 

    //bzero(p, sizeof(struct spead_packet));

    nread = recvfrom(s->s_fd, p->data, SPEAD_MAX_PACKET_LEN, 0, (struct sockaddr *) &peer_addr, &peer_addr_len);
    if (nread <= 0){
#if DEBUG>1
      fprintf(stderr, "%s: rcount [%lu] unable to recvfrom: %s\n", __func__, rcount, strerror(errno));
#endif
#if 1
      if (push_hash_o(hs->s_list, o) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: cannot push object!\n", __func__);
#endif
      }
#endif
      break;
    }

    lock_mutex(&(s->s_m));
    s->s_bc += nread;
    unlock_mutex(&(s->s_m));

#if 1
    if (process_packet_hs(s->s_hs, o) < 0){
#if DEBUG>1
      fprintf(stderr, "%s: cannot process packet return object!\n", __func__);
#endif
      if (push_hash_o(hs->s_list, o) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: cannot push object!\n", __func__);
#endif
      }

      //continue; 
    }
#endif

#if 0
    if(write(cfd, &nread, sizeof(nread)) < 0)
      continue;
#endif
    
    bcount += nread;

  }

  
  lock_mutex(&(s->s_m));
  gettimeofday(&now, NULL);
  sub_time(&delta, &now, &prev);
  print_time(&delta, bcount);

  close(cfd);
  
#ifdef DEBUG
  //fprintf(stderr, "\tCHILD[%d]: exiting with bytes: %lu\n", getpid(), bcount);
#endif
  print_format_bitrate('T', bcount);
  unlock_mutex(&(s->s_m));

  return 0;
}


int spawn_workers_us(struct u_server *s, uint64_t hashes, uint64_t hashsize)
{
  struct spead_heap_store *hs;
  struct u_child *c;
  int status, i, hi_fd, rtn;
  fd_set ins;
  pid_t sp;
  sigset_t empty_mask;
  uint64_t rr, total;
#if 0
  struct timespec ts;
#endif
  struct sigaction sa;

  hs = NULL;
  rr = 0;
  total = 0;
#if 0
  ts.tv_sec = 1;
  ts.tv_nsec = 0;
#endif
  
  if (s == NULL || hashes < 1 || hashsize < 1)
    return -1;
 
  hs = create_store_hs((hashes * hashsize), hashes, hashsize);
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

  sigfillset(&sa.sa_mask);
  sa.sa_handler   = timer_us;
  sa.sa_flags     = 0;

  sigaction(SIGALRM, &sa, NULL);
  
  sigemptyset(&empty_mask);

  alarm(1);

  hi_fd = 0;

  while(run) {

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
    
    rtn = pselect(hi_fd + 1, &ins, (fd_set *) NULL, (fd_set *) NULL, NULL, &empty_mask);
    if (rtn < 0){
      switch(errno){
        case EAGAIN:
        case EINTR:
          break;
        default:
#ifdef DEBUG
          fprintf(stderr, "%s: pselect error\n", __func__);
#endif    
          run = 0;
          continue;
      }
    } 
    
    if (timer){
      lock_mutex(&(s->s_m));
      total = s->s_bc - total;
      unlock_mutex(&(s->s_m));
      print_format_bitrate('R', total);
#if 0 
      def DATA
      if (total > 0) {
        fprintf(stderr, "SERVER recv:\t%ld Bps\n", total);
      }
#endif
      alarm(1);
      timer = 0;
      lock_mutex(&(s->s_m));
      total = s->s_bc;
      unlock_mutex(&(s->s_m));
      continue;
    }

#if 0
    for (i=0; i<s->s_cpus; i++){
      c = s->s_cs[i];
      if (c != NULL){
        if (FD_ISSET(c->c_fd, &ins)){
          
#if DEBUG>1
          fprintf(stderr, "%s: FD %d ISSET\n", __func__, c->c_fd);
#endif    
          if(read(c->c_fd, &rr, sizeof(rr))){
            s->s_bc += rr;
          }
        }
      }
    }
#endif

  }

  //s->s_bc = total;
  
  i = 0;
  do {
    
    sp = waitpid(-1, &status, 0);

#if DEBUG>1
    fprintf(stderr,"%s: PARENT waitpid [%d]\n", __func__, sp);
#endif

    if (WIFEXITED(status)) {
#if DEBUG>1
      fprintf(stderr, "exited, status=%d\n", WEXITSTATUS(status));
#endif
    } else if (WIFSIGNALED(status)) {
#ifdef DEBUG
      fprintf(stderr, "killed by signal %d\n", WTERMSIG(status));
#endif
    } else if (WIFSTOPPED(status)) {
#ifdef DEBUG
      fprintf(stderr, "stopped by signal %d\n", WSTOPSIG(status));
#endif
    } else if (WIFCONTINUED(status)) {
#ifdef DEBUG
      fprintf(stderr, "continued\n");
#endif
    }

  } while (i++ < s->s_cpus);

#ifdef DEBUG
  //fprintf(stderr, "%s: final recv count:\t%ld bytes\n", __func__, s->s_bc);
#endif
  print_format_bitrate('T', s->s_bc);

  return 0;
}

int register_client_handler_server(int (*client_data_fn)(), char *port, long cpus, uint64_t hashes, uint64_t hashsize)
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

  if (spawn_workers_us(s, hashes, hashsize) < 0){ 
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
  
  return 0;
}

int main(int argc, char *argv[])
{
  long cpus;
  int i, j, c;
  char *port;
  uint64_t hashes, hashsize;

  i = 1;
  j = 1;

  hashes = 10;
  hashsize = 10;

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
          fprintf(stderr, "usage:\n\t%s\n\t\t-w [workers (d:%ld)]\n\t\t-p [port (d:%s)]\n\t\t-b [buffers (d:%ld)]\n\t\t-l [buffer length (d:%ld)]\n\n", argv[0], cpus, port, hashes, hashsize);
          return EX_OK;

        /*settings*/
        case 'p':
        case 'w':
        case 'b':
        case 'l':
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
              cpus = atoll(argv[i] + j);
              break;
            case 'b':
              hashes = atoll(argv[i] + j);
              break;
            case 'l':
              hashsize = atol(argv[i] + j);
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

  return register_client_handler_server(&capture_client_data , port, cpus, hashes, hashsize);
}
