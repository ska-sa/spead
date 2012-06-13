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
#include <sys/time.h>

#include <arpa/inet.h>
#include <netinet/in.h>

#include "spead_api.h"
#include "server.h"


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

struct u_server *create_server_us(int (*cdfn)(struct u_client *c), long cpus)
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
  s->s_sps  = NULL;

#endif
  return s;
}

void destroy_server_us(struct u_server *s)
{
  int i;

  if (s){
    
    if (s->s_sps){

      /*WARNING: review code --ed should be ok now*/
      for (i=0; i < s->s_cpus; i++){
        kill(s->s_sps[i], SIGTERM);
      }

      free(s->s_sps);
    }

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

  reuse_addr = 12*1024*1024;
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
  fprintf(stderr,"%s: server pid: %d running on port: %s\n", __func__, getpid(), port);
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

int worker_task_us(struct u_server *s)
{
  struct spead_packet *p;
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

#ifdef DEBUG
  fprintf(stderr, "%s:\tCHILD: first lite getpid [%d]\n", __func__, getpid());
#endif

  p = create_spead_packet();
  if (p == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot allocate memory for spead_packet\n", __func__);
#endif
    return -1;
  }

  while (run) {

#if 1
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

    fwrite(p->data, 1, nread, stdout);
    //fflush(stdout);

#if 0
    destroy_spead_packet(p);

    if (process_packet_hs(hs, p) < 0){
      destroy_spead_packet(p);
      continue; 
    }
#endif


#if 0
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
#endif 

    gettimeofday(&now, NULL);
//    sub_time(&delta, &now, &prev);
//    print_time(&delta, nread);
#endif

#if 0
def DEBUG
    fprintf(stderr, "[%d]: recvd %llu bytes\n", pid, bcount);
#endif

#if 0
def DEBUG
    fflush(stderr);
#endif
  }

#ifdef DEBUG
  fprintf(stderr, "%s:\tCHILD: last lite getpid [%d] bytes: %lu\n", __func__, getpid(), bcount);
#endif

  destroy_spead_packet(p);
   
  return 0;
}

int add_sp_us(struct u_server *s, pid_t sp, int size)
{
  if (s == NULL)
    return -1;

  s->s_sps = realloc(s->s_sps, sizeof(pid_t) * (size+1));
  if (s->s_sps == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: logic error cannot realloc for worker list\n", __func__);
#endif 
    return -1;
  }

  s->s_sps[size] = sp;
  
  return size + 1;
}

int spawn_workers_us(struct u_server *s)
{
  struct spead_heap_store *hs;
  int status, i;
  pid_t sp;

  hs = NULL;
  
  if (s == NULL)
    return -1;
 
  hs = create_store_hs();
  if (hs == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot create spead_heap_store\n", __func__);
#endif
    return -1;
  }

  i = 0;
  
  do {

    sp = fork_child_sp(s, &worker_task_us);
    if (sp < 0){
#ifdef DEBUG
      fprintf(stderr, "$s: fork_child_sp fail\n", __func__);
#endif
      continue;
    }

    if (add_sp_us(s, sp, i) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: could not store worker pid [%d]\n", __func__, sp);
#endif
      if (kill(sp, 2) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: kill error (%s)\n", __func__, strerror(errno));
#endif
      }
    } else
      i++;

  } while (i < s->s_cpus);

#ifdef DEBUG
  fprintf(stderr, "%s: PARENT about to loop\n", __func__);
#endif
  
  while(run) {
    
    //sleep(100);
    sp = waitpid(-1, &status, 0);

#ifdef DEBUG
    fprintf(stderr,"%s: PARENT waitpid [%d]\n", __func__, sp);
    if (WIFEXITED(status)) {
      fprintf(stderr, "exited, status=%d\n", WEXITSTATUS(status));
    } else if (WIFSIGNALED(status)) {
      fprintf(stderr, "killed by signal %d\n", WTERMSIG(status));
    } else if (WIFSTOPPED(status)) {
      fprintf(stderr, "stopped by signal %d\n", WSTOPSIG(status));
    } else if (WIFCONTINUED(status)) {
      fprintf(stderr, "continued\n");
    }
#endif
      
  }

#ifdef DEBUG
  fprintf(stderr, "%s: final recv count: %ld bytes\n", __func__, s->s_bc);
#endif

  
  destroy_store_hs(hs);

  return 0;
}

int register_client_handler_server(int (*client_data_fn)(struct u_client *c), char *port, long cpus)
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

  if (startup_server_us(s, port) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error in startup\n", __func__);
#endif
    shutdown_server_us(s);
    return -1;
  }

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


int capture_client_data(struct u_client *c)
{

}

int main(int argc, char *argv[])
{
  long cpus;

  cpus = sysconf(_SC_NPROCESSORS_ONLN);
#ifdef DEBUG
  fprintf(stderr, "SPEAD SERVER\n\tCPUs present: %ld\n", cpus);
#endif  

  return register_client_handler_server(&capture_client_data , PORT, cpus + 1 );
}
