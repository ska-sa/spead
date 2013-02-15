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
#include <fcntl.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/utsname.h>
#include <sys/time.h>
#include <sys/mman.h>

#include <arpa/inet.h>
#include <netinet/in.h>

#ifndef IKATCP
#include <katcl.h>
#include <katcp.h>
#endif

#include "server.h"
#include "spead_api.h"
#include "hash.h"
#include "mutex.h"


static volatile int run = 1;
static volatile int timer = 0;
static volatile int child = 0;

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
  //int64_t us;
  //int64_t bpus;

  //us = result->tv_sec*1000*1000 + result->tv_usec;
  //bpus = bytes / us * 1000 * 1000 / 1024 / 1024;
  //bpus = (bytes / us) * 1000 * 1000;
  //print_format_bitrate('R', bpus);

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

void child_us(int signum)
{
  child = 1;
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

#if 0
  if (sigaction(SIGFPE, &sa, NULL) < 0)
    return -1;
#endif

  sa.sa_handler   = child_us;

  if (sigaction(SIGCHLD, &sa, NULL) < 0)
    return -1;

  return 0;
}

struct u_server *create_server_us(struct spead_api_module *m, long cpus, char *raw_pkt_file)
{
  struct u_server *s;

#if 0
  if (cdfn == NULL)
    return NULL;
#endif

  if (cpus < 1){
#ifdef DEBUG
    fprintf(stderr, "%s: must have at least 1 cpu\n", __func__);
#endif
    return NULL;
  }

  s = mmap(NULL, sizeof(struct u_server), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, (-1), 0);
  if (s == MAP_FAILED)
    return NULL;
  
  s->s_x       = NULL;
  s->s_w       = NULL;
  s->s_f       = NULL;
  s->s_fd      = 0;
  s->s_bc      = 0;
  s->s_pc      = 0;
  s->s_hpcount = 0;
  s->s_hdcount = 0;
  s->s_cpus    = cpus;
  s->s_cs      = NULL;
  s->s_hs      = NULL;
  s->s_m       = 0;
  s->s_mod     = m;
#ifndef IKATCP
  s->s_kl      = NULL;
#endif

  if (raw_pkt_file){
    s->s_f = write_raw_data_file(raw_pkt_file);
  }

  return s;
}

void destroy_server_us(struct u_server *s)
{
  int i;

  if (s){
    
    if (s->s_cs){

      for (i=0; i < s->s_cpus; i++){
        destroy_child_sp(s->s_cs[i]);
      }

      free(s->s_cs);
    }
    
    destroy_spead_socket(s->s_x);
    destroy_spead_workers(s->s_w);
    
    destroy_raw_data_file(s->s_f);

#ifndef IKATCP
    if (s->s_kl){
      destroy_katcl(s->s_kl, 0);
    }
#endif

    unload_api_user_module(s->s_mod);

    destroy_store_hs(s->s_hs);
    munmap(s, sizeof(struct u_server));
  }

#ifdef DEBUG
  fprintf(stderr, "%s: destroyed server\n",  __func__);
#endif
}

int startup_server_us(struct u_server *s, char *port, int broadcast)
{
  if (s == NULL || port == NULL)
    return -1;
  
  s->s_x = create_spead_socket(NULL, port);
  if (s->s_x == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot create spead socket\n", __func__);
#endif
    return -1;
  }
  
  if (bind_spead_socket(s->s_x) < 0)
    return -1;
  
  if (broadcast){
    if (set_broadcast_opt_spead_socket(s->s_x) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: WARN spead socket broadcast option not set\n", __func__);
#endif
    }
  }

  s->s_fd = get_fd_spead_socket(s->s_x);

#ifdef DEBUG
  fprintf(stderr,"\tSERVER:\t\t[%d]\n\tport:\t\t%s\n\tnice:\t\t%d\n", getpid(), port, nice(0));
#endif

  return 0;
}

void shutdown_server_us(struct u_server *s)
{
  if (s){
#if 0
    if (close(s->s_fd) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error server shutdown: %s\n", __func__, strerror(errno));
#endif
    }
#endif
    destroy_server_us(s);
  }
#ifdef DEBUG
  fprintf(stderr, "%s: server shutdown complete\n", __func__);
#endif
} 


void print_format_bitrate(struct u_server *s, char x, uint64_t bps)
{
  char *rates[] = {"bits", "kbits", "mbits", "gbits", "tbits"};
  int i;
  double style;
  uint64_t bitsps;
  
  if (s == NULL)
    return;

  bitsps = bps * 8;
  bps = bitsps;
  style = 0;

#ifdef DATA
  if (bps > 0){
    
    for (i=0; (bitsps / 1024) > 0; i++, bitsps /= 1024){
      style = bitsps / 1024.0;
    }
    
    switch(x){

      case 'T':
#ifdef IKATCP
        fprintf(stderr, "TOTAL\t[%d]:\t%10.6f %s\n", getpid(), style, rates[i]);
#else
        log_message_katcl(s->s_kl, KATCP_LEVEL_INFO, NULL, "TOTAL\t[%d]:\t%10.6f %s\n", getpid(), style, rates[i]);
#endif
        break;

      case 'R':
#ifdef IKATCP
        fprintf(stderr, "RATE\t[%d]:\t%10.6f %sps %ld bps\n", getpid(), style, rates[i], bps);
#else
        log_message_katcl(s->s_kl, KATCP_LEVEL_INFO, NULL, "RATE\t[%d]:\t%10.6f %sps %ld bps\n", getpid(), style, rates[i], bps);
#endif
        break;

      case 'D':
#ifdef IKATCP
        fprintf(stderr, "DATA\t[%d]:\t%10.6f %s\n", getpid(), style, rates[i]);
#else
        log_message_katcl(s->s_kl, KATCP_LEVEL_INFO, NULL, "DATA\t[%d]:\t%10.6f %s\n", getpid(), style, rates[i]);
#endif
        break;

    }
  }
#endif
}

int worker_task_us(void *data, struct spead_api_module *m, int cfd)
{
  struct u_server *s;
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
  int rtn;

  pid_t pid;

  rtn    = 0;
  rcount = 0;
  bcount = 0;
  p      = NULL;
  o      = NULL;
  pid    = getpid();
  

  s = data;

  if (s == NULL || s->s_hs == NULL)
    return -1;

  hs = s->s_hs;

#ifdef DEBUG
  fprintf(stderr, "\t  CHILD\t\t[%d]\n", pid);
#endif
  
#ifdef RATE
  p = malloc(sizeof(struct spead_packet));
  if (p == NULL)
    return -1;
#endif

  gettimeofday(&prev, NULL);

  peer_addr_len = sizeof(struct sockaddr_storage); 

  while (run) {

#ifndef RATE
    o = pop_hash_o(hs->s_list);
    if (o == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: cannot pop object!\n", __func__);
#endif
      run = 0;
      break;
      //sleep(1);
      //continue;
    }
#endif

#ifndef RATE
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

    bzero(p, sizeof(struct spead_packet));

    nread = recvfrom(s->s_fd, p->data, SPEAD_MAX_PACKET_LEN, 0, (struct sockaddr *) &peer_addr, &peer_addr_len);
    if (nread <= 0){
#if DEBUG>1
      fprintf(stderr, "%s: rcount [%lu] unable to recvfrom: %s\n", __func__, rcount, strerror(errno));
#endif

#ifndef RATE
      if (push_hash_o(hs->s_list, o) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: cannot push object!\n", __func__);
#endif
      }
#endif
      continue;
      //break;
    }

#ifndef RATE
    if ((rtn = process_packet_hs(s, m, o)) < 0){
      if (rtn == -1){
#ifdef DEBUG
        fprintf(stderr, "%s: cannot process packet return object!\n", __func__);
#endif
        if (push_hash_o(hs->s_list, o) < 0){
#ifdef DEBUG
          fprintf(stderr, "%s: cannot push object!\n", __func__);
#endif
        }
      }
    }
#endif

    lock_mutex(&(s->s_m));
    s->s_bc += nread;
    s->s_pc++;
    unlock_mutex(&(s->s_m));
    
    bcount += nread;
    rcount++;

  }

  
  lock_mutex(&(s->s_m));
  gettimeofday(&now, NULL);
  sub_time(&delta, &now, &prev);
  print_time(&delta, bcount);

  close(cfd);
  
#ifdef DEBUG
  fprintf(stderr, "\tCHILD[%d]: exiting with bytes: %ld rcount: %ld\n", getpid(), bcount, rcount);
#endif
  print_format_bitrate(s, 'T', bcount);
  unlock_mutex(&(s->s_m));

#ifdef RATE
  if (p)
    free(p);
#endif

  return 0;
}

int spawn_workers_us(struct u_server *s, uint64_t hashes, uint64_t hashsize)
{
  struct spead_heap_store *hs;
  struct u_child *c;
  int status, i, hi_fd, rtn;
#if 0
  int rb;
  unsigned char buf[BUF];
#endif
  fd_set ins;
  pid_t sp;
  sigset_t empty_mask;
  uint64_t total;
#if 0
  struct timespec ts;
#endif
  struct sigaction sa;

  hs = NULL;
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

    c = fork_child_sp(s->s_mod, s, &worker_task_us);
    if (c == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: fork_child_sp fail\n", __func__);
#endif
      continue;
    }

    if (add_child_us(&s->s_cs, c, i) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: could not store worker pid [%d]\n", __func__, c->c_pid);
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
    
#if 0
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
#endif

    rtn = pselect(hi_fd + 1, &ins, (fd_set *) NULL, (fd_set *) NULL, NULL, &empty_mask);
    if (rtn < 0){
      switch(errno){
        case EAGAIN:
        case EINTR:
          //continue;
          break;
        default:
#ifdef DEBUG
          fprintf(stderr, "%s: pselect error\n", __func__);
#endif    
          run = 0;
          continue;
      }
    }

#if 0
    for (i=0; i<s->s_cpus; i++){
      c = s->s_cs[i];
      if (c == NULL){
        continue;
      }
      if (FD_ISSET(c->c_fd, &ins)){
#ifdef DATA
        fprintf(stderr, "\tCHILD [%d] has data for parent\n", c->c_pid);
#endif
        rb = read(c->c_fd, buf, BUF);
        if (rb < 0){
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
        } else if (rb == 0){
#ifdef DATA
          fprintf(stderr, "\tread EOF\n");
#endif
        } else{
#ifdef DATA
          fprintf(stderr, "\tread %d bytes [%s]\n", rb, (unsigned char*)buf);
#endif
        }
      }
    }
#endif

#ifdef DATA
    if (timer){
      lock_mutex(&(s->s_m));
      total = s->s_bc - total;
      unlock_mutex(&(s->s_m));
      print_format_bitrate(s, 'R', total);

      if (s->s_hpcount > 0){
        fprintf(stderr, "\theaps \033[32mprocessed: %d\033[0m\n", s->s_hpcount);
      }
      if (s->s_hdcount > 0){
        fprintf(stderr, "\theaps \033[31mdiscarded: %d\033[0m\n", s->s_hdcount);
      }
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
      s->s_hpcount = 0;
      s->s_hdcount = 0;
      unlock_mutex(&(s->s_m));
      continue;
    }
#endif

    if (child){      
      sp = waitpid(-1, &status, 0);
#ifdef DEBUG
      fprintf(stderr,"SIGCHLD waitpid [%d]\n", sp);
#endif
      child = 0;
    }

  }

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
  print_format_bitrate(s, 'T', s->s_bc);

  fprintf(stderr, "%s: final packet count: %ld\n", __func__, s->s_pc);

  return 0;
}

#ifndef IKATCP
int setup_katcp_us(struct u_server *s)
{
  struct katcl_line *kl;

  if (s == NULL)
    return -1;

#if 0
ndef DEBUG
  flags = fcntl(STDOUT_FILENO, F_GETFL, NULL);
  if (flags >= 0){
    flags = fcntl(STDOUT_FILENO, F_SETFL, flags | O_NONBLOCK);
  }
#endif

  kl = create_katcl(STDOUT_FILENO);
  if (kl == NULL)
    return -1;

  
  append_string_katcl(kl, KATCP_FLAG_FIRST | KATCP_FLAG_STRING, KATCP_VERSION_CONNECT_INFORM);
  append_string_katcl(kl,                    KATCP_FLAG_STRING, "spead-server");
  append_string_katcl(kl, KATCP_FLAG_LAST  | KATCP_FLAG_STRING, VERSION);
  
  while (write_katcl(kl) == 0);

  s->s_kl = kl;

  return 0;
}
#endif

int server_run_loop(struct u_server *s)
{
  struct sigaction sa;
  uint64_t total;
  int rtn;
  sigset_t empty_mask;

  if (s == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif  
    return -1;
  }

  sigfillset(&sa.sa_mask);
  sa.sa_handler   = timer_us;
  sa.sa_flags     = 0;

  sigaction(SIGALRM, &sa, NULL);
  
  alarm(1);

  sigemptyset(&empty_mask);

  total = 0;

  while (run){

    if (populate_fdset_spead_workers(s->s_w) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error populating fdset\n", __func__);
#endif
      run = 0;
      break;
    }

    rtn = pselect(get_high_fd_spead_workers(s->s_w) + 1, get_in_fd_set_spead_workers(s->s_w), (fd_set *) NULL, (fd_set *) NULL, NULL, &empty_mask);
    if (rtn < 0){
      switch(errno){
        case EAGAIN:
        case EINTR:
          //continue;
          break;
        default:
#ifdef DEBUG
          fprintf(stderr, "%s: pselect error\n", __func__);
#endif    
          run = 0;
          continue;
      }
    }
    

    //fprintf(stderr, ".");
    //sleep(1);

    /*do stuff*/
    
    /*saw a SIGCHLD*/
    if (child){
      wait_spead_workers(s->s_w);
    }
    
    if (timer){
      lock_mutex(&(s->s_m));
      total = s->s_bc - total;
      unlock_mutex(&(s->s_m));
      print_format_bitrate(s, 'R', total);
      alarm(1);
      timer = 0;
      lock_mutex(&(s->s_m));
      total = s->s_bc;
      unlock_mutex(&(s->s_m));
    }

  }

  fprintf(stderr, "%s: final packet count: %ld\n", __func__, s->s_pc);

#ifdef DEBUG
  fprintf(stderr, "%s: rx exiting cleanly\n", __func__);
#endif

  return 0;
}

int raw_spead_cap_worker(void *data, struct spead_api_module *m, int cfd)
{
  struct u_server *s;
  struct spead_packet *p;

  struct sockaddr_storage peer_addr;
  socklen_t peer_addr_len;

  ssize_t nread;
  
  struct data_file *f;

  s = data;
  if (s == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }
  
  f = s->s_f;
  if (f == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: data file not ready\n", __func__);
#endif
    return -1;
  }

  p = malloc(sizeof(struct spead_packet));
  if (p == NULL) {
#ifdef DEBUG
    fprintf(stderr, "%s: logic error cannot malloc\n", __func__);
#endif
    return -1;
  }
  
  peer_addr_len = sizeof(struct sockaddr_storage);

  while(run){
    
    nread = recvfrom(s->s_fd, p->data, SPEAD_MAX_PACKET_LEN, 0, (struct sockaddr *) &peer_addr, &peer_addr_len);
    if (nread < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: unable to recvfrom: %s\n", __func__, strerror(errno));
#endif
      continue;
    } else if (nread == 0){
#ifdef DEBUG
      fprintf(stderr, "%s: peer shutdown detected\n", __func__);
#endif
      continue;
    }
      
    
    if (write_next_chunk_raw_data_file(f, p->data, nread) < 0) {
#ifdef DEBUG
      fprintf(stderr, "%s: write next chunk fail\n", __func__);
#endif
    }
    
    lock_mutex(&(s->s_m));
    s->s_bc += nread;
    s->s_pc++;
    unlock_mutex(&(s->s_m));

  }
  
  if (p)
    free(p);

  return 0;
}

int register_client_handler_server(struct spead_api_module *m, char *port, long cpus, uint64_t hashes, uint64_t hashsize, int broadcast, char *raw_pkt_file)
{
  struct u_server *s;
  
  if (register_signals_us() < 0){
    fprintf(stderr, "%s: error register signals\n", __func__);
    return -1;
  }

  s = create_server_us(m, cpus, raw_pkt_file);
  if (s == NULL){
    fprintf(stderr, "%s: error could not create server\n", __func__);
    unload_api_user_module(m);
    return -1;
  }

  if (startup_server_us(s, port, broadcast) < 0){
    fprintf(stderr,"%s: error in startup\n", __func__);
    shutdown_server_us(s);
    return -1;
  }

#ifndef IKATCP
  if (setup_katcp_us(s) < 0){
    fprintf(stderr,"%s: error in startup\n", __func__);
    shutdown_server_us(s);
    return -1;
  }
#endif

  if (raw_pkt_file){

    s->s_w = create_spead_workers(s, cpus, &raw_spead_cap_worker);
    if (s->s_w == NULL){
      shutdown_server_us(s);
      fprintf(stderr, "%s: create spead workers failed\n", __func__);
      return -1;
    }

    if (server_run_loop(s) < 0){
      shutdown_server_us(s);
      fprintf(stderr, "%s: server run loop failed\n", __func__);
      return -1;
    }

  } else {

    if (spawn_workers_us(s, hashes, hashsize) < 0){ 
      fprintf(stderr,"%s: error during run\n", __func__);
      shutdown_server_us(s);
      return -1;
    }

  }
  
  shutdown_server_us(s);

#ifdef DEBUG
  fprintf(stderr,"%s: server exiting\n", __func__);
#endif

  return 0;
}


int main(int argc, char *argv[])
{
  long cpus;
  int i, j, c, broadcast;
  char *port, *dylib, *raw_pkt_file;
  uint64_t hashes, hashsize;

  struct spead_api_module *m;

  i = 1;
  j = 1;
  broadcast = 0;

  hashes   = 1000;
  hashsize = 100;
  
  dylib = NULL;
  m     = NULL;
  raw_pkt_file = NULL;

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
        case 'x':
          j++;
          broadcast = 1;
          break;

        case 'h':
          fprintf(stderr, "usage:\n\t%s\n\t\t-w [workers (d:%ld)]\n\t\t-p [port (d:%s)]\n\t\t-d [data sink module]\n\t\t-b [buffers (d:%ld)]\n\t\t-l [buffer length (d:%ld)]\n\t\t-x (enable receive from broadcast [priv])\n\t\t-r (dump raw spead packets)\n\n", argv[0], cpus, port, hashes, hashsize);
          return EX_OK;

        /*settings*/
        case 'r':
        case 'd':
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
            case 'r':
              raw_pkt_file = argv[i] +j;
              break;
            case 'd':
              dylib = argv[i] + j;  
              break;
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

  if (dylib != NULL){

    m = load_api_user_module(dylib);
    if (m == NULL){
      fprintf(stderr, "Could not load user api module <%s>\n", dylib);
      return EX_USAGE;
    }

  }

  return register_client_handler_server(m, port, cpus, hashes, hashsize, broadcast, raw_pkt_file);
}
