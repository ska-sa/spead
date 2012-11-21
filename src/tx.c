/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sysexits.h>
#include <signal.h>


#include "spead_api.h"

static volatile int run = 1;
static volatile int child = 0;

struct spead_tx {
  struct spead_socket   *t_x;
  struct spead_workers  *t_w;
 
};



void handle_us(int signum) 
{
  run = 0;
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

  sa.sa_handler   = child_us;

  if (sigaction(SIGCHLD, &sa, NULL) < 0)
    return -1;

  return 0;
}

void destroy_speadtx(struct spead_tx *tx)
{
  if (tx){
    destroy_spead_socket(tx->t_x);
    destroy_spead_workers(tx->t_w);
    free(tx);
  }
}

struct spead_tx *create_speadtx(char *host, char *port, char bcast)
{
  struct spead_tx *tx; 

  tx = malloc(sizeof(struct spead_tx));
  if (tx == NULL){
#ifdef DEUBG
    fprintf(stderr, "%s: logic cannot malloc\n", __func__);
#endif
    return NULL;
  }

  tx->t_x = NULL;

  tx->t_x = create_spead_socket(host, port);
  if (tx->t_x == NULL){
    destroy_speadtx(tx);
    return NULL;
  }
  
  if (connect_spead_socket(tx->t_x) < 0){
    destroy_speadtx(tx);
    return NULL;
  }

  if (bcast){
    set_broadcast_opt_spead_socket(tx->t_x);
  }
  
  return tx;
}

int worker_task_speadtx(void *data, struct spead_api_module *m, int cfd)
{
  struct spead_tx *tx;

  tx = data;
  if (tx == NULL)
    return -1;

#ifdef DEBUG
  fprintf(stderr, "%s: SPEADTX worker [%d] cfd[%d]\n", __func__, getpid(), cfd);
#endif

  return 0;
}

int register_speadtx(char *host, char *port, long workers, char broadcast)
{
  struct spead_tx *tx;

  if (register_signals_us() < 0)
    return EX_SOFTWARE;
  
  tx = create_speadtx(host, port, broadcast);
  if (tx == NULL)
    return EX_SOFTWARE;
  
  tx->t_w = create_spead_workers(tx, workers, &worker_task_speadtx);
  if (tx->t_w == NULL){
    destroy_speadtx(tx);
    return EX_SOFTWARE;
  }


  
  destroy_speadtx(tx);
  
  return 0;
}

int usage(char **argv, long cpus)
{
  fprintf(stderr, "usage:\n\t%s (options) destination port\n\n\tOptions\n\t\t-w [workers (d:%ld)]\n\t\t-x (enable send to broadcast [priv])\n\n", argv[0], cpus);
  return EX_USAGE;
}

int main(int argc, char **argv)
{
  long cpus;
  char c, *port, *host, broadcast;
  int i,j,k;

  i = 1;
  j = 1;
  k = 0;

  host = NULL;

  broadcast = 0;
  
  port = PORT;
  cpus = sysconf(_SC_NPROCESSORS_ONLN);

  if (argc < 2)
    return usage(argv, cpus);

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
          return usage(argv, cpus);

        /*settings*/
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
            case 'w':
              cpus = atoll(argv[i] + j);
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
      /*parameters*/
      switch (k){
        case 0:
          host = argv[i];
          k++;
          break;
        case 1:
          port = argv[i];
          k++;
          break;
        default:
          fprintf(stderr, "%s: extra argument %s\n", argv[0], argv[i]);
          return EX_USAGE;
      }
      i++;
      j=1;
    }
  }
  
  

  return register_speadtx(host, port, cpus, broadcast);
}
  
