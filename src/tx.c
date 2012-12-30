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
#include <netdb.h>

#include <sys/mman.h>

#include "avltree.h"
#include "spead_api.h"

static volatile int run = 1;
static volatile int child = 0;

struct spead_tx {
  mutex                     t_m;
  struct spead_socket       *t_x;
  struct spead_workers      *t_w;
  struct data_file          *t_f;
  int                       t_pkt_size; 
  struct avl_tree           *t_t;
  struct spead_heap_store   *t_hs;
  uint64_t                  t_count;
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
    destroy_spead_workers(tx->t_w);
    destroy_spead_socket(tx->t_x);
    destroy_store_hs(tx->t_hs);
    destroy_raw_data_file(tx->t_f);
    munmap(tx, sizeof(struct spead_tx));
  }
}

struct spead_tx *create_speadtx(char *host, char *port, char bcast, int pkt_size)
{
  struct spead_tx *tx; 

  tx = mmap(NULL, sizeof(struct spead_tx), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, (-1), 0);
  if (tx == MAP_FAILED){
#ifdef DEUBG
    fprintf(stderr, "%s: logic cannot malloc\n", __func__);
#endif
    return NULL;
  }

  tx->t_m         = 0;
  tx->t_x         = NULL;
  tx->t_w         = NULL;
  tx->t_f         = NULL;
  tx->t_pkt_size  = pkt_size;
  tx->t_t         = NULL;
  tx->t_hs        = NULL;
  tx->t_count     = 0;

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

#ifdef DEBUG
  fprintf(stderr, "%s: pktsize: %d\n", __func__, pkt_size);
#endif
  
  return tx;
}

uint64_t get_count_speadtx(struct spead_tx *tx)
{
  if (tx == NULL)
    return -1;

  lock_mutex(&(tx->t_m));
  
  tx->t_count++;

  unlock_mutex(&(tx->t_m));

  return tx->t_count;
}

int worker_task_speadtx(void *data, struct spead_api_module *m, int cfd)
{
  struct spead_item_group *ig;
  struct spead_api_item *itm;
  struct spead_tx *tx;
  struct hash_table *ht;
  pid_t pid;

  uint64_t hid;

  tx = data;
  if (tx == NULL)
    return -1;

  pid = getpid();

#ifdef DEBUG
  fprintf(stderr, "%s: SPEADTX worker [%d] cfd[%d]\n", __func__, pid, cfd);
#endif

  ig = create_item_group(8192, 4);
  if (ig == NULL)
    return -1;

  //print_list_stats(tx->t_hs->s_list, __func__);

  itm = new_item_from_group(ig, 2048);
  if (set_item_data_ones(itm) < 0) {}
  //print_data(itm->i_data, itm->i_len);

  itm = new_item_from_group(ig, 2048);
  if (set_item_data_zeros(itm) < 0) {}
  //print_data(itm->i_data, itm->i_len);

  itm = new_item_from_group(ig, 2048);
  if (set_item_data_ramp(itm) < 0) {}
  //print_data(itm->i_data, itm->i_len);

  itm = new_item_from_group(ig, 2048);
  if (set_item_data_ones(itm) < 0) {}
  
  //hid = get_count_speadtx(tx);
  
  //while (run && hid < 1) {
  while (run) {

    hid = get_count_speadtx(tx);

    ht = packetize_item_group(tx->t_hs, ig, tx->t_pkt_size, hid);
    if (ht == NULL){
      destroy_item_group(ig);
#ifdef DEBUG
      fprintf(stderr, "Packetize error\n");
#endif
      return -1;
    }

    if (inorder_traverse_hash_table(ht, &send_packet_spead_socket, tx->t_x) < 0){
      unlock_mutex(&(ht->t_m));
      destroy_item_group(ig);
#ifdef DEBUG
      fprintf(stderr, "%s: send inorder trav fail\n", __func__);
#endif
      return -1;
    }

    if (empty_hash_table(ht, 0) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error empting hash table", __func__);
#endif
      unlock_mutex(&(ht->t_m));
      destroy_item_group(ig);
      return -1;
    }

    unlock_mutex(&(ht->t_m));

#if 0 
def DATA
    fprintf(stderr, "[%d] %s: hid %ld\n", pid, __func__, hid);
#endif

    usleep(10);
  }

  //unlock_mutex(&(ht->t_m));

  destroy_item_group(ig);

#ifdef DEBUG
  fprintf(stderr, "%s: SPEADTX worker [%d] ending\n", __func__, pid);
#endif

  return 0;
}

struct avl_tree *create_spead_database()
{
  return create_avltree(&compare_spead_workers);
}

int register_speadtx(char *host, char *port, long workers, char broadcast, int pkt_size)
{
  struct spead_tx *tx;
  uint64_t heaps, packets;
  sigset_t empty_mask;
  int rtn;
  
  if (register_signals_us() < 0)
    return EX_SOFTWARE;
  
  tx = create_speadtx(host, port, broadcast, pkt_size);
  if (tx == NULL)
    return EX_SOFTWARE;
#if 0
  tx->t_f = load_raw_data_file("/srv/pulsar/test.dat");
  if (tx->t_f == NULL){
    destroy_speadtx(tx);
    return EX_SOFTWARE;
  }
#endif

  heaps = 10;
  packets = 10;

  tx->t_hs = create_store_hs(heaps*packets, heaps, packets);
  if (tx->t_hs == NULL){
    destroy_speadtx(tx);
    return EX_SOFTWARE;
  }
  
#if 0
  tx->t_t = create_spead_database();
  if (tx->t_t == NULL){
    destroy_speadtx(tx);
    return EX_SOFTWARE;
  }
#endif
  
  tx->t_w = create_spead_workers(tx, workers, &worker_task_speadtx);
  if (tx->t_w == NULL){
    destroy_speadtx(tx);
    return EX_SOFTWARE;
  }

  sigemptyset(&empty_mask);

  while (run){

    if (populate_fdset_spead_workers(tx->t_w) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error populating fdset\n", __func__);
#endif
      run = 0;
      break;
    }

    rtn = pselect(get_high_fd_spead_workers(tx->t_w) + 1, get_in_fd_set_spead_workers(tx->t_w), (fd_set *) NULL, (fd_set *) NULL, NULL, &empty_mask);
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
    

    fprintf(stderr, ".");
    //sleep(1);

    /*do stuff*/
    
    /*saw a SIGCHLD*/
    if (child){
      wait_spead_workers(tx->t_w);
    }
    
  }

  if (send_spead_stream_terminator(tx->t_x) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: could not terminat stream\n", __func__);
#endif
    destroy_speadtx(tx);
    return -1;
  }
  
  destroy_speadtx(tx);

#ifdef DEBUG
  fprintf(stderr, "%s: tx exiting cleanly\n", __func__);
#endif
  
  return 0;
}

int usage(char **argv, long cpus)
{
  fprintf(stderr, "usage:\n\t%s (options) destination port\n\n\tOptions\n\t\t-w [workers (d:%ld)]\n\t\t-x (enable send to broadcast [priv])\n\t\t-s [spead packet size]\n\n", argv[0], cpus);
  return EX_USAGE;
}

int main(int argc, char **argv)
{
  long cpus;
  char c, *port, *host, broadcast;
  int i,j,k, pkt_size;

  i = 1;
  j = 1;
  k = 0;

  host = NULL;

  broadcast = 0;
  
  pkt_size  = 1024;
  port      = PORT;
  cpus      = sysconf(_SC_NPROCESSORS_ONLN);

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
        case 's':
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
            case 's':
              pkt_size = atoi(argv[i] + j);
              if (pkt_size == 0)
                return usage(argv, cpus);
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
  
  

  return register_speadtx(host, port, cpus, broadcast, pkt_size);
}
  
