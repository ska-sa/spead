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
#include "tx.h"

static volatile int run = 1;
static volatile int child = 0;

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

struct spead_tx *create_speadtx(char *host, char *port, char bcast, int pkt_size, int chunk_size)
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
  tx->t_chunk_size= chunk_size;
  tx->t_t         = NULL;
  tx->t_hs        = NULL;
  tx->t_count     = 0;
  tx->t_pc        = 0;

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
  struct spead_api_item *itm, *itm2, *itm3;
  struct spead_tx *tx;
  struct hash_table *ht;
  pid_t pid;

  void *ptr;
  uint64_t hid, got, off;

  size_t size;
  char   *name;

  tx = data;
  if (tx == NULL)
    return -1;

  pid = getpid();
  hid = 0;

  size = get_data_file_size(tx->t_f);
  name = get_data_file_name(tx->t_f);


#ifdef DEBUG
  fprintf(stderr, "%s: SPEADTX worker [%d] cfd[%d]\n", __func__, pid, cfd);
#endif

#if 0
  ig = create_item_group(8192, 4);
  if (ig == NULL)
    return -1;

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
#endif

  ig = create_item_group(tx->t_chunk_size + 2*sizeof(uint64_t) /*+ sizeof(size_t)*/ + strlen(name) + 1, 4);
  if (ig == NULL)
    return -1;

#if 0
  itm = new_item_from_group(ig, sizeof(size_t));
  if (itm == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot create item\n", __func__);
#endif
  } else {
    itm->i_id = SPEADTX_IID_FILESIZE;
  }
  if (copy_to_spead_item(itm, &size, sizeof(size_t)) < 0){
    destroy_item_group(ig);
    return -1;
  }
#endif

  /*filename*/
  itm = new_item_from_group(ig, strlen(name) + 1);
  if (itm == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot create item\n", __func__);
#endif
  } else {
    itm->i_id = SPEADTX_IID_FILENAME;
  }
  if (copy_to_spead_item(itm, name, strlen(name)+1) < 0){
    destroy_item_group(ig);
    return -1;
  }

  /*chunk id*/
  itm2 = new_item_from_group(ig, sizeof(uint64_t));
  if (itm2 == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot create item\n", __func__);
#endif
  } else {
    itm2->i_id = SPEADTX_CHUNK_ID;
  }
  
  /*data offset from file*/
  itm3 = new_item_from_group(ig, sizeof(uint64_t));
  if (itm3 == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot create item\n", __func__);
#endif
  } else {
    itm3->i_id = SPEADTX_OFF_ID;
  }

  /*data from file*/
  itm  = new_item_from_group(ig, tx->t_chunk_size);
  if (itm == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot create item\n", __func__);
#endif
  } else {
    itm->i_id = SPEADTX_DATA_ID;
  }

  //hid = get_count_speadtx(tx);
  
  //while (run && hid < 1) {
  while (run) {

    got = request_chunk_datafile(tx->t_f, tx->t_chunk_size, &ptr, &off);
    if (got == 0){
#ifdef DEBUG
      fprintf(stderr, "%s: got 0 ending\n", __func__);
#endif
      run = 0;
      break;
    } else if (got < 0){
      destroy_item_group(ig);
      return -1;
    }

    if (copy_to_spead_item(itm, ptr, got) < 0){
      destroy_item_group(ig);
      return -1;
    }

    if (copy_to_spead_item(itm3, &off, sizeof(uint64_t)) < 0){
      destroy_item_group(ig);
      return -1;
    }

#if 0
def DEBUG
    print_data(itm->i_data, itm->i_data_len);
#endif

    hid = get_count_speadtx(tx);

    if (copy_to_spead_item(itm2, &hid, sizeof(uint64_t)) < 0){
      destroy_item_group(ig);
      return -1;
    }

#if 0
def DEBUG
    print_data(itm2->i_data, itm2->i_data_len);
#endif

    ht = packetize_item_group(tx->t_hs, ig, tx->t_pkt_size, hid);
    //ht = packetize_item_group(tx->t_hs, ig, tx->t_pkt_size, pid);
    if (ht == NULL){
      destroy_item_group(ig);
#ifdef DEBUG
      fprintf(stderr, "Packetize error\n");
#endif
      return -1;
    }

    if (inorder_traverse_hash_table(ht, &send_packet_spead_socket, tx) < 0){
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

#if 0 
int send_init_info_speadtx(struct spead_tx *tx)
{
  struct spead_item_group *ig;
  struct spead_api_item    *itm;

  struct hash_table *ht;
  uint64_t hid;
  
  size_t size;
  char   *name;

  if (tx == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }

  size = get_data_file_size(tx->t_f);
  name = get_data_file_name(tx->t_f);

  ig = create_item_group(sizeof(size_t)+strlen(name)+1, 2);
  if (ig == NULL)

    return -1;


  itm = new_item_from_group(ig, sizeof(size_t));
  if (itm == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot create item\n", __func__);
#endif
  } else {
    itm->i_id = SPEADTX_IID_FILESIZE;
  }
  if (copy_to_spead_item(itm, &size, sizeof(size_t)) < 0){
    destroy_item_group(ig);
    return -1;
  }
  
  
  itm = new_item_from_group(ig, strlen(name)+1);
  if (itm == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot create item\n", __func__);
#endif
  } else {
    itm->i_id = SPEADTX_IID_FILENAME;
  }
  if (copy_to_spead_item(itm, name, strlen(name)+1) < 0){
    destroy_item_group(ig);
    return -1;
  }


  hid = get_count_speadtx(tx);

  ht = packetize_item_group(tx->t_hs, ig, tx->t_pkt_size, hid);
  if (ht == NULL){
    destroy_item_group(ig);
#ifdef DEBUG
    fprintf(stderr, "Packetize error\n");
#endif
    return -1;
  }

  destroy_item_group(ig);

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

  return 0;
}
#endif

int register_speadtx(char *host, char *port, long workers, char broadcast, int pkt_size, int chunk_size, char *ifile)
{
  struct spead_tx *tx;
  uint64_t heaps, packets;
  sigset_t empty_mask;
  int rtn;
  
  if (register_signals_us() < 0)
    return EX_SOFTWARE;
  
  tx = create_speadtx(host, port, broadcast, pkt_size, chunk_size);
  if (tx == NULL)
    return EX_SOFTWARE;

#if 1
  tx->t_f = load_raw_data_file(ifile);
  if (tx->t_f == NULL){
    fprintf(stderr, "%s: FATAL could not load file\n", __func__);
    destroy_speadtx(tx);
    return EX_SOFTWARE;
  }
#endif

  heaps = workers*2;
  packets = chunk_size/pkt_size+2;

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

#if 0
  if (send_init_info_speadtx(tx) < 0){
    destroy_speadtx(tx);
    return EX_SOFTWARE;
  }
#endif

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

  if (send_spead_stream_terminator(tx) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: could not terminat stream\n", __func__);
#endif
    destroy_speadtx(tx);
    return -1;
  }

  fprintf(stderr, "%s: final packet count: %ld\n", __func__, tx->t_pc);

  destroy_speadtx(tx);

#ifdef DEBUG
  fprintf(stderr, "%s: tx exiting cleanly\n", __func__);
#endif
  
  return 0;
}

int usage(char **argv, long cpus)
{
  fprintf(stderr, "usage:\n\t%s (options) destination port\n\n\tOptions\n\t\t-w [workers (d:%ld)]\n\t\t-x (enable send to broadcast [priv])\n\t\t-s [spead packet size]\n\t\t-i [input file]\n\t\t-c [chunk size]\n\n", argv[0], cpus);
  return EX_USAGE;
}

int main(int argc, char **argv)
{
  long cpus;
  char c, *port, *host, broadcast, *ifile;
  int i,j,k, pkt_size, chunk_size;

  i = 1;
  j = 1;
  k = 0;

  host = NULL;
  ifile = NULL;

  broadcast = 0;
  
  pkt_size  = 1024;
  chunk_size= 8192;
  port      = PORT;
  cpus      = sysconf(_SC_NPROCESSORS_ONLN);

  if (argc < 4)
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
        case 'c':
        case 'i':
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
            case 'c':
              chunk_size = atoi(argv[i] + j);
              if (chunk_size == 0)
                return usage(argv, cpus);
              break;
            case 'w':
              cpus = atoll(argv[i] + j);
              break;
            case 's':
              pkt_size = atoi(argv[i] + j);
              if (pkt_size == 0)
                return usage(argv, cpus);
              break;
            case 'i':
              ifile = argv[i] + j;
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

  return register_speadtx(host, port, cpus, broadcast, pkt_size, chunk_size, ifile);
}
  
