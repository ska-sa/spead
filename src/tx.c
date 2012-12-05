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
  struct spead_socket       *t_x;
  struct spead_workers      *t_w;
  struct data_file          *t_f;
  int                       t_pkt_size; 
  struct avl_tree           *t_t;
  struct spead_heap_store   *t_hs;
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

  tx->t_x         = NULL;
  tx->t_w         = NULL;
  tx->t_f         = NULL;
  tx->t_pkt_size  = pkt_size;
  tx->t_t         = NULL;
  tx->t_hs        = NULL;

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

int worker_task_speadtx(void *data, struct spead_api_module *m, int cfd)
{
  struct spead_item_group *ig;
  struct spead_api_item *itm;
  struct spead_packet *p;
  struct spead_tx *tx;
  struct addrinfo *dst;
  struct hash_table *ht;
  pid_t pid;
  int i, sb, sfd;

  tx = data;
  if (tx == NULL)
    return -1;

  pid = getpid();
  sfd = get_fd_spead_socket(tx->t_x);
  dst = get_addr_spead_socket(tx->t_x);

  if (sfd <=0 || dst == NULL)
    return -1;

#ifdef DEBUG
  fprintf(stderr, "%s: SPEADTX worker [%d] cfd[%d]\n", __func__, pid, cfd);
#endif

  ig = create_item_group(1536, 3);
  if (ig == NULL)
    return -1;

  print_list_stats(tx->t_hs->s_list, __func__);

  itm = new_item_from_group(ig, 512);
  if (set_item_data_ones(itm) < 0) {}

  itm = new_item_from_group(ig, 512);
  if (set_item_data_zeros(itm) < 0) {}

  itm = new_item_from_group(ig, 512);
  if (set_item_data_ramp(itm) < 0) {}
    
  ht = packetize_item_group(tx->t_hs, ig, 384, pid);
  if (ht == NULL){
    destroy_item_group(ig);
#ifdef DEBUG
    fprintf(stderr, "Packetize error\n");
#endif
    return -1;
  }
  
  struct hash_o *o;

  i = 0; 
  int state = S_GET_OBJECT;    
  while (state){
    
    switch(state){
      
      case S_GET_OBJECT:
        if (i < ht->t_len){
          o = ht->t_os[i];
          if (o == NULL){
            i++;
            state = S_GET_OBJECT;
            break;
          }
          state = S_GET_PACKET;
        } else 
          state = S_END;
        break;

      case S_GET_PACKET:
        p = get_data_hash_o(o);
        if (p == NULL){
          state = S_NEXT_PACKET;
          break;
        }
        
        sb = sendto(sfd, p->data, SPEAD_MAX_PACKET_LEN, 0, dst->ai_addr, dst->ai_addrlen);
        
#ifdef DEBUG
        fprintf(stderr, "%s: packet %d (%p) sb [%d] bytes\n", __func__, i, p, sb);
#endif
        
        state = S_NEXT_PACKET;
        break;

      case S_NEXT_PACKET:
        if (o->o_next != NULL){
          o = o->o_next;
          state = S_GET_PACKET;
        } else {
          i++;
          state = S_GET_OBJECT;
        }
        break;

    }

  }
  
   
  
  
  unlock_mutex(&(ht->t_m));
  
  destroy_item_group(ig);


#if 0
  p = malloc(sizeof(struct spead_packet));
  if (p == NULL)
    return -1;

  bzero(p, sizeof(struct spead_packet));
  
  spead_packet_init(p);
  
  p->n_items=6;
  p->is_stream_ctrl_term = SPEAD_STREAM_CTRL_TERM_VAL;
  
  SPEAD_SET_ITEM(p->data, 0, SPEAD_HEADER_BUILD(p->n_items));
  
  SPEAD_SET_ITEM(p->data, 1, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_HEAP_CNT_ID, 0x0));
  SPEAD_SET_ITEM(p->data, 2, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_HEAP_LEN_ID, 0x00));
  SPEAD_SET_ITEM(p->data, 3, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_PAYLOAD_OFF_ID, 0x0));
  SPEAD_SET_ITEM(p->data, 4, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_PAYLOAD_LEN_ID, 0x00));
  SPEAD_SET_ITEM(p->data, 5, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, 0xFD, 12345));
  SPEAD_SET_ITEM(p->data, 6, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_STREAM_CTRL_ID, SPEAD_STREAM_CTRL_TERM_VAL));

#if 0
  print_data(p->data, SPEAD_MAX_PACKET_LEN); 

  while(run){
    
    fprintf(stderr, "[%d]", pid);
    sleep(5);

  }
#endif

  sb = sendto(sfd, p->data, SPEAD_MAX_PACKET_LEN, 0, dst->ai_addr, dst->ai_addrlen);

#ifdef DEBUG
  fprintf(stderr, "[%d] sendto: %d bytes\n", pid, sb); 
#endif

  if (p)
    free(p);
#endif


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
  
  if (register_signals_us() < 0)
    return EX_SOFTWARE;
  
  tx = create_speadtx(host, port, broadcast, pkt_size);
  if (tx == NULL)
    return EX_SOFTWARE;

  tx->t_f = load_raw_data_file("/srv/pulsar/test.dat");
  if (tx->t_f == NULL){
    destroy_speadtx(tx);
    return EX_SOFTWARE;
  }

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

  


  
  

  
  while (run){
    
    fprintf(stderr, ".");
    sleep(1);


    /*do stuff*/
    
    /*saw a SIGCHLD*/
    if (child){
      wait_spead_workers(tx->t_w);
    }
    
    

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
  
