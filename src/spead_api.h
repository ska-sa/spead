/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#ifndef SPEAD_API_H
#define SPEAD_API_H

#include <sys/stat.h>
#include <sys/select.h>

#include "spead_packet.h"
#include "hash.h"
#include "server.h"
#include "tx.h"


#define SPEAD_ZEROS_ID     0x112
#define SPEAD_ONES_ID      0x113
#define SPEAD_RAMP_ID      0x114


/*API DATASTRUCTURES*/ 

/*modules api*/
#define SAPI_CALLBACK         "spead_api_callback"
#define SAPI_SETUP            "spead_api_setup"
#define SAPI_DESTROY          "spead_api_destroy"
#define SAPI_TIMER_CALLBACK   "spead_api_timer_callback"

struct spead_api_module_shared {
  mutex   s_m;
  void    *s_data;
  size_t  s_data_size;
};

struct spead_item_group;

struct spead_api_module {
  struct spead_api_module_shared *m_s;
  void *m_handle;
  void *(*m_setup)(struct spead_api_module_shared *s);
  int  (*m_cdfn)(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data);
  int  (*m_destroy)(struct spead_api_module_shared *s, void *data);
  int  (*m_timer)(struct spead_api_module_shared *s, void *data);
  void *m_data;
};


/*spead store api*/
#define S_END             0
#define S_MODE            1
#define S_MODE_IMMEDIATE  2
#define S_MODE_DIRECT     3
#define S_GET_PACKET      4
#define S_NEXT_PACKET     5
#define S_GET_ITEM        6
#define S_NEXT_ITEM       7
#define S_GET_OBJECT      8
#define S_DIRECT_COPY     9
#define DC_NEXT_PACKET   10

struct spead_heap_store{
  int64_t             s_backlog;
  struct hash_o_list  *s_list;
  struct hash_table   **s_hash;
};

struct spead_api_item{
  int           i_valid;
  int           i_id;
  void          *io_data;
  size_t        io_size;
  uint64_t      i_len;
  uint64_t      i_data_len;
  unsigned char i_data[];
};

struct spead_api_item2{
  int           i_id;
  int           i_mode;
  int64_t       i_off;
  uint64_t      i_len;
};

struct spead_item_group {
  char      g_cd;
  uint64_t  g_items;
  uint64_t  g_size;
  uint64_t  g_off;
  void      *g_map;
};

struct coalesce_spead_data {
  struct spead_item_group *d_ig;
  struct stack *d_stack;
  int      d_imm;
  uint64_t d_len;
  uint64_t d_off;
  uint64_t s_off;
  uint64_t d_remaining;
};

struct coalesce_parcel {
  struct coalesce_spead_data  *p_c;
  struct hash_table           *p_ht;
  struct spead_api_item       *p_i;
};

/*spead shared_mem api*/

#define SHARED_MEM_REGION_SIZE  1024*1024

struct shared_mem {
  mutex                     m_m;
  struct avl_tree           *m_free;
  struct shared_mem_region  *m_current;
  uint64_t                  m_next_id;
};

struct shared_mem_region {
  mutex                       r_m;
  uint64_t                    r_id;
  uint64_t                    r_size;
  uint64_t                    r_off;
  struct shared_mem_region   *r_next;
  void                        *r_ptr;
};

struct shared_mem_free {
  struct shared_mem_free *f_next;
};

struct shared_mem_size {
  mutex                   s_m;
  size_t                  s_size;
  struct shared_mem_free  *s_top;
};

/*spead data_file api*/
#define DF_STREAM 0
#define DF_FILE   1
struct data_file{
  int         f_state;
  mutex       f_m;
  char        *f_name;
  struct stat f_fs;
  int         f_fd;
  void        *f_fmap;
  uint64_t    f_off;
};

/*spead_socket api*/
#define XSOCK_NONE       0
#define XSOCK_BOUND      1
#define XSOCK_CONNECTED  2
#define XSOCK_BOTH       3

struct spead_socket {
  char              *x_host;
  char              *x_port;
  struct addrinfo   *x_res;
  struct addrinfo   *x_active;  /*pointer to current active addrinfo in x_res*/
  int               x_fd;
  char              x_mode;
  struct ip_mreq    *x_grp;
};


/*subprocess api*/
struct u_child {
  pid_t c_pid;
  int c_fd;
};

struct spead_workers {
  struct avl_tree   *w_tree;
  int               w_count;
  fd_set            w_in;
  int               w_hfd;
};

struct spead_pipeline {
  struct stack *l_mods;
};

/*API FUNCTIONS*/

/*spead module api*/
struct spead_api_module *load_api_user_module(char *mod);
void unload_api_user_module(void *data);
int setup_api_user_module(struct spead_api_module *m);
int destroy_api_user_module(struct spead_api_module *m);
int run_api_user_callback_module(struct spead_api_module *m, struct spead_item_group *ig);

int run_module_timer_callbacks(struct spead_api_module *m);

void lock_spead_api_module_shared(struct spead_api_module_shared *s);
void unlock_spead_api_module_shared(struct spead_api_module_shared *s);
void set_data_spead_api_module_shared(struct spead_api_module_shared *s, void *data, size_t size);
void *get_data_spead_api_module_shared(struct spead_api_module_shared *s);
void clear_data_spead_api_module_shared(struct spead_api_module_shared *s);
size_t get_data_size_spead_api_module_shared(struct spead_api_module_shared *s);

/*spead pipeline api*/
struct spead_pipeline *create_spead_pipeline(struct stack *pl);
void unload_spead_pipeline(void *data);
int setup_spead_pipeline(struct spead_pipeline *l);
void destroy_spead_pipeline(void *data);
int run_callbacks_spead_pipeline(struct spead_pipeline *l, void *data);
int run_timers_spead_pipeline(struct spead_pipeline *l);


void print_data(unsigned char *buf, int size);

/*spead store api*/
struct spead_heap_store *create_store_hs(uint64_t list_len, uint64_t hash_table_count, uint64_t hash_table_size);
void destroy_store_hs(struct spead_heap_store *hs);
#if 0
int add_heap_hs(struct spead_heap_store *hs, struct spead_heap *h);
struct spead_heap *get_heap_hs(struct spead_heap_store *hs, int64_t hid);
#endif

void print_store_stats(struct spead_heap_store *hs);


void set_descriptor_flag_item_group(struct spead_item_group *ig);
int is_item_descriptor_item_group(struct spead_item_group *ig);
struct spead_item_group *create_item_group(uint64_t datasize, uint64_t nitems);
void destroy_item_group(struct spead_item_group *ig);
struct spead_api_item *new_item_from_group(struct spead_item_group *ig, uint64_t size);
struct hash_table *packetize_item_group(struct spead_heap_store *hs, struct spead_item_group *ig, int pkt_size, uint64_t hid);
int inorder_traverse_hash_table(struct hash_table *ht, int (*call)(void *data, struct spead_packet *p), void *data);
int single_traverse_hash_table(struct hash_table *ht, int (*call)(void *data, struct spead_packet *p), void *data);
void end_single_traverse_hash_table();

void print_spead_item(struct spead_api_item *itm);

#if 0
int grow_spead_item_group(struct spead_item_group *ig, uint64_t extradata, uint64_t extranitems);
struct spead_api_item *get_spead_item(struct spead_item_group *ig, uint64_t n);
#endif
struct spead_api_item *get_next_spead_item(struct spead_item_group *ig, struct spead_api_item *itm);
struct spead_api_item *get_spead_item_with_id(struct spead_item_group *ig, uint64_t iid);
struct spead_api_item *get_spead_item_at_off(struct spead_item_group *ig, uint64_t off);
int set_spead_item_io_data(struct spead_api_item *itm, void *ptr, size_t size);
int copy_to_spead_item(struct spead_api_item *itm, void *src, size_t len);
int append_copy_to_spead_item(struct spead_api_item *itm, void *src, size_t len);
int set_item_data_ones(struct spead_api_item *itm);
int set_item_data_zeros(struct spead_api_item *itm);
int set_item_data_ramp(struct spead_api_item *itm);


void destroy_spead_item2(void *data);
#if 0
struct spead_api_item *init_spead_api_item(struct spead_api_item *itm, int vaild, int id, int len, unsigned char *data);
#endif

int process_packet_hs(struct u_server *s, struct spead_pipeline *l, struct hash_o *o);  

char *hr_spead_id(uint64_t sid);

/*shared_mem api*/
int create_shared_mem();
void destroy_shared_mem();
void *shared_malloc(size_t size);
void shared_free(void *ptr, size_t size);


/*spead data file*/
struct data_file *load_raw_data_file(char *fname);
struct data_file *write_raw_data_file(char *fname);
void destroy_raw_data_file(struct data_file *f);
size_t get_data_file_size(struct data_file *f);
char *get_data_file_name(struct data_file *f);
void *get_data_file_ptr_at_off(struct data_file *f, uint64_t off);
int64_t request_chunk_datafile(struct data_file *f, uint64_t len, void **ptr, uint64_t *chunk_off_rtn);
int64_t request_packet_raw_packet_datafile(struct data_file *f, void **ptr);
int write_chunk_raw_data_file(struct data_file *f, uint64_t off, void *src, uint64_t len);
int write_next_chunk_raw_data_file(struct data_file *f, void *src, uint64_t len);

char *itoa(int64_t i, char b[]);

/*spead socket api*/
void destroy_spead_socket(struct spead_socket *x);
struct spead_socket *create_spead_socket(char *host, char *port);
struct spead_socket *create_raw_ip_spead_socket(char *host);
int bind_spead_socket(struct spead_socket *x);
int connect_spead_socket(struct spead_socket *x);
int set_broadcast_opt_spead_socket(struct spead_socket *x);
int set_multicast_send_opts_spead_socket(struct spead_socket *x, char *host);
int set_multicast_receive_opts_spead_socket(struct spead_socket *x, char *grp, char *interface);
int unset_multicast_receive_opts_spead_socket(struct spead_socket *x);
int get_fd_spead_socket(struct spead_socket *x);
struct addrinfo *get_addr_spead_socket(struct spead_socket *x);
int send_packet_spead_socket(void *data, struct spead_packet *p); // data should be a spead_tx data structure
int send_raw_data_spead_socket(void *obj, void *data, uint64_t len); //obj should be a spead_tx
int send_spead_stream_terminator(struct spead_tx *tx);

/*spead workers subprocess api*/
void destroy_child_sp(void *data);
struct u_child *fork_child_sp(struct spead_pipeline *l, void *data, int (*call)(void *data, struct spead_pipeline *l, int cfd));
int add_child_us(struct u_child ***cs, struct u_child *c, int size);

struct spead_workers *create_spead_workers(struct spead_pipeline *l, void *data, long count, int (*call)(void *data, struct spead_pipeline *l, int cfd));
void destroy_spead_workers(struct spead_workers *w);
int wait_spead_workers(struct spead_workers *w);
int get_count_spead_workers(struct spead_workers *w);
int populate_fdset_spead_workers(struct spead_workers *w);

int get_high_fd_spead_workers(struct spead_workers *w);
fd_set *get_in_fd_set_spead_workers(struct spead_workers *w);

/*spead worker compare function*/
int compare_spead_workers(const void *v1, const void *v2);

int check_spead_version(char *version);

int sub_time(struct timeval *delta, struct timeval *alpha, struct timeval *beta);
void print_time(struct timeval *result, uint64_t bytes);

#endif

