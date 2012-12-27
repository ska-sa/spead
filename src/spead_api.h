/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#ifndef SPEAD_API_H
#define SPEAD_API_H

#include <sys/stat.h>
#include <sys/select.h>

#include "spead_packet.h"
#include "hash.h"
#include "server.h"


#define SPEAD_ZEROS_ID     0x112
#define SPEAD_ONES_ID      0x113
#define SPEAD_RAMP_ID      0x114


/*API DATASTRUCTURES*/ 

/*modules api*/
#define SAPI_CALLBACK "spead_api_callback"
#define SAPI_SETUP    "spead_api_setup"
#define SAPI_DESTROY  "spead_api_destroy"

struct spead_api_module {
  void *m_handle;
  void *(*m_setup)();
  int  (*m_cdfn)();
  int  (*m_destroy)(void *data);
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
  unsigned char i_data[];
};

struct spead_item_group {
  uint64_t  g_items;
  uint64_t  g_size;
  uint64_t  g_off;
  void      *g_map;
};


/*spead shared_mem api*/
struct shared_mem {
  uint64_t  m_size;
  uint64_t  m_off;
  void      *m_ptr;
};


/*spead data_file api*/
struct data_file{
  struct stat f_fs;
  int         f_fd;
  void        *f_fmap;
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


/*API FUNCTIONS*/

/*spead module api*/
struct spead_api_module *load_api_user_module(char *mod);
void unload_api_user_module(struct spead_api_module *m);
int setup_api_user_module(struct spead_api_module *m);
int destroy_api_user_module(struct spead_api_module *m);
int run_api_user_callback_module(struct spead_api_module *m, struct spead_item_group *ig);
void print_data(unsigned char *buf, int size);


/*spead store api*/
struct spead_heap_store *create_store_hs(uint64_t list_len, uint64_t hash_table_count, uint64_t hash_table_size);
void destroy_store_hs(struct spead_heap_store *hs);
#if 0
int add_heap_hs(struct spead_heap_store *hs, struct spead_heap *h);
struct spead_heap *get_heap_hs(struct spead_heap_store *hs, int64_t hid);
#endif

void print_store_stats(struct spead_heap_store *hs);

struct spead_item_group *create_item_group(uint64_t datasize, uint64_t nitems);
void destroy_item_group(struct spead_item_group *ig);
struct spead_api_item *new_item_from_group(struct spead_item_group *ig, uint64_t size);
struct hash_table *packetize_item_group(struct spead_heap_store *hs, struct spead_item_group *ig, int pkt_size, uint64_t hid);
int inorder_traverse_hash_table(struct hash_table *ht, int (*call)(void *data, struct spead_packet *p), void *data);
#if 0
int grow_spead_item_group(struct spead_item_group *ig, uint64_t extradata, uint64_t extranitems);
struct spead_api_item *get_spead_item(struct spead_item_group *ig, uint64_t n);
#endif
struct spead_api_item *get_next_spead_item(struct spead_item_group *ig, struct spead_api_item *itm);
struct spead_api_item *get_spead_item_at_off(struct spead_item_group *ig, uint64_t off);
int set_spead_item_io_data(struct spead_api_item *itm, void *ptr, size_t size);
int copy_to_spead_item(struct spead_api_item *itm, void *src, size_t len);

int set_item_data_ones(struct spead_api_item *itm);
int set_item_data_zeros(struct spead_api_item *itm);
int set_item_data_ramp(struct spead_api_item *itm);

#if 0
struct spead_api_item *init_spead_api_item(struct spead_api_item *itm, int vaild, int id, int len, unsigned char *data);
#endif

int process_packet_hs(struct u_server *s, struct spead_api_module *m, struct hash_o *o);  


/*shared_mem api*/
int create_shared_mem(uint64_t size);
void destroy_shared_mem();
void *shared_malloc(size_t size);


/*spead data file*/
struct data_file *load_raw_data_file(char *fname);
void destroy_raw_data_file(struct data_file *f);
size_t get_data_file_size(struct data_file *f);
void *get_data_file_ptr_at_off(struct data_file *f, uint64_t off);


/*spead socket api*/
void destroy_spead_socket(struct spead_socket *x);
struct spead_socket *create_spead_socket(char *host, char *port);
int bind_spead_socket(struct spead_socket *x);
int connect_spead_socket(struct spead_socket *x);
int set_broadcast_opt_spead_socket(struct spead_socket *x);
int get_fd_spead_socket(struct spead_socket *x);
struct addrinfo *get_addr_spead_socket(struct spead_socket *x);
int send_packet_spead_socket(void *data, struct spead_packet *p); // data should be a spead_socket
int send_spead_stream_terminator(struct spead_socket *x);


/*spead workers subprocess api*/
void destroy_child_sp(void *data);
struct u_child *fork_child_sp(struct spead_api_module *m, void *data, int (*call)(void *data, struct spead_api_module *m, int cfd));
int add_child_us(struct u_child ***cs, struct u_child *c, int size);

struct spead_workers *create_spead_workers(void *data, long count, int (*call)(void *data, struct spead_api_module *m, int cfd));
void destroy_spead_workers(struct spead_workers *w);
int wait_spead_workers(struct spead_workers *w);
int get_count_spead_workers(struct spead_workers *w);
int populate_fdset_spead_workers(struct spead_workers *w);

int get_high_fd_spead_workers(struct spead_workers *w);
fd_set *get_in_fd_set_spead_workers(struct spead_workers *w);

/*spead worker compare function*/
int compare_spead_workers(const void *v1, const void *v2);



#endif

