#ifndef SPEAD_API_H
#define SPEAD_API_H

#include "spead_packet.h"
#include "hash.h"
#include "server.h"

#define CALLBACK "spead_api_callback"

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

void print_data(unsigned char *buf, int size);


struct spead_heap_store{
  int64_t             s_backlog;
  struct hash_o_list  *s_list;
  struct hash_table   **s_hash;
};

struct spead_heap_store *create_store_hs(uint64_t list_len, uint64_t hash_table_count, uint64_t hash_table_size);
void destroy_store_hs(struct spead_heap_store *hs);
int add_heap_hs(struct spead_heap_store *hs, struct spead_heap *h);
struct spead_heap *get_heap_hs(struct spead_heap_store *hs, int64_t hid);

void print_store_stats(struct spead_heap_store *hs);

struct spead_api_item{
  int           i_valid;
  int           i_id;
  uint64_t      i_len;
  unsigned char i_data[];
};

struct spead_item_group {
  uint64_t  g_items;
  uint64_t  g_size;
  uint64_t  g_off;
  void      *g_map;
};

struct spead_item_group *create_item_group(uint64_t datasize, uint64_t nitems);
void destroy_item_group(struct spead_item_group *ig);
struct spead_api_item *new_item_from_group(struct spead_item_group *ig, uint64_t size);

struct spead_api_item *get_spead_item(struct spead_item_group *ig, uint64_t n);

int process_packet_hs(struct u_server *s, struct hash_o *o);

  

#endif

