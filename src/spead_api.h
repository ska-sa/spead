#ifndef SPEAD_API_H
#define SPEAD_API_H

#include "spead_packet.h"
#include "hash.h"
#include "server.h"

struct spead_heap_store{
  int64_t s_backlog;
#if 0
  struct spead_heap **s_heaps;
#endif
  struct spead_heap *s_shipping;
  
  struct hash_o_list *s_list;
  struct hash_table **s_hash;
};

struct spead_api_item{
  int i_valid;
  int i_id;
  uint64_t i_len;
  unsigned char i_data[];
#if 0
  unsigned char i_data[SPEAD_ADDRLEN];
#endif
};

struct spead_item_group {
  uint64_t g_items;
  uint64_t g_size;
  uint64_t g_off;
  void *g_map;
};

struct spead_item_group *create_item_group(uint64_t datasize, uint64_t nitems);
void destroy_item_group(struct spead_item_group *ig);
struct spead_api_item *new_item_from_group(struct spead_item_group *ig, uint64_t size);

struct spead_heap *create_spead_heap();
void destroy_spead_heap(struct spead_heap *h);
#if 0
struct spead_packet *create_spead_packet();
#endif
void *create_spead_packet();
void destroy_spead_packet(void *data);


struct spead_heap_store *create_store_hs(uint64_t list_len, uint64_t hash_table_count, uint64_t hash_table_size);
void destroy_store_hs(struct spead_heap_store *hs);
int add_heap_hs(struct spead_heap_store *hs, struct spead_heap *h);
struct spead_heap *get_heap_hs(struct spead_heap_store *hs, int64_t hid);


int process_packet_hs(struct u_server *s, struct hash_o *o);

#if 0
int ship_heap_hs(struct spead_heap_store *hs, int64_t id); 
int process_heap_hs(struct spead_heap_store *hs, struct spead_heap *h);
#endif
  



#endif

