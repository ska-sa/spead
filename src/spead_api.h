#ifndef SPEAD_API_H
#define SPEAD_API_H

#include "spead_packet.h"
#include "hash.h"

struct spead_heap_store{
  int64_t s_backlog;
  int64_t s_count;
#if 0
  struct spead_heap **s_heaps;
#endif
  struct spead_heap *s_shipping;
  
  struct hash_o_list *s_list;
  struct hash_table **s_hash;
};


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


int process_packet_hs(struct spead_heap_store *hs, struct hash_o *o);

int ship_heap_hs(struct spead_heap_store *hs, int64_t id); 

int process_heap_hs(struct spead_heap_store *hs, struct spead_heap *h);

  



#endif

