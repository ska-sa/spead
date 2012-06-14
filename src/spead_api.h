#ifndef SPEAD_API_H
#define SPEAD_API_H

#include "spead_packet.h"

struct spead_heap_store{
  int64_t s_backlog;
  int64_t s_count;
  struct spead_heap **s_heaps;
  struct spead_heap *s_shipping;
};


struct spead_heap *create_spead_heap();
void destroy_spead_heap(struct spead_heap *h);
struct spead_packet *create_spead_packet();
void destroy_spead_packet(struct spead_packet *p);


struct spead_heap_store *create_store_hs();
void destroy_store_hs(struct spead_heap_store *hs);
int add_heap_hs(struct spead_heap_store *hs, struct spead_heap *h);
struct spead_heap *get_heap_hs(struct spead_heap_store *hs, int64_t hid);


int process_packet_hs(struct spead_heap_store *hs, struct spead_packet *p);

int ship_heap_hs(struct spead_heap_store *hs, int64_t id); 

int process_heap_hs(struct spead_heap_store *hs, struct spead_heap *h);

  



#endif

