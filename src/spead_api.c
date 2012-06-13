/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include "spead_api.h"

struct spead_heap *create_spead_heap()
{
  struct spead_heap *h;

  h = malloc(sizeof(struct spead_heap));
  if (h == NULL)
    return h;

  spead_heap_init(h);

  return h;
}

void destroy_spead_heap(struct spead_heap *h)
{
  if (h != NULL){
    spead_heap_wipe(h);
    free(h);
  }
}


struct spead_packet *create_spead_packet()
{
  struct spead_packet *p;

  p = malloc(sizeof(struct spead_packet));
  if (p == NULL)
    return p;

  spead_packet_init(p);

  return p;
}

void destroy_spead_packet(struct spead_packet *p)
{
  if (p != NULL){
    free(p);
  }
}


struct spead_heap_store *create_store_hs()
{
  struct spead_heap_store *hs;

  hs = malloc(sizeof(struct spead_heap_store));
  if (hs == NULL)
    return NULL;

  hs->s_backlog  = 3;
  hs->s_count    = 0;
  hs->s_heaps    = NULL;
  hs->s_shipping = NULL;

  return hs;
}

void destroy_store_hs(struct spead_heap_store *hs)
{
  int i;
  if (hs){
    if (hs->s_heaps){
      for (i=0; i<hs->s_count; i++){
        destroy_spead_heap(hs->s_heaps[i]);
      }
      free(hs->s_heaps);
    }
    if (hs->s_shipping)
      free(hs->s_shipping);
    free(hs);
  }
}

int64_t get_id_hs(struct spead_heap_store *hs, int64_t hid)
{
  if (hs == NULL)
    return -1;

  return hid % hs->s_backlog;
}

int add_heap_hs(struct spead_heap_store *hs, struct spead_heap *h)
{
  int64_t id;
  int i;
  
  if (hs == NULL || h == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error cannot work with NULL heap store or NULL heap\n", __func__);
#endif 
    return -1;
  }
  
  id = h->heap_cnt % hs->s_backlog;

#ifdef DEBUG
  fprintf(stderr, "%s: calculated id to [%ld]\n", __func__, id);
#endif
  
  if (hs->s_count <= id) {
    hs->s_heaps = realloc(hs->s_heaps, sizeof(struct spead_heap*) * (id + 1));
    if (hs->s_heaps == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: logic error cannot realloc for heap store\n", __func__);
#endif 
      return -1;
    }
    
    for (i = hs->s_count; i<id+1; i++)
      hs->s_heaps[i] = NULL;
      
    hs->s_count = id + 1;
  }

  hs->s_heaps[id] = h;

#ifdef DEBUG
  fprintf(stderr, "%s: inserted heap [%ld] @ id: [%ld] into heap_store [sc: %ld]\n", __func__, h->heap_cnt, id, hs->s_count);
#endif
  
  return id;
}

int ship_heap_hs(struct spead_heap_store *hs, int64_t id)
{
  int rtn;

  if (hs == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot ship null heap\n", __func__);
#endif
    return -1;
  }
  
  if (hs->s_count <= id) {
#ifdef DEBUG
    fprintf(stderr, "%s: ALL heaps shipped\n", __func__);
#endif
    return -1;
  }

#ifdef DEBUG
  fprintf(stderr, "%s:\tSHIP HEAP [%ld]\n", __func__, id);
#endif

  hs->s_shipping = hs->s_heaps[id];

  hs->s_heaps[id] = NULL;

  rtn = spead_heap_got_all_packets(hs->s_shipping);
  if (rtn) {
#ifdef DATA
    fprintf(stderr, "%s: COMPLETED HEAP [%ld] rtn=[%d]\n", __func__, hs->s_shipping->heap_cnt, rtn);
#endif
  } else {
#ifdef DATA
    fprintf(stderr, "%s: PARTIAL HEAP [%ld] rtn=[%d]\n", __func__, hs->s_shipping->heap_cnt, rtn);
#endif
  }

  if (spead_heap_finalize(hs->s_shipping) == SPEAD_ERR){
#ifdef DEBUG
    fprintf(stderr, "%s: error trying to finalize spead heap\n", __func__);
#endif 
  } else {
#ifdef DEBUG
    fprintf(stderr, "%s: finalize spead heap SUCCESS!!\n", __func__);
#endif 
  }

  destroy_spead_heap(hs->s_shipping);
  
  hs->s_shipping = NULL;

  return 0;
}

struct spead_heap *get_heap_hs(struct spead_heap_store *hs, int64_t hid)
{
  struct spead_heap *h;
  int id;
  
  if (hs == NULL || hs->s_heaps == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error cannot address requested heap\n", __func__);
#endif 
    return NULL;
  }
  
  id = hid % hs->s_backlog;
#if 0
  def DEBUG
  fprintf(stderr, "%s: calculated id to [%d]\n", __func__, id);
#endif

  if (hs->s_count <= id){
#ifdef DEBUG
    fprintf(stderr, "%s: error cannot address requested heap\n", __func__);
#endif 
    return NULL;
  }

  h = hs->s_heaps[id];
  if (h == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: requested heap is null\n", __func__);
#endif 
    return NULL;
  }

  if (h->heap_cnt != hid){
#ifdef DEBUG
    fprintf(stderr, "%s: heap id [%ld] not in heap store\n", __func__, h->heap_cnt);
#endif 
  
    if (ship_heap_hs(hs, id) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error shipping heap [%d]\n", __func__, id);
#endif
    }

    return NULL;
  }

  return h;
}

int store_packet_hs(struct spead_heap_store *hs, struct spead_packet *p)
{
  struct spead_heap *h;
  int64_t id;

  if (hs == NULL || p == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error invalid data\n", __func__);
#endif 
    return -1;
  }
  
  h = get_heap_hs(hs, p->heap_cnt);

  if (h == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: first packet for heap [%ld]\n", __func__, p->heap_cnt);
#endif 

    h = create_spead_heap();
    if (h == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: error could not create heap\n", __func__);
#endif 
      return -1;
    }

    if (spead_heap_add_packet(h, p) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error could not add packet to heap\n", __func__);
#endif 
      destroy_spead_heap(h);
      return -1;
    }

    if (add_heap_hs(hs, h) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error could not add heap to heap store\n", __func__);
#endif 
      destroy_spead_heap(h);
      return -1;
    }
    
    return 0;
  }

  if (spead_heap_add_packet(h, p) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: error could not add packet to heap\n", __func__);
#endif 
    return -1;
  }

  if (spead_heap_got_all_packets(h)){
#ifdef DATA
    fprintf(stderr, "%s: COMPLETE heap about to ship\n", __func__);
#endif 
    
    id = get_id_hs(hs, h->heap_cnt);
    
    if (ship_heap_hs(hs, id) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error shipping heap [%ld]\n", __func__, id);
#endif
    }

  }
    
  return 0;
}

int process_packet_hs(struct spead_heap_store *hs, struct spead_packet *p)
{
  int rtn, i;

  if (hs == NULL || p == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error invalid data\n", __func__);
#endif 
    return -1;
  }
  
  if (spead_packet_unpack_header(p) == SPEAD_ERR){
#ifdef DEBUG
    fprintf(stderr, "%s: unable to unpack spead header for packet (%p)\n", __func__, p);
#endif
    return -1;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: unpacked spead header for packet (%p)\n", __func__, p);
#endif

  if (spead_packet_unpack_items(p) == SPEAD_ERR){
#ifdef DEBUG
    fprintf(stderr, "%s: unable to unpack spead items for packet (%p)\n", __func__, p);
#endif
    return -1;
  } 

#ifdef DEBUG
  fprintf(stderr, "%s: unpacked spead items for packet (%p) from heap %ld\n", __func__, p, p->heap_cnt);
#endif

  if (p->is_stream_ctrl_term){
#ifdef DEBUG
    fprintf(stderr, "%s: GOT STREAM TERMINATOR\n", __func__);
#endif
    
    for (i=0; !ship_heap_hs(hs, i); i++);

    return 0;
  } else {
    rtn = store_packet_hs(hs, p);
  }

  return rtn;
}
