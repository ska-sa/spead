/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "hash.h"
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


void *create_spead_packet()
{
  struct spead_packet *p;

  p = malloc(sizeof(struct spead_packet));
  if (p == NULL)
    return p;

  spead_packet_init(p);

  return p;
}

void destroy_spead_packet(void *data)
{
  struct spead_packet *p;
  p = data;
  if (p != NULL){
    free(p);
  }
}

uint64_t hash_fn_spead_packet(struct hash_table *t, uint64_t in)
{
  if (t == NULL || t->t_len < 0)
    return -1;
  return in % t->t_len;
}

struct spead_heap_store *create_store_hs(uint64_t list_len)
{
  struct spead_heap_store *hs;

  hs = malloc(sizeof(struct spead_heap_store));
  if (hs == NULL)
    return NULL;

  hs->s_backlog  = 10;
  hs->s_count    = 0;
  hs->s_heaps    = NULL;
  hs->s_shipping = NULL;

  hs->s_hash  = NULL;
  hs->s_list  = NULL;

  hs->s_list = create_o_list(list_len, &create_spead_packet, &destroy_spead_packet, sizeof(struct spead_packet));
  if (hs->s_list == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: failed to create spead packet bank size [%ld]\n", __func__, list_len);
#endif
    destroy_store_hs(hs);
    return NULL;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: created spead packet bank of size [%ld]\n", __func__, list_len);
#endif

  hs->s_hash = create_hash_table(hs->s_list, 0, list_len, &hash_fn_spead_packet);
  if (hs->s_hash == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: failed to create spead packet hash table size [%ld]\n", __func__, list_len);
#endif
    destroy_store_hs(hs);
    return NULL; 
  }

#ifdef DEBUG
  fprintf(stderr, "%s: created spead packet hash table of size [%ld]\n", __func__, list_len);
#endif

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

    if (hs->s_hash)
      destroy_hash_table(hs->s_hash);
  
    if (hs->s_list)
      destroy_o_list(hs->s_list);
    

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

int process_heap_hs(struct spead_heap_store *hs, struct spead_heap *h)
{
  struct spead_packet *p;
  struct spead_heap *th;
  struct spead_item *itm, *itm2;
  int i;

  if (h == NULL)
    return -1;

  h = hs->s_shipping;
  itm = h->head_item;

     
  do { 
    
#ifdef DEBUG
    fprintf(stderr, "ITEM\n\tis_valid:\t%d\n\tid:\t%d\n\tlen:\t%ld\n", itm->is_valid, itm->id, itm->len);
#endif

    switch(itm->id){

      case SPEAD_DESCRIPTOR_ID:
        p = create_spead_packet();
        if (p == NULL){
#ifdef DEBUG
          fprintf(stderr,"%s: cannot create temp packet\n", __func__);
#endif
          return -1;
        }
        
        /*we have a spead packet inside itm->val*/
        memcpy(p->data, itm->val, itm->len);

        if (spead_packet_unpack_header(p) == SPEAD_ERR){
#ifdef DEBUG
          fprintf(stderr, "%s: unable to unpack spead header for Item Descriptor packet (%p)\n", __func__, p);
#endif
          break;
        }

#ifdef DEBUG
        fprintf(stderr, "%s: unpacked spead header for Item Descriptor packet (%p)\n", __func__, p);
#endif

        if (spead_packet_unpack_items(p) == SPEAD_ERR){
#ifdef DEBUG
          fprintf(stderr, "%s: unable to unpack spead items for Item Descriptor packet (%p)\n", __func__, p);
#endif
          break;
        } 

#ifdef DEBUG
        fprintf(stderr, "%s: unpacked spead items for Item Descriptor packet (%p) from heap %ld po %ld of %ld\n", __func__, p, p->heap_cnt, p->payload_off, p->heap_len);
#endif

        th = create_spead_heap();
        if (th == NULL){
#ifdef DEBUG
          fprintf(stderr, "%s: unable to create Item Descriptor temp heap\n", __func__);
#endif
          break;
        }

        if (spead_heap_add_packet(th, p) < 0){
#ifdef DEBUG
          fprintf(stderr, "%s: error could not add Item Descriptor packet to heap\n", __func__);
#endif 
          destroy_spead_packet(p);
          destroy_spead_heap(h);
          return -1;
        }

        if (spead_heap_got_all_packets(th)) {
#ifdef DEBUG
          fprintf(stderr, "[%d] %s: COMPLETED Item Descriptor HEAP [%ld]\n", getpid(), __func__, th->heap_cnt);
#endif
        } else {
#ifdef DEBUG
          fprintf(stderr, "[%d] %s: PARTIAL Item Descriptor HEAP [%ld]\n", getpid(), __func__, th->heap_cnt);
#endif
        }

        if (spead_heap_finalize(th) == SPEAD_ERR){
#ifdef DEBUG
          fprintf(stderr, "%s: error trying to finalize Item Descriptor spead heap\n", __func__);
#endif 
        } else {
#ifdef DEBUG
          fprintf(stderr, "%s: finalize Item Descriptor spead heap SUCCESS!!\n", __func__);
#endif 
        }

        itm2 = th->head_item;
        do {
#ifdef DEBUG
          fprintf(stderr, "ITEM DESCRIPTOR\n\tis_valid:\t%d\n\tid:\t%d\n\tlen:\t%ld\n\tval:\n", itm2->is_valid, itm2->id, itm2->len);
          for(i=0; i<itm2->len; i++){
            fprintf(stderr,"%X[%c] ", itm2->val[i], itm2->val[i]);
          }
          fprintf(stderr,"\n");
#endif

        } while((itm2 = itm2->next) != NULL);

        destroy_spead_heap(th);
        
        break;
      
      default:
        continue;
    }
#if 0
    for(i=0; i<itm->len; i++){
      fprintf(stderr,"0x%X ", itm->val[i]);
    }
    fprintf(stderr,"\n");
#endif
  } while ((itm = itm->next) != NULL);
  
  return 0;
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

  hs->s_shipping = hs->s_heaps[id];

#ifdef DEBUG
  fprintf(stderr, "%s:\tSHIP HEAP hs:[%ld]->[%ld]\n", __func__, id, hs->s_shipping->heap_cnt);
#endif

  hs->s_heaps[id] = NULL;

  rtn = spead_heap_got_all_packets(hs->s_shipping);
  if (rtn) {
#ifdef DATA
    fprintf(stderr, "[%d] %s: COMPLETED HEAP [%ld] SHIPPED rtn=[%d]\n", getpid(), __func__, hs->s_shipping->heap_cnt, rtn);
#endif
  } else {
#ifdef DATA
    fprintf(stderr, "[%d] %s: PARTIAL HEAP [%ld] SHIPPED rtn=[%d]\n", getpid(), __func__, hs->s_shipping->heap_cnt, rtn);
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
  
  if (process_heap_hs(hs, hs->s_shipping) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot process heap\n", __func__);
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
    
    if (spead_heap_got_all_packets(h)){
#ifdef DEBUG
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

  if (spead_heap_add_packet(h, p) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: error could not add packet to heap\n", __func__);
#endif 
    return -1;
  }

  if (spead_heap_got_all_packets(h)){
#ifdef DEBUG
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
  int rtn;

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
  fprintf(stderr, "%s: unpacked spead items for packet (%p) from heap %ld po %ld of %ld\n", __func__, p, p->heap_cnt, p->payload_off, p->heap_len);
#endif

  if (p->is_stream_ctrl_term){
#ifdef DEBUG
    fprintf(stderr, "%s: GOT STREAM TERMINATOR\n", __func__);
#endif
    
    //for (i=0; !ship_heap_hs(hs, i); i++);

    return 0;
  } else {
    rtn = store_packet_hs(hs, p);
  }

  return rtn;
}

