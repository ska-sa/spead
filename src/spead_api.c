/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <string.h>
#include <unistd.h>
#include <endian.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "hash.h"
#include "spead_api.h"
#include "sharedmem.h"
#include "server.h"

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

  p = shared_malloc(sizeof(struct spead_packet));
  if (p == NULL)
    return p;

  spead_packet_init(p);

  return p;
}

void destroy_spead_packet(void *data)
{
#if 0
  struct spead_packet *p;
  p = data;
  if (p != NULL){
    free(p);
  }
#endif
}

uint64_t hash_fn_spead_packet(struct hash_table *t, struct hash_o *o)
{
  struct spead_packet *p;
  uint64_t po, hl, id;

  po = 0;
  hl = 0;

  if (t == NULL || o == NULL)
    return -1;

  p = get_data_hash_o(o);
  if (p == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot get packet from object (%p)\n", __func__, o);
#endif 
    return -1;
  }
  
  po = p->payload_off;
  hl = p->heap_len;

  if (t->t_len <= 0)
    return -1;

  id = po / (hl / t->t_len);
  
  return id;
}

struct spead_heap_store *create_store_hs(uint64_t list_len, uint64_t hash_table_count, uint64_t hash_table_size)
{
  struct spead_heap_store *hs;
  int i;

  hs = malloc(sizeof(struct spead_heap_store));
  if (hs == NULL)
    return NULL;

  hs->s_backlog  = hash_table_count;
#if 0
  hs->s_heaps    = NULL;
#endif
  hs->s_shipping = NULL;

  hs->s_hash  = NULL;
  hs->s_list  = NULL;

  hs->s_list = create_o_list(list_len, hash_table_count, hash_table_size, &create_spead_packet, &destroy_spead_packet, sizeof(struct spead_packet));
  if (hs->s_list == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: failed to create spead packet bank size [%ld]\n", __func__, list_len);
#endif
    destroy_store_hs(hs);
    return NULL;
  }

#if DEBUG>1
  fprintf(stderr, "%s: created spead packet bank of size [%ld]\n", __func__, list_len);
#endif

  hs->s_hash = shared_malloc(sizeof(struct hash_table *) * hash_table_count);
  if (hs->s_hash == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: could not get shared memory space for hash_tables\n", __func__);
#endif
    destroy_store_hs(hs);
    return NULL;
  }
    
  for (i=0; i<hs->s_backlog; i++){
    hs->s_hash[i] = create_hash_table(hs->s_list, i, hash_table_size, &hash_fn_spead_packet);
    if (hs->s_hash[i] == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: failed to create spead packet hash table size [%ld]\n", __func__, list_len);
#endif
      destroy_store_hs(hs);
      return NULL; 
    }
  }

#if DEBUG>1
  fprintf(stderr, "%s: created [%ld] spead packet hash tables of size [%ld]\n", __func__, hash_table_count, hash_table_size);
#endif

  return hs;
}

void destroy_store_hs(struct spead_heap_store *hs)
{
  int i;

  if (hs){

    if (hs->s_hash){
      for (i=0; i<hs->s_backlog; i++){
        destroy_hash_table(hs->s_hash[i]);
      }
    }

    if (hs->s_shipping)
      free(hs->s_shipping);

#if 0
    if (hs->s_hash)
      destroy_hash_table(hs->s_hash);
#endif
  
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


struct hash_table *get_ht_hs(struct spead_heap_store *hs, uint64_t hid)
{
  uint64_t id;
  struct hash_table *ht;

  if (hs == NULL || id < 0)
    return NULL;

  id = get_id_hs(hs, hid);
  
  if (hs->s_hash == NULL)
    return NULL;

  ht = hs->s_hash[id];
  if (ht == NULL)
    return NULL;

  if (ht->t_data_id < 0){
#if 0
    lock_sem(ht->t_semid);
#endif
    ht->t_data_id = hid;
#if 0
    unlock_sem(ht->t_semid);
#endif
  } 
  
  if (ht->t_data_id != hid){
#ifdef DEBUG
    fprintf(stderr, "%s: hash table [%ld] / packet set [%ld] miss match\n", __func__, hid, ht->t_data_id);
#endif
    return NULL;
  }

  return ht;
}

struct spead_item_group *create_item_group(uint64_t datasize, uint64_t nitems)
{
  struct spead_item_group *ig;
  
  if (datasize <= 0 || nitems <= 0)
    return NULL;

  ig = malloc(sizeof(struct spead_item_group));
  if (ig == NULL)
    return NULL;

  ig->g_items = nitems;
  ig->g_off   = 0;
  ig->g_size  = datasize + nitems*(sizeof(struct spead_api_item) + sizeof(uint64_t));
  ig->g_map   = mmap(NULL, ig->g_size, PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, (-1), 0);

  if (ig->g_map == NULL){
    free(ig);
    return NULL;
  }
#ifdef DEBUG
  fprintf(stderr, "CREATE ITEM GROUP with map size [%ld] bytes\n", ig->g_size);
#endif
  return ig;
}

void destroy_item_group(struct spead_item_group *ig)
{
  if (ig){
    
    if (ig->g_map)
      munmap(ig->g_map, ig->g_size); 
  
    free(ig);
  }
}

struct spead_api_item *new_item_from_group(struct spead_item_group *ig, uint64_t size)
{
  struct spead_api_item *itm;
  
  if (ig == NULL || size <= 0)
    return NULL;
    
  if ((ig->g_off + size) > ig->g_size){
#ifdef DEBUG
    fprintf(stderr, "%s: parameter error (ig->g_off + size) %ld ig->g_size %ld\n", __func__, ig->g_off+size, ig->g_size);
#endif
    return NULL;
  }

  itm = (struct spead_api_item *) (ig->g_map + ig->g_off);
  if (itm == NULL)
    return NULL;

  ig->g_off += size;
  ig->g_items++;

#if DEBUG>2
  fprintf(stderr, "GROUP map (%p): size %ld offset: %ld data: %p\n", ig->g_map, ig->g_size, ig->g_off, itm->i_data);
#endif

  return itm;
}

int process_items(struct hash_table *ht, int (*cdfn)())
{
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

  struct spead_item_group *ig;
  struct spead_api_item   *itm;

  struct hash_o *o;
  struct spead_packet *p;
  int i, j, id, mode, state;
  int64_t iptr, off, data64;

  struct process_state {
    int i;
    int j;
    int id;
    int state;
    int64_t off;
    struct spead_packet *p;
    struct hash_o *o;
  } ps;

  struct data_state {
    int i;
    int64_t off;
    int64_t cc;
    struct spead_packet *p;
    struct hash_o *o;
  } ds;

  ps.i     = 0;
  ps.j     = 0;
  ps.id    = 0;
  ps.state = (-1);
  ps.off   = (-1);
  ps.p     = NULL;
  ps.o     = NULL;

  ds.i     = 0;
  ds.cc    = 0;
  ds.off   = (-1);
  ds.p     = NULL;
  ds.o     = NULL;

  if (ht == NULL || ht->t_os == NULL)
    return -1;

#ifdef DEBUG
  fprintf(stderr, "HEAP CNT [%ld] HEAP ITEMS [%ld] HEAP DATA [%ld bytes]\n", ht->t_data_id, ht->t_items, ht->t_data_count); 
#endif

  ig = create_item_group(ht->t_data_count, ht->t_items);
  if (ig == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: failed to create item group\n", __func__);
#endif
    return -1;
  }

#ifdef PROCESS 
  fprintf(stderr, "--PROCESS-[%d]-BEGIN---\n",getpid());
#endif

  id     = 0;
  iptr   = 0;
  i      = 0;
  j      = 0;
  mode   = 0;
  itm    = NULL;
  o      = NULL;
  p      = NULL;
  state  = S_GET_OBJECT;

  while (state){

    switch (state){

      case S_GET_OBJECT:
        if (i < ht->t_len){
          o = ht->t_os[i];
          if (o == NULL){
            i++;
            j=0;
            state = S_GET_OBJECT;
            break;
          } 
          state = S_GET_PACKET;
        } else {
          /*still need to get any direct items*/
#ifdef PROCESS
          fprintf(stderr, "Last Direct ITEM[%d] in pkt(%p) SIZE [%ld] bytes\n", ps.j, ps.p, p->heap_len - ps.off);
#endif
          ps.state = S_END;

          ds.cc = p->heap_len - ps.off; 
                    
          itm = new_item_from_group(ig, sizeof(struct spead_api_item) + ds.cc);
          if (itm){
            itm->i_id   = ps.id;
            itm->i_len  = ds.cc;

            if (ds.off == -1){
              ds.i   = i;
              ds.p   = p;
              ds.o   = o;
              ds.off = ps.off;
            }
          }

          state = S_DIRECT_COPY;
        }
        break;

      case S_GET_PACKET:
        p = get_data_hash_o(o);
        if (p == NULL){
          state = S_NEXT_PACKET;
          break;
        }
#if DEBUG>2
        fprintf(stderr, "--pkt-- in o (%p) has p (%p)\n  payload_off: %ld payload_len: %ld payload: %p\n", o, p, p->payload_off, p->payload_len, p->payload);
#endif
        state = S_GET_ITEM;
        break;

      case S_NEXT_PACKET:
        j=0;
        if (o->o_next != NULL){
          o = o->o_next;
          state = S_GET_PACKET;
        } else {
          i++;
          state = S_GET_OBJECT;
        }
        break;

      case S_GET_ITEM:
        if (j < p->n_items){
          iptr = SPEAD_ITEM(p->data, (j+1));
          id   = SPEAD_ITEM_ID(iptr);
          mode = SPEAD_ITEM_MODE(iptr);
#ifdef PROCESS
          fprintf(stderr, "@@@ITEM[%d] mode[%d] id[%d] 0x%lx\n", j, mode, id, iptr);
#endif
          //state = S_NEXT_ITEM;
          state = S_MODE;
        } else {
          state = S_NEXT_PACKET;
        }
        
        break;
        
      case S_NEXT_ITEM:
        j++;
        state = S_GET_ITEM;
        break;

      case S_MODE:
        state = S_NEXT_ITEM;

        switch(id){
          case SPEAD_HEAP_CNT_ID:
          case SPEAD_HEAP_LEN_ID:
          case SPEAD_PAYLOAD_OFF_ID:
          case SPEAD_PAYLOAD_LEN_ID:
          case SPEAD_STREAM_CTRL_ID:
            continue; /*pass control back to the beginning of the loop with state S_NEXT_ITEM*/
          case SPEAD_DESCRIPTOR_ID:
#ifdef PROCESS
            fprintf(stderr, "\tITEM_DESCRIPTOR_ID\n");
#endif
            break;
          default:
            break;
        }

#ifdef PROCESS
        fprintf(stderr, "ITEM[%d] mode[ %s ] id[%d] 0x%lx\n", j, ((mode == SPEAD_DIRECTADDR)?"DIRECT   ":"IMMEDIATE"), id, iptr);
#endif
        switch (mode){
          case SPEAD_DIRECTADDR:
            state = S_MODE_DIRECT;
            break;
          case SPEAD_IMMEDIATEADDR:
            state = S_MODE_IMMEDIATE;
            break;
        }
        break;


      case S_MODE_IMMEDIATE:
        /*this macro call results in the data correctly formatted*/
        data64 = (int64_t) SPEAD_ITEM_ADDR(iptr);
#if 0
        for (k=0; k<SPEAD_ADDRLEN; k++){
          val[k] = 0xFF & (iptr >> (8*(SPEAD_ADDRLEN-k-1)));
        }
#endif
#ifdef PROCESS
        fprintf(stderr, "\tdata: 0x%lx | %ld\n", data64, data64);
#endif
        
        itm = new_item_from_group(ig, sizeof(struct spead_api_item) + sizeof(data64));
        if (itm){
          itm->i_id = id;
          itm->i_len = sizeof(data64);
          memcpy(itm->i_data, &data64, sizeof(data64));
          itm->i_valid = 1;
        } else {      
#ifdef DEBUG
          fprintf(stderr, "MALFORMED packet\n");
#endif
          state = S_NEXT_ITEM;
          break;
        }

#if 0
        ps.i   = i;
        ps.j   = j;
        ps.off = off;
        ps.id  = id;
        ps.p   = p;
        ps.o   = o;
#endif
        state = S_NEXT_ITEM;
        break;


      case S_MODE_DIRECT:
        /*TODO: Not fully spead protocol correct*/
        off = (int64_t) SPEAD_ITEM_ADDR(iptr);
#ifdef PROCESS
        fprintf(stderr, "\toffset: 0x%lx\n", off);
#endif
        
        if (ps.off > -1){
          /*need to process the previous direct addressed item*/
#ifdef PROCESS
          fprintf(stderr, "Direct ITEM[%d] in obj[%d] pkt(%p) SIZE [%ld] bytes\n", ps.j, ps.i, ps.p, off - ps.off);
#endif
          itm = new_item_from_group(ig, sizeof(struct spead_api_item) + (off - ps.off));
          if (itm){
            itm->i_id   = ps.id;
            itm->i_len  = (off - ps.off);
            ds.cc       = itm->i_len;
            
            if (ds.off == -1){
              ds.i   = ps.i;
              ds.p   = ps.p;
              ds.o   = ps.o;
              ds.off = ps.off;
            }
            
            state = S_DIRECT_COPY;
          } else {      
#ifdef DEBUG
            fprintf(stderr, "MALFORMED packet\n");
#endif
            state = S_NEXT_ITEM;
            break;
          }


        }

        ps.i   = i;
        ps.j   = j;
        ps.off = off;
        ps.id  = id;
        ps.p   = p;
        ps.o   = o;

        state = (state == S_DIRECT_COPY) ? S_DIRECT_COPY : S_NEXT_ITEM;
        break;

      case DC_NEXT_PACKET:
        
#ifdef PROCESS
        fprintf(stderr, "===dc get next packet===\n");
#endif
DC_NXT_PKT:
        if (ds.o->o_next != NULL){
          ds.o = ds.o->o_next;
          //state = S_GET_PACKET;
          goto DC_GET_PKT;
        } else {
          ds.i++;

DC_GET_OBJ:
          if (ds.i < ht->t_len){
            ds.o = ht->t_os[ds.i];
            if (ds.o == NULL){
              ds.i++;
              //state = S_GET_OBJECT;
              goto DC_GET_OBJ;
            } 
            //state = S_GET_PACKET;
            goto DC_GET_PKT;
          } else {
            
#ifdef PROCESS
            fprintf(stderr, "end of stream!!\n");
#endif

            state = (ps.state == S_END)? ps.state : S_NEXT_ITEM;
            break;
          }
        }

DC_GET_PKT:
        ds.p = get_data_hash_o(ds.o);
        if (ds.p == NULL){
          //state = S_NEXT_PACKET;
          goto DC_NXT_PKT;
        }

#if DEBUG>2
        fprintf(stderr, "\tDC:--pkt-- in ds.o (%p) has ds.p (%p)\n\t   payload_off: %ld payload_len: %ld payload: %p\n", ds.o, ds.p, ds.p->payload_off, ds.p->payload_len, ds.p->payload);
#endif
        state = S_DIRECT_COPY;

        break;

      case S_DIRECT_COPY:

#ifdef PROCESS
        fprintf(stderr, "++++direct copy start++++\n");
        //fprintf(stderr, "\tds off: %ld ds.cc %ld i: %d p @ %p payload_len %ld\n\titm id: %d len: %ld\n", ds.off, ds.cc, ds.i, ds.p, ds.p->payload_len, itm->i_id, itm->i_len);
        fprintf(stderr, "\tcan copy %ld\n\tpayload len %ld\n\tsource %p + %ld\n\tdestination %p + %ld\n\t", ds.cc, ds.p->payload_len, ds.p->payload, ds.off, itm->i_data, (itm->i_len-ds.cc));
#endif

        if (ds.off + ds.cc <= ds.p->payload_len){
#ifdef PROCESS
          fprintf(stderr, "\tDC payload in current packet [%ld] copy [%ld] bytes\n", ds.off + ds.cc, ds.cc);
          //fprintf(stderr, "\tdestination offset [%ld]\n", itm->i_len - ds.cc);
#endif

          memcpy(itm->i_data + (itm->i_len - ds.cc), ds.p->payload + ds.off, ds.cc);

          ds.off += ds.cc;
          ds.cc = 0;

        } else {
#ifdef PROCESS
          fprintf(stderr, "\tDC payload over current packet [%ld] copy [%ld] bytes\n", ds.off + ds.cc, ds.p->payload_len - ds.off);
          //fprintf(stderr, "\tdestination offset [%ld]\n", itm->i_len - ds.cc);
#endif

          memcpy(itm->i_data + (itm->i_len - ds.cc), ds.p->payload + ds.off, ds.p->payload_len - ds.off);
          
          ds.cc  -= ds.p->payload_len - ds.off;
          ds.off = 0;

          state = DC_NEXT_PACKET;

        }

#ifdef PROCESS
        fprintf(stderr, "-----direct copy end-----still need to copy %ld bytes\n", ds.cc);
#endif
        state = (state == DC_NEXT_PACKET) ? state : ((ps.state == S_END) ? ps.state : S_NEXT_ITEM);
        break;

    }

  }

#ifdef PROCESS
  fprintf(stderr, "--PROCESS-[%d]-END-----\n", getpid());
#endif

  if (empty_hash_table(ht) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: error empting hash table", __func__);
#endif
    return -1;
  }

#ifdef DEBUG
  fprintf(stderr, "[%d] %s:\033[32m DONE empting hash table [%ld] \033[0m\n", getpid(), __func__, ht->t_id);
#endif

  /*DUMP each item*/

#if 0
  off = 0;
  uint64_t count;

  while (off < ig->g_size){
  
    itm = (struct spead_api_item *) (ig->g_map + off);

    if (itm->i_len == 0)
      goto skip;

    count = 0;
#ifdef DEBUG
    fprintf(stderr, "ITEM id[%d] vaild [%d] len [%ld]\n\t0x%06x | ", itm->i_id, itm->i_valid, itm->i_len, count);
    
    for (;count<itm->i_len; count++){
      
      fprintf(stderr, "%02X", itm->i_data[count]);
      if ((count+1) % 20 == 0){
        fprintf(stderr,"\n\t0x%06x | ", count);
      } else {
        fprintf(stderr," ");
      }

    }
    fprintf(stderr,"\n");

#endif
skip:
    off += sizeof(struct spead_api_item) + itm->i_len;
    
  }
#endif

  



  destroy_item_group(ig);
   
  return 0;
}

int store_packet_hs(struct u_server *s, struct hash_o *o)
{
#if 0
  struct spead_heap *h;
  int64_t id;
#endif
  struct spead_heap_store   *hs;
  struct spead_packet       *p;
  struct hash_table         *ht;
  int flag_processing;

  if (s == NULL)
    return -1;

  hs = s->s_hs;

  flag_processing = 0;

  if (hs == NULL || o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error invalid data\n", __func__);
#endif 
    return -1;
  }

  p = get_data_hash_o(o);
  if (p == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot get packet from object (%p)\n", __func__, o);
#endif 
    return -1;
  }
  
  ht = get_ht_hs(hs, p->heap_cnt);
  if (ht == NULL){
    /*TODO: we have a newer set from packet must process partial*/
    /*or discard set at current position*/
#ifdef DATA
    fprintf(stderr, "%s: backlog collision\n", __func__);
#endif

    return -1;
  }

  if (add_o_ht(ht, o) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: could not add packet to hash table [%ld]\n", __func__, p->heap_cnt);
#endif 
    return -1;
  }

#if 0
def DEBUG
  fprintf(stderr, "Packet has [%d] items\n", p->n_items);  
#endif
  
#if 0
  lock_sem(ht->t_semid);
#endif
  
  lock_mutex(&(ht->t_m));

  ht->t_data_count += p->payload_len;
  ht->t_items      += p->n_items;

  if (ht->t_data_count == p->heap_len && !ht->t_processing){
    ht->t_processing = 1;
    flag_processing  = 1;
  }

  unlock_mutex(&(ht->t_m));

#if 0  
  unlock_sem(ht->t_semid);
#endif

  /*have all packets by data count must process*/
  if (flag_processing){
#ifdef DEBUG
    fprintf(stderr, "[%d] data_count = %ld bytes == heap_len\n", getpid(), ht->t_data_count);
#endif

    if (process_items(ht, s->s_cdfn) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error processing items in ht (%p)\n", __func__, ht);
#endif
      return -1;
    }
    
    lock_mutex(&(s->s_m));
    s->s_hpcount++;
    unlock_mutex(&(s->s_m));
    
  }
 
  return 0;
}

int process_packet_hs(struct u_server *s, struct hash_o *o)
{
  struct spead_heap_store *hs;
  struct spead_packet *p;
  int rtn;

  if (s == NULL)
    return -1;

  hs = s->s_hs;

  if (hs == NULL || o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error invalid data\n", __func__);
#endif 
    return -1;
  }

  p = get_data_hash_o(o);
  if (p == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot get packet from object (%p)\n", __func__, o);
#endif 
    return -1;
  }
  
  if (spead_packet_unpack_header(p) == SPEAD_ERR){
#ifdef DEBUG
    fprintf(stderr, "%s: unable to unpack spead header for packet (%p)\n", __func__, p);
#endif
    return -1;
  }

#if DEBUG>1
  fprintf(stderr, "%s: unpacked spead header for packet (%p)\n", __func__, p);
#endif

  if (spead_packet_unpack_items(p) == SPEAD_ERR){
#ifdef DEBUG
    fprintf(stderr, "%s: unable to unpack spead items for packet (%p)\n", __func__, p);
#endif
    return -1;
  } 

#if DEBUG>1
  fprintf(stderr, "%s: unpacked spead items for packet (%p) from heap %ld po %ld of %ld\n", __func__, p, p->heap_cnt, p->payload_off, p->heap_len);
#endif

  if (p->is_stream_ctrl_term){
#ifdef DEBUG
    fprintf(stderr, "%s: GOT STREAM TERMINATOR\n", __func__);
#endif
    
    //for (i=0; !ship_heap_hs(hs, i); i++);

    return -1;
  } else {
    rtn = store_packet_hs(s, o);
  }

  return rtn;
}







/***
  This is old naive code
  
***/
#if 0
int store_packet(){
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
}

int add_heap_hs(struct spead_heap_store *hs, struct spead_heap *h)
{
  int64_t id;
  int i;

#if 0
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

#if 0
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
#endif

  return 0;
}

struct spead_heap *get_heap_hs(struct spead_heap_store *hs, int64_t hid)
{
  struct spead_heap *h;
  int id;
  
  h == NULL;
#if 0
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
#endif
  return h;
}
#endif

