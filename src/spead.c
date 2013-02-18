/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <string.h>
#include <unistd.h>
#include <endian.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "hash.h"
#include "spead_api.h"
#include "server.h"
#include "stack.h"
#include "tx.h"

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

  if (t == NULL || o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: null param\n", __func__);
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
  
  po = (uint64_t) p->payload_off;
  hl = (uint64_t) p->heap_len;

  if (hl <= 0){
#ifdef DEBUG
    fprintf(stderr, "%s: packet heap len %ld\n", __func__, hl);
#endif  
    return 0;
  }

  if (t->t_len <= 0){
#ifdef DEBUG
    fprintf(stderr, "%s: table t_len %ld\n", __func__, t->t_len);
#endif  
    return -1;
  }

#if 0
  id = hl / (t->t_len-1);

  if (id <= 0){
#ifdef DEBUG
    fprintf(stderr, "%s: hl[%ld] / t_len[%ld] = %ld\n", __func__, hl, t->t_len, id);
#endif  
    return 0;
  }

  id = po / id;
#endif
  if ((float) ((float)hl / (float)(t->t_len-1)) <= 0)
    return 0;

  /*TODO: FIX THIS*/
  id = (uint64_t)((float)po / (float)((float)hl / ((float)t->t_len-1.0)));

#if DEBUG>1
  fprintf(stderr, "%s: po [%ld] hl [%ld] tlen [%ld] id [%ld]\n", __func__,  po, hl, t->t_len, id);
#endif 
  
  return id;
}

int64_t hash_heap_hs(struct spead_heap_store *hs, int64_t hid)
{
  if (hs == NULL)
    return -1;

  return hid % hs->s_backlog;
}

struct hash_table *get_ht_hs(struct u_server *s, struct spead_heap_store *hs, uint64_t hid)
{
  uint64_t id;
  struct hash_table *ht;

  if (hs == NULL || id < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: parameter error", __func__);
#endif
    return NULL;
  }

  id = hash_heap_hs(hs, hid);
  
  if (hs->s_hash == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: hs->s_hash is null", __func__);
#endif
    return NULL;
  }

  ht = hs->s_hash[id];
  if (ht == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: hs->s_hash[%ld] is null", __func__, id);
#endif
    return NULL;
  }

  lock_mutex(&(ht->t_m));
  if (ht->t_data_id < 0){
    ht->t_data_id = hid;
  } 
  
  if (ht->t_data_id != hid){
#ifdef DISCARD
    fprintf(stderr, "heap_cnt[%ld] maps to[%ld] / however have [%ld] at [%ld]\n", hid, id, ht->t_data_id, id);
    fprintf(stderr, "old heap has datacount [%ld]\n", ht->t_data_count);
#endif
   
    if (empty_hash_table(ht, 0) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error empting hash table", __func__);
#endif
      unlock_mutex(&(ht->t_m));
      return NULL;
    }


    if (s){
      lock_mutex(&(s->s_m));
      s->s_hdcount++;
      unlock_mutex(&(s->s_m));
    }

#if 0
    unlock_mutex(&(ht->t_m));
    return NULL;
#endif
    

  }

  //unlock_mutex(&(ht->t_m));

  return ht;
}

struct spead_heap_store *create_store_hs(uint64_t list_len, uint64_t hash_table_count, uint64_t hash_table_size)
{
  struct spead_heap_store *hs;
  int i;

  hs = malloc(sizeof(struct spead_heap_store));
  if (hs == NULL)
    return NULL;

  hs->s_backlog  = hash_table_count;

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

#ifdef DEBUG
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

#ifdef DEBUG
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

#if 0
    if (hs->s_hash)
      destroy_hash_table(hs->s_hash);
#endif
  
    if (hs->s_list)
      destroy_o_list(hs->s_list);

    free(hs);
  }
}

void print_data(unsigned char *buf, int size)
{
#define COLS 24
#define ROWS 900
  int count, count2;

  count = 0;
  fprintf(stderr, "\t\t   ");
  for (count2=0; count2<COLS; count2++){
    fprintf(stderr, "%02x ", count2+1);
  }
  fprintf(stderr,"\n\t\t   ");
  for (count2=0; count2<COLS; count2++){
    fprintf(stderr, "---");
  }
  fprintf(stderr,"\n\t0x%06x | ", count);
  for (;count<size; count++){

    fprintf(stderr, "%02X", buf[count]);

    if ((count+1) % COLS == 0){

      if ((count+1) % ROWS == 0){
        fprintf(stderr, "\n\n\t\t   ");
        for (count2=0; count2<COLS; count2++){
          fprintf(stderr, "%02x ", count2);
        }
        fprintf(stderr,"\n\t\t   ");
        for (count2=0; count2<COLS; count2++){
          fprintf(stderr, "---");
        }
      }

      fprintf(stderr,"\n\t0x%06x | ", count+1);
    } else {
      fprintf(stderr," ");
    }

  }
  fprintf(stderr,"\n");
#undef COLS
#undef ROWS
}


struct hash_table *packetize_item_group(struct spead_heap_store *hs, struct spead_item_group *ig, int pkt_size, uint64_t hid)
{
#define PZ_END              0 
#define PZ_GETPACKET        1
#define PZ_COPYDATA         2
#define PZ_ADDIG_ITEMS      3
#define PZ_INIT_PACKET      4
#define PZ_HASHPACKET       5

  struct spead_api_item *itm;
  struct hash_table *ht;
  struct hash_o *o;
  struct spead_packet *p;

  int state;
  uint64_t *pktd, payload_off, payload_len, heap_len, nitems, count, ioff, off, remain, didcopy;
  
  if (hs == NULL || ig == NULL || pkt_size <= 0){
#ifdef DEBUG
    fprintf(stderr, "%s: parameter error\n", __func__);
#endif
    return NULL;
  }

  /************************************/
  /*NOTE: mutex is locked for table   */
  /*      after this call             */
  ht = get_ht_hs(NULL, hs, hid);
  if (ht == NULL){
    return NULL;
  }
  /***********************************/

  /*do some cals*/
  p           = NULL;
  o           = NULL;
  payload_off = 0;
  payload_len = pkt_size;
  didcopy     = 0;
  heap_len    = 0;
  nitems      = 4;
  count       = 0;
  off         = 0;
  ioff        = 0;
  remain      = 0;

  itm         = NULL;
  while ((itm = get_next_spead_item(ig, itm))){
    heap_len += itm->i_data_len;
    nitems++;
  }

#if 0 
def DEBUG
  print_data(ig->g_map, ig->g_size);
#endif

#ifdef DEBUG
  fprintf(stderr, "%s: [%ld] igitems [%ld] nitems into ht [%ld] heap_len [%ld] into [%d] byte packets\n", __func__, ig->g_items, nitems, ht->t_id, heap_len, pkt_size);
#endif

  state       = PZ_GETPACKET;

  while (state){
    
    switch (state){
      
      case PZ_GETPACKET:

#ifdef PROCESS
        fprintf(stderr, "%s: GET PACKET\n", __func__);
#endif

        o = pop_hash_o(hs->s_list);
        if (o == NULL){
          state = PZ_END;
          break;
        }

        p = get_data_hash_o(o);
        if (p == NULL){
          state = PZ_END;
          break;
        }

#ifdef PROCESS
        fprintf(stderr, "%s: payload off %ld\n", __func__, payload_off);
#endif
        pktd = (uint64_t *)p->data;

        SPEAD_SET_ITEM(pktd, 0, SPEAD_HEADER_BUILD(nitems));
        SPEAD_SET_ITEM(p->data, 1, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_HEAP_CNT_ID, hid));
        SPEAD_SET_ITEM(p->data, 2, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_HEAP_LEN_ID, heap_len));
        SPEAD_SET_ITEM(p->data, 3, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_PAYLOAD_OFF_ID, payload_off));
#if 0
        SPEAD_SET_ITEM(p->data, 4, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_PAYLOAD_LEN_ID, payload_len));
#endif

        if (nitems > 4 && count == 0){
          state = PZ_ADDIG_ITEMS;
          break;
        }
        
        state = PZ_INIT_PACKET;
        break;

      case PZ_ADDIG_ITEMS:

#ifdef PROCESS
        fprintf(stderr, "%s: ADD Items\n", __func__);
#endif

        nitems = 5;
        itm    = NULL;
        count  = 0;

        while ((itm = get_next_spead_item(ig, itm))){
          /*TODO deal with immediate items also*/
          if (itm){
            SPEAD_SET_ITEM(p->data, nitems++, SPEAD_ITEM_BUILD(SPEAD_DIRECTADDR, itm->i_id, count));
#ifdef PROCESS
            fprintf(stderr, "%s: Item at offset [%ld or 0x%lx]\n", __func__, count, count);
#endif
            count += itm->i_data_len;
          }
        }

        nitems = 4;
        itm    = NULL;

        state = PZ_INIT_PACKET;
        break;


      case PZ_INIT_PACKET:

        if (spead_packet_unpack_header(p) < 0){
#ifdef DEBUG 
          fprintf(stderr, "%s: error unpacking spead header\n", __func__);
#endif
          state = PZ_END;
          break;
        }
        
        if (spead_packet_unpack_items(p) == SPEAD_ERR){
#ifdef DEBUG
          fprintf(stderr, "%s: unable to unpack spead items for packet (%p)\n", __func__, p);
#endif
          state = PZ_END;
          break;
        } 

        if (SPEAD_HEADERLEN + p->n_items * SPEAD_ITEMLEN + payload_len >= SPEAD_MAX_PACKET_LEN) {
#ifdef PROCESS
          fprintf(stderr, "%s: error items and payload will not fit in packet!!!\n", __func__);
#endif
          state = PZ_END;
          break;
        }

        state = PZ_COPYDATA;
        break;


      case PZ_COPYDATA:
        
        /*TODO: think about including the header into the total packet size specified*/
        //copied = 0;
        
#ifdef PROCESS
        fprintf(stderr, "%s: COPY DATA\n", __func__);
#endif

        if (!remain){
#ifdef PROCESS
          fprintf(stderr, "%s: GET NEXT ITEM (%p)\n", __func__, p);
#endif
          itm = get_next_spead_item(ig, itm);
          if (itm == NULL){
            state = PZ_END;
            break;
          }
          remain = itm->i_data_len;
          ioff   = 0;
        } else {
          off = 0;
        }

#ifdef PROCESS
        fprintf(stderr, "%s:\t\tcount %ld off %ld remain %ld didcopy %ld ioff %ld\n", __func__, count, off, remain, didcopy, ioff);
#endif

        if (off + remain < pkt_size) {

          memcpy(p->payload + off, itm->i_data + ioff, remain);

          ioff    += remain;

          didcopy += remain;
          count   -= remain;
          off     += remain;
          remain   = 0;
          
          SPEAD_SET_ITEM(p->data, 4, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_PAYLOAD_LEN_ID, off));

#ifdef PROCESS
          fprintf(stderr, "%s: COPYMORE set packet payload len to: %ld\n", __func__, off);
          fprintf(stderr, "%s:\t\tcount %ld off %ld remain %ld didcopy %ld ioff %ld\n", __func__, count, off, remain, didcopy, ioff);
#endif

          state = (count > 0) ? PZ_COPYDATA : PZ_HASHPACKET;

        } else if (off + remain >= pkt_size){
          
          uint64_t cancopy = (remain < pkt_size - off) ? remain : pkt_size - off;

          memcpy(p->payload + off, itm->i_data + ioff, cancopy);

          ioff    += cancopy;
          
          didcopy += cancopy;
          count   -= cancopy;
          remain   = (remain < pkt_size - off) ? 0 : remain - (pkt_size - off);
          off      = 0;

#ifdef PROCESS
          fprintf(stderr, "%s: NEW PCKT\n", __func__);
          fprintf(stderr, "%s:\t\tcount %ld off %ld remain %ld didcopy %ld ioff %ld\n", __func__, count, off, remain, didcopy, ioff);
#endif

          state = PZ_HASHPACKET;
        }
        
        SPEAD_SET_ITEM(p->data, 4, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_PAYLOAD_LEN_ID, didcopy));
#ifdef PROCESS
        fprintf(stderr, "%s: update payload_len to %ld bytes\n", __func__, didcopy);
        fprintf(stderr, "%s: end copy data------\n", __func__);
#endif
        break;
    
      case PZ_HASHPACKET:
        state = PZ_GETPACKET;
        /*****************extra*********************/
        if (spead_packet_unpack_header(p) < 0){
#ifdef DEBUG 
          fprintf(stderr, "%s: error unpacking spead header\n", __func__);
#endif
          state = PZ_END;
          break;
        }
        if (spead_packet_unpack_items(p) == SPEAD_ERR){
#ifdef DEBUG
          fprintf(stderr, "%s: unable to unpack spead items for packet (%p)\n", __func__, p);
#endif
          state = PZ_END;
          break;
        } 
        /*******************************************/

        if (add_o_ht(ht, o) < 0){
#ifdef DEBUG 
          fprintf(stderr, "%s: add o ht error\n", __func__);
#endif
          state = PZ_END;
          break;
        }

        if (count == 0 && remain == 0){
#ifdef DEBUG
          fprintf(stderr, "%s: count and remain 0 END\n", __func__);
#endif
          state = PZ_END;
          break;
        }
        
        payload_off += pkt_size;
        didcopy = 0; 

#if 0 
def PROCESS
        print_data(p->payload, p->payload_len);
#endif  

        break;

      case PZ_END:
      default:
#ifdef PROCESS
        fprintf(stderr, "%s: packetize end\n", __func__);
#endif
        break;

    }
  }

  /***********************/
  /* NOTE ht is returned */
  /* with mutex set      */
  /***********************/
  return ht;
}


int send_spead_stream_terminator(struct spead_tx *tx)
{
  struct spead_packet pkt, *p;
  uint64_t *pktd;
  
  if (tx == NULL || tx->t_x == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }

  p = &pkt;

  bzero(p, sizeof(struct spead_packet));
  spead_packet_init(p);

  p->n_items = 3;
  p->is_stream_ctrl_term = SPEAD_STREAM_CTRL_TERM_VAL;
  
  pktd = (uint64_t *)p->data;
  SPEAD_SET_ITEM(pktd, 0, SPEAD_HEADER_BUILD(p->n_items));
  
  SPEAD_SET_ITEM(p->data, 1, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_STREAM_CTRL_ID, SPEAD_STREAM_CTRL_TERM_VAL));
  SPEAD_SET_ITEM(p->data, 2, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_HEAP_LEN_ID, 0x0));
  //SPEAD_SET_ITEM(p->data, 2, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_PAYLOAD_OFF_ID, 0x0));
  //SPEAD_SET_ITEM(p->data, 3, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_PAYLOAD_LEN_ID, 0x0));
  SPEAD_SET_ITEM(p->data, 3, SPEAD_ITEM_BUILD(SPEAD_IMMEDIATEADDR, SPEAD_HEAP_CNT_ID, 0xFFFFFFFF));

#if 1
  if (spead_packet_unpack_header(p) < 0){
#ifdef PROCESS
    fprintf(stderr, "%s: error unpacking spead header\n", __func__);
#endif
  }
  if (spead_packet_unpack_items(p) == SPEAD_ERR){
#ifdef PROCESS
    fprintf(stderr, "%s: unable to unpack spead items for packet (%p)\n", __func__, p);
#endif
  } 
#endif
        
  if (send_packet_spead_socket(tx, p) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: could not send packet to spead_socket\n", __func__);
#endif
    return -1;
  }

  return 0;
}



void process_descriptor_item(struct spead_api_item *itm)
{
  int state;
  uint64_t hdr;

  uint64_t item;
  int i, j;

  int id, mode;
  int64_t iptr, off, lastoff, data64;

  struct dpkt {
    int64_t p_hc;
    int64_t p_hl;
    int p_nitms;
    int p_sctrl;
    int64_t p_plen;
    int64_t p_poff;
    unsigned char *data;
    unsigned char *payload;
  } p;

  if (itm == NULL)
    return;

  p.data = itm->i_data;

  state = S_GET_ITEM;

  hdr = (uint64_t) SPEAD_HEADER(p.data);
  if ((SPEAD_GET_MAGIC(hdr) != SPEAD_MAGIC) || 
      (SPEAD_GET_VERSION(hdr) != SPEAD_VERSION) ||
      (SPEAD_GET_ITEMSIZE(hdr) != SPEAD_ITEM_PTR_WIDTH) || 
      (SPEAD_GET_ADDRSIZE(hdr) != SPEAD_HEAP_ADDR_WIDTH)) {
    return;
  }
  p.p_nitms = SPEAD_GET_NITEMS(hdr);
  p.payload = p.data + SPEAD_HEADERLEN + p.p_nitms * SPEAD_ITEMLEN;

  // Read each raw item, starting at 1 to skip header
  for (i=1; i <= p.p_nitms; i++) {
    item = SPEAD_ITEM(p.data, i);
    //printf("   %02x%02x%02x%02x%02x%02x%02x%02x\n", ((char *)&item)[0], ((char *)&item)[1], ((char *)&item)[2], ((char *)&item)[3], ((char *)&item)[4], ((char *)&item)[5], ((char *)&item)[6], ((char *)&item)[7]);
    //printf("item%d: mode=%lld, id=%lld, val=%lld\n", i, SPEAD_ITEM_MODE(item), SPEAD_ITEM_ID(item), SPEAD_ITEM_ADDR(item));
    switch (SPEAD_ITEM_ID(item)) {
      case SPEAD_HEAP_CNT_ID:    p.p_hc    = (int64_t) SPEAD_ITEM_ADDR(item); break;
      case SPEAD_HEAP_LEN_ID:    p.p_hl    = (int64_t) SPEAD_ITEM_ADDR(item); break;
      case SPEAD_PAYLOAD_OFF_ID: p.p_poff  = (int64_t) SPEAD_ITEM_ADDR(item); break;
      case SPEAD_PAYLOAD_LEN_ID: p.p_plen  = (int64_t) SPEAD_ITEM_ADDR(item); break;
      case SPEAD_STREAM_CTRL_ID: if (SPEAD_ITEM_ADDR(item) == SPEAD_STREAM_CTRL_TERM_VAL) p.p_sctrl = 1; break;
      default: break;
    }
  }

  i = 0;
  j = 0;
  id = 0;
  mode = 0;
  iptr = 0;
  lastoff = (-1);
  off = (-1);

  while(state){

    switch(state) {
      case S_GET_ITEM:
        if (j < p.p_nitms){
          iptr = SPEAD_ITEM(p.data, (j+1));
          id   = SPEAD_ITEM_ID(iptr);
          mode = SPEAD_ITEM_MODE(iptr);
#ifdef PROCESS 
          fprintf(stderr, "@@@ITEM[%d] mode[%d] id[%d or 0x%x] 0x%lx\n", j, mode, id, id, iptr);
#endif
          //state = S_NEXT_ITEM;
          state = S_MODE;
        } else {
          state = S_END;

          if (mode == SPEAD_DIRECTADDR){
#ifdef PROCESS
            fprintf(stderr, "--ITEM[%d] mode[ %s ] id[%d or 0x%x] 0x%lx\n", j, ((mode == SPEAD_DIRECTADDR)?"DIRECT   ":"IMMEDIATE"), id, id, iptr);
#endif
#ifdef PROCESS
            fprintf(stderr, "\tstart final direct copy: len: %ld\n", off - lastoff);
            print_data((unsigned char *)(p.payload+off), off-lastoff);
#endif
          }

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
#ifdef PROCESS
            fprintf(stderr, "ITEM[%d] mode[ %s ] id[%d or 0x%x] 0x%lx\n", j, ((mode == SPEAD_DIRECTADDR)?"DIRECT   ":"IMMEDIATE"), id, id, iptr);
#endif
            continue; /*pass control back to the beginning of the loop with state S_NEXT_ITEM*/
          case SPEAD_DESCRIPTOR_ID:
#ifdef PROCESS
            fprintf(stderr, "\tITEM_DESCRIPTOR_ID\n");
#endif
            break;
          default:
            break;
        }
        switch(id){
          case D_NAME_ID:
#ifdef PROCESS
            fprintf(stderr, "NAME ID\n");
#endif
            break;
          case D_DESC_ID:
#ifdef PROCESS
            fprintf(stderr, "DESCRIPTION ID\n");
#endif
            break;
          case D_SHAPE_ID:
#ifdef PROCESS
            fprintf(stderr, "SHAPE ID\n");
#endif
            break;
          case D_FORMAT_ID:
#ifdef PROCESS
            fprintf(stderr, "FORMAT ID\n");
#endif
            break;
          case D_ID_ID:
#ifdef PROCESS
            fprintf(stderr, "ID ID\n");
#endif
            break;
          case D_TYPE_ID:
#ifdef PROCESS
            fprintf(stderr, "TYPE ID\n");
#endif
            break;
        }

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
#ifdef PROCESS
        fprintf(stderr, "==ITEM[%d] mode[ %s ] id[%d or 0x%x] 0x%lx\n", j, ((mode == SPEAD_DIRECTADDR)?"DIRECT   ":"IMMEDIATE"), id, id, iptr);
#endif
        data64 = (int64_t) SPEAD_ITEM_ADDR(iptr);
#ifdef PROCESS
        fprintf(stderr, "\tdata: 0x%lx | %ld\n", data64, data64);
#endif
        state = S_NEXT_ITEM;
        break;

      case S_MODE_DIRECT:
        lastoff = off;
        off = (int64_t) SPEAD_ITEM_ADDR(iptr);
#ifdef PROCESS
        fprintf(stderr, "\toffset: 0x%lx\n\tlastoff: 0x%lx\n", off, lastoff);
#endif 
        state = (lastoff > -1) ? S_DIRECT_COPY : S_NEXT_ITEM;
        break;

      case S_DIRECT_COPY:
#ifdef DEBUG
        fprintf(stderr, "++ITEM[%d] mode[ %s ] id[%d or 0x%x] 0x%lx\n", j, ((mode == SPEAD_DIRECTADDR)?"DIRECT   ":"IMMEDIATE"), id, id, iptr);
#endif
            
#ifdef PROCESS
        fprintf(stderr, "\tstart direct copy: len: %ld\n", off - lastoff);
        print_data((unsigned char *)(p.payload+off), off-lastoff);
#endif

        state = S_NEXT_ITEM;
        break;

    }

  }

#ifdef DEBUG
  fprintf(stderr, "%s: unpacked spead items (%d) from heap %ld po %ld of %ld\n", __func__, p.p_nitms, p.p_hc, p.p_poff, p.p_hl);
#endif

}

int coalesce_spead_items(void *data, struct spead_packet *p)
{
  struct spead_api_item2 *itm;
  struct coalesce_spead_data *cd;

  int64_t iptr;
  uint64_t id;

  int j, mode;

  if (data == NULL || p == NULL)
    return -1;

  cd = data;
  
#ifdef PROCESS
  fprintf(stderr, "%s: [GET PACKET] --pkt-- [%d] items\n\tpayload_off: %ld\n\tpayload_len: %ld\n", __func__, p->n_items, p->payload_off, p->payload_len);
#endif
  
  for (j=0; j<p->n_items; j++){
    iptr = SPEAD_ITEM(p->data, (j+1));
    id   = SPEAD_ITEM_ID(iptr);
    mode = SPEAD_ITEM_MODE(iptr);
    
    switch(id){
      case SPEAD_HEAP_CNT_ID:
      case SPEAD_HEAP_LEN_ID:
      case SPEAD_PAYLOAD_OFF_ID:
      case SPEAD_PAYLOAD_LEN_ID:
      case SPEAD_STREAM_CTRL_ID:
        continue; /*pass control back to the beginning of the loop with state S_NEXT_ITEM*/
      case SPEAD_DESCRIPTOR_ID:
#ifdef PROCESS
        fprintf(stderr, "%s: ITEM_DESCRIPTOR_ID\n", __func__);
#endif
        break;
      default:
        break;
    }
    
#ifdef PROCESS
    fprintf(stderr, "%s: [GET ITEM] @@@ITEM[%d] mode[%d] id[%ld] 0x%lx\n", __func__, j, mode, id, iptr);
#endif
    
    /*TODO: remove this malloc use shared malloc*/
    itm = malloc(sizeof(struct spead_api_item2));
    if (itm == NULL)
      return -1;
  
    itm->i_id   = id;
    itm->i_mode = mode;
    itm->i_off  = (int64_t) SPEAD_ITEM_ADDR(iptr);
    itm->i_len  = 0;

    if (push_stack(cd->d_stack, itm) < 0){
      if (itm)
        free(itm);
      return -1;
    }

  }

#if 0
  /*TODO: look into*/
  if (memcpy(cd->d_data + cd->d_off, p->payload, p->payload_len) == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: memcpy fail\n", __func__);
#endif
    return -1;
  }
#endif

  cd->d_off += p->payload_len;
  
  return 0;
}

void destroy_spead_item2(void *data)
{
  struct spead_api_item2 *itm;
  itm = data;
  if (itm){
    free(itm);
  }
}

void print_spead_item(void *so, void *data)
{
  struct spead_api_item2 *itm;
  itm = so;
  if (itm){
#ifdef DEBUG
    fprintf(stderr, "%s: item [mode %d id %d off %ld len %ld]\n", __func__, itm->i_mode, itm->i_id, itm->i_off, itm->i_len);
#endif
  }
}

int calculate_lengths(void *so, void *data)
{
  struct coalesce_spead_data *cd;
  struct spead_api_item2     *itm;

  if (so == NULL || data == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: null params\n", __func__);
#endif
    return -1;
  }

  cd = data;
  itm = so;
  
  if (itm->i_mode == SPEAD_DIRECTADDR){
    itm->i_len = cd->d_off - itm->i_off;
    cd->d_off -= itm->i_len;
#ifdef PROCESS 
    fprintf(stderr, "%s: DIRECT item [%d] length [%ld]\n", __func__, itm->i_id, itm->i_len);
#endif
  } else {
    itm->i_len = sizeof(int64_t);
    cd->d_imm++;
#ifdef PROCESS
    fprintf(stderr, "%s: IMMEDIATE item [%d]\n", __func__, itm->i_id);
#endif
  }

  return 0;
}

int copy_direct_spead_item(void *data, struct spead_packet *p)
{
  struct coalesce_parcel *cp;
  struct coalesce_spead_data *cd;
  struct hash_table *ht;
  struct spead_api_item *itm;
  
  uint64_t cc;

  cp = data;

  if (cp == NULL || p == NULL)
    return -1;
  
  cd = cp->p_c;
  ht = cp->p_ht;
  itm = cp->p_i;
  
  if (cd == NULL || ht == NULL || itm == NULL)
    return -1;

  cc = p->payload_len - cd->d_off;

#ifdef PROCESS
  fprintf(stderr, "%s: CAN COPY [%ld]\n", __func__, cc);
#endif

  if (itm->i_len < cc) {

    if (append_copy_to_spead_item(itm, p->payload + cd->d_off, itm->i_len) < 0)
      return -1;
    
    cd->d_off += itm->i_len;

#ifdef PROCESS
    fprintf(stderr, "%s: copied [%ld] say in same packet start with off %ld\n", __func__, cc, cd->d_off);
#endif

    return 0;
  } else if (itm->i_len >= cc){
    
    if (append_copy_to_spead_item(itm, p->payload + cd->d_off, cc) < 0)
      return -1;

    cd->d_off = 0;

#ifdef PROCESS
    fprintf(stderr, "%s: copied [%ld] advance to next packet start with off %ld\n", __func__, cc, cd->d_off);
#endif

    return 1;
  }


#if 0
  /*TODO: look into*/
  if (memcpy(cd->d_data + cd->d_off, p->payload, p->payload_len) == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: memcpy fail\n", __func__);
#endif
    return -1;
  }
#endif

  return 0;
} 

int convert_to_ig(void *so, void *data)
{
  struct hash_table *ht;
  struct coalesce_parcel *cp;
  struct coalesce_spead_data *cd;
  struct spead_api_item2 *i2;
  struct spead_api_item  *itm;

  if (so == NULL || data == NULL)
    return -1;

  i2 = so;
  cp = data;

  cd = cp->p_c;
  ht = cp->p_ht;
  
  if (cd == NULL || ht == NULL)
    return -1;

  itm = new_item_from_group(cd->d_ig, i2->i_len);
  if (itm == NULL)
    return -1;

  itm->i_id = i2->i_id;
  itm->i_valid = i2->i_mode;

  cp->p_i = itm;
  
  if (i2->i_mode == SPEAD_DIRECTADDR){
#ifdef PROCESS
    fprintf(stderr, "%s: about to copy off %ld len %ld\n", __func__, i2->i_off, i2->i_len);
    //print_data(cd->d_data + i2->i_off, i2->i_len);
#endif
    
    while (single_traverse_hash_table(ht, &copy_direct_spead_item, cp) > 0){}

#if 0
    if (copy_to_spead_item(itm, cd->d_data + i2->i_off, i2->i_len) < 0){
      destroy_spead_item2(i2);
      return -1;
    }
#endif
  } else {
#ifdef PROCESS
    fprintf(stderr, "%s: DIRECT item [%d] length [%ld]\n", __func__, itm->i_id, itm->i_len);
#endif
#if 1
    if (copy_to_spead_item(itm, &(i2->i_off), sizeof(int64_t)) < 0){
      destroy_spead_item2(i2);
      return -1;
    }
#endif
  }

  destroy_spead_item2(i2);

  return 0;
}

struct spead_item_group *process_items(struct hash_table *ht)
{
  struct spead_item_group *ig;
  struct stack *temp;
  struct coalesce_spead_data cd;
  struct coalesce_parcel cp;

  if (ht == NULL || ht->t_os == NULL)
    return NULL;

  ig = NULL;

  //return NULL;
  
#ifdef PROCESS 
  fprintf(stderr, "--PROCESS-[%d]-BEGIN---\n",getpid());
#endif

#if 1

  cd.d_imm = 0;

#if 1
  cd.d_stack = create_stack();
  if (cd.d_stack == NULL){
    return NULL;
  }

  temp = create_stack();
  if (temp == NULL){
    destroy_stack(cd.d_stack, &destroy_spead_item2);
    return NULL;
  }
#endif

  cd.d_len = ht->t_data_count;
  cd.d_off = 0;

  if (inorder_traverse_hash_table(ht, &coalesce_spead_items, &cd) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: coalesce_spead_items FAILED\n", __func__);
#endif
    destroy_stack(cd.d_stack, &destroy_spead_item2);
    destroy_stack(temp, &destroy_spead_item2);
#if 0
    if (cd.d_data)
      free(cd.d_data);
#endif
    return NULL;
  }
  
  if (funnel_stack(cd.d_stack, temp, &calculate_lengths, &cd) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: calculate lengths FAILED\n", __func__);
#endif
    destroy_stack(cd.d_stack, &destroy_spead_item2);
    destroy_stack(temp, &destroy_spead_item2);
#if 0
    if (cd.d_data)
      free(cd.d_data);
#endif
    return NULL;
  }
 
  destroy_stack(cd.d_stack, &destroy_spead_item2);

  cd.d_stack = temp;
  
#if 0 
  def DEBUG
  traverse_stack(cd.d_stack, &print_spead_item, NULL);
  print_data(cd.d_data, cd.d_len);
#endif

  /*TODO: shared mem managed*/
  ig = create_item_group(cd.d_len + cd.d_imm * sizeof(int64_t), get_size_stack(cd.d_stack));
  if (ig == NULL){
    destroy_stack(cd.d_stack, &destroy_spead_item2);
#if 0
    if (cd.d_data)
      free(cd.d_data);
#endif
    return NULL;
  }

  cd.d_ig = ig;
  cd.d_off = 0;

  cp.p_c = &cd;
  cp.p_ht = ht;
  cp.p_i = NULL;

  if (funnel_stack(cd.d_stack, NULL, &convert_to_ig, &cp) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: convert to item group FAILED\n", __func__);
#endif
    destroy_item_group(ig);
    destroy_stack(cd.d_stack, &destroy_spead_item2);
#if 0
    if (cd.d_data)
      free(cd.d_data);
#endif
    return NULL;
  }


  destroy_stack(cd.d_stack, &destroy_spead_item2);
#if 0
  if (cd.d_data)
    free(cd.d_data);
#endif

#endif

#if 0
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


  if (ht->t_data_id < 0){
#ifdef DATA
    fprintf(stderr, "table has been emptied !!!!!\n");
#endif
    return NULL;
  }

#ifdef DEBUG
  fprintf(stderr, "HEAP CNT [%ld] HEAP ITEMS [%ld] HEAP DATA [%ld bytes]\n", ht->t_data_id, ht->t_items, ht->t_data_count); 
#endif

  ig = create_item_group(ht->t_data_count, ht->t_items);
  if (ig == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: failed to create item group\n", __func__);
#endif
    return NULL;
  }


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

          if (ps.p == NULL){
            state = S_END;
            break;
          }
#ifdef PROCESS
          fprintf(stderr, "[GET OBJECT] Last Direct ITEM[%d] in pkt(%p) SIZE [%ld] bytes\n", ps.j, ps.p, p->heap_len - ps.off);
#endif
          ps.state = S_END;

          ds.cc = p->heap_len - ps.off; 
                    
          itm = new_item_from_group(ig, ds.cc);
          if (itm){
            itm->i_id   = ps.id;
            itm->i_len  = ds.cc;

            if (ds.off == -1){
              ds.i   = i;
              ds.p   = p;
              ds.o   = o;
              ds.off = ps.off;
            }
          } else {
            fprintf(stderr, "\033[31mMALFORMED packet\033[0m\n");
            state = S_END;
            break;
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
#ifdef PROCESS
        //fprintf(stderr, "--pkt-- in o (%p) has p (%p)\n  payload_off: %ld payload_len: %ld payload: %p\n", o, p, p->payload_off, p->payload_len, p->payload);
        fprintf(stderr, "[GET PACKET] --pkt--\n\tpayload_off: %ld\n\tpayload_len: %ld\n", p->payload_off, p->payload_len);
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
          fprintf(stderr, "[GET ITEM] @@@ITEM[%d] mode[%d] id[%d] 0x%lx\n", j, mode, id, iptr);
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

        /*TODO: watchout for data IDs inside the reserved id range! DANGEROUS*/
        
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
        fprintf(stderr, "[MODE] ITEM[%d] mode[ %s ] id[%d] 0x%lx\n", j, ((mode == SPEAD_DIRECTADDR)?"DIRECT   ":"IMMEDIATE"), id, iptr);
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
        
        itm = new_item_from_group(ig, sizeof(data64));
        if (itm){
          itm->i_id = id;
          itm->i_len = sizeof(int64_t);
          memcpy(itm->i_data, &data64, sizeof(int64_t));
          itm->i_valid = 1;
        } else {      
          fprintf(stderr, "\033[31mMALFORMED packet\033[0m\n");
          state = S_NEXT_PACKET;
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
          itm = new_item_from_group(ig, (off - ps.off));
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
            fprintf(stderr, "\033[31mMALFORMED packet\033[0m\n");
            state = S_END;
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
        if (ds.o != NULL && ds.o->o_next != NULL){
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
#ifdef PROCESS
          fprintf(stderr, "go to next packet\n");
#endif  
          goto DC_NXT_PKT;
        }

#ifdef PROCESS
        fprintf(stderr, "\tDC:--pkt-- in ds.o (%p) has ds.p (%p)\n\t   payload_off: %ld payload_len: %ld payload: %p\n", ds.o, ds.p, ds.p->payload_off, ds.p->payload_len, ds.p->payload);
#endif
        state = S_DIRECT_COPY;

        break;

      case S_DIRECT_COPY:

#ifdef PROCESS
        fprintf(stderr, "++++direct copy start++++\n");
        //fprintf(stderr, "\tds off: %ld ds.cc %ld i: %d p @ %p payload_len %ld\n\titm id: %d len: %ld\n", ds.off, ds.cc, ds.i, ds.p, ds.p->payload_len, itm->i_id, itm->i_len);
        fprintf(stderr, "\tcan copy %ld\n\tpayload len %ld\n\tsource %p + %ld\n\tdestination %p + %ld\n", ds.cc, ds.p->payload_len, ds.p->payload, ds.off, itm->i_data, (itm->i_len-ds.cc));
#endif

        if (ds.off + ds.cc <= ds.p->payload_len){
#ifdef PROCESS
          fprintf(stderr, "\tDC payload in current packet [%ld] copy [%ld] bytes\n", ds.off + ds.cc, ds.cc);
          //fprintf(stderr, "\tdestination offset [%ld]\n", itm->i_len - ds.cc);
#endif
  
          /*TODO: more checks*/
          if (ds.cc < 0){ 
            fprintf(stderr, "\033[31mDATA STATE ERROR ds.cc %ld\033[0m\n", ds.cc);
            state = S_END;
            break;
          }

          memcpy(itm->i_data + (itm->i_len - ds.cc), ds.p->payload + ds.off, ds.cc);

          ds.off += ds.cc;
          ds.cc = 0;

          itm->i_valid = 1;

          if (itm->i_id == SPEAD_DESCRIPTOR_ID){
#ifdef DEBUG
            fprintf(stderr, "PROCESS DESCRIPTOR\n");
#endif
            process_descriptor_item(itm);
          }

        } else {
#ifdef PROCESS
          fprintf(stderr, "\tDC payload over current packet [%ld] copy [%ld] bytes\n", ds.off + ds.cc, ds.p->payload_len - ds.off);
          //fprintf(stderr, "\tdestination offset [%ld]\n", itm->i_len - ds.cc);
#endif
          if (ds.p->payload_len - ds.off < 0){
            fprintf(stderr, "\033[31mDATA STATE ERROR payload_len - ds.off %ld\033[0m\n", ds.p->payload_len - ds.off);
            state = S_END;
            break;
          }

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
#endif

#ifdef PROCESS
  fprintf(stderr, "--PROCESS-[%d]-END-----\n", getpid());
#endif

  return ig;
}

void print_store_stats(struct spead_heap_store *hs)
{
  if (hs == NULL)
    return;
}

int store_packet_hs(struct u_server *s, struct spead_api_module *m, struct hash_o *o)
{
  struct spead_heap_store   *hs;
  struct spead_packet       *p;
  struct hash_table         *ht;
  int flag_processing;
#if 0
  int (*cdfn)(struct spead_item_group *ig, void *data);
  void *data;
#endif
  struct spead_item_group *ig;
  
#if 0
  cdfn = NULL;
  data = NULL;
#endif
  ig   = NULL;

  if (s == NULL)
    return -1;
#if 0
  if (m){
    cdfn = m->m_cdfn;
    data = m->m_data;
  }
#endif

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
  
  ht = get_ht_hs(s, hs, p->heap_cnt);
  /*NOTE: if we return null the mutex is not set*/
  if (ht == NULL){
    /*we have a newer set from packet must process partial*/
    /*or discard set at current position*/
    /*will never get here since if there is a hash clash*/
    /*we empty hash table and then return fresh*/
#if 0 
def DATA
    fprintf(stderr, "new heap has size [%ld]\n", p->heap_len);
    fprintf(stderr, "%s: backlog collision\n", __func__);
    //print_store_stats(hs);
#endif
    return -1;
  }

  /*NOTE: if we are here the mutex is set*/

  if (add_o_ht(ht, o) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: could not add packet to hash table [%ld]\n", __func__, p->heap_cnt);
#endif 
    unlock_mutex(&(ht->t_m));
    return -1;
  }

#if 0
def DEBUG
  fprintf(stderr, "Packet has [%d] items\n", p->n_items);  
#endif
  
  //lock_mutex(&(ht->t_m));

  ht->t_data_count += p->payload_len;
  ht->t_items      += p->n_items;

  if (ht->t_data_count == p->heap_len && !ht->t_processing){
    ht->t_processing = 1;
    flag_processing  = 1;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: HID [%ld] HCNT [%ld] data count: [%ld] packet heap len [%ld] total items in ht [%ld]\n", __func__, ht->t_id, p->heap_cnt, ht->t_data_count, p->heap_len, ht->t_items);
#endif

  /*have all packets by data count must process*/
  if (flag_processing){
#ifdef DEBUG
    fprintf(stderr, "FLAG ROCESSING [%d] data_count = %ld bytes == heap_len total items [%ld]\n", getpid(), ht->t_data_count, ht->t_items);
#endif

    
    ig = process_items(ht);


    if (empty_hash_table(ht, 0) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error empting hash table", __func__);
#endif
      unlock_mutex(&(ht->t_m));
      destroy_item_group(ig);
      return -2;
    }

#ifdef DEBUG
    fprintf(stderr, "[%d] %s:\033[32m DONE empting hash table [%ld] \033[0m\n", getpid(), __func__, ht->t_id);
#endif

    if (ig == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: error processing items in ht (%p)\n", __func__, ht);
#endif
      unlock_mutex(&(ht->t_m));
      lock_mutex(&(s->s_m));
      s->s_hdcount++;
      unlock_mutex(&(s->s_m));
      return -2;
    }

    unlock_mutex(&(ht->t_m));


    lock_mutex(&(s->s_m));
    s->s_hpcount++;
    unlock_mutex(&(s->s_m));


    /*SPEAD_API_MODULE CALLBACK*/
#if 0
    if (cdfn != NULL){
      if((*cdfn)(ig, data) < 0){
#ifdef DEBUG 
        fprintf(stderr, "%s: user callback failed\n", __func__);
#endif
      }
    }
#endif

    if(run_api_user_callback_module(m, ig) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: user callback failed\n", __func__);
#endif
    }

    destroy_item_group(ig);
    
  }
  else {
    unlock_mutex(&(ht->t_m));
  }

#if 0
def DEBUG
  fprintf(stderr, "%s: complete\n", __func__);
#endif

  return 0;
}

int process_packet_hs(struct u_server *s, struct spead_api_module *m, struct hash_o *o)
{
  struct spead_heap_store *hs;
  struct spead_packet *p;
  int rtn;
#ifdef DEBUG
  int i, id, mode;
  uint64_t iptr;
#endif

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

#if 0
def DEBUG
  for (i=0; i<p->n_items; i++){
    iptr = SPEAD_ITEM(p->data, (i+1));
    id   = SPEAD_ITEM_ID(iptr);
    mode = SPEAD_ITEM_MODE(iptr);
    fprintf(stderr, "%s: ITEM[%d] mode[%d] id[%d or 0x%x] 0x%lx\n", __func__, i, mode, id, id, iptr);
  }
  //print_data(p->payload, p->payload_len);
#endif

  if (p->is_stream_ctrl_term){
#ifdef DEBUG
    fprintf(stderr, "%s: GOT STREAM TERMINATOR\n", __func__);
#endif

    return store_packet_hs(s, m, o);
  } else {
    rtn = store_packet_hs(s, m, o);
  }

  return rtn;
}



