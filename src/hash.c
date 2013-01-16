/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>

#include "spead_api.h"
#include "hash.h"
#include "mutex.h"

void print_list_stats(struct hash_o_list *l, const char *func) 
{
  uint64_t total, obt, len=0;
  
  if (l){
    
    if (l->l_top){
      len = l->l_olen;
    }

    total = (sizeof(struct hash_o) + len) * l->l_len + sizeof(struct hash_o_list);
    obt   = len * l->l_len;

    fprintf(stderr, "%s: object bank (%p)\n\tobjects\t\t%ld\n\tbank size\t%ld bytes\n\to->o size\t%ld bytes\n", func, l, l->l_len, total, obt);
  } else {
    fprintf(stderr, "%s: param error (null list)\n", func);
  } 
}

struct hash_table *create_hash_table(struct hash_o_list *l, uint64_t id, uint64_t len, uint64_t (*hfn)(struct hash_table *t, struct hash_o *o))
{
  struct hash_table *t;

  if (l == NULL || len < 0 || id < 0 || hfn == NULL)
    return NULL;

  t = shared_malloc(sizeof(struct hash_table));
  if (t == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot allocate hash_table from shared mem\n", __func__);
#endif
    return NULL;
  }

  t->t_id         = id;
  t->t_hfn        = hfn;
  t->t_len        = len;
  t->t_os         = NULL;
  t->t_l          = l;
  t->t_data_count = 0;
  t->t_data_id    = (-1);
  t->t_items      = 0;
  t->t_m          = 0; 
  t->t_processing = 0; 

  lock_mutex(&(t->t_m));

  t->t_os  = shared_malloc(sizeof(struct hash_o*) * len);
  if (t->t_os == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot allocate hash_table object store from shared mem\n", __func__);
#endif
    destroy_hash_table(t);
    return NULL;
  }

  memset(t->t_os, 0, len*sizeof(struct hash_o*));
  
#if DEBUG>1
  fprintf(stderr, "HT[%ld/%ld] ", id, len);
#endif

  //print_list_stats(t->t_l, __func__);

  unlock_mutex(&(t->t_m));

  return t;
}

void destroy_hash_table(struct hash_table *t)
{
  uint64_t i;
  if (t){

    if (t->t_os){
      if (t->t_l && t->t_l){

        for (i=0; i<t->t_len; i++){
          if (t->t_os[i] != NULL){
            if (push_hash_o(t->t_l, t->t_os[i]) < 0){
#ifdef DEBUG
              fprintf(stderr, "%s: failed to push o (%p) onto list (%p) from table [%ld]\n", __func__, t->t_os[i], t->t_l, t->t_id);
#endif
              destroy_hash_o(t->t_l, t->t_os[i]);
            }
          }
        }

      }
      //free(t->t_os);
    }

#if DEBUG>1
    print_list_stats(t->t_l, __func__);
#endif

    //free(t);
  }

}

int empty_hash_table(struct hash_table *ht, int flag)
{
  struct hash_o_list *l;
  struct hash_o *o, *on;
  uint64_t i;

  if (ht == NULL || ht->t_os == NULL)
    return -1;

  l = ht->t_l;
  if (l == NULL)
    return -1;

#if 0
def DEBUG
  fprintf(stderr, "%s: about to empty\n", __func__);
#endif
  if (flag)
    lock_mutex(&(ht->t_m));

  for (i=0; i<ht->t_len; i++) {
    
    o = ht->t_os[i];
    if (o == NULL)
      continue;

#if 0
def DEBUG
    fprintf(stderr, "%s: o[%ld]\n", __func__, i);
#endif

    do {
      
      on = o->o_next;

#if 0
def DEBUG
      fprintf(stderr, "%s: o (%p)\n", __func__, o);
#endif
      
      push_hash_o(l, o); 

      o = on;

    } while (o != NULL);

  }

  ht->t_data_count = 0;
  ht->t_data_id    = (-1);
  ht->t_items      = 0;
  ht->t_processing = 0;

  memset(ht->t_os, 0, ht->t_len*sizeof(struct hash_o*));

  if (flag)
    unlock_mutex(&(ht->t_m));

  //print_list_stats(l, __func__);

  return 0;
}

struct hash_o *create_hash_o(void *(*create)(), void (*destroy)(void *data), uint64_t len)
{
  struct hash_o *o;

#if 0
  o = malloc(sizeof(struct hash_o));
#endif
  o = shared_malloc(sizeof(struct hash_o));
  if (o == NULL)
    return NULL;

  o->o         = NULL;
  o->o_next    = NULL;

  if (create != NULL && destroy != NULL){

    o->o = (*create)();

    if (o->o == NULL){
      destroy_hash_o(NULL, o);
      return NULL;
    }

  }
 
  return o;
}

void destroy_hash_o(struct hash_o_list *l, struct hash_o *o)
{
#if 0
  if (o){
    if (o->o && l && l->l_destroy){
      (*l->l_destroy)(o->o);
    }

    free(o);
  }
#endif
}

struct hash_o_list *create_o_list(uint64_t len, uint64_t hlen, uint64_t hsize, void *(*create)(), void (*destroy)(void *data), uint64_t size)
{
  struct hash_o_list *l;
  struct hash_o *o;
  uint64_t i;
  uint64_t req_size;

  req_size = (size + sizeof(struct hash_o)) * len 
           + sizeof(struct hash_o_list) 
           + (sizeof(struct hash_table *) 
           + sizeof(struct hash_table)) * hlen 
           + sizeof(struct hash_o *) * hsize * hlen;

#if DEBUG>1
  fprintf(stderr, "REQ: (hash_o [%ld] + data size [%ld]) * len [%ld] + hash_o_list [%ld] + (hash_table [%ld] + hash_table ptr [%ld])* hlen [%ld] + hash_o ptr [%ld] * hsize [%ld]\n", sizeof(struct hash_o), size, len, sizeof(struct hash_o_list), sizeof(struct hash_table), sizeof(struct hash_table *), hlen, sizeof(struct hash_o *), hsize);
  fprintf(stderr, "%s: calculated size required for all sharedmem [%ld]\n", __func__, req_size);
#endif

  if (create_shared_mem(req_size) < 0) {  /*hashtables objects*/
#ifdef DEBUG
    fprintf(stderr, "%s: could not create shared memory\n", __func__);
#endif
    return NULL;
  }

  l = shared_malloc(sizeof(struct hash_o_list));
  if (l == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: could not allocate hash_o_list from shared memory\n", __func__);
#endif
    destroy_shared_mem();
    return NULL;
  }

  l->l_len     = len;
  l->l_olen    = size;
  l->l_top     = NULL;
  l->l_create  = create;
  l->l_destroy = destroy;
  l->l_m       = 0;

  lock_mutex(&(l->l_m));

  for (i=0; i<len; i++){
    o = create_hash_o(create, destroy, size);
    if (o == NULL){
      destroy_o_list(l);
      return NULL;
    }
    o->o_next = l->l_top;
    l->l_top  = o;
  }

  unlock_mutex(&(l->l_m));

  //print_list_stats(l, __func__);

  return l;
}

void destroy_o_list(struct hash_o_list *l)
{
  /*TODO: is this the final place mem is needed?*/
  destroy_shared_mem();

#if 0
  struct hash_o *o;
  if (l){

    o = l->l_top;
    do {
      l->l_top = o->o_next;
      destroy_hash_o(l, o);
      o = l->l_top;
    } while(o != NULL);

    free(l);
  }
#endif
}

struct hash_o *get_o_ht(struct hash_table *t, uint64_t id)
{
  if (t == NULL || id < 0 || id > t->t_len)
    return NULL;

  return t->t_os[id];
}

void *get_data_hash_o(struct hash_o *o)
{
  if (o == NULL)
    return NULL;

  return o->o;
} 

int add_o_ht(struct hash_table *t, struct hash_o *o)
{
  struct hash_o *to;
  int64_t id;
  int i;

  if (t == NULL || t->t_hfn == NULL || t->t_os == NULL || o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }
  
  id = (*t->t_hfn)(t, o);
  
#if DEBUG>1
  fprintf(stderr, "%s: api hashfn return id [%ld]\n", __func__, id);
#endif

  if (id < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: hfn error!\n", __func__);
#endif
    return -1;
  }
  //lock_mutex(&(t->t_m));

  if (t->t_os[id] == NULL){
    /*simple case*/
    t->t_os[id] = o;

    //unlock_mutex(&(t->t_m));

#ifdef DEBUG
    fprintf(stderr, "[%d] HASH into [%ld] @ [%ld]\t(%p)\n", getpid(), t->t_id, id, o);
#endif

    return 0;
  }
 
  /*list case*/

  to = t->t_os[id];

  for (i=0; to->o_next != NULL; i++){
    to = to->o_next;
  }
  
  if (to == NULL){
    //unlock_mutex(&(t->t_m));
    return -1;
  }

  to->o_next = o;

  //unlock_mutex(&(t->t_m));

#ifdef DEBUG
  fprintf(stderr, "[%d] HASHED into [%ld] @ [%ld] LIST pos [%d]\t(%p)\n", getpid(), t->t_id, id, i, o);
#endif

  return 0;
}

/*Critical section*/
struct hash_o *pop_hash_o(struct hash_o_list *l)
{
  struct hash_o *o;

  if (l == NULL)
    return NULL;

  lock_mutex(&(l->l_m));

  o = l->l_top;
  
  if (o == NULL){
    unlock_mutex(&(l->l_m));
#ifdef DEBUG
    fprintf(stderr, "%s: err no more objects in bank\n", __func__);
#endif
    return NULL;
  }

  l->l_top = o->o_next;
  l->l_len--;

  o->o_next = NULL;

  unlock_mutex(&(l->l_m));

#if DEBUG>1
  fprintf(stderr, "[%d] %s: poped\t\t(%p)\n", getpid(), __func__, o);
  print_list_stats(l, __func__);
#endif

  return o;
}

/*Critical section*/
int push_hash_o(struct hash_o_list *l, struct hash_o *o)
{
  if (l == NULL || o == NULL)
    return -1;

  lock_mutex(&(l->l_m));

  o->o_next = l->l_top;
  l->l_top = o;
  l->l_len++;

  unlock_mutex(&(l->l_m));

#if DEBUG>1
  fprintf(stderr, "[%d] %s: pushed\t\t(%p)\n", getpid(), __func__, o);
  print_list_stats(l, __func__);
#endif

  return 0;
}

int inorder_traverse_hash_table(struct hash_table *ht, int (*call)(void *data, struct spead_packet *p), void *data)
{
  struct hash_o *o;
  struct spead_packet *p;

  int state, i;
  
  if (ht == NULL || call == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }

  i = 0; 
  o = NULL;
  p = NULL;
  state = S_GET_OBJECT;    

  while (state){
    
    switch(state){
      
      case S_GET_OBJECT:
        if (i < ht->t_len){
          o = ht->t_os[i];
          if (o == NULL){
            i++;
            state = S_GET_OBJECT;
            break;
          }
          state = S_GET_PACKET;
        } else 
          state = S_END;
        break;

      case S_GET_PACKET:
        p = get_data_hash_o(o);
        if (p == NULL){
          state = S_NEXT_PACKET;
          break;
        }
        
        if ((*call)(data, p) < 0)
          return -1;

        state = S_NEXT_PACKET;
        break;

      case S_NEXT_PACKET:
        if (o->o_next != NULL){
          o = o->o_next;
          state = S_GET_PACKET;
        } else {
          i++;
          state = S_GET_OBJECT;
        }
        break;
    }
  }

  return 0;
}











#ifdef TEST_HASH
#ifdef DEBUG
void *create_test()
{
  char *obj;
  obj = shared_malloc(sizeof(char)*8);
  if (obj == NULL)
    return NULL;

  return obj;
}

void del_test(void *obj)
{
  if (obj){
    //free(obj);
  }
}

uint64_t hash_fn(struct hash_table *t, uint64_t in)
{
  if (t == NULL || in < 0)
    return -1;
  return in % t->t_len;
}

int main(int argc, char **argv)
{
  struct hash_o_list *b;
  struct hash_table *t[10];
  int i;

  b = create_o_list(1000000, &create_test, &del_test, sizeof(char)*8);
  if (b == NULL){
    fprintf(stderr, "err: create_o_list\n");
    return 1;
  }
 
#if 1
  for (i=0;i<10;i++){
    t[i] = create_hash_table(b, i, 100000, &hash_fn);
    if (t[i] == NULL){
      for (;i>=0;i--){
        destroy_hash_table(t[i]);
      }
      destroy_o_list(b);
      fprintf(stderr, "err: create_hash_table\n");
      return 1;
    }
  }
#endif
 
  sleep(10);
  
  for (i=0;i<10;i++){
    destroy_hash_table(t[i]);
  }

#if 0
  destroy_hash_table(t[1]);
#endif

  destroy_o_list(b);

  return 0;
}
#endif
#endif


