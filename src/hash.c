#include <stdlib.h>
#include <stdint.h>
#include <strings.h>

#include "hash.h"


struct hash_table *create_hash_table(struct hash_o_list *l, uint64_t id, uint64_t len, uint64_t (*hfn)(uint64_t in))
{
  struct hash_table *t;
  uint64_t i;

  if (l == NULL || len < 0 || id < 0 || hfn == NULL)
    return NULL;

  t = malloc(sizeof(struct hash_table));
  if (t == NULL)
    return NULL;

  t->t_id  = 0;
  t->t_hfn = hfn;
  t->t_len = len;

  t->t_os = malloc(sizeof(struct hash_o*) * len);
  if (t->t_os = NULL){
    free(t);
    return NULL;
  }

  bzero(t->t_os, len);
  
  for (i=0;i<len; i++){
    t->t_os[i] = pop_hash_o(l);
    if (t->t_os[i] == NULL){
      destroy_hash_table(t);
      return NULL;
    }
  }

  return t;
}

void destroy_hash_table(struct hash_table *t)
{
  uint64_t i;
  if (t){

    if (t->t_os){
      for (i=0; i<t->t_len; i++){
        destroy_hash_o(t->t_os[i]);
      }
    }

    free(t);
  }
}

struct hash_o *create_hash_o(void *(*create)(), void (*destroy)(void *data), uint64_t len)
{
  struct hash_o *o;

  o = malloc(sizeof(struct hash_o));
  if (o == NULL)
    return NULL;

  o->o         = NULL;
  o->o_len     = 0;
  o->o_next    = NULL;
  o->o_create  = NULL;
  o->o_destroy = NULL;

  if (create != NULL && destroy != NULL){
    o->o_create  = create;
    o->o_destroy = destroy;

    o->o = (*create)();

    if (o->o == NULL){
      destroy_hash_o(o);
      return NULL;
    }

    o->o_len = len;

  }
  
  return o;
}

void destroy_hash_o(struct hash_o *o)
{
  if (o){
    if (o->o && o->o_destroy){
      (*o->o_destroy)(o->o);
    }

    free(o);
  }
}

struct hash_o *get_o_ht(struct hash_table *t, uint64_t id)
{
  if (t == NULL || id < 0 || id > t->t_len)
    return NULL;

  return t->t_os[id];
}

int add_o_ht(struct hash_table *t, uint64_t id, void *data, uint64_t len)
{
  struct hash_o *o;

  if (t == NULL || id < 0 || data == NULL || len < 0)
    return -1;

  o = get_o_ht(t, t->t_hfn(id));
  if (o == NULL)
    return -1;

  if (o->o == NULL)
    return -1;

  o->o      = data;
  o->o_len  = len;
  o->o_next = NULL;
  
  return 0;
}


struct hash_o_list *create_o_list(uint64_t len, void *(*create)(), void (*destroy)(void *data), uint64_t size)
{
  struct hash_o_list *l;
  struct hash_o *o;
  uint64_t i;

  l = malloc(sizeof(struct hash_o_list));
  if (l == NULL)
    return NULL;

  l->l_len = len;
  l->l_top = NULL;
  l->l_create = create;
  l->l_destroy = destroy;
  
  for (i=0; i<len; i++){
    o = create_hash_o(create, destroy, size);
    if (o == NULL){
      destroy_o_list(l);
      return NULL;
    }
    o->o_next = l->l_top;
    l->l_top  = o;
  }

  return l;
}

void destroy_o_list(struct hash_o_list *l)
{
  struct hash_o *o;
  if (l){
    o = l->l_top;
    do {
      l->l_top = o->o_next;
      destroy_hash_o(o);
      o = l->l_top;
    } while(o != NULL);

    free(l);
  }
}

struct hash_o *pop_hash_o(struct hash_o_list *l)
{
  struct hash_o *o;
  if (l == NULL)
    return NULL;
  
  o = l->l_top;
  
  if (o == NULL)
    return NULL;

  l->l_top = o->o_next;
  l->l_len--;

  o->o_next = NULL;

  return o;
}

int push_hash_o(struct hash_o_list *l, struct hash_o *o)
{
  if (l == NULL || o == NULL || o->o_next != NULL)
    return -1;

  o->o_next = l->l_top;
  l->l_top = o;
  l->l_len++;

  return 0;
}




