#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <strings.h>

#include "hash.h"

void print_list_stats(struct hash_o_list *l, const char *func) 
{
  uint64_t total, obt, len=0;
  
#ifdef DEBUG
  if (l){
    
    if (l->l_top){
      len = l->l_top->o_len;
    }

    total = (sizeof(struct hash_o) + len) * l->l_len + sizeof(struct hash_o_list);
    obt   = len * l->l_len;

    fprintf(stderr, "%s: object bank (%p)\n\tobjects\t\t%ld\n\tbank size\t%ld bytes\n\to->o size\t%ld bytes\n", func, l, l->l_len, total, obt);
  }
#endif
}

struct hash_table *create_hash_table(struct hash_o_list *l, uint64_t id, uint64_t len, uint64_t (*hfn)(struct hash_table *t, uint64_t in))
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
  t->t_os  = NULL;
  t->t_l   = l;

  t->t_os = malloc(sizeof(struct hash_o*) * len);
  if (t->t_os == NULL){
    free(t);
    return NULL;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: about to start getting objects from bank\n", __func__);
#endif

  bzero(t->t_os, len);
  
  for (i=0; i<len; i++){
    t->t_os[i] = pop_hash_o(l);
    if (t->t_os[i] == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: err pop_hash_o\n", __func__);
#endif
      destroy_hash_table(t);
      return NULL;
    }
  }

#ifdef DEBUG
  fprintf(stderr, "%s: done from bank\n", __func__);
#endif

  print_list_stats(l, __func__);

  return t;
}

void destroy_hash_table(struct hash_table *t)
{
  uint64_t i;
  if (t){

    if (t->t_os){
      if (t->t_l && t->t_l){
        for (i=0; i<t->t_len; i++){
          if (push_hash_o(t->t_l, t->t_os[i]) < 0){
#ifdef DEBUG
            fprintf(stderr, "%s: failed to push o (%p) onto list (%p) from table [%ld]\n", __func__, t->t_os[i], t->t_l, t->t_id);
#endif
            destroy_hash_o(t->t_l, t->t_os[i]);
          }
        }
      }
      free(t->t_os);
    }
    
    print_list_stats(t->t_l, __func__);

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
#if 0
  o->o_create  = NULL;
  o->o_destroy = NULL;
#endif

  if (create != NULL && destroy != NULL){
#if 0
    o->o_create  = create;
    o->o_destroy = destroy;
#endif

    o->o = (*create)();

    if (o->o == NULL){
      destroy_hash_o(NULL, o);
      return NULL;
    }

    o->o_len = len;

  }
 
  return o;
}

void destroy_hash_o(struct hash_o_list *l, struct hash_o *o)
{
  if (o){
    if (o->o && l && l->l_destroy){
      (*l->l_destroy)(o->o);
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

  o = get_o_ht(t, (*t->t_hfn)(t, id));
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

  print_list_stats(l, __func__);

  return l;
}

void destroy_o_list(struct hash_o_list *l)
{
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
}

struct hash_o *pop_hash_o(struct hash_o_list *l)
{
  struct hash_o *o;

  if (l == NULL)
    return NULL;
  
  o = l->l_top;
  
  if (o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: err list object pointer is null\n", __func__);
#endif
    return NULL;
  }

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

#ifdef TEST_HASH
#ifdef DEBUG
void *create_test()
{
  char *obj;
  obj = malloc(sizeof(char));
  if (obj == NULL)
    return NULL;

  return obj;
}

void del_test(void *obj)
{
  if (obj){
    free(obj);
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

  b = create_o_list(100000, &create_test, &del_test, sizeof(char));
  if (b == NULL){
    fprintf(stderr, "err: create_o_list\n");
    return 1;
  }
  
  t[0] = create_hash_table(b, 0, 100, &hash_fn);
  if (t[0] == NULL){
    destroy_o_list(b);
    fprintf(stderr, "err: create_hash_table\n");
    return 1;
  }
  
  t[1] = create_hash_table(b, 1, 101010, &hash_fn);
  if (t[1] == NULL){
    destroy_o_list(b);
    fprintf(stderr, "err: create_hash_table\n");
    return 1;
  }

  

  
  
  destroy_hash_table(t[0]);
  destroy_hash_table(t[1]);


  destroy_o_list(b);

  return 0;
}
#endif
#endif


