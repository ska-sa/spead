/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "stack.h"
#include "spead_api.h"

struct stack *create_stack()
{
  struct stack *s;
  
  s = shared_malloc(sizeof(struct stack));
  if (s == NULL)
    return NULL;

  s->s_data = NULL;
  s->s_size = 0;

  return s;
}

void empty_stack(struct stack *s, void (*call)(void *data))
{
  void *d;
  if (s){
    while (pop_stack(s, &d) == 0) {
      if (call){
        (*call)(d);
      }
    }
  }
}

void destroy_stack(struct stack *s, void (*call)(void *data))
{
  if (s){
    empty_stack(s, call);
    shared_free(s, sizeof(struct stack));
  }
}

int push_stack(struct stack *s, void *o)
{
  struct stack_o *so;

  if (s == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error param\n", __func__);
#endif
    return -1;
  }
  
  so = shared_malloc(sizeof(struct stack_o));
  if (so == NULL)
    return -1;
  
  so->data = o;
  so->o_next = s->s_data;
  
  s->s_data = so;

  s->s_size++;

  return 0;
}

int pop_stack(struct stack *s, void **o)
{
  struct stack_o *so;

  if (s == NULL || o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error param\n", __func__);
#endif
    return -1;
  }
  
  if (s->s_size <= 0 || s->s_data == NULL){
    return -1;
  }

  so = s->s_data;

  s->s_data = so->o_next;
  s->s_size--;
    
  *o = so->data;

  shared_free(so, sizeof(struct stack_o));

  return 0;
}

void traverse_stack(struct stack *s, void (*call)(void *so, void *data), void *data)
{
  struct stack_o *so;
  uint64_t i;
  void *o;

  if (s && call){
      
    so = s->s_data;

    for(i=0; i<s->s_size; i++){
      
      if (so){
        o = so->data;
        (*call)(o, data);
        so = so->o_next;
      }

    }
  } else {
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
  }
}

int funnel_stack(struct stack *src, struct stack *dst, int (*call)(void *so, void *data), void *data)
{
  void *o;

  if (src == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }

#if 0
#ifdef DEBUG
    fprintf(stderr, "%s: pop error\n", __func__);
#endif
    return -1;
#endif

  while (!pop_stack(src, (void **) &o)){
  
    if (call){
      if ((*call)(o, data) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: callback error\n", __func__);
#endif
        return -1;
      }
    }

    if (dst){
      if (push_stack(dst, o) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: push error\n", __func__);
#endif
        return -1;
      }
    }
  
  }
  
  
  return 0;
}

uint64_t get_size_stack(struct stack *s)
{
  return (s) ? s->s_size : -1;
}
