/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "stack.h"

struct stack *create_stack()
{
  struct stack *s;
  
  s = malloc(sizeof(struct stack));
  if (s == NULL)
    return NULL;

  s->s_data = NULL;
  s->s_size = 0;

  return s;
}

void destroy_stack(struct stack *s, void (*call)(void *data))
{
  int i;
  if (s){
    if (s->s_data){
      for (i=0; i<s->s_size; i++){
        if (call){
          (*call)(s->s_data[i]);
        }
      }
      free(s->s_data);
    }
    free(s);
  }
}

int push_stack(struct stack *s, void *o)
{
  if (s == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error param\n", __func__);
#endif
    return -1;
  }

  s->s_data = realloc(s->s_data, sizeof(void*) * (s->s_size+1));
  if (s->s_data == NULL)
    return -1;

  s->s_data[s->s_size] = o;
  s->s_size++;

  return 0;
}

int pop_stack(struct stack *s, void **o)
{
  void *obj;

  if (s == NULL || o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error param\n", __func__);
#endif
    return -1;
  }
  
  if (s->s_size <= 0){
    return -1;
  }

  obj = s->s_data[s->s_size - 1];

  /*TODO: ADDRESS*/
  s->s_data = realloc(s->s_data, sizeof(void*) * (s->s_size-1));
  s->s_size--;

  *o = obj;

  return 0;
}

void traverse_stack(struct stack *s, void (*call)(void *so, void *data), void *data)
{
  uint64_t i;
  void *o;

  if (s && call){
      
    for(i=0; i<s->s_size; i++){

      if (s->s_data){
        o = s->s_data[i];

        (*call)(o, data);

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
