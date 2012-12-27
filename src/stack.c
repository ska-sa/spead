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

void destroy_stack(struct stack *s)
{
  if (s){
    if (s->s_data)
      free(s->s_data);
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

  s->s_data = realloc(s->s_data, sizeof(void*) * (s->s_size-1));
  s->s_size--;

  *o = obj;

  return 0;
}

