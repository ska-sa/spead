/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "queue.h"
#include "spead_api.h"

struct queue *create_queue()
{
  struct queue *q;

  q = shared_malloc(sizeof(struct queue));
  if (q == NULL)
    return NULL;

  q->q_front = NULL;
  q->q_back  = NULL;
  
  return q;
}


void destroy_queue(struct queue *q, void (*call)(void *data))
{
  void *d;
  if (q){
    while (dequeue(q, &d) == 0){
      if (call){
        (*call)(d);
      }
    }
  }
}


struct queue_o *new_node_xll(struct queue_o *prev, struct queue_o *cur, void *data)
{
  struct queue_o *next;

  next = shared_malloc(sizeof(struct queue_o));
  if (next == NULL)
    return NULL;

  next->data = data;
  next->o_xor = cur;

  if (cur == NULL)
    return next;
  else if (prev == NULL) {
    cur->o_xor = next;
    next->o_xor = cur;
  } else {
    cur->o_xor = (struct queue_o*)((intptr_t)prev ^ (intptr_t)next);
  }

  return next;
}

int enqueue(struct queue *q, void *o)
{
  if (q == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error param\n", __func__);
#endif
    return -1;
  }
  
  if (q->q_front == NULL){
    q->q_front = new_node_xll(NULL, NULL, o);
    q->q_back = q->q_front;
  }
  else
    q->q_back = new_node_xll((q->q_back ? q->q_back->o_xor : NULL) , q->q_back, o);

#ifdef DEBUG
  fprintf(stderr, "%s: qf %p qb %p\n", __func__, q->q_front, q->q_back);
#endif

  return 0;
}


int dequeue(struct queue *q, void **o)
{
  struct queue_o *qo, *prev, *cur;
  
  if (q == NULL)
    return -1;

  qo = q->q_front;
  if (qo == NULL)
    return -1;

  *o = qo->data; 
  cur = qo->o_xor;
  q->q_front = cur;

  if (cur == NULL){
    q->q_back = NULL;
    return 0;
  }

  prev = (struct queue_o *) (cur->o_xor ^ (intptr_t)qo);
  cur->o_xor = (intptr_t)prev; 

  q->q_front = cur;

#if 0 
def DEBUG
  fprintf(stderr, "%s: from back cur %p cxor %p prev %p pxor %p\n", __func__, cur, cur->o_xor, prev, prev->o_xor);
#endif

  //shared_free(qo, sizeof(struct queue_o));

  return 0;
}

int traverse_queue(struct queue_o *start)
{
  struct queue_o *prev, *cur, *save;
  
  if (start == NULL)
    return -1;

  cur   = start;
  prev  = start;
  while(cur){
    fprintf(stderr, "%s: <%p> value [%d] oxor %p\n", __func__, cur, *((int*)(cur->data)), cur->o_xor);
    if (cur->o_xor == cur)
      break;
    if (cur == prev)
      cur = cur->o_xor;
    else {
      save = cur;
      cur = (struct queue_o*)((intptr_t)prev ^ (intptr_t)cur->o_xor);
      prev = save;
    }
  }

  return 0;
}


#ifdef TEST_QUEUE
#include <string.h>

int main(int argc, char *argv[])
{
  struct queue *q;
  int i, *o;


  q = create_queue();

  for (i=0; i< 10; i++){
    int *data = malloc(sizeof(int));
    memcpy(data, &i, sizeof(int));
    enqueue(q, data);
  }

  

  traverse_queue(q->q_front);
  fprintf(stderr, "----\n");
  traverse_queue(q->q_back);


  for (i=0; i<7; i++){
    if (dequeue(q, &o) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: err dequeue\n", __func__);
#endif
    }

    fprintf(stderr, "got [%d]\n", *o);

    free(o);

  }
  
  for (i=0; i< 10; i++){
    int *data = malloc(sizeof(int));
    memcpy(data, &i, sizeof(int));
    enqueue(q, data);
  }

  
  traverse_queue(q->q_front);

  fprintf(stderr, "----\n");

  traverse_queue(q->q_back);


  destroy_queue(q, &free);
  
  //destroy_shared_mem();
  
  return 0; 
}
#endif


