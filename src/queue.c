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
  q->q_size  = 0;
  
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

int enqueue(struct queue *q, void *o)
{
  struct queue_o *qo;
  
  if (q == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error param\n", __func__);
#endif
    return -1;
  }

  qo = shared_malloc(sizeof(struct queue_o));
  if (qo == NULL)
    return -1;
  
  if (q->q_back == NULL){
    q->q_back = qo;
  }

  qo->data    = o;

  qo->o_xor   = (uintptr_t) (void *) q->q_front ^ (uintptr_t) (void *) qo;
  q->q_front  = qo;


  q->q_size++;

  return 0;
}

int dequeue(struct queue *q, void **o)
{
  
  if (q == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error param\n", __func__);
#endif
    return -1;
  }
  
  if (q->q_back == NULL)
    return -1;
  
  q->q_back = (void *) ((uintptr_t) (void *) q->q_back ^ o->o_xor);

  *o = q->q_back;
  


  return 0;
}

