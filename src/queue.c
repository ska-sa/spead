/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "queue.h"
#include "spead_api.h"

struct queue *create_queue(int pid)
{
  struct queue *q;

  q = shared_malloc(sizeof(struct queue));
  if (q == NULL)
    return NULL;

  q->q_front = NULL;
  q->q_back  = NULL;
  q->q_id    = pid;
  
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
    shared_free(q, sizeof(struct queue));
  }
}

int compare_priority_queues(const void *v1, const void *v2)
{
  if (*(int*)v1 < *(int*)v2)
    return -1;
  else if (*(int*)v1 > *(int*)v2)
    return 1;
  return 0;
}

struct priority_queue *create_priority_queue()
{
  struct priority_queue *pq;

  pq = shared_malloc(sizeof(struct priority_queue));
  if (pq == NULL)
    return NULL;
#if 0
  pq->pq_priorities = create_queue();
  if (pq->pq_priorities == NULL){
    shared_free(pq, sizeof(struct priority_queue));
    return NULL;
  }
#endif
  
  pq->pq_tree = create_avltree(&compare_priority_queues);
  if (pq->pq_tree == NULL){
#if 0
    destroy_queue(pq->pq_priorities, NULL);
#endif
    shared_free(pq, sizeof(struct priority_queue));
    return NULL;
  }
    
  return pq;
}

void destroy_priority_queue(struct priority_queue *pq, void (*call)(void *data))
{
  void *d;
  if (pq){
#if 0
    destroy_queue(pq->pq_priorities, NULL); /*Address this*/
#endif
    destroy_avltree(pq->pq_tree, call);
    shared_free(pq, sizeof(struct priority_queue));
  }
}

int insert_with_priority_queue(struct priority_queue *pq, int priority, void *data)
{
  struct queue *q;

  if (pq == NULL)
    return -1;
    
  q = find_name_node_avltree(pq->pq_tree, priority);
  if (q == NULL){
    
    if (priority == NULL)
      return -1;

    q = create_queue(priority);
    if(q == NULL){
      return -1;
    }
    
    if (store_named_node_avltree(pq->pq_tree, &(q->q_id), q) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: unable to store named node\n", __func__);
#endif
      destroy_queue(q, NULL);
      return -1;
    }

#ifdef DEBUG
    fprintf(stderr, "%s: ENQUEUE into new pq[%ld]\n", __func__, priority);
#endif

    if (enqueue(q, data) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: unable to store named node\n", __func__);
#endif
      destroy_queue(q, NULL);
      return -1;
    }

    return 0;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: ENQUEUE into existing pq[%ld]\n", __func__, priority);
#endif

  if (enqueue(q, data) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: unable to store named node\n", __func__);
#endif
    destroy_queue(q, NULL);
    return -1;
  }
  
  return 0;
}


int pull_highest_priority(struct priority_queue *pq, void **data)
{
  if (pq == NULL)
    return -1;
  


  
  return 0;
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


  q = create_queue(0);

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
  destroy_shared_mem();
  
  return 0; 
}
#endif

#ifdef TEST_PQUEUE
#include <string.h>

int walk_callback_priority_queue(void *data, void *node_data)
{
  struct queue *pq=NULL;

  pq = node_data;

  if (pq == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: null data\n", __func__);
#endif
    return -1;
  }

  traverse_queue(pq->q_front);

  return 0;
}

int main(int argc, char *argv[])
{
  struct priority_queue *pq;

  pq = create_priority_queue();

  int p1 = 2, p2 = 1;


  insert_with_priority_queue(pq, p1, NULL);
  insert_with_priority_queue(pq, p1, NULL);

  insert_with_priority_queue(pq, p2, NULL);
  insert_with_priority_queue(pq, p2, NULL);
  
#if 0
  while (walk_inorder_avltree(pq->pq_tree, &walk_callback_priority_queue, NULL) < 0){

#ifdef DEBUG
    fprintf(stderr, "%s: walk\n", __func__);
#endif
    
  }
#endif

  //destroy_priority_queue(pq, NULL);

  destroy_shared_mem();

  return 0;
}

#endif
