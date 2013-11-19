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
    
  pq->pq_highest;

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
    
  q = find_data_avltree(pq->pq_tree, &priority);
  if (q == NULL){
    
    if (priority == NULL)
      return -1;

    q = create_queue(priority);
    if(q == NULL){
      return -1;
    }
  
    if (pq->pq_highest == NULL){
      pq->pq_highest = q;
    } else if (q->q_id > pq->pq_highest->q_id){
      pq->pq_highest = q;
#ifdef DEBUG
      fprintf(stderr, "%s: new highest priority queue [%d]\n", __func__, q->q_id);
#endif
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

  if (pq->pq_highest == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s cannot access highest priority queue\n", __func__);
#endif
    return -1;
  }
/*

  TODO: steps dequeue from highest 
              if dequeue fails remove highest from tree (deletion)
              search for new max in tree set to highest
              try dequeue if fail repeat else return data


  */

#ifdef DEBUG
  fprintf(stderr, "%s: trying dequeue from queue [%d]\n", __func__, pq->pq_highest->q_id);
#endif

  if (dequeue(pq->pq_highest, data) < 0){
    
    if (del_name_node_avltree(pq->pq_tree, &(pq->pq_highest->q_id), NULL) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: failed 0\n", __func__);
#endif
      return -1;
    }

    destroy_queue(pq->pq_highest, NULL);
    pq->pq_highest = NULL;
    
    pq->pq_highest = get_max_data_avltree(pq->pq_tree);
    if (pq->pq_highest == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: get max fail DONE pq\n", __func__);
#endif
      return -2;
    }

    if (dequeue(pq->pq_highest, data) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: failed 2\n", __func__);
#endif
      return -1;
    }
    
  }
  
#ifdef DEBUG
  fprintf(stderr, "%s: pull highest done <%p>\n", __func__, data);
#endif
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
    //fprintf(stderr, "%s: <%p> value [%d] oxor %p\n", __func__, cur, *((int*)(cur->data)), cur->o_xor);
    fprintf(stderr, "%s: <%p> oxor %p\n", __func__, cur, cur->o_xor);
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

  fprintf(stderr, "%s: trav\n", __func__);
  traverse_queue(pq->q_front);

  return 0;
}

int main(int argc, char *argv[])
{
  struct priority_queue *pq;
  struct queue *q;
  
  int id = 555, id2 = 6666;

  pq = create_priority_queue();

  insert_with_priority_queue(pq, 1, &id);
  insert_with_priority_queue(pq, 1, &id);
  insert_with_priority_queue(pq, 1, &id);
  //q = get_max_data_avltree(pq->pq_tree);
  //fprintf(stderr, "MAX: %d\n", q->q_id);
  insert_with_priority_queue(pq, 2, &id);
  insert_with_priority_queue(pq, 2, &id);
  insert_with_priority_queue(pq, 2, &id);
  insert_with_priority_queue(pq, 1, &id2);
  insert_with_priority_queue(pq, 1, &id2);
  insert_with_priority_queue(pq, 1, &id2);
  //q = get_max_data_avltree(pq->pq_tree);
  //fprintf(stderr, "MAX: %d\n", q->q_id);
  insert_with_priority_queue(pq, 3, &id);
  insert_with_priority_queue(pq, 3, &id2);
  insert_with_priority_queue(pq, 3, &id);

  //q = get_max_data_avltree(pq->pq_tree);
  //fprintf(stderr, "MAX: %d\n", q->q_id);
 
  int *data;

  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  if (pull_highest_priority(pq, &data) == 0)
    fprintf(stderr, "%s: data %d\n", __func__, *data);
  //destroy_priority_queue(pq, NULL);

  destroy_shared_mem();

  return 0;
}

#endif
