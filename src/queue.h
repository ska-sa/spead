#ifndef QUEUE_H
#define QUEUE_H

#include <stdint.h>
#include "avltree.h"

struct queue_o {
  intptr_t o_xor;
  void *data;
};

struct queue {
  int             q_id;
  struct queue_o *q_front;
  struct queue_o *q_back;
};

struct queue *create_queue(int pid);
void destroy_queue(struct queue *q, void (*call)(void *data));
int enqueue(struct queue *q, void *o);
int dequeue(struct queue *q, void **o);

struct priority_queue{
  struct avl_tree *pq_tree; 
  struct queue    *pq_highest;
};

struct priority_queue *create_priority_queue();
void destroy_priority_queue(struct priority_queue *p, void (*call)(void *data));
int insert_with_priority_queue(struct priority_queue *pq, int priority, void *data);
int pull_highest_priority(struct priority_queue *pq, void **data);

#endif
