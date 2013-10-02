#ifndef QUEUE_H
#define QUEUE_H

#include <stdint.h>

struct queue_o {
  intptr_t o_xor;
  void *data;
};

struct queue {
  struct queue_o *q_front;
  struct queue_o *q_back;
  uint64_t q_size;
};

struct queue *create_queue();
void destroy_queue(struct queue *q, void (*call)(void *data));
int enqueue(struct queue *q, void *o);
int dequeue(struct queue *q, void **o);

#endif
