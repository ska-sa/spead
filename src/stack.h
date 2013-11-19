#ifndef STACK_H
#define STACK_H

#include <stdint.h>

struct stack_o {
  struct stack_o *o_next;
  void *data;
};

struct stack {
  struct stack_o *s_data;
  uint64_t  s_size;
};

struct stack *create_stack();
void destroy_stack(struct stack *s, void (*call)(void *data));
int push_stack(struct stack *s, void *o);
int pop_stack(struct stack *s, void **o);
void traverse_stack(struct stack *s, void (*call)(void *so, void *data), void *data);
int funnel_stack(struct stack *src, struct stack *dst, int (*call)(void *so, void *data), void *data);
uint64_t get_size_stack(struct stack *s);

void empty_stack(struct stack *s, void (*call)(void *data));

#endif

