#ifndef STACK_H
#define STACK_H

struct stack {
  void      **s_data;
  uint64_t  s_size;
};

struct stack *create_stack();
void destroy_stack(struct stack *s);
int push_stack(struct stack *s, void *o);
int pop_stack(struct stack *s, void **o);

#endif

