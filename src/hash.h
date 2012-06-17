#ifndef HASH_H
#define HASH_H

struct hash_table;

struct hash_o_list {
  uint64_t l_len;
  struct hash_o *l_top;
  void *(*l_create)();
  void (*l_destroy)(void *data);
};

struct hash_o_list *create_o_list(uint64_t len, void *(*create)(), void (*destroy)(void *data), uint64_t size);
void destroy_o_list(struct hash_o_list *l);

int push_hash_o(struct hash_o_list *l, struct hash_o *o);
struct hash_o *pop_hash_o(struct hash_o_list *l);


struct hash_o {
  uint64_t o_len;
  void *o;
  struct hash_o *o_next;
#if 0
  void *(*o_create)();
  void (*o_destroy)(void *data);
#endif
};

void destroy_hash_o(struct hash_o_list *l, struct hash_o *o);
struct hash_o *create_hash_o(void *(*create)(), void (*destroy)(void *data), uint64_t len);

struct hash_o *get_o_ht(struct hash_table *t, uint64_t id);
int add_o_ht(struct hash_table *t, uint64_t id, void *data, uint64_t len);


struct hash_table {
  uint64_t      t_id;
  uint64_t      t_len;
  struct hash_o_list *t_l;
  struct hash_o **t_os;

  uint64_t (*t_hfn)(struct hash_table *t, uint64_t in);
};


void destroy_hash_table(struct hash_table *t);
struct hash_table *create_hash_table(struct hash_o_list *l, uint64_t id, uint64_t len, uint64_t (*hfn)(struct hash_table *t, uint64_t in));




#endif
