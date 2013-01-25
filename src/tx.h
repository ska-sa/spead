#ifndef TX_H
#define TX_h
  
#define SPEADTX_IID_FILESIZE  0x333
#define SPEADTX_IID_FILENAME  0x334


struct spead_tx {
  mutex                     t_m;
  struct spead_socket       *t_x;
  struct spead_workers      *t_w;
  struct data_file          *t_f;
  int                       t_pkt_size; 
  int                       t_chunk_size; 
  struct avl_tree           *t_t;
  struct spead_heap_store   *t_hs;
  uint64_t                  t_count;
};



#endif
