#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <sys/mman.h>

#include <spead_api.h>
#include <tx.h>

#define S_END     0
#define S_STAT    1
#define S_WRITE   2

struct write_file {
  int         w_fd;
  int         w_state;
  struct stat w_stat;
  char        *w_name;
  uint64_t    w_size;
  void        *w_data;
};

void spead_api_destroy(void *data)
{
  struct write_file *ws;
  ws = data;
  if (ws){
    if (ws->w_name)
      free(ws->w_name);
    if (ws->w_data){
      munmap(ws->w_data, ws->w_size);
    }
    free(ws);
  }
}

void *spead_api_setup()
{
  struct write_file *ws;

  ws = malloc(sizeof(struct write_file));
  if (ws == NULL)
    return NULL;
  
  ws->w_name = NULL;
  ws->w_size = 0;
  ws->w_data = NULL;

  return ws;
}

int spead_api_callback(struct spead_item_group *ig, void *data)
{
  struct spead_api_item *itm;
  struct write_file *w;
  int64_t size;

  w = data;
  if (w == NULL)
    return -1;
  
  itm = NULL;

  while ((itm = get_next_spead_item(ig, itm))){
    //print_data(itm->i_data, itm->i_data_len);
    if (itm->i_id == SPEADTX_IID_FILENAME){
      fprintf(stderr, "%s: FILENAME [%s]\n", __func__, itm->i_data);
    }
    if (itm->i_id == SPEADTX_IID_FILESIZE){
      memcpy(&size, itm->i_data, sizeof(int64_t));
      fprintf(stderr, "%s: FILESIZE [%ld]\n", __func__, size);
    }
     
    
    
  }

  return 0;
}



