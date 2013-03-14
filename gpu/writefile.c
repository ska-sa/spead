#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <spead_api.h>
#include <tx.h>

#define S_END     0
#define S_STAT    1
#define S_WRITE   2

struct write_file {
  int         w_fd;
  int         w_state;
  char        *w_name;
  uint64_t    w_size;
  void        *w_data;
};

void spead_api_destroy(struct spead_api_module_shared *s, void *data)
{
  struct write_file *ws;
  ws = data;
  if (ws){
    if (ws->w_name)
      free(ws->w_name);
#if 0
    if (ws->w_data)
      munmap(ws->w_data, ws->w_size);
#endif
    if (ws->w_fd > 0)
      close(ws->w_fd);
    free(ws);
  }
}

void *spead_api_setup(struct spead_api_module_shared *s)
{
  struct write_file *ws;

  ws = malloc(sizeof(struct write_file));
  if (ws == NULL)
    return NULL;
  
  ws->w_fd   = 0;
  ws->w_state= S_STAT;
  ws->w_name = NULL;
#if 0
  ws->w_size = 0;
#endif
  ws->w_data = NULL;

  return ws;
}

int spead_api_callback(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data)
{
  struct spead_api_item *data_item, *itm;
  struct write_file *ws;
  uint64_t chunk_id, off, wb, sw, pos;

  ws = data;
  if (ws == NULL)
    return -1;
  
  wb =0;
  sw =0;
  pos=0;
  itm = NULL;
  data_item = NULL;

  switch (ws->w_state){

    case S_STAT:

      while ((itm = get_next_spead_item(ig, itm))){
        if (itm->i_id == SPEADTX_IID_FILENAME){
          ws->w_name = strdup((const char*)itm->i_data); 
        }
#if 0
        if (itm->i_id == SPEADTX_IID_FILESIZE){
          memcpy(&(ws->w_size), itm->i_data, sizeof(int64_t));
          //fprintf(stderr, "%s: FILESIZE [%ld]\n", __func__, size);
        }
#endif
        if (itm->i_id == SPEADTX_CHUNK_ID)
          break;
      }

      ws->w_fd = open(ws->w_name, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
      if (ws->w_fd < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: open error (%s)\n", __func__, strerror(errno));
#endif
        return -1;
      }

#ifdef DEBUG
     fprintf(stderr, "[%d] %s: STAT %s fd %d\n", getpid(), __func__, ws->w_name, ws->w_fd);
#endif

      ws->w_state = S_WRITE;

    case S_WRITE:
      
      do {
        if (itm){
          
          if (itm->i_id == SPEADTX_CHUNK_ID){
            memcpy(&chunk_id, itm->i_data, sizeof(int64_t));
          }
          if (itm->i_id == SPEADTX_OFF_ID){
            memcpy(&off, itm->i_data, sizeof(int64_t));
          }
          if (itm->i_id == SPEADTX_DATA_ID){
            data_item = itm;
          }

        }
      } while ((itm = get_next_spead_item(ig, itm)));

#ifdef DEBUG
     fprintf(stderr, "[%d] %s: WRITE id %ld off %ld\n", getpid(), __func__, chunk_id, off);
#endif
      
      sw = data_item->i_data_len;
      pos = 0;

      do { 
        wb = pwrite(ws->w_fd, data_item->i_data + pos, sw, off + pos);
        if (wb <= 0){
          if (wb < 0){
            switch(errno){
              case EINTR:
              case EAGAIN:
                continue;
              default:
#ifdef DEBUG
                fprintf(stderr, "%s: pwrite error (%s)\n", __func__, strerror(errno));
#endif
                return -1;
            }
          } else {
            /*should never reach here*/
            if (pos == data_item->i_data_len)
              break;
          }
        } else {
          pos += wb;
          sw  -= wb;
        }
      } while(wb == data_item->i_data_len || sw <= 0);
      
      break;

    case S_END:

      break;
  }

  return 0;
}

int spead_api_timer_callback(struct spead_api_module_shared *s, void *data)
{

  return 0;
}

