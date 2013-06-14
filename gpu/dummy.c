#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <spead_api.h>

#include "shared.h"

#define COUNT           500
#define SPEAD_DATA_ID   0xb001

#define IN_DATA_SET(val) ((val) == 0x08 || (val) == 0x09 || (val) == 0x0 || (val) == 0xb001)

struct sapi_o {
  int o_count;
  int o_chunk;
  int o_off;
  int o_item_size;
  void *o_data;
};

void spead_api_destroy(struct spead_api_module_shared *s, void *data)
{
  struct sapi_o *o;
  o = data;

  if (o){
    if (o->o_data != NULL){
      free(o->o_data);
    }
    free(o);
  }

}

void *spead_api_setup(struct spead_api_module_shared *s)
{
  struct sapi_o *o;

  o = malloc(sizeof(struct sapi_o));
  if (o == NULL){
    return NULL;
  }

  o->o_count     = COUNT;
  o->o_chunk     = 0;
  o->o_data      = NULL;
  o->o_off       = 0;
  o->o_item_size = sizeof(float2);
    
  //fprintf(stdout, "set term png size 1280,720\nset view map\nsplot '-' matrix with image\n");

  return o;
}

int setup_mem(struct sapi_o *o, int len)
{
  if (o == NULL || len < 0)
    return -1;

  o->o_chunk = len;
  
  o->o_data = malloc(o->o_item_size * o->o_chunk * o->o_count);
  if (o->o_data == NULL)
    return -1;

  return 0;
}

int spead_api_callback(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data)
{
  struct sapi_o *o;
  struct spead_api_item *itm;
  float2 *pow;
  int i,j;

  o = data;
  if (o == NULL){
    return -1;
  }
  
  itm = get_spead_item_with_id(ig, SPEAD_DATA_ID);
  if (itm == NULL)
    return -1;

  if (o->o_data == NULL){
    if (setup_mem(o, itm->io_size) < 0){
      return -1;
    }
  }
  

  memcpy(o->o_data + o->o_item_size*o->o_off*o->o_chunk, itm->io_data, o->o_item_size*o->o_chunk);
  o->o_off++;

  if (o->o_off == COUNT){
    fprintf(stdout, "set term x11 size 1280,720\nset view map\nsplot '-' matrix with image\n");


    for (j=0; j<o->o_count; j++){
      pow = (float2*) (o->o_data + o->o_item_size*j*o->o_chunk);
      for (i=o->o_chunk/2; i<o->o_chunk; i++){
        fprintf(stdout,"%0.11f ",pow[i].x);
      }
      fprintf(stdout, "\n");
    }

    fprintf(stdout, "e\ne\n");

    o->o_off = 0; 
  }

  
#if 0
  struct spead_api_item *itm;
  uint64_t off, i, j;
  float2 *pow;

  off = 0;
  while (off < ig->g_size){
    itm = get_spead_item_at_off(ig, off);

    if (itm == NULL)
      return -1;

    if (IN_DATA_SET(itm->i_id)){
#if 0
      def DEBUG
      fprintf(stderr, "ITEM id[0x%x] vaild [%d] len [%ld]\n", itm->i_id, itm->i_valid, itm->i_len);
#endif

#if 0
      print_data(itm->io_data, itm->io_size);
      print_data(itm->i_data, sizeof(unsigned char)*itm->i_len);
      fprintf(stdout, "splot '-' matrix with image\n");
      for (i=0; i<CHAN; i++){
        for(j=0; j<SAMP; j++){
          fprintf(stdout, "%d %d ", itm->i_data[i*SAMP+j], itm->i_data[i*SAMP+j+1]);
        }
        fprintf(stdout,"\n");
      }
      fprintf(stdout,"e\ne\n");
#endif

      pow = (float2*) itm->io_data;

      for (i=1; i<itm->io_size;i++){
      //for (i=itm->io_size/2; i<itm->io_size;i++){
        fprintf(stdout,"%0.11f ",pow[i].x);
      }
      fprintf(stdout, "\n");

      break;
    }

    off += sizeof(struct spead_api_item) + itm->i_len;
  }
#endif
  
  return 0;
}

int spead_api_timer_callback(struct spead_api_module_shared *s, void *data)
{

  return 0;
}

