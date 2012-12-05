#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <spead_api.h>

#define CHAN    64
#define SAMP    128

#define IN_DATA_SET(val) ((val) == 0x08 || (val) == 0x09 || (val) == 0x0)

void spead_api_destroy(void *data)
{
  fprintf(stdout, "e\ne\n");
}

void *spead_api_setup()
{
  
  fprintf(stdout, "set term x11 size 1280,720\nset view map\nsplot '-' matrix with image\n");
  //fprintf(stdout, "set term png size 1280,720\nset view map\nsplot '-' matrix with image\n");

  return NULL;
}

int spead_api_callback(struct spead_item_group *ig, void *data)
{
  struct spead_api_item *itm;
  uint64_t off, i, j;
  float *pow;

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
      pow = (float*) itm->io_data;

      for (i=1; i<itm->io_size / sizeof(float);i++){
        fprintf(stdout,"%0.11f ",pow[i]);
      }
      fprintf(stdout, "\n");

      break;
    }

    off += sizeof(struct spead_api_item) + itm->i_len;
  }

  
  return 0;
}


