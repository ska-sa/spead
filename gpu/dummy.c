#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <spead_api.h>

#define CHAN    1024
#define SAMP    128

#define IN_DATA_SET(val) ((val) == 0x08 || (val) == 0x09)

void spead_api_destroy(void *data)
{

}

void *spead_api_setup()
{
  return NULL;
}

int spead_api_callback(struct spead_item_group *ig, void *data)
{
  struct spead_api_item *itm;
  uint64_t off, i, j;

  off = 0;
  while (off < ig->g_size){
    itm = (struct spead_api_item *) (ig->g_map + off);

    if (itm->i_len == 0)
      goto skip;

    if (IN_DATA_SET(itm->i_id)){
#ifdef DEBUG
      //fprintf(stderr, "ITEM id[0x%x] vaild [%d] len [%ld]\n", itm->i_id, itm->i_valid, itm->i_len);
#endif
      //print_data(itm->i_data, sizeof(unsigned char)*itm->i_len);

      for (i=0; i<CHAN; i++){
        
        for(j=0; j<SAMP; j++){
          
          fprintf(stdout, "%d ", itm->i_data[i*SAMP+j]);
  
        }

        fprintf(stdout,"\n");
      
      }

    }

skip:
    off += sizeof(struct spead_api_item) + itm->i_len;
  }

  
  return 0;
}


