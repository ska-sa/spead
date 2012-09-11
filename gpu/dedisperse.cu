#include <stdio.h>
#include <math.h>

#include <cufft.h>

#include <spead_api.h>

void *spead_api_setup()
{
  

}

void spead_api_destroy(void *data)
{
  if (data){

    
  }
}

int spead_api_callback(struct spead_item_group *ig)
{
 
#ifdef DEBUG
  fprintf(stderr, "we are in the callback\n");
#endif


#if 0

  uint64_t off = 0;
  uint64_t count;

  struct spead_api_item *itm;

  while (off < ig->g_size){
  
    itm = (struct spead_api_item *) (ig->g_map + off);

    if (itm->i_len == 0)
      goto skip;

    count = 0;
    fprintf(stderr, "ITEM id[%d] vaild [%d] len [%ld]\n", itm->i_id, itm->i_valid, itm->i_len);
    
    print_data(itm->i_data, itm->i_len);

skip:
    off += sizeof(struct spead_api_item) + itm->i_len;
    
  }

#endif

  return 0;
}


