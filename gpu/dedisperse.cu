#include <stdio.h>
#include <math.h>

#include <cufft.h>

#include <spead_api.h>


int spead_api_callback(struct spead_item_group *ig)
{
 
#ifdef DEBUG
  fprintf(stderr, "we are in the callback\n");
#endif

  uint64_t off = 0;
  uint64_t count;

  struct spead_api_item *itm;

  while (off < ig->g_size){
  
    itm = (struct spead_api_item *) (ig->g_map + off);

    if (itm->i_len == 0)
      goto skip;

    count = 0;
#ifdef DEBUG
    fprintf(stderr, "ITEM id[%d] vaild [%d] len [%ld]\n", itm->i_id, itm->i_valid, itm->i_len);
#endif
    
    print_data(itm->i_data, itm->i_len);

skip:
    off += sizeof(struct spead_api_item) + itm->i_len;
    
  }



  return 0;
}


