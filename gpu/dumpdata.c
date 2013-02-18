#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <spead_api.h>

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
  
  itm = NULL;

  while ((itm = get_next_spead_item(ig, itm))){
    
    print_data(itm->i_data, itm->i_data_len);

  }

  return 0;
}
