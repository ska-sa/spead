#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <spead_api.h>



void spead_api_destroy(struct spead_api_module_shared *s, void *data)
{

}

void *spead_api_setup(struct spead_api_module_shared *s)
{

  return NULL;
}

int spead_api_callback(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data)
{
  
#ifdef DEBUG
  fprintf(stderr, "%s: PID[%d] in module <%s>\n", __func__, getpid(), __FILE__);
#endif

  return 0;
}

int spead_api_timer_callback(struct spead_api_module_shared *s, void *data)
{

  return 0;
}
