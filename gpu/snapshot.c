#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <spead_api.h>

struct snap_shot {
  int flag;
};

void spead_api_destroy(void *data)
{
  struct snap_shot *ss;
  ss = data;
  if (ss){
    shared_free(ss, sizeof(struct snap_shot));
  }
}

void *spead_api_setup()
{
  struct snap_shot *ss;

  ss = shared_malloc(sizeof(struct snap_shot));
  if (ss == NULL)
    return NULL;
  
  ss->flag = 0;

  return ss;
}

int spead_api_callback(struct spead_item_group *ig, void *data)
{
  struct snap_shot *ss;

  ss = data;
  if (ss == NULL)
    return -1;

  


  return 0;
}
