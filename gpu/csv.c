#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <spead_api.h>
#include <avltree.h>

#include "shared.h"

struct json_file {
  char *j_name;
  struct data_file *j_df;
};


void spead_api_destroy(struct spead_api_module_shared *s, void *data)
{
  struct json_file *json;
  
  lock_spead_api_module_shared(s);

  if ((json = get_data_spead_api_module_shared(s))){ 
    
    if (json->j_name)
      shared_free(json->j_name, sizeof(char)*(strlen(json->j_name) + 1));

    destroy_raw_data_file(json->j_df);

    shared_free(json, sizeof(struct json_file));

    clear_data_spead_api_module_shared(s);
  }

  unlock_spead_api_module_shared(s);
}

void *spead_api_setup(struct spead_api_module_shared *s)
{
  struct json_file *json;
  
  lock_spead_api_module_shared(s);

  if (!(json = get_data_spead_api_module_shared(s))){

    json = shared_malloc(sizeof(struct json_file));
    if (json == NULL)
      return NULL;
  
    set_data_spead_api_module_shared(s, json, sizeof(struct json_file));

    json->j_name = NULL;
    json->j_df   = NULL;

    json->j_name = gen_id_avltree("id.json");
    if (json->j_name == NULL){
      spead_api_destroy(s, json);
      return NULL;
    }

    json->j_df = write_raw_data_file(json->j_name);
    if (json->j_df == NULL){
      spead_api_destroy(s, json);
      return NULL;
    }

  }

  unlock_spead_api_module_shared(s);
  
  return NULL;
}

int spead_api_callback(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data)
{
  struct json_file *json;
  struct spead_api_item *itm;
  char buf[12];
  char *hdr = "{ \"items\": [ ";
  char *ftr = "] }";

  char *src;

  json = get_data_spead_api_module_shared(s);

  lock_spead_api_module_shared(s);
  if (json == NULL){
    unlock_spead_api_module_shared(s);
    return -1;
  }

  write_next_chunk_raw_data_file(json->j_df, hdr, strlen(hdr));
  
  itm = NULL;
  src = NULL;

  while((itm = get_next_spead_item(ig, itm))){
    
    write_next_chunk_raw_data_file(json->j_df, "{", strlen("{"));
    
    src = itoa(itm->i_id, buf);
    write_next_chunk_raw_data_file(json->j_df, src, strlen(buf));
    write_next_chunk_raw_data_file(json->j_df, ":", strlen(":"));

    write_next_chunk_raw_data_file(json->j_df, itm->i_data, itm->i_data_len);
    write_next_chunk_raw_data_file(json->j_df, "}", strlen("}"));

  }

  write_next_chunk_raw_data_file(json->j_df, ftr, strlen(ftr));
  

  unlock_spead_api_module_shared(s);

  return 0;
}


int spead_api_timer_callback(struct spead_api_module_shared *s, void *data)
{
  return 0; 
}
