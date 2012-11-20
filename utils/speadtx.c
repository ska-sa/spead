/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>

#include <sys/stat.h>
#include <sys/mman.h>

#include <spead_api.h>

struct spead_tx {
  struct stat   t_fs;
  int           t_fd;
  uint8_t       *t_fmap;

};




void destroy_spead_tx(struct spead_tx *t)
{

}









