/* (c) 2010,2011 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sysexits.h>
#include <string.h>

#include "server.h"

pid_t fork_child_sp(struct u_server *s, int (*call)(struct u_server *s))
{
  pid_t cpid;

  if (call == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: null callback\n", __func__);
#endif
    return -1;
  }
  
  cpid = fork();

  if (cpid < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: fork fail: %s\n", __func__, strerror(errno));
#endif
    return -1;
  }

  if (cpid > 0) {
    /*in parent*/
#ifdef DEBUG
    fprintf(stderr, "%s:\tnew child pid [%d]\n", __func__, cpid);
#endif
    return cpid;
  }

  /*in child use exit not return*/ 
  (*call)(s);

  exit(EX_OK);
  return 0;
}

