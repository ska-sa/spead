/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sysexits.h>


#include "spead_api.h"




int init_speadtx(char *host, char *port, long workers, char broadcast)
{

#ifdef DEBUG
  fprintf(stderr, "%s: %s:%s workers [%ld] broadcast [%d]\n", __func__, host, port, workers, broadcast);
#endif
 
  return 0;
}

int main(int argc, char **argv)
{
  long cpus;
  char c, *port, *host, broadcast;
  int i,j,k;

  i = 1;
  j = 1;
  k = 0;

  host = NULL;

  broadcast = 0;
  
  port = PORT;
  cpus = sysconf(_SC_NPROCESSORS_ONLN);


  while (i < argc){
    if (argv[i][0] == '-'){
      c = argv[i][j];

      switch(c){
        case '\0':
          j = 1;
          i++;
          break;
        case '-':
          j++;
          break;

        /*switches*/  
        case 'x':
          j++;
          broadcast = 1;
          break;

        case 'h':
          fprintf(stderr, "usage:\n\t%s (options) destination port\n\n\tOptions\n\t\t-w [workers (d:%ld)]\n\t\t-x (enable send to broadcast [priv])\n\n", argv[0], cpus);
          return EX_OK;

        /*settings*/
        case 'w':
          j++;
          if (argv[i][j] == '\0'){
            j = 0;
            i++;
          }
          if (i >= argc){
            fprintf(stderr, "%s: option -%c requires a parameter\n", argv[0], c);
            return EX_USAGE;
          }
          switch (c){
            case 'w':
              cpus = atoll(argv[i] + j);
              break;
          }
          i++;
          j = 1;
          break;

        default:
          fprintf(stderr, "%s: unknown option -%c\n", argv[0], c);
          return EX_USAGE;
      }

    } else {
      /*parameters*/
      switch (k){
        case 0:
          host = argv[i];
          k++;
          break;
        case 1:
          port = argv[i];
          k++;
          break;
        default:
          fprintf(stderr, "%s: extra argument %s\n", argv[0], argv[i]);
          return EX_USAGE;
      }
      i++;
      j=1;
    }
  }
  
  

  return init_speadtx(host, port, cpus, broadcast);
}
  
