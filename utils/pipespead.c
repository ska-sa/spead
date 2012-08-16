#include <stdio.h>
#include <sysexits.h>

#define BUF 5000


void usage(char *argv[])
{
  fprintf(stderr, "%s used to remove any data b4 spead magic and retransmit to UDP\n\tusage %s dst-host dst-port\n\n", argv[0], argv[0]);
}

int main(int argc, char *argv[])
{
  int i, j;
  char c;

  i=0;
  j=0;

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
        case 'h':
          usage(argv);
          return EX_OK;

        /*settings*/
#if 0
        case 'p':
        case 'w':
        case 'b':
#endif
        case 'l':
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
#if 0
            case 'p':
              port = argv[i] + j;  
              break;
            case 'w':
              cpus = atoll(argv[i] + j);
              break;
            case 'b':
              hashes = atoll(argv[i] + j);
              break;
#endif
            case 'l':
              fprintf(stderr, "param l\n");
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
#if 0
      fprintf(stderr, "%s: extra argument %s\n", argv[0], argv[i]);
#endif
      usage(argv);
      return EX_USAGE;
    }
    
  }


  
  return 0;
}
