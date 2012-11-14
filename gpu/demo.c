#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>



int main(int argc, char *argv[])
{
#if 0
  struct sapi_o *a;
#endif
  struct stat fs;
  
  uint8_t *data;
  int fd;
  char *fname;

  uint64_t off, chunk, have;

  if (argc < 2){
#ifdef DEBUG
    fprintf(stderr, "e: usage: %s datafile\n", argv[0]);
#endif
    return 1;
  }
  
  fname = argv[1];

  /*map file into memory*/
  if (stat(fname, &fs) < 0){
#ifdef DEBUG
    fprintf(stderr, "e: stat error: %s\n", strerror(errno));
#endif
    return 1;
  }
  
  fd = open(fname, O_RDONLY);
  if (fd < 0){
#ifdef DEBUG
    fprintf(stderr, "e: open error: %s\n", strerror(errno));
#endif
    return 1;
  }

  data = mmap(NULL, fs.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: mmap error: %s\n", strerror(errno));
#endif
    close(fd);
    return 1;
  }
  

#ifdef DEBUG
  fprintf(stderr, "mapped <%s> into (%p) size [%d] bytes\n", fname, data, (int) fs.st_size);
#endif


#if 0
  a = spead_api_setup();
  if (a == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: spead api setup\n"); 
#endif
    munmap(data, fs.st_size);
    close(fd);
    return 1;
  }
  

  off   = 0;
  chunk = 1024;
  have  = fs.st_size;

  do {

    print_data(data+off, (have < chunk) ? have : chunk);

    off += chunk;
    have -= chunk;

#ifdef DEBUG
    fprintf(stderr, "\n");
#endif

  } while (off < fs.st_size);

  spead_api_destroy(a);
#endif


  munmap(data, fs.st_size);
  close(fd);
    
  return 0;
}
