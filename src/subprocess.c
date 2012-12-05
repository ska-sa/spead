/* (c) 2010,2011 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sysexits.h>
#include <string.h>
#include <signal.h>

#include <sys/types.h> 
#include <sys/wait.h>

#include "server.h"
#include "spead_api.h"
#include "avltree.h"

void destroy_spead_workers(struct spead_workers *w)
{
  if (w){
    destroy_avltree(w->w_tree, &destroy_child_sp); 
    free(w);
  }
}

int compare_spead_workers(const void *v1, const void *v2)
{
  if (*(int *) v1 < *(int *) v2)
    return -1;
  if (*(int *) v1 > *(int *) v2)
    return 1;
  return 0;
}

struct spead_workers *create_spead_workers(void *data, long count, int (*call)(void *data, struct spead_api_module *m, int cfd))
{
  struct spead_workers  *w;
  struct u_child        *c;
  int i;

  w = malloc(sizeof(struct spead_workers));
  if (w == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: logic cannot malloc\n", __func__);
#endif
    return NULL;
  }

  w->w_tree   = NULL;
  w->w_count  = 0;

  w->w_tree = create_avltree(&compare_spead_workers);
  if (w->w_tree == NULL){
    destroy_spead_workers(w);
    return NULL;
  }

  for (i=0; i<count; i++){
  
    c = fork_child_sp(NULL, data, call);
    if (c == NULL)
      continue;         /* continue starting other children*/
    
    if (store_named_node_avltree(w->w_tree, &(c->c_pid), c) < 0){
      destroy_child_sp(c);
      continue;         /* continue starting other children*/
    }
    
    w->w_count++;       /* ends up containing amount of started childred*/

  }

  return w;
}

int get_count_spead_workers(struct spead_workers *w)
{
  return (w) ? w->w_count : 0;
}

int wait_spead_workers(struct spead_workers *w)
{
  pid_t pid;
  int status;

  if (w == NULL)
    return -1;
  
  while((pid = waitpid(WAIT_ANY, &status, WNOHANG)) > 0){

    if (WIFEXITED(status)) {
#ifdef DEBUG
      fprintf(stderr, "exited, status=%d\n", WEXITSTATUS(status));
#endif
      
      if (del_name_node_avltree(w->w_tree, &pid, &destroy_child_sp) == 0){
        w->w_count--;
      }

    } else if (WIFSIGNALED(status)) {
#ifdef DEBUG
      fprintf(stderr, "killed by signal %d\n", WTERMSIG(status));
#endif

      if (del_name_node_avltree(w->w_tree, &pid, &destroy_child_sp) == 0){
        w->w_count--;
      }

    } else if (WIFSTOPPED(status)) {
#ifdef DEBUG
      fprintf(stderr, "stopped by signal %d\n", WSTOPSIG(status));
#endif

      if (del_name_node_avltree(w->w_tree, &pid, &destroy_child_sp) == 0){
        w->w_count--;
      }

    } else if (WIFCONTINUED(status)) {
#ifdef DEBUG
      fprintf(stderr, "continued\n");
#endif
    }

  }

  return 0;
}


struct u_child *create_child_sp(pid_t pid, int cfd)
{
  struct u_child *c;

  c = malloc(sizeof(struct u_child));
  if (c == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: logic error cannot malloc child\n", __func__);
#endif
    return NULL;
  }

  c->c_pid = pid;
  c->c_fd  = cfd;

  return c;
}

void destroy_child_sp(void *data)
{
  struct u_child *c;
  c = data;
  if (c){
    
    if (c->c_fd)
      close(c->c_fd);
    
    kill(c->c_pid, SIGTERM); 
    
    free(c);
  }
}



int add_child_us(struct u_child ***cs, struct u_child *c, int size)
{
  if (cs == NULL || c == NULL)
    return -1;

  *cs = realloc(*cs, sizeof(struct u_child*) * (size+1));
  if (*cs == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: logic error cannot realloc for worker list\n", __func__);
#endif 
    return -1;
  }

  (*cs)[size] = c;
  
  return size + 1;
}

#if 0
struct u_child *fork_child_sp(struct u_server *s, int (*call)(struct u_server *s, struct spead_api_module *m, int cfd))
#endif
struct u_child *fork_child_sp(struct spead_api_module *m, void *data, int (*call)(void *data, struct spead_api_module *m, int cfd))
{
  int pipefd[2];
  pid_t cpid;

  if (call == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: null callback\n", __func__);
#endif
    return NULL;
  }
  
  if (pipe(pipefd) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: pipe fail: %s\n", __func__, strerror(errno));
#endif
    return NULL;
  } 

  cpid = fork();

  if (cpid < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: fork fail: %s\n", __func__, strerror(errno));
#endif
    return NULL;
  }

  if (cpid > 0) {
    /*in parent*/
    close(pipefd[1]); /*close write end parent*/
#if 0
    def DEBUG
    fprintf(stderr, "%s:\tnew child pid [%d]\n", __func__, cpid);
#endif

    /*child structure has child pid and read end of pipe*/
    return create_child_sp(cpid, pipefd[0]);
  }

  close(pipefd[0]); /*close read end in child*/
  
  /*setup module data*/
  if (setup_api_user_module(m) == 0){
#ifdef DEBUG
    fprintf(stderr, "%s: child [%d] has api data @ (%p)\n", __func__, getpid(), m->m_data);
#endif
  }

  /*in child use exit not return*/ 
  if ((*call)(data, m, pipefd[1]) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: child [%d] task returned error\n", __func__, getpid());
#endif
  }

  /*destroy module data*/
  if (destroy_api_user_module(m) == 0){
#ifdef DEBUG
    fprintf(stderr, "%s: child [%d] has called destroy api data\n", __func__, getpid());
#endif
  }

  exit(EX_OK);
  return NULL;
}

