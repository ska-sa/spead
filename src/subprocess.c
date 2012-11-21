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

#include "server.h"
#include "spead_api.h"

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

void destroy_child_sp(struct u_child *c)
{
  if (c){
    
    if (c->c_fd)
      close(c->c_fd);
    
    kill(c->c_pid, SIGTERM); 
    
    free(c);
  }
}

struct u_child *fork_child_sp(struct u_server *s, int (*call)(struct u_server *s, struct spead_api_module *m, int cfd))
{
  int pipefd[2];
  pid_t cpid;
  struct spead_api_module *m;

  m = NULL;

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
  if (s){
    m = s->s_mod;
    if (setup_api_user_module(m) == 0){
#ifdef DEBUG
      fprintf(stderr, "%s: child [%d] has api data @ (%p)\n", __func__, getpid(), m->m_data);
#endif
    }

#if 0
    if (m){
      if (m->m_setup){
        m->m_data = (*m->m_setup)();
#ifdef DEBUG
        fprintf(stderr, "%s: child [%d] has api data @ (%p)\n", __func__, getpid(), m->m_data);
#endif
      }
    }
#endif
  }

  /*in child use exit not return*/ 
  (*call)(s, m, pipefd[1]);

  /*destroy module data*/
  if (destroy_api_user_module(m) == 0){
#ifdef DEBUG
    fprintf(stderr, "%s: child [%d] has called destroy api data\n", __func__, getpid());
#endif
  }
#if 0
  if (m){
    if (m->m_destroy){
      (*m->m_destroy)(m->m_data);
#ifdef DEBUG
      fprintf(stderr, "%s: child [%d] has called destroy api data\n", __func__, getpid());
#endif
    }
  }
#endif

  exit(EX_OK);
  return NULL;
}

