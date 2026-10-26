/* A small HTTP/1.1 server over POSIX sockets for runq serve (http.c): an accept loop, one thread per
 * connection (at most max_conns; more get a 503), keep-alive (HTTP/1.1 default, `Connection: close`
 * honoured, HTTP/1.0 keep-alive on request), request bodies by Content-Length only (chunked uploads get
 * 411), limits on the header and body sizes (431 / 413), timeouts for a slow or idle client, responses
 * with a Content-Length or chunked (streaming), `Expect: 100-continue`, shutdown on SIGINT / SIGTERM.
 * Not a general web server: no TLS, no pipelining guarantees beyond sequential requests, no ranges.
 */
#ifndef RUNQ_HTTP_H
#define RUNQ_HTTP_H

#include <stddef.h>

typedef struct {
    char method[16];
    char target[2048];   // as received (path + query)
    char path[2048];     // without the query string
    int minor;           // HTTP/1.<minor>
    int keep_alive;      // the connection stays open after the response
    char auth[512];      // Authorization header
    char *body;          // Content-Length bytes (NUL-terminated), or NULL
    size_t body_len;
    char peer[64];       // client address
    double t0;           // rq_now() when the request was complete
} HttpReq;

typedef struct HttpConn HttpConn;
typedef void (*HttpHandler)(void *app, HttpConn *c, HttpReq *r);

typedef struct {
    const char *host;
    int port;            // 0: any free port (printed)
    size_t max_header;   // bytes of the request line + headers
    size_t max_body;
    int max_conns;
    double read_timeout; // seconds for a request to arrive completely once started
    double idle_timeout; // seconds a keep-alive connection may stay idle
} HttpOpts;

// Binds and listens (the address and the port actually bound in *bound_port). Returns a listening
// socket or -1 with err
int http_listen(const HttpOpts *o, int *bound_port, char *err, size_t errlen);
// The accept loop on the socket until SIGINT / SIGTERM (http_signals must have been installed)
void http_run(int lfd, const HttpOpts *o, HttpHandler h, void *app);
// Installs the SIGINT / SIGTERM handlers and blocks those signals in the calling thread (call it
// before creating any other thread; http_run unblocks them in its own thread only)
int http_signals(void);
int http_shutting_down(void);
// After http_run returned: wait until every connection thread has ended (1) or the timeout (0)
int http_wait_conns(double seconds);

// One complete response (Content-Length). extra: more header lines, each ending in \r\n, or NULL
int http_respond(HttpConn *c, HttpReq *r, int status, const char *ctype, const char *body, size_t n, const char *extra);
// A streamed response (Transfer-Encoding: chunked): start, chunks, end
int http_stream_start(HttpConn *c, HttpReq *r, int status, const char *ctype);
int http_stream_write(HttpConn *c, const char *s, size_t n);
int http_stream_end(HttpConn *c);
int http_peer_gone(HttpConn *c);    // the client closed the connection (non-blocking check)
const char *http_reason(int status);

#endif
