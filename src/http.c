/* HTTP/1.1 server for runq serve (http.h). POSIX only (serve.c reports the server as unavailable on
 * Windows). */
#ifndef _WIN32
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include "runq.h"
#include "http.h"

#ifndef POLLRDHUP
#define POLLRDHUP 0
#endif

struct HttpConn {
    int fd;
    char peer[64];
    char *buf;           // received bytes not consumed yet (the next request's start)
    size_t n, cap;
    int chunked;         // a chunked response is open
    int dead;            // a write failed: the connection is closed after the request
    const HttpOpts *o;
    HttpHandler h;
    void *app;
};

static int g_stop = 0; // (set by the signal handler, read by every thread: atomic loads / stores, lock-free)
static int g_pipe[2] = { -1, -1 };
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static int g_conns = 0;

static int stopping(void) { return __atomic_load_n(&g_stop, __ATOMIC_ACQUIRE); }
int http_shutting_down(void) { return stopping(); }

static void on_signal(int sig) {
    (void)sig;
    if (stopping()) _exit(130); // (a second signal: no more grace)
    __atomic_store_n(&g_stop, 1, __ATOMIC_RELEASE);
    if (g_pipe[1] >= 0) {
        ssize_t r = write(g_pipe[1], "x", 1);
        (void)r;
    }
}

int http_signals(void) {
    if (pipe(g_pipe)) return -1;
    fcntl(g_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(g_pipe[1], F_SETFL, O_NONBLOCK);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &set, NULL); // (the threads created from here on inherit it)
    return 0;
}

const char *http_reason(int s) {
    switch (s) {
        case 100: return "Continue";
        case 200: return "OK";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 414: return "URI Too Long";
        case 417: return "Expectation Failed";
        case 431: return "Request Header Fields Too Large";
        case 499: return "Client Closed Request";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        case 504: return "Gateway Timeout";
        case 505: return "HTTP Version Not Supported";
        default: return "Error";
    }
}

// ------------------------------------------------------------------------------------------- I/O

static int write_all(HttpConn *c, const char *s, size_t n) {
    if (c->dead) return -1;
    while (n > 0) {
        ssize_t w = send(c->fd, s, n, MSG_NOSIGNAL);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) {
            c->dead = 1;
            return -1;
        }
        s += w;
        n -= (size_t)w;
    }
    return 0;
}

static int fill(HttpConn *c, double deadline, size_t limit) {
    // read more bytes into buf (at most up to limit buffered); 1 = got some, 0 = closed, -1 = error /
    // timeout / shutdown (checked every 200 ms)
    if (c->n >= limit) return -1;
    if (c->cap < limit + 1) {
        char *b = realloc(c->buf, limit + 1);
        if (!b) return -1;
        c->buf = b;
        c->cap = limit + 1;
    }
    for (;;) {
        double left = deadline - rq_now();
        if (left <= 0) return -2;
        struct pollfd p = { c->fd, POLLIN, 0 };
        int r = poll(&p, 1, left > 0.2 ? 200 : (int)(left * 1000) + 1);
        if (r < 0 && errno != EINTR) return -1;
        if (r <= 0) {
            if (stopping()) return -3;
            continue;
        }
        ssize_t k = recv(c->fd, c->buf + c->n, limit - c->n, 0);
        if (k < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (k < 0) return -1;
        if (k == 0) return 0;
        c->n += (size_t)k;
        return 1;
    }
}

int http_peer_gone(HttpConn *c) {
    if (c->dead) return 1;
    struct pollfd p = { c->fd, POLLIN | POLLRDHUP, 0 };
    if (poll(&p, 1, 0) <= 0) return 0;
    if (p.revents & (POLLHUP | POLLERR | POLLRDHUP)) return 1;
    if (p.revents & POLLIN) {
        char b;
        ssize_t k = recv(c->fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
        return k == 0 || (k < 0 && errno != EAGAIN && errno != EINTR);
    }
    return 0;
}

static void hdr_common(char *h, size_t cap, HttpReq *r, int status, const char *ctype) {
    snprintf(h, cap, "HTTP/1.1 %d %s\r\nServer: runq\r\nContent-Type: %s\r\nConnection: %s\r\n", status,
             http_reason(status), ctype, r && r->keep_alive ? "keep-alive" : "close");
}

int http_respond(HttpConn *c, HttpReq *r, int status, const char *ctype, const char *body, size_t n, const char *extra) {
    char h[1024];
    hdr_common(h, sizeof(h), r, status, ctype);
    size_t hl = strlen(h);
    snprintf(h + hl, sizeof(h) - hl, "%sContent-Length: %zu\r\n\r\n", extra ? extra : "", n);
    if (write_all(c, h, strlen(h))) return -1;
    if (r && !strcmp(r->method, "HEAD")) return 0;
    return n ? write_all(c, body, n) : 0;
}

int http_stream_start(HttpConn *c, HttpReq *r, int status, const char *ctype) {
    char h[1024];
    hdr_common(h, sizeof(h), r, status, ctype);
    size_t hl = strlen(h);
    snprintf(h + hl, sizeof(h) - hl, "Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n");
    c->chunked = 1;
    return write_all(c, h, strlen(h));
}

int http_stream_write(HttpConn *c, const char *s, size_t n) {
    if (n == 0) return 0;
    char h[32];
    snprintf(h, sizeof(h), "%zx\r\n", n);
    if (write_all(c, h, strlen(h)) || write_all(c, s, n) || write_all(c, "\r\n", 2)) return -1;
    return 0;
}

int http_stream_end(HttpConn *c) {
    c->chunked = 0;
    return write_all(c, "0\r\n\r\n", 5);
}

// -------------------------------------------------------------------------------------- requests

static void send_error(HttpConn *c, int status, const char *msg) {
    // a plain error response before a request could be parsed (the connection is closed after it)
    char body[512];
    int n = snprintf(body, sizeof(body), "{\"error\": {\"message\": \"%s\", \"type\": \"invalid_request_error\", \"param\": null, \"code\": null}}", msg);
    HttpReq r;
    memset(&r, 0, sizeof(r));
    strcpy(r.method, "GET");
    http_respond(c, &r, status, "application/json", body, (size_t)n, NULL);
}

static char *find_head_end(char *b, size_t n, size_t *hl) {
    // end of the request head: \r\n\r\n (or \n\n); *hl = the head's length including it
    for (size_t i = 0; i + 1 < n; i++) {
        if (b[i] == '\n' && b[i + 1] == '\n') { *hl = i + 2; return b + i; }
        if (i + 3 < n && b[i] == '\r' && b[i + 1] == '\n' && b[i + 2] == '\r' && b[i + 3] == '\n') { *hl = i + 4; return b + i; }
    }
    return NULL;
}

static int token_char(int c) {
    return c > 32 && c < 127 && !strchr("()<>@,;:\\\"/[]?={}", c);
}

// read and parse one request; 1 = ok (r filled), 0 = connection done (closed / idle / shutdown), -1 =
// error response sent (close)
static int read_request(HttpConn *c, HttpReq *r) {
    const HttpOpts *o = c->o;
    memset(r, 0, sizeof(*r));
    snprintf(r->peer, sizeof(r->peer), "%s", c->peer);
    size_t hl = 0;
    double deadline = rq_now() + o->idle_timeout;
    int started = c->n > 0;
    while (!find_head_end(c->buf ? c->buf : "", c->n, &hl)) {
        if (c->n >= o->max_header) { send_error(c, 431, "request head too large"); return -1; }
        int k = fill(c, deadline, o->max_header);
        if (k == 0) return 0;                   // closed
        if (k < 0) {
            if (started && k == -2) send_error(c, 408, "timeout reading the request");
            return started && k == -2 ? -1 : 0;
        }
        if (!started) {
            started = 1;
            deadline = rq_now() + o->read_timeout;
        }
    }
    char *h = c->buf;
    h[hl - 1] = 0;
    // request line: METHOD SP target SP HTTP/1.x
    char *line_end = strchr(h, '\n');
    if (!line_end) { send_error(c, 400, "bad request line"); return -1; }
    *line_end = 0;
    if (line_end > h && line_end[-1] == '\r') line_end[-1] = 0;
    char *sp1 = strchr(h, ' '), *sp2 = sp1 ? strchr(sp1 + 1, ' ') : NULL;
    if (!sp1 || !sp2 || sp1 == h || sp2 == sp1 + 1 || strchr(sp2 + 1, ' ')) { send_error(c, 400, "bad request line"); return -1; }
    if ((size_t)(sp1 - h) >= sizeof(r->method)) { send_error(c, 501, "unknown method"); return -1; }
    for (char *p = h; p < sp1; p++)
        if (!token_char((unsigned char)*p)) { send_error(c, 400, "bad method"); return -1; }
    memcpy(r->method, h, sp1 - h);
    if ((size_t)(sp2 - sp1 - 1) >= sizeof(r->target)) { send_error(c, 414, "URI too long"); return -1; }
    memcpy(r->target, sp1 + 1, sp2 - sp1 - 1);
    for (char *p = r->target; *p; p++)
        if ((unsigned char)*p <= 32 || (unsigned char)*p >= 127) { send_error(c, 400, "bad request target"); return -1; }
    if (strncmp(sp2 + 1, "HTTP/1.", 7) || !sp2[8] || sp2[8] < '0' || sp2[8] > '9' || sp2[9]) {
        send_error(c, strncmp(sp2 + 1, "HTTP/", 5) ? 400 : 505, "unsupported protocol version");
        return -1;
    }
    r->minor = sp2[8] - '0';
    snprintf(r->path, sizeof(r->path), "%.*s", (int)strcspn(r->target, "?#"), r->target);
    r->keep_alive = r->minor >= 1;
    // headers
    long long clen = -1;
    int expect_continue = 0, te = 0;
    for (char *p = line_end + 1; *p;) {
        char *e = strchr(p, '\n');
        if (!e) e = p + strlen(p);
        char *next = *e ? e + 1 : e;
        *e = 0;
        if (e > p && e[-1] == '\r') e[-1] = 0;
        if (!*p) break;
        if (*p == ' ' || *p == '\t') { send_error(c, 400, "obsolete header folding"); return -1; }
        char *colon = strchr(p, ':');
        if (!colon || colon == p) { send_error(c, 400, "bad header line"); return -1; }
        for (char *q = p; q < colon; q++)
            if (!token_char((unsigned char)*q)) { send_error(c, 400, "bad header name"); return -1; }
        *colon = 0;
        char *v = colon + 1;
        while (*v == ' ' || *v == '\t') v++;
        size_t vl = strlen(v);
        while (vl > 0 && (v[vl - 1] == ' ' || v[vl - 1] == '\t')) v[--vl] = 0;
        if (!strcasecmp(p, "Content-Length")) {
            char *end;
            if (!*v || strspn(v, "0123456789") != vl || vl > 15) { send_error(c, 400, "bad Content-Length"); return -1; }
            long long x = strtoll(v, &end, 10);
            if (clen >= 0 && x != clen) { send_error(c, 400, "conflicting Content-Length headers"); return -1; }
            clen = x;
        } else if (!strcasecmp(p, "Transfer-Encoding")) {
            te = 1;
        } else if (!strcasecmp(p, "Connection")) {
            if (strcasestr(v, "close")) r->keep_alive = 0;
            else if (strcasestr(v, "keep-alive")) r->keep_alive = 1;
        } else if (!strcasecmp(p, "Expect")) {
            if (!strcasecmp(v, "100-continue")) expect_continue = 1;
            else { send_error(c, 417, "unsupported Expect"); return -1; }
        } else if (!strcasecmp(p, "Authorization")) {
            snprintf(r->auth, sizeof(r->auth), "%s", v);
        }
        p = next;
    }
    // the head is consumed; keep what follows it
    memmove(c->buf, c->buf + hl, c->n - hl);
    c->n -= hl;
    if (te) { send_error(c, 411, "chunked request bodies are not supported: send a Content-Length"); return -1; }
    if (clen < 0) clen = 0;
    if ((unsigned long long)clen > o->max_body) { send_error(c, 413, "request body too large"); return -1; }
    if (clen > 0) {
        if (expect_continue && c->n == 0 && write_all(c, "HTTP/1.1 100 Continue\r\n\r\n", 25)) return 0;
        r->body = malloc((size_t)clen + 1);
        if (!r->body) { send_error(c, 500, "out of memory"); return -1; }
        size_t got = c->n < (size_t)clen ? c->n : (size_t)clen;
        memcpy(r->body, c->buf, got);
        memmove(c->buf, c->buf + got, c->n - got);
        c->n -= got;
        double dl = rq_now() + o->read_timeout;
        while (got < (size_t)clen) {
            size_t want = (size_t)clen - got;
            size_t lim = want < 65536 ? want : 65536;
            int k = fill(c, dl, lim);
            if (k <= 0) {
                if (k == -2) send_error(c, 408, "timeout reading the request body");
                free(r->body);
                r->body = NULL;
                return k == -2 ? -1 : 0;
            }
            memcpy(r->body + got, c->buf, c->n);
            got += c->n;
            c->n = 0;
        }
        r->body[clen] = 0;
        r->body_len = (size_t)clen;
    }
    r->t0 = rq_now();
    return 1;
}

static void *conn_main(void *arg) {
    HttpConn *c = arg;
    for (;;) {
        HttpReq r;
        int k = read_request(c, &r);
        if (k <= 0) break;
        if (stopping()) r.keep_alive = 0;
        c->h(c->app, c, &r);
        free(r.body);
        if (!r.keep_alive || c->dead || stopping()) break;
    }
    shutdown(c->fd, SHUT_WR);
    close(c->fd);
    free(c->buf);
    free(c);
    pthread_mutex_lock(&g_mu);
    g_conns--;
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
    return NULL;
}

int http_listen(const HttpOpts *o, int *bound_port, char *err, size_t errlen) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
    char port[16];
    snprintf(port, sizeof(port), "%d", o->port);
    int rc = getaddrinfo(o->host, port, &hints, &res);
    if (rc) {
        snprintf(err, errlen, "cannot resolve %s: %s", o->host, gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *a = res; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, a->ai_addr, a->ai_addrlen) == 0 && listen(fd, 128) == 0) break;
        snprintf(err, errlen, "cannot listen on %s:%d: %s", o->host, o->port, strerror(errno));
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) return -1;
    struct sockaddr_storage ss;
    socklen_t sl = sizeof(ss);
    *bound_port = o->port;
    if (getsockname(fd, (struct sockaddr *)&ss, &sl) == 0)
        *bound_port = ss.ss_family == AF_INET6 ? ntohs(((struct sockaddr_in6 *)&ss)->sin6_port) : ntohs(((struct sockaddr_in *)&ss)->sin_port);
    return fd;
}

void http_run(int lfd, const HttpOpts *o, HttpHandler h, void *app) {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    pthread_sigmask(SIG_UNBLOCK, &set, NULL);
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&at, 256 * 1024);
    while (!stopping()) {
        struct pollfd p[2] = { { lfd, POLLIN, 0 }, { g_pipe[0], POLLIN, 0 } };
        int r = poll(p, 2, 1000);
        if (r <= 0) continue;
        if (!(p[0].revents & POLLIN)) continue;
        struct sockaddr_storage ss;
        socklen_t sl = sizeof(ss);
        int fd = accept4(lfd, (struct sockaddr *)&ss, &sl, SOCK_CLOEXEC);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        struct timeval tv = { 30, 0 };
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        HttpConn *c = calloc(1, sizeof(HttpConn));
        if (!c) { close(fd); continue; }
        c->fd = fd;
        c->o = o;
        c->h = h;
        c->app = app;
        char a[INET6_ADDRSTRLEN] = "?";
        int port = 0;
        if (ss.ss_family == AF_INET) {
            inet_ntop(AF_INET, &((struct sockaddr_in *)&ss)->sin_addr, a, sizeof(a));
            port = ntohs(((struct sockaddr_in *)&ss)->sin_port);
        } else if (ss.ss_family == AF_INET6) {
            inet_ntop(AF_INET6, &((struct sockaddr_in6 *)&ss)->sin6_addr, a, sizeof(a));
            port = ntohs(((struct sockaddr_in6 *)&ss)->sin6_port);
        }
        snprintf(c->peer, sizeof(c->peer), "%s:%d", a, port);
        pthread_mutex_lock(&g_mu);
        int full = g_conns >= o->max_conns;
        if (!full) g_conns++;
        pthread_mutex_unlock(&g_mu);
        pthread_t th;
        pthread_sigmask(SIG_BLOCK, &set, NULL); // (the connection threads never take the signals)
        int failed = full || pthread_create(&th, &at, conn_main, c);
        pthread_sigmask(SIG_UNBLOCK, &set, NULL);
        if (failed) {
            if (!full) {
                pthread_mutex_lock(&g_mu);
                g_conns--;
                pthread_mutex_unlock(&g_mu);
            }
            send_error(c, 503, "too many connections");
            close(fd);
            free(c);
        }
    }
    pthread_attr_destroy(&at);
    close(lfd);
}

int http_wait_conns(double seconds) {
    double end = rq_now() + seconds;
    pthread_mutex_lock(&g_mu);
    while (g_conns > 0 && rq_now() < end) {
        pthread_mutex_unlock(&g_mu);
        usleep(20000);
        pthread_mutex_lock(&g_mu);
    }
    int ok = g_conns == 0;
    pthread_mutex_unlock(&g_mu);
    return ok;
}
#endif
