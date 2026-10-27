/* runq serve: an OpenAI-compatible HTTP server on the library (runq.h). Layers, one file each:
 *   http.c       HTTP/1.1 over POSIX sockets: accept loop, one thread per connection, request parsing
 *                (Content-Length bodies, limits), responses (fixed length or chunked), shutdown
 *   serve.c      the OpenAI layer: routes, request JSON -> a Job (chat template, sampling, stop strings,
 *                response_format), the Job's output -> JSON responses / SSE chunks, errors, logging
 *   engine.c     the engine: one thread running every Job on one compute context with continuous
 *                batching (one batched pass per step over all running jobs: decode rows + prompt
 *                chunks), a pool of sequences (slots) with prefix KV reuse; the only code calling
 *                forward passes
 *   constrain.c  constrained decoding for json_schema with a finite set of outputs: schema -> canonical
 *                strings -> byte trie; per step the forced tokens or the token mask
 *   json.c       JSON parser / writer
 * serve.c and http.c do not depend on how Jobs are run (Step 10b replaced Step 9's one-job-at-a-time loop
 * with the scheduler behind the same Job interface).
 */
#ifndef RUNQ_SERVE_H
#define RUNQ_SERVE_H

#include "runq.h"
#include "json.h"

int serve_main(int argc, char **argv); // `runq serve ...` (cli.c dispatches here)

// --------------------------------------------------------------------------- constrained decoding
// A finite set of output strings (the canonical JSON texts a schema allows) and the token-level view
// of it. Decisions are masked at the byte level: a token is allowed if its bytes keep the output a
// prefix of some valid string (any tokenization of a valid string can be produced). Where only one
// continuation is possible up to the next decision, the canonical tokens of that part are forced
// (appended without a forward pass of their own).
typedef struct ConSet ConSet;     // compiled schema (immutable, shared by the requests using it)
typedef struct ConVocab ConVocab; // token bytes -> ids (built once per model)

// schema: the JSON schema object (response_format.json_schema.schema). NULL with a message on error
ConSet *con_compile(const Js *schema, char *err, size_t errlen);
void con_free(ConSet *cs);
int con_count(const ConSet *cs);                  // number of valid strings
const char *con_string(const ConSet *cs, int i, int *len);

ConVocab *con_vocab_new(const RqModel *m);
void con_vocab_free(ConVocab *v);

typedef struct {
    const ConSet *cs;
    int node;     // trie node of the output so far
    int done;     // a complete valid string
} ConState;
void con_start(ConState *st, const ConSet *cs);
// tokens forced from the current state (appended to out; state advanced over them); RQ_OK or an error
int con_forced(ConState *st, const RqModel *m, RqTokens *out);
// the allowed tokens at the current state: allow[t] = 1 for each (listed in list[0..*n)); the caller
// clears them after the step (allow[list[i]] = 0). Returns the count (0: no token fits)
int con_mask(const ConState *st, const ConVocab *v, unsigned char *allow, int *list, int maxlist);
// advance over the bytes of a sampled token (0 if they leave the valid strings)
int con_advance(ConState *st, const char *bytes, int len);

// ------------------------------------------------------------------------------------------- engine
enum { JOB_QUEUED = 0, JOB_RUNNING = 1, JOB_DONE = 2 };
enum { FINISH_NONE = 0, FINISH_STOP = 1, FINISH_LENGTH = 2 };

typedef struct Job Job;
struct Job {
    // request (filled by serve.c before job_submit; read-only for the engine)
    RqTokens prompt;            // the rendered conversation, ending with the assistant header
    int max_new;                // generated tokens at most (>= 1)
    float temperature, top_p;
    unsigned long long seed;
    int thinking;               // the reply starts in a <think> block: reasoning / content split
    const ConSet *con;          // json_schema constraint, or NULL
    char **stop;                // stop strings (content only)
    int nstop;
    int top_logprobs;           // < 0: no logprobs; else the number of alternatives per token (<= 20)
    int no_cache;               // prefill from position 0 (cache_prompt: false)
    double deadline;            // rq_now() time limit, 0 = none

    // output (written by the engine under mu; serve.c reads it under mu)
    void *mu, *cv;              // pthread mutex / condition (engine.c)
    JBuf content, reasoning;    // text so far (content: what passed the stop strings)
    JBuf logprobs;              // JSON array items (comma-separated) of the generated tokens
    int state;                  // JOB_*
    int finish;                 // FINISH_*
    int status;                 // 0, or an HTTP error status (with errmsg / errcode)
    char errmsg[300], errcode[48];
    int n_prompt, n_cached, n_gen;
    double t_submit, t_start, t_first, t_end; // rq_now()
    int cancel;                 // set by the connection (client gone): the engine stops at the next step
    int refs;                   // the submitter and the engine
    int slot;                   // the sequence (slot) used
};

typedef struct {
    const char *model_path;
    int ctx;                    // positions per sequence (slot)
    int kv_type;
    int batch;                  // prompt tokens per pass when no job is decoding (-b)
    int chunk;                  // prompt tokens per pass while other jobs decode (--chunk)
    int parallel;               // running jobs at most (--parallel; 0: 16)
    int slots;                  // sequences in the pool (running jobs + prefix caches; 0: parallel + 4)
    double kv_mem;              // GiB for the pool at most (slots x rq_seq_mem of ctx positions; 0: half the RAM, at most MemAvailable)
    int copy_min;               // copy a running job's prefix if that saves >= this many tokens (0: never)
    int queue;                  // queued jobs at most (waiting, not running)
    int verbose;
} EngineOpts;

typedef struct Engine Engine;
Engine *engine_start(RqModel *m, const EngineOpts *o, char *err, size_t errlen);
// queue the job (refs: the engine takes one); 0 = queued, -1 = queue full, -2 = shutting down
int engine_submit(Engine *e, Job *j);
void engine_stop(Engine *e, double grace); // reject the queue, let the running jobs finish (grace s), join
void engine_free(Engine *e);                // (after engine_stop, once no connection uses it)
int engine_queued(Engine *e, int *running); // waiting jobs; *running: jobs running
void engine_describe(Engine *e, char *buf, size_t n); // the effective pool / batching settings (one line)
const ConVocab *engine_vocab(Engine *e);

Job *job_new(void);
void job_release(Job *j);                    // drop a reference (freed at 0)
void job_lock(Job *j);
void job_unlock(Job *j);
// under the lock: one wait for news of at most `seconds` (monotonic; may wake early: re-check the job's
// state); 0 on timeout
int job_wait(Job *j, double seconds);

#endif
