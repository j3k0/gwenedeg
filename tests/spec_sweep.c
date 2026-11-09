/* The speculative loop's settings x prompts in one process (the library API, src/runq.h; TODO S9d, PLAN-S9-mtp.md 8):
 * the model, its MTP view (rq_mtp_load; one at a time, reloaded when a setting's --mtp-vocab differs) and optionally
 * a draft model are loaded once, then for each reasoning mode (-r), prompt and setting (-S), -R times: fresh sequences,
 * the prompt rendered as the CLI's chat renders one -i turn (rq_chat_system without a system message + rq_chat_user;
 * its assistant header a checkpoint, as there), rq_generate_spec with -n new tokens, and one JSON line (the prompt's
 * prefill is not in tok/s: n_gen / t_gen). The first setting is always plain decoding (prepended if not given): every
 * other run's tokens, stop and RNG state afterwards must equal its (the output is plain decoding's for any draft, k
 * and tau; plain's later reps too), else "match": false and exit status 1. At the end per setting: tok/s per prompt (the best of the reps),
 * the mean and x plain, accepted / drafted, tokens per pass. Built by `make spec-sweep` (build/spec_sweep); run under
 * scripts/s9_gate.sh for the 27B (S9d.2).
 *
 *   build/spec_sweep MODEL [-S SETTING]... [-P chat,code,math,summ] [-i TEXT]... [-r 1,0] [-n 256] [-c 4096] [-k f16]
 *                          [-b 512] [-t 0] [-p 0.95] [-K 0] [-s 42] [-R 1] [-D DRAFT.gguf] [--mem GiB] [--list]
 * SETTING: plain, or mtp (the model's MTP block) / draft (-D's model), then runq's options for it: -G k (default 3),
 * --tau T, --mtp-vocab N (mtp), --lookup N, --lookup-min L, --spec auto; e.g. -S "mtp -G 6 --tau 0.3", -S "draft -G 3"
 * (lookup alone: "plain --lookup 4"). Default settings: plain, "mtp -G 3". -P: built-in prompts (TODO S7's, from
 * scripts/spec_accept.py: chat TCP vs UDP, code a C linked-list reversal, math the train's average speed, summ the
 * bike-lane article in three bullet points); -i adds one of its own (custom1, ...). -r: the reasoning modes (1: on,
 * Qwen3.8's default effort; 0: off). -t / -p / -K / -s: sampling as runq's (-t 0: greedy; seeded runs equal plain's
 * too). --mem: the loads' budget in GiB, used up in turn (the model's estimate checked against it, the MTP view's
 * against what is left, the draft's against the rest; RqModelOpts.mem / RqMtpOpts.mem); not given: MemAvailable at
 * each load. --list: the plan only (no model loaded), one line "plan ctx C kv K kmax X mtp 0|1 vocabs V,.. draft
 * PATH|- runs N" for s9_gate.sh's estimate, then the runs. Threads: OMP_NUM_THREADS (else the default). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/runq.h"

static const char *ARTICLE =
    "The city council voted on Tuesday to approve a new bike lane network covering 42 kilometers of "
    "downtown streets. The plan, which will cost 18 million euros over four years, replaces roughly "
    "1,200 on-street parking spaces with protected lanes. Supporters argue it will cut traffic "
    "accidents and emissions, while shopkeepers worry about lost customers. The council also "
    "promised a 3 million euro fund to compensate affected businesses, and a review after two years. "
    "Construction begins in April on the northern avenues, with the riverside section last.";

#define PMAX 16
#define SMAX 64
#define RMAX 4

typedef struct { char name[16]; char *text; } Prompt;
typedef struct {
    char desc[160];
    int src;          // 0 plain, 1 mtp, 2 draft
    int k, vocab, lookup, lookup_min, autom;
    float tau;
} Setting;

static void die(const char *what, int e, RqCtx *c) {
    fprintf(stderr, "spec_sweep: %s: %s %s\n", what, rq_strerror(e), c && rq_ctx_error(c)[0] ? rq_ctx_error(c) : "");
    exit(1);
}

static int parse_setting(const char *s, Setting *o) {
    // "plain" / "mtp" / "draft" + runq's options for it; 0 on a bad one
    memset(o, 0, sizeof(*o));
    snprintf(o->desc, sizeof(o->desc), "%s", s);
    char b[160];
    snprintf(b, sizeof(b), "%s", s);
    char *t = strtok(b, " ");
    if (!t) return 0;
    if (!strcmp(t, "plain")) o->src = 0;
    else if (!strcmp(t, "mtp")) o->src = 1;
    else if (!strcmp(t, "draft")) o->src = 2;
    else return 0;
    o->k = o->src ? 3 : 0;
    while ((t = strtok(NULL, " "))) {
        char *v = strtok(NULL, " "), *end;
        if (!v) return 0;
        double x = strtod(v, &end);
        if (end == v || *end) {
            if (!strcmp(t, "--spec") && (!strcmp(v, "auto") || !strcmp(v, "fixed"))) { o->autom = !strcmp(v, "auto"); continue; }
            return 0;
        }
        if (!strcmp(t, "-G") && o->src && x >= 1 && x <= RQ_SPEC_K_MAX && x == (int)x) o->k = (int)x;
        else if (!strcmp(t, "--tau") && o->src && x >= 0 && x <= 1) o->tau = (float)x;
        else if (!strcmp(t, "--mtp-vocab") && o->src == 1 && x >= 0 && x == (int)x) o->vocab = (int)x;
        else if (!strcmp(t, "--lookup") && x >= 0 && x <= RQ_SPEC_K_MAX && x == (int)x) o->lookup = (int)x;
        else if (!strcmp(t, "--lookup-min") && x >= 2 && x <= RQ_LOOKUP_LMAX && x == (int)x) o->lookup_min = (int)x;
        else return 0;
    }
    if (o->autom && !o->src && !o->lookup) return 0; // (auto needs a source to choose from)
    return 1;
}

typedef struct { RqTokens out; } Collect;

static int on_token(void *user, int token) {
    Collect *c = user;
    return rq_tokens_push(&c->out, &token, 1) != RQ_OK; // (stops on a malloc failure: then a mismatch)
}

static int parse_list(const char *s, int *v, int max) {
    int n = 0;
    while (*s && n < max) {
        v[n++] = atoi(s);
        const char *c = strchr(s, ',');
        if (!c) break;
        s = c + 1;
    }
    return n;
}

static const char *stop_name(int s) {
    return s == RQ_STOP_END ? "end" : s == RQ_STOP_LIMIT ? "limit" : s == RQ_STOP_CONTEXT ? "context" : s == RQ_STOP_CALLBACK ? "callback" : "?";
}

static void print_mem(const char *what, const RqModel *m) {
    RqModelMem mm;
    if (rq_model_mem(m, &mm) != RQ_OK) return;
    const double G = 1073741824.0;
    printf("# [memory] %s estimate %.2f GiB: weights %.2f + runtime %.2f + a sequence %.2f (KV %.2f, state %.1f MiB x (1 + %d))%s\n",
           what, mm.total / G, mm.weights / G, mm.runtime / G, mm.seq.total / G, mm.seq.kv / G, mm.seq.state / 1048576.0,
           mm.seq.checkpoints, mm.budget_src == 2 ? " (--mem)" : "");
}

static double used_gib(const RqModel *m) {
    RqModelMem mm;
    return rq_model_mem(m, &mm) == RQ_OK ? mm.total / 1073741824.0 : 0;
}

int main(int argc, char **argv) {
    const char *path = NULL, *draft_path = NULL, *kvs = "f16", *plist = "chat,code,math,summ";
    const char *sargs[SMAX], *custom[PMAX];
    int ns_arg = 0, ncustom = 0, thinks[2] = { 1 }, nthink = 1, max_new = 256, ctx = 4096, batch = RQ_BATCH_MAX, reps = 1, list = 0;
    float temp = 0, topp = 0.95f;
    int top_k = 0;
    unsigned long long seed = 42;
    double mem = 0;
    int mem_given = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            if (path) goto usage;
            path = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "--list")) { list = 1; continue; }
        if (i + 1 >= argc) goto usage;
        const char *f = argv[i], *v = argv[++i];
        if (!strcmp(f, "-S") && ns_arg < SMAX) sargs[ns_arg++] = v;
        else if (!strcmp(f, "-P")) plist = v;
        else if (!strcmp(f, "-i") && ncustom < PMAX) custom[ncustom++] = v;
        else if (!strcmp(f, "-r")) nthink = parse_list(v, thinks, 2);
        else if (!strcmp(f, "-n")) max_new = atoi(v);
        else if (!strcmp(f, "-c")) ctx = atoi(v);
        else if (!strcmp(f, "-k")) kvs = v;
        else if (!strcmp(f, "-b")) batch = atoi(v);
        else if (!strcmp(f, "-t")) temp = (float)atof(v);
        else if (!strcmp(f, "-p")) topp = (float)atof(v);
        else if (!strcmp(f, "-K")) top_k = atoi(v);
        else if (!strcmp(f, "-s")) seed = strtoull(v, NULL, 10);
        else if (!strcmp(f, "-R")) reps = atoi(v);
        else if (!strcmp(f, "-D")) draft_path = v;
        else if (!strcmp(f, "--mem")) {
            char *end;
            mem = strtod(v, &end);
            if (end == v || *end || !(mem > 0)) goto usage; // (a budget > 0 only: s9_gate.sh never passes 0)
            mem_given = 1;
        } else goto usage;
    }
    int kv = rq_kv_parse(kvs);
    if (!path || kv < 0 || nthink < 1 || max_new < 1 || ctx < 64 || batch < 0 || batch > RQ_BATCH_MAX || reps < 1 || reps > RMAX || temp < 0)
        goto usage;
    for (int i = 0; i < nthink; i++)
        if (thinks[i] != 0 && thinks[i] != 1) goto usage;

    // the settings (plain first) and the prompts
    Setting set[SMAX + 1];
    int ns = 0;
    for (int i = 0; i < ns_arg; i++) {
        if (!parse_setting(sargs[i], &set[ns + 1])) { fprintf(stderr, "spec_sweep: bad setting \"%s\"\n", sargs[i]); goto usage; }
        if (set[ns + 1].src == 2 && !draft_path) { fprintf(stderr, "spec_sweep: \"%s\" needs -D DRAFT\n", sargs[i]); return 2; }
        ns++;
    }
    if (!ns) {
        parse_setting("mtp -G 3", &set[1]);
        ns = 1;
    }
    if (set[1].src == 0 && !set[1].lookup) // (plain given first: it is the reference)
        memmove(set, set + 1, ns-- * sizeof(Setting));
    else
        parse_setting("plain", &set[0]);
    ns++;
    Prompt pr[PMAX];
    int np = 0;
    {
        char b[256];
        snprintf(b, sizeof(b), "%s", plist);
        for (char *t = strtok(b, ","); t && np < PMAX; t = strtok(NULL, ",")) {
            const char *txt = !strcmp(t, "chat") ? "What are the main differences between TCP and UDP?"
                            : !strcmp(t, "code") ? "Write a C function that reverses a singly linked list in place."
                            : !strcmp(t, "math") ? "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed "
                                                   "for the whole trip? Show the steps."
                            : !strcmp(t, "summ") ? "" : NULL;
            if (!txt) { fprintf(stderr, "spec_sweep: unknown prompt %s\n", t); goto usage; }
            snprintf(pr[np].name, sizeof(pr[np].name), "%s", t);
            if (!strcmp(t, "summ")) {
                const char *h = "Summarize the text below in three bullet points.\n\n";
                pr[np].text = malloc(strlen(h) + strlen(ARTICLE) + 1);
                if (!pr[np].text) die("malloc", RQ_ERR_NOMEM, NULL);
                strcpy(pr[np].text, h);
                strcat(pr[np].text, ARTICLE);
            } else pr[np].text = strdup(txt);
            np++;
        }
        for (int i = 0; i < ncustom && np < PMAX; i++, np++) {
            snprintf(pr[np].name, sizeof(pr[np].name), "custom%d", i + 1);
            pr[np].text = strdup(custom[i]);
        }
    }
    int kmax = 0, use_mtp = 0, use_draft = 0;
    char vocabs[256] = "";
    for (int i = 0; i < ns; i++) {
        int kk = set[i].k > set[i].lookup ? set[i].k : set[i].lookup;
        kmax = kk > kmax ? kk : kmax;
        use_mtp |= set[i].src == 1;
        use_draft |= set[i].src == 2;
        if (set[i].src == 1) {
            char v[24];
            snprintf(v, sizeof(v), ",%d,", set[i].vocab);
            char have[260];
            snprintf(have, sizeof(have), ",%s,", vocabs);
            if (!strstr(have, v)) snprintf(vocabs + strlen(vocabs), sizeof(vocabs) - strlen(vocabs), "%s%d", vocabs[0] ? "," : "", set[i].vocab);
        }
    }
    int runs = nthink * np * ns * reps;
    printf("plan ctx %d kv %s kmax %d mtp %d vocabs %s draft %s runs %d\n", ctx, rq_kv_name(kv), kmax, use_mtp, vocabs[0] ? vocabs : "-",
           use_draft ? draft_path : "-", runs);
    for (int i = 0; i < ns; i++)
        printf("# setting %d: %s%s\n", i, set[i].desc, i ? "" : " (the reference)");
    fflush(stdout);
    if (list) return 0;

    // the models: the target, the draft model (if a setting uses it), the MTP view (loaded per vocabulary below)
    const char *w = rq_threads_init(0);
    if (w) fprintf(stderr, "%s\n", w);
    char err[512];
    int e;
    RqModelOpts o = { 0 };
    o.ctx_len = ctx;
    o.mem = mem_given ? mem : 0;
    o.kv_bits = rq_kv_bits(kv);
    double t0 = rq_now();
    RqModel *m = rq_model_load(path, &o, err, sizeof(err));
    if (!m) { fprintf(stderr, "spec_sweep: %s\n", err); return 1; }
    double tload = rq_now() - t0, left = mem - used_gib(m);
    RqModelInfo I;
    rq_model_info(m, &I);
    if (I.warning) fprintf(stderr, "warning: %s\n", I.warning);
    char load[64] = "";
    FILE *lf = fopen("/proc/loadavg", "r");
    if (lf) { if (!fgets(load, sizeof(load), lf)) load[0] = 0; fclose(lf); load[strcspn(load, "\n")] = 0; }
    printf("# %s: dim %d, %d layers (%d linear), vocab %d, chat %s; %d threads, ctx %d, KV %s, -b %d; load %.1f s; loadavg %s\n", path,
           I.dim, I.n_layers, I.linear_layers, I.vocab_size, rq_chat_name(I.chat), rq_threads(), I.seq_len, rq_kv_name(kv), batch,
           tload, load);
    print_mem("model", m);
    for (int i = 0; i < nthink; i++)
        if (rq_chat_check(m, thinks[i] ? RQ_THINK_ON : 0, err, sizeof(err)) != RQ_OK) { fprintf(stderr, "spec_sweep: %s\n", err); return 1; }
    RqModel *dm = NULL;
    RqCtx *dmc = NULL;
    RqSeq *dms = NULL;
    RqCtx *c = rq_ctx_new(m, batch, 0, &e);
    RqSeq *s = c ? rq_seq_new(m, 0, kv, &e) : NULL;
    if (!s) die("rq_ctx_new / rq_seq_new", e, NULL);
    RqModel *view = NULL;
    RqCtx *vc = NULL;
    RqSeq *vs = NULL;
    int view_vocab = -1;
    if (use_mtp) { // (the first vocabulary now: its estimate counted before the draft's)
        for (int i = 0; i < ns; i++)
            if (set[i].src == 1) { view_vocab = set[i].vocab; break; }
        RqMtpOpts mo = { view_vocab, mem_given ? left : 0 };
        if (mem_given && left <= 0) { fprintf(stderr, "spec_sweep: --mem %.2f GiB: nothing left for the MTP view\n", mem); return 1; }
        if (!(view = rq_mtp_load(m, &mo, err, sizeof(err)))) { fprintf(stderr, "spec_sweep: rq_mtp_load: %s\n", err); return 1; }
        print_mem("mtp", view);
        left -= used_gib(view);
        if (!(vc = rq_ctx_new(view, batch, 0, &e)) || !(vs = rq_seq_new(view, 0, kv, &e))) die("rq_ctx_new / rq_seq_new (MTP)", e, NULL);
    }
    if (use_draft) {
        RqModelOpts d = o;
        if (mem_given && left <= 0) { fprintf(stderr, "spec_sweep: --mem %.2f GiB: nothing left for the draft\n", mem); return 1; }
        d.mem = mem_given ? left : 0;
        if (!(dm = rq_model_load(draft_path, &d, err, sizeof(err)))) { fprintf(stderr, "spec_sweep: draft: %s\n", err); return 1; }
        if (rq_spec_check(m, dm, err, sizeof(err)) != RQ_OK) { fprintf(stderr, "spec_sweep: -D %s: %s\n", draft_path, err); return 1; }
        print_mem("draft", dm);
        if (!(dmc = rq_ctx_new(dm, batch, 0, &e)) || !(dms = rq_seq_new(dm, 0, kv, &e))) die("rq_ctx_new / rq_seq_new (draft)", e, NULL);
    }
    fflush(stdout);

    // the runs
    RqSampling cfg = { temp, topp, NULL, top_k };
    RqPassCost tc, dc_mtp, dc_draft;
    int have_tc = 0, have_mtp = 0, have_draft = 0, bad = 0;
    // per setting, mode and prompt: the best tok/s, its stats (for the summary)
    typedef struct { double toks, tps; int drafted, accepted, passes, n_gen, done; } Best;
    Best *best = calloc((size_t)ns * nthink * np, sizeof(Best));
    RqTokens ref[2][PMAX];
    int ref_stop[2][PMAX];
    unsigned long long ref_rng[2][PMAX];
    memset(ref, 0, sizeof(ref));
    if (!best) die("calloc", RQ_ERR_NOMEM, NULL);
    for (int ti = 0; ti < nthink; ti++) {
        int think = thinks[ti] ? RQ_THINK_ON : 0;
        for (int pi = 0; pi < np; pi++) {
            RqTokens pt = { 0 };
            if ((e = rq_chat_system(m, NULL, think, &pt)) != RQ_OK || (e = rq_chat_user(m, pr[pi].text, think, &pt)) != RQ_OK)
                die("rq_chat_system / user", e, NULL);
            int ckp = rq_chat_checkpoint_pos(m, pt.v, pt.n);
            for (int si = 0; si < ns; si++) {
                Setting *st = &set[si];
                if (st->src == 1 && st->vocab != view_vocab) { // (another vocabulary subset: the view again)
                    rq_ctx_free(vc);
                    rq_seq_free(vs);
                    rq_model_free(view);
                    RqMtpOpts mo = { st->vocab, mem_given ? mem - used_gib(m) - (dm ? used_gib(dm) : 0) : 0 };
                    if (mem_given && mo.mem <= 0) { fprintf(stderr, "spec_sweep: --mem %.2f GiB: nothing left for the MTP view\n", mem); return 1; }
                    if (!(view = rq_mtp_load(m, &mo, err, sizeof(err)))) { fprintf(stderr, "spec_sweep: rq_mtp_load: %s\n", err); return 1; }
                    if (!(vc = rq_ctx_new(view, batch, 0, &e)) || !(vs = rq_seq_new(view, 0, kv, &e))) die("rq_ctx_new / rq_seq_new (MTP)", e, NULL);
                    view_vocab = st->vocab;
                    have_mtp = 0;
                }
                RqCtx *dc = st->src == 1 ? vc : st->src == 2 ? dmc : NULL;
                RqSeq *ds = st->src == 1 ? vs : st->src == 2 ? dms : NULL;
                RqSpecOpts so;
                memset(&so, 0, sizeof(so));
                so.k = st->src ? st->k : 0;
                so.tau = st->tau;
                so.lookup = st->lookup;
                so.lookup_min = st->lookup_min;
                if (st->autom) { // (--spec auto: the cost models as the CLI calibrates them, once per model)
                    int L = rq_seq_capacity(s) - 32 < 8192 ? rq_seq_capacity(s) - 32 : 8192;
                    if (!have_tc) {
                        rq_seq_reset(s);
                        if (batch < 16 || rq_pass_cost_measure(c, s, L, 5, &tc) != RQ_OK) rq_pass_cost_default(m, &tc);
                        have_tc = 1;
                    }
                    RqPassCost *dcost = st->src == 1 ? &dc_mtp : &dc_draft;
                    int *have = st->src == 1 ? &have_mtp : &have_draft;
                    if (ds && !*have) {
                        rq_seq_reset(ds);
                        int Ld = L < rq_seq_capacity(ds) - 32 ? L : rq_seq_capacity(ds) - 32;
                        if (batch < 16 || rq_pass_cost_measure(dc, ds, Ld, 5, dcost) != RQ_OK) rq_pass_cost_default(st->src == 1 ? view : dm, dcost);
                        *have = 1;
                    }
                    so.cost = &tc;
                    so.dcost = ds ? dcost : NULL;
                }
                for (int rep = 0; rep < reps; rep++) {
                    rq_seq_reset(s);
                    rq_seq_seed(s, seed);
                    if (ckp >= 0 && (e = rq_seq_checkpoint_at(s, ckp)) != RQ_OK) die("rq_seq_checkpoint_at", e, NULL);
                    if (ds) {
                        rq_seq_reset(ds);
                        if (ckp >= 0 && ckp <= rq_seq_capacity(ds) && (e = rq_seq_checkpoint_at(ds, ckp)) != RQ_OK) die("rq_seq_checkpoint_at", e, NULL);
                    }
                    Collect col = { { 0 } };
                    RqCallbacks cb = { on_token, NULL, &col };
                    RqSpecStats ss;
                    memset(&ss, 0, sizeof(ss));
                    if ((e = rq_generate_spec(c, s, dc, ds, pt.v, pt.n, &cfg, max_new, &so, &cb, &ss)) != RQ_OK) die("rq_generate_spec", e, c);
                    unsigned long long rng = *rq_seq_rng(s);
                    int match = 1;
                    if (si == 0 && rep == 0) {
                        ref[ti][pi] = col.out;
                        ref_stop[ti][pi] = ss.gen.stop;
                        ref_rng[ti][pi] = rng;
                    } else {
                        const RqTokens *r = &ref[ti][pi];
                        match = col.out.n == r->n && !memcmp(col.out.v, r->v, (size_t)r->n * sizeof(int)) && ss.gen.stop == ref_stop[ti][pi] &&
                                rng == ref_rng[ti][pi];
                        bad += !match;
                    }
                    double tps = ss.gen.n_gen / (ss.gen.t_gen > 0 ? ss.gen.t_gen : 1e-9);
                    int passes = ss.passes ? ss.passes : ss.gen.n_gen; // (plain: one pass per token)
                    printf("{\"setting\": \"%s\", \"prompt\": \"%s\", \"think\": %d, \"rep\": %d, \"n_prompt\": %d, \"tokens\": %d, \"stop\": \"%s\", "
                           "\"passes\": %d, \"drafted\": %d, \"accepted\": %d, \"draft_steps\": %d, \"mtp_rows\": %d, \"mtp_approx\": %d, "
                           "\"lookup_passes\": %d, \"lookup_drafted\": %d, \"lookup_accepted\": %d, \"refed\": %d, \"t_prefill\": %.3f, "
                           "\"t_gen\": %.3f, \"t_draft\": %.3f, \"t_verify\": %.3f, \"tok_s\": %.2f, \"tok_per_pass\": %.3f, \"match\": %s}\n",
                           st->desc, pr[pi].name, thinks[ti], rep, ss.gen.n_prompt, ss.gen.n_gen, stop_name(ss.gen.stop), passes, ss.drafted,
                           ss.accepted, ss.draft_steps, ss.mtp_rows, ss.mtp_approx, ss.lookup_passes, ss.lookup_drafted, ss.lookup_accepted,
                           ss.refed, ss.gen.t_prefill, ss.gen.t_gen, ss.t_draft, ss.t_verify, tps,
                           (double)ss.gen.n_gen / (passes > 0 ? passes : 1), match ? "true" : "false");
                    fflush(stdout);
                    Best *b = &best[((size_t)si * nthink + ti) * np + pi];
                    if (!b->done || tps > b->tps)
                        *b = (Best){ (double)ss.gen.n_gen / (passes > 0 ? passes : 1), tps, ss.drafted, ss.accepted, passes, ss.gen.n_gen, 1 };
                    if (si || rep) rq_tokens_free(&col.out);
                }
            }
            rq_tokens_free(&pt);
        }
    }

    // the summary: per mode and setting, tok/s per prompt (best rep), the mean, x plain, accepted, tokens per pass
    for (int ti = 0; ti < nthink; ti++) {
        printf("# thinking %s: tok/s (accepted %%, tokens per pass) per prompt, the best of %d rep(s); mean; x plain\n",
               thinks[ti] ? "on" : "off", reps);
        printf("# %-36s", "setting");
        for (int pi = 0; pi < np; pi++) printf(" %-20s", pr[pi].name);
        printf(" %6s %6s\n", "mean", "x");
        double plain_mean = 0;
        for (int si = 0; si < ns; si++) {
            double sum = 0;
            printf("# %-36s", set[si].desc);
            for (int pi = 0; pi < np; pi++) {
                Best *b = &best[((size_t)si * nthink + ti) * np + pi];
                char cell[40];
                if (si) snprintf(cell, sizeof(cell), "%.1f (%.0f%%, %.2f)", b->tps, 100.0 * b->accepted / (b->drafted ? b->drafted : 1), b->toks);
                else snprintf(cell, sizeof(cell), "%.1f", b->tps);
                printf(" %-20s", cell);
                sum += b->tps;
            }
            double mean = sum / np;
            if (!si) plain_mean = mean;
            printf(" %6.1f %5.2fx\n", mean, mean / (plain_mean > 0 ? plain_mean : 1));
        }
    }
    printf("# outputs == plain: %d of %d runs%s\n", runs - nthink * np - bad, runs - nthink * np, bad ? " (MISMATCH)" : "");
    fflush(stdout);

    for (int ti = 0; ti < nthink; ti++)
        for (int pi = 0; pi < np; pi++) rq_tokens_free(&ref[ti][pi]);
    for (int pi = 0; pi < np; pi++) free(pr[pi].text);
    free(best);
    // (the views and the draft before the model: an MTP view shares its mapping, tokenizer and classifier)
    if (view) {
        rq_ctx_free(vc);
        rq_seq_free(vs);
        rq_model_free(view);
    }
    if (dm) {
        rq_ctx_free(dmc);
        rq_seq_free(dms);
        rq_model_free(dm);
    }
    rq_ctx_free(c);
    rq_seq_free(s);
    rq_model_free(m);
    return bad ? 1 : 0;
usage:
    fprintf(stderr, "usage: spec_sweep MODEL [-S SETTING]... [-P chat,code,math,summ] [-i TEXT]... [-r 1,0] [-n 256] [-c 4096] [-k f16]\n"
                    "                  [-b 512] [-t 0] [-p 0.95] [-K 0] [-s 42] [-R 1..%d] [-D DRAFT.gguf] [--mem GiB] [--list]\n"
                    "  SETTING: plain | mtp | draft, then -G k, --tau T, --mtp-vocab N, --lookup N, --lookup-min L, --spec auto\n", RMAX);
    return 2;
}
