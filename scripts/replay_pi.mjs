#!/usr/bin/env node
// Replays one recorded pi session (~/.pi/agent/sessions/*/*.jsonl) against an OpenAI-compatible server (`runq serve`)
// with pi's own code: the session's entries go through pi-coding-agent's context building (buildSessionContext,
// convertToLlm) and pi-ai's openai-completions provider (streamSimple: the request body, the SSE parsing into an
// assistant message), so each request is the one pi would send. Run by scripts/replay_sessions.py (the server, the
// report); alone:
//
//   node scripts/replay_pi.mjs --pi PI_DIR --session FILE --url http://127.0.0.1:8080/v1 --model ID --ctx 98304 \
//        [--mode force|live] [--live-tokens 512] [--reserve N] [--keep N] [--max-requests N] [--thinking LEVEL]
//
// PI_DIR: pi's install (the directory holding node_modules/@earendil-works/...). Per request, one JSON line on stdout:
// kind (turn / summary / turn_prefix), the session entry, usage, the server's timings (runq: prefilled, cached, the
// slot's length and common prefix), finish reason, wall time.
//
// The replies. --mode force (default): every reply is forced to the recorded one (runq's force_reply: the recorded
// assistant message rendered as the template renders it, fed as if generated), and the next request carries what the
// server returned for it, parsed by pi (reasoning_content, content, tool_calls): the cache holds exactly what a model
// writing in the template's format would have generated, and the round trip (the server's parse -> pi's message ->
// pi's serialization -> the server's rendering) is the real one. --mode live: the model generates up to
// --live-tokens tokens and that reply goes on in the history (the recorded tool results follow it: a session whose
// replies are the model's own, to measure how its real outputs re-serialize).
//
// Compaction: as pi with the server's context window (--ctx): after each reply (and before a new user prompt) the
// projected context is checked (shouldCompact: tokens > ctx - reserveTokens); then pi's prepareCompaction / compact
// run, their summary request(s) go to the server (forced to a recorded compaction summary of the session, or a
// synthetic one of the same format), and the compaction entry is appended as pi appends it (its system message
// checkpoint included). The recorded compaction entries (made at the original model's window) are skipped. A request
// over the context (400 context_length_exceeded) is compacted and retried once, as pi's overflow recovery.
import { pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";

const args = {};
for (let i = 2; i < process.argv.length; i++) {
    const k = process.argv[i];
    if (!k.startsWith("--")) throw new Error(`unexpected argument ${k}`);
    args[k.slice(2)] = process.argv[++i];
}
for (const k of ["pi", "session", "url", "model", "ctx"]) if (!args[k]) throw new Error(`--${k} is required`);
const mode = args.mode || "force";
const ctx = parseInt(args.ctx, 10);
const liveTokens = parseInt(args["live-tokens"] || "512", 10);
const maxRequests = parseInt(args["max-requests"] || "0", 10);
const settings = {
    enabled: true,
    reserveTokens: parseInt(args.reserve || String(Math.min(16384, Math.floor(ctx / 4))), 10),
    keepRecentTokens: parseInt(args.keep || String(Math.min(20000, Math.floor(ctx / 4))), 10),
};

const mod = (p) => import(pathToFileURL(`${args.pi}/node_modules/@earendil-works/${p}`).href);
// pi's HTTP idle timeout (settings httpIdleTimeoutMs, default 300 s: a prefill longer than that without a byte ends
// the request); --http-timeout MS, default 0: none (runq serve sends SSE keep-alive comments, older builds do not)
{
    const undici = await import(pathToFileURL(`${args.pi}/node_modules/undici/index.js`).href);
    const ms = parseInt(args["http-timeout"] || "0", 10);
    undici.setGlobalDispatcher(new undici.Agent({ headersTimeout: ms, bodyTimeout: ms }));
    undici.install?.(); // (fetch on the same undici, as pi does)
}
const sm = await mod("pi-coding-agent/dist/core/session-manager.js");
const { convertToLlm } = await mod("pi-coding-agent/dist/core/messages.js");
const cmp = await mod("pi-coding-agent/dist/core/compaction/compaction.js");
const { streamSimple } = await mod("pi-ai/dist/api/openai-completions.js");
const { getCurrentSystemMessage } = await mod("pi-ai/dist/utils/transcript.js");

const model = {
    id: args.model, name: args.model, api: "openai-completions", provider: "runq", baseUrl: args.url, reasoning: true,
    input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: ctx,
    maxTokens: mode === "live" ? liveTokens : ctx,
    // docs/server.md's provider entry
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: true, supportsStore: false, maxTokensField: "max_tokens" },
};
const sessionId = randomUUID();

// ------------------------------------------------------------------------------------------- the session

const all = sm.loadEntriesFromFile(args.session);
sm.migrateSessionEntries(all);
const entries = all.filter((e) => e.type !== "session" && e.id);
const byId = new Map(entries.map((e) => [e.id, e]));
const path = []; // the active branch, root first (the leaf: the last entry, as pi resumes it)
for (let e = entries[entries.length - 1]; e; e = e.parentId ? byId.get(e.parentId) : undefined) path.unshift(e);
const summaries = entries.filter((e) => e.type === "compaction" && e.summary).map((e) => e.summary.split("\n<read-files>")[0].split("\n<modified-files>")[0]);

const textOf = (content) => (typeof content === "string" ? content : (content || []).filter((b) => b.type === "text").map((b) => b.text).join(""));

function forcedReply(msg) {
    // a recorded assistant message as an OpenAI message (what the model would have written: runq renders it)
    const thinking = msg.content.filter((b) => b.type === "thinking" && !b.redacted).map((b) => b.thinking).join("\n");
    const calls = msg.content.filter((b) => b.type === "toolCall").map((b) => ({
        type: "function", function: { name: b.name, arguments: JSON.stringify(b.arguments ?? {}) } }));
    return { content: msg.content.filter((b) => b.type === "text").map((b) => b.text).join(""), reasoning_content: thinking,
             ...(calls.length ? { tool_calls: calls } : {}) };
}

function syntheticSummary(prep) {
    // pi's summary format, ~4-5k characters, for sessions with no recorded compaction
    const first = path.find((e) => e.type === "message" && e.message.role === "user");
    const goal = first ? textOf(first.message.content).slice(0, 1200) : "Continue the task.";
    const files = [...(prep.fileOps?.read ?? []), ...(prep.fileOps?.edited ?? []), ...(prep.fileOps?.written ?? [])].slice(0, 40);
    return `## Goal\n${goal}\n\n## Constraints & Preferences\n- Keep the existing code style.\n- Run the tests after each change.\n\n` +
        `## Progress\n### Done\n${files.map((f) => `- [x] Reviewed ${f}`).join("\n")}\n\n### In Progress\n- [ ] Finish the current change.\n\n` +
        `### Blocked\n- None\n\n## Key Decisions\n- **Approach**: follow the plan agreed with the user.\n\n## Next Steps\n1. Continue the work.\n` +
        `2. Verify the result.\n\n## Critical Context\n${files.map((f) => `- ${f}: relevant to the task`).join("\n")}`;
}

// ------------------------------------------------------------------------------------------- requests

let seq = 0, stopAll = false, after = null, afterCompaction = false;
const out = (o) => process.stdout.write(JSON.stringify(o) + "\n");

function streamFn(kind, entry, force) {
    // pi's streamSimple with the force_reply hook and the raw SSE captured (the server's timings)
    return (m, context, options) => {
        const rec = { seq: seq++, kind, entry, msgs: context.messages.length, ...(kind === "turn" ? { after, after_compaction: afterCompaction } : {}) };
        let capture = Promise.resolve("");
        const t0 = Date.now();
        const fetchHook = async (url, init) => {
            const res = await fetch(url, init);
            capture = res.clone().text().catch(() => "");
            return res;
        };
        const s = streamSimple(m, context, {
            ...options, apiKey: "none",
            onPayload: (params) => (force !== undefined && mode === "force" ? { ...params, force_reply: force } : params),
            fetch: fetchHook,
        });
        const result = s.result.bind(s);
        s.result = async () => {
            const msg = await result();
            const raw = await capture;
            rec.wall_ms = Date.now() - t0;
            for (const line of raw.split("\n")) {
                if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
                try {
                    const d = JSON.parse(line.slice(6));
                    if (d.usage) rec.usage = d.usage;
                    if (d.timings) rec.timings = d.timings;
                    if (d.error) rec.error = d.error;
                } catch { /* not JSON */ }
            }
            if (!raw.startsWith("data:") && raw) try { rec.error = JSON.parse(raw).error; } catch { rec.error = { message: raw.slice(0, 300) }; }
            rec.stop = msg.stopReason;
            if (msg.errorMessage) rec.error_message = msg.errorMessage.slice(0, 300);
            out(rec);
            if (maxRequests && seq >= maxRequests) stopAll = true;
            return msg;
        };
        return s;
    };
}

let replayed = []; // the session as replayed so far (entries; assistant messages substituted, compactions simulated)
let thinking = "medium";
let nsum = 0;

async function maybeCompact(reason) {
    // pi's threshold check (agent-session: the projected context against contextWindow - reserveTokens), then compact()
    const projection = sm.buildSessionProjection(replayed);
    const tokens = cmp.estimateProjectedContextTokens(projection, replayed).tokens;
    if (reason !== "overflow" && !cmp.shouldCompact(tokens, ctx, settings)) return false;
    const prep = cmp.prepareCompaction(replayed, settings);
    if (!prep) return false;
    const text = summaries.length ? summaries[nsum++ % summaries.length] : syntheticSummary(prep);
    const fn = (m, c, o) => {
        // (the turn-prefix summary of a split turn is a second request)
        const isPrefix = c.messages[0] && textOf(c.messages[c.messages.length - 1].content).startsWith("# Conversation");
        return streamFn(isPrefix ? "turn_prefix" : "summary", replayed[replayed.length - 1]?.id, { content: isPrefix ? text.slice(0, 1500) : text, reasoning_content: "" })(m, c, o);
    };
    let res;
    try {
        res = await cmp.compact(prep, model, "none", undefined, undefined, undefined, thinking, fn, undefined, { maxRetries: 0 }, undefined, sessionId);
    } catch (err) {
        out({ seq: seq++, kind: "compaction_failed", reason, error_message: String(err).slice(0, 300) });
        return false;
    }
    const last = replayed[replayed.length - 1];
    const id = randomUUID().slice(0, 8);
    const systemMessage = getCurrentSystemMessage(sm.buildSessionProjection(replayed).messages);
    replayed.push({ type: "compaction", id, parentId: last.id, timestamp: new Date().toISOString(), summary: res.summary,
                    firstKeptEntryId: res.firstKeptEntryId, tokensBefore: res.tokensBefore, details: res.details,
                    ...(systemMessage ? { systemMessage } : {}) });
    out({ seq: seq++, kind: "compaction", reason, tokens_before: res.tokensBefore, estimate: tokens });
    afterCompaction = true;
    return true;
}

async function request(entry) {
    // the assistant reply of `entry`: the request pi sends for it, the reply as the server returns it
    const rec = entry.message;
    for (let attempt = 0; attempt < 2; attempt++) {
        const context = { messages: convertToLlm(sm.buildSessionContext(replayed).messages) };
        const s = streamFn("turn", entry.id, forcedReply(rec))(model, context, { reasoning: thinking === "off" ? undefined : thinking, sessionId });
        const msg = await s.result();
        if (msg.stopReason === "error" && /context|maximum|too long/i.test(msg.errorMessage || "") && attempt === 0) {
            if (await maybeCompact("overflow")) continue;
        }
        if (msg.stopReason === "error") return null;
        if (rec.stopReason === "aborted" || rec.stopReason === "error") msg.stopReason = rec.stopReason; // (pi drops those)
        return msg;
    }
    return null;
}

let prevRole = null;
for (const e of path) {
    if (stopAll) break;
    if (e.type === "compaction") continue; // (the original model's window; simulated below at ours)
    if (e.type === "thinking_level_change") thinking = e.thinkingLevel;
    if (e.type === "message" && e.message.role === "assistant") {
        if (prevRole === "toolResult" || prevRole === "user") await maybeCompact("threshold");
        after = prevRole;
        const msg = await request(e);
        afterCompaction = false;
        if (!msg) {
            out({ seq: seq++, kind: "stopped", entry: e.id });
            break;
        }
        replayed.push({ ...e, parentId: replayed.length ? replayed[replayed.length - 1].id : null, message: msg });
        prevRole = "assistant";
        continue;
    }
    if (e.type === "message" && e.message.role === "user" && prevRole === "assistant") await maybeCompact("threshold");
    replayed.push({ ...e, parentId: replayed.length ? replayed[replayed.length - 1].id : null });
    if (e.type === "message") prevRole = e.message.role;
}
