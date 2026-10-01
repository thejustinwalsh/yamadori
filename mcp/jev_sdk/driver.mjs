// TypeSafe's JavaScript SDK as a client of the Jev API. Run by
// mcp/test_jev_sdk.py with node; the SDK is imported from the pinned install
// in tools/typesafe-sdk/js/node_modules (--sdk-dir), never a global one.
//
//   node driver.mjs --base-url URL --plan PLAN.json --sdk-dir DIR
//
// The key comes from TYPESAFE_API_KEY (the SDK's own variable) and is never
// printed. One JSON line per step on stdout; each step catches its own
// failure. The harness reads the header `x-jev-test` (its scenario).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, v, i, a) => (v.startsWith("--") ? [...acc, [v.slice(2), a[i + 1]]] : acc), []),
);
const plan = JSON.parse(readFileSync(args.plan, "utf8"));
const live = Boolean(plan.live);
const sdkPkg = join(args["sdk-dir"], "node_modules", "@typesafe-ai", "sdk");
const sdk = await import(pathToFileURL(join(sdkPkg, "dist", "index.mjs")).href);
const { TypeSafeClient } = sdk;

const out = (rec) => process.stdout.write(JSON.stringify(rec) + "\n");
const err = (e) => {
  const chain = [];
  for (let p = Object.getPrototypeOf(e); p && p.constructor && p.constructor !== Object; p = Object.getPrototypeOf(p)) {
    chain.push(p.constructor.name);
  }
  return {
    error_class: e?.constructor?.name ?? null,
    error_mro: chain,
    message: String(e?.message ?? e).slice(0, 600),
    status: e?.status ?? null,
    request_id: e?.requestId ?? null,
    retry_after_ms: e?.retryAfterMs ?? null,
    body: e?.body ?? null,
  };
};
const step = async (name, fn) => {
  const t0 = Date.now();
  let rec;
  try {
    rec = await fn();
    rec.ok = true;
  } catch (e) {
    rec = { ...err(e), ok: false, trace: String(e?.stack ?? "").split("\n").slice(0, 3) };
  }
  rec.step = name;
  rec.seconds = (Date.now() - t0) / 1000;
  out(rec);
};
const withResponse = async (p) => {
  if (typeof p.withResponse === "function") {
    const w = await p.withResponse();
    return { data: w.data, requestId: w.requestId, header: w.response?.headers?.get("x-typesafe-request-id") ?? null };
  }
  return { data: await p, requestId: null, header: null, note: "APIPromise has no withResponse()" };
};

const pj = JSON.parse(readFileSync(join(sdkPkg, "package.json"), "utf8"));
out({ step: "sdk", ok: true, sdk: "js", version: pj.version, VERSION: sdk.VERSION ?? null, node: process.version });
const client = new TypeSafeClient({ baseURL: args["base-url"], logLevel: "off" });

await step("models", async () => {
  const w = await withResponse(client.models.list({ headers: { "x-jev-test": "models" } }));
  const list = Array.isArray(w.data) ? w.data : w.data?.models;
  return {
    class: Array.isArray(w.data) ? "Array" : typeof w.data,
    models: (list ?? []).map((m) => ({ name: m.name, description: m.description, release_date: m.release_date })),
    request_id: w.requestId,
    header_request_id: w.header,
  };
});

for (const [i, ex] of plan.examples.entries()) {
  const req = ex.request;
  await step(`example:${i}`, async () => {
    const w = await withResponse(
      client.systemOne(
        { state: req.state, questions: req.questions, model: req.model },
        { headers: { "x-jev-test": `example:${i}` } },
      ),
    );
    return { i, model: w.data.model, usage: w.data.usage, answers: w.data.answers, request_id: w.requestId, header_request_id: w.header };
  });
}

await step("invalid_422", async () => {
  await client.systemOne(
    { state: "state", questions: { q: { type: "score", instructions: "Level?", criteria: Array.from({ length: 11 }, (_, n) => `level ${n}`) } } },
    { headers: { "x-jev-test": "invalid" }, retry: { maxRetries: 0 } },
  );
  return { error_class: null, note: "no exception raised" };
});

if (!live) {
  const q = { is_urgent: { type: "noul", instructions: "Does this convey urgency?" } };
  await step("busy_no_retry", async () => {
    await client.systemOne({ state: "Help!", questions: q }, { headers: { "x-jev-test": "busy_always" }, retry: { maxRetries: 0 } });
    return { error_class: null, note: "no exception raised" };
  });
  await step("busy_retried", async () => {
    const w = await withResponse(client.systemOne({ state: "Help!", questions: q }, { headers: { "x-jev-test": `busy_once:js:${Date.now()}` } }));
    return { model: w.data.model, request_id: w.requestId };
  });
  await step("overload_retried", async () => {
    const w = await withResponse(client.systemOne({ state: "Help!", questions: q }, { headers: { "x-jev-test": `overload_once:js:${Date.now()}` } }));
    return { model: w.data.model, request_id: w.requestId };
  });
}
out({ step: "done", ok: true });
