// An MCP client for checking a loadout's MCP servers WITHOUT a model
// (docs/HARNESS-SANDBOX.md "Default loadout"). Runs inside the harness box
// image, with the @modelcontextprotocol/sdk the box already pins (cclsp's
// dependency): no download.
//
//   node /probe/mcp_probe.mjs '<spec json>'
//   spec: {"command": "...", "args": [...], "env": {...}, "cwd": "...",
//          "calls": [{"tool": "...", "arguments": {...}}, ...]}
//
// Prints one JSON object: the server's tools (name, chars of the tool as an
// OpenAI function definition: the size a harness offers the model) and each
// call's text result.
import { Client } from "/opt/harness/node_modules/@modelcontextprotocol/sdk/dist/esm/client/index.js";
import { StdioClientTransport } from "/opt/harness/node_modules/@modelcontextprotocol/sdk/dist/esm/client/stdio.js";

const spec = JSON.parse(process.argv[2]);
const transport = new StdioClientTransport({
  command: spec.command, args: spec.args || [], cwd: spec.cwd,
  env: { ...Object.fromEntries(Object.entries(process.env).filter(([k]) =>
    ["PATH", "HOME", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"].includes(k))),
         ...(spec.env || {}) },
  stderr: "pipe",
});
let stderr = "";
transport.stderr?.on("data", (d) => { stderr += d.toString(); });
const client = new Client({ name: "yamadori-mcp-probe", version: "1" });
const out = { tools: [], calls: [] };
const t0 = Date.now();
try {
  await client.connect(transport);
  const { tools } = await client.listTools();
  out.tools = tools.map((t) => ({
    name: t.name,
    chars: JSON.stringify({ type: "function", function: { name: t.name, description: t.description, parameters: t.inputSchema } }).length,
    read_only: t.annotations?.readOnlyHint ?? null,
  }));
  for (const c of spec.calls || []) {
    if (c.sleep_ms) { await new Promise((r) => setTimeout(r, c.sleep_ms)); continue; }
    try {
      const r = await client.callTool({ name: c.tool, arguments: c.arguments || {} }, undefined, { timeout: c.timeout_ms || 120000 });
      out.calls.push({ tool: c.tool, isError: !!r.isError,
                       text: (r.content || []).filter((p) => p.type === "text").map((p) => p.text).join("\n").slice(0, 4000),
                       images: (r.content || []).filter((p) => p.type === "image").length });
    } catch (e) {
      out.calls.push({ tool: c.tool, error: String(e).slice(0, 1000) });
    }
  }
} catch (e) {
  out.error = String(e).slice(0, 1000);
}
out.ms = Date.now() - t0;
out.stderr_tail = stderr.slice(-1500);
console.log(JSON.stringify(out));
await client.close().catch(() => {});
process.exit(0);
