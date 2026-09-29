// What the proxy hosts and decides beside the model, for the cockpit:
//   /dash/api/mcp    mcp/dash_mcp.py -> mcp/mcp_host.py status(): the MCP
//                    servers the proxy runs for the model (read-only)
//   /dash/api/deep   mcp/dash_deep.py -> mcp/deep_learn.py overview(): deep
//                    thinking's triggers, thresholds and labels
// and pure helpers over them, so what the panels show is testable.
import type { LoadedModel, Serving } from './types';

export const MCP_PATH = '/dash/api/mcp';
export const DEEP_PATH = '/dash/api/deep';

export type McpTool = { name: string; upstream?: string; description?: string; overlaps?: string[] };
export type McpStatus = {
  id?: string;
  /** stopped / starting / ready / exited / failed */
  state?: string;
  why?: string | null;
  starts?: number;
  server?: { name?: string; version?: string; protocol?: string } | null;
  upstream_tools?: string[];
  ready_since?: number | null;
  network?: string | null;
};
export type McpServer = {
  id: string;
  title?: string;
  enabled?: boolean;
  runtime?: string;
  package?: string;
  licence?: string;
  image?: string;
  network?: string;
  call_timeout_s?: number;
  call_timeout_why?: string;
  tools?: McpTool[];
  /** this process's state; absent when the host never started it */
  status?: McpStatus | null;
};
export type Mcp = {
  host: { enabled: boolean; switch?: string };
  config: { path: string; exists: boolean; version?: number | null };
  servers: McpServer[];
};

/** A server's state as one word: its status, else "not started". */
export function mcpState(s: McpServer): string {
  if (s.enabled === false) return 'disabled';
  return s.status?.state || 'not started';
}

export type DeepDay = { day: string; requests: number; runs: Record<string, number>; labels: Record<string, number> };
export type Deep = {
  thresholds: Record<string, { value: number; source: string; bounds?: number[]; n?: number }>;
  adjustments?: unknown[];
  proposals?: unknown[];
  per_day: DeepDay[];
  recent?: { id: string; created: number; tier?: string; route?: string; trigger?: string; ran?: number; label?: string | null; traffic?: string }[];
  learning?: { idle?: boolean; why?: string; pending?: number };
  labels?: string[];
  note?: string;
};

/** Deep-thinking runs by trigger over the last `days` days of per_day, newest last. */
export function deepRuns(d: Deep | null | undefined, days = 7): { days: DeepDay[]; byTrigger: [string, number][]; requests: number; runs: number } {
  const rows = Array.isArray(d?.per_day) ? d!.per_day.slice(-days) : [];
  const by: Record<string, number> = {};
  let requests = 0;
  for (const r of rows) {
    requests += typeof r.requests === 'number' ? r.requests : 0;
    for (const [k, v] of Object.entries(r.runs ?? {})) by[k] = (by[k] ?? 0) + (typeof v === 'number' ? v : 0);
  }
  const byTrigger = Object.entries(by).sort((a, b) => b[1] - a[1]);
  return { days: rows, byTrigger, requests, runs: byTrigger.reduce((a, [, v]) => a + v, 0) };
}

/**
 * The model the main card is serving, from vitals.serving (max mode aware),
 * as {model, gguf}; null when the server predates the field or llama-swap
 * could not be read.
 */
export function servedOnCard(s: Serving | null | undefined): { model: string; gguf: string | null; max: boolean } | null {
  if (!s || !s.on_card) return null;
  const row: LoadedModel | undefined = s.loaded?.find((x) => x.model === s.on_card);
  return { model: s.on_card, gguf: row?.gguf ? row.gguf.replace(/\.gguf$/i, '') : null, max: !!s.max && s.on_card === s.max };
}
