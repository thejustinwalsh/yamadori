// What the proxy hosts and decides beside the model, for the cockpit:
//   /dash/api/mcp    mcp/dash_mcp.py -> mcp/mcp_host.py status(): the MCP
//                    servers the proxy runs for the model (read-only)
// and pure helpers over them, so what the panels show is testable.
import type { LoadedModel, Serving } from './types';

export const MCP_PATH = '/dash/api/mcp';

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
