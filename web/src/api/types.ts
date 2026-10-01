// Shapes of the /dash/api/* JSON, read from the Python that produces them
// (mcp/vitals.py, mcp/budget.py, mcp/datasets.py, mcp/jobs.py,
// mcp/concept_seed.py; the JJAVA and SOKUDO pages'
// in ./stats.ts). Every field the server can omit
// is optional here, so the compiler forces each panel to handle its absence.

export type Gpu = {
  index: number;
  name: string;
  used_mib: number;
  total_mib: number;
  free_mib: number;
  pct: number;
  /** free_mib under floor_mib: the card's own live free-VRAM floor */
  tight: boolean;
  /** the free-VRAM floor this card is sized to, read live by the server (absent on older servers) */
  floor_mib?: number | null;
  util: number;
  /** nvidia-smi uuid, power.draw and power.limit (W); absent on older servers, null for [N/A] */
  uuid?: string | null;
  watts?: number | null;
  watts_limit?: number | null;
  /** temperature.gpu (°C), 2026-09-30; absent on older servers, null for [N/A] */
  temp_c?: number | null;
  /** true on the card the main model runs on (by UUID, mcp/vitals.py); absent on older servers */
  main?: boolean;
};

/** mcp/power.py rate_at(): the DTE D1.11 price of a kWh now. */
export type PowerRate = {
  period: 'peak' | 'off_peak' | 'flat';
  season: 'summer' | 'winter' | null;
  cents_per_kwh: number;
  flat: boolean;
  /** epoch of the next period change; null for a flat rate */
  until: number | null;
  until_local: string | null;
  components: Record<string, number> | null;
  source: string;
};

/** mcp/power.py day_summary(): one ledger day. null = nothing measured that day. */
export type PowerDay = {
  date: string;
  kwh: number | null;
  cents: number | null;
  kwh_peak: number | null;
  kwh_off_peak: number | null;
  kwh_flat: number | null;
  cents_peak: number | null;
  cents_off_peak: number | null;
  cents_flat: number | null;
  gpu_kwh: number | null;
  extra_kwh: number | null;
  measured_seconds: number;
  gap_seconds: number;
};

export type PowerBasis = {
  measured: string;
  not_measured: string;
  extra_watts: number;
  extra_watts_env: string;
  rate: string;
  flat_rate_env: string;
  holidays: string;
  timezone: string;
};

/** /dash/api/power and `power` on the vitals snapshot (mcp/power.py live()). */
export type PowerLive = {
  running: boolean;
  at: number | null;
  age_s: number | null;
  interval_s?: number;
  gpus: { index: number; name: string; uuid: string | null; watts: number | null; watts_limit: number | null }[];
  gpu_watts: number | null;
  extra_watts: number;
  total_watts: number | null;
  rolling_watts?: number | null;
  rolling_seconds?: number;
  rate: PowerRate;
  dollars_per_hour: number | null;
  today: PowerDay;
  days: PowerDay[];
  errors?: number;
  last_error: string | null;
  basis: PowerBasis;
};

export type Proc = { pid: number | null; ppid: number | null; what: string; port: string; started: string };
export type Listener = { port: number; role: string; addr: string; pid: string; conflict: boolean };
export type Endpoint = { name: string; ok: boolean; code: number; ms: number };
export type Duplicate = { what: string; pids: number[] };

/** mcp/budget.py budgets(), or {error} when the model server did not answer. */
export type ContextPool =
  | {
      pool: number;
      main: number;
      /** tokens per helper context (the child slot in the cap layout) */
      helper: number;
      /** how many helper contexts; absent on servers older than the field */
      helpers?: number;
      reserve: number;
      gib: number;
      /** 'cap': main = the VRAM line (THE CAP LAYOUT, 2026-09-28); 'split': the pool divided; absent on older servers */
      layout?: 'cap' | 'split' | string;
      /** where the main cap came from, or why there is none */
      cap_source?: string;
      /** the most one request may occupy */
      window?: number;
      /** /props kv_vram_cells as served (engine patch 0041); null when not reported; absent on older servers */
      vram_line?: number | null;
      /** budget.child(): the child slot's role, named by the server (layout v2: "decider lane"); absent on older servers */
      child?: { role?: string; tokens?: number; serves?: string[]; kept?: boolean; ranked?: boolean; rank?: number | null; slot?: number | null } | null;
      /** how the pool was read this time (2026-09-30: a view never loads a model): direct from the main model's server, or cached and why */
      pool_read?: string;
    }
  | { error: string; pool_read?: string };

/** mcp/concept_seed.py record(): the concept seed most recently put in a prompt. */
export type Seed = {
  word: string;
  u32: number;
  hex: string;
  token_id: number | null;
  where: string;
  at: number;
};

/** mcp/vitals.py strata: result-tier hit counts over a trailing window. */
export type Strata = {
  taproot: number;
  branch: number;
  shoot: number;
  searches: number;
  window_seconds: number;
};

export type Vitals = {
  at: number;
  gpus: Gpu[];
  processes: Proc[];
  listeners: Listener[];
  duplicates: Duplicate[];
  endpoints: Endpoint[];
  context: ContextPool;
  /**
   * Absent: the running server predates the field (needs a restart).
   * null: the server has it, and no seed has ever been injected.
   */
  seed?: Seed | null;
  /** result-tier hits across find_by_meaning searches; absent on older servers */
  strata?: Strata | null;
  /** the pulse fields below are also on the full snapshot; absent on older servers */
  slots?: Slots;
  lanes?: Lanes | null;
  tools?: ToolActivity | null;
  queue?: JobQueue | null;
  /** GPU power and electricity cost; absent on servers older than mcp/power.py */
  power?: PowerLive | Partial<PowerLive> | null;
  /** mcp/tree_sources.py: index breadth (nebari), staleness (moss) and the
   *  last requests' recall; absent on older servers. Read by
   *  bonsai/mapping.ts treeSources(), which checks every field it uses. */
  tree?: Record<string, unknown> | null;
  /** mcp/vitals.py serving(): which main model holds the card (max mode); absent on older servers */
  serving?: Serving | null;
  warnings: string[];
};

/** A llama-swap /running row as vitals reports it. */
export type LoadedModel = { model: string; state: string; port: number | null; gguf: string | null };

/** mcp/vitals.py serving() over mcp/max_mode.py. */
export type Serving = {
  /** max mode is configured (YAMADORI_MAX_MODEL) */
  enabled: boolean;
  main: string;
  max: string | null;
  max_tier: string;
  /** the main model or the max model, whichever llama-swap has loaded; null when neither or unread */
  on_card: string | null;
  /** max mode holds the card: a max request in flight or waiting to swap in */
  max_active: boolean;
  inflight: Record<string, number>;
  switching_to: string | null;
  last_max_end: number | null;
  /** the operator's idle seconds before swapping back; null until given */
  idle_s: number | null;
  /** llama-swap /running; null when it could not be read */
  loaded: LoadedModel[] | null;
  error: string | null;
};

/** mcp/vitals.py slots(): llama-server /slots, one row per slot. */
export type SlotState = 'idle' | 'prefill' | 'decode';
export type Slot = {
  id: number;
  state: SlotState;
  n_ctx: number;
  /** tokens this slot holds (prompt processed + decoded) */
  ctx: number;
  prompt: number;
  processed: number;
  decoded: number;
  remain: number;
  /** decode tokens/s since the server's previous read */
  tps: number;
  /** prefill tokens/s since the server's previous read */
  pps: number;
  /** the slot layout (mcp/slots.py): conversations on 0..n-2, the child n-1; absent on older servers */
  role?: 'conversation' | 'child';
  /** a conversation is pinned here (proxy process only) */
  pinned?: boolean;
  /** the primary conversation's slot (slots RANKS) */
  primary?: boolean;
};
export type Slots = {
  ok: boolean;
  slots: Slot[];
  ms: number;
  error?: string;
  /** the model whose server answered (the max model's while it holds the card); absent on older servers */
  model?: string | null;
  off_card?: boolean;
  decoding?: number;
  prefilling?: number;
  tps?: number;
};

/** mcp/vitals.py lanes(): admission lanes held right now. */
export type Lanes = {
  main: number;
  helper: number;
  main_lanes: number | null;
  helper_lanes: number | null;
  admitted?: number;
  queued?: number;
  refused?: number;
  /** false: read outside the proxy, so the counts are not the proxy's */
  in_proxy: boolean;
};

export type ToolEvent = { id: number; name: string | null; at: number; ms?: number | null; empty?: boolean };

/** mcp/vitals.py tools(): the newest tool calls in the corpus log. */
export type ToolActivity = {
  last: ToolEvent | null;
  running: ToolEvent | null;
  recent: ToolEvent[];
  calls_window: number;
  turns_window: number;
  answers_window: number;
  window_seconds: number;
  last_turn: { id: number; at: number; open: boolean } | null;
};

export type JobQueue = { states: Record<string, number>; oldest_queued_age: number | null };

/** /dash/api/vitals/pulse (mcp/vitals.py pulse()): the fast subset, ~1 s. */
export type Pulse = {
  at: number;
  gpus: Gpu[];
  slots: Slots;
  lanes: Lanes | null;
  tools: ToolActivity | null;
  queue: JobQueue | null;
  seed: Seed | null;
  strata: Strata | null;
  context: ContextPool;
  power?: PowerLive | Partial<PowerLive> | null;
};

export type JobRow = {
  id: string;
  queue: string;
  lane: string;
  state: string;
  stage: string | null;
  attempts: number;
  max_attempts: number;
  error: string | null;
  progress: string | null;
  created: number;
  [k: string]: unknown;
};

export type Missing = { field?: string; name?: string; what: string; why?: string; severity?: string };
export type DatasetWarning = { kind?: string; what: string };

export type Dataset = {
  id: string;
  name: string;
  prompt: string;
  kind: string;
  source: string | null;
  source_name: string | null;
  source_url: string | null;
  licence: string | null;
  language: string | null;
  domains: string[];
  notes: string | null;
  stage: string;
  counts: Record<string, number>;
  created: number;
  updated: number;
  recipe_file?: string | null;
  jobs?: Record<string, number>;
  review?: Record<string, number> | null;
  missing?: Missing[];
  warnings?: DatasetWarning[];
  next_stage?: string | null;
  errored_jobs?: JobRow[];
  running_jobs?: JobRow[];
  job_rows?: JobRow[];
  /** datasets.field_states(): each clarifying field and where its value came from. */
  field_states?: FieldState[];
  /** The raw `assist` column: provenance entries, what was searched, what was discarded. */
  assist?: AssistRecord;
  assist_jobs?: JobRow[];
};

/** Where a field's value came from (mcp/datasets.py field_states). */
export type Provenance = 'evidence' | 'proposed' | 'operator' | 'needs_you';

export type FieldState = {
  field: string;
  label: string;
  input: string;
  choices?: string[] | null;
  severity: string;
  why: string;
  value: string | string[] | null;
  assisted: boolean;
  state: Provenance;
  /** evidence only: the verbatim quote, and where it was actually found */
  quote?: string | null;
  found_in?: string | null;
  /** proposed only: what the proposal rests on */
  basis?: string;
  /** needs_you only: what is unanswered, and what the assist already searched */
  what?: string;
  searched?: string;
  /** what the assist offered and discarded, and why */
  rejected?: { why?: string; values?: string[] };
  /** operator only: the model's entry this answer replaced */
  overrode?: { provenance?: string; value?: unknown; quote?: string; found_in?: string };
};

export type SearchedRecord = {
  where: string;
  kind: 'source' | 'licence_file' | string;
  chars?: number;
  shown?: string;
  status?: number;
  result?: string;
  final_url?: string;
};

export type AssistRecord = {
  job?: string;
  at?: number;
  searched?: SearchedRecord[];
  not_found?: Record<string, string>;
  rejected?: Record<string, { why?: string; values?: string[] }>;
  /** set when a restricted licence the assist found keeps the dataset in clarify */
  held?: string | null;
};

export type Field = {
  name: string;
  label: string;
  input: string;
  required: boolean;
  severity: string;
  why: string;
  choices?: string[];
};

export type QueueSnapshot = {
  states: Record<string, number>;
  lanes: Record<string, Record<string, number>>;
  lane_limits: Record<string, number>;
  oldest_queued_age: number | null;
  db: string;
};

export type DatasetsOverview = {
  datasets: Dataset[];
  stages: string[];
  optional_stages: string[];
  human_stages: string[];
  enqueues: Record<string, { queue: string; lane: string }>;
  fields: Field[];
  /** Absent: the running server predates the assist (needs a restart). */
  assist?: { queue: string; lane: string; fields: string[] };
  lanes: Record<string, number>;
  queue: QueueSnapshot;
  worker: { ever_claimed: number; last_heartbeat: number | null; last_started: number | null };
  read_at: number;
};

export type TierSpec = {
  thinks: boolean; floor?: number; effort: string;
  /** retired with hints (2026-09-26): skills are the knowledge system; kept optional for old payloads */
  hints?: boolean;
  skills?: boolean;
  why: string;
  /** the model that serves the tier (mcp/max_mode.py model_for); absent on older servers */
  model?: string | null;
  /** the reasoning_effort the proxy actually sends (tiers.safe_effort); absent until the server reports it */
  sent_effort?: string;
  /** what RUNS at this tier, one cell per feature column (tiers.features, the one matrix the docs also read) */
  features?: Record<string, string>;
};
export type Tiers = {
  order: string[];
  tiers: Record<string, TierSpec>;
  /** the feature matrix's columns, in order (tiers.FEATURE_COLUMNS) */
  feature_columns?: string[];
  default?: string;
  ceiling?: string;
  /** max mode (mcp/max_mode.py); null where it cannot be read, absent on older servers */
  max_mode?: { enabled: boolean; main: string; max: string | null; tier: string } | null;
};
