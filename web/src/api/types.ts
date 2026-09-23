// Shapes of the /dash/api/* JSON, read from the Python that produces them
// (mcp/vitals.py, mcp/budget.py, mcp/datasets.py, mcp/jobs.py,
// mcp/dash_results.py, mcp/concept_seed.py). Every field the server can omit
// is optional here, so the compiler forces each panel to handle its absence.

export type Gpu = {
  index: number;
  name: string;
  used_mib: number;
  total_mib: number;
  free_mib: number;
  pct: number;
  tight: boolean;
  util: number;
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
      /** tokens per deep-thinking context */
      helper: number;
      /** how many deep-thinking contexts; absent on servers older than the field */
      helpers?: number;
      reserve: number;
      gib: number;
    }
  | { error: string };

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
  warnings: string[];
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
};
export type Slots = { ok: boolean; slots: Slot[]; ms: number; error?: string; decoding?: number; prefilling?: number; tps?: number };

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

export type Power = {
  n_paired: number;
  min_discordant_for_sig: number | null;
  max_possible_discordant: number;
  alpha: number;
  comparison_possible: boolean;
};

export type Pair = {
  a: string;
  b: string;
  b_only: number;
  a_only: number;
  discordant: number;
  p: number;
  verdict: string | null;
  underpowered: boolean;
  both_solved?: number;
};

export type SectionMeta = { file: string; exists: boolean; file_age_s: number | null; bad_lines: number };

export type EmptySection = SectionMeta & { state: 'empty'; how: string; note: string };
export type ErrorSection = { state: 'error'; component: string; error: string; check: string };

/** where a scored row ended: passed, or the grader stage it failed at */
export type StageCounts = { pass?: number; extract: number; compile: number; test: number };

export type LcbArm = {
  arm: string; k: number; n: number; rate: number | null; lo: number; hi: number;
  mean_s: number | null; tok_in: number | null; tok_out: number | null;
  empty?: number; len_cut?: number; errors?: number;
  // bench/domain/analyse.py only (absent in the livecodebench section)
  stages?: StageCounts;
  median_s?: number | null;
  tok_total?: number | null;
  /** mean tool rounds the proxy ran for the model (x_yamadori.hops - 1, summed over requests) */
  tool_hops?: number | null;
  investigate_hops?: number | null;
  /** rows whose final answer got past extract and compile */
  final_compiles?: number;
  self_check?: boolean;
  // S arms (the model may call check_solution) only
  twin?: string;
  check_rounds_mean?: number | null;
  check_rounds_median?: number | null;
  check_rounds_max?: number | null;
  checked_any?: number;
  last_check_ok?: number;
  final_public_ok?: number;
  final_public_n?: number;
  rounds_mean?: number | null;
};

/** an S arm against its one-shot twin (S0 vs A0, S5 vs A5, S6 vs A6) */
export type SelfCheckTwin = {
  s: string; one_shot: string; n_paired: number;
  s_pass: number; one_shot_pass: number; s_only: number; one_shot_only: number;
  discordant: number; p: number; p_bonferroni: number | null;
  /** one-shot failed at compile, S arm passed */
  fixed_by_checking: number;
  compile_to_test: number; extract_to_pass: number; test_to_pass: number;
  one_shot_compile_fail: number; s_compile_fail: number;
  one_shot_compiles: number; s_compiles: number;
};

export type DomainPair = {
  domain: string; a: string; b: string; n_paired: number;
  b_only: number; a_only: number; p_uncorrected: number;
};
export type LcbSection = SectionMeta & {
  state: 'ready';
  arms: string[];
  suspect: { identical_across_arms: boolean; extreme: string[] };
  progress: { rows: number; attempted: number; scored_rows: number; errors: number; complete_all_arms: number; errors_by_arm: Record<string, Record<string, number>> };
  power: Power;
  arm_table: LcbArm[];
  raw: { arm: string; attempts: number; passed: number; rate: number; lo: number; hi: number }[];
  pairs: Pair[];
  difficulty: {
    difficulty: string;
    n: number;
    by_arm: Record<string, { k: number; n: number; rate: number; lo?: number; hi?: number; stages?: StageCounts }>;
  }[];
  failures: Record<string, { passed: number; failed: number; buckets: { bucket: string; n: number; scaffolding: boolean }[] }>;
  scaffolding_total: number;
  // bench/domain/analyse.py only
  twins?: SelfCheckTwin[];
  domain_pairs?: DomainPair[];
  excluded_contaminated?: { tasks: number; domains: string[] };
};

export type RetrievalArm = {
  arm: string; n: number; hit1: number; hit1_rate: number; hit1_lo: number; hit1_hi: number;
  hit5: number; hit5_rate: number; hit5_lo: number; hit5_hi: number; mrr: number;
  mean_ms: number; absent: number; extreme: boolean; cost_x: number;
};
export type RetrievalSection = SectionMeta & {
  state: 'ready';
  n: number;
  arms: string[];
  power: Power;
  arm_table: RetrievalArm[];
  pairs: Pair[];
  sources: { source: string; n: number; by_arm: Record<string, number> }[];
  caveats: string[];
};

export type RecipeSection = SectionMeta & {
  state: 'ready';
  arms: string[];
  progress: { attempted: number; errors: number; complete_all_arms: number };
  power: Power;
  arm_table: { arm: string; k: number; n: number; rate: number | null; lo: number; hi: number; median_max_s: number | null; median_peak_kb: number | null }[];
  oracle: { k: number; n: number; rate: number | null };
};

export type Section<T> = T | EmptySection | ErrorSection;

export type Results = {
  served_at: number;
  sections: {
    /** bench/domain/analyse.py dashboard_section(): the newest domain run, lcb-shaped */
    domain?: Section<LcbSection>;
    lcb?: Section<LcbSection>;
    retrieval?: Section<RetrievalSection>;
    recipe?: Section<RecipeSection>;
    [k: string]: unknown;
  };
};

export type TierSpec = {
  thinks: boolean; floor: number; effort: string; retrieval: boolean; hints: boolean;
  fanout: number; investigate: boolean; why: string;
  /** the reasoning_effort the proxy actually sends (tiers.safe_effort); absent until the server reports it */
  sent_effort?: string;
};
export type Tiers = { order: string[]; tiers: Record<string, TierSpec>; default?: string; ceiling?: string };
