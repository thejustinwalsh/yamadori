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
  /** nvidia-smi uuid, power.draw and power.limit (W); absent on older servers, null for [N/A] */
  uuid?: string | null;
  watts?: number | null;
  watts_limit?: number | null;
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
  /** GPU power and electricity cost; absent on servers older than mcp/power.py */
  power?: PowerLive | Partial<PowerLive> | null;
  /** mcp/tree_sources.py: index breadth (nebari), staleness (moss) and the
   *  last requests' fan-out and recall; absent on older servers. Read by
   *  bonsai/mapping.ts treeSources(), which checks every field it uses. */
  tree?: Record<string, unknown> | null;
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
  // the separate style score (grade_style.py); never part of pass/fail
  lint_n?: number;
  lint_errors_mean?: number | null;
  lint_warnings_mean?: number | null;
  lint_clean?: number;
  modern_flags?: Record<string, number>;
  modern_credits?: Record<string, number>;
  /** per mechanism: rows where the arm allowed it, and of those where it ran / produced data */
  mechanism_health?: Record<string, MechanismHealth>;
};

export type MechanismHealth = {
  rows: number; allowed: number; allowed_pct: number | null;
  ran: number; ran_pct: number | null; data: number; data_pct: number | null;
};

/** an S arm against its one-shot twin (S0 vs A0, S5 vs A5, S6 vs A6) */
export type SelfCheckTwin = {
  s: string; one_shot: string; n_paired: number;
  s_pass: number; one_shot_pass: number; s_only: number; one_shot_only: number;
  discordant: number; p: number; p_bonferroni: number | null;
  /** one-shot failed at compile, S arm passed */
  fixed_by_checking: number;
  /** fixed_by_checking, counting only S rows that called check_solution at least once */
  fixed_with_check?: number;
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
  /** clean=false: seconds and tokens/s were measured sharing the GPU with `concurrent_with` */
  timing?: { concurrent_with: string[]; clean: boolean; label: string };
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

/**
 * One arm, in the columns of README "Effort tiers" (mcp/dash_results.py
 * legend_from_features / legend_from_tier). "auto" = the tier allows it and
 * mcp/selection.py decides per request.
 */
export type LegendRow = {
  arm: string;
  kind: 'features' | 'tier' | 'unknown';
  note?: string;
  tier?: string;
  thinking_sent?: string | null;
  retrieval?: string;
  hints?: string;
  fanout?: string;
  deep_thinking?: string;
  check?: string;
  repair?: string;
  client_tool?: string | null;
  /** SWE-bench: reasoning_effort sent in the body (the tier), beside the header */
  body_effort?: string | null;
  reasoning_cap?: number;
};

export type Ci = [number, number] | null;
export type Dist = { n: number; median: number | null; p90: number | null; max: number | null; total: number };

/** bench/livebench/mechanisms.py health(): per mechanism, counts over answers */
export type MechHealth = {
  answers: number;
  recorded: number;
  stack_errors?: number;
  mechanisms?: Record<string, Record<string, number | null>>;
};

export type LbCategory = {
  score: number | null;
  ci95: Ci;
  n: number | null;
  tasks: Record<string, { score: number | null; ci95: Ci; n: number | null }>;
  finish_reason: Record<string, number>;
  seconds?: Dist | null;
  output_tokens?: Dist | null;
};

export type LbArm = {
  arm: string;
  label: string | null;
  overall: { score: number | null; ci95: Ci; categories_included: string[] };
  categories: Record<string, LbCategory>;
  status: Record<string, number>;
  finish_reason: Record<string, number>;
  rows: number;
  mechanism_health: MechHealth;
  /** estimate: seconds per question x measured generating watts (mcp/power.py benchmark_estimate) */
  electricity?: LbElectricity | null;
};

/** [cheapest, dearest] cell of the rate table, or a flat rate twice */
export type Range2 = [number, number];
export type LbElectricity = {
  n: number;
  seconds_per_question: number;
  watts: number;
  wh_per_question: number;
  kwh_per_100: number;
  cents_per_question: Range2;
  dollars_per_100: Range2;
  cents_per_kwh: Range2;
  seconds_from: 'summary' | 'rows';
};
export type LbElectricityBasis = {
  estimate: true;
  watts: number;
  gpu_watts: number;
  extra_watts: number;
  evidence: string;
  per_gpu: Record<string, number>;
  seconds: string;
  overlap: string;
  cents_per_kwh: Range2;
  rate: string;
  priced: string;
};

export type LbPaired = {
  a: string;
  b: string;
  n_pairs: number | null;
  diff: number | null;
  ci95: Ci;
  categories: Record<
    string,
    {
      n_pairs: number | null; a_score: number | null; b_score: number | null; diff: number | null; ci95: Ci;
      binary: boolean | null; b_only_correct: number | null; a_only_correct: number | null; mcnemar_p: number | null;
    }
  >;
};

export type LbProgress = {
  arm: string; category: string; log: string; log_age_s: number | null;
  at: string | null; answered: number; of: number | null; err_rows: number;
};

export type LbRef = { model: string; overall: number | null; n: number | null; missing: number | null; categories: Record<string, number | null> };

export type LbRun = {
  run_id: string;
  dir: string;
  state: 'ready' | 'running' | 'empty';
  frozen: string | null;
  condition: { min_p?: number; server_sampling?: string; requests_in_flight?: number; shared_with?: string; note?: string; [k: string]: unknown };
  generated_at: string | null;
  summary_age_s: number | null;
  release: string[] | string | null;
  population: Record<string, number>;
  method: { bootstrap_B?: number; ci?: string; aggregation?: string };
  arms: LbArm[];
  legend: LegendRow[];
  paired: LbPaired[];
  progress: LbProgress[];
  same_question_refs: Record<string, LbRef[]>;
  published_2024_11_25: Record<string, { categories: Record<string, number>; global: number }>;
  current_leaderboard_base: { date: string | null; models: Record<string, { global: number; categories: Record<string, number> }>; comparable: false } | null;
  electricity_basis?: LbElectricityBasis | null;
};

export type LiveBenchSection = SectionMeta & { state: 'ready'; current: string | null; runs: LbRun[] };

export type SweArm = {
  arm: string; attempted: number; scored: number; resolved: number;
  rate: number | null; lo: number | null; hi: number | null;
  outcomes: Record<string, number>;
  exit_status: Record<string, number>;
  median_steps: number | null; median_seconds: number | null;
  median_prompt_tokens: number | null; median_completion_tokens: number | null;
  format_errors: number; length_events: number; proxy_error_lines: number;
  finish_reasons: Record<string, number>;
  mechanisms: Record<string, number>;
  tiers_reported: Record<string, number>;
  instances: { instance: string; eval_status: string; exit_status: string; steps: number | null; seconds: number | null; prompt_tokens: number | null; completion_tokens: number | null }[];
};

export type SweBoard = {
  source: string;
  n_instances?: number;
  tables: { key: string; caption: string; same_scaffold: boolean; rows: { model: string; verified: number | null; detail: Record<string, string> }[] }[];
};

export type SweSection = SectionMeta & {
  state: 'ready' | 'running';
  run_id: string; subset: string | null; dataset: string | null; planned: number | null;
  stopped: string | null; arms: SweArm[]; legend: LegendRow[]; leaderboard: SweBoard;
};

export type MtpBuild = { run: string; build: string; head: string; bi: string; mean_tps: number | null; by_content: Record<string, number | null>; acceptance: string | null };
export type MtpAcceptance = { content: string; n_prompts: number | null; acceptance: number | null; accepted: number | null; drafted: number | null; tps_on: number | null; tps_off: number | null };

export type LongCtxCell = { L: number; k: number; n: number; rate: number | null; lo: number | null; hi: number | null; budget?: number; stack_errors?: number };
export type LongCtxSection = SectionMeta & {
  state: 'ready';
  arms: string[];
  arm_table: {
    arm: string; model: string | null; n_ctx: number | null; usable_context: number | null; ref_L: number | null;
    decode_tps_first: number | null; decode_tps_longest: number | null; prefill_tps_first: number | null; prefill_tps_longest: number | null;
    speculating: boolean; draft_accept_first: number | null; draft_accept_longest: number | null; errors: number;
  }[];
  curves: Record<string, { speed: { L: number; n: number; decode_tps: number | null; prefill_tps: number | null; draft_accept_rate: number | null; vram_peak_mib: number | null }[]; accuracy: Record<string, LongCtxCell[]> }>;
  note?: string;
};

export type SpeedSection = {
  state: 'ready';
  mtp: {
    source: string;
    section6: string | null;
    builds: { heading: string; rows: MtpBuild[]; method: string } | null;
    acceptance: { file: string; rows: MtpAcceptance[] }[];
    file_age_s?: number | null;
  };
  longctx: LongCtxSection | EmptySection | ErrorSection;
};

export type ImageConfig = {
  config: string; size: string; rows: number; status: Record<string, number>; images: number; attempts_run?: number;
  steps: number | null; median_wall_s: number | null; min_wall_s: number | null; max_wall_s: number | null;
  median_sample_s: number | null; median_sec_per_step: number | null; median_decode_s: number | null; median_encode_s: number | null;
  max_delta_peak_mib: number | null; min_free_mib: number | null; max_temp_c: number | null;
  kills: { why: string; min_free_mib: number | null; delta_peak_mib: number | null }[];
  skips: string[]; errors: string[];
};

export type ImagegenSection = SectionMeta & {
  state: 'ready';
  configs: ImageConfig[];
  server: { config: string | null; images: number; ok: number; median_seconds: number | null; delta_peak_mib: number | null; min_free_mib: number | null; killed: boolean | null }[];
  idle_probes: { base_used_mib: number | null; idle_after_mib: number | null; after_kill_mib: number | null; gen_512_s: number | null; at: string | null }[];
  partiprompts: { file: string; prompts: number | null; categories: Record<string, number> } | null;
};

export type ModelCardSection = {
  state: 'ready';
  publisher: string;
  source: string;
  measured_here: false;
  columns: string[];
  rows: { bench: string; values: (number | null)[] }[];
};

/** A bench/domain/analyse.py comparison (pairwise, on the tasks both arms scored). */
export type DomainComparison = {
  a: string; b: string; n_paired: number; a_pass: number; b_pass: number;
  b_only: number; a_only: number; discordant: number; both_solved: number;
  p: number; p_bonferroni?: number; family?: string; family_size?: number; could_reach_significance?: boolean;
};
export type DomainBlock = {
  label?: string; tasks?: number;
  families?: Record<string, { size: number; alpha_corrected: number; min_discordant_for_sig: number | null }>;
  comparisons?: DomainComparison[];
};
export type DomainArmStats = {
  attempted_pairs: number; scored: number; passed: number; rate: number | null; lo: number; hi: number;
  stack_error_rows: number; stack_error_kinds: Record<string, number>; fail_stages: Record<string, number>;
  median_seconds: number | null; median_prompt_tokens: number | null; median_completion_tokens: number | null;
  stages?: StageCounts; lint_n?: number; lint_errors_mean?: number | null; lint_clean?: number;
  modern_flags?: Record<string, number>; modern_credits?: Record<string, number>;
  tool_hops_mean?: number | null; investigate_hops_mean?: number | null; final_compiles?: number; n?: number;
  /** bench/domain/analyse.py mechanism_health(): per mechanism over the arm's scored rows */
  mechanism_health?: Record<string, DomainMechCount>;
};
export type DomainMechCount = { rows: number; allowed: number; ran: number; data: number; allowed_pct?: number | null; ran_pct?: number | null; data_pct?: number | null };
export type DomainGroup = {
  group: string; dirs: string[]; age_s: number; rows: number; stale_rows: number;
  stale: { reason: string; rows: number; arms: Record<string, number> }[];
  planned_pairs: number | null; scored_pairs: number; arms: string[]; conditions: Record<string, unknown>;
};
export type Rate = { k: number; n: number; rate: number };
export type DomainTriggers = {
  n: number;
  tools_offered?: Rate; hints_selected?: Rate; hints_emitted?: Rate; investigate_chosen?: Rate;
  investigate_searched?: Rate; investigate_injected?: Rate; fanned_out?: Rate; main_hops_gt1?: Rate;
};
export type DomainExtras = {
  run_id?: string; group?: string; merged_run_dirs?: string[];
  current?: DomainGroup;
  groups?: DomainGroup[];
  conditions?: Record<string, unknown>;
  rows_total?: number; stale_rows?: number;
  stale?: { reason: string; rows: number; arms: Record<string, number> }[];
  task_counts?: { all: number; uncontaminated: number; contaminated: number; contaminated_domains: string[] };
  caveats?: string[];
  headline?: DomainBlock;
  contaminated_block?: DomainBlock;
  per_arm_headline?: Record<string, DomainArmStats>;
  by_domain?: Record<string, {
    contaminated: boolean; tasks: number;
    by_arm: Record<string, { k: number; n: number; rate: number | null; lo: number; hi: number; stages: StageCounts; stack_errors: number }>;
    vs_A0: DomainComparison[];
  }>;
  triggers?: Record<string, DomainTriggers>;
  legend?: LegendRow[];
};

/** a domain group that is planned or running with nothing scorable yet */
export type DomainRunning = SectionMeta & DomainExtras & { state: 'running'; how: string };

export type Results = {
  served_at: number;
  sections: {
    /** bench/domain/analyse.py: the named group (overnight-0923-minp0), lcb-shaped plus the analysis */
    domain?: Section<LcbSection & DomainExtras> | DomainRunning;
    livebench?: Section<LiveBenchSection>;
    swebench?: Section<SweSection> | (EmptySection & { leaderboard?: SweBoard });
    speed?: Section<SpeedSection>;
    imagegen?: Section<ImagegenSection>;
    model_card?: Section<ModelCardSection>;
    lcb?: Section<LcbSection>;
    retrieval?: Section<RetrievalSection>;
    recipe?: Section<RecipeSection>;
    [k: string]: unknown;
  };
};

export type TierSpec = {
  thinks: boolean; floor: number; effort: string; retrieval: boolean; hints: boolean;
  fanout: number; investigate: boolean; why: string;
  check_code?: boolean; repair?: boolean;
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
};
