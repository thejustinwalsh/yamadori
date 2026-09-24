// Speed (MTP staging + long context), image generation, the published model
// card, and the tier ladder every arms legend refers to.
import type { ImagegenSection, LongCtxSection, ModelCardSection, Section, SpeedSection, Tiers } from '../../api/types';
import type { Poll } from '../../api/usePoll';
import { n } from '../../format';
import { Panel } from '../../ui/Panel';
import { Chip } from '../../ui/primitives';
import { PollState, StateView } from '../../ui/StateView';
import { Table } from '../../ui/Table';
import { f1, f2, isEmpty, isError, pc0 } from './model';
import { type Bar, Chips, CompareBars, H3, Lines, NotReady, tally, What } from './parts';

// ------------------------------------------------------------------ tiers --

export function TierLadder({ tiers }: { tiers: Poll<Tiers> }) {
  const t = tiers.data;
  return (
    <Panel id="sec-tiers" kanji="段" title="EFFORT TIERS" tag="WHAT EACH ARM MAPS TO" tagTone="muted" fill>
      <What>
        A client's reasoning_effort picks a tier; each tier allows more of the stack. Fan-out and deep thinking, where
        allowed, are chosen per request. The columns are what runs (mcp/tiers.py features, the same matrix README and
        AGENTS.md carry). A benchmark arm is either a tier or a header that forces each switch.
      </What>
      {!t ? (
        <PollState path="/dash/api/tiers" failure={tiers.failure} />
      ) : (
        <Table
          rows={(t.order ?? Object.keys(t.tiers ?? {})).flatMap((k) => (t.tiers?.[k] ? [{ k, v: t.tiers[k] }] : []))}
          rowKey={(r) => r.k}
          caption="effort tiers (mcp/tiers.py TIERS, features)"
          columns={[
            {
              key: 'k',
              head: 'reasoning_effort',
              cell: ({ k }) => (
                <span>
                  <strong>{k}</strong>
                  {t.default === k ? ' · default' : ''}
                </span>
              ),
            },
            ...(t.feature_columns ?? []).map((c) => ({
              key: `f-${c}`,
              head: c,
              cell: ({ v }: { v: Tiers['tiers'][string] }) => v.features?.[c] ?? '—',
            })),
          ]}
        />
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ speed --

function LongCtx({ lc }: { lc: SpeedSection['longctx'] }) {
  if (!lc || isEmpty(lc) || isError(lc)) return <NotReady sec={lc as never} what="long-context sweep" />;
  const r = lc as LongCtxSection;
  const tasks = Array.from(new Set(Object.values(r.curves ?? {}).flatMap((c) => Object.keys(c.accuracy ?? {}))));
  return (
    <>
      <Table
        rows={r.arm_table}
        rowKey={(a) => a.arm}
        caption="long-context configs"
        columns={[
          { key: 'a', head: 'config', cell: (a) => a.arm },
          { key: 'c', head: 'n_ctx', num: true, cell: (a) => n(a.n_ctx) },
          { key: 'u', head: 'usable context', num: true, cell: (a) => n(a.usable_context) },
          { key: 'd', head: 'decode tok/s first → longest', num: true, cell: (a) => `${f1(a.decode_tps_first)} → ${f1(a.decode_tps_longest)}` },
          { key: 'p', head: 'prefill tok/s first → longest', num: true, cell: (a) => `${n(a.prefill_tps_first)} → ${n(a.prefill_tps_longest)}` },
          { key: 's', head: 'draft acceptance first → longest', num: true, cell: (a) => (a.speculating ? `${f1(a.draft_accept_first)} → ${f1(a.draft_accept_longest)}` : 'not speculating') },
          { key: 'e', head: 'errors', num: true, cell: (a) => a.errors },
        ]}
      />
      {Object.entries(r.curves ?? {}).map(([arm, c]) => (
        <Table
          key={arm}
          rows={c.speed ?? []}
          rowKey={(x) => String(x.L)}
          caption={`${arm}: speed and accuracy against context length`}
          columns={[
            { key: 'L', head: `${arm} · context`, num: true, cell: (x) => n(x.L) },
            { key: 'd', head: 'decode tok/s', num: true, cell: (x) => f1(x.decode_tps) },
            { key: 'p', head: 'prefill tok/s', num: true, cell: (x) => n(x.prefill_tps) },
            { key: 'a', head: 'draft accept', num: true, cell: (x) => (x.draft_accept_rate === null ? '—' : x.draft_accept_rate.toFixed(3)) },
            { key: 'v', head: 'VRAM peak MiB', num: true, cell: (x) => n(x.vram_peak_mib) },
            ...tasks.map((t) => ({
              key: t,
              head: `${t} correct · Wilson`,
              num: true,
              cell: (x: { L: number }) => {
                const cell = (c.accuracy?.[t] ?? []).find((y) => y.L === x.L);
                return cell && cell.n ? `${cell.k}/${cell.n} [${pc0(cell.lo)}–${pc0(cell.hi)}]` : '—';
              },
            })),
          ]}
        />
      ))}
      {r.note ? <Lines items={[r.note]} /> : null}
    </>
  );
}

export function Speed({ sec }: { sec: Section<SpeedSection> | undefined }) {
  const sp = sec && !isEmpty(sec) && !isError(sec) ? (sec as SpeedSection) : null;
  const b = sp?.mtp?.builds;
  const top = Math.max(60, ...(b?.rows ?? []).map((r) => r.mean_tps ?? 0));
  const axis = Math.ceil(top / 10) * 10;
  const bars: Bar[] = (b?.rows ?? []).map((r) => ({
    label: `${r.run} · ${r.build}`,
    value: r.mean_tps === null ? null : r.mean_tps / axis,
    text: `${f2(r.mean_tps)} tok/s · head ${r.head} · BI ${r.bi}`,
    tone: r.head === 'on' ? 'cyan' : 'muted',
  }));
  return (
    <Panel id="sec-speed" kanji="速" title="SPEED" tag="TOK/S" tagTone="cyan" fill>
      <What>
        Single-stream decode speed per llama.cpp build and model file, and how often the MTP draft head's guess is accepted, by
        content type. Numbers are llama-server's own timings.
      </What>
      {!sp ? (
        <NotReady sec={sec as never} what="speed" />
      ) : (
        <>
          <Chips>
            <Chip tone="muted">{sp.mtp.source}</Chip>
            {sp.mtp.section6 && <Chip tone="rose">MEASURED · {sp.mtp.section6.toUpperCase()}</Chip>}
          </Chips>
          {b ? (
            <>
              <H3>{b.heading} · axis 0–{axis} tok/s</H3>
              <CompareBars bars={bars} label="decode tokens per second per build" />
              <Lines items={[b.method]} />
              <Table
                rows={b.rows}
                rowKey={(r) => r.run}
                caption="decode tok/s by content type"
                columns={[
                  { key: 'r', head: 'run', cell: (r) => r.run },
                  { key: 'b', head: 'binary / file', cell: (r) => r.build },
                  { key: 'h', head: 'MTP head', cell: (r) => r.head },
                  ...Object.keys(b.rows[0]?.by_content ?? {}).map((c) => ({
                    key: c,
                    head: `${c} tok/s`,
                    num: true,
                    cell: (r: (typeof b.rows)[number]) => f2(r.by_content[c]),
                  })),
                  { key: 'm', head: 'mean', num: true, cell: (r) => f2(r.mean_tps) },
                  { key: 'a', head: 'acceptance', num: true, cell: (r) => r.acceptance ?? '—' },
                ]}
              />
            </>
          ) : (
            <StateView kind="empty" title={`${sp.mtp.source} · no section 6 build table found`} />
          )}
          {sp.mtp.acceptance.map((t) => (
            <Table
              key={t.file}
              rows={t.rows}
              rowKey={(r) => r.content}
              caption={`draft acceptance by content · ${t.file}`}
              columns={[
                { key: 'c', head: `acceptance · ${t.file.replace(/`/g, '')}`, cell: (r) => r.content },
                { key: 'n', head: 'prompts', num: true, cell: (r) => n(r.n_prompts) },
                { key: 'a', head: 'accepted / drafted', num: true, cell: (r) => (r.acceptance === null ? '—' : `${r.acceptance.toFixed(3)} (${n(r.accepted)}/${n(r.drafted)})`) },
                { key: 'on', head: 'tok/s head on', num: true, cell: (r) => f1(r.tps_on) },
                { key: 'off', head: 'tok/s head off', num: true, cell: (r) => (r.tps_off === null ? 'not run' : f1(r.tps_off)) },
              ]}
            />
          ))}
          <H3>Speed and accuracy against context length · bench/longctx</H3>
          <LongCtx lc={sp.longctx} />
        </>
      )}
    </Panel>
  );
}

// ----------------------------------------------------------------- images --

const STATUS_TONE: Record<string, 'moss' | 'muted' | 'crimson' | 'rose'> = {
  ok: 'moss',
  errors_logged: 'rose',
  killed: 'crimson',
  failed: 'crimson',
  skipped: 'muted',
};
const STATUS_LABEL: Record<string, string> = {
  ok: 'IMAGE',
  errors_logged: 'IMAGE · ERRORS IN LOG',
  killed: 'KILLED',
  failed: 'NO IMAGE',
  skipped: 'SKIPPED',
};

export function Images({ sec }: { sec: Section<ImagegenSection> | undefined }) {
  const im = sec && !isEmpty(sec) && !isError(sec) ? (sec as ImagegenSection) : null;
  return (
    <Panel id="sec-images" kanji="画" title="IMAGES" tag="S / IMAGE" tagTone="cyan" fill>
      <What>
        Image generation (Qwen-Image 2.1 in stable-diffusion.cpp): wall seconds per image, denoising seconds per step, and the
        card memory each configuration took, from nvidia-smi polled while it ran.
      </What>
      {!im ? (
        <NotReady sec={sec as never} what="image generation" />
      ) : (
        <>
          <Table
            rows={im.configs}
            rowKey={(c) => `${c.config}|${c.size}`}
            caption="per configuration and size"
            columns={[
              { key: 'c', head: 'config', cell: (c) => c.config },
              { key: 'z', head: 'size', cell: (c) => c.size },
              {
                key: 'st',
                head: 'attempts',
                cell: (c) => (
                  <Chips>
                    {Object.entries(c.status).map(([k, v]) => (
                      <Chip key={k} tone={STATUS_TONE[k] ?? 'muted'}>
                        {v} {STATUS_LABEL[k] ?? k}
                      </Chip>
                    ))}
                  </Chips>
                ),
              },
              { key: 'w', head: 'median s / image (min–max)', num: true, cell: (c) => (c.images ? `${f1(c.median_wall_s)} (${f1(c.min_wall_s)}–${f1(c.max_wall_s)}) · n ${c.images}` : '—') },
              { key: 'sp', head: 's / step', num: true, cell: (c) => (c.median_sec_per_step === null ? '—' : c.median_sec_per_step.toFixed(2)) },
              { key: 'dc', head: 'decode s', num: true, cell: (c) => f1(c.median_decode_s) },
              { key: 'v', head: 'VRAM Δ peak MiB', num: true, cell: (c) => n(c.max_delta_peak_mib) },
              { key: 'f', head: 'min free MiB', num: true, cell: (c) => n(c.min_free_mib) },
              { key: 't', head: 'peak °C', num: true, cell: (c) => n(c.max_temp_c) },
            ]}
          />
          <Lines
            items={im.configs.flatMap((c) => [
              ...c.kills.map((k) => `${c.config} ${c.size}: killed${k.why ? ` (${k.why})` : ''} · min free ${n(k.min_free_mib)} MiB · Δ peak ${n(k.delta_peak_mib)} MiB`),
              ...c.skips.map((k) => `${c.config} ${c.size}: skipped · ${k}`),
            ])}
            warn
          />
          {im.configs.some((c) => c.errors.length) ? (
            <Lines items={Array.from(new Set(im.configs.flatMap((c) => c.errors.map((e) => `${c.config}: ${e}`))))} />
          ) : null}
          {im.server.length ? (
            <Table
              rows={im.server}
              rowKey={(x, i) => `${x.config}|${i}`}
              caption="through the image server"
              columns={[
                { key: 'c', head: 'served path', cell: (x) => x.config ?? '—' },
                { key: 'i', head: 'images ok', num: true, cell: (x) => `${x.ok}/${x.images}` },
                { key: 's', head: 'median s (client)', num: true, cell: (x) => f1(x.median_seconds) },
                { key: 'v', head: 'VRAM Δ peak MiB', num: true, cell: (x) => n(x.delta_peak_mib) },
                { key: 'f', head: 'min free MiB', num: true, cell: (x) => n(x.min_free_mib) },
              ]}
            />
          ) : null}
          {im.idle_probes.length ? (
            <Lines
              items={im.idle_probes.map(
                (p) => `idle probe: loaded ${n(p.base_used_mib)} MiB → ${n(p.idle_after_mib)} MiB after generating (median) → ${n(p.after_kill_mib)} MiB after the server stopped`,
              )}
            />
          ) : null}
          {im.partiprompts ? (
            <Lines items={[`PartiPrompts sample: ${im.partiprompts.file} · ${n(im.partiprompts.prompts)} prompts · ${tally(im.partiprompts.categories)}`]} />
          ) : (
            <StateView kind="empty" title="no PartiPrompts sample manifest under bench/imagegen" />
          )}
        </>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------- model card --

export function ModelCard({ sec }: { sec: Section<ModelCardSection> | undefined }) {
  const c = sec && !isEmpty(sec) && !isError(sec) ? (sec as ModelCardSection) : null;
  return (
    <Panel id="sec-card" kanji="札" title="MODEL CARD" tag="PUBLISHED · NOT MEASURED HERE" tagTone="rose" fill>
      <What>The base model's quantisation, as its publisher reports it against the full-precision model it came from.</What>
      {!c ? (
        <NotReady sec={sec as never} what="model card" />
      ) : (
        <>
          <Chips>
            <Chip tone="rose">PUBLISHED BY {c.publisher.toUpperCase()}</Chip>
          </Chips>
          <Table
            rows={c.rows}
            rowKey={(r) => r.bench}
            caption={`published by ${c.publisher}, not measured here`}
            columns={[
              { key: 'b', head: 'benchmark', cell: (r) => r.bench },
              ...c.columns.map((col, i) => ({ key: col, head: col, num: true, cell: (r: (typeof c.rows)[number]) => (r.values[i] === null || r.values[i] === undefined ? '—' : String(r.values[i])) })),
            ]}
          />
          <Lines items={[<a key="s" href={c.source} target="_blank" rel="noreferrer" style={{ color: 'inherit' }}>{c.source}</a>]} />
        </>
      )}
    </Panel>
  );
}
