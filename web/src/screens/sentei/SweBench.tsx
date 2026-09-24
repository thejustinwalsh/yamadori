// SWE-bench Verified Mini through mini-swe-agent's bash-only config, graded
// by the swebench harness; the leaderboard beside it is published, full 500.
import type { EmptySection, ErrorSection, SweBoard, SweSection } from '../../api/types';
import { ago, n } from '../../format';
import { Panel } from '../../ui/Panel';
import { Chip } from '../../ui/primitives';
import { StateView } from '../../ui/StateView';
import { Table } from '../../ui/Table';
import { f1, isEmpty, isError, kOfN, mechRowsSwe, pc0 } from './model';
import { ArmsLegend, type Bar, Chips, Collapse, CompareBars, H3, Lines, MechTable, NotReady, tally, What } from './parts';

type Sec = SweSection | (EmptySection & { leaderboard?: SweBoard }) | ErrorSection | undefined;

function boardBars(board: SweBoard | undefined): Bar[] {
  const out: Bar[] = [];
  for (const t of board?.tables ?? []) {
    for (const r of t.rows) {
      out.push({
        label: r.model,
        value: r.verified === null ? null : r.verified / 100,
        text: `${f1(r.verified)}%`,
        group: t.same_scaffold
          ? `PUBLISHED · FULL ${board?.n_instances ?? 500} · SAME SCAFFOLD (mini-swe-agent v2, bash only)`
          : `PUBLISHED · FULL ${board?.n_instances ?? 500} · OLDER MINI v1.x (text actions)`,
      });
    }
  }
  return out;
}

export function SweBench({ sec }: { sec: Sec }) {
  const ready = sec && !isEmpty(sec) && !isError(sec) ? (sec as SweSection) : null;
  const board = ready?.leaderboard ?? (sec && isEmpty(sec) ? (sec as EmptySection & { leaderboard?: SweBoard }).leaderboard : undefined);
  const ours: Bar[] = (ready?.arms ?? []).map((a) => ({
    label: `${a.arm} (ours)`,
    value: a.rate,
    lo: a.lo,
    hi: a.hi,
    text: `${a.resolved}/${a.scored} · ${pc0(a.rate)} [${pc0(a.lo)}–${pc0(a.hi)}]`,
    ours: true,
    group: `MEASURED HERE · ${ready?.subset ?? 'subset'} · ${a.scored} OF ${ready?.planned ?? '?'} GRADED`,
  }));
  const mech = (ready?.arms ?? []).flatMap(mechRowsSwe);
  return (
    <Panel id="sec-swebench" kanji="修" title="SWE-BENCH VERIFIED" tag="RESOLVED" tagTone="cyan" fill>
      <What>
        Real GitHub issues: the agent edits the repository with bash only (mini-swe-agent 2.1.0, the leaderboard's config), and an
        instance counts as resolved when the swebench harness's hidden tests pass in Docker.
      </What>
      {!ready ? (
        <NotReady sec={sec as never} what="SWE-bench" />
      ) : (
        <>
          <Chips>
            <Chip tone="muted">{ready.run_id}</Chip>
            {ready.dataset && <Chip tone="muted">{ready.dataset}</Chip>}
            <Chip tone={ready.arms.length ? 'cyan' : 'muted'}>
              {ready.arms.reduce((s, a) => s + a.scored, 0)} GRADED OF {ready.planned ?? '?'} PLANNED
            </Chip>
            <Chip tone="muted">WRITTEN {ago(ready.file_age_s)} AGO</Chip>
          </Chips>
          {ready.stopped ? <Lines items={[ready.stopped]} warn /> : null}
          <ArmsLegend rows={ready.legend} />
          {ready.legend.some((l) => l.body_effort) ? (
            <Lines items={[`every arm also sends reasoning_effort "${ready.legend.find((l) => l.body_effort)?.body_effort}" in the body, so the tier allows everything and the header decides`]} />
          ) : null}
          <Table
            rows={ready.arms}
            rowKey={(a) => a.arm}
            caption="resolved per arm with Wilson 95% interval, and cost"
            columns={[
              { key: 'a', head: 'arm', cell: (a) => a.arm },
              { key: 'r', head: 'resolved / graded · Wilson 95%', num: true, cell: (a) => kOfN(a.resolved, a.scored, a.lo, a.hi) },
              { key: 'o', head: 'outcomes', cell: (a) => tally(a.outcomes) },
              { key: 'st', head: 'median steps', num: true, cell: (a) => n(a.median_steps) },
              { key: 's', head: 'median wall s', num: true, cell: (a) => n(a.median_seconds === null ? null : Math.round(a.median_seconds)) },
              { key: 'pt', head: 'median prompt tok (all steps)', num: true, cell: (a) => n(a.median_prompt_tokens === null ? null : Math.round(a.median_prompt_tokens)) },
              { key: 'ct', head: 'median completion tok', num: true, cell: (a) => n(a.median_completion_tokens === null ? null : Math.round(a.median_completion_tokens)) },
              { key: 'f', head: 'finish reasons', cell: (a) => tally(a.finish_reasons) },
              { key: 'e', head: 'format err · length', num: true, cell: (a) => `${a.format_errors} · ${a.length_events}` },
            ]}
          />
        </>
      )}
      {ours.length || board?.tables?.length ? (
        <>
          <H3>Resolved rate · ours against the published leaderboard</H3>
          <CompareBars bars={[...ours, ...boardBars(board)]} label="SWE-bench Verified resolved rate: measured here and published" />
          {board ? <Lines items={[`published: ${board.source} section 7 (SWE-bench leaderboard), % resolved of the full ${board.n_instances ?? 500}`]} /> : null}
        </>
      ) : ready ? (
        <StateView kind="empty" title="no leaderboard table found in docs/SWE-BENCH.md section 7" />
      ) : null}
      {ready ? (
        <>
          {mech.length ? (
            <>
              <H3>Mechanism health · per agent turn</H3>
              <MechTable rows={mech} caption="what the proxy reported on each agent turn (x_yamadori)" />
            </>
          ) : null}
          <Collapse summary={`Instances · ${ready.arms.reduce((s, a) => s + a.instances.length, 0)}`}>
            <Table
              rows={ready.arms.flatMap((a) => a.instances.map((i) => ({ ...i, arm: a.arm })))}
              rowKey={(r) => `${r.arm}|${r.instance}`}
              caption="every graded instance"
              columns={[
                { key: 'a', head: 'arm', cell: (r) => r.arm },
                { key: 'i', head: 'instance', cell: (r) => r.instance },
                { key: 'v', head: 'outcome', cell: (r) => r.eval_status },
                { key: 'x', head: 'agent exit', cell: (r) => r.exit_status },
                { key: 'st', head: 'steps', num: true, cell: (r) => n(r.steps) },
                { key: 's', head: 'wall s', num: true, cell: (r) => n(r.seconds === null ? null : Math.round(r.seconds)) },
                { key: 'p', head: 'prompt tok', num: true, cell: (r) => n(r.prompt_tokens) },
                { key: 'c', head: 'completion tok', num: true, cell: (r) => n(r.completion_tokens) },
              ]}
            />
          </Collapse>
        </>
      ) : null}
    </Panel>
  );
}
