// DENKI · ELECTRICITY: live GPU watts, the DTE D1.11 period and price, $/h at
// this draw, today's and the last seven days' kWh and dollars, from
// mcp/power.py live() (the `power` field of /dash/api/vitals). GPU draw only,
// and the panel says so under every figure.
import * as stylex from '@stylexjs/stylex';
import { basisLines, centsText, dayLabel, dollarsText, isPower, kwhText, periodName, wattsText, weekTotal } from '../api/power';
import type { PowerDay, PowerLive, Vitals } from '../api/types';
import { ago } from '../format';
import { colors, space } from '../tokens/tokens.stylex';
import { Panel } from './Panel';
import { Label, layout, Meter, Stat } from './primitives';
import { StateView } from './StateView';
import { Table } from './Table';
import { text } from './text';

const s = stylex.create({
  gpu: {
    display: 'flex',
    flexDirection: 'column',
    gap: space.spaceXs,
    padding: space.spaceXs,
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 25%, transparent)`,
  },
  watts: { color: colors.primary },
  basis: { margin: 0, color: colors.onSurfaceVariant },
  err: { margin: 0, color: colors.secondary },
});

type P = { v: Vitals | null; stale?: boolean; fill?: boolean };

export function PowerPanel({ v, stale, fill }: P) {
  const p = v?.power;
  const live = isPower(p) ? p : null;
  const rate = live?.rate;
  const peak = rate?.period === 'peak';
  return (
    <Panel
      kanji="電"
      title="DENKI · ELECTRICITY"
      tag={rate ? `${periodName(rate.period)} · ${centsText(rate.cents_per_kwh, 3)}/kWh` : '—'}
      tagTone={peak ? 'rose' : 'moss'}
      flag={live ? `nvidia-smi · ${live.interval_s ?? 1} s` : 'nvidia-smi'}
      stale={stale}
      edge={peak ? 'crimson' : undefined}
      fill={fill}
    >
      {!v ? (
        <StateView kind="loading" title="reading /dash/api/vitals" />
      ) : p === undefined ? (
        <StateView kind="inert" title="this server predates the power sampler" detail="Restart the proxy (mcp/server.py) to start mcp/power.py." />
      ) : !live ? (
        <StateView kind="error" title="no power payload" detail={(p && typeof p.last_error === 'string' && p.last_error) || undefined} />
      ) : (
        <PowerBody p={live} />
      )}
    </Panel>
  );
}

function PowerBody({ p }: { p: PowerLive }) {
  const week = weekTotal(p.days);
  const extra = p.extra_watts > 0 ? ` + ${Math.round(p.extra_watts)} W EXTRA` : '';
  const until = p.rate.until_local ? ` · UNTIL ${p.rate.until_local.toUpperCase()}` : '';
  return (
    <>
      {!p.running && (
        <p {...stylex.props(text.labelXs, s.err)}>
          [ ! ] SAMPLER NOT LIVE{p.age_s !== null ? ` · LAST SAMPLE ${ago(p.age_s)} AGO` : ''}
          {p.last_error ? ` · ${p.last_error}` : ''}
        </p>
      )}
      <div {...stylex.props(layout.grid4)}>
        <Stat label="DRAW" value={wattsText(p.total_watts)} sub={`GPU${extra}`} tone="moss" />
        <Stat label="RATE" value={centsText(p.rate.cents_per_kwh, 3)} sub={`${periodName(p.rate.period)}${until}`} tone={p.rate.period === 'peak' ? 'rose' : undefined} />
        <Stat label="PER HOUR" value={dollarsText(p.dollars_per_hour)} sub="AT THIS DRAW" />
        <Stat label="TODAY" value={dollarsText(p.today.cents === null ? null : p.today.cents / 100)} sub={kwhText(p.today.kwh)} tone="cyan" />
      </div>
      {p.gpus.length ? (
        <div {...stylex.props(layout.grid2)}>
          {p.gpus.map((g) => (
            <div key={g.uuid ?? g.index} {...stylex.props(s.gpu)} title={g.uuid ?? undefined}>
              <div {...stylex.props(layout.between)}>
                <Label>GPU{g.index} · {g.name.replace(/^NVIDIA (GeForce )?/, '')}</Label>
                <span {...stylex.props(text.labelMd, text.num, s.watts)}>{wattsText(g.watts)}</span>
              </div>
              <Meter
                value={g.watts !== null && g.watts_limit ? g.watts / g.watts_limit : null}
                tone="moss"
                label={`GPU${g.index} drawing ${wattsText(g.watts)} of a ${wattsText(g.watts_limit)} limit`}
              />
              <Label>LIMIT {wattsText(g.watts_limit)}</Label>
            </div>
          ))}
        </div>
      ) : null}
      <Table
        rows={p.days}
        rowKey={(d: PowerDay) => d.date}
        caption="GPU electricity per day, last seven days"
        columns={[
          { key: 'd', head: 'day', cell: (d: PowerDay) => dayLabel(d.date, p.today.date) },
          { key: 'k', head: 'kWh', num: true, cell: (d: PowerDay) => (d.kwh === null ? '—' : d.kwh.toFixed(3)) },
          { key: 'pk', head: 'peak', num: true, cell: (d: PowerDay) => (d.kwh_peak === null ? '—' : d.kwh_peak.toFixed(3)) },
          { key: 'op', head: 'off-peak', num: true, cell: (d: PowerDay) => (d.kwh_off_peak === null ? '—' : d.kwh_off_peak.toFixed(3)) },
          { key: 'c', head: '$', num: true, cell: (d: PowerDay) => dollarsText(d.cents === null ? null : d.cents / 100) },
          { key: 'h', head: 'sampled', num: true, cell: (d: PowerDay) => (d.measured_seconds ? `${(d.measured_seconds / 3600).toFixed(1)} h` : '—') },
        ]}
      />
      <Label>
        7 DAYS · {kwhText(week.kwh)} · {dollarsText(week.cents === null ? null : week.cents / 100)}
        {week.days ? ` · ${week.days} DAY${week.days === 1 ? '' : 'S'} SAMPLED` : ' · NOTHING SAMPLED YET'}
      </Label>
      <div {...stylex.props(layout.stack)}>
        {basisLines(p.basis, p.rate.flat).map((line) => (
          <p key={line} {...stylex.props(text.labelXs, s.basis)}>
            {line}
          </p>
        ))}
      </div>
    </>
  );
}
