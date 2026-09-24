// Static renders of the electricity panel and the LiveBench estimate (vitest
// runs in node: react-dom/server markup), plus the pure helpers behind them.
// The payloads are shaped exactly as mcp/power.py live() and
// dash_results._livebench_electricity() return them.
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { basisLines, dayLabel, dollarsText, electricityCells, isPower, periodName, weekTotal } from '../api/power';
import type { LbRun, PowerDay, PowerLive, Vitals } from '../api/types';
import { Electricity } from '../screens/sentei/LiveBench';
import { PowerPanel } from './PowerPanel';

const day = (date: string, kwh: number | null, cents: number | null, over: Partial<PowerDay> = {}): PowerDay => ({
  date,
  kwh,
  cents,
  kwh_peak: kwh === null ? null : kwh / 4,
  kwh_off_peak: kwh === null ? null : (3 * kwh) / 4,
  kwh_flat: kwh === null ? null : 0,
  cents_peak: null,
  cents_off_peak: null,
  cents_flat: null,
  gpu_kwh: kwh,
  extra_kwh: kwh === null ? null : 0,
  measured_seconds: kwh === null ? 0 : 36000,
  gap_seconds: 0,
  ...over,
});

const basis = {
  measured: 'GPU board power, nvidia-smi power.draw, every card',
  not_measured: 'CPU, motherboard, RAM, drives, fans, PSU loss',
  extra_watts: 0,
  extra_watts_env: 'YAMADORI_POWER_EXTRA_WATTS',
  rate: 'DTE Energy residential Time of Day 3-7 p.m. standard rate D1.11, rate card in effect 2025-02-06, excluding surcharges and taxes',
  flat_rate_env: 'YAMADORI_POWER_RATE_CENTS',
  holidays: 'not modelled',
  timezone: 'machine local time',
};

const power = (over: Partial<PowerLive> = {}): PowerLive => ({
  running: true,
  at: 1790250000,
  age_s: 0.4,
  interval_s: 1,
  gpus: [
    { index: 0, name: 'NVIDIA GeForce RTX 5060 Ti', uuid: 'GPU-de66', watts: 132.3, watts_limit: 180 },
    { index: 1, name: 'NVIDIA RTX A4000', uuid: 'GPU-43e3', watts: 6.2, watts_limit: 140 },
  ],
  gpu_watts: 138.5,
  extra_watts: 0,
  total_watts: 138.5,
  rolling_watts: 137.9,
  rolling_seconds: 60,
  rate: {
    period: 'off_peak',
    season: 'summer',
    cents_per_kwh: 18.435,
    flat: false,
    until: 1790276400,
    until_local: 'Thu 15:00',
    components: { capacity: 3.24, non_capacity: 5.469, distribution: 9.726 },
    source: basis.rate,
  },
  dollars_per_hour: 0.0257,
  today: day('2026-09-24', 1.2, 22.5),
  days: [
    day('2026-09-24', 1.2, 22.5),
    day('2026-09-23', 2.5, 50.1),
    day('2026-09-22', null, null),
    day('2026-09-21', null, null),
    day('2026-09-20', null, null),
    day('2026-09-19', null, null),
    day('2026-09-18', null, null),
  ],
  errors: 0,
  last_error: null,
  basis,
  ...over,
});

const vitals = (p: unknown): Vitals =>
  ({ at: 1, gpus: [], processes: [], listeners: [], duplicates: [], endpoints: [], context: { pool: 1, main: 1, helper: 1, reserve: 0, gib: 1 }, warnings: [], ...(p === undefined ? {} : { power: p }) }) as Vitals;

const render = (v: Vitals | null) => renderToStaticMarkup(createElement(PowerPanel, { v }));

describe('PowerPanel', () => {
  it('shows live watts per GPU and in total, the period and price, $/h and today', () => {
    const html = render(vitals(power()));
    expect(html).toContain('DENKI · ELECTRICITY');
    expect(html).toContain('OFF-PEAK · 18.435¢/kWh');
    expect(html).toContain('>139 W<'); // total
    expect(html).toContain('RTX 5060 Ti');
    expect(html).toContain('>132 W<');
    expect(html).toContain('RTX A4000');
    expect(html).toContain('>6 W<');
    expect(html).toContain('UNTIL THU 15:00');
    expect(html).toContain('$0.026'); // per hour
    expect(html).toContain('$0.225'); // today, 22.5 cents
    expect(html).toContain('1.200 kWh');
  });
  it('lists seven days, unmeasured days as dashes, and the week total', () => {
    const html = render(vitals(power()));
    expect(html).toContain('TODAY');
    expect(html).toContain('WED 23');
    expect(html.match(/<tr/g)?.length).toBe(8); // header + 7 days
    expect(html).toContain('7 DAYS · 3.700 kWh · $0.726 · 2 DAYS SAMPLED');
  });
  it('labels the figures GPU draw only, at DTE D1.11 before surcharges and taxes, with the extra watts', () => {
    const html = render(vitals(power()));
    expect(html).toContain('GPU DRAW ONLY');
    expect(html).toContain('NOT MEASURED: CPU, motherboard, RAM, drives, fans, PSU loss');
    expect(html).toContain('DTE D1.11');
    expect(html).toContain('BEFORE SURCHARGES AND TAXES');
    expect(html).toContain('YAMADORI_POWER_EXTRA_WATTS = 0');
    const extra = render(vitals(power({ extra_watts: 85, total_watts: 223.5, basis: { ...basis, extra_watts: 85 } })));
    expect(extra).toContain('GPU + 85 W EXTRA');
    expect(extra).toContain('PLUS 85 W CONFIGURED FOR THE REST OF THE SYSTEM');
  });
  it('peak is marked, and a flat rate names its env var', () => {
    const peak = render(vitals(power({ rate: { ...power().rate, period: 'peak', cents_per_kwh: 24.133, until_local: 'Thu 19:00' } })));
    expect(peak).toContain('PEAK · 24.133¢/kWh');
    expect(peak).toContain('UNTIL THU 19:00');
    const flat = render(vitals(power({ rate: { ...power().rate, period: 'flat', flat: true, cents_per_kwh: 21.5, until_local: null }, basis: { ...basis, rate: 'YAMADORI_POWER_RATE_CENTS flat rate' } })));
    expect(flat).toContain('FLAT RATE · 21.500¢/kWh');
    expect(flat).toContain('YAMADORI_POWER_RATE_CENTS');
    expect(flat).not.toContain('UNTIL');
  });
  it('a sampler that is not live says so and shows no invented draw', () => {
    const html = render(vitals(power({ running: false, gpus: [], gpu_watts: null, total_watts: null, dollars_per_hour: null, age_s: 3600, last_error: 'nvidia-smi returned no rows' })));
    expect(html).toContain('SAMPLER NOT LIVE');
    expect(html).toContain('nvidia-smi returned no rows');
    expect(html).not.toContain('>139 W<');
  });
  it('an older server (no power field) asks for a restart; no data yet is a loading state', () => {
    expect(render(vitals(undefined))).toContain('predates the power sampler');
    expect(render(null)).toContain('reading /dash/api/vitals');
    expect(render(vitals({ running: false, last_error: 'boom' }))).toContain('boom');
  });
});

describe('power helpers', () => {
  it('isPower needs rate, days and basis', () => {
    expect(isPower(power())).toBe(true);
    expect(isPower({ running: false })).toBe(false);
    expect(isPower(null)).toBe(false);
  });
  it('names the periods', () => {
    expect(periodName('peak')).toBe('PEAK');
    expect(periodName('off_peak')).toBe('OFF-PEAK');
    expect(periodName('flat')).toBe('FLAT RATE');
  });
  it('formats dollars with enough digits to be non-zero', () => {
    expect(dollarsText(12.345)).toBe('$12.35');
    expect(dollarsText(0.0257)).toBe('$0.026');
    expect(dollarsText(0.0012)).toBe('$0.0012');
    expect(dollarsText(null)).toBe('—');
  });
  it('labels days by calendar date, not UTC midnight', () => {
    expect(dayLabel('2026-09-24', '2026-09-24')).toBe('TODAY');
    expect(dayLabel('2026-09-20', '2026-09-24')).toBe('SUN 20');
  });
  it('the week total skips unmeasured days rather than counting them as zero', () => {
    expect(weekTotal(power().days)).toEqual({ kwh: 3.7, cents: 72.6, days: 2 });
    expect(weekTotal([day('2026-09-24', null, null)])).toEqual({ kwh: null, cents: null, days: 0 });
  });
  it('basis lines are empty without a basis', () => {
    expect(basisLines(undefined, false)).toEqual([]);
  });
});

describe('LiveBench electricity', () => {
  const run = {
    arms: [
      {
        arm: 'bonsai',
        electricity: { n: 21, seconds_per_question: 226.6, watts: 138.5, wh_per_question: 8.718, kwh_per_100: 0.8718, cents_per_question: [0.1607, 0.2104], dollars_per_100: [0.1607, 0.2104], cents_per_kwh: [18.435, 24.133], seconds_from: 'summary' },
      },
      { arm: 'no-seconds', electricity: null },
    ],
    electricity_basis: {
      estimate: true, watts: 138.5, gpu_watts: 138.5, extra_watts: 0,
      evidence: 'operator measurement, 594 one-second nvidia-smi power.draw samples while generating; script not in the repo; n=1 run',
      per_gpu: {}, seconds: 's', overlap: 'wall seconds charge the whole GPU to each question', cents_per_kwh: [18.435, 24.133], rate: basis.rate, priced: 'cheapest and dearest cell of the rate table',
    },
  } as unknown as LbRun;
  it('shows per question and per 100 for each arm with seconds, labelled an estimate with its basis', () => {
    const html = renderToStaticMarkup(createElement(Electricity, { run }));
    expect(html).toContain('Electricity · estimate');
    expect(html).toContain('8.7 Wh');
    expect(html).toContain('0.16–0.21¢');
    expect(html).toContain('0.87 kWh');
    expect(html).toContain('$0.16–$0.21');
    expect(html).toContain('227 s');
    expect(html).toContain('ESTIMATE, not metered');
    expect(html).toContain('138.5 W');
    expect(html).toContain('594 one-second');
    expect(html).toContain('18.435–24.133¢/kWh');
    expect(html).not.toContain('no-seconds');
  });
  it('renders nothing without a basis', () => {
    expect(renderToStaticMarkup(createElement(Electricity, { run: { ...run, electricity_basis: null } }))).toBe('');
  });
  it('cells are null for an arm with no estimate', () => {
    expect(electricityCells(null)).toBeNull();
  });
});
