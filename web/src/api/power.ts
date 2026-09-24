// Pure helpers over mcp/power.py's payloads: the electricity panel on the
// cockpit (/dash/api/vitals `power`) and the LiveBench electricity estimate.
// Every figure is GPU draw only; the panel prints the basis beside it.
import type { LbElectricity, PowerBasis, PowerDay, PowerLive, Range2 } from './types';

/** A full live() payload, as opposed to an absent field (older server) or a bare error. */
export function isPower(p: unknown): p is PowerLive {
  return !!p && typeof p === 'object' && 'rate' in p && 'days' in p && 'basis' in p;
}

export const periodName = (p: string | null | undefined): string =>
  p === 'peak' ? 'PEAK' : p === 'off_peak' ? 'OFF-PEAK' : p === 'flat' ? 'FLAT RATE' : '—';

const ok = (x: number | null | undefined): x is number => typeof x === 'number' && Number.isFinite(x);

export const wattsText = (w: number | null | undefined): string => (ok(w) ? `${Math.round(w)} W` : '—');

export const centsText = (c: number | null | undefined, digits = 2): string => (ok(c) ? `${c.toFixed(digits)}¢` : '—');

/** Dollars with enough digits to be non-zero for small sums (a day of idle GPUs is cents). */
export function dollarsText(d: number | null | undefined, digits?: number): string {
  if (!ok(d)) return '—';
  const dp = digits ?? (Math.abs(d) >= 1 ? 2 : Math.abs(d) >= 0.01 ? 3 : 4);
  return `$${d.toFixed(dp)}`;
}

export const kwhText = (k: number | null | undefined, digits = 3): string => (ok(k) ? `${k.toFixed(digits)} kWh` : '—');

/** "TODAY", else "MON 21" from an ISO date (parsed as a calendar date, not UTC midnight). */
export function dayLabel(date: string, today: string): string {
  if (date === today) return 'TODAY';
  const [y, m, d] = date.split('-').map(Number);
  if (!y || !m || !d) return date;
  const wd = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'][new Date(y, m - 1, d).getDay()];
  return `${wd} ${String(d).padStart(2, '0')}`;
}

/** Sum of the days that measured anything; null when none did. */
export function weekTotal(days: PowerDay[]): { kwh: number | null; cents: number | null; days: number } {
  const measured = days.filter((d) => ok(d.kwh));
  if (!measured.length) return { kwh: null, cents: null, days: 0 };
  return {
    kwh: measured.reduce((a, d) => a + (d.kwh ?? 0), 0),
    cents: measured.reduce((a, d) => a + (d.cents ?? 0), 0),
    days: measured.length,
  };
}

/** The label under every figure: what is measured, what is not, and the price's source. */
export function basisLines(b: PowerBasis | undefined, flat: boolean): string[] {
  if (!b) return [];
  const extra = ok(b.extra_watts) && b.extra_watts > 0
    ? `PLUS ${Math.round(b.extra_watts)} W CONFIGURED FOR THE REST OF THE SYSTEM (${b.extra_watts_env})`
    : `NO REST-OF-SYSTEM WATTS CONFIGURED (${b.extra_watts_env} = 0)`;
  return [
    `GPU DRAW ONLY · nvidia-smi power.draw · NOT MEASURED: ${b.not_measured}`,
    extra,
    flat ? `PRICE: ${b.rate} (${b.flat_rate_env})` : 'PRICE: DTE D1.11 TIME OF DAY 3–7 PM · BEFORE SURCHARGES AND TAXES · HOLIDAYS NOT MODELLED',
  ];
}

export const rangeText = (r: Range2 | null | undefined, f: (x: number) => string): string =>
  !r ? '—' : r[0] === r[1] ? f(r[0]) : `${f(r[0])}–${f(r[1])}`;

/** One LiveBench arm's estimate as table cells. */
export function electricityCells(e: LbElectricity | null | undefined) {
  if (!e) return null;
  return {
    perQuestionWh: `${e.wh_per_question.toFixed(1)} Wh`,
    perQuestionCents: rangeText(e.cents_per_question, (x) => x.toFixed(2)) + '¢',
    per100Kwh: `${e.kwh_per_100.toFixed(2)} kWh`,
    per100Dollars: rangeText(e.dollars_per_100, (x) => `$${x.toFixed(2)}`),
    seconds: `${Math.round(e.seconds_per_question)} s`,
  };
}
