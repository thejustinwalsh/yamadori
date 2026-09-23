const nf = new Intl.NumberFormat('en-US');

export const n = (x: number | null | undefined): string => (x === null || x === undefined || !Number.isFinite(x) ? '—' : nf.format(x));

export const gib = (mib: number): string => (mib / 1024).toFixed(1);

export const pct = (x: number | null | undefined, digits = 1): string =>
  x === null || x === undefined || !Number.isFinite(x) ? '—' : `${(x * 100).toFixed(digits)}%`;

export function ago(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—';
  const s = Math.max(0, Math.round(seconds));
  if (s < 90) return `${s}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  if (s < 172800) return `${(s / 3600).toFixed(1)}h`;
  return `${Math.round(s / 86400)}d`;
}

export const clock = (epochSeconds: number): string => new Date(epochSeconds * 1000).toLocaleTimeString([], { hour12: false });

export const pValue = (p: number): string => (p < 1e-4 ? p.toExponential(1) : p.toFixed(4));
