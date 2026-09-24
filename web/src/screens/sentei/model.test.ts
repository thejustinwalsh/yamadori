// The benchmark page's pure reads: the headline strip picks numbers the
// server carries and never makes one up; mechanism rows keep "not exposed"
// (null) apart from "no such count" (undefined).
import { describe, expect, it } from 'vitest';
import type { MechHealth, Results } from '../../api/types';
import { kOfN, mechRowsDomain, mechRowsLiveBench, scoreCi, summaryTiles } from './model';

const lbRun = (over: object) => ({
  run_id: 'lb-2', dir: 'd', state: 'ready', frozen: null, condition: {}, generated_at: null, summary_age_s: 1,
  release: ['2024-11-25'], population: {}, method: {}, legend: [], paired: [], progress: [], same_question_refs: {},
  published_2024_11_25: {}, current_leaderboard_base: null,
  arms: [
    {
      arm: 'bonsai', label: null, status: { ok: 21 }, finish_reason: {}, rows: 21, mechanism_health: { answers: 21, recorded: 0 },
      overall: { score: 70.4545, ci95: [51.36, 85.9], categories_included: ['coding'] },
      categories: { coding: { score: 70.45, ci95: [51.36, 85.9], n: 21, tasks: {}, finish_reason: {} } },
    },
  ],
  ...over,
});

describe('summaryTiles', () => {
  it('headlines each benchmark with its n, and says when a section is empty or errored', () => {
    const r = {
      served_at: 0,
      sections: {
        livebench: { state: 'ready', file: 'f', exists: true, file_age_s: 1, bad_lines: 0, current: 'lb-2', runs: [lbRun({})] },
        swebench: { state: 'empty', file: 'f', exists: false, file_age_s: null, bad_lines: 0, how: 'run x', note: '' },
        imagegen: { state: 'error', component: 'c', error: 'E: boom', check: 'x' },
        model_card: { state: 'ready', publisher: 'PrismML', source: 's', measured_here: false, columns: ['A', 'B'], rows: [{ bench: '20-benchmark mean', values: [85.4, 83.9] }] },
      },
    } as unknown as Results;
    const t = Object.fromEntries(summaryTiles(r).map((x) => [x.key, x]));
    expect(t.lb?.lines[0]).toEqual({ label: 'bonsai', value: '70.5 [51.4–85.9]', sub: 'n 21 · coding' });
    expect(t.swe?.state).toBe('empty');
    expect(t.swe?.lines).toEqual([]);
    expect(t.img?.state).toBe('error');
    expect(t.card?.published).toBe(true);
    expect(t.card?.note).toContain('not measured here');
    expect(t.card?.lines.map((l) => l.value)).toEqual(['85.4', '83.9']);
  });

  it('a run with no summary is running, with its answered count, and shows no score', () => {
    const r = {
      served_at: 0,
      sections: {
        livebench: {
          state: 'ready', file: 'f', exists: true, file_age_s: 1, bad_lines: 0, current: 'lb-2',
          runs: [lbRun({ state: 'running', arms: [], progress: [{ arm: 'bonsai', category: 'coding', answered: 4, of: 21, err_rows: 0 }] })],
        },
      },
    } as unknown as Results;
    const lb = summaryTiles(r).find((x) => x.key === 'lb');
    expect(lb?.state).toBe('running');
    expect(lb?.lines).toEqual([]);
    expect(lb?.note).toContain('4 answered, not yet scored');
  });
});

describe('formatting', () => {
  it('prints k/n with its Wilson interval, and no interval when there is none', () => {
    expect(kOfN(2, 2, 0.342, 1)).toBe('2/2 · 100% [34%–100%]');
    expect(kOfN(0, 0)).toBe('0/0');
    expect(scoreCi(null, null)).toBe('—');
  });
});

describe('mechanism rows', () => {
  it('keeps not-exposed (null) apart from no-such-count (undefined)', () => {
    const h: MechHealth = {
      answers: 3, recorded: 2, stack_errors: 0,
      mechanisms: { retrieval: { allowed: 2, ran: 1, produced: null }, hints: { allowed: 2, decided: 1, ran: 1, produced: 1, injected_total: 3 } },
    };
    const rows = mechRowsLiveBench('yamadori', h);
    const ret = rows.find((x) => x.mechanism === 'retrieval');
    expect(ret?.produced).toBeNull();
    expect(ret?.decided).toBeUndefined();
    expect(ret?.n).toBe(2);
    expect(rows.find((x) => x.mechanism === 'hints')?.extra).toBe('injected_total 3');
    expect(mechRowsLiveBench('x', { answers: 21, recorded: 0 })).toEqual([]);
  });

  it('reads the domain triggers as counts of n tasks', () => {
    const rows = mechRowsDomain('A6', {
      n: 4, tools_offered: { k: 4, n: 4, rate: 1 }, investigate_chosen: { k: 3, n: 4, rate: 0.75 },
      investigate_searched: { k: 2, n: 4, rate: 0.5 }, investigate_injected: { k: 1, n: 4, rate: 0.25 },
    });
    const dt = rows.find((x) => x.mechanism === 'deep thinking');
    expect([dt?.decided, dt?.ran, dt?.produced, dt?.n]).toEqual([3, 2, 1, 4]);
    expect(mechRowsDomain('A0', { n: 0 })).toEqual([]);
  });
});
