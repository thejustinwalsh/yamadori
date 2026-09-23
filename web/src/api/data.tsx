// One poll per endpoint for the whole app, so the header, the panels and the
// tree all read the same snapshot and agree about its age.
import { createContext, useContext, type ReactNode } from 'react';
import type { DatasetsOverview, Tiers, Vitals } from './types';
import { usePoll, type Poll } from './usePoll';

export const PATHS = {
  vitals: '/dash/api/vitals',
  datasets: '/dash/api/datasets',
  results: '/dash/api/results',
  stats: '/dash/api/stats',
  tiers: '/dash/api/tiers',
  dataset: (id: string) => `/dash/api/datasets/${encodeURIComponent(id)}`,
} as const;

type Shared = { vitals: Poll<Vitals>; datasets: Poll<DatasetsOverview>; tiers: Poll<Tiers> };
const Ctx = createContext<Shared | null>(null);

export function DataProvider({ children }: { children: ReactNode }) {
  // 5 s is the Python vitals page's EVERY; each snapshot costs nvidia-smi,
  // a PowerShell process listing and two HTTP probes, so no faster.
  const vitals = usePoll<Vitals>(PATHS.vitals, 5000);
  const datasets = usePoll<DatasetsOverview>(PATHS.datasets, 10000);
  const tiers = usePoll<Tiers>(PATHS.tiers, 60000);
  return <Ctx.Provider value={{ vitals, datasets, tiers }}>{children}</Ctx.Provider>;
}

export function useShared(): Shared {
  const v = useContext(Ctx);
  if (!v) throw new Error('useShared outside DataProvider');
  return v;
}
