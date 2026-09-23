// Pre-ES2023 style: copy-then-sort, a reverse loop, and a map for the replacement.
export interface Score { player: string; points: number; at: number }

export const leaderboard = (scores: readonly Score[]): Score[] =>
  [...scores].sort((a, b) => (a.points !== b.points ? b.points - a.points : a.at - b.at));

export function latestFor(scores: readonly Score[], player: string): Score | undefined {
  for (let i = scores.length - 1; i >= 0; i--) {
    if (scores[i]!.player === player) return scores[i];
  }
  return undefined;
}

export function withCorrection(scores: readonly Score[], index: number, points: number): Score[] {
  const target = index >= 0 ? index : scores.length + index;
  if (!Number.isInteger(target) || target < 0 || target >= scores.length) throw new RangeError('bad index');
  return scores.map((s, i) => (i === target ? { player: s.player, points, at: s.at } : s));
}
