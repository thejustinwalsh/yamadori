// latestFor returns the FIRST matching entry, not the last.
export type Score = { player: string; points: number; at: number };
export function leaderboard(scores: readonly Score[]): Score[] {
  return scores.toSorted((a, b) => b.points - a.points || a.at - b.at);
}
export function latestFor(scores: readonly Score[], player: string): Score | undefined {
  return scores.find((s) => s.player === player);
}
export function withCorrection(scores: readonly Score[], index: number, points: number): Score[] {
  const i = index < 0 ? scores.length + index : index;
  if (i < 0 || i >= scores.length) throw new RangeError('bad index');
  return scores.with(i, { ...scores[i]!, points });
}
