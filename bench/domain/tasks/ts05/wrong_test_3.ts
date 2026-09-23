// Ties are broken the wrong way round (latest first).
export type Score = { player: string; points: number; at: number };
export function leaderboard(scores: readonly Score[]): Score[] {
  return scores.toSorted((a, b) => b.points - a.points || b.at - a.at);
}
export function latestFor(scores: readonly Score[], player: string): Score | undefined {
  return scores.findLast((s) => s.player === player);
}
export function withCorrection(scores: readonly Score[], index: number, points: number): Score[] {
  return scores.with(index, { ...scores.at(index)!, points });
}
