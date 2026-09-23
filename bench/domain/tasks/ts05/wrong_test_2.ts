// withCorrection edits the element object in place (it is shared with the frozen input).
export type Score = { player: string; points: number; at: number };
export function leaderboard(scores: readonly Score[]): Score[] {
  return [...scores].sort((a, b) => b.points - a.points || a.at - b.at);
}
export function latestFor(scores: readonly Score[], player: string): Score | undefined {
  return scores.findLast((s) => s.player === player);
}
export function withCorrection(scores: readonly Score[], index: number, points: number): Score[] {
  const copy = [...scores];
  const i = index < 0 ? copy.length + index : index;
  if (i < 0 || i >= copy.length) throw new RangeError('bad index');
  copy[i]!.points = points;
  return copy;
}
