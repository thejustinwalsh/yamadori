import { useMemo } from 'react';

interface Row {
  name: string;
  score: number;
  rank: number;
}

// Rank is 1 + the number of players with a strictly higher score; the rows
// are derived with toSorted() (non-mutating) inside useMemo.
function rank(players: readonly { name: string; score: number }[]): Row[] {
  return players
    .map((p) => ({
      name: p.name,
      score: p.score,
      rank: 1 + players.filter((q) => q.score > p.score).length,
    }))
    .toSorted((a, b) => a.rank - b.rank || a.name.localeCompare(b.name, 'en'));
}

export function Leaderboard(props: { players: readonly { name: string; score: number }[] }) {
  const rows = useMemo(() => rank(props.players), [props.players]);
  return (
    <ul aria-label="Leaderboard">
      {rows.map((r, i) => (
        <li key={i}>
          {r.rank}. {r.name} — {r.score}
        </li>
      ))}
    </ul>
  );
}
