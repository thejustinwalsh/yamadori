// Wrong: dense ranking -- after a tie the next rank is not skipped (1, 2, 2, 3).
type Player = { name: string; score: number };

export function Leaderboard({ players }: { players: readonly Player[] }) {
  const sorted = [...players].sort(
    (a, b) => b.score - a.score || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0),
  );
  const distinct = [...new Set(sorted.map((p) => p.score))];
  return (
    <ol>
      {sorted.map((p) => (
        <li key={p.name}>{`${distinct.indexOf(p.score) + 1}. ${p.name} — ${p.score}`}</li>
      ))}
    </ol>
  );
}
