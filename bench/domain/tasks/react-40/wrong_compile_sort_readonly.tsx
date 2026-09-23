// Wrong: (strict) sort() does not exist on a readonly array.
type Player = { name: string; score: number };

export function Leaderboard({ players }: { players: readonly Player[] }) {
  const sorted = players.sort((a, b) => b.score - a.score || a.name.localeCompare(b.name));
  return (
    <ol>
      {sorted.map((p, i) => (
        <li key={p.name}>{`${i + 1}. ${p.name} — ${p.score}`}</li>
      ))}
    </ol>
  );
}
