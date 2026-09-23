// Wrong: sorts the players prop in place (a cast hides the readonly), which
// throws on a frozen array and reorders the caller's array.
type Player = { name: string; score: number };

export function Leaderboard({ players }: { players: readonly Player[] }) {
  const sorted = (players as Player[]).sort(
    (a, b) => b.score - a.score || a.name.localeCompare(b.name),
  );
  let rank = 0;
  return (
    <ol>
      {sorted.map((p, i) => {
        if (i === 0 || sorted[i - 1].score !== p.score) rank = i + 1;
        return <li key={p.name}>{`${rank}. ${p.name} — ${p.score}`}</li>;
      })}
    </ol>
  );
}
