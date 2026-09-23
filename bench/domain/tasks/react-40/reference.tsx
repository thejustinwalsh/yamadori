type Player = { name: string; score: number };

export function Leaderboard({ players }: { players: readonly Player[] }) {
  const sorted = [...players].sort(
    (a, b) => b.score - a.score || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0),
  );
  return (
    <ol>
      {sorted.map((p, i) => {
        const rank = i > 0 && sorted[i - 1].score === p.score
          ? sorted.findIndex((q) => q.score === p.score) + 1
          : i + 1;
        return (
          <li key={`${p.name}:${i}`}>
            {`${rank}. ${p.name} — ${p.score}`}
          </li>
        );
      })}
    </ol>
  );
}
