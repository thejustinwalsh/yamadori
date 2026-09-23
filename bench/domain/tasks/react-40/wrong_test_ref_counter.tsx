// Wrong: the running position lives in a ref that is advanced during render
// and never reset, so a second render (StrictMode, or any re-render) keeps
// counting from where the last one stopped.
import { useRef } from 'react';

type Player = { name: string; score: number };

export function Leaderboard({ players }: { players: readonly Player[] }) {
  const position = useRef(0);
  const sorted = [...players].sort(
    (a, b) => b.score - a.score || a.name.localeCompare(b.name),
  );
  let rank = 0;
  let lastScore: number | null = null;
  return (
    <ol>
      {sorted.map((p) => {
        position.current += 1;
        if (p.score !== lastScore) rank = position.current;
        lastScore = p.score;
        return <li key={p.name}>{`${rank}. ${p.name} — ${p.score}`}</li>;
      })}
    </ol>
  );
}
