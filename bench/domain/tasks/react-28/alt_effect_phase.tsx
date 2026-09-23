import { useEffect, useId, useReducer } from 'react';

// Phase machine in a reducer; the delay timer lives in an effect keyed on the
// phase (so any phase change or unmount clears it); the tooltip element is
// always rendered and toggled with the `hidden` attribute; Escape is caught by
// a document listener while open.
type Phase = 'idle' | 'waiting' | 'open';
type Event = 'arm' | 'fire' | 'close';

function step(phase: Phase, ev: Event): Phase {
  if (ev === 'close') return 'idle';
  if (ev === 'arm') return phase === 'open' ? 'open' : 'waiting';
  return phase === 'waiting' ? 'open' : phase;
}

interface Props {
  content: string;
  label: string;
  delay?: number;
}

export function Tooltip({ content, label, delay }: Props) {
  const [phase, send] = useReducer(step, 'idle');
  const tipId = `tip-${useId().replace(/:/g, '')}`;
  const wait = delay ?? 300;

  useEffect(() => {
    if (phase !== 'waiting') return;
    const t = setTimeout(() => send('fire'), wait);
    return () => clearTimeout(t);
  }, [phase, wait]);

  useEffect(() => {
    if (phase !== 'open') return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') send('close');
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [phase]);

  const open = phase === 'open';
  return (
    <div style={{ display: 'inline-block', position: 'relative' }}>
      <button
        {...(open ? { 'aria-describedby': tipId } : {})}
        onMouseEnter={() => send('arm')}
        onFocus={() => send('arm')}
        onMouseLeave={() => send('close')}
        onBlur={() => send('close')}
      >
        {label}
      </button>
      <div role="tooltip" id={tipId} hidden={!open}>
        {content}
      </div>
    </div>
  );
}
