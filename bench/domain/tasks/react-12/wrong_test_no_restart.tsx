// Wrong: each copy starts a new timer without clearing the previous one, so
// the first timer flips `copied` back early.
import { useEffect, useRef, useState } from 'react';

export function useCopyToClipboard(resetMs = 2000) {
  const [copied, setCopied] = useState(false);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const copy = async (text: string): Promise<boolean> => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      timers.current.push(setTimeout(() => setCopied(false), resetMs));
      return true;
    } catch {
      return false;
    }
  };

  return { copied, copy };
}
