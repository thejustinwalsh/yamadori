import { useCallback, useEffect, useRef, useState } from 'react';

export function useCopyToClipboard(resetMs: number = 2000): {
  copied: boolean;
  copy: (text: string) => Promise<boolean>;
} {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clear = () => {
    if (timer.current !== null) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  };

  useEffect(() => clear, []);

  const copy = useCallback(
    async (text: string) => {
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        return false;
      }
      clear();
      setCopied(true);
      timer.current = setTimeout(() => {
        timer.current = null;
        setCopied(false);
      }, resetMs);
      return true;
    },
    [resetMs],
  );

  return { copied, copy };
}
