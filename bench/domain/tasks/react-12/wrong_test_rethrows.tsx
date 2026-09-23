// Wrong: a rejected clipboard write propagates out of copy() instead of
// resolving to false.
import { useEffect, useRef, useState } from 'react';

export function useCopyToClipboard(resetMs = 2000) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => () => clearTimeout(timer.current), []);

  const copy = async (text: string): Promise<boolean> => {
    await navigator.clipboard.writeText(text);
    clearTimeout(timer.current);
    setCopied(true);
    timer.current = setTimeout(() => setCopied(false), resetMs);
    return true;
  };

  return { copied, copy };
}
