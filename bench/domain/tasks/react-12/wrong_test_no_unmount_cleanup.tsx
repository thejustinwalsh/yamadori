// Wrong: the reset timer is restarted correctly but never cleared on
// unmount, so it outlives the component.
import { useRef, useState } from 'react';

export function useCopyToClipboard(resetMs = 2000) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const copy = async (text: string): Promise<boolean> => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      return false;
    }
    clearTimeout(timer.current);
    setCopied(true);
    timer.current = setTimeout(() => setCopied(false), resetMs);
    return true;
  };

  return { copied, copy };
}
