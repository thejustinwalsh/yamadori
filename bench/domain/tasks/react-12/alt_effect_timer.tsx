import { useEffect, useState } from 'react';

// The timer lives in an effect keyed on a copy counter instead of a ref; the
// effect's own cleanup restarts the countdown and handles unmount.
export function useCopyToClipboard(resetMs = 2000) {
  const [stamp, setStamp] = useState(0);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (stamp === 0) return;
    const id = setTimeout(() => setCopied(false), resetMs);
    return () => clearTimeout(id);
  }, [stamp, resetMs]);

  function copy(text: string): Promise<boolean> {
    return navigator.clipboard.writeText(text).then(
      () => {
        setCopied(true);
        setStamp((n) => n + 1);
        return true;
      },
      () => false,
    );
  }

  return { copied, copy };
}
