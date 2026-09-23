// Loaded by vitest before every react task's test.tsx (grade_react.py).
// globals are off, so @testing-library/react cannot register its own
// afterEach(cleanup); do it here, and reset everything a test may have faked.
import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  try {
    window.localStorage.clear();
  } catch {
    // a test stubbed storage away; nothing to clear
  }
  document.body.innerHTML = '';
});
