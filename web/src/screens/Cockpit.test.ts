import { describe, expect, it } from 'vitest';
import { startedText } from './Cockpit';

describe('startedText (PROCESSES)', () => {
  it('prints the server ISO time with a space', () => {
    expect(startedText('2026-09-29T07:01:02')).toBe('2026-09-29 07:01:02');
  });
  it('reads the pre-fix "/Date(<ms>" a server older than 2026-09-29 sends', () => {
    const got = startedText('/Date(1790680397077');
    expect(got).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/);
    expect(got).not.toContain('Date');
  });
  it('a dash when absent', () => {
    expect(startedText('')).toBe('—');
    expect(startedText(null)).toBe('—');
  });
});
