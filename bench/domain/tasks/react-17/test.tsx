import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Accordion } from './solution';

const ITEMS = [
  { id: 'ship', title: 'Shipping', content: <p>Ships in two days.</p> },
  { id: 'ret', title: 'Returns', content: <p>Thirty-day returns.</p> },
  { id: 'war', title: 'Warranty', content: <p>One year warranty.</p> },
];

const header = (name: string) => screen.getByRole('button', { name });
const expanded = () =>
  screen
    .getAllByRole('button')
    .filter((b) => b.getAttribute('aria-expanded') === 'true')
    .map((b) => b.textContent);
// queryAllByRole skips elements that are `hidden`, so this lists open panels.
const openRegions = () => screen.queryAllByRole('region').map((r) => r.textContent);

describe('Accordion', () => {
  it('starts with every section closed', () => {
    render(<Accordion items={ITEMS} />);
    for (const t of ['Shipping', 'Returns', 'Warranty']) {
      expect(header(t).tagName).toBe('BUTTON');
      expect(header(t).getAttribute('aria-expanded')).toBe('false');
    }
    expect(openRegions()).toEqual([]);
  });

  it('opens a section and links header and panel', async () => {
    const user = userEvent.setup({ delay: null });
    render(<Accordion items={ITEMS} />);
    await user.click(header('Returns'));
    expect(header('Returns').getAttribute('aria-expanded')).toBe('true');
    const region = screen.getByRole('region', { name: 'Returns' });
    expect(region.textContent).toBe('Thirty-day returns.');
    expect(region.id).not.toBe('');
    expect(header('Returns').getAttribute('aria-controls')).toBe(region.id);
    expect(region.getAttribute('aria-labelledby')).toBe(header('Returns').id);
    expect(openRegions()).toEqual(['Thirty-day returns.']);
  });

  it('closes an open section when its header is activated again', async () => {
    const user = userEvent.setup({ delay: null });
    render(<Accordion items={ITEMS} />);
    await user.click(header('Shipping'));
    await user.click(header('Shipping'));
    expect(header('Shipping').getAttribute('aria-expanded')).toBe('false');
    expect(openRegions()).toEqual([]);
  });

  it('keeps at most one section open by default', async () => {
    const user = userEvent.setup({ delay: null });
    render(<Accordion items={ITEMS} />);
    await user.click(header('Shipping'));
    await user.click(header('Warranty'));
    expect(expanded()).toEqual(['Warranty']);
    expect(openRegions()).toEqual(['One year warranty.']);
    await user.click(header('Returns'));
    expect(expanded()).toEqual(['Returns']);
    expect(openRegions()).toEqual(['Thirty-day returns.']);
  });

  it('keeps other sections open with allowMultiple', async () => {
    const user = userEvent.setup({ delay: null });
    render(<Accordion items={ITEMS} allowMultiple />);
    await user.click(header('Shipping'));
    await user.click(header('Warranty'));
    expect(expanded()).toEqual(['Shipping', 'Warranty']);
    expect(openRegions()).toEqual(['Ships in two days.', 'One year warranty.']);
    await user.click(header('Shipping'));
    expect(expanded()).toEqual(['Warranty']);
    expect(openRegions()).toEqual(['One year warranty.']);
  });

  it('works from the keyboard with Enter and Space', async () => {
    const user = userEvent.setup({ delay: null });
    render(<Accordion items={ITEMS} />);
    await user.tab();
    expect(document.activeElement).toBe(header('Shipping'));
    await user.keyboard('{Enter}');
    expect(expanded()).toEqual(['Shipping']);
    await user.keyboard(' ');
    expect(expanded()).toEqual([]);
    expect(openRegions()).toEqual([]);
  });
});
