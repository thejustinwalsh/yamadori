import { describe, it, expect } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Tabs } from './solution';

const TABS = [
  { id: 'one', label: 'One', content: <p>First panel</p> },
  { id: 'two', label: 'Two', content: <p>Second panel</p> },
  { id: 'three', label: 'Three', content: <p>Third panel</p> },
];

const tab = (name: string) => screen.getByRole('tab', { name });
const selectedNames = () =>
  screen
    .getAllByRole('tab')
    .filter((t) => t.getAttribute('aria-selected') === 'true')
    .map((t) => t.textContent);

function expectSelected(name: string, text: string) {
  expect(selectedNames()).toEqual([name]);
  for (const t of screen.getAllByRole('tab')) {
    const on = t.textContent === name;
    expect(t.getAttribute('aria-selected')).toBe(on ? 'true' : 'false');
    expect(t.tabIndex).toBe(on ? 0 : -1);
  }
  const panels = screen.getAllByRole('tabpanel');
  expect(panels.length).toBe(1);
  expect(panels[0].textContent).toBe(text);
}

describe('Tabs', () => {
  it('renders a tablist of tabs and selects the first by default', () => {
    render(<Tabs tabs={TABS} />);
    const list = screen.getByRole('tablist');
    expect(within(list).getAllByRole('tab').map((t) => t.textContent)).toEqual([
      'One',
      'Two',
      'Three',
    ]);
    for (const t of within(list).getAllByRole('tab')) expect(t.tagName).toBe('BUTTON');
    expectSelected('One', 'First panel');
  });

  it('honours defaultTabId', () => {
    render(<Tabs tabs={TABS} defaultTabId="two" />);
    expectSelected('Two', 'Second panel');
  });

  it('links the selected tab and the visible panel both ways', async () => {
    const user = userEvent.setup({ delay: null });
    render(<Tabs tabs={TABS} />);
    await user.click(tab('Three'));
    expectSelected('Three', 'Third panel');
    const t = tab('Three');
    const panel = screen.getByRole('tabpanel');
    expect(t.id).not.toBe('');
    expect(panel.id).not.toBe('');
    expect(t.getAttribute('aria-controls')).toBe(panel.id);
    expect(panel.getAttribute('aria-labelledby')).toBe(t.id);
    for (const other of screen.getAllByRole('tab')) {
      expect(other.id).not.toBe('');
      expect(other.getAttribute('aria-controls')).toBeTruthy();
    }
  });

  it('Tab from before the component lands on the selected tab only', async () => {
    const user = userEvent.setup({ delay: null });
    render(
      <>
        <button>before</button>
        <Tabs tabs={TABS} defaultTabId="two" />
      </>,
    );
    await user.click(screen.getByRole('button', { name: 'before' }));
    await user.tab();
    expect(document.activeElement).toBe(tab('Two'));
    await user.tab();
    const a = document.activeElement;
    expect(a === tab('One') || a === tab('Three')).toBe(false);
  });

  it('ArrowRight / ArrowLeft move focus and selection', async () => {
    const user = userEvent.setup({ delay: null });
    render(<Tabs tabs={TABS} />);
    await user.click(tab('One'));
    await user.keyboard('{ArrowRight}');
    expect(document.activeElement).toBe(tab('Two'));
    expectSelected('Two', 'Second panel');
    await user.keyboard('{ArrowRight}');
    expect(document.activeElement).toBe(tab('Three'));
    expectSelected('Three', 'Third panel');
    await user.keyboard('{ArrowLeft}');
    expect(document.activeElement).toBe(tab('Two'));
    expectSelected('Two', 'Second panel');
  });

  it('arrow keys wrap around at both ends', async () => {
    const user = userEvent.setup({ delay: null });
    render(<Tabs tabs={TABS} defaultTabId="three" />);
    await user.click(tab('Three'));
    await user.keyboard('{ArrowRight}');
    expect(document.activeElement).toBe(tab('One'));
    expectSelected('One', 'First panel');
    await user.keyboard('{ArrowLeft}');
    expect(document.activeElement).toBe(tab('Three'));
    expectSelected('Three', 'Third panel');
  });

  it('Home and End jump to the first and last tab', async () => {
    const user = userEvent.setup({ delay: null });
    render(<Tabs tabs={TABS} defaultTabId="two" />);
    await user.click(tab('Two'));
    await user.keyboard('{End}');
    expect(document.activeElement).toBe(tab('Three'));
    expectSelected('Three', 'Third panel');
    await user.keyboard('{Home}');
    expect(document.activeElement).toBe(tab('One'));
    expectSelected('One', 'First panel');
  });

  it('keeps ids unique across two instances with the same tab ids', async () => {
    const user = userEvent.setup({ delay: null });
    render(
      <>
        <Tabs tabs={TABS} />
        <Tabs tabs={TABS} defaultTabId="two" />
      </>,
    );
    const ids = [...document.querySelectorAll('[id]')].map((e) => e.id);
    expect(new Set(ids).size).toBe(ids.length);
    const [, second] = screen.getAllByRole('tablist');
    await user.click(within(second).getByRole('tab', { name: 'Three' }));
    for (const panel of screen.getAllByRole('tabpanel')) {
      const owner = document.getElementById(panel.getAttribute('aria-labelledby') ?? '');
      expect(owner).not.toBeNull();
      expect(owner!.getAttribute('aria-controls')).toBe(panel.id);
      expect(owner!.getAttribute('aria-selected')).toBe('true');
    }
    expect(screen.getAllByRole('tabpanel').map((p) => p.textContent)).toEqual([
      'First panel',
      'Third panel',
    ]);
  });
});
