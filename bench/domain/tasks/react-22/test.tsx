import { describe, it, expect, vi } from 'vitest';
import { useState } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { StarRating } from './solution';

const radios = () => screen.getAllByRole('radio');
const checked = () =>
  radios()
    .filter((r) => r.getAttribute('aria-checked') === 'true')
    .map((r) => r.getAttribute('aria-label') ?? r.textContent);

function Stateful({ start, max, spy }: { start: number; max?: number; spy: (v: number) => void }) {
  const [v, setV] = useState(start);
  return (
    <StarRating
      value={v}
      max={max}
      onChange={(n) => {
        spy(n);
        setV(n);
      }}
    />
  );
}

describe('StarRating', () => {
  it('renders a Rating radiogroup with five named stars by default', () => {
    render(<StarRating value={0} onChange={() => {}} />);
    const group = screen.getByRole('radiogroup', { name: 'Rating' });
    expect(group.tabIndex).toBe(0);
    expect(radios().length).toBe(5);
    for (const [i, name] of ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'].entries()) {
      expect(screen.getByRole('radio', { name })).toBe(radios()[i]);
      expect(radios()[i].getAttribute('aria-checked')).toBe('false');
    }
  });

  it('checks exactly the star equal to value and honours max', () => {
    render(<StarRating value={3} max={7} onChange={() => {}} />);
    expect(radios().length).toBe(7);
    screen.getByRole('radio', { name: '7 stars' });
    for (const [i, r] of radios().entries()) {
      expect(r.getAttribute('aria-checked')).toBe(i === 2 ? 'true' : 'false');
    }
    expect(screen.getByRole('radio', { name: '3 stars' }).getAttribute('aria-checked')).toBe('true');
  });

  it('clicking star n calls onChange(n)', async () => {
    const user = userEvent.setup({ delay: null });
    const onChange = vi.fn<(v: number) => void>();
    render(<StarRating value={2} onChange={onChange} />);
    await user.click(screen.getByRole('radio', { name: '4 stars' }));
    expect(onChange.mock.calls).toEqual([[4]]);
    await user.click(screen.getByRole('radio', { name: '1 star' }));
    expect(onChange.mock.calls).toEqual([[4], [1]]);
  });

  it('is controlled: it only shows what value says', async () => {
    const user = userEvent.setup({ delay: null });
    const onChange = vi.fn<(v: number) => void>();
    render(<StarRating value={2} onChange={onChange} />);
    await user.click(screen.getByRole('radio', { name: '5 stars' }));
    expect(screen.getByRole('radio', { name: '2 stars' }).getAttribute('aria-checked')).toBe('true');
    expect(screen.getByRole('radio', { name: '5 stars' }).getAttribute('aria-checked')).toBe('false');
  });

  it('ArrowRight / ArrowLeft step by one within 1..max', async () => {
    const user = userEvent.setup({ delay: null });
    const spy = vi.fn<(v: number) => void>();
    render(<Stateful start={0} max={3} spy={spy} />);
    await user.tab();
    expect(document.activeElement).toBe(screen.getByRole('radiogroup', { name: 'Rating' }));
    await user.keyboard('{ArrowRight}');
    expect(spy.mock.calls).toEqual([[1]]);
    await user.keyboard('{ArrowRight}{ArrowRight}{ArrowRight}');
    expect(spy.mock.calls).toEqual([[1], [2], [3]]);
    expect(screen.getByRole('radio', { name: '3 stars' }).getAttribute('aria-checked')).toBe('true');
    await user.keyboard('{ArrowLeft}{ArrowLeft}{ArrowLeft}');
    expect(spy.mock.calls).toEqual([[1], [2], [3], [2], [1]]);
    expect(screen.getByRole('radio', { name: '1 star' }).getAttribute('aria-checked')).toBe('true');
  });

  it('readOnly: stars are aria-disabled and nothing calls onChange', async () => {
    const user = userEvent.setup({ delay: null });
    const onChange = vi.fn<(v: number) => void>();
    render(<StarRating value={2} onChange={onChange} readOnly />);
    for (const r of radios()) expect(r.getAttribute('aria-disabled')).toBe('true');
    await user.click(screen.getByRole('radio', { name: '4 stars' }));
    await user.tab();
    await user.keyboard('{ArrowRight}{ArrowLeft}');
    expect(onChange).not.toHaveBeenCalled();
    expect(checked().length).toBe(1);
  });
});
