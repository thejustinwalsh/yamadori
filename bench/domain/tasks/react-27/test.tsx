import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CheckboxTree } from './solution';
import type { TreeNode } from './solution';

const TREE: TreeNode[] = [
  {
    id: 'produce',
    label: 'Produce',
    children: [
      {
        id: 'fruit',
        label: 'Fruit',
        children: [
          { id: 'apple', label: 'Apple' },
          { id: 'pear', label: 'Pear' },
        ],
      },
      {
        id: 'veg',
        label: 'Vegetables',
        children: [
          { id: 'kale', label: 'Kale' },
          { id: 'leek', label: 'Leek' },
          {
            id: 'onions',
            label: 'Onions',
            children: [
              { id: 'red', label: 'Red onion' },
              { id: 'shallot', label: 'Shallot' },
            ],
          },
        ],
      },
    ],
  },
  { id: 'bread', label: 'Bread', children: [] },
];

const LABELS = ['Produce', 'Fruit', 'Apple', 'Pear', 'Vegetables', 'Kale', 'Leek', 'Onions', 'Red onion', 'Shallot', 'Bread'];

// getByLabelText rather than getByRole for speed; the first test checks the roles.
function box(label: string): HTMLInputElement {
  const hit = screen
    .getAllByLabelText(label)
    .find((e): e is HTMLInputElement => e instanceof HTMLInputElement && e.type === 'checkbox');
  if (!hit) throw new Error(`no checkbox labelled ${label}`);
  return hit;
}

type State = 'on' | 'off' | 'mixed' | 'inconsistent';
function state(label: string): State {
  const b = box(label);
  const aria = b.getAttribute('aria-checked') === 'mixed';
  if (b.indeterminate && aria && !b.checked) return 'mixed';
  if (b.indeterminate || aria) return 'inconsistent';
  return b.checked ? 'on' : 'off';
}

function setup() {
  const onChange = vi.fn<(ids: string[]) => void>();
  render(<CheckboxTree nodes={TREE} onChange={onChange} />);
  const user = userEvent.setup({ delay: null });
  const last = () => onChange.mock.calls[onChange.mock.calls.length - 1][0];
  return { onChange, user, last };
}

describe('CheckboxTree', () => {
  it('renders one unchecked native checkbox per node, labelled by its label', () => {
    setup();
    expect(screen.getAllByRole('checkbox').length).toBe(LABELS.length);
    for (const l of LABELS) {
      expect(screen.getByRole('checkbox', { name: l })).toBe(box(l));
      expect(state(l)).toBe('off');
    }
  });

  it('toggles a leaf and reports the checked leaves', async () => {
    const { user, onChange, last } = setup();
    await user.click(box('Kale'));
    expect(state('Kale')).toBe('on');
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(last()).toEqual(['kale']);
    await user.click(box('Kale'));
    expect(state('Kale')).toBe('off');
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(last()).toEqual([]);
  });

  it('shows a parent as indeterminate when some of its leaves are checked, and checked when all are', async () => {
    const { user } = setup();
    await user.click(box('Apple'));
    expect(state('Fruit')).toBe('mixed');
    expect(state('Produce')).toBe('mixed');
    expect(state('Vegetables')).toBe('off');
    await user.click(box('Pear'));
    expect(state('Fruit')).toBe('on');
    expect(state('Produce')).toBe('mixed');
    await user.click(box('Apple'));
    expect(state('Fruit')).toBe('mixed');
    await user.click(box('Pear'));
    expect(state('Fruit')).toBe('off');
    expect(state('Produce')).toBe('off');
  });

  it('derives every ancestor from all leaves below it, three levels deep', async () => {
    const { user } = setup();
    await user.click(box('Shallot'));
    expect(state('Onions')).toBe('mixed');
    expect(state('Vegetables')).toBe('mixed');
    expect(state('Produce')).toBe('mixed');
    await user.click(box('Red onion'));
    expect(state('Onions')).toBe('on');
    expect(state('Vegetables')).toBe('mixed');
    await user.click(box('Kale'));
    await user.click(box('Leek'));
    expect(state('Vegetables')).toBe('on');
    expect(state('Produce')).toBe('mixed');
  });

  it('checks and then unchecks every leaf below a parent', async () => {
    const { user, last } = setup();
    await user.click(box('Vegetables'));
    for (const l of ['Vegetables', 'Kale', 'Leek', 'Onions', 'Red onion', 'Shallot']) expect(state(l)).toBe('on');
    expect(state('Fruit')).toBe('off');
    expect(state('Produce')).toBe('mixed');
    expect(last()).toEqual(['kale', 'leek', 'red', 'shallot']);
    await user.click(box('Vegetables'));
    for (const l of ['Vegetables', 'Kale', 'Leek', 'Onions', 'Red onion', 'Shallot']) expect(state(l)).toBe('off');
    expect(last()).toEqual([]);
  });

  it('checks every leaf when an indeterminate parent is clicked', async () => {
    const { user, last } = setup();
    await user.click(box('Red onion'));
    await user.click(box('Bread'));
    await user.click(box('Produce'));
    for (const l of LABELS) expect(state(l)).toBe('on');
    expect(last()).toEqual(['apple', 'pear', 'kale', 'leek', 'red', 'shallot', 'bread']);
    await user.click(box('Produce'));
    expect(state('Bread')).toBe('on');
    expect(state('Produce')).toBe('off');
    expect(last()).toEqual(['bread']);
  });

  it('reports checked leaves in tree order, not click order', async () => {
    const { user, last } = setup();
    await user.click(box('Bread'));
    await user.click(box('Shallot'));
    await user.click(box('Apple'));
    expect(last()).toEqual(['apple', 'shallot', 'bread']);
  });
});
