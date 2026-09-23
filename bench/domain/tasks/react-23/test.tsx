import { describe, it, expect } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DataTable } from './solution';

type Row = { id: number; name: string; age: number };

const RAW: Row[] = [
  { id: 1, name: 'Mia', age: 31 },
  { id: 2, name: 'arlo', age: 25 },
  { id: 3, name: 'Zoe', age: 40 },
  { id: 4, name: 'Ben', age: 25 },
  { id: 5, name: 'Emma', age: 19 },
  { id: 6, name: 'Liam', age: 52 },
  { id: 7, name: 'Noah', age: 33 },
];
const frozen = (): readonly Row[] => Object.freeze(RAW.map((r) => Object.freeze({ ...r })));

function setup(pageSize = 3) {
  const user = userEvent.setup({ delay: null });
  const rows = frozen();
  render(<DataTable rows={rows} pageSize={pageSize} />);
  return { user, rows };
}

const bodyRows = () => {
  const table = screen.getByRole('table');
  const all = within(table).getAllByRole('row');
  return all.filter((r) => within(r).queryAllByRole('columnheader').length === 0);
};
const names = () => bodyRows().map((r) => within(r).getAllByRole('cell')[0].textContent);
const ages = () => bodyRows().map((r) => within(r).getAllByRole('cell')[1].textContent);
const header = (name: string) => screen.getByRole('columnheader', { name });
const sortOf = () => [header('Name').getAttribute('aria-sort'), header('Age').getAttribute('aria-sort')];
const btn = (name: string) => screen.getByRole('button', { name }) as HTMLButtonElement;
const pageText = () => screen.getByText(/^Page \d+ of \d+$/).textContent;

describe('DataTable', () => {
  it('shows the first page unsorted with two sortable headers', () => {
    setup();
    expect(within(screen.getByRole('table')).getAllByRole('columnheader').map((h) => h.textContent)).toEqual([
      'Name',
      'Age',
    ]);
    expect(within(header('Name')).getByRole('button', { name: 'Name' })).toBeTruthy();
    expect(within(header('Age')).getByRole('button', { name: 'Age' })).toBeTruthy();
    expect(sortOf()).toEqual(['none', 'none']);
    expect(names()).toEqual(['Mia', 'arlo', 'Zoe']);
    expect(ages()).toEqual(['31', '25', '40']);
    expect(pageText()).toBe('Page 1 of 3');
    expect(btn('Previous').disabled).toBe(true);
    expect(btn('Next').disabled).toBe(false);
  });

  it('pages with Previous / Next, disabled at the ends', async () => {
    const { user } = setup();
    await user.click(btn('Next'));
    expect(names()).toEqual(['Ben', 'Emma', 'Liam']);
    expect(pageText()).toBe('Page 2 of 3');
    expect(btn('Previous').disabled).toBe(false);
    await user.click(btn('Next'));
    expect(names()).toEqual(['Noah']);
    expect(pageText()).toBe('Page 3 of 3');
    expect(btn('Next').disabled).toBe(true);
    await user.click(btn('Previous'));
    expect(pageText()).toBe('Page 2 of 3');
  });

  it('sorts by age ascending, then descending, without mutating rows', async () => {
    const { user, rows } = setup(10);
    await user.click(btn('Age'));
    expect(sortOf()).toEqual(['none', 'ascending']);
    expect(names()).toEqual(['Emma', 'arlo', 'Ben', 'Mia', 'Noah', 'Zoe', 'Liam']);
    await user.click(btn('Age'));
    expect(sortOf()).toEqual(['none', 'descending']);
    expect(names()).toEqual(['Liam', 'Zoe', 'Noah', 'Mia', 'arlo', 'Ben', 'Emma']);
    await user.click(btn('Age'));
    expect(sortOf()).toEqual(['none', 'ascending']);
    expect(rows.map((r) => r.id)).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it('sorts by name with localeCompare and switches column', async () => {
    const { user } = setup(10);
    await user.click(btn('Age'));
    await user.click(btn('Age'));
    await user.click(btn('Name'));
    expect(sortOf()).toEqual(['ascending', 'none']);
    const expected = RAW.map((r) => r.name).sort((a, b) => a.localeCompare(b));
    expect(names()).toEqual(expected);
    await user.click(btn('Name'));
    expect(sortOf()).toEqual(['descending', 'none']);
    expect(names()).toEqual([...expected].reverse());
  });

  it('sorting applies before pagination', async () => {
    const { user } = setup(3);
    await user.click(btn('Age'));
    await user.click(btn('Age'));
    expect(pageText()).toBe('Page 1 of 3');
    expect(names()).toEqual(['Liam', 'Zoe', 'Noah']);
  });

  it('filters by name case-insensitively and returns to page 1', async () => {
    const { user } = setup(2);
    await user.click(btn('Next'));
    await user.click(btn('Next'));
    expect(pageText()).toBe('Page 3 of 4');
    await user.type(screen.getByLabelText('Filter'), 'M');
    expect(pageText()).toBe('Page 1 of 2');
    expect(names()).toEqual(['Mia', 'Emma']);
    await user.click(btn('Next'));
    expect(names()).toEqual(['Liam']);
    await user.type(screen.getByLabelText('Filter'), 'i');
    expect(pageText()).toBe('Page 1 of 1');
    expect(names()).toEqual(['Mia']);
    expect(btn('Previous').disabled).toBe(true);
    expect(btn('Next').disabled).toBe(true);
  });

  it('shows Page 1 of 1 and no body rows when nothing matches', async () => {
    const { user } = setup(3);
    await user.type(screen.getByLabelText('Filter'), 'xyz');
    expect(bodyRows()).toEqual([]);
    expect(pageText()).toBe('Page 1 of 1');
    expect(btn('Previous').disabled).toBe(true);
    expect(btn('Next').disabled).toBe(true);
  });
});
