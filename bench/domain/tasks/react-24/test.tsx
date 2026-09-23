import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FileDropzone } from './solution';

function file(name: string, type: string, size: number): File {
  return new File(['x'.repeat(size)], name, { type });
}

// A FileList stand-in: indexable, iterable, with length and item().
function fileList(files: File[]) {
  return Object.assign([...files], { item: (i: number) => files[i] ?? null });
}

function drop(zone: Element, files: File[]): boolean {
  return fireEvent.drop(zone, { dataTransfer: { files: fileList(files), types: ['Files'] } });
}

function setup(accept: string[] = ['image/*', 'application/pdf'], max = 100) {
  const onFiles = vi.fn<(files: File[]) => void>();
  render(<FileDropzone accept={accept} maxSizeBytes={max} onFiles={onFiles} />);
  const zone = screen.getByRole('button', { name: 'Upload files' });
  return { onFiles, zone };
}

const alerts = () => screen.queryAllByRole('alert').map((a) => a.textContent);
const names = (files: File[]) => files.map((f) => f.name);

describe('FileDropzone', () => {
  it('renders the drop area and a multi-file input labelled "Choose files"', () => {
    setup();
    const input = screen.getByLabelText('Choose files') as HTMLInputElement;
    expect(input.type).toBe('file');
    expect(input.multiple).toBe(true);
    expect(screen.queryByText('Drop files here')).toBeNull();
  });

  it('shows "Drop files here" only while a drag is over the drop area', () => {
    const { zone } = setup();
    fireEvent.dragEnter(zone, { dataTransfer: { types: ['Files'] } });
    expect(screen.queryByText('Drop files here')).not.toBeNull();
    fireEvent.dragLeave(zone, { dataTransfer: { types: ['Files'] } });
    expect(screen.queryByText('Drop files here')).toBeNull();
    fireEvent.dragOver(zone, { dataTransfer: { types: ['Files'] } });
    expect(screen.queryByText('Drop files here')).not.toBeNull();
    drop(zone, [file('a.png', 'image/png', 1)]);
    expect(screen.queryByText('Drop files here')).toBeNull();
  });

  it('prevents the browser default on dragover and drop', () => {
    const { zone } = setup();
    expect(fireEvent.dragOver(zone, { dataTransfer: { types: ['Files'] } })).toBe(false);
    expect(drop(zone, [file('a.png', 'image/png', 1)])).toBe(false);
  });

  it('passes accepted files in one call and matches wildcard and exact types', () => {
    const { onFiles, zone } = setup();
    drop(zone, [
      file('a.png', 'image/png', 10),
      file('notes.txt', 'text/plain', 10),
      file('b.jpeg', 'image/jpeg', 10),
      file('doc.pdf', 'application/pdf', 10),
    ]);
    expect(onFiles).toHaveBeenCalledTimes(1);
    expect(names(onFiles.mock.calls[0][0])).toEqual(['a.png', 'b.jpeg', 'doc.pdf']);
    expect(alerts()).toEqual(['notes.txt: unsupported type']);
  });

  it('accepts a file of exactly maxSizeBytes and rejects one byte more', () => {
    const { onFiles, zone } = setup(['image/*'], 100);
    drop(zone, [file('exact.png', 'image/png', 100), file('big.png', 'image/png', 101)]);
    expect(onFiles).toHaveBeenCalledTimes(1);
    expect(names(onFiles.mock.calls[0][0])).toEqual(['exact.png']);
    expect(alerts()).toEqual(['big.png: too large']);
  });

  it('reports the type before the size and skips onFiles when nothing is accepted', () => {
    const { onFiles, zone } = setup(['application/pdf'], 5);
    drop(zone, [file('huge.txt', 'text/plain', 50), file('huge.pdf', 'application/pdf', 50)]);
    expect(onFiles).not.toHaveBeenCalled();
    expect(alerts()).toEqual(['huge.txt: unsupported type', 'huge.pdf: too large']);
  });

  it('replaces the previous alerts with each new batch', () => {
    const { onFiles, zone } = setup();
    drop(zone, [file('x.txt', 'text/plain', 1), file('y.txt', 'text/plain', 1)]);
    expect(alerts()).toEqual(['x.txt: unsupported type', 'y.txt: unsupported type']);
    drop(zone, [file('z.gif', 'image/gif', 1)]);
    expect(alerts()).toEqual([]);
    expect(names(onFiles.mock.calls[0][0])).toEqual(['z.gif']);
  });

  it('checks files chosen through the input the same way', async () => {
    const { onFiles } = setup(['image/*'], 20);
    const user = userEvent.setup({ applyAccept: false });
    const input = screen.getByLabelText('Choose files') as HTMLInputElement;
    await user.upload(input, [
      file('ok.webp', 'image/webp', 20),
      file('clip.mp4', 'video/mp4', 5),
      file('wide.png', 'image/png', 21),
    ]);
    expect(onFiles).toHaveBeenCalledTimes(1);
    expect(names(onFiles.mock.calls[0][0])).toEqual(['ok.webp']);
    expect(alerts()).toEqual(['clip.mp4: unsupported type', 'wide.png: too large']);
  });
});
