// Wrong (strict): input.files is FileList | null and is passed on without a null check.
import { useState } from 'react';
import type { ChangeEvent, DragEvent } from 'react';

function matches(type: string, accept: string[]): boolean {
  return accept.some((a) =>
    a.endsWith('/*') ? type.startsWith(a.slice(0, -1)) : type === a,
  );
}

export function FileDropzone({
  accept,
  maxSizeBytes,
  onFiles,
}: {
  accept: string[];
  maxSizeBytes: number;
  onFiles: (files: File[]) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);

  const handle = (list: ArrayLike<File>) => {
    const ok: File[] = [];
    const bad: string[] = [];
    for (const f of Array.from(list)) {
      if (!matches(f.type, accept)) bad.push(`${f.name}: unsupported type`);
      else if (f.size > maxSizeBytes) bad.push(`${f.name}: too large`);
      else ok.push(f);
    }
    setErrors(bad);
    if (ok.length > 0) onFiles(ok);
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
  };
  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    handle(e.dataTransfer.files);
  };
  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const chosen: FileList = e.target.files;
    handle(chosen);
    e.target.value = '';
  };

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload files"
        onDragEnter={() => setDragging(true)}
        onDragOver={onDragOver}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {dragging ? 'Drop files here' : 'Drag files or choose them'}
      </div>
      <input type="file" multiple aria-label="Choose files" hidden onChange={onChange} />
      {errors.map((msg, i) => (
        <p key={i} role="alert">
          {msg}
        </p>
      ))}
    </div>
  );
}
