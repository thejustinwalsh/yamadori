// Wrong: a parent is indeterminate only when some DIRECT child is fully checked, so a grandparent over a partly checked subtree shows as unchecked.
import { useEffect, useRef, useState } from 'react';

export type TreeNode = { id: string; label: string; children?: TreeNode[] };

function leavesOf(node: TreeNode): string[] {
  if (!node.children || node.children.length === 0) return [node.id];
  return node.children.flatMap(leavesOf);
}

function Box({
  label,
  checked,
  mixed,
  onToggle,
}: {
  label: string;
  checked: boolean;
  mixed: boolean;
  onToggle: () => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = mixed;
  });
  return (
    <label>
      <input
        ref={ref}
        type="checkbox"
        checked={checked}
        aria-checked={mixed ? 'mixed' : undefined}
        onChange={onToggle}
      />
      {label}
    </label>
  );
}

export function CheckboxTree({
  nodes,
  onChange,
}: {
  nodes: TreeNode[];
  onChange?: (checkedLeafIds: string[]) => void;
}) {
  const [checked, setChecked] = useState<ReadonlySet<string>>(new Set());

  const commit = (next: Set<string>) => {
    setChecked(next);
    onChange?.(nodes.flatMap(leavesOf).filter((id) => next.has(id)));
  };

  const toggle = (node: TreeNode) => {
    const leaves = leavesOf(node);
    const all = leaves.every((id) => checked.has(id));
    const next = new Set(checked);
    for (const id of leaves) {
      if (all) next.delete(id);
      else next.add(id);
    }
    commit(next);
  };

  // a parent looks only at whether each direct child is checked
  const status = (node: TreeNode): boolean => {
    if (!node.children || node.children.length === 0) return checked.has(node.id);
    return node.children.every(status);
  };

  const renderNode = (node: TreeNode) => {
    const isParent = !!node.children && node.children.length > 0;
    const kids = isParent ? node.children!.map(status) : [];
    const on = status(node);
    return (
      <li key={node.id}>
        <Box
          label={node.label}
          checked={on}
          mixed={!on && kids.some(Boolean)}
          onToggle={() => toggle(node)}
        />
        {isParent && <ul>{node.children!.map(renderNode)}</ul>}
      </li>
    );
  };

  return <ul>{nodes.map(renderNode)}</ul>;
}
