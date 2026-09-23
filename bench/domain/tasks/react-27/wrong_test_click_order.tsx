// Wrong: onChange reports the checked leaves in the order they were checked (Set insertion order), not in tree order.
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
    onChange?.([...next]);
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

  const renderNode = (node: TreeNode) => {
    const leaves = leavesOf(node);
    const on = leaves.filter((id) => checked.has(id)).length;
    const isParent = !!node.children && node.children.length > 0;
    return (
      <li key={node.id}>
        <Box
          label={node.label}
          checked={on === leaves.length}
          mixed={on > 0 && on < leaves.length}
          onToggle={() => toggle(node)}
        />
        {isParent && <ul>{node.children!.map(renderNode)}</ul>}
      </li>
    );
  };

  return <ul>{nodes.map(renderNode)}</ul>;
}
