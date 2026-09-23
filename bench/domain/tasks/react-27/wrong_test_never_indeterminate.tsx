// Wrong: a partly checked parent just shows as unchecked -- indeterminate is never set and aria-checked is never "mixed".
import { useState } from 'react';

export type TreeNode = { id: string; label: string; children?: TreeNode[] };

function leavesOf(node: TreeNode): string[] {
  if (!node.children || node.children.length === 0) return [node.id];
  return node.children.flatMap(leavesOf);
}

function Box({
  label,
  checked,
  onToggle,
}: {
  label: string;
  checked: boolean;
  mixed: boolean;
  onToggle: () => void;
}) {
  return (
    <label>
      <input
        type="checkbox"
        checked={checked}
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
