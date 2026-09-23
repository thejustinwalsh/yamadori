import { useId, useReducer } from 'react';

// Recursive item component, a reducer over a leaf-id -> boolean record, the
// indeterminate flag set from a callback ref, and aria-labelledby instead of a
// wrapping <label>. onChange is fired from the click handler with the state
// the reducer is about to produce.
export type TreeNode = { id: string; label: string; children?: TreeNode[] };

type Checked = Record<string, boolean>;

const isLeaf = (n: TreeNode) => (n.children?.length ?? 0) === 0;

function collectLeaves(n: TreeNode, out: string[] = []): string[] {
  if (isLeaf(n)) out.push(n.id);
  else for (const c of n.children ?? []) collectLeaves(c, out);
  return out;
}

function apply(state: Checked, action: { leaves: string[] }): Checked {
  const allOn = action.leaves.every((id) => state[id]);
  const next = { ...state };
  for (const id of action.leaves) next[id] = !allOn;
  return next;
}

function Item({
  node,
  checked,
  toggle,
}: {
  node: TreeNode;
  checked: Checked;
  toggle: (leaves: string[]) => void;
}) {
  const labelId = useId();
  const leaves = collectLeaves(node);
  let count = 0;
  for (const id of leaves) if (checked[id]) count++;
  const mixed = count > 0 && count < leaves.length;
  return (
    <div role="group" aria-labelledby={labelId} style={{ paddingLeft: 16 }}>
      <input
        type="checkbox"
        aria-labelledby={labelId}
        checked={count === leaves.length}
        {...(mixed ? { 'aria-checked': 'mixed' as const } : {})}
        ref={(el) => {
          if (el) el.indeterminate = mixed;
        }}
        onChange={() => toggle(leaves)}
      />
      <span id={labelId}>{node.label}</span>
      {!isLeaf(node) &&
        (node.children ?? []).map((c) => <Item key={c.id} node={c} checked={checked} toggle={toggle} />)}
    </div>
  );
}

export function CheckboxTree(props: { nodes: TreeNode[]; onChange?: (checkedLeafIds: string[]) => void }) {
  const [checked, dispatch] = useReducer(apply, {});
  const order = props.nodes.flatMap((n) => collectLeaves(n));

  const toggle = (leaves: string[]) => {
    const next = apply(checked, { leaves });
    dispatch({ leaves });
    props.onChange?.(order.filter((id) => next[id]));
  };

  return (
    <div>
      {props.nodes.map((n) => (
        <Item key={n.id} node={n} checked={checked} toggle={toggle} />
      ))}
    </div>
  );
}
