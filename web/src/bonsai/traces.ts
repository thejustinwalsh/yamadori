// Ground circuit traces on the slab, one per probed service (BONSAI-VIZ §2).
// Axis-aligned, PCB-style, laid out purely from the trace count: no RNG, so
// the same set of services always draws the same board.
export type Seg = { x: number; z: number; w: number; d: number };

export const SLAB = { w: 4.8, d: 2.9, h: 0.26 } as const;

export function traceLayout(count: number): Seg[][] {
  const out: Seg[][] = [];
  const hw = SLAB.w / 2 - 0.18;
  const hd = SLAB.d / 2 - 0.18;
  const T = 0.035;
  for (let i = 0; i < count; i++) {
    const side = i % 2 === 0 ? 1 : -1; // alternate left / right of the trunk
    const lane = Math.floor(i / 2);
    const z0 = (lane % 2 === 0 ? 1 : -1) * (0.18 + 0.22 * lane);
    const x0 = side * 0.62;
    const x1 = side * (1.05 + 0.3 * lane);
    const z1 = Math.max(-hd, Math.min(hd, z0 + (z0 >= 0 ? 1 : -1) * (0.45 + 0.2 * lane)));
    const x2 = side * hw;
    out.push([
      { x: (x0 + x1) / 2, z: z0, w: Math.abs(x1 - x0), d: T },
      { x: x1, z: (z0 + z1) / 2, w: T, d: Math.abs(z1 - z0) + T },
      { x: (x1 + x2) / 2, z: z1, w: Math.abs(x2 - x1), d: T },
      // vias at each corner
      { x: x0, z: z0, w: 0.09, d: 0.09 },
      { x: x1, z: z1, w: 0.07, d: 0.07 },
    ]);
  }
  return out;
}
