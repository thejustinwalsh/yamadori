// The Phase 0 gate (design/STACK.md): a spinning cube with the compiler on,
// a ref mutated in useFrame, geometry derived in render. Reached at
// /dash/phase0. Every number it shows is read from `probe`, which the
// browser check also reads through window.__phase0.
import { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber/webgpu';
import { SpinningCube } from './SpinningCube';
import { DerivedRing, DerivedRingUncompiled } from './DerivedRing';
import { probe } from './probe';
import { color } from '../tokens/design';

export function Phase0() {
  const [speed, setSpeed] = useState(1);
  const [segments, setSegments] = useState(24);
  // An unrelated state change that re-renders both rings with identical
  // geometry props. The compiled ring must not rebuild on it.
  const [tick, setTick] = useState(0);
  const [snapshot, setSnapshot] = useState(() => ({ ...probe }));

  useEffect(() => {
    window.__phase0 = probe;
    const id = window.setInterval(() => {
      setTick((t) => t + 1);
      setSnapshot({ ...probe });
    }, 500);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div style={{ display: 'grid', gridTemplateRows: '360px auto', gap: 12, padding: 16 }}>
      <Canvas camera={{ position: [0, 0, 5] }} dpr={[1, 1.5]}>
        <ambientLight intensity={0.3} />
        <directionalLight position={[3, 4, 5]} intensity={1.5} />
        <SpinningCube speed={speed} color={color.tertiaryContainer} />
        <DerivedRing radius={0.6} segments={segments} label={`ring-${tick}`} />
        <DerivedRingUncompiled radius={0.6} segments={segments} label={`ring-${tick}`} />
      </Canvas>
      <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
        <div>
          <button data-testid="speed" onClick={() => setSpeed((s) => (s === 1 ? 4 : 1))}>
            speed = {speed} (toggle 1/4)
          </button>{' '}
          <button data-testid="segments" onClick={() => setSegments((s) => s + 8)}>
            segments = {segments} (+8)
          </button>
        </div>
        <pre data-testid="probe">{JSON.stringify({ tick, ...snapshot }, null, 2)}</pre>
      </div>
    </div>
  );
}
