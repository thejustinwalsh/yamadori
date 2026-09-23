// Phase 0, case 2: geometry derived in render, with no useMemo.
//
// Without the compiler, `makeRing(...)` runs on every render and allocates a
// new BufferGeometry each time. With the compiler it should be cached on
// (radius, segments) and survive re-renders caused by `label` alone.
//
// Two copies of the same component: one compiled, one opted out with
// 'use no memo'. The opted-out copy is the control -- its build counter must
// climb with every render, which proves the compiled copy's flat counter is
// the compiler's doing and not an accident of React skipping renders.
import { useEffect } from 'react';
import { TorusGeometry } from 'three';
import { probe } from './probe';
import { color } from '../tokens/design';

type RingProps = { radius: number; segments: number; label: string };

function makeRing(radius: number, segments: number): TorusGeometry {
  probe.geometryBuilds += 1;
  return new TorusGeometry(radius, 0.12, 12, segments);
}

function makeRingUncompiled(radius: number, segments: number): TorusGeometry {
  probe.uncompiledGeometryBuilds += 1;
  return new TorusGeometry(radius, 0.12, 12, segments);
}

export function DerivedRing({ radius, segments, label }: RingProps) {
  const geometry = makeRing(radius, segments);
  useEffect(() => {
    probe.ringCommits += 1;
    probe.ringTubularSegments = geometry.parameters.tubularSegments;
  });
  // A geometry made in render must be disposed by whoever made it; r3f only
  // disposes what it constructed itself.
  useEffect(
    () => () => {
      geometry.dispose();
      probe.geometryDisposals += 1;
    },
    [geometry],
  );
  return (
    <mesh geometry={geometry} name={label} position={[1.2, 0, 0]}>
      <meshStandardMaterial color={color.primaryContainer} emissive={color.primaryContainer} emissiveIntensity={0.25} />
    </mesh>
  );
}

export function DerivedRingUncompiled({ radius, segments, label }: RingProps) {
  'use no memo';
  const geometry = makeRingUncompiled(radius, segments);
  useEffect(() => {
    probe.uncompiledRingCommits += 1;
  });
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <mesh geometry={geometry} name={label} position={[1.2, -1.6, 0]} scale={0.5}>
      <meshStandardMaterial color={color.tertiaryContainer} />
    </mesh>
  );
}
