// Phase 0, case 1: a component that mutates a ref inside useFrame.
//
// The compiler memoizes the useFrame callback on the values it closes over
// (`speed`). The failure being looked for is a stale closure: the callback
// captured with the first `speed` keeps running after the prop changes. The
// harness changes `speed` and checks the rotation rate follows.
import { useRef } from 'react';
import { useFrame } from '@react-three/fiber/webgpu';
import type { Mesh } from 'three';
import { probe } from './probe';

export function SpinningCube({ speed, color }: { speed: number; color: string }) {
  const mesh = useRef<Mesh>(null);
  useFrame((_state, delta) => {
    const m = mesh.current;
    if (!m) return;
    m.rotation.x += delta * speed * 0.5;
    m.rotation.y += delta * speed;
    probe.frames += 1;
    probe.cubeRotationY = m.rotation.y;
    probe.cubeSpeed = speed;
  });
  return (
    <mesh ref={mesh} position={[-1.2, 0, 0]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}
