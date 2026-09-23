// The tokonoma tree, rendered with r3f 10's default WebGPU renderer and
// three's own TSL post-processing (selective bloom on the MRT emissive
// target, BONSAI-VIZ §5). Where WebGPU is unavailable, three's WebGPURenderer
// falls back to its WebGL2 backend by itself (WebGPURenderer.js: getFallback
// when the WebGPU backend fails to init), so the tree always draws.
//
// frameloop="demand", driven by LiveDriver: every frame while the model is
// working or a pulse is in flight, IDLE_FPS while it rests (the slow breath
// and the calm sway still move), and with reduced motion only while a value
// is easing -- the tree stays, the motion stops (§6).
import { Canvas, useFrame, useRenderPipeline, useThree } from '@react-three/fiber/webgpu';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three/webgpu';
import { emissive, mrt, output } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { color as token } from '../../tokens/design';
import { allocateBark, BLOBS, padBlobs, padCount, writeBark } from '../bark';
import { LiveBus, windOffset, type WindState } from '../live';
import type { TreeParams } from '../mapping';
import { poseFrames } from '../pose';
import type { Skeleton } from '../skeleton';
import { ParamSprings } from '../spring';
import { SLAB, traceLayout } from '../traces';
import { barkMaterial, capMaterial, foliageMaterial, rimMaterial, shaftMaterial, slabMaterial, traceMaterial } from './materials';
import { U } from './uniforms';

/** Frame rate while the model rests: enough for the breath and the calm sway. */
const IDLE_FPS = 30;

export type SceneStats = {
  frames: number;
  /** frames drawn over the last second */
  fps: number;
  settled: boolean;
  lastFrameMs: number;
  backend: string;
  branches: number;
  vertices: number;
  pads: number;
};

declare global {
  interface Window {
    __bonsai?: SceneStats;
  }
}

const stats: SceneStats = { frames: 0, fps: 0, settled: false, lastFrameMs: 0, backend: '?', branches: 0, vertices: 0, pads: 0 };

// ------------------------------------------------------------------- tree

/**
 * The tree's three.js objects and springs, owned outside React.
 *
 * React Compiler bails out of a component that assigns to a property of a
 * value returned from a hook (`built.geo.getAttribute(n).needsUpdate = true`
 * on a useMemo result was exactly that, found in the Phase 0 build). The r3f
 * idiom is mutation, so the mutation lives here, behind methods, and the
 * component holds one instance for its lifetime (a Tree is keyed by genome,
 * so its skeleton never changes).
 */
class TreeRig {
  readonly group = new THREE.Group();
  private readonly buf;
  private readonly geo = new THREE.BufferGeometry();
  private readonly bark;
  private readonly foliage;
  private readonly caps;
  private readonly twigs: number[];
  private readonly springs: ParamSprings;
  private readonly blobs: Float32Array;
  private nBlobs = 0;
  /** bark positions as posed, before the wind moves them */
  private readonly rest: Float32Array;
  /** tip caps as posed: x, y, z, scale */
  private readonly capRest: Float32Array;
  private readonly off = new Float32Array(3);
  private posed = false;
  private swayed = false;
  private readonly m = new THREE.Matrix4();

  constructor(private readonly sk: Skeleton, initial: TreeParams['branches']) {
    this.buf = allocateBark(sk);
    this.rest = new Float32Array(this.buf.position.length);
    const g = this.geo;
    g.setAttribute('position', new THREE.BufferAttribute(this.buf.position, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('normal', new THREE.BufferAttribute(this.buf.normal, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('uv', new THREE.BufferAttribute(this.buf.uv, 2).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aState', new THREE.BufferAttribute(this.buf.state, 4).setUsage(THREE.DynamicDrawUsage));
    g.setIndex(new THREE.BufferAttribute(this.buf.index, 1));
    this.bark = new THREE.Mesh(g, barkMaterial());
    this.bark.castShadow = true;
    this.bark.receiveShadow = true;
    this.bark.frustumCulled = false;

    const pads = padCount(sk);
    const blob = new THREE.IcosahedronGeometry(1, 1);
    blob.scale(1, 0.5, 1); // flattened clouds, the bonsai pad silhouette
    this.foliage = new THREE.InstancedMesh(blob, foliageMaterial(), pads * BLOBS);
    this.foliage.castShadow = true;
    this.foliage.receiveShadow = true;
    this.foliage.frustumCulled = false;
    this.blobs = new Float32Array(pads * BLOBS * 4);

    this.twigs = sk.branches.filter((b) => b.kind === 'twig').map((b) => b.id);
    this.caps = new THREE.InstancedMesh(new THREE.SphereGeometry(0.045, 12, 8), capMaterial(), this.twigs.length);
    this.capRest = new Float32Array(this.twigs.length * 4);
    this.caps.frustumCulled = false;
    this.group.add(this.bark, this.foliage, this.caps);

    // Everything starts retracted: the first frames are the tree growing into
    // what was measured, never a pop to it.
    this.springs = new ParamSprings(initial.map(retracted));
  }

  /**
   * Advance springs and, while they move, re-pose the rest shape; then lay
   * the wind over it. Returns true while a spring is moving.
   */
  step(targets: TreeParams['branches'], dt: number, wind: WindState): boolean {
    const t0 = performance.now();
    const moving = this.springs.step(targets, dt);
    const repose = moving || !this.posed;
    if (repose) {
      const cur = Array.from({ length: this.springs.n }, (_, i) => this.springs.get(i));
      const frames = poseFrames(this.sk, cur);
      writeBark(this.buf, this.sk, frames, cur);
      this.rest.set(this.buf.position);
      for (const name of ['normal', 'uv', 'aState'] as const) this.geo.getAttribute(name).needsUpdate = true;
      this.nBlobs = padBlobs(this.sk, frames, cur, this.blobs);
      this.twigs.forEach((id, i) => {
        const f = frames[id]!;
        const c = cur[id]!.cap;
        this.capRest.set([f.end[0], f.end[1], f.end[2], c < 0.01 ? 0 : 0.6 + 0.9 * c], i * 4);
      });
      this.posed = true;
    }
    // The wind is laid over the rest shape every frame it blows, and once
    // more when it drops to nothing, so the tree settles back exactly.
    if (repose || wind.amp > 0 || this.swayed) this.place(wind);
    this.swayed = wind.amp > 0;

    stats.lastFrameMs = performance.now() - t0;
    stats.branches = this.sk.branches.length;
    stats.vertices = this.buf.vertexCount;
    stats.pads = this.nBlobs / BLOBS;
    stats.settled = !moving;
    return moving;
  }

  private place(wind: WindState) {
    const P = this.buf.position;
    const R = this.rest;
    const o = this.off;
    const still = wind.amp === 0;
    if (still) P.set(R);
    else {
      for (let i = 0; i < R.length; i += 3) {
        windOffset(R[i]!, R[i + 1]!, R[i + 2]!, wind, o);
        P[i] = R[i]! + o[0]!;
        P[i + 1] = R[i + 1]! + o[1]!;
        P[i + 2] = R[i + 2]! + o[2]!;
      }
    }
    this.geo.getAttribute('position').needsUpdate = true;

    const B = this.blobs;
    for (let i = 0; i < this.nBlobs; i++) {
      const sc = B[i * 4 + 3]!;
      if (still) o.fill(0);
      else windOffset(B[i * 4]!, B[i * 4 + 1]!, B[i * 4 + 2]!, wind, o);
      this.m.makeScale(sc, sc, sc).setPosition(B[i * 4]! + o[0]!, B[i * 4 + 1]! + o[1]!, B[i * 4 + 2]! + o[2]!);
      this.foliage.setMatrixAt(i, this.m);
    }
    this.foliage.instanceMatrix.needsUpdate = true;

    const C = this.capRest;
    for (let i = 0; i < this.twigs.length; i++) {
      const sc = C[i * 4 + 3]!;
      if (still) o.fill(0);
      else windOffset(C[i * 4]!, C[i * 4 + 1]!, C[i * 4 + 2]!, wind, o);
      this.m.makeScale(sc, sc, sc).setPosition(C[i * 4]! + o[0]!, C[i * 4 + 1]! + o[1]!, C[i * 4 + 2]! + o[2]!);
      this.caps.setMatrixAt(i, this.m);
    }
    this.caps.instanceMatrix.needsUpdate = true;
  }

  dispose() {
    this.geo.dispose();
    this.foliage.geometry.dispose();
    this.caps.geometry.dispose();
    for (const o of [this.bark, this.foliage, this.caps]) (o.material as THREE.Material).dispose();
  }
}

const retracted = (p: TreeParams['branches'][number]) => ({ ...p, growth: 0, foliage: 0, emissive: 0, cap: 0 });

function Tree({
  sk,
  target,
  dying,
  onGone,
  bus,
  still,
}: {
  sk: Skeleton;
  target: TreeParams;
  dying: boolean;
  onGone?: () => void;
  bus: LiveBus;
  still: boolean;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const [rig] = useState(() => new TreeRig(sk, target.branches));
  useEffect(() => () => rig.dispose(), [rig]);

  // A dying tree (its genome was replaced) retracts to nothing, then leaves.
  const goal = dying ? target.branches.map(retracted) : target.branches;
  const goalRef = useRef(goal);
  const dyingRef = useRef(dying);
  const gone = useRef(false);
  useEffect(() => {
    goalRef.current = goal;
    dyingRef.current = dying;
    invalidate();
  }, [goal, dying, invalidate]);

  useFrame((_s, dt) => {
    const moving = rig.step(goalRef.current, dt, bus.windState(still));
    if (moving) invalidate();
    else if (dyingRef.current && !gone.current) {
      gone.current = true;
      onGone?.();
    }
  });

  return <primitive object={rig.group} />;
}

// ------------------------------------------------------------------- slab

function Slab({ traces, alarm }: { traces: { name: string; ok: boolean }[]; alarm: number }) {
  const invalidate = useThree((s) => s.invalidate);
  const group = useMemo(() => {
    const g = new THREE.Group();
    const slab = new THREE.Mesh(new THREE.BoxGeometry(SLAB.w, SLAB.h, SLAB.d), slabMaterial());
    slab.position.y = -SLAB.h / 2;
    slab.receiveShadow = true;
    g.add(slab);
    // The rim glows crimson only while a GPU is below its free-memory floor.
    const rim = new THREE.Mesh(new THREE.BoxGeometry(SLAB.w + 0.02, 0.025, SLAB.d + 0.02), rimMaterial());
    rim.position.y = -0.012;
    rim.name = 'rim';
    g.add(rim);
    return g;
  }, []);
  const traceGroup = useMemo(() => {
    const g = new THREE.Group();
    const okMat = traceMaterial(true);
    const badMat = traceMaterial(false);
    traceLayout(traces.length).forEach((segs, i) => {
      for (const s of segs) {
        const box = new THREE.Mesh(new THREE.BoxGeometry(s.w, 0.012, s.d), traces[i]!.ok ? okMat : badMat);
        box.position.set(s.x, 0.006, s.z);
        g.add(box);
      }
    });
    return g;
  }, [traces]);
  useEffect(() => {
    const rim = group.getObjectByName('rim');
    if (rim) rim.visible = alarm > 0;
    invalidate();
  }, [alarm, group, invalidate]);
  useEffect(() => {
    invalidate();
    return () => traceGroup.traverse((o) => (o as THREE.Mesh).geometry?.dispose());
  }, [traceGroup, invalidate]);
  return (
    <>
      <primitive object={group} />
      <primitive object={traceGroup} />
    </>
  );
}

// -------------------------------------------------------- lights, camera

function Stage({ haze }: { haze: number }) {
  const { camera, gl, invalidate } = useThree();
  useEffect(() => invalidate(), [haze, invalidate]);

  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement);
    controls.target.set(0, 0.95, 0);
    controls.enablePan = false;
    controls.minDistance = 3.5;
    controls.maxDistance = 11;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.update();
    const onChange = () => invalidate();
    controls.addEventListener('change', onChange);
    return () => {
      controls.removeEventListener('change', onChange);
      controls.dispose();
    };
  }, [camera, gl, invalidate]);

  return (
    <>
      <color attach="background" args={[token.surfaceContainerLowest]} />
      {/* GPU memory pressure thickens the haze (§2: slab tilt, thermal haze). */}
      <fog attach="fog" args={[token.surfaceContainerLowest, 7 + (1 - haze) * 7, 15 + (1 - haze) * 14]} />
      <hemisphereLight args={[token.inverseSurface, token.surfaceContainerLowest, 0.35]} />
      {/* Warm key, high: the shafts. */}
      <directionalLight
        position={KEY_LIGHT.toArray()}
        intensity={3.4}
        color={token.primary}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-4}
        shadow-camera-right={4}
        shadow-camera-top={5}
        shadow-camera-bottom={-3}
        shadow-bias={-0.0004}
      />
      {/* Cold cyan rim, behind-left, separating bark from the void. */}
      <directionalLight position={[-5, 3, -4.5]} intensity={1.1} color={token.tertiaryContainer} />
    </>
  );
}

const KEY_LIGHT = new THREE.Vector3(3.2, 7.5, 2.4);

function Shafts() {
  const group = useMemo(() => {
    const g = new THREE.Group();
    const mat = shaftMaterial();
    const dir = KEY_LIGHT.clone().normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    // Three overlapping cones, slightly offset: layered depth reads as volume.
    [
      [0, 1.0, 0.1, 0.95],
      [0.45, 0.7, 0.06, 0.55],
      [-0.5, 1.3, 0.05, 0.4],
    ].forEach(([ox, oy, rt, rb]) => {
      const cone = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, 9, 32, 1, true), mat);
      cone.quaternion.copy(q);
      cone.position.copy(dir.clone().multiplyScalar(4.1)).add(new THREE.Vector3(ox!, oy!, 0));
      cone.renderOrder = 10;
      g.add(cone);
    });
    return g;
  }, []);
  return <primitive object={group} />;
}

function Bloom() {
  useRenderPipeline(
    ({ renderPipeline, passes }) => {
      const scene = passes.scenePass.getTextureNode('output');
      const glow = passes.scenePass.getTextureNode('emissive');
      // Selective: only what is emissive blooms. Full-scene bloom washes the
      // bark out (§5).
      renderPipeline.outputNode = scene.add(bloom(glow, 1.1, 0.5, 0.0));
    },
    ({ passes }) => {
      passes.scenePass.setMRT(mrt({ output, emissive }));
    },
  );
  return null;
}

/**
 * Ticks the live bus, writes the scene uniforms, and decides when the next
 * frame is drawn (see the header). Registered before the trees, so they read
 * this frame's wind.
 */
function LiveDriver({ bus, still }: { bus: LiveBus; still: boolean }) {
  const invalidate = useThree((st) => st.invalidate);
  const timer = useRef<number | null>(null);
  useEffect(() => bus.listen(() => invalidate()), [bus, invalidate]);
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
  useFrame((_s, dt) => {
    bus.tick(dt, still);
    U.time.value = bus.t;
    U.glow.value = bus.glowNow(still);
    // Reduced motion: no travelling pulses; brightness still eases.
    U.request.value = still ? -100 : bus.fired.request;
    U.tool.value = still ? -100 : bus.fired.tool;
    U.seed.value = still ? -100 : bus.fired.seed;
    if (still) {
      if (bus.animating()) invalidate();
    } else if (bus.lively()) {
      invalidate();
    } else if (timer.current === null) {
      timer.current = window.setTimeout(() => {
        timer.current = null;
        invalidate();
      }, 1000 / IDLE_FPS);
    }
  });
  return null;
}

function Stats() {
  const renderer = useThree((s) => s.renderer as unknown as { backend?: { isWebGPUBackend?: boolean } });
  useEffect(() => {
    stats.backend = renderer?.backend?.isWebGPUBackend ? 'WebGPU' : 'WebGL2 fallback';
    window.__bonsai = stats;
  }, [renderer]);
  const win = useRef<number[]>([]);
  useFrame(() => {
    stats.frames += 1;
    const now = performance.now();
    const w = win.current;
    w.push(now);
    while (w.length && now - w[0]! > 1000) w.shift();
    stats.fps = w.length;
  });
  return null;
}

export type TreeLayer = { key: string; sk: Skeleton; target: TreeParams; dying: boolean };

export function BonsaiCanvas({
  layers,
  scene,
  onLayerGone,
  bus,
  still,
}: {
  layers: TreeLayer[];
  scene: TreeParams['scene'];
  onLayerGone: (key: string) => void;
  bus: LiveBus;
  /** prefers-reduced-motion: the tree stays, its motion stops */
  still: boolean;
}) {
  return (
    <Canvas
      frameloop="demand"
      dpr={[1, 1.5]}
      shadows
      camera={{ position: [4.1, 1.7, 5.0], fov: 30, near: 0.1, far: 60 }}
      onCreated={(s) => {
        const r = s.renderer as unknown as THREE.WebGPURenderer;
        r.toneMapping = THREE.ACESFilmicToneMapping;
        r.toneMappingExposure = 1.05;
      }}
    >
      <Stage haze={scene.haze} />
      <Bloom />
      <Shafts />
      <Stats />
      <LiveDriver bus={bus} still={still} />
      <Slab traces={scene.traces} alarm={scene.alarm} />
      {layers.map((l) => (
        <Tree key={l.key} sk={l.sk} target={l.target} dying={l.dying} onGone={() => onLayerGone(l.key)} bus={bus} still={still} />
      ))}
    </Canvas>
  );
}
