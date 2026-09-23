// TSL materials for the tree. Palette from the frontmatter tokens only.
import * as THREE from 'three/webgpu';
import {
  abs, attribute, color, exp, float, fract, length, max, mix, mx_noise_float, normalView, normalWorld, positionWorld, smoothstep, uv, vec3,
} from 'three/tsl';
import { color as token } from '../../tokens/design';
import { U } from './uniforms';

type Vec3Node = THREE.Node<'vec3'>;
type FloatNode = THREE.Node<'float'>;

// Event tints: a request is the primary green, a tool call cyan, a concept
// seed draw the pale fixed green. Crimson stays reserved for alarms.
const EVENTS = [
  [U.request, token.primaryContainer],
  [U.tool, token.tertiaryContainer],
  [U.seed, token.primaryFixed],
] as const;

/** A band of light climbing the bark from the slab after each event. */
function climbing(): Vec3Node {
  let sum: Vec3Node = vec3(0, 0, 0);
  for (const [at, tint] of EVENTS) {
    const since = U.time.sub(at);
    const off = positionWorld.y.sub(since.mul(1.6).sub(0.15)).div(0.2);
    const band = exp(off.mul(off).mul(-1)).mul(exp(since.mul(-0.55)));
    sum = sum.add(color(tint).mul(band));
  }
  return sum;
}

/** Rings of light spreading across the slab from the trunk after each event. */
function ripples(d: FloatNode): Vec3Node {
  let sum: Vec3Node = vec3(0, 0, 0);
  for (const [at, tint] of EVENTS) {
    const since = U.time.sub(at);
    const radius = since.mul(1.25).add(0.55);
    const band = smoothstep(0.12, 0.0, abs(d.sub(radius))).mul(exp(since.mul(-0.8)));
    sum = sum.add(color(tint).mul(band));
  }
  return sum;
}

export function barkMaterial() {
  const m = new THREE.MeshStandardNodeMaterial();
  const st = attribute('aState', 'vec4'); // emissive, bleach, inert, depth
  const u = uv();
  // Bark grain: low-frequency noise in world space, so taper never stretches
  // it (the triplanar intent of BONSAI-VIZ §5, at a fraction of the cost).
  const grain = mx_noise_float(positionWorld.mul(vec3(9, 2.2, 9))).mul(0.5).add(0.5);
  const fissure = smoothstep(0.35, 0.05, abs(fract(u.x.mul(5).add(grain.mul(0.6))).sub(0.5)));
  const bark = mix(color(token.surfaceContainerHighest), color(token.surfaceBright), grain).mul(float(1).sub(fissure.mul(0.45)));
  const inertGrey = color(token.outline).mul(0.55);
  // Shari: bleach by pulling albedo to bone and raising roughness, never by
  // tinting (BONSAI-VIZ §5: tinting reads as plastic).
  const bone = color(token.inverseSurface).mul(float(0.82).add(grain.mul(0.18)));
  m.colorNode = mix(mix(bark, inertGrey, st.z.mul(0.55)), bone, st.y);
  m.roughnessNode = mix(float(0.78), float(1.0), st.y);
  m.metalnessNode = float(0.0);
  // Circuit threads: three longitudinal traces and sparse rings, lit only by
  // the measured activity channel.
  const thread = smoothstep(0.025, 0.0, abs(fract(u.x.mul(2)).sub(0.5)));
  const ring = smoothstep(0.06, 0.0, abs(fract(u.y.mul(0.9)).sub(0.5))).mul(0.7);
  const pattern = max(thread, ring);
  // Event pulses climb the same threads and rings, with a faint wash of the
  // whole surface so the band reads on the trunk, where no arcs run.
  m.emissiveNode = color(token.tertiaryContainer)
    .mul(pattern)
    .mul(st.x)
    .mul(3.2)
    .add(climbing().mul(pattern.mul(2.4).add(0.06)));
  return m;
}

export function foliageMaterial() {
  const m = new THREE.MeshStandardNodeMaterial();
  const n = mx_noise_float(positionWorld.mul(7)).mul(0.5).add(0.5);
  // INERT foliage (no hint signal exists): matte moss, unlit. When a hint
  // channel exists it will drive `foliageLive` toward primaryContainer.
  const leaf = mix(color(token.onPrimaryFixedVariant), color(token.outlineVariant), n);
  // Sun-side tops read lighter, undersides fall into shadow: form, not glow.
  const top = smoothstep(-0.2, 0.9, normalWorld.y);
  m.colorNode = mix(leaf.mul(0.55), leaf.mul(1.55), top);
  m.roughnessNode = float(0.95);
  m.metalnessNode = float(0);
  // Faceted, not smooth: the chisel-cut finish DESIGN.md asks of every shape.
  m.flatShading = true;
  return m;
}

export function capMaterial() {
  const m = new THREE.MeshStandardNodeMaterial();
  m.colorNode = color(token.secondaryContainer);
  m.emissiveNode = color(token.secondaryContainer).mul(2.6);
  m.roughnessNode = float(0.4);
  return m;
}

export function slabMaterial() {
  const m = new THREE.MeshStandardNodeMaterial();
  const n = mx_noise_float(positionWorld.mul(vec3(3, 12, 3))).mul(0.5).add(0.5);
  m.colorNode = mix(color(token.surfaceContainer), color(token.surfaceContainerHigh), n);
  m.roughnessNode = float(0.9);
  m.metalnessNode = float(0.05);
  // Raked rings around the tree on the slab's top face: faint at rest,
  // brightening with activity (U.glow), crossed by event ripples.
  const d = length(positionWorld.xz);
  const top = smoothstep(-0.03, -0.004, positionWorld.y);
  const rake = smoothstep(0.06, 0.0, abs(fract(d.mul(3)).sub(0.5)))
    .mul(smoothstep(0.7, 0.95, d))
    .mul(smoothstep(2.5, 1.9, d));
  m.emissiveNode = color(token.primaryContainer)
    .mul(rake.mul(U.glow.mul(0.9).add(0.03)))
    .add(ripples(d).mul(1.4))
    .mul(top);
  return m;
}

export function traceMaterial(ok: boolean) {
  const m = new THREE.MeshStandardNodeMaterial();
  const c = color(ok ? token.tertiaryContainer : token.secondaryContainer);
  m.colorNode = c.mul(0.3);
  // Healthy traces breathe with the model (dim at rest, bright at work) and
  // flash as a ripple passes; a failing trace holds a steady crimson.
  m.emissiveNode = ok ? c.mul(U.glow.mul(1.5).add(0.22)).add(ripples(length(positionWorld.xz)).mul(1.2)) : c.mul(2.0);
  m.roughnessNode = float(0.5);
  return m;
}

export function rimMaterial() {
  const m = new THREE.MeshStandardNodeMaterial();
  m.colorNode = color(token.secondaryContainer).mul(0.3);
  m.emissiveNode = color(token.secondaryContainer).mul(2.2);
  return m;
}

/**
 * A soft light shaft from the key light: additive, depth-test on, no depth
 * write, faded toward its silhouette edge and its far end. Screen-cheap
 * stand-in for raymarched volumetrics against a black void (BONSAI-VIZ §5).
 * Lighting, not data: it is constant and does not animate.
 */
export function shaftMaterial() {
  const m = new THREE.MeshBasicNodeMaterial();
  const u = uv();
  const edge = abs(normalView.z).pow(2.5);
  const along = smoothstep(0.0, 0.35, u.y).mul(smoothstep(1.0, 0.55, u.y));
  m.colorNode = color(token.primary);
  m.opacityNode = edge.mul(along).mul(0.045);
  m.transparent = true;
  m.depthWrite = false;
  m.blending = THREE.AdditiveBlending;
  m.side = THREE.DoubleSide;
  return m;
}
