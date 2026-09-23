// Counters the Phase 0 harness exposes on window.__phase0, so the browser
// check reads numbers instead of eyeballing a canvas. Written only from
// effects, useFrame callbacks and non-component helpers -- never from a
// render body, where the compiler would (correctly) refuse to compile a
// component that mutates module state.
export type Phase0Probe = {
  frames: number;
  cubeRotationY: number;
  cubeSpeed: number;
  geometryBuilds: number;
  geometryDisposals: number;
  ringCommits: number;
  ringTubularSegments: number;
  uncompiledGeometryBuilds: number;
  uncompiledRingCommits: number;
};

export const probe: Phase0Probe = {
  frames: 0,
  cubeRotationY: 0,
  cubeSpeed: 0,
  geometryBuilds: 0,
  geometryDisposals: 0,
  ringCommits: 0,
  ringTubularSegments: 0,
  uncompiledGeometryBuilds: 0,
  uncompiledRingCommits: 0,
};

declare global {
  interface Window {
    __phase0?: Phase0Probe;
  }
}
