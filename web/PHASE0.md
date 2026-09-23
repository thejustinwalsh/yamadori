# Phase 0 — React Compiler vs r3f idioms

**Verdict: PASS. Keep the compiler on everywhere. No `'use no memo'` is
needed for the r3f subtree.** One real conflict was found and fixed with a
pattern (below), not by opting out. Measured 2026-09-22 on Node 24.21.0, the
production build (`npm run build` + `vite preview`), in installed Chrome 153.0.8010.53 with
WebGPU, driven by `playwright-core` so requestAnimationFrame ran unthrottled.

## 1. What ran

`/phase0` (`src/phase0/`), on r3f 10's **WebGPU** entry
(`@react-three/fiber/webgpu`). It exposes `window.__phase0` counters so the
check reads numbers, not pixels:

- `SpinningCube`: mutates `mesh.current.rotation` in `useFrame`. `speed` is a
  prop the harness toggles 1 → 4.
- `DerivedRing`: `new TorusGeometry(...)` derived in render, **no useMemo**.
  A `label` prop changes every 500 ms to force re-renders. `segments` can be
  bumped by +8.
- `DerivedRingUncompiled`: the same component, opted out with
  `'use no memo'`. This is the control.

## 2. Results (production build)

| check | observed | pass |
|---|---|---|
| frame loop runs | 56.7 fps | yes |
| ref mutated in `useFrame` animates | `cubeRotationY` rate **0.999 rad/s** at speed 1 | yes |
| no stale closure after a prop change | rate **4.002 rad/s** at speed 4; `cubeSpeed` read inside the running callback = 4 | yes |
| derived geometry is not rebuilt on unrelated renders | 13 ring commits in the window, `geometryBuilds` stayed **1** | yes |
| the control proves it is the compiler | uncompiled copy: **13** builds for 13 commits | yes |
| props change → exactly one rebuild | `segments` 24→32: `geometryBuilds` **2**, `geometryDisposals` **1**, torus `tubularSegments` **32** | yes |
| page errors | none | yes |

The dev server (StrictMode) agreed. The compiled ring built once, and the
control built 22 times for 11 commits, twice per commit, as StrictMode's
double render predicts.

**Compiled output, checked.** `SpinningCube` becomes `const $ = _c(6)` with
the `useFrame` callback cached on `$[0] !== speed`. `DerivedRing` caches
`makeRing(radius, segments)` on `$[0] !== radius || $[1] !== segments`. Both
import `c` from `react/compiler-runtime`. The first production build, which held only
the Phase 0 files, contained 11 `react.memo_cache_sentinel` markers.

**Why the stale closure cannot happen.** In r3f 10.0.0-alpha.5, `useFrame`
wraps the callback in `useMutableCallback` and calls
`callbackRef.current?.(...)` every frame (`dist/*/index.mjs`, `function
useFrame`). Whatever identity the compiler hands it, the latest one runs.

## 3. The conflict that WAS found, and the fix

`build/babel-transform.ts` routes the compiler's logger into Vite warnings,
because by default the compiler skips any function it cannot compile and
says nothing. The first full build of the real tree reported **4 bail-outs**:

| where | compiler said | fix |
|---|---|---|
| `bonsai/scene/Scene.tsx` `Tree` | "Modifying a value previously passed as an argument to a hook is not allowed" | the r3f idiom itself: `useMemo(() => three objects)`, then `geo.getAttribute(n).needsUpdate = true` inside `useFrame` |
| `bonsai/scene/Scene.tsx` `Stage` | "Modifying a value returned from a hook is not allowed" | `scene.background = …` / `scene.fog = …` on `useThree().scene` |
| `api/usePoll.ts` | "Handle TryStatement with a finalizer" | try/finally is unsupported |
| `bonsai/scene/Tokonoma.tsx` | "Support value blocks … within a try/catch" | a ternary inside a try |

A bail-out is SAFE: the function runs uncompiled and still behaves
correctly. It is also silent unless logged, which is why the logger exists.

The fixes, which are now the rules for tree code:

1. **Own imperative three.js state in a plain class, not in hook values.**
   `TreeRig` holds the geometry, instanced meshes and springs. The component
   takes one instance with `useState(() => new TreeRig(...))` and mutates it
   only through methods (`rig.step(goal, dt)`) inside `useFrame`. Refs
   mutated in `useFrame` were always fine (see the cube).
2. **Scene state goes through JSX, not assignment.** Use
   `<color attach="background" />` and `<fog attach="fog" />`.
3. **No try/finally, and no conditional inside a try, in a component or
   hook.** Move it to a module-level helper.

After the fixes: **59 functions compiled, 0 bail-outs, 1 skip** (the
deliberate `'use no memo'` control). `Tree`, `Stage`, `Slab`, `Shafts`,
`Bloom`, `Stats` and `BonsaiCanvas` all compile.

## 4. Toolchain facts found on the way (each checked, not assumed)

- **Node 20.15 cannot run this stack.** Vite 8.3.0, rolldown 1.2.9 and its
  native binding need `^20.19.0 || >=22.12.0`. npm skipped the binding
  silently, as an optional dependency with a mismatched engine, and Vite then
  died at startup. Now on Node 24.21.0; `web/.npmrc` sets
  `engine-strict=true`, so a too-old Node fails at install time.
- **`@react-three/fiber/legacy` (WebGL) does not build** against three
  0.186.0. Rolldown reports `MISSING_EXPORT` for `MeshBasicNodeMaterial`,
  `Node` and `NodeUpdateType` imported from `'three'`. The newest canary
  (`10.0.0-canary.24d1f81`) has the same import line.
- **Renderer decision, made by building:** r3f 10's WebGPU renderer plus
  three's own TSL post-processing. Bloom (`three/addons/tsl/display/BloomNode.js`)
  is selective, fed from the MRT `emissive` target via `useRenderPipeline`.
  Chrome reports `backend: WebGPU`. `postprocessing` 6.39.5 is WebGL-only, so
  it is not used; it is still installed as pinned and can be removed.
- `useRenderPipeline` is exported only from `@react-three/fiber/webgpu`, not
  the default entry. All r3f imports use `/webgpu`, so the bundle carries one
  copy.
- Babel resolves plugin names against the process cwd. A dev server started
  from the repo root failed until plugins were resolved to absolute paths
  from `web/` (`build/stylex-options.js`, `build/babel-transform.ts`).
