# Quarantine status (2026-10-07)

Names, ids and blocker classes only -- no skills data. Generated read-only from the skill store (`index/jobs.sqlite3`).
A skill's blocker is the first failing gate of its latest version.

| blocker | skills |
|---|---|
| prove-worse | 56 |
| activation case | 11 |
| screen | 3 |
| model screen | 1 |
| faithfulness | 2 |
| validate | 2 |
| in pipeline | 3 |

## prove-worse (56)

PROVE found WITH the skill a check that passed WITHOUT it failed. Records with `rule: one-sample` were decided before the repeat rule (REPEATS=1) and never re-proved: the 43 non-pagoda re-proofs were held (not_before 30 days out) and are idle-gated; `python mcp/skill_prove.py --reprove` lists them.

| skill | id | status | note |
|---|---|---|---|
| browser-app-entry-point | 508dc52b9091 | quarantined | worse checks: absent, parse; one-sample rule (not re-proved) |
| design-visual-brief-authority-2 | 33e4401b13a9 | quarantined | worse checks: parse; rule repeat/1 |
| design-visual-decoration-5 | dbd308680af6 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| design-visual-delight-specificity-6 | 29523dfb21f8 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| design-visual-section-numbering-22 | c826068b9232 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| design-visual-squint-test-23 | 089741588f21 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| design-visual-target-size-24 | 9d9e482ee17b | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| design-visual-will-change-26 | 77b9846fffa6 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| emscripten-embind-base-class-bindings-1 | 3f06d8f41348 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| emscripten-embind-e-g | e289f3fd9ead | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| emscripten-embind-explicit-deletion-2 | e486e79e39b7 | quarantined | worse checks: present; one-sample rule (not re-proved) |
| emscripten-interacting-with-code-64-bit-integer-handling-1 | 697829cb090e | quarantined | worse checks: absent; one-sample rule (not re-proved) |
| emscripten-interacting-with-code-c-function-exporting | 4856a20abba6 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| emscripten-interacting-with-code-string-conversion-5 | 44cbd16966e2 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| fe-handbook-system-design-ui-components-e-g | 68be742d5a63 | quarantined | worse checks: parse; rule repeat/1 |
| modern-practice-build-tooling-1 | 658f5acb342b | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| modern-practice-build-tooling-3-2 | 62e9a05fbd28 | quarantined | worse checks: present; one-sample rule (not re-proved) |
| modern-practice-modern-vs-outdated-1 | 42c6fc7801a1 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| modern-practice-modern-vs-outdated-3 | 948f32d5a3ef | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| modern-practice-modern-vs-outdated-7-2 | 347449338d92 | quarantined | worse checks: present; one-sample rule (not re-proved) |
| modern-practice-type-safety-1 | 2b21b5117559 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| module-exports-match-callers | b5d16254a780 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| r3f-events-9 | 83287a8c811b | quarantined | worse checks: parse; rule repeat/1 |
| r3f-v10-setup-21 | a7ef0b5c3d57 | quarantined | worse checks: absent; rule repeat/1 |
| react-dev-blog-react-19-3-fragment-refs-1 | 72480494980b | quarantined | worse checks: types; rule repeat/1 |
| react-dev-form-component-form-action-prop | 6e590a22168f | quarantined | worse checks: present; rule repeat/1 |
| react-dev-use-effect-event-dependency-array | f5cdce5e2375 | quarantined | worse checks: types; rule repeat/1 |
| rust-nomicon-ffi-async-callbacks-1 | 7905f05b839b | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| rust-nomicon-ffi-ffi-opaque-types-3 | 103a1526ce64 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| rust-nomicon-other-reprs-repr-c-ffi-safe-2 | e6329c2eec32 | quarantined | worse checks: absent; one-sample rule (not re-proved) |
| rust-reference-external-blocks-link-name-attribute | 6be5a323d303 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| sudheerj-react-interview-questions-event-handling-6 | bd62988f08b4 | quarantined | worse checks: present; rule repeat/1 |
| sudheerj-react-interview-questions-react-context-15 | 9733e75e78f4 | quarantined | worse checks: present; rule repeat/1 |
| sudheerj-react-interview-questions-use-hook-33 | 88097ea4972b | quarantined | worse checks: types; rule repeat/1 |
| systems-canonical-style-2 | 7d6154a83b30 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| systems-canonical-style-2-2 | ccb26a8d9993 | quarantined | worse checks: absent; one-sample rule (not re-proved) |
| systems-memory-layout-1 | b40621615dd7 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| threejs-llms-full-tsl-compute-shader-lifecycle-2 | ba3b1745c027 | quarantined | worse checks: parse; rule repeat/1 |
| typegpu-three-tsl-integration-2 | c45cc261b2c0 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| typescript-6-0-release-notes-syntax-migration-3 | c018bc1d1878 | quarantined | worse checks: types; rule repeat/1 |
| typescript-types-branding-2 | bb0b5fccdf69 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| typescript-types-expect-type-6 | f3a3ec654ffa | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| typescript-types-intrinsics-8 | 04309b251e98 | quarantined | worse checks: present; one-sample rule (not re-proved) |
| typescript-types-modifiers-9 | 7f8f580284bb | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| typescript-types-union-limit | 8784e66a2870 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| wasm-basic-c-abi-red-zone-optimization-2 | d77e44d5f03a | quarantined | worse checks: absent; one-sample rule (not re-proved) |
| wasm-bindgen-closures-closure-api-selection | ec3115025dab | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| wasm-bindgen-promises-futures-heterogeneous-promise-all-2 | cb4422d3713d | quarantined | worse checks: absent; one-sample rule (not re-proved) |
| web-gpu-bind-group-layouts-3 | 2b46e1c28a52 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| web-gpu-bitecs-1 | f45602cb3849 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| web-gpu-render-bundles-15 | 69348b57475b | quarantined | worse checks: absent, present; one-sample rule (not re-proved) |
| web-gpu-tgsl-externals-19 | 40f3097f49e5 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| web-gpu-update-frequency-grouping-23 | 6a81b7385bb2 | quarantined | worse checks: parse; one-sample rule (not re-proved) |
| web-gpu-use-transition-24 | 231f3fcd2301 | quarantined | worse checks: present; rule repeat/1 |
| web-gpu-workers-sharedarraybuffer-25 | d41f2a434be8 | quarantined | worse checks: present; one-sample rule (not re-proved) |
| webgpufundamentals-bind-group-layouts-bind-group-layouts-1 | c5a9b8d47dfd | quarantined | worse checks: present; one-sample rule (not re-proved) |

## activation case (11)

The text matcher's activation tests failed (a near-miss selects it, or a should-case does not). With a package channel the skill can be armed package-only: `skill_pipeline.admit_package_only(id)` then PROVE (idle-gated). `skill_pipeline.promote_package_only` promotes one whose tests now pass under the matcher's version/negation gates.

| skill | id | status | note |
|---|---|---|---|
| koota-composable-systems | 5cf16e75cd0c | quarantined | has a package channel (admit_package_only applies); should: "I'm building a koota-based TypeScript project in src/game.ts. I currently have o" -> none (no multi_f |
| koota-ecs-core | 15441a593e1f | quarantined | has a package channel (admit_package_only applies); should: 'I need to refactor my Koota ECS game to use tag traits. Currently I have an Enti' -> none (phase is n |
| koota-entity-operations | f4adc9fd28ad | quarantined | has a package channel (admit_package_only applies); should: 'I have a Koota scene where players pick up collectible items. The item is a spaw' -> none (phase is n |
| koota-queries | 5325ca04a6fc | quarantined | has a package channel (admit_package_only applies); should: 'I have a Koota world and I need to batch-update the Health trait of every entity' -> none (phase is n |
| koota-relations | 9193b5c59bdd | quarantined | has a package channel (admit_package_only applies); should: 'In my Koota project I need each container entity to hold multiple item entities,' -> none (one plain  |
| koota-runtime-systems | 0d2f2ed3fa9b | quarantined | has a package channel (admit_package_only applies); should: 'I have a Koota ECS world with position and velocity components attached to entit' -> none (one plain  |
| koota-traits-actions-over-classes | d4349e72aa18 | quarantined | has a package channel (admit_package_only applies); should: "I'm building a koota game loop and I have all my player state (position, health," -> none (one plain  |
| r3f-frame-reuse-objects | b2f436fe5992 | quarantined | NO package channel (area not a registry package); should_not: "I'm building a standalone Three.js web page in plain HTML and JavaScript (no Rea" -> inject; shou |
| react-dev-compiler-setup-vite | f96ecb5aa1fd | quarantined | NO package channel (area not a registry package); should_not: "I'm building an Expo React app with JavaScript and I want to add reactCompilerPr" -> inject; shou |
| react-dev-ref-as-prop | 39d1be7596af | quarantined | NO package channel (area not a registry package); should_not: "I'm working on a React 18 project and I need to set up forwardRef and useImperat" -> inject; shou |
| typescript-tsconfig-app-setup | 4c2161dbf28e | quarantined | NO package channel (area not a registry package); should_not: "I'm building a TypeScript LIBRARY that I'll publish to npm. What should my tscon" -> inject |

## screen (3)

The deterministic screen (mcp/skill_screen.py) quarantined the source. A pinned source may be cleaned first (mcp/skill_clean.py, meta fetch_clean.pinned_sha256).

| skill | id | status | note |
|---|---|---|---|
| koota_react_actions | f6e95de8b402 | quarantined | screen: exfiltration: a markdown image whose URL carries a query string (a beacon) (line 1) |
| koota_react_app_setup | fb9a09997c3f | quarantined | screen: exfiltration: a markdown image whose URL carries a query string (a beacon) (line 1) |
| koota_react_reactive_hooks | a8af58c8e943 | quarantined | screen: exfiltration: a markdown image whose URL carries a query string (a beacon) (line 1) |

## model screen (1)

The model screen reported ai_directed text in the source.

| skill | id | status | note |
|---|---|---|---|
| tsl_compute_shaders | 3ca21fd2465e | quarantined | model screen: model_screen: the model's check reports ai_directed; model_screen: the model's check reports ai_directed |

## faithfulness (2)

validate: no item survived the faithfulness check.

| skill | id | status | note |
|---|---|---|---|
| koota_data_oriented_design | 235fec45c8ec | failed | validate: no item survived the faithfulness check (2 dropped) |
| koota_trait_types | fd48fc62b261 | failed | validate: no item survived the faithfulness check (1 dropped) |

## validate (2)

A validate-stage failure other than faithfulness.

| skill | id | status | note |
|---|---|---|---|
| koota_react_integration | 1de2934e62bd | failed | validate: no item survived validation (3 dropped) |
| koota_trait_naming | 797192824940 | failed | validate: no item survived validation (3 dropped) |

## in pipeline (3)

Not quarantined: still running through the pipeline.

| skill | id | status | note |
|---|---|---|---|
| koota_directory_structure | 6a359a237f64 | pipeline | v1 running at validate |
| koota_view_decoupling | 2644a1a9c7c5 | pipeline | v1 running at validate |
| pmath_spherical_orbit | afeae5f1284f | pipeline | v1 running at distil |
