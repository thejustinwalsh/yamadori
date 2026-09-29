Excerpt: three.js r185 (2431a09f46f3) docs/llms-full.txt lines 451-460 (MIT License, three.js authors); the U+200B pair of line 457 kept.

    - `gaussianBlur()`: Double render-pass gaussian blur node. It can be used in the material or in post-processing through a single function.
  - Easy access to renderer buffers using TSL functions like: 
    - `viewportSharedTexture()`: Accesses the beauty what has already been rendered, preserving the render-order.
    - `viewportLinearDepth()`: Accesses the depth what has already been rendered, preserving the render-order.
  - Integrated Compute Shaders
    - Perform calculations on buffers using compute stage directly during an object's rendering.
  - TSL allows dynamic manipulation of renderer functions, which makes it more customizable than intermediate languages ​​that would have to use flags in fixed pipelines for this.
  - You just need to use the events of a Node for the renderer manipulations, without needing to modify the core.

### Automatic Optimization and Workarounds
