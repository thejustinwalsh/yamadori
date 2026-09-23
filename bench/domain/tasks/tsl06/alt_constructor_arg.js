import { pass } from 'three/tsl';
import { RenderPipeline } from 'three/webgpu';

export const createPipeline = ( renderer, scene, camera ) => new RenderPipeline( renderer, pass( scene, camera ) );
