import { normalWorld, packNormalToRGB } from 'three/tsl';

const n = normalWorld.normalize();

export const encodedNormal = packNormalToRGB( n );
