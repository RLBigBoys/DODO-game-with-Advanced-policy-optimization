/**
 * block.js — Pizza box block creation, texturing, and slicing logic
 *
 * Matches the DODO Pizza reference: labeled side faces with brand names,
 * solid opaque pastel materials, proportional canvas textures that
 * clip text naturally when boxes are sliced smaller.
 */
import * as THREE from 'three';
import {
  BLOCK_HEIGHT,
  INITIAL_SIZE,
  PERFECT_THRESHOLD,
  PPU,
  BOX_VARIANTS,
  rlRandom
} from './config.js?v=6';

// Re-export for any consumers
export { BLOCK_HEIGHT, INITIAL_SIZE, PERFECT_THRESHOLD };

// Sequential variant index (cycles through palette)
// No longer used, we pick randomly now.
// let variantIndex = 0;

/**
 * Get next variant randomly.
 */
function nextVariant() {
  const index = Math.floor(rlRandom() * BOX_VARIANTS.length);
  return BOX_VARIANTS[index];
}

/**
 * Reset variant index on game restart.
 */
export function resetColorSequence() {
  // no-op
}

// ─── Texture creation ───

/**
 * Create a side-face canvas texture with a pizza label.
 *
 * Canvas width is proportional to the face's world-space width,
 * so when the box is sliced narrower, the canvas is smaller and
 * the text clips naturally at the right edge — matching the reference.
 *
 * @param {number} faceWidth – world units (box width or depth)
 * @param {string} bgColor   – CSS hex color
 * @param {string} textColor – CSS hex color
 * @param {string} label     – brand text (e.g. "MARGHERITA")
 */
function makeSideTexture(faceWidth, bgColor, textColor, label) {
  const canvasW = Math.max(32, Math.round(faceWidth * PPU));
  const canvasH = Math.max(12, Math.round(BLOCK_HEIGHT * PPU)); // ~35px

  const canvas = document.createElement('canvas');
  canvas.width = canvasW;
  canvas.height = canvasH;
  const ctx = canvas.getContext('2d');

  // Solid background fill
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, canvasW, canvasH);

  // Bold label text — left-aligned with padding
  const fontSize = Math.round(canvasH * 0.58);
  ctx.fillStyle = textColor;
  ctx.font = `900 ${fontSize}px "Arial", "Helvetica Neue", sans-serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  const pad = Math.round(canvasH * 0.35);
  ctx.fillText(label, pad, canvasH / 2 + 1);

  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Build 6-material array for a BoxGeometry(width, BLOCK_HEIGHT, depth).
 *
 * Face order: +X, -X, +Y (top), -Y (bottom), +Z (front), -Z (back)
 *
 * - +X/-X side faces are (depth × BLOCK_HEIGHT) → label canvas width ∝ depth
 * - +Z/-Z side faces are (width × BLOCK_HEIGHT) → label canvas width ∝ width
 * - +Y top: solid lighter shade, no text
 * - -Y bottom: solid slightly darker shade
 *
 * ALL materials are fully opaque — no transparency.
 */
function makeBoxMaterials(variant, boxWidth, boxDepth) {
  const { bg, text, label } = variant;
  const baseColor = new THREE.Color(bg);

  // Side face textures (proportional to face world-width)
  const xSideTex = makeSideTexture(boxDepth, bg, text, label);
  const zSideTex = makeSideTexture(boxWidth, bg, text, label);

  const xSideMat = new THREE.MeshLambertMaterial({ map: xSideTex });
  const zSideMat = new THREE.MeshLambertMaterial({ map: zSideTex });

  // Top face — solid, slightly lighter (clean pizza-box lid look)
  const topColor = baseColor.clone().lerp(new THREE.Color('#FFFFFF'), 0.10);
  const topMat = new THREE.MeshLambertMaterial({ color: topColor });

  // Bottom face — solid, slightly darker
  const bottomColor = baseColor.clone().lerp(new THREE.Color('#000000'), 0.06);
  const bottomMat = new THREE.MeshLambertMaterial({ color: bottomColor });

  return [
    xSideMat,   // +X  (depth × height)
    xSideMat,   // -X  (depth × height)
    topMat,     // +Y  top
    bottomMat,  // -Y  bottom
    zSideMat,   // +Z  front (width × height)
    zSideMat,   // -Z  back  (width × height)
  ];
}

// ─── Public API ───

/**
 * Create a new pizza-box mesh.
 *
 * @param {number} width  – size along X axis
 * @param {number} depth  – size along Z axis
 * @param {object} [variant] – { bg, text, label } override (for sliced pieces)
 * @returns {THREE.Mesh}
 */
export function createBlock(width, depth, variant) {
  const v = variant || nextVariant();
  const geom = new THREE.BoxGeometry(width, BLOCK_HEIGHT, depth);
  const mats = makeBoxMaterials(v, width, depth);
  const mesh = new THREE.Mesh(geom, mats);
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  // Metadata for slicing
  mesh.userData.width = width;
  mesh.userData.depth = depth;
  mesh.userData.variant = v;
  return mesh;
}

/**
 * Slice a block along an axis.
 *
 * Returns { placed, cut, perfect, gameOver }
 */
export function sliceBlock(current, previous, axis, scene) {
  const prop = axis === 'x' ? 'width' : 'depth';
  const posAxis = axis;

  const curPos = current.position[posAxis];
  const prevPos = previous.position[posAxis];
  const curSize = current.userData[prop];

  const delta = curPos - prevPos;
  const absDelta = Math.abs(delta);

  // ── Game Over: no overlap ──
  if (absDelta >= curSize) {
    return { placed: null, cut: null, perfect: false, gameOver: true };
  }

  // ── Perfect Match ──
  if (absDelta < PERFECT_THRESHOLD) {
    current.position[posAxis] = prevPos;
    return { placed: current, cut: null, perfect: true, gameOver: false };
  }

  // ── Partial overlap → slice ──
  const overlap = curSize - absDelta;
  const direction = Math.sign(delta);

  const otherProp = prop === 'width' ? 'depth' : 'width';
  const otherSize = current.userData[otherProp];

  // Placed piece (stays on tower)
  const placedCenter = prevPos + (delta / 2);
  const placedWidth = axis === 'x' ? overlap : otherSize;
  const placedDepth = axis === 'z' ? overlap : otherSize;

  const placed = createBlock(placedWidth, placedDepth, current.userData.variant);
  placed.position.copy(current.position);
  placed.position[posAxis] = placedCenter;
  placed.position.y = current.position.y;

  // Cut piece (falls off)
  const cutSize = absDelta;
  const cutCenter = curPos + direction * (overlap / 2);
  const cutWidth = axis === 'x' ? cutSize : otherSize;
  const cutDepth = axis === 'z' ? cutSize : otherSize;

  const cut = createBlock(cutWidth, cutDepth, current.userData.variant);
  cut.position.copy(current.position);
  cut.position[posAxis] = cutCenter;
  cut.position.y = current.position.y;

  // Remove original
  scene.remove(current);
  if (current.geometry) current.geometry.dispose();
  if (Array.isArray(current.material)) {
    current.material.forEach(m => { if (m.map) m.map.dispose(); m.dispose(); });
  }

  return { placed, cut, perfect: false, gameOver: false };
}

/**
 * Spawn sparkle particles for a perfect match.
 */
export function spawnPerfectParticles(block, scene) {
  const particles = [];
  const count = 10;
  const w = block.userData.width / 2;
  const d = block.userData.depth / 2;
  const py = block.position.y;

  for (let i = 0; i < count; i++) {
    const geom = new THREE.SphereGeometry(0.06, 8, 8);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xFFFFFF,
      transparent: true,
      opacity: 1,
    });
    const spark = new THREE.Mesh(geom, mat);

    const angle = (Math.PI * 2 * i) / count;
    spark.position.set(
      block.position.x + Math.cos(angle) * w,
      py + BLOCK_HEIGHT,
      block.position.z + Math.sin(angle) * d
    );

    const vel = new THREE.Vector3(
      Math.cos(angle) * (2 + rlRandom() * 2),
      1.5 + rlRandom() * 2,
      Math.sin(angle) * (2 + rlRandom() * 2)
    );

    scene.add(spark);
    particles.push({ mesh: spark, vel, life: 0.6 });
  }
  return particles;
}
