/**
 * config.js — Centralized game configuration
 *
 * All tunable constants in one place: colors, speeds, physics,
 * block geometry, UI parameters, etc.
 */

// ═══════════════ RL SEEDING ═══════════════
let _rlSeed = null;
let _rlPrngState = 0;

window.setRLSeed = function (seed) {
    if (seed === null || seed === undefined) {
        _rlSeed = null;
    } else {
        _rlSeed = seed;
        _rlPrngState = seed >>> 0;
    }
};

export function rlRandom() {
    if (_rlSeed === null) {
        return Math.random();
    }
    let t = _rlPrngState += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
}

// ═══════════════ SCENE ═══════════════
export const BG_COLOR = 0x8A2BE2;          // scene background (purple)

// ═══════════════ CAMERA ═══════════════
export const CAMERA_FRUSTUM = 6;           // orthographic half-extent
export const CAMERA_LERP = 0.06;        // follow smoothness (0–1)

// ═══════════════ LIGHTING ═══════════════
export const AMBIENT_INTENSITY = 0.75;
export const DIR_LIGHT_INTENSITY = 0.85;

// ═══════════════ BLOCK GEOMETRY ═══════════════
export const BLOCK_HEIGHT = 0.22;     // thin pizza-box height
export const INITIAL_SIZE = 3;        // starting square size (X & Z)
export const PERFECT_THRESHOLD = 0.1;      // snap-to-center tolerance

// ═══════════════ BLOCK TEXTURES ═══════════════
export const PPU = 160;                    // pixels-per-world-unit for canvas textures

// ═══════════════ BLOCK MOVEMENT ═══════════════
export const MOVE_RANGE = 5;              // how far blocks slide off-center
export const MIN_SPEED = 7.0;            // min units per second
export const MAX_SPEED = 10.0;           // max units per second
export const SPEED_BUMP = 0.05;           // +5 % speed every 10 levels

// ═══════════════ FALLING PIECES ═══════════════
export const FALL_GRAVITY = 15;            // gravity for cut pieces (units/s²)
export const FALL_SPIN = 2.5;           // rotation speed for falling pieces

// ═══════════════ PARTICLES ═══════════════
export const PARTICLE_DECAY = 1.8;         // how fast perfect-match particles fade

// ═══════════════ TOWER PHYSICS (chain simulation) ═══════════════
export const JOINT_STIFFNESS = 15.0;   // low = slow gentle oscillation
export const JOINT_DAMPING = 0.25;   // very low = oscillations last long
export const GROUND_STIFFNESS = 90.0;  // stiff ground — base barely moves
export const GROUND_DAMPING = 4.0;    // ground settles quickly
export const STRUCTURAL_DAMPING = 0.02;  // minimal air drag
export const P_DELTA_FACTOR = 0.18;   // gravity overturning: M = P·Δ
export const IMPACT_IMPULSE = 0.05;   // gentle kick — small sway
export const MAX_ANGLE = 0.05;   // ~2.9° — allows more energy before clamping
export const SUBSTEPS = 8;      // physics substeps per render frame
export const SWAY_AMPLITUDE = 1.8;    // smaller horizontal shift per angle
export const FLEX_POWER = 1.5;    // (retained for reference, unused in chain)

// ═══════════════ UI / ECONOMY ═══════════════
export const COINS_PER_BLOCK = 5;
export const COINS_TARGET = 350;

// ═══════════════ PIZZA BOX VARIANTS ═══════════════
// { bg, text, label } — cycled sequentially
export const BOX_VARIANTS = [
    { bg: '#7CB342', text: '#FFFFFF', label: 'DODO KIDS' },     // bright lime green
    { bg: '#E8A0AA', text: '#FFFFFF', label: 'MARGHERITA' },    // soft warm pink
    { bg: '#F0C0C8', text: '#884050', label: 'DODO MIX' },     // light blush pink
    { bg: '#A4B83C', text: '#FFFFFF', label: 'DODO 2023' },     // warm chartreuse/yellow-green
    { bg: '#E8C058', text: '#5D4037', label: 'PERONI' },        // golden yellow
    { bg: '#E89060', text: '#FFFFFF', label: 'DODO PIZZA' },    // warm orange
    { bg: '#64B0D8', text: '#FFFFFF', label: 'PEPPERONI' },     // sky blue
    { bg: '#E08868', text: '#FFFFFF', label: 'HAWAII' },        // warm coral/salmon
    { bg: '#C0AFA0', text: '#4E342E', label: 'FOUR CHEESE' },   // warm beige/cream
];
