/**
 * physics.js — Multi-body chain simulation for tower dynamics
 *
 * Each placed block has its own angular state (θ, ω) for TWO rotation
 * axes (around Z for X-sway, around X for Z-sway). Adjacent blocks are
 * coupled by damped rotational springs. The base block is anchored to
 * the ground with a stiff spring.
 *
 * Key features:
 *   1. Whip/phase-lag effect — impulse at the top propagates as a wave
 *      down through the spring chain and reflects back.
 *   2. P-Delta instability — gravity overturning moment grows with
 *      height: M = P·Δ  (blocks above × horizontal offset).
 *   3. Height-dependent frequency — ω ≈ 1/H emerges naturally from
 *      the chain's lowest vibration mode.
 *   4. Substep integration for numerical stability at 20+ blocks.
 */
import {
    JOINT_STIFFNESS, JOINT_DAMPING,
    GROUND_STIFFNESS, GROUND_DAMPING,
    STRUCTURAL_DAMPING, P_DELTA_FACTOR,
    IMPACT_IMPULSE, MAX_ANGLE, SUBSTEPS,
    BLOCK_HEIGHT, SWAY_AMPLITUDE,
} from './config.js?v=6';

export class TowerPhysics {
    /**
     * @param {THREE.Group} towerGroup – group that parents all stacked blocks
     */
    constructor(towerGroup) {
        this.group = towerGroup;

        // Per-block angular state (only for PLACED blocks, not the base)
        // Index 0 = first placed block, 1 = second, etc.
        this.anglesZ = [];   // rotation around Z axis (sway in X)
        this.anglesX = [];   // rotation around X axis (sway in Z)
        this.omegasZ = [];   // angular velocity around Z
        this.omegasX = [];   // angular velocity around X

        this.blockCount = 0;  // number of placed blocks (excludes base)
    }

    /**
     * Called when a new block is placed on the tower.
     * Adds it to the chain and applies an impact impulse at the top.
     */
    onBlockPlaced(placed, previous, moveAxis) {
        // Compute rest position = prev's rest + slice offset
        // This prevents double-displacement: _applyToBlocks adds sway on top
        const sliceOffsetX = placed.position.x - previous.position.x;
        const sliceOffsetZ = placed.position.z - previous.position.z;
        const prevRestX = previous.userData.restX !== undefined ? previous.userData.restX : 0;
        const prevRestZ = previous.userData.restZ !== undefined ? previous.userData.restZ : 0;
        placed.userData.restX = prevRestX + sliceOffsetX;
        placed.userData.restZ = prevRestZ + sliceOffsetZ;

        // Add new link to the chain
        this.anglesZ.push(0);
        this.anglesX.push(0);
        this.omegasZ.push(0);
        this.omegasX.push(0);
        this.blockCount++;

        // Impact impulse at the TOP of the chain
        const offsetX = placed.position.x - previous.position.x;
        const offsetZ = placed.position.z - previous.position.z;
        const area = placed.userData.width * placed.userData.depth;
        const idx = this.blockCount - 1;

        // Height amplification — taller tower = less stable
        const heightAmp = 1 + idx * 0.15;

        // Spread impulse across the top several blocks so they sway together
        const spreadCount = Math.min(this.blockCount, 6);
        for (let j = 0; j < spreadCount; j++) {
            const blockIdx = this.blockCount - 1 - j;
            const falloff = 1.0 / (1 + j * 0.5); // 1.0, 0.67, 0.5, 0.4, ...

            if (moveAxis === 'x') {
                this.omegasZ[blockIdx] += -offsetX * area * IMPACT_IMPULSE * heightAmp * falloff;
            } else {
                this.omegasX[blockIdx] += offsetZ * area * IMPACT_IMPULSE * heightAmp * falloff;
            }
        }
    }

    /**
     * Advance the chain simulation by dt, then apply results to visual blocks.
     */
    update(dt) {
        const N = this.blockCount;
        if (N === 0) return;

        // Don't rotate the group itself
        this.group.rotation.z = 0;
        this.group.rotation.x = 0;

        const subDt = dt / SUBSTEPS;

        for (let sub = 0; sub < SUBSTEPS; sub++) {
            this._substep(subDt, N);
        }

        this._applyToBlocks(N);
    }

    /**
     * One substep of the chain physics.
     */
    _substep(dt, N) {
        // Temporary acceleration arrays
        const aZ = new Float64Array(N);
        const aX = new Float64Array(N);

        for (let i = 0; i < N; i++) {
            // ── Angles & velocities of this block and its neighbor below ──
            const thetaZ = this.anglesZ[i];
            const thetaX = this.anglesX[i];
            const wZ = this.omegasZ[i];
            const wX = this.omegasX[i];

            const belowZ = i > 0 ? this.anglesZ[i - 1] : 0; // ground = 0
            const belowX = i > 0 ? this.anglesX[i - 1] : 0;
            const belowWZ = i > 0 ? this.omegasZ[i - 1] : 0;
            const belowWX = i > 0 ? this.omegasX[i - 1] : 0;

            // ── 1. Spring from block below (or ground) ──
            // First block uses stiffer ground spring
            const kDown = i === 0 ? GROUND_STIFFNESS : JOINT_STIFFNESS;
            const dDown = i === 0 ? GROUND_DAMPING : JOINT_DAMPING;

            const springDownZ = -kDown * (thetaZ - belowZ) - dDown * (wZ - belowWZ);
            const springDownX = -kDown * (thetaX - belowX) - dDown * (wX - belowWX);

            // ── 2. Spring from block above (if exists) ──
            let springUpZ = 0, springUpX = 0;
            if (i < N - 1) {
                const aboveZ = this.anglesZ[i + 1];
                const aboveX = this.anglesX[i + 1];
                const aboveWZ = this.omegasZ[i + 1];
                const aboveWX = this.omegasX[i + 1];

                springUpZ = -JOINT_STIFFNESS * (thetaZ - aboveZ)
                    - JOINT_DAMPING * (wZ - aboveWZ);
                springUpX = -JOINT_STIFFNESS * (thetaX - aboveX)
                    - JOINT_DAMPING * (wX - aboveWX);
            }

            // ── 3. P-Delta gravity: M = P·Δ ──
            // Weight of blocks above creates an overturning moment
            // proportional to the sin of the absolute tilt angle.
            const blocksAbove = N - i;
            const pDeltaZ = blocksAbove * P_DELTA_FACTOR * Math.sin(thetaZ);
            const pDeltaX = blocksAbove * P_DELTA_FACTOR * Math.sin(thetaX);

            // ── 4. Structural damping (air resistance) ──
            const dragZ = -STRUCTURAL_DAMPING * wZ;
            const dragX = -STRUCTURAL_DAMPING * wX;

            // ── Total acceleration ──
            aZ[i] = springDownZ + springUpZ + pDeltaZ + dragZ;
            aX[i] = springDownX + springUpX + pDeltaX + dragX;
        }

        // ── Semi-implicit Euler integration ──
        for (let i = 0; i < N; i++) {
            this.omegasZ[i] += aZ[i] * dt;
            this.omegasX[i] += aX[i] * dt;
            this.anglesZ[i] += this.omegasZ[i] * dt;
            this.anglesX[i] += this.omegasX[i] * dt;

            // Clamp to prevent explosion
            this.anglesZ[i] = Math.max(-MAX_ANGLE, Math.min(MAX_ANGLE, this.anglesZ[i]));
            this.anglesX[i] = Math.max(-MAX_ANGLE, Math.min(MAX_ANGLE, this.anglesX[i]));
        }
    }

    /**
     * Forward kinematics: apply chain angles as POSITION OFFSETS only.
     * Blocks stay flat (no rotation) and tightly stacked.
     * The tower curves purely by shifting each block horizontally
     * based on the cumulative angular chain state below it.
     */
    _applyToBlocks(N) {
        const blocks = this.group.children;
        // blocks[0] = base (no chain entry)
        // blocks[1..N] = placed blocks → chain indices 0..N-1

        let cumOffsetX = 0;
        let cumOffsetZ = 0;

        for (let i = 0; i < N; i++) {
            const block = blocks[i + 1]; // visual block (offset by 1 for base)
            if (!block) continue;

            // NO rotation — blocks stay perfectly flat
            block.rotation.z = 0;
            block.rotation.x = 0;

            // Accumulate THIS block's angle into cumulative offset FIRST
            // so that this block is displaced by its own angle too
            const angleZ = this.anglesZ[i];
            const angleX = this.anglesX[i];
            cumOffsetX += Math.sin(angleZ) * BLOCK_HEIGHT * SWAY_AMPLITUDE;
            cumOffsetZ -= Math.sin(angleX) * BLOCK_HEIGHT * SWAY_AMPLITUDE;

            // Then shift position horizontally
            if (block.userData.restX !== undefined) {
                block.position.x = block.userData.restX + cumOffsetX;
                block.position.z = block.userData.restZ + cumOffsetZ;
            }
        }
    }

    /**
     * Reset all physics state (on game restart).
     */
    reset() {
        // Restore visual blocks
        for (const block of this.group.children) {
            block.rotation.z = 0;
            block.rotation.x = 0;
            if (block.userData.restX !== undefined) {
                block.position.x = block.userData.restX;
                block.position.z = block.userData.restZ;
            }
        }

        this.anglesZ = [];
        this.anglesX = [];
        this.omegasZ = [];
        this.omegasX = [];
        this.blockCount = 0;

        this.group.rotation.z = 0;
        this.group.rotation.x = 0;
    }
}
