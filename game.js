/**
 * game.js — Main game loop for DODO Stack
 *
 * Three.js scene with orthographic camera, pizza box stacking,
 * slicing mechanics, camera follow, and falling physics.
 */
import * as THREE from 'three';
import {
    createBlock,
    sliceBlock,
    spawnPerfectParticles,
    resetColorSequence,
} from './block.js?v=6';
import {
    updateScore,
    showGameOver,
    hideGameOver,
    resetUI,
    onClose,
    waitForTapToStart,
} from './ui.js?v=6';
import { TowerPhysics } from './physics.js?v=6';
import {
    BG_COLOR, MOVE_RANGE, MIN_SPEED, MAX_SPEED, SPEED_BUMP,
    CAMERA_LERP, CAMERA_FRUSTUM,
    FALL_GRAVITY, FALL_SPIN, PARTICLE_DECAY,
    BLOCK_HEIGHT, INITIAL_SIZE,
    AMBIENT_INTENSITY, DIR_LIGHT_INTENSITY,
    rlRandom
} from './config.js?v=6';

// ======================== STATE ========================
let scene, camera, renderer;
let ambientLight, dirLight;

let stack = [];      // list of placed block meshes
let score = 0;
let speed = MIN_SPEED; // will be overwritten in startGame
let gameState = 'idle';  // idle | playing | gameover
let movingBlock = null;
let moveAxis = 'x';   // alternates each level
let moveDir = 1;     // +1 or -1

// Falling "cut" pieces
let fallingPieces = [];  // { mesh, velY, rotAxis }

// Perfect-match particles
let particles = [];      // { mesh, vel, life }

// Target camera Y
let cameraTargetY = 0;

// Tower physics group & simulation
let towerGroup = null;
let towerPhysics = null;

// ======================== INIT ========================
function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(BG_COLOR);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.body.prepend(renderer.domElement);

    // Camera (orthographic, ~45° isometric)
    setupCamera();

    // Lights
    ambientLight = new THREE.AmbientLight(0xffffff, AMBIENT_INTENSITY);
    scene.add(ambientLight);

    dirLight = new THREE.DirectionalLight(0xffffff, DIR_LIGHT_INTENSITY);
    dirLight.position.set(5, 10, 7);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.set(1024, 1024);
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 50;
    dirLight.shadow.camera.left = -10;
    dirLight.shadow.camera.right = 10;
    dirLight.shadow.camera.top = 10;
    dirLight.shadow.camera.bottom = -10;
    scene.add(dirLight);

    // Events
    window.addEventListener('resize', onResize);
    window.addEventListener('pointerdown', onTap);

    // Close button → trigger game over
    onClose(() => {
        if (gameState === 'playing') {
            endGame();
        }
    });

    // Start the first round
    startGame();
}

function setupCamera() {
    const aspect = window.innerWidth / window.innerHeight;
    camera = new THREE.OrthographicCamera(
        -CAMERA_FRUSTUM * aspect, CAMERA_FRUSTUM * aspect,
        CAMERA_FRUSTUM, -CAMERA_FRUSTUM,
        0.1, 100
    );
    // Position camera for ~45° isometric view
    camera.position.set(8, 8, 8);
    camera.lookAt(0, 0, 0);
}

function onResize() {
    const aspect = window.innerWidth / window.innerHeight;
    camera.left = -CAMERA_FRUSTUM * aspect;
    camera.right = CAMERA_FRUSTUM * aspect;
    camera.top = CAMERA_FRUSTUM;
    camera.bottom = -CAMERA_FRUSTUM;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function disposeObject(obj) {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
        if (Array.isArray(obj.material)) {
            obj.material.forEach(m => m.dispose());
        } else {
            obj.material.dispose();
        }
    }
    if (obj.children) {
        obj.children.forEach(child => disposeObject(child));
    }
}

// ======================== GAME FLOW ========================
let isAnimating = false;

async function startGame() {
    // Clear everything
    resetScene();
    resetUI();
    resetColorSequence();
    score = 0;
    speed = MIN_SPEED + Math.random() * (MAX_SPEED - MIN_SPEED);
    moveAxis = 'x';
    gameState = 'idle';
    cameraTargetY = 0;

    // Create tower physics group
    towerGroup = new THREE.Group();
    scene.add(towerGroup);
    towerPhysics = new TowerPhysics(towerGroup);
    // DEBUG: expose for console inspection
    window._towerGroup = towerGroup;
    window._towerPhysics = towerPhysics;

    // Create base block
    const base = createBlock(INITIAL_SIZE, INITIAL_SIZE);
    base.position.set(0, 0, 0);
    base.userData.restX = 0;
    base.userData.restZ = 0;
    towerGroup.add(base);
    stack.push(base);

    // Wait for tap
    await waitForTapToStart();

    gameState = 'playing';
    spawnMovingBlock();
    if (!isAnimating) {
        isAnimating = true;
        animate(performance.now());
    }
}

function resetScene() {
    // Remove all meshes from the scene
    while (scene && scene.children.length > 0) {
        const obj = scene.children[0];
        scene.remove(obj);
        disposeObject(obj);
    }
    stack = [];
    fallingPieces = [];
    particles = [];

    // Re-add lights
    if (ambientLight) scene.add(ambientLight);
    if (dirLight) scene.add(dirLight);
}

function spawnMovingBlock() {
    const prev = stack[stack.length - 1];
    const w = prev.userData.width;
    const d = prev.userData.depth;

    const block = createBlock(w, d);
    // Use world position of the previous block (it's inside towerGroup)
    const prevWorld = new THREE.Vector3();
    prev.getWorldPosition(prevWorld);
    const y = prevWorld.y + BLOCK_HEIGHT;
    block.position.y = y;

    // Start off-screen on the current axis
    // Use rest positions (not physics-displaced) for the non-sliding axis
    const restX = prev.userData.restX !== undefined ? prev.userData.restX : prev.position.x;
    const restZ = prev.userData.restZ !== undefined ? prev.userData.restZ : prev.position.z;
    if (moveAxis === 'x') {
        block.position.x = -MOVE_RANGE;
        block.position.z = restZ;
    } else {
        block.position.z = -MOVE_RANGE;
        block.position.x = restX;
    }

    moveDir = 1;
    movingBlock = block;
    scene.add(block); // moving block stays in scene (not in towerGroup) until placed
}

function endGame() {
    gameState = 'gameover';
    movingBlock = null;

    showGameOver(score, () => {
        // Restart callback
        startGame();
    });
}

// ======================== TAP HANDLER ========================
function onTap(e) {
    if (gameState !== 'playing' || !movingBlock) return;

    // Prevent default to avoid double-firing on mobile
    if (e) e.preventDefault();

    // Manual clicks are intercepted for the Python loop if running RL
    if (e && e.isTrusted && window.rlMode) {
        window.pythonRlAction = 'c';
        // In RL mode (both auto and manual step), Python will explicitly call window.executeDropBlock() 
        // when it processes the action. Doing it here causes a double-drop.
        return;
    }

    window.executeDropBlock();
}

// Expose drop block logic so Python can trigger it cleanly without fake clicks
window.executeDropBlock = function () {
    if (gameState !== 'playing' || !movingBlock) return { dropped: false, success: false, areaRatio: 0, perfect: false };

    const prev = stack[stack.length - 1];
    const result = sliceBlock(movingBlock, prev, moveAxis, scene);

    if (result.gameOver) {
        // Block completely missed — make it fall
        addFallingPiece(movingBlock);
        movingBlock = null;

        // Delay game-over popup slightly so user sees the block fall
        setTimeout(() => endGame(), 800);
        return { dropped: true, success: false, areaRatio: 0, perfect: false };
    }

    // Success! Place the block
    scene.remove(movingBlock); // remove original if still there
    const prevBlock = stack[stack.length - 1];

    let placedBlock = result.placed;
    let cutPieces = result.cut ? [result.cut] : [];

    // ── Cross-axis slice: check the OTHER axis for sway-induced overhang ──
    if (placedBlock && !result.perfect) {
        const crossAxis = moveAxis === 'x' ? 'z' : 'x';
        const crossResult = sliceBlock(placedBlock, prevBlock, crossAxis, scene);

        if (crossResult.gameOver) {
            // Completely missed on cross-axis too
            addFallingPiece(placedBlock);
            cutPieces.forEach(c => { scene.add(c); addFallingPiece(c); });
            movingBlock = null;
            setTimeout(() => endGame(), 800);
            return { dropped: true, success: false, areaRatio: 0, perfect: false };
        }

        placedBlock = crossResult.placed;
        if (crossResult.cut) cutPieces.push(crossResult.cut);
    }

    if (placedBlock) {
        towerGroup.add(placedBlock);
        stack.push(placedBlock);

        // Apply physics impulse
        towerPhysics.onBlockPlaced(placedBlock, prevBlock, moveAxis);

        // Immediately sync visual positions
        towerPhysics.update(0);
    }

    // Handle cut pieces
    for (const cut of cutPieces) {
        scene.add(cut);
        addFallingPiece(cut);
    }

    // Perfect match particles
    if (result.perfect && result.placed) {
        const p = spawnPerfectParticles(result.placed, scene);
        particles.push(...p);
    }

    // Update score
    score++;
    updateScore(score);

    // Randomize speed from uniform [MIN_SPEED, MAX_SPEED], then scale up every 10 levels
    const baseSpeed = MIN_SPEED + rlRandom() * (MAX_SPEED - MIN_SPEED);
    speed = baseSpeed * (1 + SPEED_BUMP * Math.floor(score / 10));

    // Camera target (use world position since block is inside towerGroup)
    if (result.placed) {
        const worldPos = new THREE.Vector3();
        result.placed.getWorldPosition(worldPos);
        cameraTargetY = worldPos.y;
    }

    // Alternate axis
    moveAxis = moveAxis === 'x' ? 'z' : 'x';

    // Spawn next
    movingBlock = null;
    spawnMovingBlock();

    let areaRatio = 1.0;
    if (placedBlock && prevBlock) {
        const prevArea = prevBlock.userData.width * prevBlock.userData.depth;
        const newArea = placedBlock.userData.width * placedBlock.userData.depth;
        areaRatio = newArea / prevArea;
    }

    return { dropped: true, success: true, areaRatio: areaRatio, perfect: result.perfect };
}

// ======================== FALLING PIECES ========================
function addFallingPiece(mesh) {
    // Random rotation axis
    const rotAxis = new THREE.Vector3(
        (rlRandom() - 0.5),
        0,
        (rlRandom() - 0.5)
    ).normalize();

    fallingPieces.push({
        mesh,
        velY: 0,
        rotAxis,
        age: 0,
    });
}

function updateFallingPieces(dt) {
    for (let i = fallingPieces.length - 1; i >= 0; i--) {
        const fp = fallingPieces[i];
        fp.velY -= FALL_GRAVITY * dt;
        fp.mesh.position.y += fp.velY * dt;
        fp.mesh.rotateOnWorldAxis(fp.rotAxis, FALL_SPIN * dt);
        fp.age += dt;

        // Remove after falling far enough
        if (fp.age > 2.5) {
            scene.remove(fp.mesh);
            disposeObject(fp.mesh);
            fallingPieces.splice(i, 1);
        }
    }
}

// ======================== PARTICLES ========================
function updateParticles(dt) {
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.life -= dt * PARTICLE_DECAY;

        if (p.life <= 0) {
            scene.remove(p.mesh);
            disposeObject(p.mesh);
            particles.splice(i, 1);
            continue;
        }

        p.mesh.position.x += p.vel.x * dt;
        p.mesh.position.y += p.vel.y * dt;
        p.mesh.position.z += p.vel.z * dt;

        // Gravity on particles
        p.vel.y -= 6 * dt;

        // Fade out
        p.mesh.material.opacity = Math.max(0, p.life);
    }
}

// ======================== TOWER COLLISION ========================
const _movingBox = new THREE.Box3();
const _towerBox = new THREE.Box3();
const _shrink = new THREE.Vector3(0.05, 0.05, 0.05); // margin to avoid false positives

function checkCollisionWithTower(block) {
    _movingBox.setFromObject(block);
    _movingBox.min.add(_shrink);    // shrink slightly inward
    _movingBox.max.sub(_shrink);

    for (const child of towerGroup.children) {
        _towerBox.setFromObject(child);
        if (_movingBox.intersectsBox(_towerBox)) {
            return true;
        }
    }
    return false;
}

// ======================== ANIMATION LOOP ========================
let lastTime = performance.now();

// Array of promise resolvers waiting for the next frame render
window._frameWaiters = [];

function updateGameData(dt) {
    // Move the current block
    if (movingBlock && gameState === 'playing') {
        const delta = speed * dt * moveDir;
        movingBlock.position[moveAxis] += delta;

        // Keep moving block aligned with swaying tower on the non-sliding axis
        const prev = stack[stack.length - 1];
        if (moveAxis === 'x') {
            movingBlock.position.z = prev.position.z;
        } else {
            movingBlock.position.x = prev.position.x;
        }

        // Check if block flew past the tower (agent missed the tap)
        if (movingBlock.position[moveAxis] > MOVE_RANGE) {
            addFallingPiece(movingBlock);
            movingBlock = null;
            setTimeout(() => endGame(), 800);
        }
    }

    // Camera follow
    if (dt > 0) {
        const targetY = cameraTargetY;
        camera.position.y += (targetY + 8 - camera.position.y) * CAMERA_LERP;
        camera.lookAt(0, targetY, 0);

        // Update shadow light position to follow the tower
        dirLight.position.y = targetY + 10;
        dirLight.target.position.set(0, targetY, 0);
        dirLight.target.updateMatrixWorld();
    }

    // Update tower physics (sway & tilt)
    if (towerPhysics) {
        towerPhysics.update(dt);
    }

    // Update falling pieces
    updateFallingPieces(dt);

    // Update particles
    updateParticles(dt);
}

// Play mode animation loop
function animate(time) {
    if (gameState === 'idle') {
        requestAnimationFrame(animate);
        return;
    }

    // if Python is in step-mode (paused), this is true
    if (window.rlPhysicsFrozen === true) {
        // Stop the loop completely, Python will manually step via rlManualStep
        return;
    }

    const dt = Math.min((time - lastTime) / 1000, 0.05); // cap delta
    lastTime = time;

    updateGameData(dt);
    renderer.render(scene, camera);

    // Resolve anyone waiting for this frame (Python auto_mode)
    const waiters = window._frameWaiters;
    window._frameWaiters = [];
    waiters.forEach(resolve => resolve());

    requestAnimationFrame(animate);
}

// ======================== RL INTERFACE ========================
// Create a small hidden canvas specifically for blazing-fast RL frame extraction
const rlCanvas = document.createElement('canvas');
rlCanvas.width = 84;
rlCanvas.height = 84;
const rlCtx = rlCanvas.getContext('2d', { willReadFrequently: true });

window.getRlFrame = function () {
    // Instantly draw and downsample the main WebGL buffer into our tiny 84x84 canvas
    rlCtx.drawImage(renderer.domElement, 0, 0, 84, 84);
    // Export minimal JPEG String
    return rlCanvas.toDataURL('image/jpeg', 0.6);
};

// Auto Mode: Yield execution until the browser's requestAnimationFrame does one naturally timed tick
window.waitForNextFrame = function () {
    return new Promise(resolve => {
        window._frameWaiters.push(resolve);
    });
};

// Step Mode: Completely freeze the loop and take manual control of time and rendering
window.rlManualStep = function (dt) {
    window.rlPhysicsFrozen = true; // freeze natural animation loop globally 
    updateGameData(dt);
    renderer.render(scene, camera);
    return window.getRlFrame();
};

window.rlResumeAuto = function () {
    if (window.rlPhysicsFrozen) {
        window.rlPhysicsFrozen = false;
        lastTime = performance.now();
        requestAnimationFrame(animate);
    }
}

// ======================== BOOT ========================
init();

// --- PYTHON RL INTEGRATION (Optional Interactive Controls) ---
window.pythonRlAction = null;
window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (['a', 'r', 's'].includes(k)) {
        window.pythonRlAction = k; // Store action for Python to poll
    }
});
