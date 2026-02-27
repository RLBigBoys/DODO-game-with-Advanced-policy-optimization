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
    BLOCK_HEIGHT,
    INITIAL_SIZE,
} from './block.js';
import {
    updateScore,
    showGameOver,
    hideGameOver,
    resetUI,
    onClose,
    waitForTapToStart,
} from './ui.js';

// ======================== CONSTANTS ========================
const BG_COLOR = 0x8A2BE2;
const MOVE_RANGE = 6;        // how far blocks move off-center
const BASE_SPEED = 12.0;     // units per second (crossing ~3u in ~0.5s)
const SPEED_BUMP = 0.05;     // +5% every 10 levels
const CAMERA_LERP = 0.06;     // camera follow smoothness
const FALL_GRAVITY = 15;       // gravity for falling pieces (units/s²)
const FALL_SPIN = 2.5;      // rotation speed for falling pieces
const PARTICLE_DECAY = 1.8;      // how fast particles fade

// ======================== STATE ========================
let scene, camera, renderer;
let ambientLight, dirLight;

let stack = [];      // list of placed block meshes
let score = 0;
let speed = BASE_SPEED;
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

// ======================== INIT ========================
function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(BG_COLOR);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    document.body.prepend(renderer.domElement);

    // Camera (orthographic, ~45° isometric)
    setupCamera();

    // Lights
    ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);

    dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
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
    const frustum = 6;
    camera = new THREE.OrthographicCamera(
        -frustum * aspect, frustum * aspect,
        frustum, -frustum,
        0.1, 100
    );
    // Position camera for ~45° isometric view
    camera.position.set(8, 8, 8);
    camera.lookAt(0, 0, 0);
}

function onResize() {
    const aspect = window.innerWidth / window.innerHeight;
    const frustum = 6;
    camera.left = -frustum * aspect;
    camera.right = frustum * aspect;
    camera.top = frustum;
    camera.bottom = -frustum;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// ======================== GAME FLOW ========================
async function startGame() {
    // Clear everything
    resetScene();
    resetUI();
    resetColorSequence();
    score = 0;
    speed = BASE_SPEED;
    moveAxis = 'x';
    gameState = 'idle';
    cameraTargetY = 0;

    // Create base block
    const base = createBlock(INITIAL_SIZE, INITIAL_SIZE);
    base.position.set(0, 0, 0);
    scene.add(base);
    stack.push(base);

    // Wait for tap
    await waitForTapToStart();

    gameState = 'playing';
    spawnMovingBlock();
    animate(performance.now());
}

function resetScene() {
    // Remove all meshes from the scene
    while (scene && scene.children.length > 0) {
        const obj = scene.children[0];
        scene.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) {
                obj.material.forEach(m => m.dispose());
            } else {
                obj.material.dispose();
            }
        }
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
    const y = prev.position.y + BLOCK_HEIGHT;
    block.position.y = y;

    // Start off-screen on the current axis
    if (moveAxis === 'x') {
        block.position.x = -MOVE_RANGE;
        block.position.z = prev.position.z;
    } else {
        block.position.z = -MOVE_RANGE;
        block.position.x = prev.position.x;
    }

    moveDir = 1;
    movingBlock = block;
    scene.add(block);
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
    e.preventDefault();

    const prev = stack[stack.length - 1];
    const result = sliceBlock(movingBlock, prev, moveAxis, scene);

    if (result.gameOver) {
        // Block completely missed — make it fall
        addFallingPiece(movingBlock);
        movingBlock = null;

        // Delay game-over popup slightly so user sees the block fall
        setTimeout(() => endGame(), 800);
        return;
    }

    // Success! Place the block
    scene.remove(movingBlock); // remove original if still there
    if (result.placed) {
        scene.add(result.placed);
        stack.push(result.placed);
    }

    // Handle cut piece
    if (result.cut) {
        scene.add(result.cut);
        addFallingPiece(result.cut);
    }

    // Perfect match particles
    if (result.perfect && result.placed) {
        const p = spawnPerfectParticles(result.placed, scene);
        particles.push(...p);
    }

    // Update score
    score++;
    updateScore(score);

    // Speed increase every 10 levels
    speed = BASE_SPEED * (1 + SPEED_BUMP * Math.floor(score / 10));

    // Camera target
    cameraTargetY = result.placed ? result.placed.position.y : cameraTargetY;

    // Alternate axis
    moveAxis = moveAxis === 'x' ? 'z' : 'x';

    // Spawn next
    movingBlock = null;
    spawnMovingBlock();
}

// ======================== FALLING PIECES ========================
function addFallingPiece(mesh) {
    // Random rotation axis
    const rotAxis = new THREE.Vector3(
        (Math.random() - 0.5),
        0,
        (Math.random() - 0.5)
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
            fp.mesh.geometry.dispose();
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
            p.mesh.geometry.dispose();
            p.mesh.material.dispose();
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

// ======================== ANIMATION LOOP ========================
let lastTime = 0;

function animate(time) {
    if (gameState === 'idle') return; // not yet started

    requestAnimationFrame(animate);

    const dt = Math.min((time - lastTime) / 1000, 0.05); // cap delta
    lastTime = time;

    // Move the current block
    if (movingBlock && gameState === 'playing') {
        const delta = speed * dt * moveDir;
        movingBlock.position[moveAxis] += delta;

        // Reverse at edges
        if (movingBlock.position[moveAxis] > MOVE_RANGE) {
            movingBlock.position[moveAxis] = MOVE_RANGE;
            moveDir = -1;
        } else if (movingBlock.position[moveAxis] < -MOVE_RANGE) {
            movingBlock.position[moveAxis] = -MOVE_RANGE;
            moveDir = 1;
        }
    }

    // Camera follow
    const targetY = cameraTargetY;
    camera.position.y += (targetY + 8 - camera.position.y) * CAMERA_LERP;
    camera.lookAt(0, targetY, 0);

    // Update shadow light position to follow the tower
    dirLight.position.y = targetY + 10;
    dirLight.target.position.set(0, targetY, 0);
    dirLight.target.updateMatrixWorld();

    // Update falling pieces
    updateFallingPieces(dt);

    // Update particles
    updateParticles(dt);

    renderer.render(scene, camera);
}

// ======================== BOOT ========================
init();
