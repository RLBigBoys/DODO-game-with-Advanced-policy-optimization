/**
 * ui.js — In-game UI management (score, coins, game-over popup)
 */

const scoreEl = document.getElementById('score');
const coinTextEl = document.getElementById('coin-text');
const goOverlay = document.getElementById('gameover-overlay');
const goReward = document.getElementById('go-reward');
const btnRestart = document.getElementById('btn-restart');
const btnClose = document.getElementById('btn-close');
const tapStart = document.getElementById('tap-to-start');

const COINS_PER_BLOCK = 5;
const COINS_TARGET = 350;

let currentScore = 0;

/**
 * Update score display with a bounce animation.
 * @param {number} score
 */
export function updateScore(score) {
    currentScore = score;
    scoreEl.textContent = score;

    // Trigger bounce animation
    scoreEl.classList.remove('bump');
    // Force reflow to restart animation
    void scoreEl.offsetWidth;
    scoreEl.classList.add('bump');

    // Update coin bar
    const coins = score * COINS_PER_BLOCK;
    coinTextEl.textContent = `${coins} / ${COINS_TARGET}`;
}

/**
 * Show game-over popup with earned coins.
 * @param {number} score
 * @param {() => void} onRestart – callback to restart the game
 */
export function showGameOver(score, onRestart) {
    const coins = score * COINS_PER_BLOCK;
    goReward.textContent = `+${coins}D`;

    goOverlay.classList.remove('hidden');
    // Trigger transition after a frame
    requestAnimationFrame(() => {
        goOverlay.classList.add('visible');
    });

    // Restart button handler (one-shot)
    const handler = () => {
        btnRestart.removeEventListener('click', handler);
        onRestart();
    };
    btnRestart.addEventListener('click', handler);
}

/**
 * Hide game-over popup.
 */
export function hideGameOver() {
    goOverlay.classList.remove('visible');
    goOverlay.classList.add('hidden');
}

/**
 * Listen for close-button clicks.
 * @param {() => void} cb – called when user taps ✕
 */
export function onClose(cb) {
    btnClose.addEventListener('click', cb);
}

/**
 * Show tap-to-start overlay and resolve when tapped.
 * @returns {Promise<void>}
 */
export function waitForTapToStart() {
    return new Promise((resolve) => {
        tapStart.classList.remove('hidden');
        const handler = () => {
            tapStart.removeEventListener('click', handler);
            tapStart.removeEventListener('touchstart', handler);
            tapStart.classList.add('hidden');
            resolve();
        };
        tapStart.addEventListener('click', handler);
        tapStart.addEventListener('touchstart', handler, { passive: true });
    });
}

/**
 * Reset all UI elements to initial state.
 */
export function resetUI() {
    currentScore = 0;
    scoreEl.textContent = '0';
    coinTextEl.textContent = `0 / ${COINS_TARGET}`;
    hideGameOver();
}
