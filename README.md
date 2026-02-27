# DODO Stack Game

A complete 3D browser-based clone of the popular "Stack" mobile game, themed with DODO Pizza boxes. Built purely with HTML, CSS, and Three.js — no build tools required.

## Features
- **3D Physics and Rendering:** Built from scratch using Three.js.
- **DODO Pizza Branding:** Custom pizza box textures that dynamically clip and adjust when sliced.
- **Stacking Mechanics:** Perfectly stack boxes for "sparkles", or mistime it and watch the overhang slice off and fall with gravity.
- **Dynamic Camera:** The isometric camera smoothly pans up as your tower grows.
- **Progressive Difficulty:** The boxes speed up as your score increases!

## How to Play

Since this game uses standard web technologies, you only need a local web server to serve the files (to avoid browser CORS restrictions when loading 3D assets/textures).

### Prerequisites
Make sure you have [Python](https://www.python.org/downloads/) installed on your computer.

### Running the Game
1. Open your terminal (Command Prompt, PowerShell, or macOS/Linux Terminal).
2. Navigate to the directory containing this project.
3. Start a local Python HTTP server by running this command:
   ```bash
   python -m http.server 3456
   ```
4. Open your favorite web browser (Chrome, Firefox, Edge, Safari).
5. Go to the following address:
   ```
   http://localhost:3456
   ```

### Controls
Enjoy the game with a single click or tap!
- **Click or Tap anywhere** on the screen to stop the moving pizza box.
- Try to perfectly align it with the box below. If you're off, the overhanging piece is sliced off, and your target area for the next box becomes smaller.
- If you miss the tower entirely, game over!
- Earn **5 DODO Coins** for every successful placement.

## Project Structure
- `index.html` - The main entry point loading Three.js and our custom scripts.
- `style.css` - UI overlay styles (score, progress bar, game over screen).
- `game.js` - Main game loop, camera management, WebGL rendering, and tap logic.
- `block.js` - Pizza box meshes, colors, branding labels, and the slicing engine.
- `ui.js` - Scoreboard and end-screen logic.
