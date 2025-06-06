importScripts('mandelbrot.js');

let wasmModule = null;

function getColor(iter, maxIter) {
  if (iter === maxIter) return [0, 0, 0];

  const t = (iter % 256) / 256.0;

  const r = (0.5 + 0.5 * Math.cos(6.28318 * t)) * 255;
  const g = (0.5 + 0.5 * Math.cos(6.28318 * t + 2.09439)) * 255;
  const b = (0.5 + 0.5 * Math.cos(6.28318 * t + 4.18879)) * 255;

  return [Math.floor(r), Math.floor(g), Math.floor(b)]; // convert to integer
}

Module().then(mod => {
  wasmModule = mod;

  const mandelbrot = wasmModule.cwrap('mandelbrot', 'number', ['number', 'number', 'number']);

  onmessage = function (e) {
    const { tile, width, height, centerX, centerY, zoom, maxIter } = e.data;
    const pixels = [];

    for (let j = 0; j < tile.h; j++) {
      for (let i = 0; i < tile.w; i++) {
        const x = tile.x + i;
        const y = tile.y + j;

        // Convert pixel coords to complex plane
        const re = centerX + (x - width / 2) / width * (4 / zoom);
        const im = centerY + (y - height / 2) / height * (4 / zoom);

        const iter = mandelbrot(re, im, maxIter);
        const color = getColor(iter, maxIter);
        pixels.push(color[0], color[1], color[2]);
      }
    }

    postMessage({ x: tile.x, y: tile.y, w: tile.w, h: tile.h, pixels });
  };
});