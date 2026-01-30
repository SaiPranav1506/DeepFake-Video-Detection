(() => {
  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  function init() {
    const body = document.body;
    if (!body || !body.classList.contains('space-ui')) return;

    // If the user prefers reduced motion, keep the theme but skip animation.
    const reduce = prefersReducedMotion();

    let scene = document.getElementById('space-scene');
    if (!scene) {
      scene = document.createElement('div');
      scene.id = 'space-scene';
      scene.className = 'space-scene';
      scene.setAttribute('aria-hidden', 'true');
      body.prepend(scene);
    }

    let canvas = document.getElementById('space-canvas');
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = 'space-canvas';
      scene.appendChild(canvas);
    }

    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    let w = 0;
    let h = 0;
    let dpr = 1;

    // Stars are static in world-space; we parallax them with the mouse.
    /** @type {{x:number,y:number,r:number,a:number,z:number}[]} */
    let stars = [];

    const state = {
      // target mouse normalized [-0.5..0.5]
      tx: 0,
      ty: 0,
      // smoothed mouse
      x: 0,
      y: 0,
    };

    function resize() {
      dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
      w = Math.max(1, window.innerWidth);
      h = Math.max(1, window.innerHeight);
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // density scales gently with area
      const n = Math.floor(Math.sqrt(w * h) * 0.9);
      stars = new Array(n).fill(0).map(() => {
        const z = Math.random() * 1.0 + 0.15; // depth
        return {
          x: Math.random() * w,
          y: Math.random() * h,
          r: (Math.random() * 1.3 + 0.3) * (1.0 / z),
          a: Math.random() * 0.65 + 0.20,
          z,
        };
      });
    }

    function setCSSVars(nx, ny) {
      body.style.setProperty('--mx', String(nx));
      body.style.setProperty('--my', String(ny));
    }

    function onPointerMove(e) {
      const nx = clamp((e.clientX / w) - 0.5, -0.5, 0.5);
      const ny = clamp((e.clientY / h) - 0.5, -0.5, 0.5);
      state.tx = nx;
      state.ty = ny;
      if (reduce) {
        // still update CSS variables so the background responds without continuous animation
        setCSSVars(nx, ny);
      }
    }

    function draw() {
      // Smooth follow to avoid jitter
      state.x += (state.tx - state.x) * 0.08;
      state.y += (state.ty - state.y) * 0.08;

      setCSSVars(state.x, state.y);

      ctx.clearRect(0, 0, w, h);

      // Parallax offsets for star layer (foreground moves more)
      const ox = state.x * 40;
      const oy = state.y * 28;

      // Subtle vignetting
      const vg = ctx.createRadialGradient(w * 0.5, h * 0.55, Math.min(w, h) * 0.2, w * 0.5, h * 0.55, Math.max(w, h) * 0.7);
      vg.addColorStop(0, 'rgba(0,0,0,0)');
      vg.addColorStop(1, 'rgba(0,0,0,0.55)');
      ctx.fillStyle = vg;
      ctx.fillRect(0, 0, w, h);

      for (const s of stars) {
        const px = s.x + ox * (1.0 / (s.z + 0.2));
        const py = s.y + oy * (1.0 / (s.z + 0.2));
        if (px < -20 || px > w + 20 || py < -20 || py > h + 20) continue;

        ctx.beginPath();
        ctx.fillStyle = `rgba(255,255,255,${s.a})`;
        ctx.arc(px, py, s.r, 0, Math.PI * 2);
        ctx.fill();
      }

      // A couple of soft "comets" that react to mouse
      ctx.globalCompositeOperation = 'lighter';
      const cometX = w * 0.22 + state.x * 120;
      const cometY = h * 0.25 + state.y * 90;
      const comet = ctx.createRadialGradient(cometX, cometY, 0, cometX, cometY, 140);
      comet.addColorStop(0, 'rgba(34,211,238,0.10)');
      comet.addColorStop(0.4, 'rgba(139,92,246,0.06)');
      comet.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = comet;
      ctx.fillRect(0, 0, w, h);
      ctx.globalCompositeOperation = 'source-over';

      requestAnimationFrame(draw);
    }

    resize();

    // Use pointer events so it works with mouse + touchpads.
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('resize', () => {
      resize();
    });

    if (!reduce) requestAnimationFrame(draw);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
