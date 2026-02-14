"""Three.js-powered interactive components for the F1 Pit Strategy Simulator."""
import streamlit.components.v1 as components


def render_particle_hero(height: int = 420):
    """Animated particle speed-trail hero background with F1 branding."""
    components.html(f"""
    <div id="hero-3d" style="position:relative;width:100%;height:{height}px;overflow:hidden;background:#0a0a0a;
         border-bottom:3px solid #e10600;">
      <canvas id="particleCanvas" style="position:absolute;top:0;left:0;width:100%;height:100%;"></canvas>
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;z-index:10;pointer-events:none;">
        <div style="font-size:3.2rem;margin-bottom:0.5rem;">🏎️</div>
        <h1 style="font-family:'Orbitron',sans-serif;font-size:2.6rem;letter-spacing:3px;margin:0;
            background:linear-gradient(90deg,#e10600,#ff4444,#e10600);-webkit-background-clip:text;
            -webkit-text-fill-color:transparent;background-size:200% 100%;
            animation:heroShimmer 3s ease-in-out infinite;">F1 PIT STRATEGY<br>SIMULATOR</h1>
        <p style="color:#999;font-family:'Inter',sans-serif;font-size:1rem;margin-top:0.6rem;">
            Machine Learning Lap-Time Prediction + Monte Carlo Strategy Optimization</p>
      </div>
      <div style="position:absolute;top:0;left:50%;transform:translateX(-50%);width:500px;height:500px;
           background:radial-gradient(circle,rgba(225,6,0,0.06) 0%,transparent 70%);pointer-events:none;"></div>
    </div>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;800&family=Inter:wght@400&display=swap');
      @keyframes heroShimmer {{ 0%,100%{{background-position:0% 50%;}} 50%{{background-position:100% 50%;}} }}
    </style>
    <script>
    (function() {{
      const canvas = document.getElementById('particleCanvas');
      const ctx = canvas.getContext('2d');
      let W, H, particles = [], mouse = {{x: -1000, y: -1000}};

      function resize() {{
        W = canvas.width = canvas.parentElement.offsetWidth;
        H = canvas.height = canvas.parentElement.offsetHeight;
      }}
      resize();
      window.addEventListener('resize', resize);

      // Mouse tracking for interactivity
      canvas.parentElement.addEventListener('mousemove', e => {{
        const rect = canvas.parentElement.getBoundingClientRect();
        mouse.x = e.clientX - rect.left;
        mouse.y = e.clientY - rect.top;
      }});
      canvas.parentElement.addEventListener('mouseleave', () => {{ mouse.x = -1000; mouse.y = -1000; }});

      const PARTICLE_COUNT = 120;
      const TRAIL_LENGTH = 12;
      const COLORS = ['#e10600','#ff4444','#ff6666','#ff8888','#ffffff','#ffaa00'];

      class Particle {{
        constructor() {{ this.reset(); }}
        reset() {{
          this.x = -20;
          this.y = Math.random() * H;
          this.vx = 2 + Math.random() * 6;
          this.vy = (Math.random() - 0.5) * 0.8;
          this.size = 1 + Math.random() * 2;
          this.color = COLORS[Math.floor(Math.random() * COLORS.length)];
          this.alpha = 0.3 + Math.random() * 0.7;
          this.trail = [];
          this.life = 0;
        }}
        update() {{
          // Mouse repulsion
          const dx = this.x - mouse.x, dy = this.y - mouse.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {{
            const force = (120 - dist) / 120 * 0.5;
            this.vy += dy / dist * force;
            this.vx += dx / dist * force * 0.3;
          }}
          // Dampen vertical drift
          this.vy *= 0.98;
          this.trail.push({{x: this.x, y: this.y, alpha: this.alpha}});
          if (this.trail.length > TRAIL_LENGTH) this.trail.shift();
          this.x += this.vx;
          this.y += this.vy;
          this.life++;
          if (this.x > W + 20 || this.y < -20 || this.y > H + 20) this.reset();
        }}
        draw(ctx) {{
          // Trail
          for (let i = 0; i < this.trail.length - 1; i++) {{
            const t = this.trail[i];
            const progress = i / this.trail.length;
            ctx.beginPath();
            ctx.moveTo(this.trail[i].x, this.trail[i].y);
            ctx.lineTo(this.trail[i+1].x, this.trail[i+1].y);
            ctx.strokeStyle = this.color;
            ctx.globalAlpha = progress * this.alpha * 0.4;
            ctx.lineWidth = this.size * progress;
            ctx.stroke();
          }}
          // Head glow
          ctx.globalAlpha = this.alpha;
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
          ctx.fillStyle = this.color;
          ctx.fill();
          // Bloom
          ctx.globalAlpha = this.alpha * 0.15;
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.size * 4, 0, Math.PI * 2);
          ctx.fillStyle = this.color;
          ctx.fill();
          ctx.globalAlpha = 1;
        }}
      }}

      for (let i = 0; i < PARTICLE_COUNT; i++) {{
        const p = new Particle();
        p.x = Math.random() * W;
        particles.push(p);
      }}

      function animate() {{
        ctx.clearRect(0, 0, W, H);
        // Subtle grid lines
        ctx.globalAlpha = 0.03;
        ctx.strokeStyle = '#e10600';
        for (let y = 0; y < H; y += 40) {{
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        }}
        ctx.globalAlpha = 1;
        particles.forEach(p => {{ p.update(); p.draw(ctx); }});
        requestAnimationFrame(animate);
      }}
      animate();
    }})();
    </script>
    """, height=height)


def render_live_telemetry(height: int = 220):
    """Clean animated telemetry dashboard with live gauges and a scrolling race data strip."""
    components.html(f"""
    <div id="telemetry" style="width:100%;height:{height}px;border-radius:16px;overflow:hidden;
         background:linear-gradient(135deg,#0a0a0a,#111);border:1px solid rgba(225,6,0,0.15);
         position:relative;">
      <canvas id="telCanvas" style="width:100%;height:100%;"></canvas>
    </div>
    <script>
    (function() {{
      const canvas = document.getElementById('telCanvas');
      const ctx = canvas.getContext('2d');
      const container = canvas.parentElement;
      let W, H, time = 0;

      function resize() {{
        W = canvas.width = container.offsetWidth;
        H = canvas.height = container.offsetHeight;
      }}
      resize();
      window.addEventListener('resize', resize);

      // Simulated telemetry data generators
      function speed(t) {{ return 180 + 120 * Math.sin(t * 0.7) + 30 * Math.sin(t * 1.3); }}
      function rpm(t) {{ return 8000 + 4000 * Math.sin(t * 0.9) + 1000 * Math.sin(t * 2.1); }}
      function throttle(t) {{ return 50 + 45 * Math.sin(t * 0.7 + 0.5); }}
      function brake(t) {{ const v = Math.sin(t * 0.7 + 3.14); return v > 0.3 ? (v - 0.3) * 100 : 0; }}

      // History buffers for scrolling chart
      const histLen = 200;
      let speedHist = new Array(histLen).fill(0);
      let throttleHist = new Array(histLen).fill(0);
      let brakeHist = new Array(histLen).fill(0);

      // Draw a mini gauge
      function drawGauge(x, y, w, h, value, max, label, color, unit) {{
        const pct = Math.min(value / max, 1);
        // Background bar
        ctx.fillStyle = '#1a1a1a';
        ctx.beginPath();
        ctx.roundRect(x, y + 18, w, 10, 4);
        ctx.fill();
        // Value bar
        const grad = ctx.createLinearGradient(x, 0, x + w, 0);
        grad.addColorStop(0, color);
        grad.addColorStop(1, color + 'aa');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y + 18, w * pct, 10, 4);
        ctx.fill();
        // Glow on top
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(x, y + 18, w * pct, 4, [4,4,0,0]);
        ctx.fill();
        ctx.globalAlpha = 1;
        // Label
        ctx.font = '9px Orbitron, sans-serif';
        ctx.fillStyle = '#666';
        ctx.textAlign = 'left';
        ctx.fillText(label, x, y + 12);
        // Value
        ctx.font = 'bold 11px Orbitron, sans-serif';
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'right';
        ctx.fillText(Math.round(value) + unit, x + w, y + 12);
      }}

      function animate() {{
        time += 0.03;
        ctx.clearRect(0, 0, W, H);

        const s = speed(time), r = rpm(time), th = throttle(time), br = brake(time);
        speedHist.push(s); speedHist.shift();
        throttleHist.push(th); throttleHist.shift();
        brakeHist.push(br); brakeHist.shift();

        const pad = 20;
        const gaugeW = W * 0.35;
        const gaugeX = pad;

        // Title
        ctx.font = '10px Orbitron, sans-serif';
        ctx.fillStyle = '#e10600';
        ctx.textAlign = 'left';
        ctx.fillText('LIVE TELEMETRY', pad, 18);
        // Blinking dot
        ctx.globalAlpha = 0.5 + 0.5 * Math.sin(time * 5);
        ctx.beginPath();
        ctx.arc(pad + 100, 14, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#e10600';
        ctx.fill();
        ctx.globalAlpha = 1;

        // Gauges
        drawGauge(gaugeX, 28, gaugeW, 30, s, 360, 'SPEED', '#e10600', ' km/h');
        drawGauge(gaugeX, 64, gaugeW, 30, r, 15000, 'RPM', '#ffaa00', '');
        drawGauge(gaugeX, 100, gaugeW, 30, th, 100, 'THROTTLE', '#43B02A', '%');
        drawGauge(gaugeX, 136, gaugeW, 30, br, 100, 'BRAKE', '#ff4444', '%');

        // Scrolling trace chart on the right
        const chartX = W * 0.42;
        const chartW = W * 0.55;
        const chartY = 30;
        const chartH = H - 50;

        // Chart background
        ctx.fillStyle = 'rgba(15,15,15,0.8)';
        ctx.beginPath();
        ctx.roundRect(chartX, chartY, chartW, chartH, 8);
        ctx.fill();
        ctx.strokeStyle = 'rgba(225,6,0,0.1)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Grid lines
        ctx.globalAlpha = 0.06;
        ctx.strokeStyle = '#fff';
        for (let i = 0; i < 5; i++) {{
          const gy = chartY + (chartH / 5) * i;
          ctx.beginPath(); ctx.moveTo(chartX, gy); ctx.lineTo(chartX + chartW, gy); ctx.stroke();
        }}
        ctx.globalAlpha = 1;

        // Draw traces
        function drawTrace(hist, max, color, alpha) {{
          ctx.beginPath();
          ctx.globalAlpha = alpha;
          for (let i = 0; i < histLen; i++) {{
            const x = chartX + (i / histLen) * chartW;
            const y = chartY + chartH - (hist[i] / max) * chartH * 0.9 - 5;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
          }}
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.globalAlpha = 1;
        }}
        drawTrace(speedHist, 360, '#e10600', 0.8);
        drawTrace(throttleHist, 100, '#43B02A', 0.5);
        drawTrace(brakeHist, 100, '#ff4444', 0.5);

        // Trace labels
        ctx.font = '8px Inter, sans-serif';
        ctx.textAlign = 'left';
        const labels = [['Speed','#e10600'],['Throttle','#43B02A'],['Brake','#ff4444']];
        labels.forEach(([l,c], i) => {{
          ctx.fillStyle = c;
          ctx.globalAlpha = 0.7;
          ctx.fillText('●', chartX + 8 + i * 60, chartY + chartH + 14);
          ctx.fillStyle = '#888';
          ctx.fillText(l, chartX + 16 + i * 60, chartY + chartH + 14);
        }});
        ctx.globalAlpha = 1;

        // Bottom race position ticker
        const tickerY = H - 4;
        ctx.globalAlpha = 0.05;
        ctx.fillStyle = '#e10600';
        ctx.fillRect(0, tickerY - 2, W, 4);
        ctx.globalAlpha = 1;

        requestAnimationFrame(animate);
      }}
      animate();
    }})();
    </script>
    """, height=height)


def render_simulation_loader(height: int = 280):
    """Animated loading spinner with racing particles while simulation runs."""
    components.html(f"""
    <div id="loader-3d" style="width:100%;height:{height}px;display:flex;flex-direction:column;
         align-items:center;justify-content:center;position:relative;overflow:hidden;">
      <canvas id="loaderCanvas" style="position:absolute;top:0;left:0;width:100%;height:100%;"></canvas>
      <div style="z-index:10;text-align:center;">
        <div style="font-size:2.5rem;animation:carBounce 1s ease-in-out infinite alternate;">🏎️</div>
        <p style="font-family:'Orbitron',sans-serif;color:#e10600;font-size:1rem;margin:0.5rem 0;
           animation:textPulse 2s ease-in-out infinite;">Running Monte Carlo Simulation</p>
        <p style="font-family:'Inter',sans-serif;color:#666;font-size:0.85rem;margin:0;">
           Analyzing thousands of strategy variations…</p>
      </div>
    </div>
    <style>
      @keyframes carBounce {{ 0% {{ transform: translateY(0); }} 100% {{ transform: translateY(-10px); }} }}
      @keyframes textPulse {{ 0%,100% {{ opacity: 0.6; }} 50% {{ opacity: 1; }} }}
    </style>
    <script>
    (function() {{
      const canvas = document.getElementById('loaderCanvas');
      const ctx = canvas.getContext('2d');
      let W, H, time = 0;
      function resize() {{
        W = canvas.width = canvas.parentElement.offsetWidth;
        H = canvas.height = canvas.parentElement.offsetHeight;
      }}
      resize();

      const rings = [];
      for (let i = 0; i < 40; i++) {{
        rings.push({{
          angle: Math.random() * Math.PI * 2,
          radius: 60 + Math.random() * 100,
          speed: 0.01 + Math.random() * 0.03,
          size: 1 + Math.random() * 2,
          color: ['#e10600','#ff4444','#ff6666','#ffaa00','#fff'][Math.floor(Math.random() * 5)],
          alpha: 0.3 + Math.random() * 0.5,
        }});
      }}

      function animate() {{
        time += 0.02;
        ctx.clearRect(0, 0, W, H);
        const cx = W / 2, cy = H / 2;

        // Orbiting ring
        ctx.globalAlpha = 0.08;
        ctx.beginPath();
        ctx.arc(cx, cy, 90, 0, Math.PI * 2);
        ctx.strokeStyle = '#e10600';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.globalAlpha = 1;

        // Particles orbiting
        rings.forEach(r => {{
          r.angle += r.speed;
          const x = cx + Math.cos(r.angle) * r.radius;
          const y = cy + Math.sin(r.angle) * r.radius * 0.5; // Elliptical
          ctx.globalAlpha = r.alpha * (0.5 + 0.5 * Math.sin(time + r.angle));
          ctx.beginPath();
          ctx.arc(x, y, r.size, 0, Math.PI * 2);
          ctx.fillStyle = r.color;
          ctx.fill();
          // Trail
          for (let j = 1; j <= 3; j++) {{
            const tx = cx + Math.cos(r.angle - j * r.speed * 3) * r.radius;
            const ty = cy + Math.sin(r.angle - j * r.speed * 3) * r.radius * 0.5;
            ctx.globalAlpha = r.alpha * (1 - j / 3) * 0.2;
            ctx.beginPath();
            ctx.arc(tx, ty, r.size * 0.6, 0, Math.PI * 2);
            ctx.fill();
          }}
        }});
        ctx.globalAlpha = 1;
        requestAnimationFrame(animate);
      }}
      animate();
    }})();
    </script>
    """, height=height)


def render_sidebar_tire_viz(compound: str = "MEDIUM", height: int = 100):
    """Small animated tire compound visualization for the sidebar."""
    color_map = {
        "SOFT": ("#FF1801", "#cc1300", "SOFT"),
        "MEDIUM": ("#FFC906", "#d4a800", "MEDIUM"),
        "HARD": ("#FFFFFF", "#cccccc", "HARD"),
        "INTERMEDIATE": ("#43B02A", "#358c22", "INTER"),
        "WET": ("#0067FF", "#0052cc", "WET"),
    }
    c1, c2, label = color_map.get(compound.upper(), color_map["MEDIUM"])
    components.html(f"""
    <div style="width:100%;height:{height}px;display:flex;align-items:center;justify-content:center;">
      <canvas id="tireViz" width="90" height="90"></canvas>
    </div>
    <script>
    (function() {{
      const canvas = document.getElementById('tireViz');
      const ctx = canvas.getContext('2d');
      let angle = 0;
      function draw() {{
        angle += 0.03;
        ctx.clearRect(0, 0, 90, 90);
        const cx = 45, cy = 45, r = 32;
        // Outer tire
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = '#222'; ctx.fill();
        ctx.lineWidth = 6; ctx.strokeStyle = '{c1}'; ctx.stroke();
        // Inner hub
        ctx.beginPath(); ctx.arc(cx, cy, 12, 0, Math.PI * 2);
        ctx.fillStyle = '#333'; ctx.fill();
        // Spokes with rotation
        for (let i = 0; i < 5; i++) {{
          const a = angle + (i * Math.PI * 2 / 5);
          ctx.beginPath();
          ctx.moveTo(cx + Math.cos(a) * 14, cy + Math.sin(a) * 14);
          ctx.lineTo(cx + Math.cos(a) * 28, cy + Math.sin(a) * 28);
          ctx.strokeStyle = '{c2}'; ctx.lineWidth = 2; ctx.stroke();
        }}
        // Label
        ctx.font = 'bold 8px Inter, sans-serif';
        ctx.fillStyle = '{c1}'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('{label}', cx, cy);
        // Glow
        ctx.beginPath(); ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
        ctx.strokeStyle = '{c1}'; ctx.lineWidth = 1;
        ctx.globalAlpha = 0.2 + 0.1 * Math.sin(angle * 3);
        ctx.stroke(); ctx.globalAlpha = 1;
        requestAnimationFrame(draw);
      }}
      draw();
    }})();
    </script>
    """, height=height)


def render_speed_gauge(value: float = 0, max_val: float = 100, label: str = "SIM PROGRESS", height: int = 130):
    """Animated speed gauge / tachometer style progress indicator."""
    pct = min(value / max_val, 1.0) if max_val > 0 else 0
    components.html(f"""
    <div style="width:100%;height:{height}px;display:flex;align-items:center;justify-content:center;">
      <canvas id="gaugeCanvas" width="200" height="120"></canvas>
    </div>
    <script>
    (function() {{
      const canvas = document.getElementById('gaugeCanvas');
      const ctx = canvas.getContext('2d');
      const target = {pct};
      let current = 0;

      function draw() {{
        current += (target - current) * 0.05;
        ctx.clearRect(0, 0, 200, 120);
        const cx = 100, cy = 95, r = 70;
        const startAngle = Math.PI * 0.8;
        const endAngle = Math.PI * 2.2;
        const range = endAngle - startAngle;

        // Background arc
        ctx.beginPath(); ctx.arc(cx, cy, r, startAngle, endAngle);
        ctx.strokeStyle = '#222'; ctx.lineWidth = 10; ctx.lineCap = 'round'; ctx.stroke();

        // Value arc
        const valAngle = startAngle + range * current;
        const grad = ctx.createLinearGradient(30, 0, 170, 0);
        grad.addColorStop(0, '#43B02A');
        grad.addColorStop(0.5, '#FFC906');
        grad.addColorStop(1, '#e10600');
        ctx.beginPath(); ctx.arc(cx, cy, r, startAngle, valAngle);
        ctx.strokeStyle = grad; ctx.lineWidth = 10; ctx.lineCap = 'round'; ctx.stroke();

        // Center text
        ctx.font = 'bold 18px Orbitron, sans-serif';
        ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(Math.round(current * 100) + '%', cx, cy - 10);
        ctx.font = '9px Inter, sans-serif';
        ctx.fillStyle = '#888';
        ctx.fillText('{label}', cx, cy + 10);

        if (Math.abs(current - target) > 0.001) requestAnimationFrame(draw);
      }}
      draw();
    }})();
    </script>
    """, height=height)
