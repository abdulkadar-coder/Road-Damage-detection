from __future__ import annotations


def motion_strip() -> str:
    """Return an isolated GSAP/Lenis motion panel with native-scroll fallback."""
    return r"""
    <!doctype html>
    <html><head>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet" href="https://unpkg.com/lenis@1.3.27/dist/lenis.css">
      <style>
        * { box-sizing: border-box; }
        html, body { margin: 0; background: transparent; color: #efeee6; font-family: Arial, sans-serif; }
        .frame { border: 1px solid #35372e; border-radius: 6px; background: #151612; padding: 12px 16px 14px; overflow: hidden; }
        .top { display:flex; justify-content:space-between; align-items:center; margin-bottom:9px; color:#a8aa9c; font:10px monospace; letter-spacing:.14em; text-transform:uppercase; }
        .top b { color:#d5ff58; font-weight:500; }
        .track-wrap { position:relative; }
        .track-wrap:before { content:""; position:absolute; height:1px; left:0; right:0; top:23px; background:#414338; }
        .beam { width:1px; height:42px; position:absolute; top:2px; left:0; z-index:2; background:#d5ff58; box-shadow:0 0 12px #d5ff58; }
        .rail { overflow-x:auto; position:relative; scrollbar-width:none; cursor:grab; }
        .rail::-webkit-scrollbar { display:none; }
        .track { display:flex; width:max-content; gap:8px; padding:0 1px 2px; }
        .card { position:relative; z-index:1; width:205px; min-height:58px; padding:9px 11px; border:1px solid #35372e; border-radius:4px; background:#1c1e19; transform-style:preserve-3d; transition:border-color .2s,background .2s; }
        .card:hover { border-color:#889354; background:#22251d; }
        .code { color:#d5ff58; font:11px monospace; letter-spacing:.08em; }
        .name { color:#efeee6; font-size:13px; margin-top:4px; }
        .dot { display:inline-block; width:5px; height:5px; margin:0 7px 1px 0; border-radius:50%; background:#fb8958; }
        .foot { display:flex; justify-content:space-between; margin-top:8px; color:#85877d; font:9px monospace; letter-spacing:.1em; text-transform:uppercase; }
        @media(prefers-reduced-motion:reduce) { * { scroll-behavior:auto !important; transition:none !important; } }
      </style>
    </head><body>
      <div class="frame">
        <div class="top"><span><b>Motion study</b> / RDD2022 class index</span><span>drag or scroll →</span></div>
        <div class="track-wrap"><div class="beam"></div><div id="rail" class="rail"><div id="track" class="track">
          <div class="card"><span class="dot"></span><span class="code">D00</span><div class="name">Longitudinal crack</div></div>
          <div class="card"><span class="dot"></span><span class="code">D10</span><div class="name">Transverse crack</div></div>
          <div class="card"><span class="dot"></span><span class="code">D20</span><div class="name">Alligator crack</div></div>
          <div class="card"><span class="dot"></span><span class="code">D40</span><div class="name">Pothole</div></div>
        </div></div></div>
        <div class="foot"><span>Four supported damage categories</span><span>Local model · uploaded media only</span></div>
      </div>
      <script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"></script>
      <script src="https://unpkg.com/lenis@1.3.27/dist/lenis.min.js"></script>
      <script>
        (function () {
          const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
          const rail = document.getElementById("rail");
          const track = document.getElementById("track");
          if (window.Lenis) {
            try { new Lenis({ wrapper: rail, content: track, orientation: "horizontal", gestureOrientation: "both", smoothWheel: true, autoRaf: true }); } catch (_) {}
          }
          if (!reduce && window.gsap) {
            gsap.from(".card", { opacity: 0, y: 12, stagger: 0.1, duration: 0.55, ease: "power2.out" });
            gsap.to(".beam", { x: 770, duration: 4, repeat: -1, ease: "none", yoyo: true });
            document.querySelectorAll(".card").forEach(function (card) {
              card.addEventListener("pointermove", function (event) {
                const r = card.getBoundingClientRect();
                const x = (event.clientX - r.left) / r.width - 0.5;
                const y = (event.clientY - r.top) / r.height - 0.5;
                gsap.to(card, { rotateY: x * 7, rotateX: -y * 6, duration: 0.25, overwrite: true });
              });
              card.addEventListener("pointerleave", function () {
                gsap.to(card, { rotateY: 0, rotateX: 0, duration: 0.35, ease: "power2.out" });
              });
            });
          }
        })();
      </script>
    </body></html>
    """

