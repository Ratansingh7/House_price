/* ============================================================
   EstateIQ — client logic
   ============================================================ */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const SVG_NS = "http://www.w3.org/2000/svg";

  const state = {
    meta: null,
    ranges: {},
    toggles: {},
    choices: {},
    lastResult: null,
    sensFeature: "area_sqft",
    sensCache: {},
  };

  const fmtMoney = (n, decimals = 0) =>
    "$" +
    Number(n || 0).toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });

  const fmtShort = (n) => {
    const v = Number(n || 0);
    if (Math.abs(v) >= 1e6) return "$" + (v / 1e6).toFixed(2) + "M";
    if (Math.abs(v) >= 1e3) return "$" + (v / 1e3).toFixed(1) + "k";
    return "$" + v.toFixed(0);
  };

  const fmtNum = (n, d = 0) =>
    Number(n || 0).toLocaleString("en-US", {
      minimumFractionDigits: d,
      maximumFractionDigits: d,
    });

  /* ---------------------------------------------------------
     Utilities
     --------------------------------------------------------- */
  function el(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function toast(message, kind = "") {
    const stack = $("#toasts");
    const node = document.createElement("div");
    node.className = "toast " + kind;
    node.innerHTML = `<span>${message}</span>`;
    stack.appendChild(node);
    setTimeout(() => {
      node.classList.add("out");
      setTimeout(() => node.remove(), 400);
    }, 3600);
  }

  function animateNumber(node, to, { decimals = 0, prefix = "", suffix = "", duration = 1100 } = {}) {
    const from = parseFloat(String(node.dataset.current || "0")) || 0;
    const start = performance.now();
    node.dataset.current = to;
    const tick = (now) => {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = from + (to - from) * eased;
      node.textContent = prefix + fmtNum(val, decimals) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function updateSliderFill(input) {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const val = parseFloat(input.value);
    const pct = max === min ? 0 : ((val - min) / (max - min)) * 100;
    input.style.setProperty("--pct", pct + "%");
  }

  function svgDefs(svg, id, from, to) {
    const defs = el("defs", {}, svg);
    const grad = el("linearGradient", { id, x1: "0", y1: "0", x2: "1", y2: "0" }, defs);
    el("stop", { offset: "0%", "stop-color": from }, grad);
    el("stop", { offset: "100%", "stop-color": to }, grad);
    return defs;
  }

  /* ---------------------------------------------------------
     Confetti
     --------------------------------------------------------- */
  const fx = (() => {
    const canvas = $("#fx-canvas");
    if (!canvas) return { burst: () => {} };
    const ctx = canvas.getContext("2d");
    let parts = [];
    const COLORS = ["#5b8cff", "#a855f7", "#22d3ee", "#34d399", "#fbbf24", "#fb7185"];

    const resize = () => {
      canvas.width = window.innerWidth * devicePixelRatio;
      canvas.height = window.innerHeight * devicePixelRatio;
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    function loop() {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      parts = parts.filter((p) => p.life > 0);
      parts.forEach((p) => {
        p.vy += 0.16;
        p.vx *= 0.995;
        p.x += p.vx;
        p.y += p.vy;
        p.rot += p.vr;
        p.life -= 1;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.globalAlpha = Math.max(0, Math.min(1, p.life / 45));
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
        ctx.restore();
      });
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);

    return {
      burst(x, y, amount = 90) {
        for (let i = 0; i < amount; i++) {
          const angle = Math.random() * Math.PI * 2;
          const speed = 3 + Math.random() * 9;
          parts.push({
            x, y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed - 4,
            w: 4 + Math.random() * 6,
            h: 3 + Math.random() * 5,
            rot: Math.random() * Math.PI,
            vr: (Math.random() - 0.5) * 0.3,
            life: 70 + Math.random() * 50,
            color: COLORS[(Math.random() * COLORS.length) | 0],
          });
        }
      },
    };
  })();

  /* ---------------------------------------------------------
     Build the form from /api/meta
     --------------------------------------------------------- */
  function buildRangeSlider(name, spec) {
    const row = document.createElement("div");
    row.className = "slider-row";

    const top = document.createElement("div");
    top.className = "slider-top";
    const label = document.createElement("span");
    label.className = "slider-name";
    label.textContent = spec.label;
    const value = document.createElement("span");
    value.className = "slider-val";
    row.appendChild(top);
    top.append(label, value);

    const stepDecimals = (s) => {
      const str = String(s);
      const dot = str.indexOf(".");
      return dot === -1 ? 0 : str.length - dot - 1;
    };
    const decimals = stepDecimals(spec.step);

    const input = document.createElement("input");
    input.type = "range";
    input.id = `f-${name}`;
    input.name = name;
    input.min = spec.min;
    input.max = spec.max;
    input.step = spec.step;
    input.value = spec.default;
    input.dataset.unit = spec.unit || "";
    input.dataset.decimals = String(decimals);
    row.appendChild(input);

    const scale = document.createElement("div");
    scale.className = "range-scale";
    scale.innerHTML = `<span>${fmtNum(spec.min, decimals)}</span><span>${fmtNum(spec.max, decimals)}</span>`;
    row.appendChild(scale);

    const fmt = (v) => {
      const d = parseFloat(input.dataset.decimals);
      const s = fmtNum(v, d);
      if (name === "area_sqft") return fmtNum(v, 0) + " ft²";
      if (name === "lot_size_sqft") return fmtNum(v, 0) + " ft²";
      if (name === "distance_to_cbd") return s + " km";
      if (name === "crime_rate") return s;
      return s + (spec.unit && spec.unit.startsWith("/") ? " " + spec.unit : "");
    };

    const sync = () => {
      value.textContent = fmt(input.value);
      updateSliderFill(input);
    };
    input.addEventListener("input", () => {
      sync();
      value.classList.add("bump");
      setTimeout(() => value.classList.remove("bump"), 180);
    });
    input.addEventListener("change", () => debouncedPredict());
    sync();

    state.ranges[name] = { input, spec };
    return row;
  }

  function buildToggle(name, label, defaultOn = false) {
    const wrap = document.createElement("label");
    wrap.className = "toggle" + (defaultOn ? " on" : "");
    wrap.innerHTML = `
      <input type="checkbox" name="${name}" ${defaultOn ? "checked" : ""} />
      <span class="track"></span><span>${label}</span>`;
    const input = wrap.querySelector("input");
    input.addEventListener("change", () => {
      wrap.classList.toggle("on", input.checked);
      debouncedPredict();
    });
    state.toggles[name] = { input, wrap };
    return wrap;
  }

  function buildForm(meta) {
    const rangesBox = $("#ranges");
    rangesBox.innerHTML = "";
    const order = [
      "area_sqft", "bedrooms", "bathrooms", "age",
      "floors", "garage", "lot_size_sqft",
      "overall_quality", "school_rating",
      "crime_rate", "distance_to_cbd",
    ];
    order.forEach((name) => {
      const spec = meta.ranges[name];
      if (spec) rangesBox.appendChild(buildRangeSlider(name, spec));
    });

    const nSel = $("#neighborhood");
    nSel.innerHTML = "";
    (meta.choices.neighborhood || []).forEach((n) => {
      const o = document.createElement("option");
      o.value = n; o.textContent = n;
      if (n === "Maplewood") o.selected = true;
      nSel.appendChild(o);
    });
    nSel.addEventListener("change", debouncedPredict);

    const rSel = $("#renovation");
    rSel.innerHTML = "";
    (meta.choices.renovation || []).forEach((r) => {
      const o = document.createElement("option");
      o.value = r;
      o.textContent = r.replace(/_/g, " ");
      if (r === "Not_Renovated") o.selected = true;
      rSel.appendChild(o);
    });
    rSel.addEventListener("change", debouncedPredict);

    const togBox = $("#toggles");
    togBox.innerHTML = "";
    const defaults = { has_pool: false, has_basement: false, air_conditioning: true, furnished: false, main_road: true };
    Object.entries(meta.toggles || {}).forEach(([name, label]) => {
      togBox.appendChild(buildToggle(name, label, defaults[name] ?? false));
    });
  }

  /* ---------------------------------------------------------
     Collect / apply payload
     --------------------------------------------------------- */
  function currentPayload() {
    const payload = {};
    Object.entries(state.ranges).forEach(([name, { input }]) => {
      payload[name] = parseFloat(input.value);
    });
    Object.entries(state.toggles).forEach(([name, { input }]) => {
      payload[name] = input.checked ? 1 : 0;
    });
    payload.neighborhood = $("#neighborhood").value;
    payload.renovation = $("#renovation").value;
    return payload;
  }

  function resetForm() {
    Object.entries(state.ranges).forEach(([name, { input, spec }]) => {
      input.value = spec.default;
      input.dispatchEvent(new Event("input"));
    });
    Object.entries(state.toggles).forEach(([name, { input }]) => {
      input.checked = name === "air_conditioning" || name === "main_road";
      input.dispatchEvent(new Event("change"));
    });
    $("#neighborhood").value = "Maplewood";
    $("#renovation").value = "Not_Renovated";
    toast("Form reset to defaults", "ok");
    predict();
  }

  /* ---------------------------------------------------------
     Prediction
     --------------------------------------------------------- */
  let predictTimer = null;
  function debouncedPredict() {
    clearTimeout(predictTimer);
    predictTimer = setTimeout(predict, 320);
  }

  async function predict(event) {
    if (event) event.preventDefault();
    const btn = $("#predict-btn");
    btn.classList.add("loading");
    btn.disabled = true;
    const payload = currentPayload();

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Prediction failed");
      state.lastResult = data.result;
      renderResult(data.result);
      syncUrl(payload);
    } catch (err) {
      toast(err.message || "Something went wrong", "err");
    } finally {
      btn.classList.remove("loading");
      btn.disabled = false;
    }
  }

  function renderResult(r) {
    const priceNode = $("#price");
    animateNumber(priceNode, r.price, { prefix: "$", duration: 1300 });
    priceNode.classList.remove("bump");
    void priceNode.offsetWidth;
    priceNode.classList.add("bump");

    $("#price-sub").textContent =
      `≈ ${fmtMoney(r.lower)} – ${fmtMoney(r.upper)} · ${r.model} ensemble`;

    // range bar — scaled to the neighbourhood so the band stays legible
    const stats = (state.meta && state.meta.price_stats) || {};
    const ns = (state.meta && state.meta.neighborhood_stats && state.meta.neighborhood_stats[r.neighborhood]) || {};
    const lo = Math.min(r.lower, ns.min ?? stats.min ?? r.lower);
    const hi = Math.max(r.upper, ns.max ?? stats.max ?? r.upper);
    const pct = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo || 1)) * 100));
    const fill = $("#range-fill");
    fill.style.width = pct(r.upper) + "%";
    fill.style.left = pct(r.lower) + "%";
    fill.style.right = "auto";
    $("#range-marker").style.left = pct(r.price) + "%";
    $("#range-low").textContent = fmtMoney(r.lower);
    $("#range-high").textContent = fmtMoney(r.upper);
    $("#range-confidence").textContent = `${r.confidence}% confidence`;

    // KPIs
    animateNumber($("#ppsqft"), r.price_per_sqft, { prefix: "$", decimals: 0, duration: 900 });
    const conf = Math.max(0, Math.min(100, r.confidence));
    $("#conf-value").textContent = conf.toFixed(0) + "%";
    const ring = $(".ring-fg");
    const C = 100.5;
    ring.style.strokeDashoffset = String(C * (1 - conf / 100));

    const vs = $("#vs-median");
    if (r.delta_vs_median === null || r.delta_vs_median === undefined) {
      vs.textContent = "—";
      vs.style.color = "";
    } else {
      const pctDelta = r.neighborhood_median ? (r.delta_vs_median / r.neighborhood_median) * 100 : 0;
      vs.textContent = `${pctDelta >= 0 ? "+" : ""}${pctDelta.toFixed(1)}%`;
      vs.style.color = pctDelta >= 0 ? "var(--good)" : "var(--bad)";
    }

    renderCompare(r);

    // confetti
    const rect = $("#price").getBoundingClientRect();
    fx.burst(rect.left + rect.width / 2, rect.top + rect.height / 2, 110);

    if (state.meta) loadSensitivity(state.sensFeature, true);
  }

  function renderCompare(r) {
    const box = $("#compare-bars");
    const hood = r.neighborhood;
    const ns = (state.meta && state.meta.neighborhood_stats && state.meta.neighborhood_stats[hood]) || {};
    const mean = ns.mean || 0;
    const median = ns.median || 0;
    const max = Math.max(r.price, mean, median) || 1;
    $("#compare-hood").textContent = hood;

    const rows = [
      { label: "This home", value: r.price, cls: "a" },
      { label: "Area mean", value: mean, cls: "b" },
      { label: "Median", value: median, cls: "b" },
    ];
    box.innerHTML = "";
    rows.forEach((row) => {
      const wrap = document.createElement("div");
      wrap.className = "cbar";
      wrap.innerHTML = `
        <span class="cbar-label">${row.label}</span>
        <span class="cbar-track"><span class="cbar-fill ${row.cls}" style="width:0%"></span></span>
        <span class="cbar-val">${fmtShort(row.value)}</span>`;
      box.appendChild(wrap);
      requestAnimationFrame(() => {
        wrap.querySelector(".cbar-fill").style.width = (row.value / max) * 100 + "%";
      });
    });
  }

  /* ---------------------------------------------------------
     Sensitivity chart
     --------------------------------------------------------- */
  const SENS_FEATURES = [
    "area_sqft", "age", "bedrooms", "bathrooms",
    "distance_to_cbd", "overall_quality", "garage", "crime_rate",
  ];

  function buildSensChips() {
    const box = $("#sens-controls");
    box.innerHTML = "";
    SENS_FEATURES.forEach((f) => {
      const spec = state.meta.ranges[f];
      if (!spec) return;
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "sens-chip" + (f === state.sensFeature ? " active" : "");
      chip.textContent = spec.label;
      chip.addEventListener("click", () => {
        state.sensFeature = f;
        $$(".sens-chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        loadSensitivity(f);
      });
      box.appendChild(chip);
    });
  }

  async function loadSensitivity(feature, silent = false) {
    if (!state.meta) return;
    const payload = currentPayload();
    payload.feature = feature;
    payload.steps = 26;
    const key = JSON.stringify(payload);
    if (state.sensCache[key]) {
      drawSensChart(state.sensCache[key]);
      return;
    }
    try {
      const res = await fetch("/api/sensitivity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error);
      state.sensCache[key] = data;
      drawSensChart(data);
      drawHeroSpark(data);
    } catch (err) {
      if (!silent) toast(err.message || "Sensitivity failed", "err");
    }
  }

  function drawSensChart(data) {
    const svg = $("#sens-chart");
    svg.innerHTML = "";
    const W = 640, H = 300, P = { t: 18, r: 18, b: 42, l: 62 };
    const pts = data.points || [];
    if (!pts.length) return;

    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.price);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const padY = (yMax - yMin) * 0.12 || yMax * 0.1 || 1;
    const lo = yMin - padY, hi = yMax + padY;

    const sx = (v) => P.l + ((v - xMin) / (xMax - xMin || 1)) * (W - P.l - P.r);
    const sy = (v) => H - P.b - ((v - lo) / (hi - lo || 1)) * (H - P.t - P.b);

    svgDefs(svg, "lineGrad", "#22d3ee", "#a855f7");

    // grid + y labels
    for (let i = 0; i <= 4; i++) {
      const y = P.t + ((H - P.t - P.b) / 4) * i;
      el("line", { x1: P.l, y1: y, x2: W - P.r, y2: y, class: "chart-grid" }, svg);
      const val = hi - ((hi - lo) / 4) * i;
      const t = el("text", { x: P.l - 10, y: y + 4, class: "chart-label", "text-anchor": "end" }, svg);
      t.textContent = fmtShort(val);
    }
    el("line", { x1: P.l, y1: H - P.b, x2: W - P.r, y2: H - P.b, class: "chart-axis" }, svg);

    // x labels
    [0, 0.5, 1].forEach((f) => {
      const v = xMin + (xMax - xMin) * f;
      const t = el("text", {
        x: sx(v), y: H - P.b + 20, class: "chart-label", "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle",
      }, svg);
      t.textContent = fmtNum(v, data.unit === "0-1" ? 2 : 0) + (data.unit && !data.unit.startsWith("/") ? " " + data.unit : "");
    });

    // area fill
    const areaD = `M ${sx(xs[0])} ${H - P.b} ` +
      pts.map((p) => `L ${sx(p.x)} ${sy(p.price)}`).join(" ") +
      ` L ${sx(xs[xs.length - 1])} ${H - P.b} Z`;
    el("path", { d: areaD, fill: "url(#lineGrad)", opacity: "0.13" }, svg);

    // line
    const lineD = pts.map((p, i) => `${i ? "L" : "M"} ${sx(p.x)} ${sy(p.price)}`).join(" ");
    const line = el("path", { d: lineD, class: "chart-line chart-line-anim" }, svg);

    // baseline marker
    if (data.baseline !== null && data.baseline !== undefined) {
      const bx = sx(data.baseline);
      el("line", {
        x1: bx, y1: P.t, x2: bx, y2: H - P.b,
        stroke: "var(--accent-3)", "stroke-width": 1.4, "stroke-dasharray": "4 4", opacity: 0.75,
      }, svg);
      const bt = el("text", { x: bx + 6, y: P.t + 12, class: "chart-label", fill: "#22d3ee" }, svg);
      bt.textContent = "current";
    }

    // interactive points
    const tip = $("#sens-tip");
    const wrap = svg.parentElement;
    pts.forEach((p) => {
      const cx = sx(p.x), cy = sy(p.price);
      el("circle", { cx, cy, r: 4, class: "chart-point" }, svg);
      const hit = el("circle", { cx, cy, r: 14, class: "chart-point-hit" }, svg);
      hit.addEventListener("mouseenter", () => {
        const box = svg.getBoundingClientRect();
        const sxr = (cx / W) * box.width;
        const syr = (cy / H) * box.height;
        tip.style.left = sxr + "px";
        tip.style.top = syr + "px";
        tip.textContent = `${fmtNum(p.x, data.unit === "0-1" ? 2 : 0)}${data.unit && !data.unit.startsWith("/") ? " " + data.unit : ""} → ${fmtMoney(p.price)}`;
        tip.classList.add("show");
      });
      hit.addEventListener("mouseleave", () => tip.classList.remove("show"));
    });

    $("#sens-caption").textContent =
      `Sweeping ${data.label} from ${fmtNum(xMin, 0)} to ${fmtNum(xMax, 0)} ${data.unit || ""} — ` +
      `value moves ${fmtMoney(yMin)} → ${fmtMoney(yMax)}.`;

    // hero spark
    drawHeroSpark(data);
  }

  function drawHeroSpark(data) {
    const svg = $("#hero-spark");
    if (!svg || !data || !data.points) return;
    svg.innerHTML = "";
    const W = 320, H = 120, P = 8;
    const pts = data.points;
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.price);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const sx = (v) => P + ((v - xMin) / (xMax - xMin || 1)) * (W - P * 2);
    const sy = (v) => H - P - ((v - yMin) / (yMax - yMin || 1)) * (H - P * 2);

    const defs = el("defs", {}, svg);
    const g = el("linearGradient", { id: "sparkGrad", x1: "0", y1: "0", x2: "0", y2: "1" }, defs);
    el("stop", { offset: "0%", "stop-color": "#5b8cff" }, g);
    el("stop", { offset: "100%", "stop-color": "#a855f7", "stop-opacity": "0" }, g);

    const d = pts.map((p, i) => `${i ? "L" : "M"} ${sx(p.x)} ${sy(p.price)}`).join(" ");
    el("path", {
      d: `${d} L ${sx(xs[xs.length - 1])} ${H} L ${sx(xs[0])} ${H} Z`,
      class: "spark-area",
    }, svg);
    el("path", { d, class: "spark-line" }, svg);
    el("circle", { cx: sx(xs[xs.length - 1]), cy: sy(ys[ys.length - 1]), r: 4, class: "spark-dot" }, svg);
  }

  /* ---------------------------------------------------------
     Feature importance
     --------------------------------------------------------- */
  function renderImportance(imp) {
    const box = $("#imp-list");
    box.innerHTML = "";
    const entries = Object.entries(imp || {}).slice(0, 12);
    if (!entries.length) {
      box.innerHTML = `<p class="chart-caption">No importance data yet.</p>`;
      return;
    }
    const max = Math.max(...entries.map(([, v]) => Math.abs(v))) || 1;
    entries.forEach(([name, value], i) => {
      const row = document.createElement("div");
      row.className = "imp";
      row.innerHTML = `
        <span class="imp-name" title="${name}">${name.replace(/_/g, " ")}</span>
        <span class="imp-track"><span class="imp-fill"></span></span>
        <span class="imp-val">${(value / max * 100).toFixed(0)}%</span>`;
      box.appendChild(row);
      const fill = row.querySelector(".imp-fill");
      setTimeout(() => { fill.style.width = (Math.abs(value) / max) * 100 + "%"; }, 60 * i);
    });
  }

  /* ---------------------------------------------------------
     Samples table
     --------------------------------------------------------- */
  async function loadSamples() {
    const tbody = $("#samples-table tbody");
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      if (!data.ok) throw new Error(data.error);
      tbody.innerHTML = "";
      data.samples.forEach((s, i) => {
        const err = s.predicted - s.actual;
        const errPct = (err / s.actual) * 100;
        const tr = document.createElement("tr");
        tr.style.animationDelay = i * 70 + "ms";
        tr.innerHTML = `
          <td>${fmtNum(s.area_sqft)} ft²</td>
          <td>${s.bedrooms}</td>
          <td>${s.bathrooms}</td>
          <td>${s.age} yr</td>
          <td>${s.neighborhood}</td>
          <td>${fmtMoney(s.actual)}</td>
          <td>${fmtMoney(s.predicted)}</td>
          <td class="${err >= 0 ? "err-up" : "err-down"}">${err >= 0 ? "+" : ""}${errPct.toFixed(1)}%</td>`;
        tbody.appendChild(tr);
      });
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="8">Could not load rows</td></tr>`;
    }
  }

  /* ---------------------------------------------------------
     URL state (shareable links)
     --------------------------------------------------------- */
  function syncUrl(payload) {
    const p = new URLSearchParams();
    Object.entries(payload).forEach(([k, v]) => p.set(k, v));
    history.replaceState(null, "", "?" + p.toString());
  }

  function applyUrlState() {
    const p = new URLSearchParams(location.search);
    if (!p.toString()) return false;
    let applied = false;
    Object.entries(state.ranges).forEach(([name, { input }]) => {
      if (p.has(name)) {
        const v = parseFloat(p.get(name));
        if (!Number.isNaN(v)) {
          input.value = Math.min(parseFloat(input.max), Math.max(parseFloat(input.min), v));
          input.dispatchEvent(new Event("input"));
          applied = true;
        }
      }
    });
    Object.entries(state.toggles).forEach(([name, { input }]) => {
      if (p.has(name)) {
        input.checked = p.get(name) === "1" || p.get(name) === "true";
        input.dispatchEvent(new Event("change"));
        applied = true;
      }
    });
    if (p.has("neighborhood")) { $("#neighborhood").value = p.get("neighborhood"); applied = true; }
    if (p.has("renovation")) { $("#renovation").value = p.get("renovation"); applied = true; }
    return applied;
  }

  /* ---------------------------------------------------------
     Tabs
     --------------------------------------------------------- */
  function initTabs() {
    $$("#tabs .tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        $$("#tabs .tab").forEach((t) => t.classList.remove("active"));
        $$(".tab-panel").forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        $("#tab-" + tab.dataset.tab).classList.add("active");
        if (tab.dataset.tab === "sensitivity" && !Object.keys(state.sensCache).length) loadSensitivity(state.sensFeature);
      });
    });
  }

  /* ---------------------------------------------------------
     Theme
     --------------------------------------------------------- */
  function initTheme() {
    const saved = localStorage.getItem("estateiq-theme");
    const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
    document.documentElement.dataset.theme = saved || (prefersLight ? "light" : "dark");
    $("#theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("estateiq-theme", next);
      toast(next === "dark" ? "Dark mode" : "Light mode", "ok");
    });
  }

  /* ---------------------------------------------------------
     Retrain
     --------------------------------------------------------- */
  function initRetrain() {
    const btn = $("#retrain-btn");
    btn.addEventListener("click", async () => {
      btn.classList.add("loading");
      btn.disabled = true;
      try {
        const res = await fetch("/api/retrain", { method: "POST" });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error);
        toast("Retraining started — this takes a moment", "ok");
        pollRetrain();
      } catch (err) {
        toast(err.message || "Could not start retraining", "err");
      } finally {
        setTimeout(() => { btn.classList.remove("loading"); btn.disabled = false; }, 1200);
      }
    });
  }

  async function pollRetrain() {
    const res = await fetch("/api/retrain/status");
    const data = await res.json();
    if (data.running) {
      setTimeout(pollRetrain, 2500);
    } else {
      toast("Model retrained and reloaded", "ok");
      await init();
    }
  }

  /* ---------------------------------------------------------
     Boot
     --------------------------------------------------------- */
  async function init() {
    const res = await fetch("/api/meta");
    const meta = await res.json();
    state.meta = meta;
    state.ranges = {};
    state.toggles = {};
    state.sensCache = {};

    buildForm(meta);
    buildSensChips();
    renderImportance(meta.feature_importance);

    $("#model-name").textContent = meta.model || "model";
    $("#trained-at").textContent = meta.trained_at ? meta.trained_at.replace("T", " ") : "—";
    $("#feature-count").textContent = (meta.features || []).length;
    $("#hero-chip").textContent = "live";

    const t = meta.metrics || {};
    if (t.r2 !== undefined) animateNumber($("#stat-strip .stat:nth-child(1) .stat-value"), t.r2 * 100, { decimals: 1, suffix: "%" });
    if (t.rmse !== undefined) animateNumber($("#stat-strip .stat:nth-child(2) .stat-value"), t.rmse / 1000, { decimals: 1, prefix: "$", suffix: "k" });
    if (t.mae !== undefined) animateNumber($("#stat-strip .stat:nth-child(3) .stat-value"), t.mae / 1000, { decimals: 1, prefix: "$", suffix: "k" });
    if (meta.n_samples) animateNumber($("#stat-strip .stat:nth-child(4) .stat-value"), meta.n_samples, { decimals: 0, suffix: "+" });

    const applied = applyUrlState();
    await predict();
    if (applied) toast("Loaded a shared configuration", "ok");
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initTabs();
    initRetrain();
    $("#predict-form").addEventListener("submit", predict);
    $("#reset-btn").addEventListener("click", resetForm);

    // reveal-on-scroll for sections below the fold
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.style.animation = "revealIn .8s var(--ease) forwards";
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.12 });
    $$(".samples-panel").forEach((n) => io.observe(n));

    init().catch((e) => toast(e.message || "Failed to start", "err"));
    loadSamples();
  });
})();
