// War-Games-Sim replay viewer (no dependencies). Format: viewer/README.md.
// Pure replay logic lives in replay-model.js (window.WarsimReplay).
(function () {
  "use strict";

  const R = window.WarsimReplay;
  const T = R.TILE;

  // ---- constants --------------------------------------------------------
  const TEAM_COLORS = ["#286b9f", "#b64c45", "#7654a5", "#28734f"];
  const TEAM_NAMES = ["Blue", "Red", "Violet", "Green"];
  const COLORS = {
    floorA: "#e7ecef", floorB: "#f0f3f5", wall: "#788795", wallTop: "#aeb9c2", wallShade: "#63717d",
    door: "#9a7958", doorEdge: "#72563a", doorOpen: "#b08b62", glass: "#6d9db8", stairs: "#d5e0e8", stairsLine: "#9aabb8",
    arrow: "#1769a6", dead: "#89949e", outline: "#ffffff", hit: "#b83e4b", miss: "#647381",
    select: "#1769a6", hover: "#263746", border: "#aab8c4",
  };
  const SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16];
  const TRACER_FRAMES = 6;       // how long a shot tracer stays visible
  const CACHE_MAX_PX = 4096;     // largest side of a pre-rendered floor canvas
  const ZOOM_MIN = 0.5;
  const ZOOM_MAX = 16;
  const FOLLOW_ZOOM = 2.2;
  const FOLLOW_EASE = 0.2;       // camera smoothing per animation frame
  const DRAG_THRESHOLD_PX = 4;
  const UNIT_RADIUS = 0.34;      // in tiles
  const SAMPLE_URL = "sample_replay.json";

  // ---- state ------------------------------------------------------------
  const S = {
    model: null,
    floorCaches: [],
    pos: 0,               // fractional frame index
    playing: false,
    speed: 1,
    floor: 0,
    followId: null,
    hoverId: null,
    cam: { cx: 0, cy: 0, zoom: 1 },
    opts: { ghost: true, others: true, tracers: true, labels: true, smooth: true },
    dirty: true,
    lastFrameIdx: -1,
    pointer: null,        // { downX, downY, dragging, camX, camY } while a button is down
    hoverPt: null,        // last pointer position over the canvas (screen px)
    onLoad: null,         // one-shot hook applying URL deep-link params
  };

  // ---- DOM helpers ------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const dom = {
    canvas: $("canvas"), stage: $("stage"), meta: $("meta"), outcome: $("outcome"),
    fileInput: $("file-input"), floorPicker: $("floor-picker"), teams: $("teams"), combat: $("combat"),
    scrub: $("scrub"), tickLabel: $("tick-label"), timeLabel: $("time-label"), speed: $("speed"),
    play: $("btn-play"), tooltip: $("tooltip"), hoverLine: $("hover-line"), empty: $("empty"),
    drop: $("drop-overlay"), error: $("error"), errorText: $("error-text"), help: $("help"),
    followChip: $("follow-chip"), followLabel: $("follow-label"),
  };
  const ctx = dom.canvas.getContext("2d");

  /** Build an element; children are nodes or strings (always inserted as text). */
  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (k === "style") Object.assign(node.style, v);
      else if (k === "class") node.className = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const c of children) if (c !== null && c !== undefined) node.append(c);
    return node;
  }
  const teamColor = (t) => TEAM_COLORS[t % TEAM_COLORS.length];
  const teamName = (t) => `Team ${t} · ${TEAM_NAMES[t % TEAM_NAMES.length]}`;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const fmt = (v, d = 1) => (Math.round(v * 10 ** d) / 10 ** d).toFixed(d);

  // ---- loading ----------------------------------------------------------
  function showError(err) {
    dom.errorText.textContent = err instanceof Error ? err.message : String(err);
    dom.error.hidden = false;
  }

  function loadModel(build, source) {
    let model;
    try {
      model = build();
    } catch (err) {
      showError(`Could not load ${source}:\n${err.message}`);
      return;
    }
    dom.error.hidden = true;
    setModel(model, source);
  }

  const loadText = (text, source) => loadModel(() => R.parseReplayText(text), source);

  function loadFile(file) {
    if (!file) return;
    file.text().then((text) => loadText(text, file.name), (err) => showError(`Could not read ${file.name}: ${err.message}`));
  }

  function loadUrl(url) {
    fetch(url)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
      .then((text) => loadText(text, url))
      .catch((err) => showError(`Could not fetch ${url}: ${err.message}\n(Opened via file://? Use "Open replay" or serve the folder: bash infra/viewer.sh)`));
  }

  function loadSample() {
    if (window.WARSIM_SAMPLE_REPLAY) loadModel(() => R.buildModel(window.WARSIM_SAMPLE_REPLAY), "sample replay");
    else loadUrl(SAMPLE_URL);
  }

  function setModel(model, source) {
    S.model = model;
    S.floorCaches = buildFloorCaches(model.map);
    S.pos = 0;
    S.playing = false;
    S.followId = null;
    S.hoverId = null;
    S.floor = model.frames[0].units.length ? model.frames[0].units[0].floor : 0;
    S.lastFrameIdx = -1;
    fitView();
    dom.empty.hidden = true;
    dom.scrub.max = String(model.frames.length - 1);
    dom.scrub.value = "0";
    buildMeta(model, source);
    buildFloorPicker(model);
    buildTeams(model);
    updatePlayButton();
    updateFollowChip();
    resize();
    S.dirty = true;
    if (S.onLoad) S.onLoad();
  }

  // ---- static panels ------------------------------------------------------
  function buildMeta(model, source) {
    const { width, height, floors } = model.map;
    const kind = typeof model.config.map_kind === "string" ? model.config.map_kind : "map";
    const last = model.frames[model.frames.length - 1].tick;
    dom.meta.replaceChildren(
      el("b", null, kind), ` ${width}×${height}×${floors}  ·  seed ${model.seed}  ·  `,
      `${model.frames.length} frames (ticks ${model.frames[0].tick}–${last})  ·  ${model.units.length} units  ·  ${source}`,
    );
    dom.meta.title = dom.meta.textContent;
    const o = model.outcome;
    dom.outcome.hidden = false;
    if (o.kind === "win") {
      dom.outcome.textContent = `${TEAM_NAMES[o.team % TEAM_NAMES.length]} wins · t${o.tick}`;
      dom.outcome.style.color = teamColor(o.team);
      dom.outcome.style.borderColor = teamColor(o.team);
    } else {
      dom.outcome.textContent = o.kind === "draw" ? `Draw · t${o.tick}` : `No winner · t${o.tick}`;
      dom.outcome.style.color = "";
      dom.outcome.style.borderColor = "";
    }
  }

  function buildFloorPicker(model) {
    dom.floorPicker.replaceChildren();
    if (model.map.floors < 2) return;
    for (let f = 0; f < model.map.floors; f++) {
      dom.floorPicker.append(el("button", {
        class: "floor-btn", type: "button", "data-floor": String(f), "aria-pressed": "false",
        title: `Floor ${f} ([ / ])`, onclick: () => setFloor(f),
      }, `F${f}`, el("span", { class: "pop" })));
    }
  }

  function buildTeams(model) {
    dom.teams.replaceChildren();
    for (const t of model.teams) {
      const rows = model.units.filter((u) => u.team === t).map((u) => el("li", {
        class: "unit-row", "data-unit": String(u.id), "aria-pressed": "false", tabindex: "0",
        title: `Follow unit ${u.id}`, onclick: () => toggleFollow(u.id),
        onkeydown: (e) => { if (e.key === "Enter") toggleFollow(u.id); },
      },
      el("span", { class: "uid", style: { color: teamColor(t) } }, `#${u.id}`),
      el("div", { class: "bar" }, el("span", { style: { background: teamColor(t), width: "100%" } })),
      el("span", { class: "ufloor" }, "F0"),
      el("span", { class: "uhp" }, "100%")));
      dom.teams.append(el("div", { class: "team", "data-team": String(t) },
        el("div", { class: "team-head" },
          el("span", { class: "team-name" }, el("i", { style: { background: teamColor(t) } }), teamName(t)),
          el("span", { class: "team-count" }, "0", el("small", null, "/0"))),
        el("div", { class: "bar team-bar" }, el("span", { style: { background: teamColor(t), width: "100%" } })),
        el("ul", { class: "units" }, ...rows)));
    }
  }

  // ---- per-frame panels ------------------------------------------------------
  function updatePanels(idx) {
    const m = S.model;
    const frame = m.frames[idx];
    const counts = R.aliveByTeam(frame, m.teams);
    for (const t of m.teams) {
      const box = dom.teams.querySelector(`.team[data-team="${t}"]`);
      if (!box) continue;
      const c = counts[t];
      const count = box.querySelector(".team-count");
      count.firstChild.textContent = String(c.alive);
      count.lastChild.textContent = `/${c.total}`;
      box.querySelector(".team-bar > span").style.width = `${c.total ? (100 * c.hp) / (c.total * m.hpScale) : 0}%`;
    }
    for (const u of frame.units) {
      const row = dom.teams.querySelector(`.unit-row[data-unit="${u.id}"]`);
      if (!row) continue;
      const pct = Math.round((100 * Math.max(0, u.hp)) / m.hpScale);
      row.classList.toggle("dead", !u.alive);
      row.querySelector(".bar > span").style.width = `${pct}%`;
      row.querySelector(".ufloor").textContent = `F${u.floor}`;
      row.querySelector(".uhp").textContent = u.alive ? `${pct}%` : "KIA";
    }
    updateCombat(idx);
    updateFloorPicker(frame);
    updateClock(idx);
  }

  function updateCombat(idx) {
    const m = S.model;
    const stats = m.stats[idx];
    const head = el("tr", null, el("th", null, "Team"), el("th", null, "Shots"), el("th", null, "Hits"), el("th", null, "Acc."));
    const rows = m.teams.map((t) => {
      const s = stats[t] || { shots: 0, hits: 0 };
      return el("tr", null,
        el("td", { style: { color: teamColor(t) } }, TEAM_NAMES[t % TEAM_NAMES.length]),
        el("td", null, String(s.shots)), el("td", null, String(s.hits)),
        el("td", null, s.shots ? `${Math.round((100 * s.hits) / s.shots)}%` : "–"));
    });
    dom.combat.replaceChildren(head, ...rows);
  }

  function updateFloorPicker(frame) {
    for (const btn of dom.floorPicker.children) {
      const f = Number(btn.dataset.floor);
      btn.setAttribute("aria-pressed", String(f === S.floor));
      const dots = frame.units.filter((u) => u.alive && u.floor === f)
        .map((u) => el("i", { style: { background: teamColor(u.team) } }));
      btn.querySelector(".pop").replaceChildren(...dots);
    }
  }

  function updateClock(idx) {
    const m = S.model;
    const f = m.frames[idx];
    const last = m.frames[m.frames.length - 1].tick;
    dom.tickLabel.textContent = `tick ${f.tick} / ${last}`;
    dom.timeLabel.textContent = `${fmt(f.tick * m.tickSeconds)} s · frame ${idx + 1}/${m.frames.length}`;
    if (document.activeElement !== dom.scrub) dom.scrub.value = String(idx);
  }

  // ---- floor caches (tiles pre-rendered once per load) ----------------------
  const isWallish = (t) => t === T.WALL || t === T.WINDOW || t === T.DOOR_CLOSED || t === T.DOOR_OPEN;

  /** true when a door/window runs left-right (walls to its left/right). */
  function runsHorizontally(map, x, y, f) {
    const h = isWallish(R.tileAt(map, x - 1, y, f)) + isWallish(R.tileAt(map, x + 1, y, f));
    const v = isWallish(R.tileAt(map, x, y - 1, f)) + isWallish(R.tileAt(map, x, y + 1, f));
    return h >= v;
  }

  function buildFloorCaches(map) {
    const px = clamp(Math.floor(CACHE_MAX_PX / Math.max(map.width, map.height)), 4, 32);
    const caches = [];
    for (let f = 0; f < map.floors; f++) {
      const c = document.createElement("canvas");
      c.width = map.width * px;
      c.height = map.height * px;
      const g = c.getContext("2d");
      for (let y = 0; y < map.height; y++) {
        for (let x = 0; x < map.width; x++) drawTile(g, map, x, y, f, px);
      }
      caches.push({ canvas: c, px });
    }
    return caches;
  }

  function drawFloorTile(g, x, y, p) {
    g.fillStyle = (x + y) % 2 ? COLORS.floorA : COLORS.floorB;
    g.fillRect(x * p, y * p, p, p);
  }

  function drawWallTile(g, map, x, y, f, p) {
    const X = x * p, Y = y * p;
    g.fillStyle = COLORS.wall;
    g.fillRect(X, Y, p, p);
    g.fillStyle = COLORS.wallTop;
    g.fillRect(X, Y, p, Math.max(1, p * 0.12));
    if (!isWallish(R.tileAt(map, x, y + 1, f))) {
      g.fillStyle = COLORS.wallShade;
      g.fillRect(X, Y + p * 0.78, p, p * 0.22);
    }
  }

  /** Door slab / window glass across the tile, oriented along its wall. */
  function drawAcross(g, x, y, p, horiz, thickness, color) {
    const X = x * p, Y = y * p, t = p * thickness;
    g.fillStyle = color;
    if (horiz) g.fillRect(X, Y + (p - t) / 2, p, t);
    else g.fillRect(X + (p - t) / 2, Y, t, p);
  }

  function drawStairs(g, x, y, p, up) {
    const X = x * p, Y = y * p;
    g.fillStyle = COLORS.stairs;
    g.fillRect(X, Y, p, p);
    g.strokeStyle = COLORS.stairsLine;
    g.lineWidth = Math.max(1, p / 16);
    for (let i = 1; i < 5; i++) {
      g.beginPath();
      g.moveTo(X + p * 0.1, Y + (p * i) / 5);
      g.lineTo(X + p * 0.9, Y + (p * i) / 5);
      g.stroke();
    }
    g.fillStyle = COLORS.arrow;
    g.beginPath();
    const cx = X + p / 2, s = p * 0.26;
    if (up) { g.moveTo(cx, Y + p / 2 - s); g.lineTo(cx + s, Y + p / 2 + s * 0.7); g.lineTo(cx - s, Y + p / 2 + s * 0.7); }
    else { g.moveTo(cx, Y + p / 2 + s); g.lineTo(cx + s, Y + p / 2 - s * 0.7); g.lineTo(cx - s, Y + p / 2 - s * 0.7); }
    g.closePath();
    g.fill();
  }

  function drawTile(g, map, x, y, f, p) {
    const t = R.tileAt(map, x, y, f);
    switch (t) {
      case T.VOID: return;
      case T.FLOOR: drawFloorTile(g, x, y, p); return;
      case T.WALL: drawWallTile(g, map, x, y, f, p); return;
      case T.DOOR_CLOSED: {
        drawFloorTile(g, x, y, p);
        const horiz = runsHorizontally(map, x, y, f);
        drawAcross(g, x, y, p, horiz, 0.42, COLORS.doorEdge);
        drawAcross(g, x, y, p, horiz, 0.3, COLORS.door);
        return;
      }
      case T.DOOR_OPEN: {
        drawFloorTile(g, x, y, p);
        const horiz = runsHorizontally(map, x, y, f);
        const X = x * p, Y = y * p, t2 = Math.max(2, p * 0.14);
        // Hinge on the first jamb; the leaf swings toward tiles drawn earlier
        // (up / left) so later tiles don't paint over it. Dotted swing arc.
        const L = p * 0.9;
        g.strokeStyle = COLORS.doorOpen;
        g.globalAlpha = 0.55;
        g.lineWidth = Math.max(1, p / 20);
        g.setLineDash([Math.max(1, p / 10), Math.max(1, p / 10)]);
        g.beginPath();
        if (horiz) g.arc(X, Y + p / 2, L, -Math.PI / 2, 0);
        else g.arc(X + p / 2, Y, L, Math.PI / 2, Math.PI);
        g.stroke();
        g.setLineDash([]);
        g.globalAlpha = 1;
        g.fillStyle = COLORS.doorOpen;
        if (horiz) g.fillRect(X, Y + p / 2 - L, t2, L);
        else g.fillRect(X + p / 2 - L, Y, L, t2);
        return;
      }
      case T.WINDOW: {
        drawWallTile(g, map, x, y, f, p);
        const horiz = runsHorizontally(map, x, y, f);
        drawAcross(g, x, y, p, horiz, 0.34, COLORS.glass);
        drawAcross(g, x, y, p, horiz, 0.08, "#d8e7ef");
        return;
      }
      case T.STAIRS_UP: drawStairs(g, x, y, p, true); return;
      case T.STAIRS_DOWN: drawStairs(g, x, y, p, false); return;
      default: // unknown tile id: make it obvious instead of silently hiding it
        g.fillStyle = "#b83e4b";
        g.fillRect(x * p, y * p, p, p);
    }
  }

  // ---- camera -----------------------------------------------------------
  const view = { w: 1, h: 1, dpr: 1 };

  function resize() {
    const rect = dom.stage.getBoundingClientRect();
    view.dpr = window.devicePixelRatio || 1;
    view.w = Math.max(1, rect.width);
    view.h = Math.max(1, rect.height);
    dom.canvas.width = Math.round(view.w * view.dpr);
    dom.canvas.height = Math.round(view.h * view.dpr);
    S.dirty = true;
  }

  function fitScale() {
    if (!S.model) return 1;
    return Math.min(view.w / S.model.map.width, view.h / S.model.map.height) * 0.94;
  }
  const scale = () => fitScale() * S.cam.zoom;

  function fitView() {
    if (!S.model) return;
    S.cam = { cx: S.model.map.width / 2, cy: S.model.map.height / 2, zoom: 1 };
    S.dirty = true;
  }

  function toScreen(wx, wy) {
    const s = scale();
    return [view.w / 2 + (wx - S.cam.cx) * s, view.h / 2 + (wy - S.cam.cy) * s];
  }
  function toWorld(sx, sy) {
    const s = scale();
    return [S.cam.cx + (sx - view.w / 2) / s, S.cam.cy + (sy - view.h / 2) / s];
  }

  function zoomAt(factor, sx, sy) {
    const [wx, wy] = toWorld(sx, sy);
    S.cam.zoom = clamp(S.cam.zoom * factor, ZOOM_MIN, ZOOM_MAX);
    if (S.followId === null) {
      const s = scale();
      S.cam.cx = wx - (sx - view.w / 2) / s;
      S.cam.cy = wy - (sy - view.h / 2) / s;
    }
    S.dirty = true;
  }

  // ---- drawing ------------------------------------------------------------
  function renderPos() {
    return S.opts.smooth ? S.pos : Math.round(S.pos);
  }

  function draw() {
    ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
    ctx.clearRect(0, 0, view.w, view.h);
    if (!S.model) return;
    const m = S.model;
    const pos = renderPos();
    const units = R.unitsAt(m, pos);
    followCamera(units);

    const s = scale();
    const [ox, oy] = toScreen(0, 0);
    const mw = m.map.width * s, mh = m.map.height * s;
    ctx.imageSmoothingEnabled = s < S.floorCaches[0].px;
    if (S.opts.ghost && S.floor > 0) {
      ctx.globalAlpha = 0.22;
      ctx.drawImage(S.floorCaches[S.floor - 1].canvas, ox, oy, mw, mh);
      ctx.globalAlpha = 1;
    }
    ctx.drawImage(S.floorCaches[S.floor].canvas, ox, oy, mw, mh);
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(ox - 0.5, oy - 0.5, mw + 1, mh + 1);

    if (S.opts.tracers) drawShots(R.recentShots(m, pos, TRACER_FRAMES), s);
    const r = Math.max(4, UNIT_RADIUS * s);
    if (S.opts.others) for (const u of units) if (u.floor !== S.floor) drawGhostUnit(u, r);
    for (const u of units) if (u.floor === S.floor && !u.alive) drawUnit(u, r);
    for (const u of units) if (u.floor === S.floor && u.alive) drawUnit(u, r);

    const idx = clamp(Math.floor(pos + 1e-6), 0, m.frames.length - 1);
    if (idx !== S.lastFrameIdx) {
      S.lastFrameIdx = idx;
      updatePanels(idx);
    }
  }

  function followCamera(units) {
    if (S.followId === null) return;
    const u = units.find((v) => v.id === S.followId);
    if (!u) return;
    if (u.floor !== S.floor) setFloor(u.floor);
    const k = S.playing ? FOLLOW_EASE : 1;
    S.cam.cx += (u.x - S.cam.cx) * k;
    S.cam.cy += (u.y - S.cam.cy) * k;
    if (Math.abs(u.x - S.cam.cx) + Math.abs(u.y - S.cam.cy) > 1e-3) S.dirty = true;
  }

  function drawShots(shots, s) {
    for (const shot of shots) {
      if (shot.floor !== S.floor) continue;
      const fade = 1 - shot.age / TRACER_FRAMES;
      if (fade <= 0) continue;
      const [x0, y0] = toScreen(shot.fromX, shot.fromY);
      const [x1, y1] = toScreen(shot.x, shot.y);
      ctx.lineCap = "round";
      ctx.globalAlpha = 0.35 * fade;
      ctx.strokeStyle = teamColor(shot.team);
      ctx.lineWidth = Math.max(3, s * 0.12);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.globalAlpha = fade;
      ctx.strokeStyle = "#fff4d6";
      ctx.lineWidth = Math.max(1, s * 0.04);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      const hit = shot.hit !== null;
      ctx.fillStyle = hit ? COLORS.hit : COLORS.miss;
      ctx.beginPath();
      ctx.arc(x1, y1, Math.max(2, s * (hit ? 0.18 : 0.08) * (1 + shot.age * 0.25)), 0, Math.PI * 2);
      ctx.fill();
      if (shot.age < 1.5) {
        ctx.fillStyle = COLORS.arrow;
        ctx.beginPath(); ctx.arc(x0, y0, Math.max(2, s * 0.1), 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  }

  function drawGhostUnit(u, r) {
    const [x, y] = toScreen(u.x, u.y);
    ctx.globalAlpha = u.alive ? 0.45 : 0.2;
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = u.alive ? teamColor(u.team) : COLORS.dead;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, r * 0.8, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    if (r >= 8) {
      ctx.fillStyle = teamColor(u.team);
      ctx.font = `600 ${Math.round(r * 0.7)}px ${getMono()}`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(u.floor > S.floor ? "▲" : "▼", x, y);
    }
    ctx.globalAlpha = 1;
  }

  function drawUnit(u, r) {
    const [x, y] = toScreen(u.x, u.y);
    const color = u.alive ? teamColor(u.team) : COLORS.dead;
    const highlighted = u.id === S.followId || u.id === S.hoverId;
    if (highlighted) {
      ctx.strokeStyle = u.id === S.followId ? COLORS.select : COLORS.hover;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, r + 4, 0, Math.PI * 2); ctx.stroke();
    }
    if (u.alive) {
      const dx = Math.cos(u.yaw), dy = Math.sin(u.yaw);
      ctx.strokeStyle = COLORS.outline;
      ctx.lineCap = "round";
      ctx.lineWidth = Math.max(3, r * 0.42);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx * r * 1.9, y + dy * r * 1.9); ctx.stroke();
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1.5, r * 0.22);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx * r * 1.85, y + dy * r * 1.85); ctx.stroke();
    }
    ctx.fillStyle = color;
    ctx.strokeStyle = COLORS.outline;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    if (!u.alive) {
      ctx.strokeStyle = "#20262c";
      ctx.lineWidth = Math.max(1.5, r * 0.2);
      const k = r * 0.45;
      ctx.beginPath(); ctx.moveTo(x - k, y - k); ctx.lineTo(x + k, y + k); ctx.moveTo(x + k, y - k); ctx.lineTo(x - k, y + k); ctx.stroke();
      return;
    }
    if (S.opts.labels && r >= 8) {
      ctx.fillStyle = COLORS.outline;
      ctx.font = `700 ${Math.round(r * 0.95)}px ${getMono()}`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(String(u.id), x, y + 0.5);
    }
    drawHpBar(x, y, r, u.hp / S.model.hpScale);
  }

  function drawHpBar(x, y, r, frac) {
    const w = Math.max(16, r * 2.2), h = Math.max(3, r * 0.22);
    const bx = x - w / 2, by = y - r - h - 5;
    ctx.fillStyle = "rgba(23,33,43,0.18)";
    ctx.fillRect(bx - 1, by - 1, w + 2, h + 2);
    const f = clamp(frac, 0, 1);
    ctx.fillStyle = f > 0.6 ? "#28734f" : f > 0.3 ? "#a77a2b" : "#b83e4b";
    ctx.fillRect(bx, by, w * f, h);
  }

  let monoFont = null;
  function getMono() {
    if (!monoFont) monoFont = getComputedStyle(document.documentElement).getPropertyValue("--mono").trim() || "monospace";
    return monoFont;
  }

  // ---- playback -----------------------------------------------------------
  function lastIndex() { return S.model ? S.model.frames.length - 1 : 0; }

  function seek(pos) {
    if (!S.model) return;
    S.pos = clamp(pos, 0, lastIndex());
    S.dirty = true;
  }
  function step(n) { setPlaying(false); seek(Math.round(S.pos) + n); }

  function setPlaying(on) {
    if (!S.model) return;
    if (on && S.pos >= lastIndex()) S.pos = 0;
    S.playing = on;
    updatePlayButton();
    S.dirty = true;
  }

  function updatePlayButton() {
    dom.play.textContent = S.playing ? "⏸" : "▶";
    dom.play.setAttribute("aria-label", S.playing ? "Pause" : "Play");
  }

  function setSpeed(v) {
    S.speed = v;
    dom.speed.value = String(v);
  }
  function nudgeSpeed(dir) {
    const i = SPEEDS.indexOf(S.speed);
    setSpeed(SPEEDS[clamp((i < 0 ? 2 : i) + dir, 0, SPEEDS.length - 1)]);
  }

  function setFloor(f) {
    if (!S.model) return;
    const nf = clamp(f, 0, S.model.map.floors - 1);
    if (nf === S.floor) return;
    S.floor = nf;
    S.lastFrameIdx = -1; // refresh floor picker
    S.dirty = true;
  }

  function toggleFollow(id) {
    if (!S.model) return;
    S.followId = S.followId === id ? null : id;
    if (S.followId !== null) S.cam.zoom = Math.max(S.cam.zoom, FOLLOW_ZOOM);
    updateFollowChip();
    S.dirty = true;
  }

  function updateFollowChip() {
    for (const row of dom.teams.querySelectorAll(".unit-row")) {
      row.setAttribute("aria-pressed", String(Number(row.dataset.unit) === S.followId));
    }
    dom.followChip.hidden = S.followId === null;
    if (S.followId !== null) dom.followLabel.textContent = `Following unit #${S.followId}`;
  }

  let lastNow = performance.now();
  function tick(now) {
    const dt = Math.min(0.1, (now - lastNow) / 1000);
    lastNow = now;
    if (S.playing && S.model) {
      S.pos += dt * S.model.framesPerSecond * S.speed;
      if (S.pos >= lastIndex()) { S.pos = lastIndex(); setPlaying(false); }
      S.dirty = true;
    }
    if (S.dirty) {
      S.dirty = false;
      draw();
      // Units move under a still pointer while playing: keep the tooltip live.
      if (S.playing && S.hoverPt && !S.pointer) updateHover(S.hoverPt[0], S.hoverPt[1]);
    }
    requestAnimationFrame(tick);
  }

  // ---- hover / tooltip ----------------------------------------------------
  function unitAtScreen(sx, sy) {
    if (!S.model) return null;
    const r = Math.max(6, UNIT_RADIUS * scale()) + 3;
    let best = null, bestD = Infinity;
    for (const u of R.unitsAt(S.model, renderPos())) {
      if (u.floor !== S.floor) continue;
      const [x, y] = toScreen(u.x, u.y);
      const d = Math.hypot(x - sx, y - sy);
      if (d <= r && d < bestD) { best = u; bestD = d; }
    }
    return best;
  }

  function updateHover(sx, sy) {
    if (!S.model) return;
    const u = unitAtScreen(sx, sy);
    const id = u ? u.id : null;
    if (id !== S.hoverId) { S.hoverId = id; S.dirty = true; }
    dom.canvas.classList.toggle("over-unit", Boolean(u));
    const [wx, wy] = toWorld(sx, sy);
    const tx = Math.floor(wx), ty = Math.floor(wy);
    const inside = tx >= 0 && ty >= 0 && tx < S.model.map.width && ty < S.model.map.height;
    dom.hoverLine.textContent = inside
      ? `x ${tx}  y ${ty}  F${S.floor}  ${R.TILE_NAMES[R.tileAt(S.model.map, tx, ty, S.floor)] || "Unknown tile"}`
      : "";
    if (!u) { dom.tooltip.hidden = true; return; }
    const row = (k, v) => el("div", null, el("span", { class: "tt-k" }, `${k} `), v);
    const deg = Math.round(((u.yaw * 180) / Math.PI + 360) % 360);
    dom.tooltip.replaceChildren(
      el("div", { class: "tt-head" }, el("span", { class: "tt-dot", style: { background: teamColor(u.team) } }),
        `Unit #${u.id} · ${TEAM_NAMES[u.team % TEAM_NAMES.length]}`),
      row("hp", u.alive ? `${Math.round((100 * u.hp) / S.model.hpScale)}%  (${fmt(u.hp, 2)})` : "dead"),
      row("pos", `${fmt(u.x, 2)}, ${fmt(u.y, 2)}  F${u.floor}`),
      row("yaw", `${deg}°`),
    );
    dom.tooltip.hidden = false;
    const tw = dom.tooltip.offsetWidth, th = dom.tooltip.offsetHeight;
    dom.tooltip.style.left = `${clamp(sx + 14, 4, view.w - tw - 4)}px`;
    dom.tooltip.style.top = `${clamp(sy + 14, 4, view.h - th - 4)}px`;
  }

  // ---- input wiring -------------------------------------------------------
  function localPoint(e) {
    const rect = dom.canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  function wirePointer() {
    const c = dom.canvas;
    c.addEventListener("pointerdown", (e) => {
      const [x, y] = localPoint(e);
      S.pointer = { downX: x, downY: y, camX: S.cam.cx, camY: S.cam.cy, dragging: false };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      const [x, y] = localPoint(e);
      const p = S.pointer;
      if (p && (p.dragging || Math.hypot(x - p.downX, y - p.downY) > DRAG_THRESHOLD_PX)) {
        if (!p.dragging) { p.dragging = true; c.classList.add("dragging"); dom.tooltip.hidden = true; }
        if (S.followId !== null) { S.followId = null; updateFollowChip(); }
        const s = scale();
        S.cam.cx = p.camX - (x - p.downX) / s;
        S.cam.cy = p.camY - (y - p.downY) / s;
        S.dirty = true;
        return;
      }
      S.hoverPt = [x, y];
      updateHover(x, y);
    });
    c.addEventListener("pointerup", (e) => {
      const p = S.pointer;
      S.pointer = null;
      c.classList.remove("dragging");
      if (p && !p.dragging) {
        const [x, y] = localPoint(e);
        const u = unitAtScreen(x, y);
        if (u) toggleFollow(u.id);
      }
    });
    c.addEventListener("pointerleave", () => {
      S.hoverPt = null;
      dom.tooltip.hidden = true;
      dom.hoverLine.textContent = "";
      if (S.hoverId !== null) { S.hoverId = null; S.dirty = true; }
    });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [x, y] = localPoint(e);
      zoomAt(Math.exp(-e.deltaY * 0.0015), x, y);
    }, { passive: false });
    c.addEventListener("dblclick", fitView);
  }

  function wireDrop() {
    let depth = 0;
    const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
    window.addEventListener("dragenter", (e) => { if (hasFiles(e)) { depth++; dom.drop.hidden = false; } });
    window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) dom.drop.hidden = true; });
    window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      depth = 0;
      dom.drop.hidden = true;
      loadFile(e.dataTransfer.files[0]);
    });
  }

  function onKey(e) {
    if (e.target instanceof HTMLElement && e.target.matches("input[type=file], select") ) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const handled = handleKey(e);
    if (handled) e.preventDefault();
  }

  function handleKey(e) {
    if (!dom.help.hidden) {
      if (e.key === "Escape" || e.key === "?") { dom.help.hidden = true; return true; }
      return false;
    }
    const big = e.shiftKey ? 10 : 1;
    switch (e.key) {
      case " ": setPlaying(!S.playing); return true;
      case "ArrowRight": step(big); return true;
      case "ArrowLeft": step(-big); return true;
      case "Home": setPlaying(false); seek(0); return true;
      case "End": setPlaying(false); seek(lastIndex()); return true;
      case "+": case "=": nudgeSpeed(1); return true;
      case "-": case "_": nudgeSpeed(-1); return true;
      case "]": case "PageUp": setFloor(S.floor + 1); return true;
      case "[": case "PageDown": setFloor(S.floor - 1); return true;
      case "0": fitView(); return true;
      case "f": case "F": if (S.hoverId !== null) toggleFollow(S.hoverId); return true;
      case "Escape": if (S.followId !== null) toggleFollow(S.followId); dom.error.hidden = true; return true;
      case "?": dom.help.hidden = false; return true;
      default:
        if (/^[1-9]$/.test(e.key) && S.model) {
          const u = S.model.units[Number(e.key) - 1];
          if (u) toggleFollow(u.id);
          return true;
        }
        return false;
    }
  }

  function wireControls() {
    dom.fileInput.addEventListener("change", () => { loadFile(dom.fileInput.files[0]); dom.fileInput.value = ""; });
    $("btn-sample").addEventListener("click", loadSample);
    $("btn-help").addEventListener("click", () => { dom.help.hidden = false; });
    $("btn-help-close").addEventListener("click", () => { dom.help.hidden = true; });
    dom.help.addEventListener("click", (e) => { if (e.target === dom.help) dom.help.hidden = true; });
    $("btn-error-close").addEventListener("click", () => { dom.error.hidden = true; });
    dom.play.addEventListener("click", () => setPlaying(!S.playing));
    $("btn-back").addEventListener("click", () => step(-1));
    $("btn-fwd").addEventListener("click", () => step(1));
    $("btn-start").addEventListener("click", () => { setPlaying(false); seek(0); });
    $("btn-end").addEventListener("click", () => { setPlaying(false); seek(lastIndex()); });
    dom.scrub.addEventListener("input", () => seek(Number(dom.scrub.value)));
    dom.speed.addEventListener("change", () => setSpeed(Number(dom.speed.value)));
    $("btn-zoom-in").addEventListener("click", () => zoomAt(1.25, view.w / 2, view.h / 2));
    $("btn-zoom-out").addEventListener("click", () => zoomAt(0.8, view.w / 2, view.h / 2));
    $("btn-fit").addEventListener("click", () => { if (S.followId !== null) toggleFollow(S.followId); fitView(); });
    $("btn-unfollow").addEventListener("click", () => { if (S.followId !== null) toggleFollow(S.followId); });
    for (const key of Object.keys(S.opts)) {
      const box = $(`opt-${key}`);
      box.addEventListener("change", () => { S.opts[key] = box.checked; S.dirty = true; });
      S.opts[key] = box.checked;
    }
    window.addEventListener("keydown", onKey);
    new ResizeObserver(resize).observe(dom.stage);
  }

  // ---- boot ---------------------------------------------------------------
  wireControls();
  wirePointer();
  wireDrop();
  resize();
  // Deep links: ?replay=<url>&t=<tick>&floor=<f>&follow=<unit id>&play=1
  const params = new URLSearchParams(window.location.search);
  S.onLoad = () => {
    S.onLoad = null;
    const num = (k) => (params.has(k) && Number.isFinite(Number(params.get(k))) ? Number(params.get(k)) : null);
    if (num("t") !== null) seek(R.frameIndexForTick(S.model, num("t")));
    if (num("floor") !== null) setFloor(num("floor"));
    if (num("follow") !== null) toggleFollow(num("follow"));
    if (params.get("play") === "1") setPlaying(true);
  };
  const replayParam = params.get("replay");
  if (replayParam) loadUrl(replayParam); else loadSample();
  requestAnimationFrame(tick);
})();
