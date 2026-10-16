// Pure replay logic for the viewer: parsing, validation, interpolation, stats.
// No DOM access. Loaded as a classic <script> (works from file://) and as a
// CommonJS module by the Node tests in viewer/tests/.
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.WarsimReplay = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const SUPPORTED_VERSION = 1;
  const TILE = Object.freeze({
    VOID: 0, FLOOR: 1, WALL: 2, DOOR_CLOSED: 3, DOOR_OPEN: 4,
    WINDOW: 5, STAIRS_UP: 6, STAIRS_DOWN: 7,
  });
  const TILE_NAMES = ["Void", "Floor", "Wall", "Door (closed)", "Door (open)",
    "Window", "Stairs up", "Stairs down"];
  const DEFAULT_TICK_SECONDS = 0.1;
  const MAX_ERRORS = 8;

  const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const isInt = (v) => Number.isInteger(v) && v >= 0;

  /** Collects human-readable problems; throws once with all of them. */
  function makeErrors() {
    const list = [];
    return {
      add(msg) { if (list.length < MAX_ERRORS) list.push(msg); },
      get count() { return list.length; },
      throwIfAny() {
        if (list.length) throw new Error("Invalid replay:\n- " + list.join("\n- "));
      },
    };
  }

  function validateMap(map, errs) {
    if (!isObj(map)) { errs.add("`map` must be an object (see docs/INTERFACE.md Map JSON)"); return null; }
    const { width, height, floors, tiles } = map;
    for (const [k, v] of [["width", width], ["height", height], ["floors", floors]]) {
      if (!isInt(v) || v < 1) errs.add(`map.${k} must be a positive integer (got ${JSON.stringify(v)})`);
    }
    if (!Array.isArray(tiles)) { errs.add("map.tiles must be an array"); return null; }
    if (errs.count) return null;
    const expected = width * height * floors;
    if (tiles.length !== expected) {
      errs.add(`map.tiles has ${tiles.length} entries, expected width*height*floors = ${expected}`);
      return null;
    }
    const out = new Uint8Array(expected);
    for (let i = 0; i < expected; i++) {
      const t = tiles[i];
      if (!isInt(t) || t > 255) { errs.add(`map.tiles[${i}] is not a tile id (got ${JSON.stringify(t)})`); return null; }
      out[i] = t;
    }
    return { width, height, floors, tiles: out };
  }

  function validateUnit(u, where, errs) {
    if (!isObj(u)) { errs.add(`${where} must be an object`); return; }
    for (const k of ["id", "team", "floor"]) if (!isInt(u[k])) errs.add(`${where}.${k} must be a non-negative integer`);
    for (const k of ["x", "y", "yaw", "hp"]) if (!isNum(u[k])) errs.add(`${where}.${k} must be a number`);
    if (typeof u.alive !== "boolean") errs.add(`${where}.alive must be true/false`);
  }

  function validateShot(s, where, errs) {
    if (!isObj(s)) { errs.add(`${where} must be an object`); return; }
    if (!isInt(s.shooter)) errs.add(`${where}.shooter must be a unit id`);
    if (!isNum(s.x) || !isNum(s.y)) errs.add(`${where}.x/y must be numbers`);
    if (!isInt(s.floor)) errs.add(`${where}.floor must be a non-negative integer`);
    if (s.hit !== undefined && s.hit !== null && !isInt(s.hit)) errs.add(`${where}.hit must be a unit id or null`);
  }

  function validateFrames(frames, errs) {
    if (!Array.isArray(frames) || frames.length === 0) { errs.add("`frames` must be a non-empty array"); return; }
    let prevTick = -Infinity;
    frames.forEach((f, i) => {
      if (errs.count >= MAX_ERRORS) return;
      if (!isObj(f)) { errs.add(`frames[${i}] must be an object`); return; }
      if (!isInt(f.tick)) errs.add(`frames[${i}].tick must be a non-negative integer`);
      else if (f.tick <= prevTick) errs.add(`frames[${i}].tick (${f.tick}) must be greater than the previous tick`);
      else prevTick = f.tick;
      if (!Array.isArray(f.units)) errs.add(`frames[${i}].units must be an array`);
      else f.units.forEach((u, j) => validateUnit(u, `frames[${i}].units[${j}]`, errs));
      if (f.shots !== undefined && !Array.isArray(f.shots)) errs.add(`frames[${i}].shots must be an array`);
      else (f.shots || []).forEach((s, j) => validateShot(s, `frames[${i}].shots[${j}]`, errs));
    });
  }

  function median(values) {
    if (!values.length) return 1;
    const s = [...values].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  }

  /** Validate a parsed replay object and build the viewer model. Throws Error. */
  function buildModel(raw) {
    const errs = makeErrors();
    if (!isObj(raw)) { errs.add("top level must be a JSON object"); errs.throwIfAny(); }
    if (raw.version !== SUPPORTED_VERSION) {
      errs.add(`unsupported version ${JSON.stringify(raw.version)} (viewer supports ${SUPPORTED_VERSION})`);
    }
    const map = validateMap(raw.map, errs);
    validateFrames(raw.frames, errs);
    errs.throwIfAny();

    const frames = raw.frames.map((f) => ({
      tick: f.tick,
      units: f.units,
      shots: (f.shots || []).map((s) => ({ ...s, hit: s.hit === undefined ? null : s.hit })),
    }));
    const config = isObj(raw.config) ? raw.config : {};
    const tickSeconds = isNum(config.tick_seconds) && config.tick_seconds > 0
      ? config.tick_seconds : DEFAULT_TICK_SECONDS;
    const tickStride = Math.max(1, median(frames.slice(1).map((f, i) => f.tick - frames[i].tick)));

    const units = new Map(); // id -> { id, team }
    let maxHp = 0;
    for (const f of frames) {
      for (const u of f.units) {
        if (!units.has(u.id)) units.set(u.id, { id: u.id, team: u.team });
        maxHp = Math.max(maxHp, u.hp);
      }
    }
    const unitList = [...units.values()].sort((a, b) => a.id - b.id);
    const teams = [...new Set(unitList.map((u) => u.team))].sort((a, b) => a - b);

    return {
      version: raw.version,
      config,
      seed: raw.seed,
      map,
      frames,
      units: unitList,
      teams,
      // Full-health reference for HP bars: config.max_hp if given, else the
      // largest hp seen (engine uses 100), else 1 (normalised hp).
      hpScale: isNum(config.max_hp) && config.max_hp > 0 ? config.max_hp : maxHp > 1 ? maxHp : 1,
      tickSeconds,
      tickStride,
      framesPerSecond: 1 / (tickSeconds * tickStride),
      stats: cumulativeStats(frames, teams),
      outcome: outcome(frames, teams),
    };
  }

  function parseReplayText(text) {
    let raw;
    try {
      raw = JSON.parse(text);
    } catch (e) {
      throw new Error("Not valid JSON: " + e.message);
    }
    return buildModel(raw);
  }

  function tileAt(map, x, y, f) {
    if (x < 0 || y < 0 || f < 0 || x >= map.width || y >= map.height || f >= map.floors) return TILE.VOID;
    return map.tiles[(f * map.height + y) * map.width + x];
  }

  function aliveByTeam(frame, teams) {
    const out = {};
    for (const t of teams) out[t] = { alive: 0, total: 0, hp: 0 };
    for (const u of frame.units) {
      const s = out[u.team] || (out[u.team] = { alive: 0, total: 0, hp: 0 });
      s.total += 1;
      if (u.alive) { s.alive += 1; s.hp += Math.max(0, u.hp); }
    }
    return out;
  }

  /** Who won, judged from the final frame (team wiped out => other team wins). */
  function outcome(frames, teams) {
    const last = frames[frames.length - 1];
    const counts = aliveByTeam(last, teams);
    const standing = teams.filter((t) => counts[t].alive > 0);
    if (standing.length === 1) return { kind: "win", team: standing[0], tick: last.tick };
    if (standing.length === 0) return { kind: "draw", team: null, tick: last.tick };
    return { kind: "open", team: null, tick: last.tick };
  }

  /** stats[i][team] = { shots, hits } summed over frames 0..=i. */
  function cumulativeStats(frames, teams) {
    const running = {};
    for (const t of teams) running[t] = { shots: 0, hits: 0 };
    return frames.map((f) => {
      const teamOf = new Map(f.units.map((u) => [u.id, u.team]));
      for (const s of f.shots) {
        const t = teamOf.get(s.shooter);
        if (t === undefined) continue;
        const r = running[t] || (running[t] = { shots: 0, hits: 0 });
        r.shots += 1;
        if (s.hit !== null) r.hits += 1;
      }
      const snap = {};
      for (const t of Object.keys(running)) snap[t] = { ...running[t] };
      return snap;
    });
  }

  /** Interpolate between angles along the shortest arc. */
  function lerpAngle(a, b, t) {
    let d = (b - a) % (2 * Math.PI);
    if (d > Math.PI) d -= 2 * Math.PI;
    if (d < -Math.PI) d += 2 * Math.PI;
    return a + d * t;
  }

  /**
   * Unit states at fractional frame position `pos` (0..frames.length-1).
   * Positions/yaw/hp interpolate; floor/alive snap (no blending across floors).
   */
  function unitsAt(model, pos) {
    const n = model.frames.length;
    const p = Math.min(Math.max(pos, 0), n - 1);
    const i = Math.floor(p);
    const t = p - i;
    const a = model.frames[i];
    if (t === 0 || i + 1 >= n) return a.units.map((u) => ({ ...u }));
    const nextById = new Map(model.frames[i + 1].units.map((u) => [u.id, u]));
    return a.units.map((u) => {
      const b = nextById.get(u.id);
      if (!b || b.floor !== u.floor || !u.alive) return { ...u };
      return {
        ...u,
        x: u.x + (b.x - u.x) * t,
        y: u.y + (b.y - u.y) * t,
        yaw: lerpAngle(u.yaw, b.yaw, t),
        hp: u.hp + (b.hp - u.hp) * t,
      };
    });
  }

  /**
   * Shots from the last `window` frames up to frame index `idx`, each with its
   * origin (shooter position in the frame it was fired) and `age` in frames.
   */
  function recentShots(model, idx, window) {
    const out = [];
    const last = Math.min(Math.floor(idx), model.frames.length - 1);
    for (let j = Math.max(0, last - window + 1); j <= last; j++) {
      const f = model.frames[j];
      if (!f.shots.length) continue;
      const byId = new Map(f.units.map((u) => [u.id, u]));
      for (const s of f.shots) {
        const from = byId.get(s.shooter);
        if (!from) continue;
        out.push({ ...s, fromX: from.x, fromY: from.y, team: from.team, age: idx - j });
      }
    }
    return out;
  }

  /** Index of the first frame whose tick is >= `tick` (clamped). */
  function frameIndexForTick(model, tick) {
    const f = model.frames;
    let lo = 0;
    let hi = f.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (f[mid].tick < tick) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  return {
    SUPPORTED_VERSION, TILE, TILE_NAMES,
    buildModel, parseReplayText, tileAt, aliveByTeam, outcome,
    lerpAngle, unitsAt, recentShots, frameIndexForTick,
  };
});
