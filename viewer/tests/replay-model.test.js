// Dependency-free tests for viewer/replay-model.js.
// Run: node --test viewer/tests/
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const R = require("../replay-model.js");

const VIEWER = path.join(__dirname, "..");
const sampleText = fs.readFileSync(path.join(VIEWER, "sample_replay.json"), "utf8");

function tinyReplay(overrides = {}) {
  return {
    version: 1,
    config: { tick_seconds: 0.1 },
    seed: 1,
    map: { width: 2, height: 1, floors: 2, tiles: [1, 2, 0, 7] },
    frames: [
      { tick: 0, units: [
        { id: 0, team: 0, x: 0.5, y: 0.5, floor: 0, yaw: 3.0, hp: 1, alive: true },
        { id: 1, team: 1, x: 1.5, y: 0.5, floor: 0, yaw: 0, hp: 1, alive: true },
      ], shots: [] },
      { tick: 1, units: [
        { id: 0, team: 0, x: 1.5, y: 0.5, floor: 0, yaw: -3.0, hp: 0.5, alive: true },
        { id: 1, team: 1, x: 1.5, y: 0.5, floor: 1, yaw: 0, hp: 0, alive: false },
      ], shots: [{ shooter: 0, x: 1.5, y: 0.5, floor: 0, hit: 1 }, { shooter: 1, x: 0, y: 0, floor: 0 }] },
    ],
    ...overrides,
  };
}

test("bundled sample parses and has the expected shape", () => {
  const m = R.parseReplayText(sampleText);
  assert.equal(m.map.floors, 2);
  assert.equal(m.units.length, 4);
  assert.deepEqual(m.teams, [0, 1]);
  assert.equal(m.outcome.kind, "win");
  assert.equal(m.outcome.team, 0);
  assert.equal(m.framesPerSecond, 10);
  const last = m.stats[m.stats.length - 1];
  assert.ok(last[0].shots > 0 && last[1].shots > 0);
});

test("sample_replay.js embeds exactly sample_replay.json", () => {
  const ctx = { window: {} };
  vm.runInNewContext(fs.readFileSync(path.join(VIEWER, "sample_replay.js"), "utf8"), ctx);
  // Compare as JSON text: objects from another vm realm have other prototypes.
  assert.equal(JSON.stringify(ctx.window.WARSIM_SAMPLE_REPLAY), JSON.stringify(JSON.parse(sampleText)));
});

test("tileAt follows the (f*H + y)*W + x layout and treats out of bounds as void", () => {
  const m = R.buildModel(tinyReplay());
  assert.equal(R.tileAt(m.map, 1, 0, 0), R.TILE.WALL);
  assert.equal(R.tileAt(m.map, 1, 0, 1), R.TILE.STAIRS_DOWN);
  assert.equal(R.tileAt(m.map, 5, 0, 0), R.TILE.VOID);
  assert.equal(R.tileAt(m.map, 0, 0, -1), R.TILE.VOID);
});

test("rejects bad input with readable errors", () => {
  assert.throws(() => R.parseReplayText("{"), /Not valid JSON/);
  assert.throws(() => R.buildModel(tinyReplay({ version: 2 })), /unsupported version 2/);
  const badMap = tinyReplay({ map: { width: 2, height: 1, floors: 2, tiles: [1] } });
  assert.throws(() => R.buildModel(badMap), /expected width\*height\*floors = 4/);
  const backwards = tinyReplay();
  backwards.frames[1].tick = 0;
  assert.throws(() => R.buildModel(backwards), /must be greater than the previous tick/);
  const badUnit = tinyReplay();
  delete badUnit.frames[0].units[0].alive;
  assert.throws(() => R.buildModel(badUnit), /units\[0\]\.alive/);
  assert.throws(() => R.buildModel(tinyReplay({ frames: [] })), /non-empty array/);
});

test("missing shots and hit fields default", () => {
  const raw = tinyReplay();
  delete raw.frames[0].shots;
  const m = R.buildModel(raw);
  assert.deepEqual(m.frames[0].shots, []);
  assert.equal(m.frames[1].shots[1].hit, null);
});

test("unitsAt interpolates position/hp and wraps yaw the short way", () => {
  const m = R.buildModel(tinyReplay());
  const [u0, u1] = R.unitsAt(m, 0.5);
  assert.equal(u0.x, 1.0);
  assert.equal(u0.hp, 0.75);
  // 3.0 -> -3.0 crosses +-pi: midpoint is near pi, not 0.
  assert.ok(Math.abs(Math.abs(u0.yaw) - Math.PI) < 0.01, `yaw ${u0.yaw}`);
  // Floor change snaps instead of blending.
  assert.equal(u1.floor, 0);
  assert.equal(u1.x, 1.5);
  assert.deepEqual(R.unitsAt(m, 99)[1].floor, 1);
});

test("recentShots carries origin, team and age", () => {
  const m = R.buildModel(tinyReplay());
  const shots = R.recentShots(m, 1.25, 4);
  assert.equal(shots.length, 2);
  assert.deepEqual([shots[0].fromX, shots[0].team, shots[0].age], [1.5, 0, 0.25]);
  assert.equal(R.recentShots(m, 0, 4).length, 0);
});

test("stats, outcome and tick lookup", () => {
  const m = R.buildModel(tinyReplay());
  assert.deepEqual(m.stats[1][0], { shots: 1, hits: 1 });
  assert.deepEqual(m.stats[1][1], { shots: 1, hits: 0 });
  assert.deepEqual(m.outcome, { kind: "win", team: 0, tick: 1 });
  assert.equal(R.frameIndexForTick(m, 1), 1);
  assert.equal(R.frameIndexForTick(m, 100), 1);
  const counts = R.aliveByTeam(m.frames[1], m.teams);
  assert.deepEqual(counts[1], { alive: 0, total: 1, hp: 0 });
});

test("hp above 1 is rescaled for bars", () => {
  const raw = tinyReplay();
  raw.frames[0].units[0].hp = 100;
  assert.equal(R.buildModel(raw).hpScale, 100);
});
