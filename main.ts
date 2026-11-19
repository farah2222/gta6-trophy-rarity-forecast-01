// gta6-trophy-rarity-forecast-01 — "Dilution-Aware Logit Forecaster"
//
// Model (built from the brief + the practice data, no starter code):
//
// Exophase's page is a snapshot, not a live feed: it freezes for hours and
// then jumps. The answer at the target is whatever the page shows at the
// target, i.e. the rates as of Exophase's last refresh before it. Two
// forces move a trophy's rate between reads:
//
//   1. DILUTION. New players arrive faster than they earn trophies, so
//      rates fall even while the absolute earner count rises. We model
//      earners E_i(t) and players N(t) separately and forecast
//      rate_i(target) = E_i(target) / N(target). Both are fitted as
//      power laws in time (log-log linear) on the distinct reads, with
//      per-trophy earner slopes shrunk toward a tier prior
//      (platinum slowest, bronze fastest).
//   2. EARNING. Story/early trophies are earned fast as players progress;
//      grind/collectible trophies barely move. The tier prior captures
//      this when a trophy's own history is too thin to fit.
//
// Frozen pages: when every read in history is identical (launch night:
// 638 players, same rates for 3+ hours), a fitted trend is exactly zero,
// which is persistence — the wrong answer, because the page WILL refresh
// before the target and the refresh dilutes every rate. For a frozen
// history we therefore use a launch-night anchor prior: the page
// refreshed at 2026-11-19T04:30:00Z with 3,868 players tracked, so we
// extrapolate log-players through (first read, N0) -> (04:30Z, 3868) and
// let earners grow only a tier fraction (beta) as fast as players.
// That makes the median DROP below the last read, which is what the
// refresh actually does.
//
// Spread: honest logit-scale uncertainty. Components: per-trophy fit
// residual, a refresh-jump term (large when the page has been frozen —
// a refresh you have not seen yet is a jump, not a drift), and a
// horizon term growing with sqrt(days). The final case's 4.5-day horizon
// gets a much wider band than the 90-minute practice horizon.
//
// Safety: any unexpected input falls back to the last read's rates with
// a wide spread. Output is validated before writing; the program never
// crashes and never writes a malformed file. Fully deterministic, no
// randomness, no network, stdlib only.

import * as fs from "node:fs";
import * as path from "node:path";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

// Launch-night refresh anchor (from the competition brief): the frozen
// page first refreshed at 2026-11-19T04:30:00Z, players 638 -> 3,868.
const ANCHOR_MS = Date.parse("2026-11-19T04:30:00Z");
const ANCHOR_N = 3868;

// Tier earner-growth prior: earners grow as beta * (players growth).
// Platinum is slowest to be earned, bronze early trophies fastest.
const BETA: Record<string, number> = { platinum: 0.35, gold: 0.5, silver: 0.55, bronze: 0.6 };

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
function logitPct(p: number): number {
  const q = clamp(p / 100, 0.0001, 0.9999);
  return Math.log(q / (1 - q));
}
function logisticPct(l: number): number {
  const e = Math.exp(-clamp(l, -30, 30));
  return (1 / (1 + e)) * 100;
}
function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function linfit(xs: number[], ys: number[]): { a: number; b: number } {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  return { a: my - b * mx, b };
}
function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = xs.reduce((s, v) => s + v, 0) / xs.length;
  return Math.sqrt(xs.reduce((s, v) => s + (v - m) * (v - m), 0) / (xs.length - 1));
}
function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  t: number; // ms since epoch
  players: number;
  rates: (number | null)[];
}

function forecastCase(c: any): { id: number; q: number[] }[] {
  const trophies: Trophy[] = Array.isArray(c.trophies) ? c.trophies : [];
  const n = trophies.length;
  const atMs = Date.parse(c.at) || 0;
  const targetMs = Date.parse(c.target) || atMs;
  const horizonDays = Math.max(0, (targetMs - atMs) / 86400000);

  const rawHist: any[] = Array.isArray(c.history) ? c.history : [];
  const reads: Read[] = [];
  for (const h of rawHist) {
    const t = Date.parse(h.at);
    if (!Number.isFinite(t)) continue;
    const players = typeof h.players_tracked === "number" && h.players_tracked > 0 ? h.players_tracked : 0;
    const rates = Array.isArray(h.rates) ? h.rates : [];
    reads.push({ t, players, rates });
  }
  reads.sort((a, b) => a.t - b.t);

  // Last non-null rate per trophy (fallback anchor).
  const lastRate: number[] = new Array(n).fill(NaN);
  for (let i = reads.length - 1; i >= 0; i--) {
    for (let j = 0; j < n; j++) {
      const r = reads[i].rates[j];
      if (Number.isNaN(lastRate[j]) && typeof r === "number" && Number.isFinite(r)) lastRate[j] = r;
    }
  }
  const tierMedianRate: Record<string, number> = {};
  for (const tier of ["platinum", "gold", "silver", "bronze"]) {
    const vals: number[] = [];
    trophies.forEach((tr, j) => {
      if (tr.tier === tier && Number.isFinite(lastRate[j])) vals.push(lastRate[j]);
    });
    tierMedianRate[tier] = vals.length ? median(vals) : 1.0;
  }
  for (let j = 0; j < n; j++) {
    if (!Number.isFinite(lastRate[j])) lastRate[j] = tierMedianRate[trophies[j].tier] ?? 1.0;
  }

  // Time axis: log hours since the first read (page time, not wall clock
  // of the release — the first read is the earliest thing we know).
  const t0 = reads.length ? reads[0].t : atMs;
  const xOf = (t: number): number => Math.log(1 + Math.max(0, (t - t0) / 3600000));
  const xTarget = xOf(targetMs);
  const xLast = reads.length ? xOf(reads[reads.length - 1].t) : xOf(atMs);

  // ---- players N(t): log-log linear fit on distinct player counts ----
  const frozen =
    reads.length >= 2 &&
    reads.every((r) => r.players === reads[0].players) &&
    reads.every((r) => JSON.stringify(r.rates) === JSON.stringify(reads[0].rates));
  const nPts: { x: number; y: number }[] = [];
  {
    let prevN = -1;
    for (const r of reads) {
      if (r.players > 0 && r.players !== prevN) {
        nPts.push({ x: xOf(r.t), y: Math.log(r.players) });
        prevN = r.players;
      }
    }
  }
  const lastN = reads.length && reads[reads.length - 1].players > 0 ? reads[reads.length - 1].players : 0;
  let nSlope = 0;
  let nIntercept = lastN > 0 ? Math.log(lastN) : 0;
  let nPred = lastN;
  let nRatio: number | null = null; // frozen case: N(target)/N(last read)
  if (nPts.length >= 2) {
    const fit = linfit(nPts.map((p) => p.x), nPts.map((p) => p.y));
    nSlope = clamp(fit.b, 0, 5);
    nIntercept = fit.a;
    nPred = Math.exp(nIntercept + nSlope * xTarget);
    // Players tracked never shrink during launch week: floor at the last
    // observed count so a noisy log-log fit cannot predict a retreat.
    if (lastN > 0) nPred = clamp(nPred, lastN, lastN * 30);
  } else if (frozen && lastN > 0 && t0 < ANCHOR_MS) {
    // Frozen page: extrapolate log-players in LINEAR time through the
    // launch-night refresh anchor (first read -> 04:30Z, 3,868 players).
    // (Log-time would mis-anchor: the page's frozen N is the value at
    // the first read, not at the last read.) nRatio is the player-growth
    // factor from the last read to the target; earners grow as a tier
    // fraction of it (see the per-trophy fit below).
    const hoursToAnchor = (ANCHOR_MS - t0) / 3600000;
    const logSlopePerHour = hoursToAnchor > 1e-9 ? (Math.log(ANCHOR_N) - Math.log(lastN)) / hoursToAnchor : 0;
    let lnNpred: number;
    if (targetMs <= ANCHOR_MS) {
      lnNpred = Math.log(lastN) + logSlopePerHour * ((targetMs - t0) / 3600000);
    } else {
      lnNpred = Math.log(ANCHOR_N) + logSlopePerHour * ((targetMs - ANCHOR_MS) / 3600000);
    }
    nPred = Math.exp(lnNpred);
    nPred = clamp(nPred, lastN, lastN * 30);
    nSlope = logSlopePerHour; // per-hour; only used as a signpost below
    nRatio = nPred / lastN;
  } else {
    nPred = lastN > 0 ? lastN : 1;
  }
  if (!Number.isFinite(nPred) || nPred <= 0) nPred = lastN > 0 ? lastN : 1;

  // ---- per-trophy earners E_i(t) = rate * N ----
  const out: { id: number; q: number[] }[] = [];
  const residByTier: Record<string, number[]> = { platinum: [], gold: [], silver: [], bronze: [] };
  const medLogitByTier: Record<string, number> = {};
  // First pass: fit each trophy, collect tier residuals.
  const fits: { logEpred: number; resid: number; nDistinct: number }[] = [];
  for (let j = 0; j < n; j++) {
    const tier = trophies[j].tier || "bronze";
    const beta = BETA[tier] ?? 0.6;
    const priorSlope = beta * nSlope;
    const pts: { x: number; y: number }[] = [];
    let prevE = -1;
    for (const r of reads) {
      const rate = r.rates[j];
      if (typeof rate !== "number" || !Number.isFinite(rate) || r.players <= 0) continue;
      const e = (Math.max(rate, 0.005) / 100) * r.players;
      if (e <= 0) continue;
      const y = Math.log(e);
      if (Math.abs(y - prevE) > 1e-9) {
        pts.push({ x: xOf(r.t), y });
        prevE = y;
      }
    }
    let m: number;
    let a: number;
    let resid = 0.15;
    if (pts.length >= 2) {
      const fit = linfit(pts.map((p) => p.x), pts.map((p) => p.y));
      const mRaw = clamp(fit.b, 0, 5);
      // Shrink toward the tier prior; more distinct points = more trust.
      const w = pts.length / (pts.length + 3);
      m = w * mRaw + (1 - w) * priorSlope;
      a = pts.reduce((s, p) => s + (p.y - m * p.x), 0) / pts.length;
      const res = pts.map((p) => p.y - (a + m * p.x));
      resid = pts.length >= 3 ? Math.max(0.05, std(res)) : 0.15;
    } else if (nRatio !== null) {
      // Frozen page: earners grow as a tier fraction (beta) of player
      // growth, so the rate scales as ratio^(beta - 1) — a dilution
      // drop, steepest for the platinum, mildest for early bronzes.
      const eLast = (Math.max(lastRate[j], 0.005) / 100) * (lastN > 0 ? lastN : 1);
      const logEpred = Math.log(eLast) + beta * Math.log(nRatio);
      // Stash via the same variables the fitted path uses.
      m = 0;
      a = 0;
      fits.push({ logEpred, resid: 0.15, nDistinct: pts.length });
      if (residByTier[tier]) residByTier[tier].push(0.15);
      continue;
    } else {
      m = priorSlope;
      const eLast = (Math.max(lastRate[j], 0.005) / 100) * (lastN > 0 ? lastN : 1);
      a = Math.log(eLast) - m * xLast;
    }
    const logEpred = a + m * xTarget;
    fits.push({ logEpred, resid, nDistinct: pts.length });
    if (residByTier[tier]) residByTier[tier].push(resid);
  }
  for (const tier of Object.keys(residByTier)) {
    medLogitByTier[tier] = residByTier[tier].length ? median(residByTier[tier]) : 0.15;
  }

  // ---- spread: residual + refresh-jump + horizon ----
  // Time since the page last visibly changed (a frozen page is a jump
  // waiting to happen, not a drift).
  let lastChangeMs = t0;
  for (let i = 1; i < reads.length; i++) {
    if (JSON.stringify(reads[i].rates) !== JSON.stringify(reads[i - 1].rates) || reads[i].players !== reads[i - 1].players) {
      lastChangeMs = reads[i].t;
    }
  }
  const hoursSinceChange = Math.max(0, (atMs - lastChangeMs) / 3600000);
  const jump = frozen ? 0.85 : hoursSinceChange > 6 ? 0.4 : 0.15;
  const horizonSigma = 0.3 * Math.sqrt(Math.min(horizonDays, 7));

  for (let j = 0; j < n; j++) {
    const tier = trophies[j].tier || "bronze";
    const rLast = lastRate[j];
    // The 100% trophy: everyone tracked has it and new players earn it
    // on starting; keep an extremely tight band near 100.
    if (rLast >= 99.99) {
      out.push({ id: trophies[j].id, q: [99.95, 99.98, 99.99, 99.99, 99.99] });
      continue;
    }
    const ratePred = clamp((Math.exp(fits[j].logEpred) / nPred) * 100, 0.01, 99.99);
    const resid = Number.isFinite(fits[j].resid) ? fits[j].resid : medLogitByTier[tier] ?? 0.15;
    const sigma = clamp(Math.sqrt(resid * resid + jump * jump + horizonSigma * horizonSigma), 0.4, 1.8);
    const mid = logitPct(ratePred);
    const qs = Z.map((z) => round4(clamp(logisticPct(mid + z * sigma), 0.01, 99.99)));
    // Enforce non-decreasing (rounding can tie, never invert, but be safe).
    for (let k = 1; k < qs.length; k++) if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
    out.push({ id: trophies[j].id, q: qs });
  }
  return out;
}

function fallbackRows(c: any): { id: number; q: number[] }[] {
  const trophies: Trophy[] = Array.isArray(c.trophies) ? c.trophies : [];
  const rawHist: any[] = Array.isArray(c.history) ? c.history : [];
  const rows: { id: number; q: number[] }[] = [];
  for (let j = 0; j < trophies.length; j++) {
    let r = 1.0;
    for (let i = rawHist.length - 1; i >= 0; i--) {
      const v = rawHist[i] && Array.isArray(rawHist[i].rates) ? rawHist[i].rates[j] : undefined;
      if (typeof v === "number" && Number.isFinite(v)) {
        r = v;
        break;
      }
    }
    if (r >= 99.99) {
      rows.push({ id: trophies[j].id, q: [99.95, 99.98, 99.99, 99.99, 99.99] });
      continue;
    }
    const mid = logitPct(clamp(r, 0.01, 99.99));
    const qs = Z.map((z) => round4(clamp(logisticPct(mid + z * 1.0), 0.01, 99.99)));
    for (let k = 1; k < qs.length; k++) if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
    rows.push({ id: trophies[j].id, q: qs });
  }
  return rows;
}

function validRows(rows: any, n: number): boolean {
  if (!Array.isArray(rows) || rows.length !== n) return false;
  const seen = new Set<number>();
  for (const row of rows) {
    if (!row || typeof row.id !== "number" || seen.has(row.id)) return false;
    seen.add(row.id);
    if (!Array.isArray(row.q) || row.q.length !== 5) return false;
    for (let k = 0; k < 5; k++) {
      const v = row.q[k];
      if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 100) return false;
      if (k > 0 && v < row.q[k - 1]) return false;
    }
    if (Object.keys(row).length !== 2) return false;
  }
  return true;
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  fs.mkdirSync(outDir, { recursive: true });
  let files: string[] = [];
  try {
    files = fs.readdirSync(inDir).filter((f) => f.endsWith(".in")).sort();
  } catch {
    files = [];
  }
  for (const f of files) {
    const id = f.slice(0, -3);
    const outPath = path.join(outDir, id + ".out");
    let rows: { id: number; q: number[] }[] | null = null;
    let n = 0;
    try {
      const c = JSON.parse(fs.readFileSync(path.join(inDir, f), "utf8"));
      n = Array.isArray(c.trophies) ? c.trophies.length : 0;
      rows = forecastCase(c);
      if (!validRows(rows, n)) rows = null;
    } catch {
      rows = null;
    }
    if (rows === null) {
      // Last-resort fallback: re-read trophies for ids, else empty list.
      try {
        const c = JSON.parse(fs.readFileSync(path.join(inDir, f), "utf8"));
        n = Array.isArray(c.trophies) ? c.trophies.length : 0;
        rows = fallbackRows(c);
        if (!validRows(rows, n)) rows = [];
      } catch {
        rows = [];
      }
    }
    const text = JSON.stringify({ trophies: rows });
    if (text.length <= 65536) {
      fs.writeFileSync(outPath, text);
    } else {
      // Extremely defensive: shrink decimals to fit (never happens at
      // this trophy count, but the cap is a hard rule).
      const slim = JSON.stringify({
        trophies: rows.map((r) => ({ id: r.id, q: r.q.map((v) => Math.round(v * 100) / 100) })),
      });
      fs.writeFileSync(outPath, slim.length <= 65536 ? slim : JSON.stringify({ trophies: [] }));
    }
  }
}

main();
