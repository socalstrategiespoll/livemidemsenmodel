// ---------------------------------------------------------------------------
// Point this at your Render service URL. No trailing slash.
// Example: https://mi-senate-model.onrender.com
// ---------------------------------------------------------------------------
const API_BASE = "https://mi-senate-model.onrender.com";

const REFRESH_MS = 15000;
const STALE_AFTER_MS = 180000; // flag the feed if nothing new lands in 3 minutes

const EL = "El-Sayed";
const ST = "Stevens";

const num = new Intl.NumberFormat("en-US");
const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function signed(v, d = 1) {
  return (v >= 0 ? "+" : "\u2212") + Math.abs(v).toFixed(d);
}

// ---------------------------------------------------------------------------
// Distribution. Drawn from the simulated percentiles the model actually
// produced, not a normal curve fitted to the median, because the posterior is
// genuinely skewed while large counties are partly counted.
// ---------------------------------------------------------------------------

const W = 720;
const H = 200;
const PAD = 8;

function density(percentiles, bins = 48) {
  const lo = percentiles[0];
  const hi = percentiles[percentiles.length - 1];
  const span = hi - lo || 1;
  const counts = new Array(bins).fill(0);

  percentiles.forEach((v) => {
    const i = Math.min(bins - 1, Math.floor(((v - lo) / span) * bins));
    counts[i] += 1;
  });

  // Light smoothing so 60 sample points read as a curve rather than a comb.
  const smooth = counts.map((_, i) => {
    const a = counts[i - 1] ?? counts[i];
    const b = counts[i];
    const c = counts[i + 1] ?? counts[i];
    return (a + 2 * b + c) / 4;
  });

  const peak = Math.max(...smooth, 1);
  return { lo, hi, span, values: smooth.map((v) => v / peak) };
}

function curvePath(d, close) {
  const step = W / (d.values.length - 1);
  const y = (v) => PAD + (1 - v) * (H - 2 * PAD - 24);
  let path = `M 0 ${y(d.values[0]).toFixed(1)}`;

  for (let i = 1; i < d.values.length; i++) {
    const x0 = (i - 1) * step;
    const x1 = i * step;
    const mid = (x0 + x1) / 2;
    path += ` C ${mid.toFixed(1)} ${y(d.values[i - 1]).toFixed(1)},` +
            ` ${mid.toFixed(1)} ${y(d.values[i]).toFixed(1)},` +
            ` ${x1.toFixed(1)} ${y(d.values[i]).toFixed(1)}`;
  }

  if (close) path += ` L ${W} ${H - 24} L 0 ${H - 24} Z`;
  return path;
}

function drawDistribution(p) {
  const pct = p.margin_percentiles;
  if (!pct || pct.length < 8) return;

  const d = density(pct);
  const xOf = (v) => ((v - d.lo) / d.span) * W;
  const clamp = (x) => Math.max(0, Math.min(W, x));

  $("dist-fill").setAttribute("d", curvePath(d, true));
  $("dist-line").setAttribute("d", curvePath(d, false));

  const band = $("dist-band");
  band.innerHTML = "";
  const rect = (x1, x2, cls) => {
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", clamp(x1));
    r.setAttribute("y", 4);
    r.setAttribute("width", Math.max(0, clamp(x2) - clamp(x1)));
    r.setAttribute("height", H - 28);
    r.setAttribute("class", cls);
    band.appendChild(r);
  };
  rect(xOf(p.interval_90[0]), xOf(p.interval_90[1]), "band-rect-90");
  rect(xOf(p.interval_50[0]), xOf(p.interval_50[1]), "band-rect-50");

  const zero = $("dist-zero");
  const zeroX = clamp(xOf(0));
  zero.setAttribute("x1", zeroX);
  zero.setAttribute("x2", zeroX);
  zero.style.display = (0 >= d.lo && 0 <= d.hi) ? "" : "none";

  const med = $("dist-median");
  const medX = clamp(xOf(p.median_margin));
  med.setAttribute("x1", medX);
  med.setAttribute("x2", medX);

  $("dist-axis").innerHTML =
    `<span>${signed(d.lo)}</span>` +
    `<span>${p.median_margin >= 0 ? EL : ST} margin</span>` +
    `<span>${signed(d.hi)}</span>`;
}


// ---------------------------------------------------------------------------
// Maps
//
// Geometry is pre-projected to SVG path strings at build time (Albers conic,
// standard parallels 42.5 and 47), so the browser needs no projection library
// and the whole state is 18 KB.
//
// Two views, and they answer different questions:
//   "Counted so far"  - the margin in votes already reported
//   "Vote still out"  - the margin the model projects for the REMAINDER
//
// The second is the interesting one. A county whose absentee batch is fully in
// still has its Election Day vote outstanding, and Election Day runs about 21
// points better for El-Sayed than the county's blended average. So a county that
// looks close on the left map can have a lopsided remainder on the right.
// ---------------------------------------------------------------------------

let GEO = null;
let MAP_MODE = "results";
let LAST_COUNTIES = [];
let PINNED = null;

const MAP_SCALE = 30; // margin points at which the color ramp saturates

function rampColor(margin) {
  if (margin === null || margin === undefined) return "#C7CCC2";
  const t = Math.max(-1, Math.min(1, margin / MAP_SCALE));
  // Two-hue diverging ramp through the paper color, so a tied county reads as
  // neutral rather than as a third category.
  const mid = [232, 234, 227];
  const end = t >= 0 ? [31, 111, 107] : [150, 112, 26];
  const k = Math.pow(Math.abs(t), 0.75);
  const c = mid.map((m, i) => Math.round(m + (end[i] - m) * k));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

async function loadGeo() {
  try {
    const res = await fetch("mi-counties.json");
    GEO = await res.json();
    buildMap();
  } catch (err) {
    document.querySelector(".maps").hidden = true;
  }
}

function buildMap() {
  if (!GEO) return;
  const g = $("map-shapes");
  g.innerHTML = "";
  Object.entries(GEO.paths).forEach(([county, d]) => {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    p.setAttribute("fill", "#C7CCC2");
    p.dataset.county = county;
    p.addEventListener("mouseenter", (e) => showTip(county, e));
    p.addEventListener("mousemove", (e) => moveTip(e));
    p.addEventListener("mouseleave", hideTip);
    p.addEventListener("click", () => { PINNED = county; paintDetail(county); });
    g.appendChild(p);
  });
  paintMap();
}

function countyRow(name) {
  return LAST_COUNTIES.find((r) => r.county === name);
}

function mapValue(row) {
  if (!row) return null;
  if (MAP_MODE === "results") return row.reporting ? row.margin : null;
  return row.remaining > 0 ? row.remainder_margin : null;
}

function paintMap() {
  if (!GEO) return;
  $("map-shapes").querySelectorAll("path").forEach((p) => {
    p.setAttribute("fill", rampColor(mapValue(countyRow(p.dataset.county))));
    p.classList.toggle("sel", p.dataset.county === PINNED);
  });

  $("map-note").textContent = MAP_MODE === "results"
    ? "Margin in the votes each county has already reported. Grey counties have not reported."
    : "Margin the model projects for the vote each county has NOT yet counted. Grey counties are finished.";

  const stops = [-1, -0.6, -0.3, 0, 0.3, 0.6, 1]
    .map((t) => `<span style="background:${rampColor(t * MAP_SCALE)}"></span>`).join("");
  $("legend").innerHTML =
    `<div class="legend-scale">${stops}</div>
     <div class="legend-ends"><span>Stevens +${MAP_SCALE}</span><span>tie</span><span>El-Sayed +${MAP_SCALE}</span></div>
     <div class="legend-none"><i></i>${MAP_MODE === "results" ? "no results yet" : "counting complete"}</div>`;

  if (PINNED) paintDetail(PINNED);
}

function showTip(county, e) {
  const row = countyRow(county);
  const v = mapValue(row);
  $("map-tip").textContent =
    `${county}  ${v === null ? (MAP_MODE === "results" ? "no results" : "complete") : signed(v)}`;
  $("map-tip").hidden = false;
  moveTip(e);
  if (!PINNED) paintDetail(county);
}

function moveTip(e) {
  const box = $("map-svg").getBoundingClientRect();
  $("map-tip").style.left = (e.clientX - box.left) + "px";
  $("map-tip").style.top = (e.clientY - box.top) + "px";
}

function hideTip() { $("map-tip").hidden = true; }

function paintDetail(county) {
  const r = countyRow(county);
  if (!r) return;
  const cls = (v) => (v >= 0 ? "v-el" : "v-st");
  const pct = r.remaining && (r.remaining_early + r.remaining_ed)
    ? Math.round(r.remaining_early / (r.remaining_early + r.remaining_ed) * 100)
    : 0;

  $("map-detail").innerHTML = `
    <h3>${county}</h3>
    <dl>
      <dt>Counted</dt><dd>${num.format(r.votes)} of ${num.format(r.projected_total)}</dd>
      <dt>Margin so far</dt>
      <dd class="${r.margin === null ? "" : cls(r.margin)}">${r.margin === null ? "—" : signed(r.margin)}</dd>
      <dt>Pre-election baseline</dt><dd>${signed(r.expected_blended)}</dd>
      <dt>Still out</dt><dd>${num.format(r.remaining)}</dd>
      <dt>Remainder projects</dt>
      <dd class="${cls(r.remainder_margin)}">${signed(r.remainder_margin)}</dd>
      <dt>County lands at</dt>
      <dd class="${cls(r.projected_final)}">${signed(r.projected_final)}</dd>
    </dl>
    <p class="split-note">
      Of what is left, about ${pct}% is early vote and ${100 - pct}% Election Day.
      This county's early vote is modeled at ${signed(r.early_margin)} and its
      Election Day vote at ${signed(r.ed_margin)}.
    </p>`;
}

function initMapTabs() {
  const set = (mode) => {
    MAP_MODE = mode;
    const isResults = mode === "results";
    $("tab-results").classList.toggle("on", isResults);
    $("tab-remaining").classList.toggle("on", !isResults);
    $("tab-results").setAttribute("aria-selected", String(isResults));
    $("tab-remaining").setAttribute("aria-selected", String(!isResults));
    paintMap();
  };
  $("tab-results").addEventListener("click", () => set("results"));
  $("tab-remaining").addEventListener("click", () => set("remaining"));
}

// ---------------------------------------------------------------------------

let lastMargin = null;

function animateMargin(el, to) {
  const from = lastMargin;
  lastMargin = to;
  if (reduceMotion || from === null || Math.abs(to - from) < 0.05) {
    el.textContent = signed(to);
    return;
  }
  const start = performance.now();
  const dur = 550;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = signed(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderCounties(all) {
  const rows = (all || []).filter((r) => r.reporting);
  const body = $("county-rows");
  if (!rows.length) {
    body.innerHTML = `<tr class="empty"><td colspan="7">No counties reporting yet.</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((r) => {
    const cls = r.margin >= 0 ? "v-el" : "v-st";
    const sw = r.vs_expected >= 0 ? "v-el" : "v-st";
    return `<tr>
      <td class="name">${r.county}</td>
      <td class="num">${r.pct_of_projected.toFixed(0)}%</td>
      <td class="num">${num.format(r.el_sayed)}</td>
      <td class="num">${num.format(r.stevens)}</td>
      <td class="num ${cls}">${signed(r.margin)}</td>
      <td class="num">${signed(r.expected_blended)}</td>
      <td class="num ${sw}">${signed(r.vs_expected)}</td>
    </tr>`;
  }).join("");
}

function renderRegions(shifts) {
  const wrap = $("regions");
  const entries = Object.entries(shifts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(2, ...entries.map(([, v]) => Math.abs(v)));

  wrap.innerHTML = entries.map(([name, v]) => {
    const half = (Math.abs(v) / max) * 50;
    const pos = v >= 0;
    return `<div class="region">
      <div>
        <div class="region-name">${name.replace(/_/g, " ")}</div>
        <div class="region-track">
          <span class="region-mid"></span>
          <span class="region-fill ${pos ? "pos" : "neg"}"
                style="left:${pos ? 50 : 50 - half}%;width:${half}%"></span>
        </div>
      </div>
      <div class="region-val ${pos ? "v-el" : "v-st"}">${signed(v, 2)}</div>
    </div>`;
  }).join("");
}

function render(data) {
  const p = data.projection;
  const c = data.counted;
  const d = data.diagnostics;
  const t = data.turnout;

  const leadEl = p.median_margin >= 0;
  const leader = leadEl ? EL : ST;
  const prob = leadEl
    ? p.el_sayed_win_probability
    : 1 - p.el_sayed_win_probability;

  $("lead-name").textContent = leader;
  $("lead-name").className = "verdict-name " + (leadEl ? "el" : "st");
  $("lead-margin").className = "verdict-number " + (leadEl ? "el" : "st");
  animateMargin($("lead-margin"), Math.abs(p.median_margin));

  $("verdict-sub").textContent = d.counties_reporting === 0
    ? "Pre-election baseline. No counties reporting."
    : `From ${d.counties_reporting} ${d.counties_reporting === 1 ? "county" : "counties"}` +
      ` and ${c.pct_of_projected_turnout.toFixed(1)}% of the projected vote.`;

  drawDistribution(p);

  $("win-prob").textContent = (prob * 100).toFixed(prob > 0.995 ? 1 : 0) + "%";
  $("win-note").textContent = `${leader} wins in ${(prob * 100).toFixed(1)}% of simulations`;

  $("counted").textContent = c.pct_of_projected_turnout.toFixed(1) + "%";
  $("precincts").textContent =
    c.pct_precincts_reporting == null ? "—" : c.pct_precincts_reporting + "%";

  $("turnout").textContent = t.projected ? num.format(t.projected) : "—";
  $("turnout-ratio").textContent = "×" + (t.pooled_ratio ?? 1).toFixed(2);

  const total = (c.el_sayed || 0) + (c.stevens || 0);
  const esShare = total ? (c.el_sayed / total) * 100 : 50;
  $("es-votes").textContent = num.format(c.el_sayed || 0);
  $("st-votes").textContent = num.format(c.stevens || 0);
  $("tally-el").style.width = esShare + "%";
  $("tally-st").style.width = (100 - esShare) + "%";
  $("es-pct").textContent = esShare.toFixed(1) + "%";
  $("st-pct").textContent = (100 - esShare).toFixed(1) + "%";
  $("other-votes").textContent = c.other
    ? `${num.format(c.other)} to other candidates, excluded from the margin`
    : "";

  LAST_COUNTIES = data.counties || [];
  renderCounties(data.counties);
  paintMap();
  renderRegions(data.regional_shift || {});

  $("d-counties").textContent = d.counties_reporting;
  $("d-shift").textContent = signed(d.implied_state_shift, 2);
  $("d-gap").textContent = "×" + d.mode_gap_multiplier.toFixed(2);
  $("d-gapn").textContent = d.counties_calibrating_gap;
  $("d-ci50").textContent = `${signed(p.interval_50[0])} to ${signed(p.interval_50[1])}`;
  $("d-ci90").textContent = `${signed(p.interval_90[0])} to ${signed(p.interval_90[1])}`;

  const stamp = new Date(data.updated_at);
  const stale = Date.now() - stamp.getTime() > STALE_AFTER_MS;
  setPulse(stale ? "stale" : "live", stale ? "feed stale" : "live", stamp);

  // A county name the model could not match is silently dropped from the
  // projection, so it gets said out loud rather than buried in a log.
  if (d.unmatched_counties && d.unmatched_counties.length) {
    $("alert").textContent =
      "Not matched to a model county, and excluded from the projection: " +
      d.unmatched_counties.join(", ");
    $("alert").hidden = false;
  } else {
    $("alert").hidden = true;
  }
}

function setPulse(state, label, stamp) {
  $("pulse").dataset.state = state;
  $("pulse-label").textContent = label;
  if (stamp) {
    $("stamp").textContent = stamp.toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  }
}

async function tick() {
  try {
    const res = await fetch(API_BASE + "/api/projection", { cache: "no-store" });
    if (res.status === 503) {
      setPulse("connecting", "waiting for first results");
      return;
    }
    if (!res.ok) throw new Error("HTTP " + res.status);
    render(await res.json());
  } catch (err) {
    setPulse("stale", "reconnecting");
  }
}

initMapTabs();
loadGeo();
tick();
setInterval(tick, REFRESH_MS);
