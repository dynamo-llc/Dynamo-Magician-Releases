"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import "./DynamoMagician.css";

const MAG       = "#f062ff";
const VIO       = "#9d4eff";
const CYAN      = "#76e9ff";
const CYAN_SOFT = "#4fb4dc";
const TEXT      = "#dbe9f7";
const TEXT_DIM  = "#8aa2bc";
const RED_VAL   = "#ff5252";
const COPPER    = "#a97d72";

// ── Real stat helpers
function shannonEntropy(s: string): number {
  if (!s.length) return 0;
  const freq: Record<string, number> = {};
  for (const c of s) freq[c] = (freq[c] || 0) + 1;
  let H = 0;
  for (const f of Object.values(freq)) { const p = f / s.length; H -= p * Math.log2(p); }
  return H;
}
function secsToHMS(s: number): string {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

interface Chip { lines: string[]; style: React.CSSProperties; }

interface QueueEntry {
  job_id: string;
  promptPreview: string;
  imageThumb: string;   // small data-URL for thumbnail
  status: 'queued' | 'running' | 'done' | 'error';
  position: number;     // 0 = running, >0 = waiting at that position
  progress: { current: number; total: number };
  outputUrl: string | null;
  errorMsg: string | null;
  previewUrl: string | null;  // live step preview JPEG (base64 data-URL)
}

interface SliderCardProps {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; accent: string; badgeColor: string;
  format: (v: number) => string;
}

interface HistoryEntry {
  job_id: string;
  timestamp: number;
  prompt: string;
  params: { width: number; height: number; steps: number; guidance: number; frames: number; seed: number };
  output_file: string;
  thumbnail_file: string | null;
  duration_secs: number;
}

const clamp    = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const snapStep = (v: number, s: number, lo = 0) => parseFloat((Math.round((v - lo) / s) * s + lo).toFixed(10));

// ── Screw ornament
function Screw({ className = "" }: { className?: string }) {
  return (
    <svg className={`dm-screw ${className}`} width="18" height="18" viewBox="0 0 18 18">
      <circle cx="9" cy="9" r="8"   fill="#0c0a08" stroke="#5a4a38" strokeWidth="1.5"/>
      <circle cx="9" cy="9" r="4"   fill="#080606" stroke="#3d3020" strokeWidth="1"/>
      <line x1="5"  y1="9" x2="13" y2="9"  stroke="#6a5040" strokeWidth="1.4"/>
      <line x1="9"  y1="5" x2="9"  y2="13" stroke="#6a5040" strokeWidth="1.4"/>
      <circle cx="9" cy="9" r="1.3" fill="#2a2018"/>
    </svg>
  );
}

// ── Aether lightning arc
function AetherLightning() {
  const bolt = "M0,28 L70,14 L120,40 L190,8 L250,36 L310,12 L370,42 L430,16 L490,36 L550,10 L610,34 L670,12 L730,40 L790,6 L850,30 L910,14 L960,36 L1000,28";
  return (
    <div className="dm-arc-wrap">
      <svg className="dm-arc-svg" viewBox="0 0 1000 56" preserveAspectRatio="none">
        <defs>
          <filter id="ag" x="-10%" y="-100%" width="120%" height="300%">
            <feGaussianBlur stdDeviation="5" result="b"/>
            <feColorMatrix in="b" type="matrix"
              values="0 0 0 0 0.46  0 0 0 0 0.91  0 0 0 0 1  0 0 0 0.9 0" result="c"/>
            <feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <filter id="ag2" x="-10%" y="-100%" width="120%" height="300%">
            <feGaussianBlur stdDeviation="12"/>
          </filter>
        </defs>
        <path d={bolt} fill="none" stroke={CYAN} strokeWidth="18" opacity="0.07" filter="url(#ag2)"/>
        <path d={bolt} fill="none" stroke={CYAN} strokeWidth="2.5" opacity="0.88"  filter="url(#ag)"  className="dm-arc-main"/>
        <path d={bolt} fill="none" stroke="white" strokeWidth="0.9" opacity="0.65" className="dm-arc-core"/>
        {[190, 370, 550, 730].map((x, i) => (
          <circle key={i} cx={x} cy={i % 2 === 0 ? 8 : 42} r="5"
            fill="white" opacity="0.85" filter="url(#ag)" className="dm-arc-node"/>
        ))}
      </svg>
    </div>
  );
}

// ── Lightning progress bar
function LightningProgressBar({ pct, phase, indeterminate }: { pct: number; phase: number; indeterminate?: boolean }) {
  // Deterministic pseudo-random hash seeded by index + phase
  const hr = (i: number) => {
    const x = Math.sin(i * 127.1 + phase * 0.41 + i * 311.7) * 43758.5453;
    return x - Math.floor(x);
  };

  // Jagged bolt polyline: segs control-points, amp controls ± swing depth
  const makeBolt = (seed: number, amp: number, segs: number): string => {
    const pts = ['0,20'];
    for (let i = 1; i < segs; i++) {
      const x = (i / segs) * 1000;
      const y = 20 + (hr(seed + i * 7.3) - 0.5) * amp * 2;
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    pts.push('1000,20');
    return pts.join(' ');
  };

  // Forking branch off from a given start point
  const makeFork = (sx: number, sy: number, seed: number, dir: 1 | -1): string => {
    const len = 55 + hr(seed) * 90;
    const mx  = sx + len * 0.45;
    const my  = sy + dir * (12 + hr(seed + 1) * 16);
    const ex  = sx + len;
    const ey  = sy + dir * (6 + hr(seed + 2) * 10);
    return `M${sx.toFixed(0)},${sy.toFixed(0)} L${mx.toFixed(0)},${my.toFixed(0)} L${ex.toFixed(0)},${ey.toFixed(0)}`;
  };

  const b1 = makeBolt(1,  28, 22);
  const b2 = makeBolt(17, 22, 18);
  const b3 = makeBolt(31, 16, 15);
  const b4 = makeBolt(47, 10, 12);
  const b5 = makeBolt(61,  5, 9);

  const forks = [110, 260, 415, 565, 710, 860].map((x, i) => {
    const y = 20 + (hr(i * 13 + 1) - 0.5) * 22;
    return makeFork(x, y, i * 5 + 3, (i % 2 === 0 ? 1 : -1));
  });

  const clipW = Math.max(0, (pct / 100) * 1000);

  return (
    <div className={`dm-prog-lightning${indeterminate ? " dm-prog-indeterminate" : ""}`}>
      <svg viewBox="0 0 1000 40" preserveAspectRatio="none" width="100%" height="100%"
        style={{ display: "block", overflow: "visible" }}>
        <defs>
          <clipPath id="lp-clip">
            <rect x="0" y="-80" width={clipW} height="200"/>
          </clipPath>
          {/* Deep outer corona */}
          <filter id="lp-corona" x="-5%" y="-600%" width="110%" height="1300%">
            <feGaussianBlur stdDeviation="32" result="bc"/>
            <feColorMatrix in="bc" type="matrix"
              values="0.04 0 0 0 0.06  0 0.04 0 0 0.22  0 0 1 0 0.9  0 0 0 2.4 0"/>
          </filter>
          {/* Mid halo */}
          <filter id="lp-halo" x="-5%" y="-400%" width="110%" height="900%">
            <feGaussianBlur stdDeviation="13" result="bh"/>
            <feColorMatrix in="bh" type="matrix"
              values="0.1 0 0 0 0.18  0 0.1 0 0 0.60  0 0 1 0 1.0  0 0 0 2.0 0"/>
          </filter>
          {/* Tight core bloom */}
          <filter id="lp-bloom" x="-5%" y="-300%" width="110%" height="700%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="3.5" result="bg"/>
            <feColorMatrix in="bg" type="matrix"
              values="0.25 0 0 0 0.35  0 0.25 0 0 0.78  0 0 1 0 1.0  0 0 0 1.7 0" result="cg"/>
            <feMerge>
              <feMergeNode in="cg"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          {/* Tip spark */}
          <filter id="lp-tip" x="-500%" y="-500%" width="1100%" height="1100%">
            <feGaussianBlur stdDeviation="9" result="bt"/>
            <feColorMatrix in="bt" type="matrix"
              values="0.5 0 0 0 0.55  0 0.5 0 0 0.88  0 0 1 0 1.0  0 0 0 2.8 0" result="ct"/>
            <feMerge><feMergeNode in="ct"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <rect x="0" y="0" width="1000" height="40" fill="#010305"/>
        {/* Ghost unlit trace */}
        <polyline points={b1} fill="none" stroke="#182848" strokeWidth="1" opacity="0.15"/>

        <g clipPath="url(#lp-clip)">
          {/* Outer violet-blue corona envelope */}
          <polyline points={b1} fill="none" stroke="#2810a0" strokeWidth="100" opacity="0.20" filter="url(#lp-corona)"/>
          <polyline points={b1} fill="none" stroke="#3858e8" strokeWidth="60"  opacity="0.26" filter="url(#lp-corona)"/>
          {/* Mid halo */}
          <polyline points={b1} fill="none" stroke="#60a8ff" strokeWidth="40"  opacity="0.32" filter="url(#lp-halo)"/>
          <polyline points={b2} fill="none" stroke="#40d8ff" strokeWidth="28"  opacity="0.24" filter="url(#lp-halo)"/>
          <polyline points={b3} fill="none" stroke="#90e8ff" strokeWidth="18"  opacity="0.16" filter="url(#lp-halo)"/>
          {/* Core bolts with bloom */}
          <polyline points={b1} fill="none" stroke="#80d0ff" strokeWidth="3.2" opacity="0.97" filter="url(#lp-bloom)" className="dm-arc-main"/>
          <polyline points={b2} fill="none" stroke="#a0e0ff" strokeWidth="2.5" opacity="0.82" filter="url(#lp-bloom)"/>
          <polyline points={b3} fill="none" stroke="#bceeff" strokeWidth="1.8" opacity="0.68" filter="url(#lp-bloom)"/>
          <polyline points={b4} fill="none" stroke="#d4f4ff" strokeWidth="1.2" opacity="0.52" filter="url(#lp-bloom)"/>
          <polyline points={b5} fill="none" stroke="#eef9ff" strokeWidth="0.8" opacity="0.36" filter="url(#lp-bloom)"/>
          {/* White-hot cores */}
          <polyline points={b1} fill="none" stroke="white" strokeWidth="1.4" opacity="0.94" className="dm-arc-core"/>
          <polyline points={b2} fill="none" stroke="white" strokeWidth="0.9" opacity="0.70"/>
          <polyline points={b3} fill="none" stroke="white" strokeWidth="0.5" opacity="0.48"/>
          {/* Fork branches */}
          {forks.map((d, i) => (
            <g key={i}>
              <path d={d} fill="none" stroke="#70c0ff" strokeWidth="18" opacity="0.20" filter="url(#lp-halo)"/>
              <path d={d} fill="none" stroke="#a0deff" strokeWidth="1.8" opacity="0.78" filter="url(#lp-bloom)" className="dm-arc-main"/>
              <path d={d} fill="none" stroke="white"   strokeWidth="0.7" opacity="0.58"/>
            </g>
          ))}
          {/* Discharge nodes */}
          {[80, 200, 340, 490, 635, 775, 920].map((x, i) => {
            const y = 20 + (hr(i * 9 + 100) - 0.5) * 18;
            return (
              <circle key={i} cx={x} cy={y} r="5.5"
                fill="white" opacity="0.88" filter="url(#lp-bloom)" className="dm-arc-node"/>
            );
          })}
        </g>

        {/* Leading-edge tip spark */}
        {pct >= 1 && pct <= 99 && (
          <g>
            <circle cx={clipW} cy="20" r="18" fill="#3878ff" opacity="0.38" filter="url(#lp-tip)"/>
            <circle cx={clipW} cy="20" r="8"  fill="white"   opacity="1.0"  filter="url(#lp-tip)" className="dm-arc-node"/>
          </g>
        )}
      </svg>
    </div>
  );
}

// ── Arc helpers for Void Gate rings
function polarToCartesian(cx: number, cy: number, r: number, deg: number): { x: number; y: number } {
  const rad = (deg - 90) * Math.PI / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}
function describeArc(cx: number, cy: number, r: number, startDeg: number, endDeg: number): string {
  const s = polarToCartesian(cx, cy, r, startDeg);
  const e = polarToCartesian(cx, cy, r, endDeg);
  return `M${s.x.toFixed(2)},${s.y.toFixed(2)} A${r},${r} 0 0 1 ${e.x.toFixed(2)},${e.y.toFixed(2)}`;
}

// ── Transmute button — VOID GATE
function TransmutePortal({ active }: { active?: boolean }) {
  const W = 310, H = 98, ch = 9;
  const cx = 155, cy = 49;
  const frame = `M${ch},0 L${W-ch},0 L${W},${ch} L${W},${H-ch} L${W-ch},${H} L${ch},${H} L0,${H-ch} L0,${ch} Z`;
  const outerSparks = [0, 110, 120, 230, 240, 350];

  return (
    <svg className="dm-vortex-svg" viewBox="0 0 310 98" preserveAspectRatio="none">
      <defs>
        <radialGradient id="vg-bg" cx="50%" cy="50%" r="65%">
          <stop offset="0%"   stopColor="#1e0045"/>
          <stop offset="42%"  stopColor="#0a001e"/>
          <stop offset="100%" stopColor="#03000a"/>
        </radialGradient>
        <filter id="vg-mag" x="-60%" y="-200%" width="220%" height="500%">
          <feGaussianBlur stdDeviation="3.5" result="b"/>
          <feColorMatrix in="b" type="matrix"
            values="1 0 0 0 0.75  0 0 0 0 0.05  0 0 1 0 0.80  0 0 0 2.8 0" result="c"/>
          <feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="vg-cyn" x="-60%" y="-200%" width="220%" height="500%">
          <feGaussianBlur stdDeviation="3.0" result="b"/>
          <feColorMatrix in="b" type="matrix"
            values="0 0 0 0 0.06  0 0 0 0 0.82  0 0 0 0 1  0 0 0 2.5 0" result="c"/>
          <feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="vg-wht" x="-100%" y="-400%" width="300%" height="900%">
          <feGaussianBlur stdDeviation="4.5" result="b"/>
          <feColorMatrix in="b" type="matrix"
            values="1 0 0 0 0.9  0 1 0 0 0.9  0 0 1 0 1  0 0 0 2.0 0" result="c"/>
          <feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="vg-amb" x="-80%" y="-300%" width="260%" height="700%">
          <feGaussianBlur stdDeviation="18"/>
        </filter>
        <filter id="vg-frm" x="-10%" y="-30%" width="120%" height="160%">
          <feGaussianBlur stdDeviation="3.5" result="b"/>
          <feColorMatrix in="b" type="matrix"
            values="1 0 0 0 0.7  0 0 0 0 0.05  0 0 1 0 0.75  0 0 0 2.2 0"/>
        </filter>
        <clipPath id="vg-cp"><path d={frame}/></clipPath>
      </defs>

      <g clipPath="url(#vg-cp)">
        {/* Deep void background */}
        <rect width="310" height="98" fill="url(#vg-bg)"/>
        {/* Ambient bloom */}
        <circle cx={cx} cy={cy} r="55" fill="#5010c0" opacity="0.28" filter="url(#vg-amb)"/>

        {/* Outer ring: 3 × 110° arcs, magenta, slow CW */}
        <g className={active ? "dm-vg-r1-a" : "dm-vg-r1"}>
          {[0, 120, 240].map(start => (
            <path key={start}
              d={describeArc(cx, cy, 40, start, start + 110)}
              fill="none" stroke="#d840ff" strokeWidth="2.1" opacity="0.90"
              filter="url(#vg-mag)"/>
          ))}
          {outerSparks.map(deg => {
            const p = polarToCartesian(cx, cy, 40, deg);
            return <circle key={deg} cx={p.x} cy={p.y} r="2.0"
              fill="#ffaaff" opacity="0.95" filter="url(#vg-mag)"/>;
          })}
        </g>

        {/* Mid ring: 4 × 80° arcs, cyan, medium CCW */}
        <g className={active ? "dm-vg-r2-a" : "dm-vg-r2"}>
          {[0, 90, 180, 270].map(start => (
            <path key={start}
              d={describeArc(cx, cy, 29, start, start + 80)}
              fill="none" stroke="#40e8ff" strokeWidth="1.5" opacity="0.80"
              filter="url(#vg-cyn)"/>
          ))}
        </g>

        {/* Inner ring: 6 radial ticks, magenta, fast CW */}
        <g className={active ? "dm-vg-r3-a" : "dm-vg-r3"}>
          {[0, 60, 120, 180, 240, 300].map(a => {
            const s = polarToCartesian(cx, cy, 14, a);
            const e = polarToCartesian(cx, cy, 22, a);
            return <line key={a}
              x1={s.x.toFixed(2)} y1={s.y.toFixed(2)}
              x2={e.x.toFixed(2)} y2={e.y.toFixed(2)}
              stroke="#f062ff" strokeWidth="1.5" opacity="0.95" filter="url(#vg-mag)"/>;
          })}
          <circle cx={cx} cy={cy} r="13" fill="none" stroke="#700898" strokeWidth="0.8" opacity="0.60"/>
        </g>

        {/* Orbiting particles — CCW ring */}
        <g className={active ? "dm-vg-pts-a" : "dm-vg-pts"}>
          {[0, 45, 90, 135, 180, 225, 270, 315].map((a, i) => {
            const p = polarToCartesian(cx, cy, 35, a);
            return <circle key={i} cx={p.x} cy={p.y}
              r={i % 2 === 0 ? 1.8 : 1.2}
              fill={i % 2 === 0 ? "#f062ff" : "#76e9ff"}
              opacity={i % 2 === 0 ? 0.90 : 0.72}
              filter={i % 2 === 0 ? "url(#vg-mag)" : "url(#vg-cyn)"}/>;
          })}
        </g>

        {/* Center sigil — slow CW rotation */}
        <g className="dm-vg-core">
          <path
            d={`M${cx},${cy-11} L${cx+3},${cy-3} L${cx+11},${cy} L${cx+3},${cy+3} L${cx},${cy+11} L${cx-3},${cy+3} L${cx-11},${cy} L${cx-3},${cy-3} Z`}
            fill="none" stroke="#ffffff" strokeWidth="1.0" opacity="0.80" filter="url(#vg-wht)"/>
          <line x1={cx-11} y1={cy} x2={cx+11} y2={cy} stroke="#ffffff" strokeWidth="0.5" opacity="0.35"/>
          <line x1={cx} y1={cy-11} x2={cx} y2={cy+11} stroke="#ffffff" strokeWidth="0.5" opacity="0.35"/>
          <circle cx={cx} cy={cy} r="3" fill="#ffffff" opacity="1.0" filter="url(#vg-wht)"/>
          <circle cx={cx} cy={cy} r="1.2" fill="#ffffff" opacity="1.0"/>
        </g>
      </g>

      {/* Frame on top */}
      <path d={frame} fill="none" stroke="#0a0416" strokeWidth="3.5" opacity="0.96"/>
      <path d={frame} fill="none" stroke="#cc40ff" strokeWidth="0.9" opacity="0.68" filter="url(#vg-frm)"/>

      {/* Corner HUD notches — magenta */}
      <polyline points={`${ch+1},1.5 1.5,1.5 1.5,${ch+1}`}             fill="none" stroke="#d840ff" strokeWidth="1.4" opacity="0.90"/>
      <polyline points={`${W-ch-1},1.5 ${W-1.5},1.5 ${W-1.5},${ch+1}`} fill="none" stroke="#d840ff" strokeWidth="1.4" opacity="0.90"/>
      <polyline points={`${ch+1},${H-1.5} 1.5,${H-1.5} 1.5,${H-ch-1}`} fill="none" stroke="#d840ff" strokeWidth="1.4" opacity="0.90"/>
      <polyline points={`${W-ch-1},${H-1.5} ${W-1.5},${H-1.5} ${W-1.5},${H-ch-1}`} fill="none" stroke="#d840ff" strokeWidth="1.4" opacity="0.90"/>
    </svg>
  );
}

// ── Icon atoms
function IconWave({ color = CYAN }: { color?: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <polyline points="1,10 4,10 5,5 7,15 9,10 11,10 12,7 14,13 16,10 19,10"
        stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
        style={{ filter: `drop-shadow(0 0 3px ${color})` }}/>
    </svg>
  );
}
function IconFingerprint({ color = CYAN }: { color?: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <path d="M10 3a7 7 0 0 1 7 7" stroke={color} strokeWidth="1.4" strokeLinecap="round"/>
      <path d="M5.5 6A6 6 0 0 1 16 10c0 2-.5 3.5-1 5" stroke={color} strokeWidth="1.3" strokeLinecap="round"/>
      <path d="M4 10a6 6 0 0 1 .5-2.5" stroke={color} strokeWidth="1.3" strokeLinecap="round"/>
      <path d="M8 10c0-1.1.9-2 2-2s2 .9 2 2c0 2-.8 4-2 5" stroke={color} strokeWidth="1.3" strokeLinecap="round"/>
      <path d="M8 10c0 1.5.3 3.5 1 5" stroke={color} strokeWidth="1.3" strokeLinecap="round"/>
      <rect x="1" y="9" width="18" height="1.5" rx="0.75" fill={color} opacity="0.15"/>
    </svg>
  );
}
function IconXCircle({ color = "#ff6f87" }: { color?: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="10" r="7" stroke={color} strokeWidth="1.5"/>
      <line x1="7" y1="7" x2="13" y2="13" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
      <line x1="13" y1="7" x2="7" y2="13" stroke={color} strokeWidth="1.5" strokeLinecap="round"/>
    </svg>
  );
}
function IconMinus({ color = TEXT_DIM }: { color?: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <line x1="4" y1="10" x2="16" y2="10" stroke={color} strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}
function IconXClose({ color = "#ff6f87" }: { color?: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <line x1="4" y1="4"  x2="16" y2="16" stroke={color} strokeWidth="2" strokeLinecap="round"/>
      <line x1="16" y1="4" x2="4"  y2="16" stroke={color} strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}
function IconDB({ color = CYAN }: { color?: string }) {
  // Server stack with right-egress arrow — DATA INGEST
  return (
    <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
      <defs>
        <filter id="ib-glow">
          <feGaussianBlur stdDeviation="1.8" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      {/* Server body */}
      <rect x="4" y="8" width="26" height="6" rx="1.5" stroke={color} strokeWidth="1.3" fill="none" filter="url(#ib-glow)"/>
      <rect x="4" y="17" width="26" height="6" rx="1.5" stroke={color} strokeWidth="1.3" fill="none" filter="url(#ib-glow)"/>
      <rect x="4" y="26" width="26" height="6" rx="1.5" stroke={color} strokeWidth="1.3" fill="none" filter="url(#ib-glow)"/>
      {/* Drive indicator dots */}
      {[11, 20, 29].map(y => (
        <circle key={y} cx="26" cy={y+0.5} r="1.4" fill={color} opacity="0.85" filter="url(#ib-glow)"/>
      ))}
      {/* Data row lines */}
      {[11, 20, 29].map(y => [
        <line key={y+"a"} x1="8" y1={y} x2="18" y2={y} stroke={color} strokeWidth="1" opacity="0.55"/>,
        <line key={y+"b"} x1="8" y1={y+2.5} x2="15" y2={y+2.5} stroke={color} strokeWidth="1" opacity="0.35"/>,
      ])}
      {/* Egress arrow (right side) */}
      <path d="M32 22 L40 22" stroke={color} strokeWidth="1.6" strokeLinecap="round"/>
      <path d="M37 18.5 L40 22 L37 25.5" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}
function IconBrain({ color = CYAN }: { color?: string }) {
  // Neural-ring: outer orbit of nodes + internal connection mesh — COGNITIVE PROCESSING
  const nodeAngs = [0, 45, 90, 135, 180, 225, 270, 315];
  const R = 13, r2 = 7;
  const nodes = nodeAngs.map(a => ({
    x: 22 + R * Math.cos(a * Math.PI / 180),
    y: 22 + R * Math.sin(a * Math.PI / 180),
  }));
  const inner = [0, 2, 4, 6].map(i => nodes[i]);
  return (
    <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
      <defs>
        <filter id="ib2-glow">
          <feGaussianBlur stdDeviation="2.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      {/* Outer ring */}
      <circle cx="22" cy="22" r={R} stroke={color} strokeWidth="0.8" opacity="0.30" strokeDasharray="3 2"/>
      {/* Spoke lines center→node */}
      {nodes.map((n, i) => (
        <line key={i} x1="22" y1="22" x2={n.x} y2={n.y}
          stroke={color} strokeWidth="0.7" opacity="0.28"/>
      ))}
      {/* Cross-connect between adjacent nodes */}
      {nodes.map((n, i) => {
        const m = nodes[(i + 2) % nodes.length];
        return <line key={"c"+i} x1={n.x} y1={n.y} x2={m.x} y2={m.y}
          stroke={color} strokeWidth="0.6" opacity="0.20"/>;
      })}
      {/* Inner quad lines */}
      {inner.map((n, i) => {
        const m = inner[(i + 1) % inner.length];
        return <line key={"in"+i} x1={n.x} y1={n.y} x2={m.x} y2={m.y}
          stroke={color} strokeWidth="0.8" opacity="0.35"/>;
      })}
      {/* Outer node dots */}
      {nodes.map((n, i) => (
        <circle key={"d"+i} cx={n.x} cy={n.y} r={i % 2 === 0 ? 1.8 : 1.2}
          fill={color} opacity={i % 2 === 0 ? 0.90 : 0.55} filter="url(#ib2-glow)"/>
      ))}
      {/* Center brain-core */}
      <circle cx="22" cy="22" r={r2} stroke={color} strokeWidth="1.2" opacity="0.60" filter="url(#ib2-glow)"/>
      <circle cx="22" cy="22" r="2.8" fill={color} opacity="0.80" filter="url(#ib2-glow)"/>
    </svg>
  );
}
function IconFlux({ color = CYAN }: { color?: string }) {
  // Hub-and-spoke with electron orbits — AETHER-FLUX
  const spokes = 8;
  const spokeData = Array.from({ length: spokes }, (_, i) => {
    const a = (i * 360 / spokes) * Math.PI / 180;
    return { x2: 22 + 16 * Math.cos(a), y2: 22 + 16 * Math.sin(a) };
  });
  const dots = Array.from({ length: spokes }, (_, i) => {
    const a = (i * 360 / spokes) * Math.PI / 180;
    return { cx: 22 + 16 * Math.cos(a), cy: 22 + 16 * Math.sin(a) };
  });
  const midDots = [1, 3, 5, 7].map(i => {
    const a = (i * 360 / spokes) * Math.PI / 180;
    return { cx: 22 + 9 * Math.cos(a), cy: 22 + 9 * Math.sin(a) };
  });
  return (
    <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
      <defs>
        <filter id="ib3-glow">
          <feGaussianBlur stdDeviation="2.2" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      {/* Outer ring */}
      <circle cx="22" cy="22" r="16" stroke={color} strokeWidth="0.8" opacity="0.30" strokeDasharray="2.5 2"/>
      {/* Spokes */}
      {spokeData.map((s, i) => (
        <line key={i} x1="22" y1="22" x2={s.x2} y2={s.y2}
          stroke={color} strokeWidth="0.9" opacity="0.45"/>
      ))}
      {/* Mid-ring connector */}
      <circle cx="22" cy="22" r="9" stroke={color} strokeWidth="0.7" opacity="0.25" strokeDasharray="1.5 3"/>
      {/* Spoke-tip nodes */}
      {dots.map((d, i) => (
        <circle key={i} cx={d.cx} cy={d.cy} r="1.6"
          fill={color} opacity="0.85" filter="url(#ib3-glow)"/>
      ))}
      {/* Mid-ring accent nodes */}
      {midDots.map((d, i) => (
        <circle key={i} cx={d.cx} cy={d.cy} r="1.0"
          fill={color} opacity="0.50"/>
      ))}
      {/* Central hub */}
      <circle cx="22" cy="22" r="4.5" stroke={color} strokeWidth="1.3" opacity="0.70" filter="url(#ib3-glow)"/>
      <circle cx="22" cy="22" r="2" fill={color} opacity="0.90" filter="url(#ib3-glow)"/>
    </svg>
  );
}

// ── Radial arc gauge meter ────────────────────────────────────────────────────
function GaugeMeter({ label, pct, color, unit }: { label: string; pct: number; color: string; unit?: string }) {
  const R = 34, CX = 50, CY = 54;
  const CIRC = 2 * Math.PI * R;
  const ARC  = CIRC * 0.75;
  const fillLen = ARC * Math.max(0, Math.min(pct, 100)) / 100;
  const crit = pct >= 85;
  const warn = pct >= 70;
  const drawColor = crit ? "#ff3c3c" : warn ? "#ffb020" : color;

  const tipAngle = (135 + 270 * (Math.max(0, Math.min(pct, 100)) / 100)) * Math.PI / 180;
  const tipX = CX + R * Math.cos(tipAngle);
  const tipY = CY + R * Math.sin(tipAngle);
  const showTip = pct >= 2;

  const majorTicks = [0, 0.25, 0.5, 0.75, 1.0].map(t => {
    const a = (135 + 270 * t) * Math.PI / 180;
    return {
      x1: CX + (R - 5) * Math.cos(a), y1: CY + (R - 5) * Math.sin(a),
      x2: CX + (R + 6) * Math.cos(a), y2: CY + (R + 6) * Math.sin(a),
    };
  });
  const minorTicks = [1,2,3,4,6,7,8,9].map(k => {
    const a = (135 + k * 27) * Math.PI / 180;
    return {
      x1: CX + (R - 2) * Math.cos(a), y1: CY + (R - 2) * Math.sin(a),
      x2: CX + (R + 3) * Math.cos(a), y2: CY + (R + 3) * Math.sin(a),
    };
  });

  const fid = `gf-${label.replace(/\W/g, '')}`;

  return (
    <svg viewBox="0 0 100 100" className={`dm-gauge-svg${crit ? " dm-gauge-crit" : ""}`} aria-label={`${label} ${pct}%`}>
      <defs>
        <filter id={fid} x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="3.5" result="b"/>
          <feColorMatrix in="b" type="saturate" values="2.5" result="c"/>
          <feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id={`${fid}t`} x="-300%" y="-300%" width="700%" height="700%">
          <feGaussianBlur stdDeviation="3.0" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <radialGradient id={`${fid}bg`} cx="50%" cy="50%">
          <stop offset="0%" stopColor={drawColor} stopOpacity="0.06"/>
          <stop offset="100%" stopColor={drawColor} stopOpacity="0"/>
        </radialGradient>
      </defs>

      <circle cx={CX} cy={CY} r="38" fill={`url(#${fid}bg)`}/>
      <circle cx={CX} cy={CY} r="26" fill="none" stroke={drawColor} strokeWidth="0.4" opacity="0.18"/>
      <circle cx={CX} cy={CY} r="20" fill="none" stroke={drawColor} strokeWidth="0.3" opacity="0.12"/>

      {minorTicks.map((t, i) => (
        <line key={i} x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2} stroke="#1a2a3a" strokeWidth="0.8"/>
      ))}
      {majorTicks.map((t, i) => (
        <line key={i} x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2} stroke="#2a3d52" strokeWidth="1.5"/>
      ))}

      {/* Track */}
      <circle cx={CX} cy={CY} r={R} fill="none" stroke="#070f18" strokeWidth="7"
        strokeDasharray={`${ARC} ${CIRC - ARC}`} strokeLinecap="round"
        transform={`rotate(135, ${CX}, ${CY})`} opacity="0.7"/>
      <circle cx={CX} cy={CY} r={R} fill="none" stroke="#0e1d2a" strokeWidth="4"
        strokeDasharray={`${ARC} ${CIRC - ARC}`} strokeLinecap="round"
        transform={`rotate(135, ${CX}, ${CY})`}/>

      {/* Fill glow */}
      {fillLen > 0 && (
        <circle cx={CX} cy={CY} r={R} fill="none" stroke={drawColor} strokeWidth="9"
          strokeDasharray={`${fillLen} ${CIRC}`} strokeLinecap="round"
          transform={`rotate(135, ${CX}, ${CY})`} opacity="0.22"
          filter={`url(#${fid})`}/>
      )}
      {/* Fill core */}
      {fillLen > 0 && (
        <circle cx={CX} cy={CY} r={R} fill="none" stroke={drawColor} strokeWidth="4"
          strokeDasharray={`${fillLen} ${CIRC}`} strokeLinecap="round"
          transform={`rotate(135, ${CX}, ${CY})`}
          filter={`url(#${fid})`}/>
      )}

      {/* Tip hotspot */}
      {showTip && (
        <>
          <circle cx={tipX} cy={tipY} r="6" fill={drawColor} opacity="0.3" filter={`url(#${fid}t)`}/>
          <circle cx={tipX} cy={tipY} r="2.2" fill="#ffffff" opacity="0.92"/>
        </>
      )}

      {/* Value */}
      <text x={CX} y={CY - 4} textAnchor="middle"
        fill={drawColor} fontSize="19" fontFamily="Consolas,monospace" fontWeight="bold"
        filter={`url(#${fid})`}>{pct}</text>
      <text x={CX} y={CY + 9} textAnchor="middle"
        fill={drawColor} fontSize="8" fontFamily="Consolas,monospace" opacity="0.80">{unit ?? '%'}</text>

      {/* Label */}
      <text x={CX} y="96" textAnchor="middle"
        fill="#f5ff30" fontSize="9.5" fontFamily="Consolas,monospace" fontWeight="bold" letterSpacing="0.08em"
        style={{ filter: "drop-shadow(0 0 4px #f5ff30cc)" }}>{label}</text>
      <circle cx="50" cy="87" r="1.8" fill={drawColor} opacity="0.80"/>
    </svg>
  );
}

// ── Circuit board background
function CircuitBg() {
  const G  = "#39ff14";   // neon green
  const GH = "#a0ffb0";   // lighter green for halos
  const W  = "#ffffff";   // white-hot core
  const op = 0.52;        // base trace opacity

  // Named paths reused for animateMotion
  const pRBus  = "M432,0 L432,780";
  const pRBusR = "M432,780 L432,0";
  const pBBot  = "M0,748 L460,748";
  const pBBotR = "M460,748 L0,748";
  const pD1    = "M380,148 L360,168 L360,248";
  const pD2    = "M370,300 L350,320 L350,370";
  const pD3    = "M375,540 L355,560 L355,620";

  // Helper: a travelling electricity pulse — halo + core at the same position
  function Pulse({ path, dur, begin = "0s", haloR = 7, coreR = 2, haloOp = 0.55 }: {
    path: string; dur: string; begin?: string; haloR?: number; coreR?: number; haloOp?: number;
  }) {
    return (
      <>
        <circle r={haloR} fill={GH} opacity={haloOp} filter="url(#cb-phalo)">
          <animateMotion path={path} dur={dur} begin={begin} repeatCount="indefinite" calcMode="linear"/>
        </circle>
        <circle r={coreR} fill={W} opacity="0.95" filter="url(#cb-pcore)">
          <animateMotion path={path} dur={dur} begin={begin} repeatCount="indefinite" calcMode="linear"/>
        </circle>
      </>
    );
  }

  // Helper: voltage spike — extremely bright fast burst
  function Spike({ path, dur, begin = "0s" }: { path: string; dur: string; begin?: string }) {
    return (
      <>
        <circle r={14} fill={G} opacity="0.85" filter="url(#cb-spike)">
          <animateMotion path={path} dur={dur} begin={begin} repeatCount="indefinite" calcMode="linear"/>
        </circle>
        <circle r={3.5} fill={W} opacity="1" filter="url(#cb-pcore)">
          <animateMotion path={path} dur={dur} begin={begin} repeatCount="indefinite" calcMode="linear"/>
        </circle>
      </>
    );
  }

  return (
    <svg className="dm-circuit-bg" viewBox="0 0 460 780" preserveAspectRatio="xMaxYMax slice" aria-hidden="true">
      <defs>
        {/* Trace edge glow */}
        <filter id="cb-glow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="2.4" result="b"/>
          <feColorMatrix in="b" type="matrix"
            values="0 0 0 0 0.1  0 1 0 0 0.2  0 0 0.1 0 0  0 0 0 2 0" result="c"/>
          <feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        {/* Soft ambient haze */}
        <filter id="cb-soft" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="5"/>
          <feColorMatrix type="matrix"
            values="0 0 0 0 0.1  0 1 0 0 0.2  0 0 0.1 0 0  0 0 0 1.6 0"/>
        </filter>
        {/* Pulse halo — wide bloom */}
        <filter id="cb-phalo" x="-800%" y="-800%" width="1700%" height="1700%">
          <feGaussianBlur stdDeviation="9" result="b"/>
          <feColorMatrix in="b" type="matrix"
            values="0 0 0 0 0.05  0 1 0 0 0.1  0 0 0 0 0  0 0 0 3.0 0"/>
        </filter>
        {/* Pulse core — tight bright bloom */}
        <filter id="cb-pcore" x="-600%" y="-600%" width="1300%" height="1300%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="2.2" result="b"/>
          <feColorMatrix in="b" type="matrix"
            values="0.3 0 0 0 0.2  0 1 0 0 0.4  0 0 0.3 0 0  0 0 0 3.5 0" result="c"/>
          <feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        {/* Voltage spike — massive bloom */}
        <filter id="cb-spike" x="-1000%" y="-1000%" width="2100%" height="2100%">
          <feGaussianBlur stdDeviation="18" result="b"/>
          <feColorMatrix in="b" type="matrix"
            values="0.3 0 0 0 0.2  0 1 0 0 0.4  0 0 0.3 0 0  0 0 0 3.5 0"/>
        </filter>
      </defs>

      {/* ══ RIGHT VERTICAL BUS ══ */}
      <line x1="432" y1="0" x2="432" y2="780" stroke={G} strokeWidth="1.6" opacity={op} filter="url(#cb-glow)"/>
      <line x1="448" y1="0" x2="448" y2="780" stroke={G} strokeWidth="0.8" opacity={op * 0.45}/>

      {/* Branches off right bus */}
      {[60,100,148,196,248,300,355,410,468,520,572,624,680,730].map((y,i) => {
        const len = 40 + (i % 3) * 20;
        return (
          <g key={y}>
            <line x1={432} y1={y} x2={432-len} y2={y} stroke={G} strokeWidth="1.2" opacity={op} filter="url(#cb-glow)"/>
            <circle cx={432} cy={y} r="3.5" fill="none" stroke={G} strokeWidth="1.2" opacity={op+0.12} filter="url(#cb-glow)"/>
            <circle cx={432} cy={y} r="1.4" fill={G} opacity={op+0.18}/>
            <circle cx={432-len} cy={y} r="3.0" fill="none" stroke={G} strokeWidth="1.1" opacity={op}/>
            <circle cx={432-len} cy={y} r="1.2" fill={G} opacity={op}/>
          </g>
        );
      })}

      {/* ══ BOTTOM HORIZONTAL BUS ══ */}
      <line x1="0" y1="748" x2="460" y2="748" stroke={G} strokeWidth="1.5" opacity={op} filter="url(#cb-glow)"/>
      <line x1="0" y1="762" x2="460" y2="762" stroke={G} strokeWidth="0.8" opacity={op * 0.45}/>

      {/* Branches off bottom bus */}
      {[38,88,140,195,252,308,362,418].map((x,i) => {
        const len = 30 + (i % 4) * 12;
        return (
          <g key={x}>
            <line x1={x} y1={748} x2={x} y2={748-len} stroke={G} strokeWidth="1.1" opacity={op} filter="url(#cb-glow)"/>
            <circle cx={x} cy={748} r="3.0" fill="none" stroke={G} strokeWidth="1.1" opacity={op+0.12} filter="url(#cb-glow)"/>
            <circle cx={x} cy={748} r="1.2" fill={G} opacity={op+0.15}/>
            <circle cx={x} cy={748-len} r="2.8" fill="none" stroke={G} strokeWidth="1.0" opacity={op}/>
            <circle cx={x} cy={748-len} r="1.1" fill={G} opacity={op}/>
          </g>
        );
      })}

      {/* ══ IC CHIP 1 ══ */}
      <rect x="388" y="90" width="36" height="50" rx="2" fill="#030a04" stroke={G} strokeWidth="1.2" opacity={op+0.1}/>
      {[4,12,20,28,36].map((dy,i) => (
        <g key={i}>
          <line x1="388" y1={98+dy} x2="382" y2={98+dy} stroke={G} strokeWidth="1.0" opacity={op}/>
          <circle cx="382" cy={98+dy} r="1.6" fill={G} opacity={op+0.1}/>
          <line x1="424" y1={98+dy} x2="430" y2={98+dy} stroke={G} strokeWidth="1.0" opacity={op}/>
          <circle cx="430" cy={98+dy} r="1.6" fill={G} opacity={op+0.1}/>
        </g>
      ))}
      <rect x="396" y="100" width="20" height="12" rx="1" fill="#081408" stroke={G} strokeWidth="0.7" opacity={op * 0.7}/>
      {/* IC-1 internal glow flicker */}
      <rect x="389" y="91" width="34" height="48" rx="2" fill={G} opacity="0">
        <animate attributeName="opacity" values="0;0;0.08;0;0.05;0;0;0.12;0" dur="2.4s" repeatCount="indefinite" begin="0.3s"/>
      </rect>

      {/* ══ IC CHIP 2 ══ */}
      <rect x="388" y="260" width="40" height="64" rx="2" fill="#030a04" stroke={G} strokeWidth="1.2" opacity={op+0.1}/>
      {[0,10,20,30,40,50].map((dy,i) => (
        <g key={i}>
          <line x1="388" y1={270+dy} x2="381" y2={270+dy} stroke={G} strokeWidth="1.0" opacity={op}/>
          <circle cx="381" cy={270+dy} r="1.6" fill={G} opacity={op+0.1}/>
          <line x1="428" y1={270+dy} x2="435" y2={270+dy} stroke={G} strokeWidth="1.0" opacity={op}/>
          <circle cx="435" cy={270+dy} r="1.6" fill={G} opacity={op+0.1}/>
        </g>
      ))}
      <rect x="395" y="272" width="26" height="14" rx="1" fill="#081408" stroke={G} strokeWidth="0.7" opacity={op * 0.7}/>
      {/* IC-2 glow flicker */}
      <rect x="389" y="261" width="38" height="62" rx="2" fill={G} opacity="0">
        <animate attributeName="opacity" values="0;0.06;0;0;0.1;0;0.04;0" dur="3.2s" repeatCount="indefinite" begin="1.1s"/>
      </rect>

      {/* ══ IC CHIP 3 (tall) ══ */}
      <rect x="390" y="440" width="32" height="80" rx="2" fill="#030a04" stroke={G} strokeWidth="1.2" opacity={op+0.1}/>
      {[0,12,24,36,48,60,72].map((dy,i) => (
        <g key={i}>
          <line x1="390" y1={452+dy} x2="384" y2={452+dy} stroke={G} strokeWidth="1.0" opacity={op}/>
          <circle cx="384" cy={452+dy} r="1.5" fill={G} opacity={op}/>
          <line x1="422" y1={452+dy} x2="428" y2={452+dy} stroke={G} strokeWidth="1.0" opacity={op}/>
          <circle cx="428" cy={452+dy} r="1.5" fill={G} opacity={op}/>
        </g>
      ))}
      {/* IC-3 glow flicker */}
      <rect x="391" y="441" width="30" height="78" rx="2" fill={G} opacity="0">
        <animate attributeName="opacity" values="0;0;0.07;0.03;0;0.09;0" dur="2.9s" repeatCount="indefinite" begin="0.6s"/>
      </rect>

      {/* ══ RESISTORS ══ */}
      {[{ x:385, y:185 },{ x:385, y:370 },{ x:385, y:600 },{ x:385, y:700 }].map(({ x, y },i) => (
        <g key={i}>
          <line x1={x} y1={y} x2={x+7} y2={y} stroke={G} strokeWidth="1.0" opacity={op}/>
          <line x1={x+19} y1={y} x2={x+26} y2={y} stroke={G} strokeWidth="1.0" opacity={op}/>
          <rect x={x+7} y={y-4} width="12" height="8" rx="1.5" fill="#061008" stroke={G} strokeWidth="1.0" opacity={op+0.1}/>
          <line x1={x+12} y1={y-4} x2={x+12} y2={y+4} stroke={G} strokeWidth="1.2" opacity={op+0.18}/>
          <circle cx={x} cy={y} r="2.2" fill="none" stroke={G} strokeWidth="1.0" opacity={op}/>
          <circle cx={x+26} cy={y} r="2.2" fill="none" stroke={G} strokeWidth="1.0" opacity={op}/>
        </g>
      ))}

      {/* ══ VIA HOLES ══ */}
      {([
        [410,160],[420,220],[405,340],[415,500],[408,560],[418,640],[412,710],
        [350,748],[270,748],[180,748],[100,748],
      ] as [number,number][]).map(([x,y],i) => (
        <g key={i}>
          <circle cx={x} cy={y} r="4.5" fill="none" stroke={G} strokeWidth="1.1" opacity={op} filter="url(#cb-soft)"/>
          <circle cx={x} cy={y} r="2.2" fill="none" stroke={G} strokeWidth="1.0" opacity={op+0.12}/>
          <circle cx={x} cy={y} r="0.9" fill={G} opacity={op+0.22}/>
          {/* Arc discharge spark */}
          <circle cx={x} cy={y} r="5.5" fill={G} opacity="0" filter="url(#cb-phalo)">
            <animate attributeName="opacity"
              values={`0;0;${0.5+i*0.04};0;0;${0.3};0`}
              dur={`${1.2+i*0.35}s`} repeatCount="indefinite" begin={`${i*0.28}s`}/>
          </circle>
        </g>
      ))}

      {/* ══ DIAGONAL TRACES ══ */}
      <path d={pD1} fill="none" stroke={G} strokeWidth="1.1" opacity={op * 0.85} filter="url(#cb-glow)"/>
      <path d={pD2} fill="none" stroke={G} strokeWidth="1.1" opacity={op * 0.85} filter="url(#cb-glow)"/>
      <path d={pD3} fill="none" stroke={G} strokeWidth="1.1" opacity={op * 0.85} filter="url(#cb-glow)"/>

      {/* ══ CAPACITORS ══ */}
      {[60,160,280,380].map((x,i) => (
        <g key={i}>
          <line x1={x} y1={748} x2={x} y2={726} stroke={G} strokeWidth="1.0" opacity={op}/>
          <rect x={x-6} y={714} width="12" height="12" rx="6" fill="#061008" stroke={G} strokeWidth="1.0" opacity={op+0.1}/>
          <line x1={x-3} y1={720} x2={x+3} y2={720} stroke={G} strokeWidth="1.3" opacity={op+0.18}/>
        </g>
      ))}

      {/* ══════════════════════════════════════════════════
           ELECTRICITY PULSES — flowing through traces
          ══════════════════════════════════════════════════ */}

      {/* RIGHT VERTICAL BUS — 4 pulses (two down, one up, one spike) */}
      <Pulse path={pRBus}  dur="4.2s" begin="0s"   haloR={9}  coreR={2.5} haloOp={0.60}/>
      <Pulse path={pRBus}  dur="2.6s" begin="1.5s"  haloR={6}  coreR={1.8} haloOp={0.45}/>
      <Pulse path={pRBusR} dur="3.5s" begin="0.8s"  haloR={7}  coreR={2.2} haloOp={0.50}/>
      <Spike path={pRBus}  dur="0.85s" begin="5.0s"/>

      {/* BOTTOM HORIZONTAL BUS — 3 pulses + spike */}
      <Pulse path={pBBot}  dur="5.0s" begin="0.3s"  haloR={9}  coreR={2.4} haloOp={0.55}/>
      <Pulse path={pBBotR} dur="3.8s" begin="2.0s"  haloR={6}  coreR={1.9} haloOp={0.45}/>
      <Pulse path={pBBot}  dur="2.2s" begin="4.1s"  haloR={5}  coreR={1.6} haloOp={0.40}/>
      <Spike path={pBBotR} dur="0.70s" begin="7.2s"/>

      {/* RIGHT BUS BRANCHES */}
      <Pulse path="M432,100 L372,100" dur="1.1s" begin="2.0s" haloR={4}  coreR={1.4} haloOp={0.55}/>
      <Pulse path="M432,248 L372,248" dur="1.3s" begin="0.4s" haloR={4}  coreR={1.4} haloOp={0.50}/>
      <Pulse path="M432,300 L352,300" dur="1.5s" begin="1.7s" haloR={5}  coreR={1.5} haloOp={0.55}/>
      <Pulse path="M432,468 L352,468" dur="1.2s" begin="3.0s" haloR={4}  coreR={1.4} haloOp={0.45}/>
      <Pulse path="M432,520 L392,520" dur="0.95s" begin="0.9s" haloR={4} coreR={1.3} haloOp={0.50}/>
      <Pulse path="M432,624 L352,624" dur="1.4s" begin="2.4s" haloR={4}  coreR={1.4} haloOp={0.45}/>
      <Pulse path="M432,730 L372,730" dur="0.9s" begin="4.2s" haloR={3}  coreR={1.2} haloOp={0.50}/>
      <Spike path="M432,355 L392,355" dur="0.5s" begin="6.0s"/>

      {/* BOTTOM BUS BRANCHES (going up) */}
      <Pulse path="M88,748 L88,706"  dur="0.8s" begin="1.5s" haloR={4}  coreR={1.3} haloOp={0.50}/>
      <Pulse path="M195,748 L195,682" dur="1.0s" begin="3.1s" haloR={4} coreR={1.3} haloOp={0.48}/>
      <Pulse path="M308,748 L308,706" dur="0.9s" begin="0.5s" haloR={4} coreR={1.3} haloOp={0.50}/>
      <Pulse path="M418,748 L418,682" dur="1.1s" begin="2.2s" haloR={4} coreR={1.3} haloOp={0.45}/>

      {/* DIAGONAL TRACES */}
      <Pulse path={pD1} dur="2.2s" begin="1.5s"  haloR={5} coreR={1.5} haloOp={0.50}/>
      <Pulse path={pD2} dur="1.9s" begin="0.7s"  haloR={4} coreR={1.4} haloOp={0.48}/>
      <Pulse path={pD3} dur="2.0s" begin="3.3s"  haloR={5} coreR={1.5} haloOp={0.50}/>
      <Spike path={pD2} dur="0.6s" begin="8.5s"/>

      {/* ══ AMBIENT GREEN EDGE GLOW ══ */}
      <rect x="360" y="0" width="100" height="780" fill={G} opacity="0.04" filter="url(#cb-soft)"/>
      <rect x="0" y="690" width="460" height="90" fill={G} opacity="0.04" filter="url(#cb-soft)"/>
    </svg>
  );
}

// ── SliderCard
function SliderCard({ label, value, min, max, step, onChange, accent, badgeColor, format }: SliderCardProps) {
  const pct = Math.round(((value - min) / (max - min)) * 100);
  const dec = () => onChange(clamp(snapStep(value - step, step, min), min, max));
  const inc = () => onChange(clamp(snapStep(value + step, step, min), min, max));
  return (
    <div className="dm-card">
      <label className="dm-card-label">{label}</label>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="dm-slider"
        style={{ "--acc": accent, "--pct": `${pct}%` } as React.CSSProperties}
      />
      <div className="dm-badge-row">
        <button className="dm-badge-arrow" onClick={dec}>◄</button>
        <div className="dm-badge" style={{ borderColor: `${accent}88`, color: badgeColor }}>
          {format(value)}
        </div>
        <button className="dm-badge-arrow" onClick={inc}>►</button>
      </div>
    </div>
  );
}

// ── Main
export default function DynamoMagician() {
  const [imageSrc,   setImageSrc]   = useState<string | null>(null);
  const [imageName,  setImageName]  = useState("");
  const [imageW,     setImageW]     = useState(0);
  const [imageH,     setImageH]     = useState(0);
  const [prompt,     setPrompt]     = useState("A cinematic aether-trace of the scene, precise lighting, vivid detail.");
  const [resolution, setResolution] = useState("480*832");
  const [width,      setWidth]      = useState(832);
  const [height,     setHeight]     = useState(480);
  const [steps,      setSteps]      = useState(40);
  const [guidance,   setGuidance]   = useState(5.0);
  const [frames,     setFrames]     = useState(81);
  const [seed,       setSeed]       = useState(-1);
  const [cyclesCur,     setCyclesCur]     = useState(0);
  const [vramHigh,      setVramHigh]      = useState(false);
  const [t5CpuOffload,  setT5CpuOffload]  = useState(true);
  const [vaeCpuOffload, setVaeCpuOffload] = useState(false);
  const [previewEvery,  setPreviewEvery]  = useState(0);
  const [isGenerating,  setIsGenerating]  = useState(false);
  const [genCycles,     setGenCycles]     = useState(0);
  const [genTotal,      setGenTotal]      = useState(0);
  const [genStatus,     setGenStatus]     = useState("AETHER-NET STATUS: ACTIVE");
  const [outputUrl,     setOutputUrl]     = useState<string | null>(null);
  const isGeneratingRef = useRef(false);
  const isSubmittingRef = useRef(false);    // true only during the encode+POST step
  const [jobQueue,      setJobQueue]      = useState<QueueEntry[]>([]);
  const [termLog,        setTermLog]        = useState<string[]>([]);
  const termLogScrollRef = useRef<HTMLDivElement>(null);
  const [historyJobs,    setHistoryJobs]    = useState<HistoryEntry[]>([]);
  const [showHistory,    setShowHistory]    = useState(false);
  const [fps,           setFps]           = useState(60);
  const [heapMB,        setHeapMB]        = useState(0);
  const [uptime,        setUptime]        = useState(0);
  const [barPhase,      setBarPhase]      = useState(0);
  const [imageFileSize, setImageFileSize] = useState(0);
  const [imageLuma,     setImageLuma]     = useState(0);
  const [liveMetrics,   setLiveMetrics]   = useState({ cpu: 0, gpu: 0, vram: 0, ram: 0, cpu_temp: 0, gpu_temp: 0, ok: false });
  const fileRef    = useRef<HTMLInputElement>(null);
  const fpsFrames  = useRef(0);
  const fpsLast    = useRef(performance.now());
  const startTime  = useRef(Date.now());
  const stepsRef   = useRef(steps);

  useEffect(() => { stepsRef.current = steps; }, [steps]);

  useEffect(() => {
    let rafId: number;
    const measureFps = (now: number) => {
      fpsFrames.current++;
      if (now - fpsLast.current >= 1000) {
        setFps(Math.min(Math.round(fpsFrames.current * 1000 / (now - fpsLast.current)), 144));
        fpsFrames.current = 0;
        fpsLast.current = now;
        const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
        if (mem) setHeapMB(+(mem.usedJSHeapSize / 1048576).toFixed(1));
      }
      rafId = requestAnimationFrame(measureFps);
    };
    rafId = requestAnimationFrame(measureFps);
    const barId = setInterval(() => setBarPhase(p => (p + 1) % 1000), 140);
    const sysId = setInterval(() => {
      setUptime(Math.floor((Date.now() - startTime.current) / 1000));
      if (!isGeneratingRef.current)
        setCyclesCur(c => (c + 1) % (stepsRef.current + 1));
    }, 1000);
    return () => { cancelAnimationFrame(rafId); clearInterval(barId); clearInterval(sysId); };
  }, []);

  // Live system metrics from backend
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch('/metrics');
        if (!cancelled && res.ok) {
          const d = await res.json();
          setLiveMetrics({ cpu: d.cpu ?? 0, gpu: d.gpu ?? 0, vram: d.vram ?? 0, ram: d.ram ?? 0, cpu_temp: d.cpu_temp ?? 0, gpu_temp: d.gpu_temp ?? 0, ok: true });
        }
      } catch { /* server not running — metrics stay at last known value */ }
      if (!cancelled) metricsTimer = window.setTimeout(poll, 2000);
    };
    let metricsTimer = window.setTimeout(poll, 400);
    return () => { cancelled = true; clearTimeout(metricsTimer); };
  }, []);

  // Job history from backend
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch('/history');
        if (res.ok) setHistoryJobs(await res.json() as HistoryEntry[]);
      } catch { /* server offline */ }
    };
    fetchHistory();
  }, []);

  const pushLog = useCallback((msg: string) => {
    const now = new Date();
    const ts = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
    setTermLog(prev => [...prev.slice(-49), `[${ts}] ${msg}`]);
  }, []);

  // Boot sequence
  useEffect(() => {
    const msgs = [
      "DYNAMO-MAGICIAN  v0.9.1  BOOT SEQUENCE",
      "AETHER-NET SUBSYSTEM.............. OK",
      "QUANTIZATION ENGINE............... OK",
      "MODEL REGISTRY................STANDBY",
      "INFERENCE WORKER.............. STANDBY",
      "AWAITING TRANSMUTATION DIRECTIVE",
    ];
    msgs.forEach((m, i) => setTimeout(() => pushLog(m), i * 320));
  }, [pushLog]);

  // Auto-scroll terminal on new log entries
  useEffect(() => {
    if (termLogScrollRef.current)
      termLogScrollRef.current.scrollTop = termLogScrollRef.current.scrollHeight;
  }, [termLog]);

  const handleFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setImageFileSize(f.size);
    const url = URL.createObjectURL(f);
    setImageSrc(url); setImageName(f.name);
    const img = new Image();
    img.onload = () => {
      setImageW(img.naturalWidth); setImageH(img.naturalHeight);
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const sw = Math.min(img.naturalWidth, 64), sh = Math.min(img.naturalHeight, 64);
        canvas.width = sw; canvas.height = sh;
        ctx.drawImage(img, 0, 0, sw, sh);
        const data = ctx.getImageData(0, 0, sw, sh).data;
        let sum = 0;
        for (let i = 0; i < data.length; i += 4)
          sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        setImageLuma(+(sum / (data.length / 4) / 255 * 100).toFixed(1));
      }
    };
    img.src = url;
    e.target.value = "";
  }, []);

  const handleResolution = useCallback((v: string) => {
    setResolution(v);
    if (v !== "Custom" && v.includes("*")) {
      const [h, w] = v.split("*");
      setHeight(Number(h)); setWidth(Number(w));
    }
  }, []);

  const handleTransmute = useCallback(async () => {
    if (isSubmittingRef.current) return;
    if (!imageSrc) { setGenStatus("ERROR: SELECT AN INPUT IMAGE FIRST"); return; }

    // Encode the loaded image to base64 via a canvas
    const encodeBlob = (): Promise<string> => new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
        canvas.getContext("2d")!.drawImage(img, 0, 0);
        const dataUrl = canvas.toDataURL("image/png");
        resolve(dataUrl.split(",")[1]); // strip data-URI prefix
      };
      img.onerror = reject;
      img.src = imageSrc;
    });

    // Build a small thumbnail for the queue panel
    const encodeThumb = (): Promise<string> => new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const S = 48;
        const canvas = document.createElement("canvas");
        const aspect = img.naturalWidth / img.naturalHeight;
        canvas.width  = aspect >= 1 ? S : Math.round(S * aspect);
        canvas.height = aspect < 1  ? S : Math.round(S / aspect);
        canvas.getContext("2d")!.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.7));
      };
      img.onerror = () => resolve("");
      img.src = imageSrc;
    });

    try {
      isSubmittingRef.current = true;

      pushLog("▶ TRANSMUTATION DIRECTIVE RECEIVED");
      pushLog(`  INPUT : ${imageName || "unknown"} (${(imageFileSize / 1024).toFixed(1)} KB)`);
      pushLog(`  PARAMS: steps=${steps}  frames=${frames}  cfg=${guidance}  ${width}×${height}`);
      pushLog(`  VRAM  : ${vramHigh ? "HIGH" : "LOW"} mode  |  T5: ${t5CpuOffload ? "CPU" : "GPU"}  |  VAE-offload: ${vaeCpuOffload ? "ON" : "OFF"}`);
      pushLog("  ENCODING INPUT IMAGE...");

      const [image_b64, thumbSrc] = await Promise.all([encodeBlob(), encodeThumb()]);

      pushLog(`  ENCODE: OK  (${(image_b64.length / 1024).toFixed(0)} KB base64)`);
      pushLog("  CONNECTING TO INFERENCE SERVER...");

      const res = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_b64,
          prompt,
          width,
          height,
          steps,
          guidance,
          frames,
          seed,
          vram_mode: vramHigh ? "high" : "low",
          t5_cpu: t5CpuOffload,
          vae_cpu_offload: vaeCpuOffload,
          preview_every: previewEvery,
        }),
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const { job_id, position } = await res.json() as { job_id: string; position: number };

      pushLog(`  SERVER ACK  →  JOB: ${job_id.slice(0, 8).toUpperCase()}  POS: ${position}`);

      // Add to queue display
      const newEntry: QueueEntry = {
        job_id,
        promptPreview: prompt.slice(0, 60) + (prompt.length > 60 ? "…" : ""),
        imageThumb: thumbSrc,
        status: position === 1 ? 'running' : 'queued',
        position,
        progress: { current: 0, total: steps },
        outputUrl: null,
        errorMsg: null,
        previewUrl: null,
      };
      setJobQueue(prev => [...prev, newEntry]);

      // If this is the first/only job, update the legacy status indicators too
      if (position === 1) {
        isGeneratingRef.current = true;
        setIsGenerating(true);
        setOutputUrl(null);
        setGenCycles(0); setGenTotal(0);
        setGenStatus(`AETHER-NET: JOB QUEUED — POS ${position}`);
      } else {
        setGenStatus(`AETHER-NET: JOB QUEUED — POS ${position}`);
        pushLog(`  QUEUED AT POSITION ${position} — WAITING FOR ACTIVE JOB TO COMPLETE`);
      }

      // Open SSE stream and drive queue entry updates
      const es = new EventSource(`/stream/${job_id}`);

      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as {
            type: string; message?: string; current?: number;
            total?: number; output?: string; position?: number;
            image?: string; step?: number;
          };

          setJobQueue(prev => prev.map(e => {
            if (e.job_id !== job_id) return e;
            if (msg.type === "queued") {
              return { ...e, status: 'queued', position: msg.position ?? e.position };
            }
            if (msg.type === "status") {
              const running = msg.message?.toLowerCase().includes("decoding") ||
                              msg.message?.toLowerCase().includes("initialising") ||
                              msg.message?.toLowerCase().includes("inference");
              return { ...e, status: running ? 'running' : e.status, position: 0 };
            }
            if (msg.type === "progress") {
              const cur = msg.current ?? 0, tot = msg.total ?? steps;
              return { ...e, status: 'running', progress: { current: cur, total: tot } };
            }
            if (msg.type === "preview") {
              return { ...e, previewUrl: msg.image ?? e.previewUrl };
            }
            if (msg.type === "complete") {
              return { ...e, status: 'done', outputUrl: msg.output ?? null, progress: { ...e.progress, current: e.progress.total } };
            }
            if (msg.type === "error") {
              return { ...e, status: 'error', errorMsg: msg.message ?? "unknown error" };
            }
            return e;
          }));

          // Legacy status/progress display (tracks the most recently active job)
          if (msg.type === "status") {
            setGenStatus(`AETHER-NET: ${msg.message ?? ""}`);
            pushLog(`[PROC] ${msg.message ?? ""}`);
          } else if (msg.type === "progress") {
            const cur = msg.current ?? 0, tot = msg.total ?? steps;
            setGenCycles(cur); setGenTotal(tot);
            setCyclesCur(cur);
            setGenStatus(`AETHER-FLUX: cycle ${cur}/${tot}`);
            const logEvery = Math.max(1, Math.ceil(tot / 8));
            if (cur === 1 || cur % logEvery === 0 || cur === tot) {
              const filled = Math.round((cur / tot) * 10);
              pushLog(`[STEP] ${String(cur).padStart(3)}/${tot}  ${"█".repeat(filled)}${"░".repeat(10 - filled)}  ${Math.round((cur / tot) * 100)}%`);
            }
          } else if (msg.type === "complete") {
            setGenStatus("AETHER-NET: TRANSMUTATION COMPLETE");
            setOutputUrl(msg.output ?? "");
            pushLog("[DONE] DIFFUSION COMPLETE — ENCODING OUTPUT");
            pushLog(`[FILE] OUTPUT READY → ${msg.output ?? ""}`);
            pushLog("▶ TRANSMUTATION SEQUENCE FINISHED ◀");
            es.close();
            isGeneratingRef.current = false;
            setIsGenerating(false);
            // Refresh history panel
            fetch('/history').then(r => r.ok ? r.json() : []).then((data: HistoryEntry[]) => setHistoryJobs(data)).catch(() => {});
          } else if (msg.type === "error") {
            setGenStatus(`ERROR: ${msg.message ?? "unknown"}`);
            pushLog(`!ERR  ${msg.message ?? "unknown error"}`);
            es.close();
            isGeneratingRef.current = false;
            setIsGenerating(false);
          }
        } catch { /* ignore malformed */ }
      };

      es.onerror = () => {
        setGenStatus("ERROR: Lost connection to server");
        pushLog("!ERR  SSE CONNECTION LOST — CHECK SERVER");
        es.close();
        setJobQueue(prev => prev.map(e =>
          e.job_id === job_id && e.status !== 'done'
            ? { ...e, status: 'error', errorMsg: 'SSE connection lost' }
            : e
        ));
        isGeneratingRef.current = false;
        setIsGenerating(false);
      };

    } catch (err) {
      setGenStatus(`ERROR: ${(err as Error).message}`);
      pushLog(`!FATAL  ${(err as Error).message}`);
      isGeneratingRef.current = false;
      setIsGenerating(false);
    } finally {
      isSubmittingRef.current = false;
    }
  }, [imageSrc, imageName, imageFileSize, prompt, width, height, steps, guidance, frames, seed, vramHigh, t5CpuOffload, vaeCpuOffload, previewEvery, pushLog]);

  const cancelJob = useCallback(async (job_id: string) => {
    try {
      await fetch(`/queue/${job_id}`, { method: "DELETE" });
    } catch { /* ignore */ }
    setJobQueue(prev => prev.map(e =>
      e.job_id === job_id ? { ...e, status: 'error', errorMsg: 'Cancelled' } : e
    ));
  }, []);

  const stem        = imageName.replace(/\.[^.]+$/, "").slice(0, 14).toUpperCase();
  const progress    = isGenerating && genTotal > 0
    ? (genCycles / genTotal) * 100
    : (cyclesCur / Math.max(steps, 1)) * 100;
  const indeterminate = isGenerating && genTotal === 0;
  // 0=DATA INGEST  1=COGNITIVE PROCESSING  2=AETHER-FLUX  -1=idle
  const activeStep = !isGenerating ? -1
    : genStatus.includes("CONNECTING") || genStatus.includes("Decoding") ? 0
    : genStatus.includes("Encoding") || genStatus.includes("COMPLETE") ? 2
    : 1;
  const entropy     = shannonEntropy(prompt).toFixed(4);
  const teraflops   = (1.0 + (fps / 60) * (steps / 40) * (guidance / 5.0) * 1.8).toFixed(3);
  const vramGb      = ((width * height * frames * 12 + 8 * 1073741824) / 1073741824).toFixed(2);
  const megapixels  = (width * height / 1e6).toFixed(2);
  const tokenEst    = Math.round(prompt.length / 4);
  const paramHash   = (((seed < 0 ? steps * 9973 : seed * 31337) + frames * 7919 + width * 3137) >>> 0);
  const paramHex    = paramHash.toString(16).padStart(8, '0').toUpperCase();
  const syncStatus  = fps >= 55 ? "LOCK" : fps >= 45 ? "SYNC" : "DRIFT";
  const lumaDisplay = imageLuma > 0 ? imageLuma.toFixed(1) : (steps * guidance / (100 * 15) * 84.2 * 2).toFixed(1);
  const fileSizeKb  = imageFileSize > 0 ? (imageFileSize / 1024).toFixed(1) : "0.0";
  const sessionHMS  = secsToHMS(uptime);
  const heapDisplay = heapMB > 0 ? heapMB.toFixed(1) : "—";

  // Simulated system utilisation (driven by barPhase + load proxy) — used as fallback
  const loadProxy  = Math.min((steps / 40) * (guidance / 5.0) * (frames / 81), 1.5);
  const cpuPct     = liveMetrics.ok ? liveMetrics.cpu  : Math.round(18 + Math.abs(Math.sin(barPhase * 0.031 + 1.1)) * 55 * Math.min(loadProxy, 1));
  const gpuPct     = liveMetrics.ok ? liveMetrics.gpu  : Math.round(32 + Math.abs(Math.sin(barPhase * 0.027 + 2.3)) * 60 * Math.min(loadProxy, 1));
  const vramPct    = liveMetrics.ok ? liveMetrics.vram : Math.round(Math.min(parseFloat(vramGb) / (vramHigh ? 48 : 24) * 100, 99));
  const sramPct    = liveMetrics.ok ? liveMetrics.ram  : Math.round(20 + Math.abs(Math.sin(barPhase * 0.019 + 0.7)) * 35 * Math.min(loadProxy, 1));
  const cpuTempPct  = liveMetrics.ok ? liveMetrics.cpu_temp : 45;
  const gpuTempPct  = liveMetrics.ok ? liveMetrics.gpu_temp : 45;

  const chips: Chip[] = imageSrc ? [
    { lines: ["SUBJECT LOCK", `ID:${paramHex.slice(0, 4)}`],  style: { top: "8%",     left: "7%"  } },
    { lines: ["SOURCE FILE",  `${imageW}×${imageH}`],         style: { top: "8%",     right: "5%" } },
    { lines: [stem,           `${fileSizeKb}KB`],             style: { bottom: "27%", left: "7%"  } },
  ] : [
    { lines: ["VRAM EST.",   `${vramGb}GB`],                  style: { top: "22%",    left: "7%"  } },
    { lines: ["MEGAPIXELS",  `${megapixels}MP`],              style: { top: "28%",    left: "38%" } },
    { lines: ["TOKENS EST.", `${tokenEst}`],                  style: { top: "52%",    right: "9%" } },
    { lines: ["HEAP",        `${heapDisplay}MB`],             style: { bottom: "33%", left: "7%"  } },
  ];

  const bayStatus = imageSrc
    ? `${imageW}×${imageH} · ${fileSizeKb}KB · LUMA:${lumaDisplay}% · LINK STABLE`
    : `FPS:${fps} · ENTROPY:${entropy} · UPTIME:${sessionHMS} · BAY ONLINE`;

  return (
    <div className="dm-bezel">
      <Screw className="dm-screw-tl" />
      <Screw className="dm-screw-tr" />
      <Screw className="dm-screw-bl" />
      <Screw className="dm-screw-br" />
      <div className="dm-scanline-overlay" />

      <div className="dm-root">

        {/* HEADER */}
        <header className="dm-header">
          <div className="dm-header-left">
            <svg className="dm-sigil" width="38" height="38" viewBox="0 0 38 38">
              <polygon points="19,2 35,11 35,27 19,36 3,27 3,11"
                fill="none" stroke={MAG} strokeWidth="1.5"
                style={{ filter: `drop-shadow(0 0 6px ${MAG})` }}/>
              <polygon points="19,9 29,15 29,23 19,29 9,23 9,15"
                fill="none" stroke={CYAN} strokeWidth="1"
                style={{ filter: `drop-shadow(0 0 3px ${CYAN})` }}/>
              <circle cx="19" cy="19" r="4.5" fill={VIO} opacity="0.85"
                style={{ filter: `drop-shadow(0 0 8px ${VIO})` }}/>
              <circle cx="19" cy="19" r="1.5" fill="white" opacity="0.6"/>
            </svg>
            <div className="dm-title-group">
              <h1 className="dm-title">DYNAMO MAGICIAN</h1>
              <span className="dm-title-sub">AETHER-NET v2.47 // CLASS-7 THAUMIC ENGINE</span>
            </div>
          </div>

          <div className="dm-hdr-readouts">
            <div className="dm-readout">
              <span className="dm-readout-lbl">TFLOPS</span>
              <span className="dm-readout-val">{teraflops}</span>
            </div>
            <div className="dm-readout">
              <span className="dm-readout-lbl">ENTROPY</span>
              <span className="dm-readout-val">{entropy}</span>
            </div>
            <div className="dm-readout">
              <span className="dm-readout-lbl">SYNC</span>
              <span className={`dm-readout-val ${syncStatus === "LOCK" ? "dm-readout-ok" : syncStatus === "DRIFT" ? "dm-readout-err" : "dm-readout-warn"}`}>
                {syncStatus}
              </span>
            </div>
          </div>

          <div className="dm-hdr-bars">
            {Array.from({ length: 14 }, (_, i) => {
              const load = Math.min((steps / 40) * (guidance / 5.0) * (frames / 81), 2);
              const h = 15 + Math.abs(Math.sin(barPhase * 0.05 + i * 0.7 + fps * 0.02)) * 70 * Math.min(load, 1.2);
              return (
                <div key={i} className="dm-hdr-bar"
                  style={{
                    background: i % 3 === 0 ? CYAN : i % 3 === 1 ? MAG : VIO,
                    opacity: 0.25 + Math.abs(Math.sin(barPhase * 0.04 + i * 0.9)) * 0.65,
                    height: `${h}%`
                  }}/>
              );
            })}
          </div>

          <span className="dm-hdr-meta">{sessionHMS} · FPS:{fps} · {heapMB > 0 ? `${heapDisplay}MB` : "MEM:—"} · 0x{paramHex.slice(0, 4)}</span>
          <div className="dm-hdr-icons">
            <button className="dm-hdr-btn" title="Signal"><IconWave /></button>
            <button className="dm-hdr-btn" title="Auth"><IconFingerprint /></button>
            <button className="dm-hdr-btn" title="Stop"><IconXCircle /></button>
            <button className="dm-hdr-btn" title="Minimise"><IconMinus /></button>
            <button className="dm-hdr-btn" title="Close"><IconXClose /></button>
          </div>
        </header>

        {/* MAIN */}
        <div className="dm-main">

          {/* LEFT BAY */}
          <div className="dm-bay-outer">
            <div className="dm-bay-inner">
              <div className="dm-bay-topbar">
                <span className="dm-bay-id">BAY-01 // OPTICAL INPUT // {imageSrc ? `${imageW}×${imageH} // ${fileSizeKb}KB` : `UPTIME:${sessionHMS}`}</span>
                <span className="dm-rec">&#11044; REC</span>
              </div>

              <button className="dm-select-btn" onClick={() => fileRef.current?.click()}>
                &#8853;  INJECT SOURCE  —  Step 1: Select Input Image
              </button>
              <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFile} />

              <div className="dm-preview-frame">
                {imageSrc ? (
                  <img src={imageSrc} alt="preview" className="dm-preview-img" />
                ) : (
                  <div className="dm-preview-empty">
                    <svg width="64" height="64" viewBox="0 0 64 64" opacity="0.4">
                      <polygon points="32,4 58,18 58,46 32,60 6,46 6,18"
                        fill="none" stroke={CYAN} strokeWidth="1.5"/>
                      <polygon points="32,16 48,25 48,43 32,52 16,43 16,25"
                        fill="none" stroke={VIO} strokeWidth="1" opacity="0.6"/>
                      <circle cx="32" cy="32" r="9" fill="none" stroke={MAG} strokeWidth="1.5"/>
                      <circle cx="32" cy="32" r="2.5" fill={MAG} opacity="0.7"/>
                      <line x1="32" y1="4"  x2="32" y2="20" stroke={CYAN} strokeWidth="0.8"/>
                      <line x1="32" y1="44" x2="32" y2="60" stroke={CYAN} strokeWidth="0.8"/>
                      <line x1="6"  y1="18" x2="21" y2="27" stroke={CYAN} strokeWidth="0.8"/>
                      <line x1="43" y1="37" x2="58" y2="46" stroke={CYAN} strokeWidth="0.8"/>
                    </svg>
                    <span className="dm-empty-title">NO IMAGE SELECTED</span>
                    <span className="dm-empty-sub">Visual bay ready for source acquisition</span>
                    <span className="dm-empty-code">AETHER-SPEC: AWAITING INPUT // 0x0000</span>
                  </div>
                )}

                <div className="dm-hud-scan" />
                <div className="dm-corner dm-corner-tl" />
                <div className="dm-corner dm-corner-tr" />
                <div className="dm-corner dm-corner-bl" />
                <div className="dm-corner dm-corner-br" />
                <div className="dm-scan-top" />
                <div className="dm-scan-btm" />

                <div className="dm-hud-badge">
                  <div>SPEC: 0x{paramHex.slice(0, 4)}</div>
                  <div>LUMA: {imageSrc ? `${lumaDisplay}%` : "N/A"}</div>
                  <div>ENTROPY: {entropy}</div>
                </div>

                {chips.map((chip, i) => (
                  <div key={i} className="dm-chip" style={chip.style}>
                    {chip.lines.map((line, j) => <div key={j}>{line}</div>)}
                  </div>
                ))}

                <div className="dm-bay-status">
                  <span className="dm-bay-status-text">{bayStatus}</span>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT PANEL */}
          <div className="dm-panel">
            <CircuitBg />
            <div className="dm-panel-id">
              PARAM-MODULE // v9.1 // THAUMIC-ENCODER // REF:0xE2C4
            </div>
            <div className="dm-panel-title">
              <div className="dm-panel-title-inner">&#11042; Parameters &#11042;</div>
            </div>

            <div className="dm-panel-body">
              <div className="dm-card">
                <label className="dm-card-label">Prompt</label>
                <textarea className="dm-textarea" value={prompt}
                  onChange={e => setPrompt(e.target.value)} rows={3} />
              </div>

              <div className="dm-card">
                <label className="dm-card-label">Resolution Preset</label>
                <div className="dm-select-wrap">
                  <select className="dm-select" value={resolution}
                    onChange={e => handleResolution(e.target.value)}>
                    {["480*832","832*480","576*1024","1024*576","Custom"].map(v => (
                      <option key={v} value={v}>{v}</option>
                    ))}
                  </select>
                  <span className="dm-select-arrow">&#9660;</span>
                </div>
                <div className="dm-dims-row">
                  <span className="dm-dim-label">W.</span>
                  <input type="number" className="dm-dim-inp" value={width}
                    onChange={e => setWidth(Number(e.target.value))} />
                  <span className="dm-dim-label">H.</span>
                  <input type="number" className="dm-dim-inp" value={height}
                    onChange={e => setHeight(Number(e.target.value))} />
                </div>
              </div>

              <SliderCard label="Sampling Steps"
                value={steps} min={10} max={100} step={1} onChange={setSteps}
                accent={MAG} badgeColor={RED_VAL} format={v => v.toFixed(2)} />

              <SliderCard label="Guidance Scale"
                value={guidance} min={1.0} max={15.0} step={0.1} onChange={setGuidance}
                accent={CYAN} badgeColor={TEXT} format={v => v.toFixed(1)} />

              <SliderCard label="Frame Count"
                value={frames} min={1} max={241} step={4} onChange={setFrames}
                accent={CYAN} badgeColor={TEXT} format={v => v.toFixed(0)} />

              <div className="dm-card">
                <label className="dm-card-label">Seed (-1 for random)</label>
                <div className="dm-entry-row">
                  <input type="number" className="dm-entry" value={seed}
                    onChange={e => setSeed(Number(e.target.value))} />
                  <div className="dm-energy-track">
                    <div className="dm-energy-fill"
                      style={{ width: "72%", background: `linear-gradient(90deg,#7b2fff,${MAG})` }} />
                    <div className="dm-energy-sparks" />
                  </div>
                </div>
              </div>

              <div className="dm-card">
                <label className="dm-card-label">Camera Control</label>
                <div className="dm-entry-row">
                  <span className="dm-cam-lbl">No camera path selected</span>
                  <button className="dm-link-btn">Link Folder</button>
                </div>
              </div>

              <div className="dm-vram-row">
                <label className="dm-toggle">
                  <input type="checkbox" className="sr-only" checked={vramHigh}
                    onChange={e => setVramHigh(e.target.checked)} />
                  <span className="dm-toggle-track" style={{ background: vramHigh ? MAG : "#2d3342" }}>
                    <span className="dm-toggle-thumb" style={{ transform: vramHigh ? "translateX(18px)" : "none" }} />
                  </span>
                </label>
                <span className="dm-vram-label">High VRAM Mode (48GB+)</span>
              </div>
              <div className="dm-vram-row" style={{ marginTop: 8 }}>
                <label className="dm-toggle">
                  <input type="checkbox" className="sr-only" checked={t5CpuOffload}
                    onChange={e => setT5CpuOffload(e.target.checked)} />
                  <span className="dm-toggle-track" style={{ background: t5CpuOffload ? CYAN_SOFT : "#2d3342" }}>
                    <span className="dm-toggle-thumb" style={{ transform: t5CpuOffload ? "translateX(18px)" : "none" }} />
                  </span>
                </label>
                <span className="dm-vram-label">T5 CPU Offload (saves VRAM)</span>
              </div>
              <div className="dm-vram-row" style={{ marginTop: 8 }}>
                <label className="dm-toggle">
                  <input type="checkbox" className="sr-only" checked={vaeCpuOffload}
                    onChange={e => setVaeCpuOffload(e.target.checked)} />
                  <span className="dm-toggle-track" style={{ background: vaeCpuOffload ? CYAN_SOFT : "#2d3342" }}>
                    <span className="dm-toggle-thumb" style={{ transform: vaeCpuOffload ? "translateX(18px)" : "none" }} />
                  </span>
                </label>
                <span className="dm-vram-label">VAE CPU Offload (saves ~1.3 GB)</span>
              </div>

              <div className="dm-vram-row">
                <span className="dm-vram-label">Step Preview</span>
                <div className="dm-select-wrap" style={{ flex: 1, minWidth: 0 }}>
                  <select className="dm-select dm-select--sm"
                    value={previewEvery}
                    onChange={e => setPreviewEvery(Number(e.target.value))}
                    disabled={vaeCpuOffload}
                    title={vaeCpuOffload ? "Disable VAE CPU Offload to enable previews" : "Decode a preview frame every N diffusion steps"}
                  >
                    <option value={0}>Off</option>
                    <option value={3}>Every 3 steps</option>
                    <option value={5}>Every 5 steps</option>
                    <option value={10}>Every 10 steps</option>
                  </select>
                  <span className="dm-select-arrow">&#9660;</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* FOOTER */}
        <footer className="dm-footer">
          <LightningProgressBar pct={progress} phase={barPhase} indeterminate={indeterminate} />

          <div className="dm-footer-body">
            <div className="dm-flow-panel">
              <div className="dm-flow-steps" style={{ justifyContent: "flex-start" }}>
                {[
                  { icon: <IconDB />,    lbl: "DATA INGEST" },
                  { icon: <IconBrain />, lbl: "COGNITIVE PROCESSING" },
                  { icon: <IconFlux />,  lbl: "AETHER-FLUX" },
                ].map(({ icon, lbl }, i) => (
                  <React.Fragment key={i}>
                    {i > 0 && (
                      <div className="dm-flow-arrow">
                        <svg width="38" height="12" viewBox="0 0 38 12" fill="none">
                          <line x1="0" y1="6" x2="31" y2="6" stroke="#4fb4dc" strokeWidth="1.2" opacity="0.65"/>
                          <polyline points="27,2 33,6 27,10" fill="none" stroke="#4fb4dc" strokeWidth="1.4" strokeLinejoin="round" opacity="0.85"/>
                        </svg>
                      </div>
                    )}
                    <div className={`dm-flow-step${activeStep === i ? " dm-step-active" : ""}`}>
                      <div className="dm-flow-icon">{icon}</div>
                      <span className="dm-flow-lbl">{lbl}</span>
                    </div>
                  </React.Fragment>
                ))}
              </div>
              <div className="dm-flow-status">
                <div className={`dm-status-primary${isGenerating ? " dm-status-active" : ""}`}>
                  {isGenerating && <span className="dm-live-dot" />}
                  {genStatus}
                </div>
                <div className="dm-status-sub">MODEL: Dynamo_Mage_v0.9.1  //  FPS:{fps}  //  {sessionHMS}</div>
              </div>
              {/* CRT PROCESS LOG TERMINAL */}
              <div className="dm-crt-terminal">
                <div className="dm-crt-header">
                  <span className="dm-crt-title">SYS▸LOG</span>
                  <span className="dm-crt-subtitle"> // PROCESS MONITOR</span>
                  <span className="dm-crt-blink">▮</span>
                </div>
                <div className="dm-crt-body" ref={termLogScrollRef}>
                  {termLog.map((line, i) => (
                    <div key={i} className="dm-crt-line">{line}</div>
                  ))}
                  <span className="dm-crt-cursor">▮</span>
                </div>
              </div>
            </div>

            {/* SYSMON GAUGES */}
            <div className="dm-sysmon-panel">
              <div className="dm-sysmon-header">
                <span className="dm-sysmon-title">⬡ SYSTEM VITALS</span>
                <span className={`dm-sysmon-live${liveMetrics.ok ? " dm-sysmon-live-on" : ""}`}>
                  {liveMetrics.ok ? "● LIVE" : "○ SIM"}
                </span>
              </div>
              <div className="dm-gauge-grid">
                <GaugeMeter label="CPU"  pct={cpuPct}  color={CYAN}/>
                <GaugeMeter label="GPU"  pct={gpuPct}  color={MAG}/>
                <GaugeMeter label="VRAM" pct={vramPct} color={VIO}/>
                <GaugeMeter label="RAM"  pct={sramPct} color={CYAN}/>
                <GaugeMeter label="CPU°C" pct={cpuTempPct} color={CYAN} unit="°C"/>
                <GaugeMeter label="GPU°C" pct={gpuTempPct} color={MAG}  unit="°C"/>
              </div>
            </div>

            <div className="dm-transmute-panel">
              {/* Step hint — only shown when idle */}
              {!isGenerating && (
                <div className="dm-initiate-badge">
                  <span className="dm-initiate-badge-dot" />
                  STEP 3 · CLICK TO INITIATE
                </div>
              )}
              <div className="dm-transmute-row">
                {/* Pulsing arrow hint — hidden while generating */}
                {!isGenerating && <div className="dm-initiate-arrow">▶▶</div>}
                <div style={{ position: "relative" }}>
                  <button
                    className={`dm-transmute-btn${isGenerating ? " dm-transmute-active" : ""}`}
                    onClick={handleTransmute}
                    title="Queue a new generation job"
                  >
                    <TransmutePortal active={isGenerating} />
                    <span className="dm-transmute-lbl">
                      {isGenerating ? `GENERATING… (${jobQueue.filter(e => e.status === 'queued').length} queued)` : "GENERATE WORLD MODEL"}
                    </span>
                  </button>
                  {/* Beacon dot — top-right corner, hidden while generating */}
                  {!isGenerating && <span className="dm-beacon" />}
                </div>
                {/* Cable bundle — decorative connectors on right edge */}
                <svg className="dm-cable-bundle" viewBox="0 0 80 98" preserveAspectRatio="none">
                  {[15, 45, 75].map((y, i) => {
                    const bend = i === 0 ? -24 : i === 1 ? 2 : 26;
                    return (
                      <g key={i}>
                        {/* Connector housing */}
                        <rect x="0" y={y - 8} width="20" height="16" rx="2.5"
                          fill="#14182a" stroke="#2a3860" strokeWidth="0.9"/>
                        <rect x="1.5" y={y - 6} width="12" height="12" rx="1.5" fill="#0a0c1a"/>
                        {[0, 3.5, 7].map(d => (
                          <rect key={d} x="3" y={y - 4.5 + d} width="8" height="2" rx="0.8" fill="#243068"/>
                        ))}
                        {/* Rubber strain-relief ridge */}
                        <rect x="18" y={y - 5} width="4" height="10" rx="2" fill="#0e1020" stroke="#222840" strokeWidth="0.7"/>
                        {/* Cable body — three-layer depth */}
                        <path d={`M22,${y} C44,${y} 55,${y + bend} 80,${y + bend}`}
                          fill="none" stroke="#0a0a1c" strokeWidth={i === 1 ? 12 : 9} strokeLinecap="round"/>
                        <path d={`M22,${y} C44,${y} 55,${y + bend} 80,${y + bend}`}
                          fill="none" stroke="#16182e" strokeWidth={i === 1 ? 7 : 5.5} strokeLinecap="round"/>
                        <path d={`M22,${y} C44,${y} 55,${y + bend} 80,${y + bend}`}
                          fill="none" stroke="#222446" strokeWidth="1.8" strokeLinecap="round"/>
                      </g>
                    );
                  })}
                </svg>
              </div>
              <div className="dm-cycles">
                CYCLES: {isGenerating ? `${genCycles}/${genTotal || steps}` : `${cyclesCur}/${steps}`}
              </div>
              {outputUrl && (
                <a className="dm-output-link" href={outputUrl} download target="_blank" rel="noreferrer">
                  &#11015; DOWNLOAD OUTPUT
                </a>
              )}

              {/* ─── JOB QUEUE PANEL ─── */}
              {jobQueue.length > 0 && (
                <div className="dm-queue-panel">
                  <div className="dm-queue-header">
                    <span className="dm-queue-title">&#11041; JOB QUEUE</span>
                    <span className="dm-queue-count">{jobQueue.length}</span>
                  </div>
                  <div className="dm-queue-list">
                    {jobQueue.map(entry => (
                      <div key={entry.job_id} className={`dm-queue-item dm-queue-item--${entry.status}`}>
                        <div className="dm-queue-thumb-wrap">
                          {entry.previewUrl ? (
                            <img src={entry.previewUrl} className="dm-queue-thumb dm-queue-thumb--preview" alt="live preview" title="Live diffusion preview" />
                          ) : entry.imageThumb ? (
                            <img src={entry.imageThumb} className="dm-queue-thumb" alt="" />
                          ) : null}
                          {entry.status === 'running' && entry.previewUrl && (
                            <span className="dm-preview-badge">LIVE</span>
                          )}
                        </div>
                        <div className="dm-queue-info">
                          <span className="dm-queue-badge">
                            {entry.status === 'queued'  ? `#${entry.position} QUEUED` :
                             entry.status === 'running' ? 'RUNNING' :
                             entry.status === 'done'    ? 'DONE' : 'ERR'}
                          </span>
                          <span className="dm-queue-prompt">{entry.promptPreview}</span>
                          {entry.status === 'running' && (
                            <div className="dm-queue-prog-wrap">
                              <div
                                className="dm-queue-prog-bar"
                                style={{ width: `${Math.round((entry.progress.current / Math.max(entry.progress.total, 1)) * 100)}%` }}
                              />
                              <span className="dm-queue-prog-txt">
                                {entry.progress.current}/{entry.progress.total}
                              </span>
                            </div>
                          )}
                          {entry.status === 'error' && entry.errorMsg && (
                            <span className="dm-queue-errmsg">{entry.errorMsg}</span>
                          )}
                        </div>
                        <div className="dm-queue-actions">
                          {entry.outputUrl && (
                            <a href={entry.outputUrl} download target="_blank" rel="noreferrer"
                               className="dm-queue-dl" title="Download output">&#11015;</a>
                          )}
                          {entry.status === 'queued' && (
                            <button
                              className="dm-queue-cancel"
                              title="Cancel job"
                              onClick={() => cancelJob(entry.job_id)}
                            >&#x2715;</button>
                          )}
                          {(entry.status === 'done' || entry.status === 'error') && (
                            <button
                              className="dm-queue-dismiss"
                              title="Dismiss"
                              onClick={() => setJobQueue(prev => prev.filter(e => e.job_id !== entry.job_id))}
                            >&#x2715;</button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="dm-new-slot" />
            </div>

            {/* ─── HISTORY GALLERY ─── */}
            {historyJobs.length > 0 && (
              <div className="dm-history-panel">
                <button className="dm-history-header" onClick={() => setShowHistory(v => !v)}>
                  <span className="dm-queue-title">&#9881; GENERATION HISTORY</span>
                  <span className="dm-queue-count">{historyJobs.length}</span>
                  <span className="dm-history-toggle">{showHistory ? "▲" : "▼"}</span>
                </button>
                {showHistory && (
                  <div className="dm-history-grid">
                    {historyJobs.map(h => (
                      <a
                        key={h.job_id}
                        className="dm-history-item"
                        href={`/output/${h.job_id}`}
                        download
                        target="_blank"
                        rel="noreferrer"
                        title={`${h.prompt}\n${h.params.width}×${h.params.height} · ${h.params.frames}fr · ${h.duration_secs}s`}
                      >
                        {h.thumbnail_file ? (
                          <img
                            src={`/thumbnail/${h.job_id}`}
                            className="dm-history-thumb"
                            alt="thumbnail"
                            loading="lazy"
                          />
                        ) : (
                          <div className="dm-history-thumb dm-history-thumb--empty">&#9651;</div>
                        )}
                        <div className="dm-history-info">
                          <span className="dm-history-prompt">{h.prompt.slice(0, 48)}{h.prompt.length > 48 ? "…" : ""}</span>
                          <span className="dm-history-meta">{h.params.width}×{h.params.height} · {h.params.frames}fr · {Math.floor(h.duration_secs)}s</span>
                          <span className="dm-history-meta">{new Date(h.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                        </div>
                      </a>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="dm-thumb-panel">
              {imageSrc ? (
                <img src={imageSrc} alt="thumb" className="dm-thumb-img" />
              ) : (
                <div className="dm-thumb-city">
                  <svg viewBox="0 0 150 90" width="100%" height="100%" preserveAspectRatio="xMidYMid slice">
                    <defs>
                      <linearGradient id="cyGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%"   stopColor="#02020e"/>
                        <stop offset="50%"  stopColor="#080524"/>
                        <stop offset="85%"  stopColor="#0e0535"/>
                        <stop offset="100%" stopColor="#160840"/>
                      </linearGradient>
                      <radialGradient id="cyHaze" cx="50%" cy="38%" r="55%">
                        <stop offset="0%"   stopColor="#1808a0" stopOpacity="0.35"/>
                        <stop offset="100%" stopColor="#000" stopOpacity="0"/>
                      </radialGradient>
                      <radialGradient id="cyGlow1" cx="30%" cy="55%" r="30%">
                        <stop offset="0%"   stopColor="#c020ff" stopOpacity="0.22"/>
                        <stop offset="100%" stopColor="#000" stopOpacity="0"/>
                      </radialGradient>
                      <radialGradient id="cyGlow2" cx="72%" cy="60%" r="28%">
                        <stop offset="0%"   stopColor="#8020ff" stopOpacity="0.18"/>
                        <stop offset="100%" stopColor="#000" stopOpacity="0"/>
                      </radialGradient>
                    </defs>
                    {/* Sky */}
                    <rect width="150" height="90" fill="url(#cyGrad)"/>
                    <rect width="150" height="90" fill="url(#cyHaze)"/>
                    <rect width="150" height="90" fill="url(#cyGlow1)"/>
                    <rect width="150" height="90" fill="url(#cyGlow2)"/>
                    {/* Background towers */}
                    {[[4,40,11,50],[16,32,9,52],[27,46,8,44],[38,28,13,58],[54,36,9,50],[66,30,11,56],[80,44,8,46],[90,24,15,62],[108,36,9,50],[120,40,8,48],[130,26,12,56],[143,38,7,48]].map(([x,h,w,y],i) => (
                      <rect key={`bt-${i}`} x={x} y={y-h} width={w} height={h}
                        fill={`rgba(${7+i},${5+i},${18+i*3},1)`}/>
                    ))}
                    {/* Vertical neon edge strips */}
                    <line x1="38" y1="20" x2="38" y2="58" stroke="#f040ff" strokeWidth="1.4" opacity="0.55"/>
                    <line x1="51" y1="18" x2="51" y2="50" stroke="#ff3090" strokeWidth="0.9" opacity="0.45"/>
                    <line x1="90" y1="14" x2="90" y2="62" stroke="#e030ff" strokeWidth="1.2" opacity="0.5"/>
                    <line x1="108" y1="20" x2="108" y2="50" stroke="#4080ff" strokeWidth="0.9" opacity="0.4"/>
                    <line x1="130" y1="18" x2="130" y2="56" stroke="#c030ff" strokeWidth="1" opacity="0.45"/>
                    {/* Foreground buildings */}
                    {[[0,28,9,90],[10,20,16,90],[28,34,18,90],[50,24,14,90],[66,30,18,90],[88,26,17,90],[108,22,15,90],[126,28,24,90]].map(([x,h,w,y],i) => (
                      <rect key={`ff-${i}`} x={x} y={y-h} width={w} height={h}
                        fill={`rgba(${3+i},${2+i},${12+i*2},1)`}/>
                    ))}
                    {/* Neon windows — pink, cyan, violet */}
                    {[[12,72],[16,68],[20,74],[32,70],[36,66],[42,73],[54,68],[58,72],[70,70],[74,76],[90,68],[96,72],[110,70],[115,66],[129,72],[133,68],[138,74]].map(([x,y],i) => (
                      <rect key={`nw-${i}`} x={x} y={y} width="2" height="2" rx="0.4"
                        fill={i%4===0?"#ff30ff":i%4===1?"#50cdff":i%4===2?"#c030ff":"#ff6090"}
                        opacity="0.92"/>
                    ))}
                    {/* Street-level magenta neon wash */}
                    <rect x="0" y="82" width="150" height="8" fill="#08021c"/>
                    <rect x="0" y="81" width="150" height="1.5" fill="#d028ff" opacity="0.35"/>
                    {/* Horizon glow band */}
                    <rect x="0" y="42" width="150" height="8" fill="rgba(28,6,85,0.3)"/>
                  </svg>
                </div>
              )}
              <div className="dm-flux-label">AETHER-FLUX: {Math.round(progress)}%</div>
              <div className="dm-new-slot" />
            </div>
          </div>
        </footer>

        {/* SESSION LOG */}
        <div className="dm-session-bar">
          <div className="dm-session-ticker">
            SESSION_LOG: INPUT ACCEPTED... {teraflops} TFLOPS...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            ENTROPY:{entropy}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            FPS:{fps}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            HEAP:{heapDisplay}MB...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            VRAM-EST:{vramGb}GB...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            TOKENS:{tokenEst}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            UPTIME:{sessionHMS}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            HASH:0x{paramHex}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            SYNC:{syncStatus}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            CYCLES:{cyclesCur}/{steps}...&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            SESSION_LOG: INPUT ACCEPTED... {teraflops} TFLOPS...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            ENTROPY:{entropy}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            FPS:{fps}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            HEAP:{heapDisplay}MB...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            VRAM-EST:{vramGb}GB...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            TOKENS:{tokenEst}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            UPTIME:{sessionHMS}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            HASH:0x{paramHex}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            SYNC:{syncStatus}...&nbsp;&nbsp;&#183;&nbsp;&nbsp;
            CYCLES:{cyclesCur}/{steps}...&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          </div>
        </div>

      </div>
    </div>
  );
}