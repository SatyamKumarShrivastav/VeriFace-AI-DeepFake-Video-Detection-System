/* ── DeepFake Detector JS ── */
'use strict';

const API = '';  // Same origin via FastAPI static serving

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropzone      = document.getElementById('dropzone');
const fileInput     = document.getElementById('file-input');
const browseBtn     = document.getElementById('browse-btn');
const dropInner     = document.getElementById('drop-inner');
const previewInner  = document.getElementById('preview-inner');
const videoPreview  = document.getElementById('video-preview');
const previewMeta   = document.getElementById('preview-meta');
const analyzeBtn    = document.getElementById('analyze-btn');
const clearBtn      = document.getElementById('clear-btn');

const uploadSection   = document.getElementById('upload-section');
const progressSection = document.getElementById('progress-section');
const progressBar     = document.getElementById('progress-bar');
const progressStatus  = document.getElementById('progress-status');
const resultsSection  = document.getElementById('results-section');
const newAnalysisBtn  = document.getElementById('new-analysis-btn');
const downloadBtn     = document.getElementById('download-btn');

let currentFile   = null;
let lastReport    = null;

// ── File Selection ────────────────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('video/')) loadFile(f);
  else showToast('⚠️ Please drop a valid video file.');
});

function loadFile(file) {
  currentFile = file;
  const url   = URL.createObjectURL(file);
  videoPreview.src = url;
  const sizeMB = (file.size / 1024 / 1024).toFixed(2);
  previewMeta.textContent = `📄 ${file.name}  •  📦 ${sizeMB} MB  •  🎬 ${file.type}`;
  dropInner.classList.add('hidden');
  previewInner.classList.remove('hidden');
}

clearBtn.addEventListener('click', () => {
  currentFile = null;
  videoPreview.src = '';
  fileInput.value = '';
  previewInner.classList.add('hidden');
  dropInner.classList.remove('hidden');
});

// ── Analysis ──────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', startAnalysis);

async function startAnalysis() {
  if (!currentFile) return;

  // Show progress
  uploadSection.classList.add('hidden');
  resultsSection.classList.add('hidden');
  progressSection.classList.remove('hidden');

  const steps = [
    { id: 'step-upload',   label: '📤 Uploading video…',               pct: 10 },
    { id: 'step-frames',   label: '🎞️ Extracting frames…',             pct: 25 },
    { id: 'step-freq',     label: '📡 Frequency domain analysis…',      pct: 40 },
    { id: 'step-temporal', label: '⏱️ Temporal consistency check…',     pct: 55 },
    { id: 'step-face',     label: '🎭 Face region forensics…',          pct: 68 },
    { id: 'step-noise',    label: '📷 Noise residual analysis…',        pct: 80 },
    { id: 'step-report',   label: '📋 Generating report…',             pct: 92 },
  ];

  // Animate steps while waiting for the API
  let stepIdx  = 0;
  const ticker = setInterval(() => {
    if (stepIdx < steps.length) {
      const s = steps[stepIdx];
      updateProgress(s.pct, s.label);
      markStep(s.id, 'active');
      if (stepIdx > 0) markStep(steps[stepIdx - 1].id, 'done');
      stepIdx++;
    }
  }, 1200);

  try {
    const form = new FormData();
    form.append('file', currentFile);

    const resp = await fetch(`${API}/analyze`, { method: 'POST', body: form });
    clearInterval(ticker);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Server error');
    }

    const report = await resp.json();
    lastReport   = report;

    // Mark all done
    steps.forEach(s => markStep(s.id, 'done'));
    updateProgress(100, '✅ Analysis complete!');

    await delay(600);
    showResults(report);

  } catch (err) {
    clearInterval(ticker);
    showToast(`❌ ${err.message}`);
    resetToUpload();
  }
}

// ── Render Results ─────────────────────────────────────────────────────────────
function showResults(r) {
  progressSection.classList.add('hidden');
  resultsSection.classList.remove('hidden');
  uploadSection.classList.add('hidden');

  const verdictCard  = document.getElementById('verdict-card');
  const verdictBadge = document.getElementById('verdict-badge');
  const verdictIcon  = document.getElementById('verdict-icon');
  const verdictValue = document.getElementById('verdict-value');

  // Verdict styling
  verdictCard.classList.remove('verdict-fake','verdict-real','verdict-uncertain');
  verdictBadge.classList.remove('verdict-fake','verdict-real','verdict-uncertain');

  const icons = { FAKE:'🚨', REAL:'✅', UNCERTAIN:'⚠️' };
  verdictIcon.textContent  = icons[r.verdict] || '🔍';
  verdictValue.textContent = r.verdict;
  verdictCard.classList.add(`verdict-${r.verdict.toLowerCase()}`);
  verdictBadge.classList.add(`verdict-${r.verdict.toLowerCase()}`);

  // Gauges (semi-circle: path length ≈ 173)
  animateGauge('gauge-fake', r.fake_probability, 173);
  animateGauge('gauge-real', r.real_probability, 173);
  document.getElementById('gauge-fake-val').textContent = `${r.fake_probability}%`;
  document.getElementById('gauge-real-val').textContent = `${r.real_probability}%`;

  document.getElementById('accuracy-val').textContent = `${r.accuracy}%`;
  document.getElementById('frames-val').textContent    = r.frames_analyzed;
  document.getElementById('summary-text').textContent  = r.summary;

  // Reasons
  const reasonsList = document.getElementById('reasons-list');
  reasonsList.innerHTML = '';
  (r.top_reasons || []).forEach((reason, i) => {
    const li   = document.createElement('li');
    li.className = 'reason-item';
    li.style.animationDelay = `${i * 80}ms`;
    // Convert **bold** markdown to <strong>
    li.innerHTML = reason.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    reasonsList.appendChild(li);
  });

  // Signals
  const grid = document.getElementById('signals-grid');
  grid.innerHTML = '';
  (r.signal_breakdown || []).forEach((sig, i) => {
    const barColor = { HIGH:'#f87171', MEDIUM:'#fb923c', LOW:'#fbbf24', CLEAN:'#22d3a0' }[sig.severity] || '#7b82a0';
    const card = document.createElement('div');
    card.className = 'signal-card';
    card.style.animationDelay = `${i * 60}ms`;
    card.innerHTML = `
      <div class="signal-head">
        <span class="signal-ico">${sig.icon}</span>
        <span class="signal-name">${sig.label}</span>
        <span class="signal-sev sev-${sig.severity}">${sig.severity}</span>
      </div>
      <div class="signal-bar-wrap">
        <div class="signal-bar" data-score="${sig.score}" style="width:0%;background:${barColor}"></div>
      </div>
      <div class="signal-detail">${sig.detail} — Score: <strong style="color:${barColor}">${sig.score}%</strong></div>
    `;
    grid.appendChild(card);
  });
  // Animate bars after render
  requestAnimationFrame(() => {
    document.querySelectorAll('.signal-bar').forEach(b => {
      b.style.width = Math.min(b.dataset.score, 100) + '%';
    });
  });

  // Video info
  const infoGrid = document.getElementById('info-grid');
  const vi = r.video_info || {};
  const items = [
    { val: vi.resolution || '—',             lbl: 'Resolution' },
    { val: `${vi.fps || 0} fps`,              lbl: 'Frame Rate' },
    { val: `${vi.duration_sec || 0}s`,        lbl: 'Duration' },
    { val: `${r.file_size_mb || '—'} MB`,     lbl: 'File Size' },
    { val: `${r.frames_analyzed}`,             lbl: 'Frames Analyzed' },
    { val: `${r.elapsed_sec || '—'}s`,         lbl: 'Analysis Time' },
  ];
  infoGrid.innerHTML = items.map(it =>
    `<div class="info-item"><div class="info-val">${it.val}</div><div class="info-lbl">${it.lbl}</div></div>`
  ).join('');

  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// ── Gauge Animation ───────────────────────────────────────────────────────────
function animateGauge(id, pct, pathLen) {
  const el  = document.getElementById(id);
  if (!el) return;
  const arc = (pct / 100) * pathLen;
  setTimeout(() => { el.style.strokeDasharray = `${arc} ${pathLen}`; }, 100);
}

// ── Progress helpers ──────────────────────────────────────────────────────────
function updateProgress(pct, label) {
  progressBar.style.width   = pct + '%';
  progressStatus.textContent = label;
}
function markStep(id, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('active','done');
  el.classList.add(cls);
}

// ── Reset ─────────────────────────────────────────────────────────────────────
newAnalysisBtn.addEventListener('click', resetToUpload);
function resetToUpload() {
  currentFile = null; lastReport = null;
  videoPreview.src = ''; fileInput.value = '';
  previewInner.classList.add('hidden');
  dropInner.classList.remove('hidden');
  progressBar.style.width = '0%';
  document.querySelectorAll('.step').forEach(s => s.classList.remove('active','done'));
  progressSection.classList.add('hidden');
  resultsSection.classList.add('hidden');
  uploadSection.classList.remove('hidden');
  uploadSection.scrollIntoView({ behavior: 'smooth' });
}

// ── Download Report ───────────────────────────────────────────────────────────
downloadBtn.addEventListener('click', () => {
  if (!lastReport) return;
  const lines = [
    `DEEPFAKE VIDEO DETECTION REPORT`,
    `================================`,
    `Analyzed by: Mr. S. K. Shrivastav's DeepFake Detection System`,
    `Date: ${new Date().toLocaleString()}`,
    ``,
    `FILE: ${lastReport.filename}`,
    `VERDICT: ${lastReport.verdict}`,
    `Fake Probability: ${lastReport.fake_probability}%`,
    `Real Probability: ${lastReport.real_probability}%`,
    `Accuracy: ${lastReport.accuracy}%`,
    `Confidence: ${lastReport.confidence}%`,
    `Frames Analyzed: ${lastReport.frames_analyzed}`,
    ``,
    `SUMMARY:`,
    lastReport.summary,
    ``,
    `KEY REASONS:`,
    ...(lastReport.top_reasons || []).map((r, i) => `${i+1}. ${r.replace(/\*\*/g,'')}`),
    ``,
    `SIGNAL BREAKDOWN:`,
    ...(lastReport.signal_breakdown || []).map(s =>
      `  ${s.label.padEnd(35)} Score: ${String(s.score+'%').padEnd(6)} Severity: ${s.severity}`
    ),
    ``,
    `VIDEO INFO:`,
    `  Resolution : ${lastReport.video_info?.resolution}`,
    `  FPS        : ${lastReport.video_info?.fps}`,
    `  Duration   : ${lastReport.video_info?.duration_sec}s`,
    `  File Size  : ${lastReport.file_size_mb} MB`,
  ];

  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `deepfake_report_${lastReport.filename.replace(/\.[^.]+$/, '')}.txt`;
  a.click();
});

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg) {
  const t   = document.createElement('div');
  t.textContent = msg;
  Object.assign(t.style, {
    position:'fixed', bottom:'30px', left:'50%', transform:'translateX(-50%)',
    background:'rgba(30,30,50,0.95)', color:'#fff', padding:'12px 24px',
    borderRadius:'10px', fontSize:'14px', zIndex:'9999',
    boxShadow:'0 4px 20px rgba(0,0,0,0.5)', border:'1px solid rgba(255,255,255,0.12)',
    animation:'fadeIn .3s ease',
  });
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ── Utils ──────────────────────────────────────────────────────────────────────
const delay = ms => new Promise(r => setTimeout(r, ms));
