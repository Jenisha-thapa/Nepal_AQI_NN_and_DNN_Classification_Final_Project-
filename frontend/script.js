const CAT_LABEL = {
  Good: 'Good', Moderate: 'Moderate', Unhealthy_SG: 'Unhealthy (Sensitive Groups)',
  Unhealthy: 'Unhealthy', Very_Unhealthy: 'Very Unhealthy', Hazardous: 'Hazardous'
};
const CAT_VAR = {
  Good: '--good', Moderate: '--moderate', Unhealthy_SG: '--usg',
  Unhealthy: '--unhealthy', Very_Unhealthy: '--veryunhealthy', Hazardous: '--hazardous'
};
const CAT_INK = {
  Good: '--good-ink', Moderate: '--moderate-ink', Unhealthy_SG: '--usg-ink',
  Unhealthy: '--unhealthy-ink', Very_Unhealthy: '--veryunhealthy-ink', Hazardous: '--hazardous-ink'
};
let CATEGORY_ORDER = ['Good', 'Moderate', 'Unhealthy_SG', 'Unhealthy', 'Very_Unhealthy', 'Hazardous'];

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// ---------------------------------------------------------------------
// A. Historical / dataset AQI -- district selector
// ---------------------------------------------------------------------
let DISTRICTS = {};

async function loadDistricts() {
  const res = await fetch('/api/districts');
  DISTRICTS = await res.json();
  const sel = document.getElementById('districtSelect');
  sel.innerHTML = '';
  Object.keys(DISTRICTS).sort().forEach(d => {
    const opt = document.createElement('option');
    opt.value = d; opt.textContent = d;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => renderDistrict(sel.value));
  renderDistrict(Object.keys(DISTRICTS).sort()[0]);
}

function renderDistrict(name) {
  const d = DISTRICTS[name];
  if (!d) return;
  document.getElementById('distName').textContent = name;

  const pill = document.getElementById('distPill');
  pill.textContent = `${CAT_LABEL[d.AQI_Status] || d.AQI_Status}  \u00b7  AQI ${Math.round(d.Calculated_AQI)}`;
  pill.style.background = cssVar(CAT_VAR[d.AQI_Status]);
  pill.style.color = cssVar(CAT_INK[d.AQI_Status]);

  document.getElementById('distMarker').style.left = Math.min(100, d.Calculated_AQI / 500 * 100) + '%';
  document.getElementById('distPM25').textContent = d.PM2_5.toFixed(1);
  document.getElementById('distPM10').textContent = d.PM10.toFixed(1);
  document.getElementById('distNO2').textContent = d.NO2.toFixed(1);
  document.getElementById('distNote').textContent =
    `Averaged over ${d.n_records.toLocaleString()} readings for ${name}.`;
}

// ---------------------------------------------------------------------
// B. AI prediction -- calls the FastAPI backend, which runs both models
// ---------------------------------------------------------------------
async function loadMeta() {
  const res = await fetch('/api/meta');
  const meta = await res.json();
  CATEGORY_ORDER = meta.category_order || CATEGORY_ORDER;
  document.getElementById('aiDesc').textContent =
    `Enter any PM2.5 / PM10 / NO2 reading and this runs both trained models on the ` +
    `legitimate, EPA-formula-derived AQI target (macro F1: MLP ${meta.mlp_macro_f1}, ` +
    `DNN ${meta.dnn_macro_f1}) -- live, on your backend, not a lookup table.`;
}

function modelCard(name, m) {
  const trained = new Set(m.trained_categories || []);
  const color = cssVar(CAT_VAR[m.category]);
  const ink = cssVar(CAT_INK[m.category]);
  const label = CAT_LABEL[m.category] || m.category;

  const rows = CATEGORY_ORDER
    .map(cat => ({ cat, p: m.confidences[cat] ?? 0 }))
    .sort((a, b) => b.p - a.p)
    .map(({ cat, p }) => {
      const isTrained = trained.size === 0 || trained.has(cat);
      const tag = isTrained ? '' : '<span class="untrained-tag">no training data</span>';
      return `
        <div class="prob-row ${isTrained ? '' : 'unsupported'}">
          <div class="prob-name">${CAT_LABEL[cat] || cat}${tag}</div>
          <div class="prob-bar-bg"><div class="prob-bar" style="width:${(p * 100).toFixed(1)}%; background:${cssVar(CAT_VAR[cat])}"></div></div>
          <div class="prob-pct">${(p * 100).toFixed(1)}%</div>
        </div>`;
    }).join('');

  const el = document.createElement('div');
  el.className = 'model-card';
  el.innerHTML = `
    <div class="m-title">
      <span>${name === 'MLP' ? 'MLP Neural Network' : 'From-Scratch DNN'}</span>
      <span class="m-conf">confidence ${(m.confidence * 100).toFixed(1)}%</span>
    </div>
    <div class="m-result">
      ${label}
      <span class="range-chip" style="background:${color}; color:${ink}">AQI ${m.aqi_range}</span>
    </div>
    ${rows}
  `;
  return el;
}

async function predict() {
  const pm25 = parseFloat(document.getElementById('inPM25').value) || 0;
  const pm10 = parseFloat(document.getElementById('inPM10').value) || 0;
  const no2 = parseFloat(document.getElementById('inNO2').value) || 0;

  const btn = document.getElementById('predictBtn');
  const loading = document.getElementById('loading');
  const errorBox = document.getElementById('errorBox');
  const resultsEl = document.getElementById('modelResults');

  btn.disabled = true; loading.style.display = 'block'; errorBox.style.display = 'none';
  resultsEl.innerHTML = '';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ PM2_5_ug_m3: pm25, PM10_ug_m3: pm10, NO2_ppb: no2 })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    for (const [name, m] of Object.entries(data.models)) {
      resultsEl.appendChild(modelCard(name, m));
    }
  } catch (err) {
    errorBox.textContent = 'Error: ' + err.message + ' -- is the FastAPI backend running?';
    errorBox.style.display = 'block';
  } finally {
    btn.disabled = false; loading.style.display = 'none';
  }
}

document.getElementById('predictBtn').addEventListener('click', predict);

loadDistricts();
loadMeta();
