/**
 * SafeRoute AI - Real-Time Public Transport Women Safety Telemetry Engine
 * Track: AI-SDLC | Review-3 Implementation | Dundigal -> Secunderabad
 */

let map = null;
let tileLayer = null;
let plannedRouteLine = null;
let busMarker = null;
let trailSegments = [];
let riskZoneLayers = [];
let ws = null;
let isJourneyRunning = true;
let isRealGpsActive = false;
let realGpsWatchId = null;
let modelChartInstance = null;
let activeScenarioName = 'Normal';
let isDemoRunning = false;
let demoTimer = null;
let lastLatLng = null;

// ---------------------------------------------------------------------------
// 1. Theme Management (Light / Dark Mode with Tile Swap)
// ---------------------------------------------------------------------------

function initTheme() {
  const saved = localStorage.getItem('saferoute-theme') || 'light';
  applyTheme(saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  const next = current === 'dark' ? 'light' : 'dark';
  applyTheme(next);
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('saferoute-theme', theme);

  const iconEl = document.getElementById('theme-icon');
  const textEl = document.getElementById('theme-text');
  if (iconEl && textEl) {
    if (theme === 'dark') {
      iconEl.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
      textEl.textContent = 'Dark';
    } else {
      iconEl.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>`;
      textEl.textContent = 'Light';
    }
  }

  // Swap Map Tiles if Leaflet is initialized
  if (map && tileLayer) {
    const tileUrl = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
    tileLayer.setUrl(tileUrl);
  }
}

// ---------------------------------------------------------------------------
// 2. Tab Navigation
// ---------------------------------------------------------------------------

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  const targetTab = document.getElementById(tabId);
  if (targetTab) {
    targetTab.classList.add('active');
  }

  const btn = document.querySelector(`.tab-btn[onclick*="${tabId}"]`);
  if (btn) btn.classList.add('active');

  // Toggle monitor class on body for clean footer management
  document.body.classList.toggle('tab-monitor-active', tabId === 'tab-monitor');

  // If leaving Live Monitor, hide caution prompt & demo banner
  if (tabId !== 'tab-monitor') {
    const cautionOverlay = document.getElementById('caution-prompt-banner');
    if (cautionOverlay) cautionOverlay.style.display = 'none';
    const demoBanner = document.getElementById('demo-banner');
    if (demoBanner) demoBanner.style.display = 'none';
  }

  if (tabId === 'tab-monitor' && map) {
    setTimeout(() => map.invalidateSize(), 200);
  }
  if (tabId === 'tab-experiments') {
    loadExperimentMetrics();
    loadFirstPrinciples();
  }
  if (tabId === 'tab-validation') {
    loadValidationReport();
  }
}

// ---------------------------------------------------------------------------
// 3. Leaflet Map Setup with CartoDB Positron & Custom Markers
// ---------------------------------------------------------------------------

async function initMap() {
  const mapEl = document.getElementById('map');
  if (!mapEl) return;

  const initialTheme = document.documentElement.getAttribute('data-theme') || 'light';
  const tileUrl = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';

  map = L.map('map', {
    center: [17.525, 78.465],
    zoom: 12,
    zoomControl: false
  });

  // Zoom control in bottom right or top right
  L.control.zoom({ position: 'topright' }).addTo(map);

  tileLayer = L.tileLayer(tileUrl, {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);

  try {
    const res = await fetch('/api/route');
    const data = await res.json();

    // 1. Planned Route Line (Muted slate-blue dashed)
    if (data.polyline && data.polyline.length > 0) {
      plannedRouteLine = L.polyline(data.polyline, {
        color: '#6366F1',
        weight: 3,
        dashArray: '6, 8',
        opacity: 0.65
      }).addTo(map);

      map.fitBounds(plannedRouteLine.getBounds(), { padding: [30, 30] });

      // Start Marker (Green dot)
      const startCoord = data.polyline[0];
      const startIcon = L.divIcon({
        className: 'route-start-pin',
        html: '<div style="width:12px;height:12px;border-radius:50%;background:#16A34A;border:2px solid #FFFFFF;box-shadow:0 1px 4px rgba(0,0,0,0.3);"></div>',
        iconSize: [12, 12],
        iconAnchor: [6, 6]
      });
      L.marker(startCoord, { icon: startIcon }).addTo(map).bindTooltip('Origin: Dundigal Terminal');

      // Destination Marker (Dark pin)
      const endCoord = data.polyline[data.polyline.length - 1];
      const endIcon = L.divIcon({
        className: 'route-end-pin',
        html: '<div style="width:14px;height:14px;border-radius:50%;background:#0F172A;border:2px solid #FFFFFF;box-shadow:0 1px 4px rgba(0,0,0,0.3);"></div>',
        iconSize: [14, 14],
        iconAnchor: [7, 7]
      });
      L.marker(endCoord, { icon: endIcon }).addTo(map).bindTooltip('Destination: Secunderabad Hub');
    }

    // 2. Risk Zones (Translucent soft circles with hover tooltip)
    if (data.risk_zones && data.risk_zones.length > 0) {
      data.risk_zones.forEach(zone => {
        const circle = L.circle([zone.lat, zone.lon], {
          radius: zone.radius_m,
          color: '#E11D48',
          weight: 1,
          fillColor: '#E11D48',
          fillOpacity: 0.12
        }).addTo(map);

        circle.bindTooltip(`Risk Zone: ${zone.name}`, { sticky: true });
        riskZoneLayers.push(circle);
      });
    }

    // 3. Custom Bus Marker (White circle with accent border + bus SVG + pulsing ring)
    const busHtml = `
      <div class="custom-bus-marker">
        <div class="bus-pulse-ring" id="bus-pulse-indicator"></div>
        <div class="bus-circle">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 6v6"/><path d="M16 6v6"/><path d="M2 12h20"/>
            <rect x="4" y="3" width="16" height="15" rx="3"/>
            <circle cx="7" cy="18" r="2"/><circle cx="17" cy="18" r="2"/>
          </svg>
        </div>
      </div>
    `;

    const busIcon = L.divIcon({
      className: 'bus-leaflet-icon',
      html: busHtml,
      iconSize: [44, 44],
      iconAnchor: [22, 22]
    });

    const initCoord = data.polyline && data.polyline.length > 0 ? data.polyline[0] : [17.525, 78.465];
    busMarker = L.marker(initCoord, { icon: busIcon }).addTo(map);

  } catch (err) {
    console.error('Failed to load initial map route data:', err);
  }
}

function recenterMap() {
  if (map && busMarker) {
    map.setView(busMarker.getLatLng(), 14, { animate: true });
  } else if (map && plannedRouteLine) {
    map.fitBounds(plannedRouteLine.getBounds(), { padding: [30, 30] });
  }
}

// ---------------------------------------------------------------------------
// 4. WebSocket Live Telemetry Stream
// ---------------------------------------------------------------------------

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    const statusDot = document.getElementById('header-connection-status');
    const statusText = document.getElementById('conn-status-text');
    if (statusDot) statusDot.classList.remove('offline');
    if (statusText) statusText.textContent = 'Live Telemetry';
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleTelemetryTick(data);
    } catch (e) {
      console.error('Failed to parse telemetry JSON:', e);
    }
  };

  ws.onclose = () => {
    const statusDot = document.getElementById('header-connection-status');
    const statusText = document.getElementById('conn-status-text');
    if (statusDot) statusDot.classList.add('offline');
    if (statusText) statusText.textContent = 'Offline (Reconnecting...)';
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error('[WS] Connection error:', err);
    ws.close();
  };
}

// ---------------------------------------------------------------------------
// 5. Telemetry Tick Processor
// ---------------------------------------------------------------------------

function handleTelemetryTick(data) {
  const { telemetry, features, prediction, alert_state, pipeline_status, total_pipeline_latency_ms } = data;

  // 1. Pipeline Stepper Latency Profiling
  if (pipeline_status && pipeline_status.length >= 5) {
    const idMap = ['lat-sim', 'lat-feat', 'lat-model', 'lat-alert', 'lat-notif'];
    const dotMap = ['dot-sim', 'dot-feat', 'dot-model', 'dot-alert', 'dot-notif'];

    pipeline_status.forEach((mod, idx) => {
      const latEl = document.getElementById(idMap[idx]);
      const dotEl = document.getElementById(dotMap[idx]);

      if (latEl) latEl.textContent = `${mod.latency_ms}ms`;
      if (dotEl) {
        dotEl.className = 'step-indicator-dot';
        if (mod.indicator === 'amber') dotEl.classList.add('amber');
        else if (mod.indicator === 'red') dotEl.classList.add('red');
      }
    });
  }

  const totLatEl = document.getElementById('pipeline-total-ms');
  if (totLatEl && total_pipeline_latency_ms !== undefined) {
    totLatEl.textContent = `${total_pipeline_latency_ms} ms`;
  }

  // 2. Bus Marker & Dynamic Segmented Risk Trail
  if (telemetry && map && busMarker) {
    const newLatLng = [telemetry.lat, telemetry.lon];
    busMarker.setLatLng(newLatLng);

    // Step Counter & Progress Fill
    const stepEl = document.getElementById('bus-step-counter');
    const barEl = document.getElementById('bus-progress-bar');
    if (stepEl && telemetry.step !== undefined) {
      stepEl.textContent = `Step ${telemetry.step}/320`;
      if (barEl) {
        const pct = Math.min(100, Math.max(0, (telemetry.step / 320) * 100));
        barEl.style.width = `${pct}%`;
      }
    }

    // Segment Trail coloring based on current risk
    if (lastLatLng) {
      let strokeColor = '#16A34A'; // Safe
      if (prediction && prediction.risk_class === 1) strokeColor = '#F59E0B'; // Caution
      else if (prediction && prediction.risk_class === 2) strokeColor = '#E11D48'; // High-Risk

      const trail = L.polyline([lastLatLng, newLatLng], {
        color: strokeColor,
        weight: 4,
        opacity: 0.85
      }).addTo(map);

      trailSegments.push(trail);
      if (trailSegments.length > 70) {
        const old = trailSegments.shift();
        map.removeLayer(old);
      }
    }
    lastLatLng = newLatLng;

    // Update pulse ring color on bus marker
    const pulseRing = document.getElementById('bus-pulse-indicator');
    if (pulseRing) {
      pulseRing.className = 'bus-pulse-ring';
      if (prediction && prediction.risk_class === 1) pulseRing.classList.add('caution');
      else if (prediction && prediction.risk_class === 2) pulseRing.classList.add('high-risk');
    }
  }

  // 3. Circular Risk Gauge (SVG Ring 140px, circumference ~326.7)
  if (prediction && prediction.probabilities) {
    const probs = prediction.probabilities;
    const safeProb = probs['Safe'] || 0;
    const cautionProb = probs['Caution'] || 0;
    const highProb = probs['High-Risk'] || 0;

    let dominantLabel = 'Safe';
    let dominantConf = safeProb;
    let dominantColor = 'var(--risk-safe)';

    if (prediction.risk_class === 2 || highProb >= 0.32) {
      dominantLabel = 'High-Risk';
      dominantConf = highProb;
      dominantColor = 'var(--risk-high)';
    } else if (prediction.risk_class === 1 || (cautionProb + highProb) >= 0.38) {
      dominantLabel = 'Caution';
      dominantConf = cautionProb;
      dominantColor = 'var(--risk-caution)';
    }

    // Update gauge text
    const labelEl = document.getElementById('risk-gauge-label');
    const confEl = document.getElementById('risk-gauge-conf');
    const circleEl = document.getElementById('risk-gauge-circle');
    const gaugeWrap = document.getElementById('risk-gauge-wrap');

    if (labelEl) {
      labelEl.textContent = dominantLabel;
      labelEl.style.color = dominantColor;
    }
    if (confEl) {
      confEl.textContent = `${Math.round(dominantConf * 100)}%`;
    }

    if (circleEl) {
      circleEl.style.stroke = dominantColor;
      // Circumference = 2 * PI * 52 = 326.7
      const offset = 326.7 * (1 - dominantConf);
      circleEl.style.strokeDashoffset = offset;
    }

    // Pulse glow on High-Risk
    if (gaugeWrap) {
      if (dominantLabel === 'High-Risk') gaugeWrap.classList.add('high-risk-pulse');
      else gaugeWrap.classList.remove('high-risk-pulse');
    }

    // Stacked Probability Bar
    const safePct = Math.round(safeProb * 100);
    const cautionPct = Math.round(cautionProb * 100);
    const highPct = Math.round(highProb * 100);

    setBarWidth('prob-safe-bar', `${safePct}%`);
    setBarWidth('prob-caution-bar', `${cautionPct}%`);
    setBarWidth('prob-high-bar', `${highPct}%`);

    setText('prob-safe-pct', `${safePct}%`);
    setText('prob-caution-pct', `${cautionPct}%`);
    setText('prob-high-pct', `${highPct}%`);
  }

  // 4. Journey Metrics (2x3 Grid with alert threshold styling)
  if (features) {
    const devM = Math.round(features.route_deviation_m);
    setText('feat-dev', `${devM} m`);
    setTileAlert('tile-dev', devM > 300 ? 'danger' : (devM > 150 ? 'warn' : ''));

    const stopS = Math.round(features.stop_duration_s);
    if (stopS > 60) {
      const mins = Math.floor(stopS / 60);
      const secs = stopS % 60;
      setText('feat-stop', `${mins}m ${secs}s`);
    } else {
      setText('feat-stop', `${stopS}s`);
    }
    setTileAlert('tile-stop', stopS > 180 ? 'danger' : (stopS > 60 ? 'warn' : ''));

    setText('feat-speed', `${features.avg_speed_kmh.toFixed(1)} km/h`);
    setTileAlert('tile-speed', features.avg_speed_kmh > 75 ? 'warn' : '');

    setText('feat-delay', `${features.delay_ratio.toFixed(2)}x`);
    setTileAlert('tile-delay', features.delay_ratio > 1.8 ? 'danger' : (features.delay_ratio > 1.3 ? 'warn' : ''));

    const nightEl = document.getElementById('feat-night');
    if (nightEl) {
      nightEl.textContent = features.is_night ? 'Night' : 'Day';
      if (features.is_night) {
        nightEl.innerHTML = '<span style="color:var(--risk-caution-text); font-weight:600;">Night</span>';
      }
    }

    setText('feat-zone', features.zone_risk_score.toFixed(2));
    setTileAlert('tile-zone', features.zone_risk_score > 0.7 ? 'danger' : (features.zone_risk_score > 0.35 ? 'warn' : ''));
  }

  // 5. Why This Alert (Attribution factors with clean SVG icons, max 3)
  if (prediction && prediction.reasons) {
    const ul = document.getElementById('reasons-list');
    if (ul && prediction.reasons.length > 0) {
      const displayReasons = prediction.reasons.slice(0, 3);
      ul.innerHTML = displayReasons.map(r => `
        <li class="reason-item">
          <svg class="reason-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
          <span>${escapeHtml(r)}</span>
        </li>
      `).join('');
    }
  }

  // 6. Caution 15s Countdown Overlay (Only show if Live Monitor is active tab)
  const cautionOverlay = document.getElementById('caution-prompt-banner');
  const timerVal = document.getElementById('caution-countdown-val');
  const isMonitorActive = document.getElementById('tab-monitor')?.classList.contains('active');
  if (cautionOverlay && alert_state) {
    if (alert_state.current_state === 'Caution' && isMonitorActive) {
      cautionOverlay.style.display = 'block';
      const rem = Math.max(0, alert_state.caution_time_remaining_s || 0);
      if (timerVal) timerVal.textContent = `${rem.toFixed(1)}s`;
    } else {
      cautionOverlay.style.display = 'none';
    }
  }

  // 7. Dispatched Alerts Handling
  if (alert_state && alert_state.alerts_dispatched && alert_state.alerts_dispatched.length > 0) {
    alert_state.alerts_dispatched.forEach(a => {
      appendAlertEvent(a);
      showToast(a.message || 'Safety event detected on transit corridor', a.risk_level || 'ALERT');
    });
  }
}

function setTileAlert(tileId, state) {
  const el = document.getElementById(tileId);
  if (!el) return;
  el.classList.remove('warn', 'danger');
  if (state === 'warn') el.classList.add('warn');
  else if (state === 'danger') el.classList.add('danger');
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setBarWidth(id, width) {
  const el = document.getElementById(id);
  if (el) el.style.width = width;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// 6. Alert Events, Timeline & Phone Mock
// ---------------------------------------------------------------------------

function showInnerTab(tab) {
  const btnTimeline = document.getElementById('tab-btn-timeline');
  const btnPhone = document.getElementById('tab-btn-phone');
  const viewTimeline = document.getElementById('view-timeline');
  const viewPhone = document.getElementById('view-phone');

  if (tab === 'timeline') {
    if (btnTimeline) btnTimeline.classList.add('active');
    if (btnPhone) btnPhone.classList.remove('active');
    if (viewTimeline) viewTimeline.style.display = 'flex';
    if (viewPhone) viewPhone.style.display = 'none';
  } else {
    if (btnTimeline) btnTimeline.classList.remove('active');
    if (btnPhone) btnPhone.classList.add('active');
    if (viewTimeline) viewTimeline.style.display = 'none';
    if (viewPhone) viewPhone.style.display = 'flex';
  }
}

function appendAlertEvent(alert) {
  const timeStr = alert.timestamp ? new Date(alert.timestamp * 1000).toLocaleTimeString() : new Date().toLocaleTimeString();
  const gmapsUrl = `https://maps.google.com/?q=${alert.lat},${alert.lon}`;

  // 1. Timeline List
  const timeline = document.getElementById('view-timeline');
  if (timeline) {
    const row = document.createElement('div');
    row.className = 'timeline-event-row';
    const dotClass = alert.risk_level === 'HIGH' || alert.risk_level === 'SOS' ? 'red' : 'amber';

    row.innerHTML = `
      <span class="event-dot ${dotClass}"></span>
      <div class="event-text-wrap">
        <span><strong>${escapeHtml(alert.recipient_name || 'Emergency Contact')}:</strong> ${escapeHtml(alert.message || 'Alert dispatched')}</span>
        <span class="event-time">${timeStr} &bull; <a href="${gmapsUrl}" target="_blank" style="color:var(--accent); text-decoration:underline;">View GPS</a></span>
      </div>
    `;
    timeline.prepend(row);
  }

  // 2. Mock Phone Feed
  const phoneFeed = document.getElementById('mock-phone-sms-feed');
  if (phoneFeed) {
    const bubble = document.createElement('div');
    bubble.className = 'phone-sms-bubble';
    bubble.innerHTML = `
      <div class="sms-header">
        <span>${escapeHtml(alert.risk_level || 'ALERT')} DISPATCH</span>
        <span style="color:var(--text-muted); font-weight:400;">${timeStr}</span>
      </div>
      <div>${escapeHtml(alert.message || 'Safety risk detected on Dundigal transit route.')}</div>
      <a href="${gmapsUrl}" target="_blank" class="sms-gmaps-btn">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
        <span>Live Location Link</span>
      </a>
    `;
    phoneFeed.prepend(bubble);
  }
}

function showToast(message, type) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast-item ${type ? type.toLowerCase() : 'caution'}`;
  toast.innerHTML = `
    <div style="flex:1;">
      <div style="font-weight:600; font-size:12.5px; color:var(--text); margin-bottom:2px;">${escapeHtml(type || 'Safety Alert')}</div>
      <div style="font-size:12px; color:var(--text-muted);">${escapeHtml(message)}</div>
    </div>
  `;

  container.prepend(toast);

  setTimeout(() => {
    toast.style.transition = 'opacity 300ms, transform 300ms';
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    setTimeout(() => toast.remove(), 300);
  }, 5000);
}

async function clearAlerts() {
  try {
    await fetch('/api/alerts', { method: 'DELETE' });
    const timeline = document.getElementById('view-timeline');
    if (timeline) timeline.innerHTML = '<div style="color:var(--text-muted); font-size:11.5px; padding:10px 0;">Alert history cleared.</div>';
    const phoneFeed = document.getElementById('mock-phone-sms-feed');
    if (phoneFeed) phoneFeed.innerHTML = '<div style="text-align:center; color:var(--text-muted); font-size:10px; margin-top:30px;">Waiting for new incoming simulated SMS alerts...</div>';
  } catch (err) {
    console.error('Failed to clear alerts:', err);
  }
}

// ---------------------------------------------------------------------------
// 7. Actions & Controls (SOS, Safe, Pause, Scenarios, Demo Mode)
// ---------------------------------------------------------------------------

async function triggerSos() {
  try {
    const res = await fetch('/api/sos', { method: 'POST' });
    const data = await res.json();
    if (data.alerts) {
      data.alerts.forEach(a => appendAlertEvent(a));
    }
    showToast('EMERGENCY SOS TRIGGERED! Instant alerts dispatched to emergency contacts & SHE teams.', 'SOS');
  } catch (err) {
    console.error('Failed to trigger SOS:', err);
  }
}

async function confirmSafe() {
  try {
    await fetch('/api/safe-confirm', { method: 'POST' });
    const overlay = document.getElementById('caution-prompt-banner');
    if (overlay) overlay.style.display = 'none';
    showToast('Passenger confirmed safe. Alert countdown cancelled.', 'Safe');
  } catch (err) {
    console.error('Failed to confirm safe:', err);
  }
}

async function toggleJourney() {
  const btn = document.getElementById('btn-toggle-journey');
  const url = isJourneyRunning ? '/api/journey/stop' : '/api/journey/start';

  try {
    await fetch(url, { method: 'POST' });
    isJourneyRunning = !isJourneyRunning;
    if (btn) {
      btn.innerHTML = isJourneyRunning
        ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg><span>Pause</span>`
        : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Resume</span>`;
    }
  } catch (err) {
    console.error('Failed to toggle journey:', err);
  }
}

async function resetJourney() {
  try {
    await fetch('/api/journey/reset', { method: 'POST' });
    trailSegments.forEach(seg => map.removeLayer(seg));
    trailSegments = [];
    lastLatLng = null;
    setScenarioButtonActive('Normal');
    recenterMap();
  } catch (err) {
    console.error('Failed to reset journey:', err);
  }
}

async function setScenario(name) {
  try {
    const res = await fetch(`/api/scenario/${encodeURIComponent(name)}`, { method: 'POST' });
    if (res.ok) {
      activeScenarioName = name;
      setScenarioButtonActive(name);
    }
  } catch (err) {
    console.error(`Failed to activate scenario ${name}:`, err);
  }
}

function setScenarioButtonActive(name) {
  const idMap = {
    'Normal': 'sc-btn-Normal',
    'Route Deviation': 'sc-btn-Route-Deviation',
    'Prolonged Stop': 'sc-btn-Prolonged-Stop',
    'Speed Anomaly': 'sc-btn-Speed-Anomaly',
    'Night + High-Risk Zone': 'sc-btn-Night-Risk',
    'Combined Worst-Case': 'sc-btn-Worst-Case'
  };

  document.querySelectorAll('.scenario-chip').forEach(b => b.classList.remove('active'));
  const activeBtnId = idMap[name];
  if (activeBtnId) {
    const btn = document.getElementById(activeBtnId);
    if (btn) btn.classList.add('active');
  }
}

async function toggleDemoMode() {
  if (!isDemoRunning) {
    startDemoMode();
  } else {
    stopDemoMode();
  }
}

async function startDemoMode() {
  try {
    await fetch('/api/demo-mode', { method: 'POST' });
    isDemoRunning = true;
    const btn = document.getElementById('btn-demo-mode');
    const label = document.getElementById('demo-btn-label');
    const banner = document.getElementById('demo-banner');
    const bannerText = document.getElementById('demo-banner-text');

    if (btn) btn.style.background = 'var(--risk-high)';
    if (label) label.textContent = 'Stop Demo';
    if (banner) banner.style.display = 'flex';
    if (bannerText) bannerText.textContent = 'Phase 1: Normal transit on Dundigal route (Baseline)';

    // Step-by-step caption updates during the automated sequence
    let step = 1;
    clearInterval(demoTimer);
    demoTimer = setInterval(() => {
      step++;
      if (step === 2 && bannerText) bannerText.textContent = 'Phase 2: Route deviation triggered toward Jeedimetla';
      if (step === 3 && bannerText) bannerText.textContent = 'Phase 3: Prolonged halt detected in high-risk zone';
      if (step === 4 && bannerText) bannerText.textContent = 'Phase 4: Caution state triggered (15s verification prompt)';
      if (step === 5 && bannerText) bannerText.textContent = 'Phase 5: Auto-escalation to emergency contacts & police';
      if (step > 6) {
        clearInterval(demoTimer);
        stopDemoMode();
      }
    }, 7000);

  } catch (err) {
    console.error('Failed to start demo mode:', err);
  }
}

async function stopDemoMode() {
  isDemoRunning = false;
  clearInterval(demoTimer);
  const btn = document.getElementById('btn-demo-mode');
  const label = document.getElementById('demo-btn-label');
  const banner = document.getElementById('demo-banner');

  if (btn) btn.style.background = 'var(--accent)';
  if (label) label.textContent = 'Demo Mode';
  if (banner) banner.style.display = 'none';

  await setScenario('Normal');
}

function updateGpsUI(active) {
  const panelBtn = document.getElementById('btn-real-gps-panel');
  const panelText = document.getElementById('real-gps-panel-text');
  const headerBtn = document.getElementById('btn-real-gps');

  if (active) {
    if (panelBtn) {
      panelBtn.style.background = 'var(--risk-safe-soft)';
      panelBtn.style.borderColor = 'var(--risk-safe)';
      panelBtn.style.color = 'var(--risk-safe-text)';
    }
    if (panelText) panelText.textContent = 'Using Real GPS (Live)';
    if (headerBtn) {
      headerBtn.style.color = 'var(--risk-safe-text)';
      const span = headerBtn.querySelector('span');
      if (span) span.textContent = 'Live Device GPS';
    }
  } else {
    if (panelBtn) {
      panelBtn.style.background = '';
      panelBtn.style.borderColor = '';
      panelBtn.style.color = '';
    }
    if (panelText) panelText.textContent = 'Use My Real GPS';
    if (headerBtn) {
      headerBtn.style.color = '';
      const span = headerBtn.querySelector('span');
      if (span) span.textContent = 'GPS Mode';
    }
  }
}

function stopRealGps() {
  if (realGpsWatchId) {
    navigator.geolocation.clearWatch(realGpsWatchId);
    realGpsWatchId = null;
  }
  isRealGpsActive = false;
  updateGpsUI(false);
  setScenario('Normal');
  showToast('Real GPS stopped. Resumed corridor route simulation.', 'Safe');
}

function toggleRealGps() {
  if (!navigator.geolocation) {
    showToast('Geolocation is not supported by this browser.', 'Caution');
    return;
  }

  if (!isRealGpsActive) {
    showToast('Requesting browser geolocation permission...', 'Safe');
    realGpsWatchId = navigator.geolocation.watchPosition(
      async (pos) => {
        try {
          const lat = pos.coords.latitude;
          const lon = pos.coords.longitude;
          const accuracy = pos.coords.accuracy || 10.0;
          const speed = pos.coords.speed ? (pos.coords.speed * 3.6) : 0.0;

          await fetch('/api/gps/real', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              lat: lat,
              lon: lon,
              accuracy: accuracy,
              speed_kmh: speed
            })
          });
        } catch (e) {
          console.error('Real GPS push failed:', e);
        }
      },
      (err) => {
        showToast(`GPS Error (${err.code}): ${err.message}. Reverting to corridor route.`, 'Caution');
        stopRealGps();
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 1000 }
    );
    isRealGpsActive = true;
    updateGpsUI(true);
    showToast('Real GPS active: streaming device coordinates to safety model.', 'Safe');
  } else {
    stopRealGps();
  }
}

// ---------------------------------------------------------------------------
// 8. Tab 2: Experiments (Chart.js Bar Chart & Metric Tables)
// ---------------------------------------------------------------------------

async function loadExperimentMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();

    const chartCanvas = document.getElementById('model-comparison-chart');
    if (chartCanvas && typeof Chart !== 'undefined' && data.model_comparison) {
      const models = Object.keys(data.model_comparison);
      const testAccs = models.map(m => (data.model_comparison[m].test_accuracy * 100).toFixed(1));
      const testF1s = models.map(m => (data.model_comparison[m].test_f1 * 100).toFixed(1));
      const cvMeans = models.map(m => (data.model_comparison[m].cv_mean * 100).toFixed(1));

      if (modelChartInstance) {
        modelChartInstance.destroy();
      }

      modelChartInstance = new Chart(chartCanvas, {
        type: 'bar',
        data: {
          labels: models,
          datasets: [
            {
              label: 'CV Mean F1 (%)',
              data: cvMeans,
              backgroundColor: '#5B5BD6',
              borderRadius: 6
            },
            {
              label: 'Test F1-Score (%)',
              data: testF1s,
              backgroundColor: '#16A34A',
              borderRadius: 6
            },
            {
              label: 'Test Accuracy (%)',
              data: testAccs,
              backgroundColor: '#8B5CF6',
              borderRadius: 6
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: {
                font: { family: 'Inter', size: 11 },
                color: document.documentElement.getAttribute('data-theme') === 'dark' ? '#E6EAF5' : '#0F172A'
              }
            }
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: { font: { family: 'Inter', size: 11 } }
            },
            y: {
              min: 70,
              max: 100,
              grid: { color: 'rgba(232, 234, 240, 0.6)' },
              ticks: { font: { family: 'JetBrains Mono', size: 11 } }
            }
          }
        }
      });
    }

    // CV Table
    const cvTbody = document.getElementById('cv-table-body');
    if (cvTbody && data.model_comparison) {
      cvTbody.innerHTML = Object.entries(data.model_comparison).map(([name, m]) => `
        <tr>
          <td><strong>${escapeHtml(name)}</strong></td>
          <td class="mono-num">${(m.cv_mean * 100).toFixed(1)}%</td>
          <td class="mono-num">&plusmn;${(m.cv_std * 100).toFixed(2)}%</td>
          <td class="mono-num">${(m.test_accuracy * 100).toFixed(1)}%</td>
          <td><span class="mono-num" style="color:var(--risk-safe-text); font-weight:700;">${(m.test_f1 * 100).toFixed(1)}%</span></td>
        </tr>
      `).join('');
    }

    // Classification Report
    const crTbody = document.getElementById('class-report-body');
    if (crTbody && data.classification_report) {
      const cr = data.classification_report;
      const classes = ['Safe', 'Caution', 'High-Risk'];
      crTbody.innerHTML = classes.map(c => {
        const row = cr[c] || {};
        return `
          <tr>
            <td><strong>${c}</strong></td>
            <td class="mono-num">${((row.precision || 0) * 100).toFixed(1)}%</td>
            <td class="mono-num">${((row.recall || 0) * 100).toFixed(1)}%</td>
            <td class="mono-num">${((row['f1-score'] || 0) * 100).toFixed(1)}%</td>
            <td class="mono-num">${row.support || '-'}</td>
          </tr>
        `;
      }).join('');
    }

    // Ablation Study Table
    const abTbody = document.getElementById('ablation-table-body');
    if (abTbody && data.ablation_study) {
      abTbody.innerHTML = Object.entries(data.ablation_study).map(([group, val]) => `
        <tr>
          <td><strong>${escapeHtml(group)}</strong></td>
          <td class="mono-num">${(val.ablated_f1 * 100).toFixed(1)}%</td>
          <td class="mono-num" style="color:var(--risk-high-text); font-weight:600;">-${(val.f1_drop * 100).toFixed(1)}%</td>
          <td><span class="status-pill fail">${val.impact}</span></td>
        </tr>
      `).join('');
    }

  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

async function retrainModel() {
  const nEst = parseInt(document.getElementById('train-estimators').value) || 100;
  const mDepth = parseInt(document.getElementById('train-depth').value) || 12;
  const seed = parseInt(document.getElementById('train-seed').value) || 42;
  const statusEl = document.getElementById('retrain-status');

  if (statusEl) statusEl.textContent = 'Retraining Random Forest pipeline...';

  try {
    const res = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_estimators: nEst, max_depth: mDepth, random_state: seed })
    });
    const result = await res.json();
    if (statusEl) {
      statusEl.textContent = `Retrained! Best Model: ${result.best_model} (Test F1: ${(result.metrics.test_f1 * 100).toFixed(1)}%)`;
    }

    const cm = document.getElementById('img-cm');
    const fi = document.getElementById('img-fi');
    if (cm) cm.src = `/artifacts/confusion_matrix.png?t=${Date.now()}`;
    if (fi) fi.src = `/artifacts/feature_importance.png?t=${Date.now()}`;

    await loadExperimentMetrics();
  } catch (err) {
    if (statusEl) statusEl.textContent = `Retrain failed: ${err.message}`;
  }
}

async function loadFirstPrinciples() {
  const el = document.getElementById('live-math-worked-example');
  if (!el) return;

  try {
    const res = await fetch('/api/first-principles');
    const math = await res.json();

    el.innerHTML = `
      <strong>Telemetry Fix Point:</strong> (${math.p_lat.toFixed(5)}, ${math.p_lon.toFixed(5)}) on Corridor Segment [${math.segment_start_name} &rarr; ${math.segment_end_name}]<br>
      &bull; Haversine Great-Circle Distance to Anchor A: <strong>${math.dist_to_anchor_m.toFixed(1)} m</strong> (R = 6,371 km)<br>
      &bull; Scalar Projection Parameter: <strong>t = ${math.t_clamped.toFixed(4)}</strong> (clamped to [0, 1])<br>
      &bull; Nearest Polyline Foot C: (${math.c_lat.toFixed(5)}, ${math.c_lon.toFixed(5)})<br>
      &bull; <strong>Orthogonal Route Deviation:</strong> <span style="color:var(--accent); font-weight:700;">${math.deviation_m.toFixed(1)} m</span>
    `;
  } catch (err) {
    el.textContent = 'Live math parameters will update on next telemetry tick.';
  }
}

// ---------------------------------------------------------------------------
// 9. Tab 3: CEP Validation (14 Scenarios Matrix & Thresholds)
// ---------------------------------------------------------------------------

async function loadValidationReport() {
  try {
    const res = await fetch('/artifacts/validation_report.json');
    const data = await res.json();
    renderValidationTable(data);
  } catch (err) {
    console.error('Failed to load validation report:', err);
  }
}

async function runValidationSuite() {
  const badge = document.getElementById('val-pass-rate-badge');
  if (badge) badge.textContent = 'Running...';

  try {
    const res = await fetch('/api/validation/run', { method: 'POST' });
    const data = await res.json();
    renderValidationTable(data);
    showToast(`Validation suite completed: ${data.pass_rate_pct.toFixed(1)}% Pass Rate`, 'Safe');
  } catch (err) {
    console.error('Failed to run validation suite:', err);
  }
}

function renderValidationTable(data) {
  const badge = document.getElementById('val-pass-rate-badge');
  if (badge && data.pass_rate_pct !== undefined) {
    badge.textContent = `${data.pass_rate_pct.toFixed(1)}%`;
  }

  const tbody = document.getElementById('val-table-body');
  if (!tbody || !data.results) return;

  tbody.innerHTML = data.results.map(r => {
    const isPass = r.passed;
    const passClass = isPass ? 'status-pill pass' : 'status-pill fail';
    const statusText = isPass ? 'PASS' : 'FAIL';
    const riskColor = r.predicted_label === 'Safe' ? 'var(--risk-safe-text)' : (r.predicted_label === 'Caution' ? 'var(--risk-caution-text)' : 'var(--risk-high-text)');

    return `
      <tr>
        <td class="mono-num">${escapeHtml(r.id)}</td>
        <td><strong>${escapeHtml(r.name)}</strong></td>
        <td><span style="font-size:12px; color:var(--text-muted);">${escapeHtml(r.category)}</span></td>
        <td>${escapeHtml(r.expected_label)}</td>
        <td><span style="color:${riskColor}; font-weight:700;">${escapeHtml(r.predicted_label)}</span></td>
        <td><span class="status-pill" style="background:var(--surface-muted); color:var(--text);">${escapeHtml(r.state_machine_state)}</span></td>
        <td><span class="${passClass}">${statusText}</span></td>
        <td style="font-size:12px; color:var(--text-muted);">${escapeHtml(r.engineering_note)}</td>
      </tr>
    `;
  }).join('');
}

function updateThresholdSliders() {
  const cSlider = document.getElementById('slider-caution');
  const hSlider = document.getElementById('slider-high');
  if (cSlider) setText('val-thresh-c-label', cSlider.value);
  if (hSlider) setText('val-thresh-h-label', hSlider.value);
}

async function applyThresholds() {
  const cSlider = document.getElementById('slider-caution');
  const hSlider = document.getElementById('slider-high');
  const msgEl = document.getElementById('thresh-save-msg');

  if (!cSlider || !hSlider) return;

  try {
    const res = await fetch('/api/threshold', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        caution_threshold: parseFloat(cSlider.value),
        high_risk_threshold: parseFloat(hSlider.value)
      })
    });
    const data = await res.json();
    if (msgEl) {
      msgEl.textContent = `Saved: Caution=${data.caution_threshold}, High-Risk=${data.high_risk_threshold}`;
      setTimeout(() => { msgEl.textContent = ''; }, 3000);
    }
  } catch (err) {
    if (msgEl) msgEl.textContent = `Error: ${err.message}`;
  }
}

function switchStakeholder(type) {
  document.querySelectorAll('.segmented-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.stakeholder-view-card').forEach(c => c.style.display = 'none');

  const btn = document.getElementById(`stk-btn-${type}`);
  const card = document.getElementById(`stk-view-${type}`);
  if (btn) btn.classList.add('active');
  if (card) card.style.display = 'block';
}

// ---------------------------------------------------------------------------
// 10. Accordion & About Modal Handlers
// ---------------------------------------------------------------------------

function toggleAccordion(btn) {
  const body = btn.nextElementSibling;
  const isExpanded = body.style.display === 'block';
  body.style.display = isExpanded ? 'none' : 'block';
  const icon = btn.querySelector('svg');
  if (icon) {
    icon.style.transform = isExpanded ? 'rotate(0deg)' : 'rotate(180deg)';
    icon.style.transition = 'transform 200ms ease';
  }
}

function openAboutModal() {
  const modal = document.getElementById('about-modal');
  if (modal) modal.style.display = 'flex';
}

function closeAboutModal() {
  const modal = document.getElementById('about-modal');
  if (modal) modal.style.display = 'none';
}

// ---------------------------------------------------------------------------
// 11. Lifecycle Initialization
// ---------------------------------------------------------------------------

window.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initMap();
  connectWebSocket();
});
