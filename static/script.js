const API_BASE =
  window.location.protocol === 'file:' || (window.location.port && window.location.port !== '5000')
    ? 'http://127.0.0.1:5000'
    : '';

const cameraPreview = document.querySelector('#cameraPreview');
const cameraFrame = document.querySelector('#cameraFrame');
const startButton = document.querySelector('#startButton');
const stopButton = document.querySelector('#stopButton');
const scannerStatus = document.querySelector('#scannerStatus');
const cameraLabel = document.querySelector('#cameraLabel');
const cameraDot = document.querySelector('#cameraDot');
const cameraFps = document.querySelector('#cameraFps');
const cameraError = document.querySelector('#cameraError');
const recordsBody = document.querySelector('#recordsBody');
const latestPlate = document.querySelector('#latestPlate');
const latestDetails = document.querySelector('#latestDetails');
const searchPlateInput = document.querySelector('#searchPlateInput');
const audioAlertToggle = document.querySelector('#audioAlertToggle');
const autoRefreshToggle = document.querySelector('#autoRefreshToggle');
const exportCsvBtn = document.querySelector('#exportCsvBtn');
const exportJsonBtn = document.querySelector('#exportJsonBtn');

// Stat counter elements
const totalScansEl = document.querySelector('#totalScans');
const avgConfidenceEl = document.querySelector('#avgConfidence');
const peakFpsEl = document.querySelector('#peakFps');

let statusTimer = null;
let recordsTimer = null;
let allRecords = [];
let lastKnownRecordTime = null;
let peakFps = 0.0;
let audioCtx = null;

// Synthesize a gentle confirmation chime when a plate is captured
function playChime() {
  if (!audioAlertToggle || !audioAlertToggle.checked) return;
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    if (!audioCtx) audioCtx = new AudioContext();
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
    osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.15); // A5

    gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.36);
  } catch (err) {
    // Audio context was blocked or not allowed by user policy
  }
}

function setLiveStatus() {
  if (cameraFrame) cameraFrame.classList.add('active');
  if (startButton) startButton.disabled = true;
  if (stopButton) stopButton.disabled = false;
  if (scannerStatus) {
    scannerStatus.textContent = 'Scanner starting';
    scannerStatus.className = 'badge badge-warning';
  }
  if (cameraLabel) cameraLabel.textContent = 'Loading YOLO on GPU';
  if (cameraDot) cameraDot.classList.add('live');
}

function startCamera() {
  if (cameraError) cameraError.classList.add('d-none');
  setLiveStatus();
  if (cameraPreview) {
    cameraPreview.src = `${API_BASE}/video_feed?time=${Date.now()}`;
  }
  if (statusTimer) window.clearInterval(statusTimer);
  statusTimer = window.setInterval(loadStatus, 1000);
}

function stopCamera() {
  if (statusTimer) window.clearInterval(statusTimer);
  if (cameraPreview) cameraPreview.removeAttribute('src');
  if (cameraFrame) cameraFrame.classList.remove('active');
  if (startButton) startButton.disabled = false;
  if (stopButton) stopButton.disabled = true;
  if (cameraFps) cameraFps.textContent = '-- FPS';
  if (scannerStatus) {
    scannerStatus.textContent = 'Scanner offline';
    scannerStatus.className = 'badge badge-secondary';
  }
  if (cameraLabel) cameraLabel.textContent = 'Scanner stopped';
  if (cameraDot) cameraDot.classList.remove('live');
}

async function loadStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/status`);
    if (!response.ok) return;
    const status = await response.json();
    if (cameraFps) {
      cameraFps.textContent = status.fps ? `${status.fps.toFixed(1)} FPS` : '-- FPS';
    }

    if (status.fps && status.fps > peakFps) {
      peakFps = status.fps;
      if (peakFpsEl) peakFpsEl.textContent = `${peakFps.toFixed(1)} FPS`;
    }

    if (status.streaming && scannerStatus) {
      scannerStatus.textContent = 'Model live';
      scannerStatus.className = 'badge badge-success';
      if (cameraLabel) {
        cameraLabel.textContent = `${status.gpu} • ${status.vram_gib} GiB`;
      }
    }
  } catch (error) {
    if (cameraError) {
      cameraError.textContent = 'Could not contact the Flask scanner.';
      cameraError.classList.remove('d-none');
    }
  }
}

function addCell(row, text, isBadge = false) {
  const cell = document.createElement('td');
  if (isBadge) {
    const span = document.createElement('span');
    span.className = 'plate-badge';
    span.textContent = text;
    cell.append(span);
  } else {
    cell.textContent = text;
  }
  row.append(cell);
}

function renderRecordsTable() {
  if (!recordsBody) return;
  recordsBody.replaceChildren();

  const query = (searchPlateInput?.value || '').trim().toUpperCase();
  const filtered = query
    ? allRecords.filter(r => (r.plate_number || '').toUpperCase().includes(query))
    : allRecords;

  const displayList = filtered.slice(-15).reverse();

  if (displayList.length === 0) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 4;
    cell.className = 'text-secondary';
    cell.textContent = query
      ? `No records found matching "${query}"`
      : 'No saved records found.';
    row.append(cell);
    recordsBody.append(row);
    return;
  }

  displayList.forEach(record => {
    const row = document.createElement('tr');
    addCell(row, record.plate_number || 'Unknown', true);
    addCell(row, record.timestamp ? new Date(record.timestamp).toLocaleString() : 'Unknown');
    addCell(row, record.ocr_confidence ? `${Math.round(record.ocr_confidence * 100)}%` : '--');
    addCell(row, record.detection_confidence ? `${Math.round(record.detection_confidence * 100)}%` : '--');
    recordsBody.append(row);
  });
}

function updateSummaryMetrics() {
  if (totalScansEl) {
    totalScansEl.textContent = allRecords.length;
  }
  if (avgConfidenceEl && allRecords.length > 0) {
    const totalConfidence = allRecords.reduce((acc, r) => acc + (r.ocr_confidence || 0), 0);
    const avg = Math.round((totalConfidence / allRecords.length) * 100);
    avgConfidenceEl.textContent = `${avg}%`;
  }
}

async function loadRecords() {
  try {
    const response = await fetch(`${API_BASE}/api/records?time=${Date.now()}`);
    if (!response.ok) throw new Error('No records file found');
    const records = await response.json();
    if (!Array.isArray(records)) return;

    // Check if new records arrived
    if (records.length > 0) {
      const latest = records[records.length - 1];
      if (lastKnownRecordTime && latest.timestamp !== lastKnownRecordTime) {
        playChime();
      }
      lastKnownRecordTime = latest.timestamp;

      if (latestPlate) latestPlate.textContent = latest.plate_number || 'Unknown';
      if (latestDetails) {
        latestDetails.textContent = latest.timestamp
          ? `Saved ${new Date(latest.timestamp).toLocaleString()}`
          : 'Saved record';
      }
    }

    allRecords = records;
    updateSummaryMetrics();
    renderRecordsTable();
  } catch (error) {
    if (recordsBody && allRecords.length === 0) {
      recordsBody.replaceChildren();
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 4;
      cell.className = 'text-secondary';
      cell.textContent = error.message === 'No saved records yet'
        ? error.message
        : 'Could not load saved records from Flask.';
      row.append(cell);
      recordsBody.append(row);
    }
  }
}

// Export Records as CSV
function exportToCsv() {
  if (allRecords.length === 0) {
    alert('No plate records available to export.');
    return;
  }
  const headers = ['Plate Number', 'Timestamp', 'OCR Confidence', 'Detection Confidence'];
  const rows = allRecords.map(r => [
    `"${r.plate_number || ''}"`,
    `"${r.timestamp || ''}"`,
    r.ocr_confidence || 0,
    r.detection_confidence || 0,
  ]);
  const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `plate_records_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Export Records as JSON
function exportToJson() {
  if (allRecords.length === 0) {
    alert('No plate records available to export.');
    return;
  }
  const jsonContent = JSON.stringify(allRecords, null, 2);
  const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `plate_records_${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}


// Event Listeners
if (startButton) startButton.addEventListener('click', startCamera);
if (stopButton) stopButton.addEventListener('click', stopCamera);

if (cameraPreview) {
  cameraPreview.addEventListener('error', () => {
    if (cameraError) {
      cameraError.textContent = 'The model stream could not start. Check the Flask terminal for the camera error.';
      cameraError.classList.remove('d-none');
    }
    stopCamera();
  });
}

const refreshBtn = document.querySelector('#refreshButton');
if (refreshBtn) refreshBtn.addEventListener('click', loadRecords);

if (searchPlateInput) {
  searchPlateInput.addEventListener('input', renderRecordsTable);
}

if (exportCsvBtn) exportCsvBtn.addEventListener('click', exportToCsv);
if (exportJsonBtn) exportJsonBtn.addEventListener('click', exportToJson);

if (autoRefreshToggle) {
  autoRefreshToggle.addEventListener('change', () => {
    if (autoRefreshToggle.checked) {
      if (!recordsTimer) recordsTimer = window.setInterval(loadRecords, 2000);
    } else {
      if (recordsTimer) {
        window.clearInterval(recordsTimer);
        recordsTimer = null;
      }
    }
  });
}

window.addEventListener('beforeunload', stopCamera);

// Initialize
loadRecords();
recordsTimer = window.setInterval(loadRecords, 2500);
