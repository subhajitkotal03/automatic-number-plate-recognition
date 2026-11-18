const API_BASE =
  window.location.protocol === 'file:' || (window.location.port && window.location.port !== '5000' && window.location.port !== '1203')
    ? 'http://127.0.0.1:1203'
    : '';

// Camera Elements
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

// Mode Switcher Elements
const modeTabCamera = document.querySelector('#modeTabCamera');
const modeTabVideo = document.querySelector('#modeTabVideo');
const liveCameraSection = document.querySelector('#liveCameraSection');
const videoUploadSection = document.querySelector('#videoUploadSection');

// Video File Upload Elements
const videoFileInput = document.querySelector('#videoFileInput');
const videoDropzone = document.querySelector('#videoDropzone');
const browseVideoBtn = document.querySelector('#browseVideoBtn');
const selectedFileInfo = document.querySelector('#selectedFileInfo');
const selectedFileName = document.querySelector('#selectedFileName');
const selectedFileSize = document.querySelector('#selectedFileSize');
const startVideoBtn = document.querySelector('#startVideoBtn');
const cancelVideoBtn = document.querySelector('#cancelVideoBtn');
const resetVideoBtn = document.querySelector('#resetVideoBtn');

// Video Processing Elements
const videoUploadCard = document.querySelector('#videoUploadCard');
const videoProcessingCard = document.querySelector('#videoProcessingCard');
const videoJobTitle = document.querySelector('#videoJobTitle');
const videoStatusBadge = document.querySelector('#videoStatusBadge');
const videoProgressBar = document.querySelector('#videoProgressBar');
const videoProgressText = document.querySelector('#videoProgressText');
const videoFramesText = document.querySelector('#videoFramesText');
const videoFpsText = document.querySelector('#videoFpsText');
const videoDurationText = document.querySelector('#videoDurationText');
const videoPlatesCountText = document.querySelector('#videoPlatesCountText');
const videoProcessingPreview = document.querySelector('#videoProcessingPreview');
const videoActionTools = document.querySelector('#videoActionTools');
const downloadAnnotatedVideoBtn = document.querySelector('#downloadAnnotatedVideoBtn');
const exportVideoCsvBtn = document.querySelector('#exportVideoCsvBtn');
const exportVideoJsonBtn = document.querySelector('#exportVideoJsonBtn');
const videoPlatesGallery = document.querySelector('#videoPlatesGallery');
const galleryPlatesCount = document.querySelector('#galleryPlatesCount');

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

// Video Job State
let selectedVideoFile = null;
let currentJobId = null;
let jobPollTimer = null;
let previewPollTimer = null;
let currentJobPlates = [];

// ============================================================================
// Audio Feedback (Gentle Chime on Detection)
// ============================================================================
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

// ============================================================================
// Operating Mode Switching (Camera vs. Video Upload)
// ============================================================================
function switchMode(mode) {
  if (mode === 'camera') {
    modeTabCamera?.classList.add('active');
    modeTabVideo?.classList.remove('active');
    if (liveCameraSection) liveCameraSection.style.display = 'block';
    if (videoUploadSection) videoUploadSection.style.display = 'none';
  } else {
    // Stop live camera if running to prioritize GPU for video analysis
    stopCamera();
    modeTabCamera?.classList.remove('active');
    modeTabVideo?.classList.add('active');
    if (liveCameraSection) liveCameraSection.style.display = 'none';
    if (videoUploadSection) videoUploadSection.style.display = 'block';
  }
}

if (modeTabCamera) modeTabCamera.addEventListener('click', () => switchMode('camera'));
if (modeTabVideo) modeTabVideo.addEventListener('click', () => switchMode('video'));

// ============================================================================
// Live Camera Controls
// ============================================================================
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

// ============================================================================
// Video Upload & File Dropzone
// ============================================================================
function handleFileSelected(file) {
  if (!file) return;
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  const validExts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'];
  if (!validExts.includes(ext)) {
    alert(`Please select a supported video file (${validExts.join(', ')}).`);
    return;
  }

  selectedVideoFile = file;
  if (selectedFileName) selectedFileName.textContent = file.name;
  if (selectedFileSize) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    selectedFileSize.textContent = `(${sizeMb} MB)`;
  }
  if (selectedFileInfo) selectedFileInfo.classList.remove('d-none');
  if (startVideoBtn) startVideoBtn.disabled = false;
}

if (browseVideoBtn) {
  browseVideoBtn.addEventListener('click', () => videoFileInput?.click());
}

if (videoFileInput) {
  videoFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });
}

if (videoDropzone) {
  ['dragenter', 'dragover'].forEach((eventName) => {
    videoDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      videoDropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    videoDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      videoDropzone.classList.remove('dragover');
    });
  });

  videoDropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const file = dt?.files?.[0];
    if (file) handleFileSelected(file);
  });
}

// ============================================================================
// Video Processing & Job Lifecycle
// ============================================================================
async function startVideoAnalysis() {
  if (!selectedVideoFile) return;

  const formData = new FormData();
  formData.append('video', selectedVideoFile);

  if (startVideoBtn) startVideoBtn.disabled = true;
  if (cancelVideoBtn) cancelVideoBtn.classList.remove('d-none');
  if (resetVideoBtn) resetVideoBtn.classList.add('d-none');
  if (videoUploadCard) videoUploadCard.classList.add('d-none');
  if (videoProcessingCard) videoProcessingCard.classList.remove('d-none');

  if (videoJobTitle) videoJobTitle.textContent = `Analyzing: ${selectedVideoFile.name}`;
  if (videoStatusBadge) {
    videoStatusBadge.textContent = 'Uploading & preparing model';
    videoStatusBadge.className = 'badge badge-warning';
  }
  if (videoProgressBar) videoProgressBar.style.width = '2%';
  if (videoProgressText) videoProgressText.textContent = 'Uploading...';
  if (videoProcessingPreview) videoProcessingPreview.style.display = 'block';
  if (videoPlatesGallery) videoPlatesGallery.replaceChildren();
  if (galleryPlatesCount) galleryPlatesCount.textContent = '0';
  currentJobPlates = [];

  try {
    const res = await fetch(`${API_BASE}/api/upload_video`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Upload failed');
    }

    const job = await res.json();
    currentJobId = job.job_id;

    // Start polling job status and preview frame
    if (jobPollTimer) clearInterval(jobPollTimer);
    jobPollTimer = setInterval(pollJobStatus, 700);

    if (previewPollTimer) clearInterval(previewPollTimer);
    previewPollTimer = setInterval(pollPreviewFrame, 350);
  } catch (error) {
    alert(`Could not start video analysis: ${error.message}`);
    resetVideoUI();
  }
}

async function pollJobStatus() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`${API_BASE}/api/video_job/${currentJobId}/status?t=${Date.now()}`);
    if (!res.ok) return;
    const data = await res.json();

    // Update progress bar & percentage
    const pct = Math.max(0, Math.min(100, data.progress || 0));
    if (videoProgressBar) videoProgressBar.style.width = `${pct}%`;
    if (videoProgressText) videoProgressText.textContent = `${pct.toFixed(1)}%`;

    // Update metrics
    if (videoFramesText) videoFramesText.textContent = `${data.current_frame || 0} / ${data.total_frames || '--'}`;
    if (videoFpsText) videoFpsText.textContent = data.processing_fps ? `${data.processing_fps.toFixed(1)} FPS` : '-- FPS';
    if (videoDurationText) videoDurationText.textContent = data.duration_seconds ? `${data.duration_seconds.toFixed(1)}s` : '--';
    if (videoPlatesCountText) videoPlatesCountText.textContent = `${data.total_plates_found || 0}`;

    // Render detected plates gallery cards
    if (Array.isArray(data.plates)) {
      renderPlatesGallery(data.plates);
    }

    // Status management
    if (data.status === 'processing') {
      if (videoStatusBadge) {
        videoStatusBadge.textContent = `Processing (${pct.toFixed(0)}%)`;
        videoStatusBadge.className = 'badge badge-warning';
      }
    } else if (data.status === 'completed') {
      finishJob(data, 'completed');
    } else if (data.status === 'cancelled') {
      finishJob(data, 'cancelled');
    } else if (data.status === 'error') {
      finishJob(data, 'error');
    }
  } catch (err) {
    // transient network error while polling
  }
}

function pollPreviewFrame() {
  if (!currentJobId || !videoProcessingPreview) return;
  videoProcessingPreview.src = `${API_BASE}/api/video_job/${currentJobId}/preview?t=${Date.now()}`;
}

function renderPlatesGallery(plates) {
  if (!videoPlatesGallery) return;
  if (plates.length === currentJobPlates.length) return; // avoid redraw if no change

  currentJobPlates = plates;
  if (galleryPlatesCount) galleryPlatesCount.textContent = plates.length;
  videoPlatesGallery.replaceChildren();

  plates.forEach((p) => {
    const card = document.createElement('div');
    card.className = 'plate-card';

    const img = document.createElement('img');
    img.className = 'plate-card-thumb';
    img.src = `${API_BASE}${p.crop_url}`;
    img.alt = `Plate ${p.plate_number}`;

    const body = document.createElement('div');
    body.className = 'plate-card-body';

    const title = document.createElement('div');
    title.className = 'plate-card-title';
    title.textContent = p.plate_number;

    const meta1 = document.createElement('div');
    meta1.className = 'plate-card-meta';
    meta1.innerHTML = `<span>⏱️ <strong>${p.timestamp}</strong></span> <span>🎯 OCR: <strong>${Math.round(p.ocr_confidence * 100)}%</strong></span>`;

    const meta2 = document.createElement('div');
    meta2.className = 'plate-card-meta';
    meta2.innerHTML = `<span>📐 YOLO: <strong>${Math.round(p.detection_confidence * 100)}%</strong></span> <span class="badge badge-secondary" style="font-size: 0.72rem;">${p.detections_count || 1} hits</span>`;

    body.append(title, meta1, meta2);
    card.append(img, body);
    videoPlatesGallery.append(card);
  });
}

function finishJob(data, status) {
  if (jobPollTimer) clearInterval(jobPollTimer);
  if (previewPollTimer) clearInterval(previewPollTimer);
  jobPollTimer = null;
  previewPollTimer = null;

  if (cancelVideoBtn) cancelVideoBtn.classList.add('d-none');
  if (resetVideoBtn) resetVideoBtn.classList.remove('d-none');

  if (status === 'completed') {
    if (videoStatusBadge) {
      videoStatusBadge.textContent = 'Analysis Complete ✓';
      videoStatusBadge.className = 'badge badge-success';
    }
    if (videoProgressBar) videoProgressBar.style.width = '100%';
    if (videoProgressText) videoProgressText.textContent = '100%';

    if (videoActionTools) videoActionTools.classList.remove('d-none');
    if (downloadAnnotatedVideoBtn && data.has_annotated_video) {
      downloadAnnotatedVideoBtn.href = `${API_BASE}/api/video_job/${currentJobId}/download`;
    }

    playChime();
    loadRecords(); // reload bottom table to show newly logged video plates
  } else if (status === 'cancelled') {
    if (videoStatusBadge) {
      videoStatusBadge.textContent = 'Cancelled by user';
      videoStatusBadge.className = 'badge badge-secondary';
    }
  } else if (status === 'error') {
    if (videoStatusBadge) {
      videoStatusBadge.textContent = `Error: ${data.error || 'Processing failed'}`;
      videoStatusBadge.className = 'badge badge-danger';
    }
  }
}

async function cancelVideoAnalysis() {
  if (!currentJobId) return;
  try {
    await fetch(`${API_BASE}/api/video_job/${currentJobId}/cancel`, { method: 'POST' });
    if (videoStatusBadge) videoStatusBadge.textContent = 'Cancelling...';
  } catch (err) {
    // ignore
  }
}

function resetVideoUI() {
  if (jobPollTimer) clearInterval(jobPollTimer);
  if (previewPollTimer) clearInterval(previewPollTimer);
  currentJobId = null;
  selectedVideoFile = null;

  if (videoFileInput) videoFileInput.value = '';
  if (selectedFileInfo) selectedFileInfo.classList.add('d-none');
  if (startVideoBtn) startVideoBtn.disabled = true;
  if (cancelVideoBtn) cancelVideoBtn.classList.add('d-none');
  if (resetVideoBtn) resetVideoBtn.classList.add('d-none');

  if (videoUploadCard) videoUploadCard.classList.remove('d-none');
  if (videoProcessingCard) videoProcessingCard.classList.add('d-none');
  if (videoActionTools) videoActionTools.classList.add('d-none');
}

if (startVideoBtn) startVideoBtn.addEventListener('click', startVideoAnalysis);
if (cancelVideoBtn) cancelVideoBtn.addEventListener('click', cancelVideoAnalysis);
if (resetVideoBtn) resetVideoBtn.addEventListener('click', resetVideoUI);

// ============================================================================
// Table Rendering & Search
// ============================================================================
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

function addSourceCell(row, source) {
  const cell = document.createElement('td');
  const span = document.createElement('span');
  const isVideo = source && source.toLowerCase().includes('video');
  span.className = `badge-source ${isVideo ? 'video' : 'camera'}`;
  span.textContent = source || 'Camera';
  cell.append(span);
  row.append(cell);
}

function renderRecordsTable() {
  if (!recordsBody) return;
  recordsBody.replaceChildren();

  const query = (searchPlateInput?.value || '').trim().toUpperCase();
  const filtered = query
    ? allRecords.filter((r) => (r.plate_number || '').toUpperCase().includes(query))
    : allRecords;

  const displayList = filtered.slice(-20).reverse();

  if (displayList.length === 0) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 5;
    cell.className = 'text-secondary';
    cell.textContent = query
      ? `No records found matching "${query}"`
      : 'No saved records found.';
    row.append(cell);
    recordsBody.append(row);
    return;
  }

  displayList.forEach((record) => {
    const row = document.createElement('tr');
    addCell(row, record.plate_number || 'Unknown', true);
    addSourceCell(row, record.source || 'Camera');
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
      cell.colSpan = 5;
      cell.className = 'text-secondary';
      cell.textContent = error.message === 'No saved records yet'
        ? error.message
        : 'Could not load saved records from Flask.';
      row.append(cell);
      recordsBody.append(row);
    }
  }
}

// ============================================================================
// Export Utilities (CSV & JSON)
// ============================================================================
function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportToCsv() {
  if (allRecords.length === 0) {
    alert('No plate records available to export.');
    return;
  }
  const headers = ['Plate Number', 'Source', 'Timestamp', 'OCR Confidence', 'Detection Confidence'];
  const rows = allRecords.map((r) => [
    `"${r.plate_number || ''}"`,
    `"${r.source || 'Camera'}"`,
    `"${r.timestamp || ''}"`,
    r.ocr_confidence || 0,
    r.detection_confidence || 0,
  ]);
  const csvContent = [headers.join(','), ...rows.map((row) => row.join(','))].join('\n');
  downloadFile(csvContent, `plate_records_${new Date().toISOString().slice(0, 10)}.csv`, 'text/csv;charset=utf-8;');
}

function exportToJson() {
  if (allRecords.length === 0) {
    alert('No plate records available to export.');
    return;
  }
  downloadFile(JSON.stringify(allRecords, null, 2), `plate_records_${new Date().toISOString().slice(0, 10)}.json`, 'application/json;charset=utf-8;');
}

function exportVideoPlatesCsv() {
  if (currentJobPlates.length === 0) {
    alert('No plates detected in this video to export.');
    return;
  }
  const headers = ['Plate Number', 'Video Timestamp', 'Frame', 'OCR Confidence', 'Detection Confidence', 'Total Hits'];
  const rows = currentJobPlates.map((p) => [
    `"${p.plate_number || ''}"`,
    `"${p.timestamp || ''}"`,
    p.frame_idx || 0,
    p.ocr_confidence || 0,
    p.detection_confidence || 0,
    p.detections_count || 1,
  ]);
  const csvContent = [headers.join(','), ...rows.map((row) => row.join(','))].join('\n');
  downloadFile(csvContent, `video_plates_${currentJobId || 'export'}.csv`, 'text/csv;charset=utf-8;');
}

function exportVideoPlatesJson() {
  if (currentJobPlates.length === 0) {
    alert('No plates detected in this video to export.');
    return;
  }
  downloadFile(JSON.stringify(currentJobPlates, null, 2), `video_plates_${currentJobId || 'export'}.json`, 'application/json;charset=utf-8;');
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
if (exportVideoCsvBtn) exportVideoCsvBtn.addEventListener('click', exportVideoPlatesCsv);
if (exportVideoJsonBtn) exportVideoJsonBtn.addEventListener('click', exportVideoPlatesJson);

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
