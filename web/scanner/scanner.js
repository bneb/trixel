// =========================================================================
// PrismCode Universal Optical Scanner (Zero-Neural High-Speed WASM Engine)
// Continuous HVS Residual Modulation + QC-LDPC Soft Belief Propagation
// =========================================================================

import initPrism, { PrismScanner } from './prism_pkg/prism_wasm.js';

// ---- DOM Elements ----
const video = document.getElementById('camera');
const viewfinder = document.getElementById('viewfinder');
const scanFrame = document.getElementById('scan-frame');
const overlay = document.getElementById('overlay');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const resultDiv = document.getElementById('result');
const resultVal = document.getElementById('result-value');
const resultOpen = document.getElementById('result-open');
const resultCopy = document.getElementById('result-copy');
const resultRescan = document.getElementById('result-rescan');
const uploadInput = document.getElementById('upload-input');
const scanLine = document.getElementById('scan-line');
const debugOverlay = document.getElementById('debug-overlay');
const cameraPrompt = document.getElementById('camera-prompt');
const startCamBtn = document.getElementById('start-cam-btn');

let wasmReady = false;
let scanning = false;
let scanTimer = null;
let prismScanner = null;
let frameIndex = 0;
let framesScanned = 0;
let lastLogTime = 0;

// Standard square sampling resolution for reticle scan (divisible by 32 & 24)
const SCAN_RES = 384;
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = SCAN_RES;
offscreenCanvas.height = SCAN_RES;
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

// ---- Initialize WASM ----
async function initWasm() {
    try {
        setStatus('loading', 'Loading PrismCode decoder...');
        await initPrism();
        prismScanner = new PrismScanner();
        wasmReady = true;
        setStatus('ready', 'PrismCode decoder ready');
    } catch (e) {
        setStatus('error', 'Failed to load decoder');
        console.error('WASM init failed:', e);
        showDebug(`WASM_INIT_ERROR: ${e.message || e}`);
    }
}

function showDebug(msg, color = '#ff4444') {
    if (!debugOverlay) return;
    debugOverlay.style.display = 'block';
    debugOverlay.style.color = color;
    debugOverlay.textContent = msg;
}

// ---- Clean Camera Teardown ----
function stopCamera() {
    stopScanLoop();
    if (video.srcObject) {
        try {
            const stream = video.srcObject;
            if (stream && stream.getTracks) {
                stream.getTracks().forEach(track => track.stop());
            }
        } catch (e) {
            console.warn('Error stopping stream tracks:', e);
        }
        video.srcObject = null;
    }
}

// ---- Android & iOS Resilient Camera Starter ----
async function startCamera() {
    stopCamera();
    setStatus('loading', 'Starting camera...');
    
    // Ensure critical mobile video element attributes
    video.muted = true;
    video.playsInline = true;
    video.setAttribute('playsinline', '');
    video.setAttribute('webkit-playsinline', '');
    video.setAttribute('muted', '');
    video.setAttribute('autoplay', '');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        cameraPrompt.classList.remove('hidden');
        setStatus('error', 'Camera not supported — use Upload');
        showDebug('getUserMedia not supported in this browser context');
        return;
    }

    let stream = null;
    const constraintList = [
        // 1. Back environment camera with ideal resolution
        { video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
        // 2. Simple environment camera without size constraints
        { video: { facingMode: 'environment' }, audio: false },
        // 3. Fallback to any available camera (front/back)
        { video: true, audio: false }
    ];

    for (const constraints of constraintList) {
        try {
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            if (stream) break;
        } catch (err) {
            console.warn('Constraint attempt failed:', constraints, err);
        }
    }

    if (!stream) {
        cameraPrompt.classList.remove('hidden');
        setStatus('error', 'Camera permission denied or blocked');
        showDebug('Permission denied. Tap the 🔒 icon in Chrome to allow camera access.');
        return;
    }

    // Handle OS suspending camera track on sleep
    stream.getVideoTracks().forEach(track => {
        track.onended = () => {
            console.warn('Camera track ended by OS/sleep');
            stopCamera();
            cameraPrompt.classList.remove('hidden');
            setStatus('ready', 'Tap to re-enable camera');
        };
    });

    cameraPrompt.classList.add('hidden');
    video.srcObject = stream;

    // Wait for Android Chrome metadata before calling play()
    await new Promise((resolve) => {
        if (video.readyState >= 2 && video.videoWidth > 0) {
            return resolve();
        }
        video.onloadedmetadata = () => resolve();
        video.onloadeddata = () => resolve();
        setTimeout(resolve, 800);
    });

    try {
        await video.play();
    } catch (e) {
        console.warn('video.play() rejected:', e);
    }

    setStatus('scanning', 'Align code within frame');
    scanning = true;
    startScanLoop();
}

startCamBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    startCamera();
});

viewfinder?.addEventListener('click', () => {
    if (!scanning && resultDiv.classList.contains('hidden')) {
        startCamera();
    }
});

// Automatic wake-up when returning from lock screen or other apps
document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState === 'visible') {
        if (resultDiv.classList.contains('hidden') && wasmReady) {
            console.log('App resumed from sleep, re-acquiring camera...');
            await startCamera();
        }
    } else {
        stopCamera();
    }
});

window.addEventListener('focus', async () => {
    if (document.visibilityState === 'visible' && !scanning && resultDiv.classList.contains('hidden') && wasmReady) {
        await startCamera();
    }
});

// Calculate video crop box corresponding to on-screen reticle
function getReticleCrop(scaleFactor = 1.0) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;

    const vRect = viewfinder.getBoundingClientRect();
    const fRect = scanFrame.getBoundingClientRect();

    const vAspect = vw / vh;
    const elAspect = vRect.width / vRect.height;

    let scale, renderW, renderH, offsetX, offsetY;
    if (elAspect > vAspect) {
        scale = vw / vRect.width;
        renderW = vRect.width;
        renderH = vRect.width / vAspect;
        offsetX = 0;
        offsetY = (renderH - vRect.height) / 2;
    } else {
        scale = vh / vRect.height;
        renderH = vRect.height;
        renderW = vRect.height * vAspect;
        offsetX = (renderW - vRect.width) / 2;
        offsetY = 0;
    }

    const relX = (fRect.left - vRect.left + offsetX);
    const relY = (fRect.top - vRect.top + offsetY);
    const relW = fRect.width;
    const relH = fRect.height;

    const centerX = relX + relW / 2;
    const centerY = relY + relH / 2;
    const scaledW = relW * scaleFactor;
    const scaledH = relH * scaleFactor;

    let sx = (centerX - scaledW / 2) * scale;
    let sy = (centerY - scaledH / 2) * scale;
    let sw = scaledW * scale;
    let sh = scaledH * scale;

    sx = Math.max(0, Math.min(vw - 1, sx));
    sy = Math.max(0, Math.min(vh - 1, sy));
    sw = Math.min(vw - sx, sw);
    sh = Math.min(vh - sy, sh);

    return { sx, sy, sw, sh };
}

// ---- Scan Loop ----
function startScanLoop() {
    if (scanTimer) return;

    function tick() {
        if (!scanning || !wasmReady || !prismScanner) {
            scanTimer = null;
            return;
        }

        try {
            const vw = video.videoWidth;
            const vh = video.videoHeight;
            if (vw === 0 || vh === 0) {
                scanTimer = requestAnimationFrame(tick);
                return;
            }

            frameIndex++;
            framesScanned++;

            // Candidate 1: Reticle crop with scale jitter
            const scales = [1.0, 0.85, 1.15, 1.30, 0.70];
            const currentScale = scales[frameIndex % scales.length];
            const crop = getReticleCrop(currentScale);

            if (crop && crop.sw > 20 && crop.sh > 20) {
                offscreenCtx.drawImage(
                    video,
                    crop.sx, crop.sy, crop.sw, crop.sh,
                    0, 0, SCAN_RES, SCAN_RES
                );

                const imgData = offscreenCtx.getImageData(0, 0, SCAN_RES, SCAN_RES);
                const prismResult = prismScanner.scan_frame(imgData.data, SCAN_RES, SCAN_RES);

                if (prismResult) {
                    showDebug('✓ ' + prismResult, '#00ff88');
                    onDecodeSuccess(prismResult);
                    return;
                }
            }

            // Candidate 2: Full-sensor central square crop (for whole-screen alignment)
            if (frameIndex % 2 === 0) {
                const minDim = Math.min(vw, vh);
                const csx = (vw - minDim) / 2;
                const csy = (vh - minDim) / 2;
                offscreenCtx.drawImage(
                    video,
                    csx, csy, minDim, minDim,
                    0, 0, SCAN_RES, SCAN_RES
                );
                const centerData = offscreenCtx.getImageData(0, 0, SCAN_RES, SCAN_RES);
                const centerResult = prismScanner.scan_frame(centerData.data, SCAN_RES, SCAN_RES);
                if (centerResult) {
                    showDebug('✓ ' + centerResult, '#00ff88');
                    onDecodeSuccess(centerResult);
                    return;
                }
            }

            // Periodic live telemetry to debug overlay
            const now = performance.now();
            if (now - lastLogTime > 1200) {
                lastLogTime = now;
                showDebug(`Camera: ${vw}x${vh} | Scanned ${framesScanned} frames | Looking for PrismCode`, '#88ccff');
            }
        } catch (err) {
            showDebug('FRAME_ERROR: ' + (err.message || err));
        }

        // 10 fps scanning rate (100ms interval)
        scanTimer = setTimeout(() => requestAnimationFrame(tick), 100);
    }

    scanTimer = requestAnimationFrame(tick);
}

function stopScanLoop() {
    scanning = false;
    if (scanTimer) {
        clearTimeout(scanTimer);
        cancelAnimationFrame(scanTimer);
        scanTimer = null;
    }
}

// ---- Decode Result ----
function onDecodeSuccess(value) {
    stopScanLoop();
    scanLine.style.animationPlayState = 'paused';

    if (navigator.vibrate) navigator.vibrate(100);

    resultVal.textContent = value;

    const isUrl = /^https?:\/\//i.test(value);
    if (isUrl) {
        resultOpen.href = value;
        resultOpen.classList.remove('hidden');
    } else {
        resultOpen.classList.add('hidden');
    }

    resultDiv.classList.remove('hidden');
    setStatus('success', 'Decoded via PrismCode!');
}

function rescan() {
    resultDiv.classList.add('hidden');
    scanLine.style.animationPlayState = 'running';
    scanning = true;
    setStatus('scanning', 'Align code within frame');
    startScanLoop();
}

// ---- File Upload ----
uploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file || !wasmReady || !prismScanner) return;

    stopScanLoop();
    setStatus('loading', 'Decoding image...');

    try {
        const img = new Image();
        const imgLoaded = new Promise((res, rej) => {
            img.onload = () => res();
            img.onerror = rej;
        });
        img.src = URL.createObjectURL(file);
        await imgLoaded;

        const offscreen = document.createElement('canvas');
        offscreen.width = img.naturalWidth;
        offscreen.height = img.naturalHeight;
        const octx = offscreen.getContext('2d', { willReadFrequently: true });
        octx.drawImage(img, 0, 0);
        const imgData = octx.getImageData(0, 0, offscreen.width, offscreen.height);

        const prismRes = prismScanner.scan_frame(imgData.data, offscreen.width, offscreen.height);
        if (prismRes) {
            onDecodeSuccess(prismRes);
        } else {
            setStatus('error', 'No pattern detected in upload');
            setTimeout(() => {
                if (scanning) setStatus('scanning', 'Align code within frame');
                else setStatus('ready', 'Ready');
            }, 3000);
        }
    } catch (err) {
        setStatus('error', `Decode failed: ${err}`);
        setTimeout(() => {
            if (scanning) setStatus('scanning', 'Align code within frame');
            else setStatus('ready', 'Ready');
        }, 3000);
    }

    uploadInput.value = '';
});

// ---- Copy Button ----
resultCopy.addEventListener('click', async () => {
    try {
        await navigator.clipboard.writeText(resultVal.textContent);
        resultCopy.textContent = 'Copied!';
        setTimeout(() => { resultCopy.textContent = 'Copy'; }, 1500);
    } catch (_) {
        resultCopy.textContent = 'Failed';
        setTimeout(() => { resultCopy.textContent = 'Copy'; }, 1500);
    }
});

// ---- Rescan Button ----
resultRescan.addEventListener('click', rescan);

// ---- Status Helper ----
function setStatus(state, text) {
    statusDot.className = '';
    switch (state) {
        case 'loading': break;
        case 'scanning': break;
        case 'ready': break;
        case 'error': statusDot.classList.add('error'); break;
        case 'success': statusDot.classList.add('success'); break;
    }
    statusText.textContent = text;
}

// ---- Boot ----
(async () => {
    await initWasm();
    await startCamera();
})();
