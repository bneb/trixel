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

let wasmReady = false;
let scanning = false;
let scanTimer = null;
let prismScanner = null;
let frameIndex = 0;

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
    }
}

// ---- Camera ----
async function startCamera() {
    try {
        setStatus('loading', 'Starting camera...');
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 1920, min: 1280 },
                height: { ideal: 1080, min: 720 },
            },
            audio: false,
        });
        video.srcObject = stream;
        await video.play();
        setStatus('scanning', 'Align code within frame');
        scanning = true;
        startScanLoop();
    } catch (e) {
        console.warn('Camera access denied or unavailable:', e);
        setStatus('ready', 'No camera — use Upload');
    }
}

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

            // Multi-scale candidate windows:
            // Frame 0: Exact reticle (1.0x)
            // Frame 1: Tight reticle (0.9x)
            // Frame 2: Wide reticle (1.15x)
            // Frame 3: Center 70% of camera
            const scales = [1.0, 0.9, 1.15, 1.3];
            const currentScale = scales[frameIndex % scales.length];

            const crop = getReticleCrop(currentScale);
            if (crop && crop.sw > 10 && crop.sh > 10) {
                offscreenCtx.drawImage(
                    video,
                    crop.sx, crop.sy, crop.sw, crop.sh,
                    0, 0, SCAN_RES, SCAN_RES
                );

                const imgData = offscreenCtx.getImageData(0, 0, SCAN_RES, SCAN_RES);
                const prismResult = prismScanner.scan_frame(imgData.data, SCAN_RES, SCAN_RES);

                if (prismResult) {
                    debugOverlay.style.display = 'block';
                    debugOverlay.textContent = '✓ ' + prismResult;
                    debugOverlay.style.color = '#00ff88';
                    onDecodeSuccess(prismResult);
                    return;
                }
            }
        } catch (err) {
            debugOverlay.style.display = 'block';
            debugOverlay.textContent = 'FRAME_ERROR: ' + err;
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
            setStatus('error', 'No pattern detected');
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
