// =========================================================================
// PrismCode Universal Optical Scanner (Zero-Neural High-Speed WASM Engine)
// Continuous HVS Residual Modulation + QC-LDPC Soft Belief Propagation
// =========================================================================

import initPrism, { PrismScanner } from './prism_pkg/prism_wasm.js';

// ---- DOM Elements ----
const video = document.getElementById('camera');
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

// Maximum dimension for downscaling (saves WASM memory on hi-res cameras)
const MAX_SCAN_DIM = 720;

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
                width: { ideal: 1280 },
                height: { ideal: 720 },
            },
            audio: false,
        });
        video.srcObject = stream;
        await video.play();
        setStatus('scanning', 'Scanning for PrismCode...');
        scanning = true;
        startScanLoop();
    } catch (e) {
        console.warn('Camera access denied or unavailable:', e);
        setStatus('ready', 'No camera — use Upload');
    }
}

// ---- Scan Loop (RGBA raw-byte path) ----
function startScanLoop() {
    if (scanTimer) return;
    const ctx = overlay.getContext('2d', { willReadFrequently: true });

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

            // Downscale to MAX_SCAN_DIM to optimize frame rate
            const scale = Math.min(1, MAX_SCAN_DIM / Math.max(vw, vh));
            const w = Math.round(vw * scale);
            const h = Math.round(vh * scale);
            overlay.width = w;
            overlay.height = h;

            // Draw video frame to canvas
            ctx.drawImage(video, 0, 0, w, h);

            // Extract RGBA pixel buffer directly
            const imageData = ctx.getImageData(0, 0, w, h);

            // PrismCode Perceptual Soft Decoder
            const prismResult = prismScanner.scan_frame(imageData.data, w, h);
            if (prismResult) {
                debugOverlay.textContent = '✓ ' + prismResult;
                debugOverlay.style.color = '#00ff88';
                onDecodeSuccess(prismResult);
                return;
            }
        } catch (err) {
            debugOverlay.style.display = 'block';
            debugOverlay.textContent = 'FRAME_ERROR: ' + err;
        }

        // ~8 fps scan rate (125ms throttle)
        scanTimer = setTimeout(() => requestAnimationFrame(tick), 125);
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
    setStatus('scanning', 'Scanning for PrismCode...');
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
            setStatus('error', 'No PrismCode pattern found in image');
            setTimeout(() => {
                if (scanning) setStatus('scanning', 'Scanning...');
                else setStatus('ready', 'Ready');
            }, 3000);
        }
    } catch (err) {
        setStatus('error', `Decode failed: ${err}`);
        setTimeout(() => {
            if (scanning) setStatus('scanning', 'Scanning...');
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
