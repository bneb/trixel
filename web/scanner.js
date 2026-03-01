// =========================================================================
// Trixel Scanner — Camera → WASM Decode Pipeline
// =========================================================================

import init, { decode_png_auto } from './pkg/trixel_scanner.js';

// ---- DOM Elements ----
const video      = document.getElementById('camera');
const overlay    = document.getElementById('overlay');
const statusDot  = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const resultDiv  = document.getElementById('result');
const resultVal  = document.getElementById('result-value');
const resultOpen = document.getElementById('result-open');
const resultCopy = document.getElementById('result-copy');
const resultRescan = document.getElementById('result-rescan');
const uploadInput  = document.getElementById('upload-input');
const scanLine     = document.getElementById('scan-line');

let wasmReady = false;
let scanning  = false;
let scanTimer = null;

// ---- Initialize WASM ----
async function initWasm() {
    try {
        setStatus('loading', 'Loading decoder...');
        await init();
        wasmReady = true;
        setStatus('ready', 'Decoder ready');
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
                width:  { ideal: 1280 },
                height: { ideal: 720 },
            },
            audio: false,
        });
        video.srcObject = stream;
        await video.play();
        setStatus('scanning', 'Scanning...');
        scanning = true;
        startScanLoop();
    } catch (e) {
        console.warn('Camera access denied or unavailable:', e);
        setStatus('ready', 'No camera — use Upload');
    }
}

// ---- Scan Loop ----
function startScanLoop() {
    if (scanTimer) return;
    const ctx = overlay.getContext('2d', { willReadFrequently: true });

    function tick() {
        if (!scanning || !wasmReady) {
            scanTimer = null;
            return;
        }

        try {
            // Size canvas to video frame
            const w = video.videoWidth;
            const h = video.videoHeight;
            if (w === 0 || h === 0) {
                scanTimer = requestAnimationFrame(tick);
                return;
            }
            overlay.width  = w;
            overlay.height = h;

            // Capture frame → PNG bytes
            ctx.drawImage(video, 0, 0, w, h);
            overlay.toBlob((blob) => {
                if (!blob || !scanning) return;

                blob.arrayBuffer().then((buf) => {
                    if (!scanning) return;
                    try {
                        const result = decode_png_auto(new Uint8Array(buf));
                        onDecodeSuccess(result);
                    } catch (_) {
                        // Decode failed — continue scanning silently
                    }
                });
            }, 'image/png');
        } catch (_) {
            // Frame capture error — continue
        }

        // ~4 fps scan rate
        scanTimer = setTimeout(() => requestAnimationFrame(tick), 250);
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

    // Vibrate on success (mobile)
    if (navigator.vibrate) navigator.vibrate(100);

    resultVal.textContent = value;

    // If it looks like a URL, show the Open button
    const isUrl = /^https?:\/\//i.test(value);
    if (isUrl) {
        resultOpen.href = value;
        resultOpen.classList.remove('hidden');
    } else {
        resultOpen.classList.add('hidden');
    }

    resultDiv.classList.remove('hidden');
    setStatus('success', 'Decoded!');
}

function rescan() {
    resultDiv.classList.add('hidden');
    scanLine.style.animationPlayState = 'running';
    scanning = true;
    setStatus('scanning', 'Scanning...');
    startScanLoop();
}

// ---- File Upload ----
uploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file || !wasmReady) return;

    stopScanLoop();
    setStatus('loading', 'Decoding image...');

    try {
        const buf = await file.arrayBuffer();
        const result = decode_png_auto(new Uint8Array(buf));
        onDecodeSuccess(result);
    } catch (err) {
        setStatus('error', `Decode failed: ${err}`);
        setTimeout(() => {
            if (scanning) setStatus('scanning', 'Scanning...');
            else setStatus('ready', 'Ready');
        }, 3000);
    }

    // Reset file input so the same file can be selected again
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
        case 'loading':  break;
        case 'scanning': break;
        case 'ready':    break;
        case 'error':    statusDot.classList.add('error'); break;
        case 'success':  statusDot.classList.add('success'); break;
    }
    statusText.textContent = text;
}

// ---- Boot ----
(async () => {
    await initWasm();
    await startCamera();
})();
