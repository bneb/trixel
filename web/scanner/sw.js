// PrismCode Scanner — Service Worker
// Network-first for HTML/JS, cache-first for WASM binary.

const CACHE_NAME = 'prism-scanner-v9';
const ASSETS = [
    './',
    './index.html',
    './style.css',
    './scanner.js',
    './prism_pkg/prism_wasm.js',
    './prism_pkg/prism_wasm_bg.wasm',
    './manifest.json',
];

// Heavy assets that benefit from cache-first (WASM binary)
const CACHE_FIRST_PATHS = ['.wasm'];

// Install: cache all app shell assets
self.addEventListener('install', (event) => {
    self.skipWaiting();
});

// Activate: purge ALL old caches aggressively
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys.map((k) => caches.delete(k))
            )
        )
    );
    self.clients.claim();
});

// Fetch: network-first for HTML/JS, cache-first for WASM
self.addEventListener('fetch', (event) => {
    const url = event.request.url;

    // Cache-first for WASM binary
    if (CACHE_FIRST_PATHS.some((path) => url.includes(path))) {
        event.respondWith(
            caches.match(event.request).then((cached) => {
                if (cached) return cached;
                return fetch(event.request).then((response) => {
                    if (response.ok) {
                        const clone = response.clone();
                        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
                    }
                    return response;
                });
            })
        );
        return;
    }

    // Network-first for all other assets (HTML, JS, CSS)
    event.respondWith(
        fetch(event.request)
            .then((response) => {
                if (response.ok) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
                }
                return response;
            })
            .catch(() => caches.match(event.request))
    );
});
