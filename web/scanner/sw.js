// Trixel Scanner — Service Worker
// Caches the app shell and WASM binary for offline use.

const CACHE_NAME = 'trixel-scanner-v2';
const ASSETS = [
    './',
    './index.html',
    './style.css',
    './scanner.js',
    './pkg/trixel_scanner.js',
    './pkg/trixel_scanner_bg.wasm',
    './manifest.json',
];

// Install: cache all app shell assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
    );
    self.skipWaiting();
});

// Activate: clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
            )
        )
    );
    self.clients.claim();
});

// Fetch: cache-first for known assets, network-first for everything else
self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((cached) => {
            return cached || fetch(event.request);
        })
    );
});
