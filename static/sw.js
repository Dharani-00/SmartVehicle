const CACHE_NAME = 'idr-nav-v1';
const TILE_CACHE = 'idr-tiles-v1';

const APP_SHELL = [
  '/',
  '/static/js/leaflet.js',
  '/static/css/leaflet.css',
  '/static/css/images/marker-icon.png',
  '/static/css/images/marker-icon-2x.png',
  '/static/css/images/marker-shadow.png',
  '/static/css/images/layers.png',
  '/static/css/images/layers-2x.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== CACHE_NAME && k !== TILE_CACHE)
          .map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

const PLACEHOLDER_TILE = (() => {
  const size = 256;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}">
    <rect width="${size}" height="${size}" fill="#1a1f36"/>
    <text x="128" y="128" text-anchor="middle" fill="#2a3050" font-size="12" font-family="sans-serif">offline</text>
  </svg>`;
  return new Blob([svg], { type: 'image/svg+xml' });
})();

function isTileRequest(url) {
  return url.includes('tile.openstreetmap.org');
}

function isApiRequest(url) {
  const u = new URL(url);
  return u.pathname.startsWith('/api/');
}

function isAppShell(url) {
  const u = new URL(url);
  return u.pathname === '/' || u.pathname.startsWith('/static/');
}

self.addEventListener('fetch', (e) => {
  const url = e.request.url;

  if (isTileRequest(url)) {
    e.respondWith(
      caches.open(TILE_CACHE).then((cache) =>
        cache.match(e.request).then((cached) => {
          const networkFetch = fetch(e.request)
            .then((resp) => {
              if (resp.ok) {
                cache.put(e.request, resp.clone());
              }
              return resp;
            })
            .catch(() => {
              if (cached) return cached;
              return new Response(PLACEHOLDER_TILE, {
                headers: { 'Content-Type': 'image/svg+xml' },
              });
            });

          return cached || networkFetch;
        })
      )
    );
    return;
  }

  if (isApiRequest(url)) {
    e.respondWith(fetch(e.request));
    return;
  }

  if (isAppShell(url)) {
    e.respondWith(
      caches.match(e.request).then((cached) => cached || fetch(e.request))
    );
    return;
  }

  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});

self.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'CACHE_TILES') {
    const { tiles } = e.data;
    if (tiles && tiles.length > 0) {
      caches.open(TILE_CACHE).then((cache) => {
        tiles.forEach((tileUrl) => {
          cache.match(tileUrl).then((existing) => {
            if (!existing) {
              fetch(tileUrl)
                .then((resp) => {
                  if (resp.ok) cache.put(tileUrl, resp);
                })
                .catch(() => {});
            }
          });
        });
      });
    }
  }
});
