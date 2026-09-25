const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3001;
const DATA_DIR = path.join(__dirname, '..', 'public', 'metadata');
const PROPERTY_API = '127.0.0.1';
const PROPERTY_API_PORT = 8001;

if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

const fetchJson = (port, host, requestPath) =>
  new Promise((resolve, reject) => {
    const req = http.get(
      { host, port, path: requestPath, headers: { accept: 'application/json' } },
      (res) => {
        let data = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => (data += chunk));
        res.on('end', () => {
          if (res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`upstream ${res.statusCode}`));
            return;
          }
          try {
            resolve(JSON.parse(data));
          } catch (error) {
            reject(error);
          }
        });
      },
    );
    req.on('error', reject);
  });

// Deterministic illustrative image via the same FNV-1a rule as the frontend
// (src/lib/propertyImage.js): (fnv1a(mreid) % 24) + 1.
const fnv1a = (input) => {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
};
const imageFor = (mreid) => (fnv1a(mreid) % 24) + 1;

const mreidToNumber = (mreid) => {
  const m = /MREID_0*(\d+)/i.exec(mreid || '');
  return m ? Number(m[1]) : null;
};

const buildMillowMetadata = (mreid, property) => {
  const facts = [
    { trait_type: 'City', value: property.city || '' },
    { trait_type: 'Location', value: property.location || '' },
    { trait_type: 'Bedrooms', value: property.bedrooms || 0 },
    { trait_type: 'Area (sqft)', value: property.area || 0 },
  ];
  return {
    name: `MILLOW Property ${mreid}`,
    property_identifier: mreid,
    description:
      'Digital property representation of a MILLOW MREID listing (technical proof-of-concept). Blockchain-based ownership record for the prototype. Legal ownership remains subject to applicable Indian property and registration law.',
    image: `/images/${imageFor(mreid)}.jpg`,
    attributes: facts,
  };
};

const nextId = () => {
  const files = fs
    .readdirSync(DATA_DIR)
    .filter((f) => f.endsWith('.json'))
    .map((f) => parseInt(f, 10))
    .filter((n) => Number.isInteger(n) && n > 0);

  return files.length ? Math.max(...files) + 1 : 1;
};

const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === 'POST' && req.url === '/api/list') {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      try {
        const data = JSON.parse(body);

        if (!data.name || !data.address || !data.image) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'name, address and image are required' }));
          return;
        }

        const id = nextId();
        const metadata = {
          name: data.name,
          address: data.address,
          description: data.description || '',
          image: data.image,
          id: String(id),
          attributes: data.attributes || [],
        };

        fs.writeFileSync(
          path.join(DATA_DIR, `${id}.json`),
          JSON.stringify(metadata, null, 4)
        );

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ id, uri: `http://localhost:${PORT}/metadata/${id}.json` }));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
    return;
  }

  const millowMatch = req.url.match(/^\/metadata\/millow\/([A-Za-z0-9_]+)\.json$/);
  if (millowMatch) {
    const mreid = millowMatch[1];
    const file = path.join(DATA_DIR, 'millow', `${mreid}.json`);
    if (fs.existsSync(file)) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(fs.readFileSync(file));
    } else {
      fetchJson(PROPERTY_API_PORT, PROPERTY_API, `/api/properties/${encodeURIComponent(mreid)}`)
        .then((detail) => {
          const metadata = buildMillowMetadata(mreid, detail.property || {});
          // Persist next to the seeded files so it is also served by the
          // create-react-app dev server (/public).
          try {
            fs.mkdirSync(path.dirname(file), { recursive: true });
            fs.writeFileSync(file, JSON.stringify(metadata, null, 2));
          } catch (_) {
            // Metadata generation is still served even if persisting fails.
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(metadata));
        })
        .catch(() => {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'not found', mreid }));
        });
    }
    return;
  }

  const match = req.url.match(/^\/metadata\/(\d+)\.json$/);
  if (match) {
    const file = path.join(DATA_DIR, `${match[1]}.json`);
    if (fs.existsSync(file)) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(fs.readFileSync(file));
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'not found' }));
    }
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found' }));
});

server.listen(PORT, () => {
  console.log(`Metadata server running at http://localhost:${PORT}`);
});
