const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3001;
const DATA_DIR = path.join(__dirname, '..', 'public', 'metadata');

if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

const nextId = () => {
  const files = fs
    .readdirSync(DATA_DIR)
    .filter((f) => f.endsWith('.json'))
    .map((f) => parseInt(f, 10));

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
