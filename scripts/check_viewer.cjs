// Headless check of docs/splat/viewer.html: serve docs/ over HTTP, open the page in Chromium
// (software WebGL via SwiftShader), wait until the splat scene is loaded and rendered, screenshot it,
// measure that the canvas is not blank, and fail on any console error / page error / failed request.
//
//   node scripts/check_viewer.cjs [playwright-module-path] [out.json] [screenshot.jpg]
//
// The first argument is a module that exports `chromium` (default: require('playwright')).
const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const DOCS = path.join(ROOT, 'docs');
const modPath = process.argv[2] || 'playwright';
const outJson = process.argv[3] || path.join(ROOT, 'results', 'viewer_check.json');
const shot = process.argv[4] || path.join(DOCS, 'img', 'viewer_headless.jpg');
const { chromium } = require(modPath);

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.splat': 'application/octet-stream', '.jpg': 'image/jpeg' };
const server = http.createServer((req, res) => {
  const p = path.join(DOCS, decodeURIComponent(req.url.split('?')[0]));
  if (!p.startsWith(DOCS) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': TYPES[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
});

(async () => {
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 960, height: 600 } });
  const errors = [], warnings = [], failed = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); else if (m.type() === 'warning') warnings.push(m.text()); });
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('requestfailed', r => failed.push(r.url() + ' ' + (r.failure() || {}).errorText));
  const t0 = Date.now();
  let ready = false;
  try {
    await page.goto(`http://127.0.0.1:${port}/splat/viewer.html`, { waitUntil: 'load', timeout: 120000 });
    await page.waitForFunction(() => window.__splatReady === true, null, { timeout: 300000, polling: 500 });
    ready = true;
  } catch (e) { errors.push('timeout/nav: ' + e.message); }
  const loadS = (Date.now() - t0) / 1000;
  await page.waitForTimeout(3000);
  // read back the WebGL canvas right after a forced render and measure how much of it is non-background
  const stats = ready ? await page.evaluate(() => {
    const v = window.__viewer; v.update(); v.render();
    const src = v.renderer.domElement;
    const c = document.createElement('canvas'); c.width = 240; c.height = 150;
    const g = c.getContext('2d'); g.drawImage(src, 0, 0, c.width, c.height);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let nonbg = 0, sum = 0, sum2 = 0; const n = d.length / 4;
    for (let i = 0; i < d.length; i += 4) {
      const y = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      sum += y; sum2 += y * y; if (Math.abs(d[i] - d[0]) + Math.abs(d[i + 1] - d[1]) + Math.abs(d[i + 2] - d[2]) > 30) nonbg++;
    }
    const mean = sum / n;
    return { canvas: [src.width, src.height], nonbackground_frac: nonbg / n, luma_mean: mean, luma_std: Math.sqrt(sum2 / n - mean * mean),
             webgl: (() => { const gl = document.createElement('canvas').getContext('webgl2'); const e = gl && gl.getExtension('WEBGL_debug_renderer_info'); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : null; })() };
  }) : null;
  await page.screenshot({ path: shot, type: 'jpeg', quality: 80 });
  await browser.close();
  server.close();
  const ok = ready && errors.length === 0 && failed.length === 0 && stats && stats.nonbackground_frac > 0.3 && stats.luma_std > 10;
  const res = { date: new Date().toISOString().slice(0, 10), page: 'docs/splat/viewer.html', chromium_args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'],
                ready, load_and_first_frames_s: Math.round(loadS * 10) / 10, console_errors: errors, failed_requests: failed,
                console_warnings: warnings.length, canvas_stats: stats, screenshot: path.relative(ROOT, shot).replace(/\\/g, '/'),
                screenshot_bytes: fs.statSync(shot).size, ok };
  fs.writeFileSync(outJson, JSON.stringify(res, null, 1));
  console.log(JSON.stringify(res, null, 1));
  process.exit(ok ? 0 : 1);
})();
