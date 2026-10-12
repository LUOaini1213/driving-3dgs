// Software-WebGL browser evidence. Old results/viewer_check.json is historical.
// node scripts/check_viewer.cjs [playwright-module-path] [new-out.json] [new-shot.jpg] [--record-in web_export.json]
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { isDeepStrictEqual } = require('node:util');
const ROOT = path.resolve(__dirname, '..');
const ARGS = ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'];
function portable(root, file) {
  const relative = path.relative(root, path.resolve(file));
  return !relative.startsWith('..') && !path.isAbsolute(relative) ? relative.replace(/\\/g, '/') : path.resolve(file);
}
function parseArgs(args) {
  const positional = []; let recordIn;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--record-in') {
      if (recordIn || !args[i + 1] || args[i + 1].startsWith('--')) throw Error('--record-in requires one export JSON path');
      recordIn = args[++i];
    } else if (args[i].startsWith('--')) throw Error('unknown option: ' + args[i]);
    else positional.push(args[i]);
  }
  if (positional.length > 3) throw Error('expected at most module, report, screenshot positional arguments');
  return { modPath: positional[0] || 'playwright', outJson: positional[1], shot: positional[2], recordIn };
}
function linkReport(root, exportFile, reportFile, result) {
  if (!result.ok) throw Error('cannot link a failed browser report');
  const original = fs.readFileSync(exportFile, 'utf8'), value = JSON.parse(original);
  const manifest = result.binding.manifest;
  if (!value.asset_binding || !isDeepStrictEqual(value.asset_binding, manifest) ||
      path.resolve(root, value.file) !== path.join(root, 'docs/splat', manifest.asset)) throw Error('export has no matching current asset binding');
  if (JSON.stringify(snapshot(root)) !== JSON.stringify(result.binding)) throw Error('content changed before linking browser report');
  value.viewer_check = { file: portable(root, reportFile), ...record(reportFile, true) };
  const temp = exportFile + '.tmp-' + crypto.randomBytes(6).toString('hex');
  try {
    fs.writeFileSync(temp, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
    if (fs.readFileSync(exportFile, 'utf8') !== original) throw Error('export JSON changed before linking browser report');
    fs.renameSync(temp, exportFile);
  } finally { if (fs.existsSync(temp)) fs.unlinkSync(temp); }
}
function record(file, text = false) {
  let bytes = fs.readFileSync(file);
  if (text) bytes = Buffer.from(bytes.toString('utf8').replace(/\r\n/g, '\n'));
  return { sha256: crypto.createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length, mode: text ? 'utf8-lf' : 'raw' };
}
function snapshot(root) {
  const dir = path.join(root, 'docs/splat'), file = path.join(dir, 'asset_manifest.json');
  const manifest = JSON.parse(fs.readFileSync(file, 'utf8'));
  const names = [manifest.asset, manifest.page];
  if (manifest.schema !== 1 || new Set(names).size !== 2 || !manifest.files ||
      Object.keys(manifest.files).sort().join('|') !== [...names].sort().join('|')) throw Error('invalid asset manifest');
  for (const name of names) {
    if (typeof name !== 'string' || !name || name === '.' || name === '..' || /[/\\:]/.test(name)) throw Error('invalid manifest filename');
    const actual = record(path.join(dir, name), name === manifest.page), expected = manifest.files[name];
    if (actual.sha256 !== expected.sha256 || actual.bytes !== expected.bytes || actual.mode !== expected.mode) throw Error('asset/page hash changed: ' + name);
  }
  return { manifest, manifest_file: record(file, true), checker: record(__filename, true) };
}
function serverFor(docs) {
  const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.splat': 'application/octet-stream', '.jpg': 'image/jpeg' };
  return http.createServer((req, res) => {
    let file;
    try {
      const requested = decodeURIComponent(req.url.split('?')[0]);
      file = path.resolve(docs, '.' + requested);
      const relative = path.relative(docs, file);
      if (relative.startsWith('..') || path.isAbsolute(relative) || requested.includes('\\')) throw Error('outside docs');
    } catch { res.writeHead(400); res.end(); return; }
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
    const stream = fs.createReadStream(file);
    stream.on('error', () => res.destroy()); stream.pipe(res);
  });
}
async function run({ chromium, root = ROOT, outJson, shot, recordIn }) {
  const binding = snapshot(root);
  // Explicit output paths must also be fresh, so historical evidence is never replaced.
  if (fs.existsSync(outJson) || fs.existsSync(shot)) throw Error('evidence output already exists; choose fresh paths');
  fs.mkdirSync(path.dirname(outJson), { recursive: true });
  fs.mkdirSync(path.dirname(shot), { recursive: true });
  const server = serverFor(path.join(root, 'docs'));
  let browser, ready = false, stats = null, loadS = null, screenshot = null;
  const errors = [], warnings = [], failed = [], responses = [], pages = [], pending = [];
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    const base = `http://127.0.0.1:${server.address().port}/splat/`;
    const assetUrl = base + encodeURIComponent(binding.manifest.asset);
    const pageUrl = base + encodeURIComponent(binding.manifest.page);
    browser = await chromium.launch({ args: ARGS });
    const page = await browser.newPage({ viewport: { width: 960, height: 600 } });
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); else if (m.type() === 'warning') warnings.push(m.text()); });
    page.on('pageerror', e => errors.push('pageerror: ' + e.message));
    page.on('requestfailed', r => failed.push(r.url() + ' ' + (r.failure() || {}).errorText));
    page.on('response', r => {
      if (r.status() >= 400) failed.push('HTTP ' + r.status() + ' ' + r.url());
      if (r.url() === assetUrl || r.url() === pageUrl) pending.push((async () => {
        let bytes = await r.body();
        const target = r.url() === assetUrl ? responses : pages;
        if (target === pages) bytes = Buffer.from(bytes.toString('utf8').replace(/\r\n/g, '\n'));
        target.push({ status: r.status(), bytes: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') });
      })().catch(e => errors.push('asset response: ' + e.message)));
    });
    const t0 = Date.now();
    await page.goto(pageUrl, { waitUntil: 'load', timeout: 120000 });
    await page.waitForFunction(() => window.__splatReady === true || Boolean(window.__splatError), null, { timeout: 300000, polling: 500 });
    const state = await page.evaluate(() => ({ ready: window.__splatReady, error: window.__splatError }));
    if (state.error || !state.ready) throw Error('viewer: ' + (state.error || 'not ready'));
    ready = true; loadS = (Date.now() - t0) / 1000;
    await page.waitForTimeout(3000);
    stats = await page.evaluate(() => {
      if (window.__splatError || !window.__splatReady) throw Error(window.__splatError || 'viewer lost readiness');
      const v = window.__viewer; v.update(); v.render();
      const src = v.renderer.domElement;
      const c = document.createElement('canvas'); c.width = 240; c.height = 150;
      const g = c.getContext('2d'); g.drawImage(src, 0, 0, c.width, c.height);
      const d = g.getImageData(0, 0, c.width, c.height).data;
      let nonbg = 0, sum = 0, sum2 = 0; const n = d.length / 4;
      for (let i = 0; i < d.length; i += 4) {
        const y = .299 * d[i] + .587 * d[i + 1] + .114 * d[i + 2]; sum += y; sum2 += y * y;
        if (Math.abs(d[i] - d[0]) + Math.abs(d[i + 1] - d[1]) + Math.abs(d[i + 2] - d[2]) > 30) nonbg++;
      }
      const mean = sum / n;
      return { canvas: [src.width, src.height], nonbackground_frac: nonbg / n, luma_mean: mean, luma_std: Math.sqrt(Math.max(0, sum2 / n - mean * mean)),
        webgl: (() => { const gl = document.createElement('canvas').getContext('webgl2'); const e = gl && gl.getExtension('WEBGL_debug_renderer_info'); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : null; })() };
    });
    await page.screenshot({ path: shot, type: 'jpeg', quality: 80 });
    screenshot = { file: portable(root, shot), ...record(shot) };
  } catch (e) { errors.push('browser check: ' + e.message); }
  finally {
    if (browser) { try { await browser.close(); } catch (e) { errors.push('browser close: ' + e.message); } }
    if (server.listening) await new Promise(resolve => server.close(resolve));
  }
  await Promise.all(pending);
  try { if (JSON.stringify(snapshot(root)) !== JSON.stringify(binding)) throw Error('content changed during browser check'); }
  catch (e) { errors.push('binding: ' + e.message); }
  const expected = binding.manifest.files[binding.manifest.asset];
  if (!responses.length || responses.some(r => r.status !== 200 || r.bytes !== expected.bytes || r.sha256 !== expected.sha256)) errors.push('served asset bytes do not match manifest');
  const expectedPage = binding.manifest.files[binding.manifest.page];
  if (!pages.length || pages.some(r => r.status !== 200 || r.bytes !== expectedPage.bytes || r.sha256 !== expectedPage.sha256)) errors.push('served page bytes do not match manifest');
  const ok = Boolean(ready && !errors.length && !failed.length && screenshot && stats && stats.nonbackground_frac > .3 && stats.luma_std > 10);
  const result = { schema: 1, date: new Date().toISOString(), page: 'docs/splat/' + binding.manifest.page, chromium_args: ARGS,
    binding, served_asset: responses, served_page: pages, ready, load_and_first_frames_s: loadS, console_errors: errors, failed_requests: failed,
    console_warnings: warnings, canvas_stats: stats, screenshot, ok };
  fs.writeFileSync(outJson, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  if (recordIn && result.ok) linkReport(root, recordIn, outJson, result);
  return result;
}
async function main() {
  const args = parseArgs(process.argv.slice(2));
  const dir = path.join(ROOT, 'results/viewer-checks', new Date().toISOString().replace(/[:.]/g, '-') + '-' + crypto.randomBytes(3).toString('hex'));
  const outJson = args.outJson || path.join(dir, 'report.json');
  const shot = args.shot || path.join(path.dirname(outJson), 'screenshot.jpg');
  const result = await run({ chromium: require(args.modPath).chromium, outJson, shot, recordIn: args.recordIn });
  console.log(JSON.stringify(result, null, 2)); process.exitCode = result.ok ? 0 : 1;
}
module.exports = { record, snapshot, serverFor, run, parseArgs, linkReport };
if (require.main === module) main().catch(e => { console.error(e); process.exitCode = 1; });
