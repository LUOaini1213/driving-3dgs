// No browser: exercise the real checker HTTP server/report lifecycle with a controlled browser boundary.
const fs = require('node:fs'), path = require('node:path');
const { root, checker, scenario, recordIn } = JSON.parse(fs.readFileSync(0, 'utf8'));
const { run } = require(checker);
const events = {}; let closed = false, evaluations = 0;
const page = {
  on(name, fn) { events[name] = fn; },
  async goto(url) {
    if (scenario === 'navigation') throw Error('navigation unavailable');
    for (const address of [url, new URL('sample.splat', url).href]) {
      const response = await fetch(address); let bytes = Buffer.from(await response.arrayBuffer());
      if (scenario === 'wrongasset' && address.endsWith('.splat')) bytes = Buffer.alloc(bytes.length);
      if (scenario === 'wrongpage' && address.endsWith('.html')) bytes = Buffer.alloc(bytes.length);
      events.response({ url: () => address, status: () => response.status, body: async () => bytes });
    }
    events.console({ type: () => 'warning', text: () => 'real warning retained' });
    if (scenario === 'http') events.response({ url: () => url + '/missing', status: () => 404 });
  },
  async waitForFunction() {}, async waitForTimeout() {},
  async evaluate() {
    evaluations++;
    if (evaluations === 1) return { ready: true };
    if (scenario === 'canvas') throw Error('canvas unavailable');
    if (scenario === 'changed') fs.appendFileSync(path.join(root, 'docs/splat/viewer.html'), 'changed');
    return { nonbackground_frac: .8, luma_std: 50 };
  },
  async screenshot({ path: file }) { if (scenario === 'screenshot') throw Error('screenshot unavailable'); fs.writeFileSync(file, 'fixture image'); }
};
const chromium = { async launch() { if (scenario === 'launch') throw Error('launch unavailable'); return { newPage: async () => page, close: async () => { closed = true; } }; } };
(async () => {
 const opts = { chromium, root, outJson: path.join(root, 'evidence/report.json'), shot: path.join(root, 'evidence/screenshot.jpg'), recordIn };
 const result = await run(opts);
 let overwriteRejected = false;
 try { await run(opts); } catch { overwriteRejected = true; }
 console.log(JSON.stringify({ result, closed, overwriteRejected }));
})().catch(error => { console.error(error); process.exitCode = 1; });
