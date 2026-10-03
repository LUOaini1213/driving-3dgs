// Actual page module, with controlled module/WebGL boundaries. No browser or GPU.
const fs = require('node:fs'), vm = require('node:vm');
const { html, scenario, preview } = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
const status = { textContent: 'loading…' }, frames = [], listeners = {}, errors = [];
let renders = 0;
const canvas = { addEventListener(name, callback) { listeners[name] = callback; } };
const context = vm.createContext({ window: { addEventListener(name, callback) { listeners[name] = callback; } },
  document: { getElementById: () => status, body: { appendChild() {} } },
  console: { error: error => errors.push(String(error)) },
  requestAnimationFrame: callback => frames.push(callback),
  addEventListener(name, callback) { listeners[name] = callback; },
  atob: value => Buffer.from(value, 'base64').toString('binary'),
  innerWidth: 960, innerHeight: 600, devicePixelRatio: 1,
});
class Viewer {
  constructor() { if (scenario === 'webgl') throw new Error('WebGL unavailable'); this.renderer = { domElement: canvas }; }
  addSplatScene() { return scenario === 'asset' ? Promise.reject(new Error('asset HTTP 404')) : Promise.resolve(); }
  start() {}
  update() {}
  render() { if (scenario === 'render') throw new Error('render failed'); renders++; }
}
const vector = () => ({ set() {} });
const three = { Scene: class {}, Color: class {}, PerspectiveCamera: class { constructor() { this.up = vector(); } },
  WebGLRenderer: class { constructor() { throw new Error('WebGL unavailable'); } } };
async function dependency(specifier) {
  if (scenario === 'cdn') throw new Error('CDN unavailable');
  const values = specifier === 'three' ? three : specifier.includes('OrbitControls') ? { OrbitControls: class {} }
    : { Viewer, SceneRevealMode: { Instant: 1 }, SceneFormat: { Splat: 1 } };
  const module = new vm.SyntheticModule(Object.keys(values), function () {
    for (const [key, value] of Object.entries(values)) this.setExport(key, value);
  }, { context });
  await module.link(() => {}); await module.evaluate(); return module;
}
(async () => {
  let uncaught = null;
  const module = new vm.SourceTextModule(source, { context, importModuleDynamically: dependency });
  const evaluated = module.link(dependency).then(() => module.evaluate()).catch(e => { uncaught = e.message; });
  for (let step = 0; step < 80; step++) {
    await new Promise(resolve => setImmediate(resolve));
    if (frames.length) frames.shift()();
    if (scenario === 'contextloss' && renders === 1 && listeners.webglcontextlost) listeners.webglcontextlost({ preventDefault() {} });
  }
  await evaluated;
  console.log(JSON.stringify({ status: status.textContent, ready: preview ? context.window.__previewReady : context.window.__splatReady,
    renders, uncaught, errors }));
})().catch(error => { console.error(error); process.exitCode = 1; });
