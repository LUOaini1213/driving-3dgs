# Isolated public viewer

The public-viewer source is in `site/`. It reuses the exact existing 9,918,624-byte
`docs/splat/full_7k.splat`, SHA-256
`2dc35c0f05432772867395d2bad81101e76e30dead666671d37c5d1a4bdfb948`, and the recorded
preview. No training data, complete checkpoints, logs or new render-quality
measurements are included. The archived results and old viewer stay unchanged.

## Build and review locally

Python 3.11+ and Node.js 24 are enough. No Torch, CUDA, model download or GPU is
needed to build the static artifact. The npm install downloads two pinned MIT
renderer packages, not ML checkpoints, and disables lifecycle scripts.

```bash
npm ci --prefix site --ignore-scripts --no-audit --no-fund
node --check site/viewer.js
node --check site/viewer-frame.js
python scripts/build_viewer.py
python scripts/build_viewer.py --check
python -m http.server --bind 127.0.0.1 --directory build/pages 8000
```

Open `http://127.0.0.1:8000/`. Direct `file://` access is unsupported. Relative URLs
also work beneath a project-site path such as `/driving-3dgs/`.

The builder accepts only a fixed file list: page/frame code, one scene, one preview,
two pinned dependency modules and a source map, licence notices and `source-index.json`.
It refuses extra files instead of deleting or uploading them, checks the original
scene and official npm module hashes, and caps the whole artifact at 15 MB.
`source-index.json` records every published file hash, the current commit, any dirty
working-tree status, the original measurement commit and scene identity. A local
dirty preview is labelled as uncommitted. A publish build requires `--require-clean`.

## Interaction and compatibility boundaries

The page starts with a recorded screenshot. It fetches no renderer or scene until
the visitor clicks **Explore in 3D**. A separate, same-origin iframe owns the 3D
renderer and worker. **Return to preview**, failure or preparation timeout removes
the frame, preventing accumulated renderer instances after retry. Parent/child
messages are checked against both the current frame window and same origin.

Download and scene preparation are distinct states. **Reset view** is enabled only
after scene loading and initial frames, and restores camera 84. Unsupported WebGL 2
or WebAssembly, missing dependencies/assets, context loss and a 180-second visible
preparation timeout show an explanation and retain the preview. Hidden tabs pause
the ready renderer and do not consume the visible preparation timeout.

The renderer uses one device pixel per CSS pixel, non-shared workers, no GPU-assisted
sorting, and renders on change. This reduces avoidable resource use; it does not
establish any phone frame-rate or memory guarantee. Emulated small-screen layout
checks cannot replace a physical-device performance run. Browser visual/rendering
checks are separate from the lightweight packaging/syntax checks above.

The `full_7k` export contains 309,957 of 557,802 trained Gaussians and only zero-order,
view-independent colour. It is not the 30k model. Blur, gaps, near-camera floaters
and unmodelled moving objects remain. The website links to the archived offline
26.56 dB subset measurement; it makes no new browser PSNR or generalisation claim.

## Reviewable Pages publication

`.github/workflows/pages.yml` builds/checks the isolated artifact for relevant PRs.
Deployment has no push trigger. It runs only when the workflow is manually dispatched
on `master`, after the reviewed change is merged and Pages is explicitly configured
to use GitHub Actions. The `github-pages` environment may apply repository approval
rules. The workflow uploads `build/pages` only, never the repository or `docs/` root.
Official actions and npm dependencies are pinned; all runtime assets are served
from the same origin rather than a CDN.

Publication and its public URL require a separate actual deployment check. A local
build or successful workflow preparation is not evidence that a public site exists.
After deployment, verify the HTTPS page and asset URLs, actual nonblank rendering,
reset/return/reload, missing-scene fallback, touch-size layout and source-index hash
inventory, then add the confirmed URL to README and the portfolio.

## Attribution

Data © 2021 Argo AI, LLC. Scene and preview are Argoverse 2 derivatives under
CC BY-NC-SA 4.0, for non-commercial technical demonstration. The page and artifact
retain attribution and licence links. MIT applies to the viewer code and the two
renderer libraries, not the data. See the published `NOTICES.txt` and licence files.
