"""CPU-only web asset decoding and content binding; no training/browser claims."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .splat import C0, decode_splat


def file_record(path: Path, *, text: bool = False) -> dict:
    data = Path(path).read_bytes()
    if text:
        data = data.decode('utf-8').replace('\r\n', '\n').encode('utf-8')
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
            'mode': 'utf8-lf' if text else 'raw'}


def make_manifest(asset: Path, page: Path) -> dict:
    asset, page = Path(asset), Path(page)
    if asset.parent.resolve() != page.parent.resolve() or asset.name == page.name:
        raise ValueError('asset and viewer must be distinct files in the same directory')
    return {'schema': 1, 'purpose': 'content-integrity-only; not a training or browser verification',
            'asset': asset.name, 'page': page.name,
            'files': {asset.name: file_record(asset), page.name: file_record(page, text=True)}}


def verify_manifest(directory: Path, manifest: dict | None = None) -> dict:
    directory = Path(directory)
    if manifest is None:
        manifest = json.loads((directory / 'asset_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 1:
        raise ValueError('unsupported web asset manifest schema')
    names = [manifest['asset'], manifest['page']]
    if len(set(names)) != 2 or set(manifest['files']) != set(names):
        raise ValueError('manifest must bind exactly one asset and one viewer')
    for name in names:
        if not isinstance(name, str) or any(c in name for c in '/\\:') or name in ('', '.', '..'):
            raise ValueError('manifest filenames must stay in the asset directory')
        expected = manifest['files'][name]
        mode = 'raw' if name == manifest['asset'] else 'utf8-lf'
        if expected.get('mode') != mode or file_record(directory / name, text=mode == 'utf8-lf') != expected:
            raise ValueError(f'web asset/page hash changed: {name}')
    return manifest


def decoded_parameters(data: bytes) -> dict[str, np.ndarray]:
    """Parameters decoded from .splat bytes for a gsplat SH0 evaluation.

    This is not a browser-renderer equivalence claim. Alpha endpoints are exact:
    sigmoid(-inf)=0 and sigmoid(inf)=1; do not clip them to arbitrary epsilons.
    """
    if not data:
        raise ValueError('cannot evaluate an empty web asset')
    decoded = decode_splat(data)
    alpha = decoded['alpha']
    with np.errstate(divide='ignore'):
        logits = np.log(alpha) - np.log1p(-alpha)
    return {key: np.asarray(value, dtype=np.float32) for key, value in {
        'means': decoded['means'], 'scales': np.log(decoded['scales']), 'quats': decoded['quats'],
        'opacities': logits, 'sh0': ((decoded['rgb'] - .5) / C0)[:, None, :],
        'shN': np.zeros((len(alpha), 0, 3)),
    }.items()}


def verify_browser_report(root: Path, export: dict) -> dict:
    """Validate a linked, portable browser record before calling it current evidence."""
    root = Path(root).resolve()
    def inside(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('browser evidence must be within the repository')
        return path
    def check_record(value):
        path = inside(value['file'])
        expected = {key: value[key] for key in ('sha256', 'bytes', 'mode')}
        if expected['mode'] not in ('raw', 'utf8-lf') or file_record(path, text=expected['mode'] == 'utf8-lf') != expected:
            raise ValueError('browser evidence hash changed: ' + value['file'])
        return path
    if export.get('file_portable') is False:
        raise ValueError('external web assets cannot be used by the repository README')
    asset = inside(export['file'])
    manifest = verify_manifest(asset.parent)
    if export.get('asset_binding') != manifest or manifest['asset'] != asset.name:
        raise ValueError('export/browser asset binding mismatch')
    ref = export['viewer_check']
    if ref['mode'] != 'utf8-lf':
        raise ValueError('browser report must use UTF-8/LF hashing')
    report = json.loads(check_record(ref).read_text(encoding='utf-8'))
    binding = report.get('binding', {})
    if (report.get('schema') != 1 or report.get('ok') is not True or report.get('ready') is not True
            or report.get('console_errors') != [] or report.get('failed_requests') != []
            or binding.get('manifest') != manifest
            or binding.get('manifest_file') != file_record(asset.parent / 'asset_manifest.json', text=True)
            or binding.get('checker') != file_record(root / 'scripts/check_viewer.cjs', text=True)):
        raise ValueError('browser report is failed, unbound, or stale')
    if report.get('page') != (asset.parent / manifest['page']).relative_to(root).as_posix():
        raise ValueError('browser page and manifest disagree')
    for key, filename in [('served_asset', manifest['asset']), ('served_page', manifest['page'])]:
        expected = manifest['files'][filename]
        responses = report.get(key)
        if not responses or any(r != dict(status=200, bytes=expected['bytes'], sha256=expected['sha256']) for r in responses):
            raise ValueError('browser served bytes and manifest disagree')
    stats = report.get('canvas_stats') or {}
    if not (stats.get('nonbackground_frac', 0) > .3 and stats.get('luma_std', 0) > 10):
        raise ValueError('browser canvas was not verified')
    check_record(report['screenshot'])
    return report
