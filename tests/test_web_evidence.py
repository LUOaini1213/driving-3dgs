"""CPU-only export/report regression tests, with generated pages executed by Node."""
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def viewer_template(preview=False):
    export = script('export')
    if preview:
        data = {'pos': 'AAAAAA==', 'col': 'AAAA', 'lo': [0, 0, 0], 'hi': [1, 1, 1], 'train': [[0, 0, 0]], 'test': []}
        return export.HTML.replace('__DATA__', json.dumps(data))
    return export.VIEWER.replace('__POS__', '[0,0,0]').replace('__LOOK__', '[0,0,1]').replace('__FILE_JSON__', '"sample.splat"')


@pytest.mark.parametrize('preview,scenario', [(False, 'cdn'), (False, 'webgl'), (False, 'asset'),
                                               (False, 'render'), (False, 'contextloss'), (True, 'cdn'), (True, 'webgl')])
def test_viewer_failures_are_visible_and_never_ready(preview, scenario):
    run = subprocess.run(['node', '--experimental-vm-modules', str(ROOT / 'tests/viewer_runtime_fixture.cjs')],
                         input=json.dumps({'html': viewer_template(preview), 'scenario': scenario, 'preview': preview}),
                         text=True, encoding='utf-8', capture_output=True, check=True)
    result = json.loads(run.stdout)
    assert not result.get('ready')
    assert 'loading' not in result['status'].lower()
    assert any(word in result['status'].lower() for word in ['failed', 'unavailable', 'error'])
    assert result['uncaught'] is None


def test_splat_ready_requires_successful_renders():
    run = subprocess.run(['node', '--experimental-vm-modules', str(ROOT / 'tests/viewer_runtime_fixture.cjs')],
                         input=json.dumps({'html': viewer_template(), 'scenario': 'success'}),
                         text=True, encoding='utf-8', capture_output=True, check=True)
    result = json.loads(run.stdout)
    assert result['ready'] is True and result['renders'] >= 10


def test_historical_web_metrics_and_browser_record_are_not_relabelled_current():
    text = script('check_readme').render_web()
    assert '量化前' in text and '历史' in text
    assert '网页实际加载的子集' not in text
    assert '不证明当前' in text


def test_decoded_asset_parameters_preserve_quantized_values():
    from d3gs.splat import C0, decode_splat, encode_splat
    from d3gs.web_asset import decoded_parameters
    means = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    data = encode_splat(means, np.log([[1, 2, 3], [2, 3, 4]]), [[1, .3, .4, .2], [0, 0, 0, 1]],
                        np.array([-100., 100.]), np.array([[[4., -.2, -4.]], [[.4, .3, .2]]]))
    decoded, params = decode_splat(data), decoded_parameters(data)
    # float32 SH reconstruction can differ by one ULP (5.96e-8 at zero RGB).
    np.testing.assert_allclose(params['sh0'][:, 0] * C0 + .5, decoded['rgb'], atol=1e-7)
    np.testing.assert_allclose(torch.sigmoid(torch.from_numpy(params['opacities'])).numpy(), decoded['alpha'])
    np.testing.assert_allclose(np.exp(params['scales']), decoded['scales'], rtol=1e-6)
    np.testing.assert_allclose(params['quats'], decoded['quats'], atol=1e-7)
    assert params['shN'].shape == (2, 0, 3)
    with pytest.raises(ValueError, match='empty'):
        decoded_parameters(b'')


def test_manifest_rejects_same_size_changes_and_missing_files(tmp_path):
    from d3gs.web_asset import make_manifest, verify_manifest
    asset, viewer = tmp_path / 'sample.splat', tmp_path / 'viewer.html'
    asset.write_bytes(b'a' * 32); viewer.write_text('<html>current</html>\n', encoding='utf-8')
    manifest = make_manifest(asset, viewer)
    verify_manifest(tmp_path, manifest)
    viewer.write_bytes(b'<html>current</html>\r\n')
    verify_manifest(tmp_path, manifest)  # Git checkout line endings do not change the page identity.
    asset.write_bytes(b'b' * 32)
    with pytest.raises(ValueError, match='hash|SHA|changed'):
        verify_manifest(tmp_path, manifest)
    asset.write_bytes(b'a' * 32); viewer.write_text('<html>changed</html>\n', encoding='utf-8')
    with pytest.raises(ValueError, match='hash|SHA|changed'):
        verify_manifest(tmp_path, manifest)
    viewer.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_manifest(tmp_path, manifest)


@pytest.mark.parametrize('sh_degree', [0, 1])
def test_export_evaluates_decoded_bytes_not_the_original_parameters(tmp_path, monkeypatch, sh_degree):
    export = script('export')
    calls = []
    def render(params, *_args):
        calls.append({k: v.detach().cpu().numpy().copy() for k, v in params.items()})
        return torch.full((1, 1, 3), .5), None, None
    monkeypatch.setattr(export, 'render', render)
    checkpoint = dict(means=np.zeros((1, 3), np.float32), scales=np.zeros((1, 3), np.float32),
                      quats=np.array([[1, .3, .4, .2]], np.float32), opacities=np.array([.003], np.float32),
                      sh0=np.array([[[3., -.2, -3.]]], np.float32), shN=np.zeros((1, 3, 3), np.float32))
    scene = SimpleNamespace(test=[0], c2w=torch.eye(4)[None], K=torch.eye(3), W=1, H=1,
                            image=lambda i: np.zeros((1, 1, 3), np.uint8), viewmat=lambda i: torch.eye(4))
    args = SimpleNamespace(splat_max=1, splat_min_opacity=0)
    result = export.export_web(args, checkpoint, scene, tmp_path, 'tiny', sh_degree, device='cpu')
    assert len(calls) == (2 if sh_degree == 0 else 3)
    from d3gs.splat import decode_splat, C0
    decoded = decode_splat((tmp_path / 'tiny.splat').read_bytes())
    np.testing.assert_allclose(calls[-1]['sh0'][:, 0] * C0 + .5, decoded['rgb'], atol=1e-7)
    np.testing.assert_allclose(torch.sigmoid(torch.from_numpy(calls[-1]['opacities'])).numpy(), decoded['alpha'], atol=1e-7)
    assert not np.array_equal(calls[-1]['quats'], checkpoint['quats'])
    assert 'decoded_asset_sh0' in result['heldout_psnr_mean']
    assert 'web_subset_sh0' not in result['heldout_psnr_mean']
    assert 'gsplat' in result['metric_semantics'] and 'browser' in result['metric_semantics']


def test_new_web_metric_label_is_decoded_gsplat_not_browser(monkeypatch):
    check = script('check_readme')
    value = json.loads((ROOT / 'results/web_export.json').read_text(encoding='utf-8'))
    value['heldout_psnr_mean'] = {'full_sh0': 21.25, 'decoded_asset_sh0': 20.5}
    monkeypatch.setattr(check, 'load_json', lambda name: value if name == 'results/web_export.json' else None)
    text = check.render_web()
    assert '实际导出字节解码后的 SH0 参数 20.50 dB' in text
    assert '不是浏览器渲染器的 PSNR' in text
    assert '量化前' not in text

@pytest.mark.parametrize('scenario', ['success', 'launch', 'navigation', 'canvas', 'screenshot', 'wrongasset', 'wrongpage', 'changed', 'http'])
def test_checker_binds_served_bytes_and_cleans_up_failures(tmp_path, scenario):
    from d3gs.web_asset import make_manifest
    directory = tmp_path / 'docs/splat'; directory.mkdir(parents=True)
    asset, viewer = directory / 'sample.splat', directory / 'viewer.html'
    asset.write_bytes(b'a' * 32); viewer.write_text('<html>fixture</html>\n', encoding='utf-8')
    (directory / 'asset_manifest.json').write_text(json.dumps(make_manifest(asset, viewer)), encoding='utf-8')
    run = subprocess.run(['node', str(ROOT / 'tests/check_viewer_fixture.cjs')],
                         input=json.dumps(dict(root=str(tmp_path), checker=str(ROOT / 'scripts/check_viewer.cjs'), scenario=scenario)),
                         text=True, encoding='utf-8', capture_output=True, check=True, timeout=20)
    value = json.loads(run.stdout)
    result = value['result']
    assert result['ok'] is (scenario == 'success')
    assert value['closed'] is (scenario != 'launch')
    assert value['overwriteRejected']
    if scenario == 'success':
        assert result['console_warnings'] == ['real warning retained']
        assert result['served_asset'][0]['sha256'] == result['binding']['manifest']['files']['sample.splat']['sha256']
        assert result['served_page'][0]['sha256'] == result['binding']['manifest']['files']['viewer.html']['sha256']
        assert result['screenshot']['sha256']
    else:
        assert result['console_errors'] or result['failed_requests']


def test_export_cli_verifies_run_before_any_write(tmp_path, monkeypatch):
    import sys
    export = script('export')
    run, out = tmp_path / 'run', tmp_path / 'out'; run.mkdir()
    (run / 'train_stats.json').write_text(json.dumps({'work': 'fixture', 'sh_degree': 0}))
    torch.save({}, run / 'ckpt.pt')
    work = tmp_path / 'work'; work.mkdir(); (work / 'init_points.npz').write_bytes(b'fixture')
    scene = SimpleNamespace(dir=work, meta={'frames': []}, train=[], test=[], lidar_split=None)
    monkeypatch.setattr(export, 'Scene', lambda work: scene)
    monkeypatch.setattr(sys, 'argv', ['export', '--run', str(run), '--out', str(out), '--skip-ply', '--skip-preview'])
    with pytest.raises(ValueError, match='legacy run'):
        export.main()
    assert not out.exists()
    captured = []
    def web(args, *unused):
        captured.append(args.run_identity)
        return {'run_identity': args.run_identity}
    monkeypatch.setattr(export, 'export_web', web)
    result = tmp_path / 'web.json'
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--allow-unverified-run', '--splat-dir', str(tmp_path / 'splat'), '--web-results', str(result)])
    export.main()
    assert captured[0]['status'] == 'unverified_legacy'
    assert json.loads(result.read_text())['run_identity'] == captured[0]


@pytest.mark.parametrize('preview,scenario', [(False, 'cdn'), (False, 'webgl'), (False, 'success'), (True, 'cdn'), (True, 'webgl')])
def test_committed_pages_execute_the_same_error_boundary(preview, scenario):
    html = (ROOT / ('docs/preview.html' if preview else 'docs/splat/viewer.html')).read_text(encoding='utf-8')
    run = subprocess.run(['node', '--experimental-vm-modules', str(ROOT / 'tests/viewer_runtime_fixture.cjs')],
                         input=json.dumps(dict(html=html, scenario=scenario, preview=preview)),
                         text=True, encoding='utf-8', capture_output=True, check=True)
    result = json.loads(run.stdout)
    assert bool(result.get('ready')) is (scenario == 'success')
    assert result['uncaught'] is None
    assert ('failed' in result['status'].lower()) is (scenario != 'success')


@pytest.mark.parametrize('mode', ['--check', '--write'])
def test_readme_rejects_same_size_asset_tampering_before_writing(tmp_path, monkeypatch, mode):
    import shutil, sys
    check = script('check_readme')
    shutil.copytree(ROOT / 'results', tmp_path / 'results')
    shutil.copytree(ROOT / 'docs', tmp_path / 'docs')
    shutil.copy2(ROOT / 'README.md', tmp_path / 'README.md')
    before = (tmp_path / 'README.md').read_bytes()
    asset = tmp_path / 'docs/splat/full_7k.splat'
    asset.write_bytes(b'\0' * asset.stat().st_size)
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['check_readme', mode])
    with pytest.raises(SystemExit, match='web content binding failed'):
        check.main()
    assert (tmp_path / 'README.md').read_bytes() == before

@pytest.mark.parametrize('offsets', [[-.25, .25, .3], [-2., 0., 2.]])
def test_offpath_report_uses_exact_offset_keys(monkeypatch, offsets):
    check, offpath = script('check_readme'), script('offpath')
    values = {offpath.offset_key(x): dict(depth_median_abs_m=abs(x), depth_within_0_5m=.5, hole_frac_mean=.2) for x in offsets}
    monkeypatch.setattr(check, 'load_json', lambda _: dict(offsets_m=offsets, runs={'fixture': {'per_offset': values}}))
    text = check.render_offpath()
    for x in offsets:
        assert f'{offpath.offset_key(x)} m' in text


def test_export_asset_path_is_portable_inside_repository(tmp_path, monkeypatch):
    export = script('export')
    monkeypatch.setattr(export, 'ROOT', tmp_path, raising=False)
    assert export.asset_location(tmp_path / 'docs/splat/tiny.splat') == ('docs/splat/tiny.splat', True)
    external = tmp_path.parent / 'outside/tiny.splat'
    assert export.asset_location(external) == (str(external.resolve()), False)

@pytest.fixture
def linked_web_report(tmp_path):
    import shutil
    from d3gs.web_asset import make_manifest
    directory = tmp_path / 'docs/splat'; directory.mkdir(parents=True)
    asset, viewer = directory / 'sample.splat', directory / 'viewer.html'
    asset.write_bytes(b'a' * 32); viewer.write_text('<html>fixture</html>\n', encoding='utf-8')
    binding = make_manifest(asset, viewer)
    (directory / 'asset_manifest.json').write_text(json.dumps(binding), encoding='utf-8')
    (tmp_path / 'scripts').mkdir()
    shutil.copy2(ROOT / 'scripts/check_viewer.cjs', tmp_path / 'scripts/check_viewer.cjs')
    export = json.loads((ROOT / 'results/web_export.json').read_text(encoding='utf-8'))
    export.update(file='docs/splat/sample.splat', bytes=32, asset_binding=binding)
    web_file = tmp_path / 'web.json'; web_file.write_text(json.dumps(export), encoding='utf-8')
    result = subprocess.run(['node', str(ROOT / 'tests/check_viewer_fixture.cjs')],
                            input=json.dumps(dict(root=str(tmp_path), checker=str(ROOT / 'scripts/check_viewer.cjs'),
                                                  scenario='success', recordIn=str(web_file))),
                            text=True, encoding='utf-8', capture_output=True, check=True, timeout=20)
    assert json.loads(result.stdout)['result']['ok']
    return tmp_path, json.loads(web_file.read_text(encoding='utf-8'))


def test_linked_current_browser_report_is_portable_and_keeps_history(linked_web_report, tmp_path, monkeypatch):
    import shutil
    from d3gs.web_asset import verify_browser_report
    root, web = linked_web_report
    assert web['viewer_check']['file'] == 'evidence/report.json'
    report = verify_browser_report(root, web)
    assert report['screenshot']['file'] == 'evidence/screenshot.jpg'
    relocated = tmp_path / 'relocated'
    shutil.copytree(root / 'docs', relocated / 'docs')
    shutil.copytree(root / 'scripts', relocated / 'scripts')
    shutil.copytree(root / 'evidence', relocated / 'evidence')
    assert verify_browser_report(relocated, web) == report
    check = script('check_readme')
    old = check.load_json('results/viewer_check.json')
    monkeypatch.setattr(check, 'ROOT', relocated)
    monkeypatch.setattr(check, 'load_json', lambda name: web if name == 'results/web_export.json' else old if name == 'results/viewer_check.json' else None)
    text = check.render_web()
    assert f"本次绑定的无头 Chromium 检查（{report['date']}）" in text
    assert f"用时 {report['load_and_first_frames_s']} s" in text
    assert f"历史无头 Chromium（{old['date']}" in text
    assert '不证明当前页面通过检查' in text


@pytest.mark.parametrize('change', ['report', 'screenshot', 'asset', 'served', 'failed', 'export_binding'])
def test_linked_report_rejects_changed_or_unbound_evidence(linked_web_report, change):
    from d3gs.web_asset import file_record, verify_browser_report
    root, web = linked_web_report
    if change == 'screenshot':
        (root / 'evidence/screenshot.jpg').write_bytes(b'changed')
    elif change == 'asset':
        (root / 'docs/splat/sample.splat').write_bytes(b'b' * 32)
    elif change == 'export_binding':
        web.pop('asset_binding')
    else:
        file = root / 'evidence/report.json'
        report = json.loads(file.read_text(encoding='utf-8'))
        if change == 'served': report['served_asset'][0]['sha256'] = '0' * 64
        elif change == 'failed': report['ok'] = False
        else: report['load_and_first_frames_s'] = 12345
        file.write_text(json.dumps(report), encoding='utf-8')
        if change != 'report': web['viewer_check'].update(file_record(file, text=True))
    with pytest.raises(ValueError):
        verify_browser_report(root, web)


def test_checker_cli_arguments_accept_explicit_evidence_paths_and_link():
    code = "const {parseArgs}=require(process.argv[1]); console.log(JSON.stringify(parseArgs(process.argv.slice(2))))"
    args = ['node', '-e', code, str(ROOT / 'scripts/check_viewer.cjs'), 'playwright', 'results/current.json',
            'docs/img/current.jpg', '--record-in', 'results/web_export.json']
    run = subprocess.run(args, capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == dict(modPath='playwright', outJson='results/current.json', shot='docs/img/current.jpg', recordIn='results/web_export.json')
    bad = subprocess.run(args[:-1], capture_output=True, text=True)
    assert bad.returncode != 0 and '--record-in requires' in bad.stderr


def test_failed_check_does_not_replace_export_reference(tmp_path):
    from d3gs.web_asset import make_manifest
    directory = tmp_path / 'docs/splat'; directory.mkdir(parents=True)
    asset, viewer = directory / 'sample.splat', directory / 'viewer.html'
    asset.write_bytes(b'a' * 32); viewer.write_text('<html>fixture</html>\n', encoding='utf-8')
    (directory / 'asset_manifest.json').write_text(json.dumps(make_manifest(asset, viewer)), encoding='utf-8')
    web = tmp_path / 'web.json'; web.write_text('{"old":"preserved"}', encoding='utf-8'); before = web.read_bytes()
    result = subprocess.run(['node', str(ROOT / 'tests/check_viewer_fixture.cjs')],
                            input=json.dumps(dict(root=str(tmp_path), checker=str(ROOT / 'scripts/check_viewer.cjs'),
                                                  scenario='navigation', recordIn=str(web))),
                            text=True, encoding='utf-8', capture_output=True, check=True, timeout=20)
    assert not json.loads(result.stdout)['result']['ok']
    assert web.read_bytes() == before

@pytest.mark.parametrize('options', [[], ['--max-points', '0'], ['--radius', 'nan']])
def test_empty_or_invalid_preview_is_rejected_before_any_output(tmp_path, monkeypatch, options):
    import sys
    export = script('export')
    run, work, out = tmp_path / 'run', tmp_path / 'work', tmp_path / 'out'
    run.mkdir(); work.mkdir(); (work / 'init_points.npz').write_bytes(b'fixture')
    (run / 'train_stats.json').write_text(json.dumps({'work': str(work), 'sh_degree': 0}))
    torch.save(dict(means=torch.tensor([[1000., 0, 0]]), opacities=torch.zeros(1), sh0=torch.zeros((1, 1, 3))), run / 'ckpt.pt')
    scene = SimpleNamespace(dir=work, meta={'frames': []}, train=[], test=[], lidar_split=None, c2w=torch.eye(4)[None])
    monkeypatch.setattr(export, 'Scene', lambda _: scene)
    monkeypatch.setattr(sys, 'argv', ['export', '--run', str(run), '--out', str(out), '--skip-ply', '--allow-unverified-run'] + options)
    with pytest.raises(ValueError, match='preview'):
        export.main()
    assert not out.exists()

@pytest.mark.parametrize('binding', [None, {'schema': 1}])
def test_successful_report_cannot_upgrade_unbound_legacy_export(linked_web_report, binding):
    root, web = linked_web_report
    export_file = root / 'web.json'
    web.pop('viewer_check')
    if binding is None: web.pop('asset_binding')
    else: web['asset_binding'] = binding
    export_file.write_text(json.dumps(web), encoding='utf-8'); before = export_file.read_bytes()
    code = "const {linkReport}=require(process.argv[1]); const fs=require('fs'); linkReport(process.argv[2],process.argv[3],process.argv[4],JSON.parse(fs.readFileSync(process.argv[4],'utf8')))"
    run = subprocess.run(['node', '-e', code, str(ROOT / 'scripts/check_viewer.cjs'), str(root), str(export_file), str(root / 'evidence/report.json')],
                         capture_output=True, text=True)
    assert run.returncode != 0 and 'matching current asset binding' in run.stderr
    assert export_file.read_bytes() == before


def test_perfect_match_export_links_in_node_and_renders_readme(tmp_path, monkeypatch):
    import shutil
    export, check = script('export'), script('check_readme')
    monkeypatch.setattr(export, 'ROOT', tmp_path)
    monkeypatch.setattr(export, 'render', lambda *_: (torch.zeros((1, 1, 3)), None, None))
    checkpoint = dict(means=np.zeros((1, 3), np.float32), scales=np.zeros((1, 3), np.float32),
                      quats=np.array([[1, 0, 0, 0]], np.float32), opacities=np.zeros(1, np.float32),
                      sh0=np.zeros((1, 1, 3), np.float32), shN=np.zeros((1, 0, 3), np.float32))
    scene = SimpleNamespace(test=[0], c2w=torch.eye(4)[None], K=torch.eye(3), W=1, H=1,
                            image=lambda _: np.zeros((1, 1, 3), np.uint8), viewmat=lambda _: torch.eye(4))
    web = export.export_web(SimpleNamespace(splat_max=1, splat_min_opacity=0), checkpoint, scene,
                            tmp_path / 'docs/splat', 'sample', 0, device='cpu')
    (tmp_path / 'scripts').mkdir()
    shutil.copy2(ROOT / 'scripts/check_viewer.cjs', tmp_path / 'scripts/check_viewer.cjs')
    target = tmp_path / 'web.json'; target.write_text(json.dumps(web), encoding='utf-8')
    run = subprocess.run(['node', str(ROOT / 'tests/check_viewer_fixture.cjs')],
                         input=json.dumps(dict(root=str(tmp_path), checker=str(ROOT / 'scripts/check_viewer.cjs'),
                                               scenario='success', recordIn=str(target))),
                         text=True, encoding='utf-8', capture_output=True, timeout=20)
    assert run.returncode == 0, run.stderr  # Real JSON.parse and --record-in implementation.
    linked = json.loads(target.read_text(encoding='utf-8'))
    assert linked['heldout_psnr_mean'] == {'full_sh0': '+Infinity', 'decoded_asset_sh0': '+Infinity'}
    assert linked['viewer_check']['file'] == 'evidence/report.json'
    json.dumps(web, allow_nan=False)  # All output remains standard JSON.
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(check, 'load_json', lambda name: linked if name == 'results/web_export.json' else None)
    text = check.render_web()
    assert '+∞ dB' in text and '本次绑定的无头 Chromium 检查' in text


def test_findings_missing_static_and_depth_support_do_not_invent_differences():
    check = script('check_readme')
    runs = check.load_runs()
    for run in runs.values():
        for method in run.get('heldout_static_mean', {}).values():
            for key in method: method[key] = None
        for method in run.get('heldout_depth', {}).values():
            if isinstance(method, dict) and 'static' in method:
                method['static']['median_abs_m'] = None
                method['static']['within_tol'] = None
    text = check.render_findings(runs)
    assert '静态 PSNR 差 — dB' in text
    assert '静态 PSNR — dB' in text
    assert '— → — m' in text
    assert 'nan' not in text.lower()


def test_findings_identical_infinite_means_are_not_a_nan_effect():
    check = script('check_readme')
    runs = check.load_runs()
    for run in runs.values():
        run['heldout_mean']['3dgs']['psnr'] = float('inf')
        run['train_views_mean_3dgs']['psnr'] = float('inf')
        if 'heldout_static_mean' in run:
            run['heldout_static_mean']['3dgs']['psnr'] = float('inf')
    text = check.render_findings(runs)
    assert 'nan' not in text.lower()
    assert '留出 PSNR — dB' in text
    assert '静态 PSNR 差 — dB' in text


def test_finite_historical_findings_keep_defined_differences():
    check = script('check_readme')
    text = check.render_findings(check.load_runs())
    assert 'nan' not in text.lower()
    assert '差值未定义' not in text
    assert '部分差值未知' not in text
