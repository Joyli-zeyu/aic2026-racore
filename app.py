# -*- coding: utf-8 -*-
import gzip
import io
import json
import os
import uuid
import zipfile
from pathlib import Path

import nibabel as nib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import reportgen
from segmenter import run_pipeline

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

import os

BRATS_DIR = BASE_DIR.parent / 'BraTS2023'
VAL_ZIP = BRATS_DIR / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip'
TRAIN_ZIP = BRATS_DIR / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip'
VAL_ZIP_PREFIX = 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/'

_env_zip = os.environ.get('BRATS_ZIP', '')
if _env_zip and Path(_env_zip).exists():
    VAL_ZIP = Path(_env_zip)
    VAL_ZIP_PREFIX = ''

MODAL_KEYS = ['t1n', 't1c', 't2w', 't2f']
SAMPLE_CACHE = []

app = FastAPI(title='RaCoRe 风险感知分割平台')


def list_samples():
    global SAMPLE_CACHE
    if SAMPLE_CACHE:
        return SAMPLE_CACHE
    if not VAL_ZIP.exists():
        return []
    with zipfile.ZipFile(VAL_ZIP) as zf:
        names = set(zf.namelist())
    basenames = {n.rsplit('/', 1)[-1] for n in names}
    cases = []
    for name in names:
        if not name.endswith('-t1c.nii.gz'):
            continue
        case_id = name.rsplit('/', 1)[-1].replace('-t1c.nii.gz', '')
        if all(f'{case_id}-{k}.nii.gz' in basenames for k in MODAL_KEYS):
            cases.append(case_id)
    SAMPLE_CACHE = sorted(cases)
    return SAMPLE_CACHE


def load_nii_from_bytes(raw: bytes, name: str) -> nib.Nifti1Image:
    if name.endswith('.gz') or raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return nib.Nifti1Image.from_bytes(raw)


def load_sample_images(case_id: str):
    with zipfile.ZipFile(VAL_ZIP) as zf:
        imgs = [load_nii_from_bytes(zf.read(f'{VAL_ZIP_PREFIX}{case_id}/{case_id}-{k}.nii.gz'), k) for k in MODAL_KEYS]
    return imgs


@app.get('/')
def index():
    return FileResponse(
        BASE_DIR / 'static' / 'index.html',
        headers={'Cache-Control': 'no-cache, no-store, must-revalidate', 'Pragma': 'no-cache'},
    )


@app.get('/api/samples')
def samples():
    cases = list_samples()
    return {'count': len(cases), 'cases': cases[:40]}


@app.get('/api/config')
def config():
    has_key = bool(reportgen._current_key())
    return {
        'deepseek_available': has_key,
        'deepseek_model': reportgen.DEEPSEEK_MODEL if has_key else 'local-template',
        'dataset_found': VAL_ZIP.exists(),
    }


@app.post('/api/set_key')
async def set_key(payload: dict):
    admin_token = os.environ.get('ADMIN_TOKEN', '')
    if admin_token:
        provided = (payload.get('admin_token') or '').strip()
        if provided != admin_token:
            raise HTTPException(403, '公网模式下已锁定 Key 配置（防止被他人篡改）。需修改请在服务器本机操作')
    key = (payload.get('api_key') or '').strip()
    if not key:
        raise HTTPException(400, 'api_key 不能为空')
    test_msg = payload.get('test', True)
    if test_msg:
        try:
            probe = {
                'model': reportgen.DEEPSEEK_MODEL,
                'messages': [{'role': 'user', 'content': 'ping'}],
                'max_tokens': 8,
                'thinking': {'type': 'disabled'},
            }
            import requests as rq
            r = rq.post(
                reportgen.DEEPSEEK_URL,
                headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                json=probe,
                timeout=30,
            )
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(400, f'Key 校验失败: {e}')
    reportgen.set_api_key(key)
    return {'ok': True, 'model': reportgen.DEEPSEEK_MODEL}


@app.post('/api/segment')
async def segment(
    sample_id: str = Form(None),
    t1n: UploadFile = File(None),
    t1c: UploadFile = File(None),
    t2w: UploadFile = File(None),
    t2f: UploadFile = File(None),
):
    case_dir_name = 'CASE_' + uuid.uuid4().hex[:8].upper()
    case_dir = RESULTS_DIR / case_dir_name
    case_dir.mkdir(parents=True, exist_ok=True)

    if sample_id:
        valid = list_samples()
        if sample_id not in valid:
            raise HTTPException(400, f'示例病例 {sample_id} 不存在')
        case_name = sample_id
        imgs = load_sample_images(sample_id)
    else:
        files = {'t1n': t1n, 't1c': t1c, 't2w': t2w, 't2f': t2f}
        missing = [k for k, v in files.items() if v is None or not v.filename]
        if missing:
            raise HTTPException(400, f'缺少模态文件: {", ".join(missing)}')
        case_name = case_dir_name
        imgs = []
        for k in MODAL_KEYS:
            raw = await files[k].read()
            if len(raw) > 300 * 1024 * 1024:
                raise HTTPException(400, f'{k} 文件过大')
            try:
                imgs.append(load_nii_from_bytes(raw, files[k].filename))
            except Exception as e:
                raise HTTPException(400, f'{k} 不是有效的 NIfTI 文件: {e}')
        shapes = [tuple(img.shape) for img in imgs]
        if len(set(shapes)) > 1:
            raise HTTPException(
                400,
                '四个模态的图像尺寸不一致，请确认上传的是同一病例：'
                + ', '.join(f'{k}={s}' for k, s in zip(MODAL_KEYS, shapes)),
            )

    try:
        features, overlay_paths, n_tumor, n_brain = run_pipeline(
            imgs, case_name, str(case_dir / case_dir_name)
        )
    except Exception as e:
        raise HTTPException(500, f'分割失败: {e}')

    rel = lambda p: f'/results/{case_dir_name}/{Path(p).name}'
    features_path = case_dir / f'{case_dir_name}_features.json'
    features_path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding='utf-8')
    return {
        'case_id': case_dir_name,
        'case_name': case_name,
        'source': 'sample' if sample_id else 'upload',
        'tumor_voxels': n_tumor,
        'brain_voxels': int(n_brain),
        'features': features,
        'images': {k: rel(v) for k, v in overlay_paths.items()},
        'features_url': f'/results/{case_dir_name}/{case_dir_name}_features.json',
    }


@app.post('/api/report')
async def report(payload: dict):
    features = payload.get('features')
    if not features:
        raise HTTPException(400, '缺少 features 数据')
    text, engine = reportgen.generate_report(features)
    return {'report': text, 'engine': engine}


@app.get('/results/{case_id}/{filename}')
def serve_result(case_id: str, filename: str):
    path = RESULTS_DIR / case_id / filename
    if not path.exists():
        raise HTTPException(404, '结果文件不存在')
    return FileResponse(path)


app.mount('/static', StaticFiles(directory=BASE_DIR / 'static'), name='static')


@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc):
    import traceback
    print(f'[错误] {request.url} -> {exc.status_code}: {exc.detail}')
    traceback.print_stack()
    return JSONResponse({'detail': exc.detail}, status_code=exc.status_code)


if __name__ == '__main__':
    import uvicorn
    print('=' * 56)
    print('本机访问:   http://127.0.0.1:8000')
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        lan_ip = s.getsockname()[0]
        s.close()
        print(f'局域网访问: http://{lan_ip}:8000（需放行防火墙）')
    except Exception:
        pass
    print('=' * 56)
    uvicorn.run(app, host='0.0.0.0', port=8000)