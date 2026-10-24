# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

import requests

CONFIG_PATH = Path(__file__).parent / 'config.json'


def _load_key():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding='utf-8')).get('deepseek_api_key', '')
        except Exception:
            pass
    return ''


def _current_key():
    return os.environ.get('DEEPSEEK_API_KEY', '') or _load_key()


def set_api_key(key: str):
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        except Exception:
            cfg = {}
    cfg['deepseek_api_key'] = key.strip()
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')


DEEPSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-pro')
DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'
TIMEOUT = 180

SYSTEM_PROMPT = (
    '你是一位三甲医院影像科的资深放射科医师，为胶质瘤MRI分割结果出具正式影像报告。'
    '要求：语言精炼克制、使用规范医学术语、不做冗余展开，全文不超过350字。'
    '严格按医院影像报告格式输出。'
)

REPORT_TEMPLATE = '''请严格按以下格式输出（各栏文字精炼，影像所见不超过120字，诊断意见2-3条、每条不超过40字）：

【影像检查报告】

检查项目：颅脑多模态 MRI（T1/T1c/T2/FLAIR）+ AI 辅助分割
病例编号：{case}

—— 影像所见 ——
（一段式描述：病灶位置、各区域构成与信号特征，直接引用测量数值）

—— 分割测量 ——
全肿瘤 WT　　xx.x cm³（最大径 xx mm）
肿瘤核心 TC　　xx.x cm³
增强肿瘤 ET　　xx.x cm³
瘤周水肿 ED　　xx.x cm³

—— 诊断意见 ——
1. （主诊断：病灶性质与级别倾向）
2. （关键征象：如坏死、强化、水肿程度）
3. （建议：随访或进一步检查，一句话）

备注：本报告由 AI 辅助生成，仅供科研教学参考。'''


def _user_prompt(features):
    return (
        f'病例编号: {features.get("case_name", "Unknown")}\n'
        f'分割特征JSON:\n```json\n{json.dumps(features, ensure_ascii=False, indent=1)}\n```\n'
        '按以下模板出具报告，将测量数值填入对应栏位，影像所见与诊断意见基于数值做专业解读：\n'
        + REPORT_TEMPLATE.format(case=features.get('case_name', 'Unknown'))
    )


def generate_with_deepseek(features, key):
    body = {
        'model': DEEPSEEK_MODEL,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': _user_prompt(features)},
        ],
        'stream': False,
        'temperature': 0.3,
        'max_tokens': 4096,
        'thinking': {'type': 'disabled'},
    }
    resp = requests.post(
        DEEPSEEK_URL,
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        json=body,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    message = resp.json()['choices'][0]['message']
    text = (message.get('content') or '').strip()
    if not text:
        raise RuntimeError('模型返回内容为空')
    return text, DEEPSEEK_MODEL


def _fmt(x, nd=1):
    return f'{x:.{nd}f}'


def generate_local(features):
    r = features.get('regions', {})
    ratios = features.get('ratios', {})
    case = features.get('case_name', '未知病例')
    wt, tc, et = r.get('WT', {}), r.get('TC', {}), r.get('ET', {})
    ncr, ed = r.get('NCR', {}), r.get('ED', {})

    def vol(e):
        return e.get('volume_cm3', 0)

    et_ratio = ratios.get('ET_over_TC', 0) or 0
    loc = wt.get('centroid_voxel', ['-', '-', '-'])
    grade = '倾向高级别胶质瘤（WHO Ⅲ-Ⅳ级）' if et_ratio > 0.5 else '级别待定，建议结合灌注评估'
    necro = '病灶内可见坏死区，增强呈不规则环形强化' if vol(ncr) > 0.5 else '未见明显坏死，增强较均匀'
    ed_dom = '瘤周水肿广泛，呈浸润性生长' if vol(ed) > vol(wt) * 0.8 else '瘤周水肿程度中等'

    lines = [
        '【影像检查报告】',
        '',
        '检查项目：颅脑多模态 MRI（T1/T1c/T2/FLAIR）+ AI 辅助分割',
        f'病例编号：{case}',
        '',
        '—— 影像所见 ——',
        f'右侧大脑半球见占位性病灶（质心体素 {loc[0]}/{loc[1]}/{loc[2]}），'
        f'全肿瘤 {_fmt(vol(wt))} cm³。FLAIR 序列病灶区信号增高，境界欠清；T1c 序列{necro}；'
        f'T1 于核心区呈低信号。{ed_dom}，TC/WT {_fmt(ratios.get("TC_over_WT", 0), 3)}，ET/TC {_fmt(et_ratio, 3)}。',
        '',
        '—— 分割测量 ——',
        f'全肿瘤 WT　　{_fmt(vol(wt))} cm³（最大径 {wt.get("max_diameter_mm", "-")} mm）',
        f'肿瘤核心 TC　　{_fmt(vol(tc))} cm³',
        f'增强肿瘤 ET　　{_fmt(vol(et))} cm³',
        f'瘤周水肿 ED　　{_fmt(vol(ed))} cm³',
        '',
        '—— 诊断意见 ——',
        f'1. 颅内占位，{grade}。',
        f'2. {necro}，伴瘤周水肿。',
        '3. 建议 MRI 增强随访，结合灌注/波谱进一步评估。',
        '',
        '备注：本报告由 AI 辅助生成，仅供科研教学参考。',
    ]
    return '\n'.join(lines), 'local-template'


def generate_report(features):
    key = _current_key()
    if key:
        try:
            return generate_with_deepseek(features, key)
        except Exception as e:
            fallback = generate_local(features)
            return fallback[0] + f'\n\n[注] DeepSeek V4 调用失败（{e}），已切换本地规则模板生成。', fallback[1]
    return generate_local(features)