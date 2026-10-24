const state = {
  files: {},
  sampleId: null,
  samples: [],
  result: null,
};

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const fileInput = $('#fileInput');
const btnRun = $('#btnRun');
const btnSample = $('#btnSample');
const sampleSelect = $('#sampleSelect');
const progressShell = $('#progressShell');
const progressFill = $('#progressFill');
const progressNote = $('#progressNote');
const resultPanel = $('#resultPanel');

let activeSlot = null;

$$('.upload-slot').forEach((slot) => {
  const modal = slot.dataset.modal;
  slot.addEventListener('click', () => {
    activeSlot = modal;
    fileInput.click();
  });
  slot.addEventListener('dragover', (e) => {
    e.preventDefault();
    slot.classList.add('dragover');
  });
  slot.addEventListener('dragleave', () => slot.classList.remove('dragover'));
  slot.addEventListener('drop', (e) => {
    e.preventDefault();
    slot.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f) acceptFile(modal, f);
  });
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0] && activeSlot) acceptFile(activeSlot, fileInput.files[0]);
  fileInput.value = '';
});

function acceptFile(modal, file) {
  if (!/\.(nii|nii\.gz|gz)$/i.test(file.name)) {
    toastSlot(modal, '仅支持 .nii / .nii.gz');
    return;
  }
  state.files[modal] = file;
  state.sampleId = null;
  sampleSelect.hidden = true;
  const slot = $(`.upload-slot[data-modal="${modal}"]`);
  slot.classList.add('filled');
  slot.querySelector('.slot-file').textContent = file.name + ' · ' + fmtSize(file.size);
  refreshRun();
}

function toastSlot(modal, msg) {
  const slot = $(`.upload-slot[data-modal="${modal}"]`);
  const el = slot.querySelector('.slot-file');
  el.textContent = msg;
  el.style.color = '#e88';
  setTimeout(() => { el.style.color = ''; }, 2200);
}

function fmtSize(n) {
  if (n > 1048576) return (n / 1048576).toFixed(1) + ' MB';
  return (n / 1024).toFixed(0) + ' KB';
}

function refreshRun() {
  btnRun.disabled = !(state.sampleId || Object.keys(state.files).length === 4);
}

btnSample.addEventListener('click', async () => {
  if (!state.samples.length) {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      state.samples = data.cases || [];
    } catch (e) {
      progressNote.textContent = '示例列表加载失败';
      return;
    }
  }
  if (!state.samples.length) {
    progressNote.textContent = '未找到 BraTS 2023 数据集';
    return;
  }
  sampleSelect.innerHTML = '';
  state.samples.forEach((c) => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    sampleSelect.appendChild(opt);
  });
  sampleSelect.hidden = false;
  sampleSelect.value = state.samples[Math.floor(Math.random() * state.samples.length)];
  state.sampleId = sampleSelect.value;
  state.files = {};
  $$('.upload-slot').forEach((s) => {
    s.classList.remove('filled');
    s.querySelector('.slot-file').textContent = '· 已选择示例 ·';
  });
  refreshRun();
});

sampleSelect.addEventListener('change', () => {
  state.sampleId = sampleSelect.value;
});

btnRun.addEventListener('click', runSegment);

async function runSegment() {
  if (btnRun.disabled) return null;
  btnRun.disabled = true;
  btnRun.textContent = '分割中…';
  progressShell.hidden = false;

  const steps = $$('.progress-steps span');
  let pct = 0;
  const timer = setInterval(() => {
    pct = Math.min(pct + Math.random() * 7, 92);
    progressFill.style.width = pct + '%';
    const idx = Math.min(Math.floor(pct / 24), 3);
    steps.forEach((s, i) => s.classList.toggle('active', i <= idx));
    progressNote.textContent = ['读取四模态体数据…', '脑区提取与强度归一化…', '风险感知分割报告生成中…', '渲染三视图叠加…'][idx];
  }, 600);

  try {
    let res;
    if (state.sampleId) {
      const fd = new FormData();
      fd.append('sample_id', state.sampleId);
      res = await fetch('/api/segment', { method: 'POST', body: fd });
    } else {
      const fd = new FormData();
      ['t1n', 't1c', 't2w', 't2f'].forEach((k) => fd.append(k, state.files[k]));
      res = await fetch('/api/segment', { method: 'POST', body: fd });
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '分割失败');
    progressFill.style.width = '100%';
    progressNote.textContent = `完成：肿瘤体素 ${data.tumor_voxels.toLocaleString()} / 脑区体素 ${data.brain_voxels.toLocaleString()}`;
    state.result = data;
    renderResult(data);
    return data;
  } catch (e) {
    progressNote.textContent = '错误：' + e.message;
    return null;
  } finally {
    clearInterval(timer);
    btnRun.disabled = false;
    btnRun.textContent = '开始分割';
  }
}

async function autoDemo() {

  document.getElementById('studio').scrollIntoView({ behavior: 'smooth' });
  try {
    if (!state.samples.length) {
      const res = await fetch('/api/samples');
      const data = await res.json();
      state.samples = data.cases || [];
    }
    if (!state.samples.length) throw new Error('未找到 BraTS 2023 数据集');

    sampleSelect.innerHTML = '';
    state.samples.forEach((c) => {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = c;
      sampleSelect.appendChild(opt);
    });
    sampleSelect.hidden = false;
    sampleSelect.value = state.samples[Math.floor(Math.random() * state.samples.length)];
    state.sampleId = sampleSelect.value;
    state.files = {};
    $$('.upload-slot').forEach((s) => {
      s.classList.remove('filled');
      s.querySelector('.slot-file').textContent = '· 已选择示例 ·';
    });
    refreshRun();

    const seg = await runSegment();
    if (!seg) throw new Error(progressNote.textContent || '分割失败');

    progressNote.textContent = '分割完成，正在出具影像报告…';
    await generateReport(seg.features);
  } catch (e) {
    if (progressNote) progressNote.textContent = '自动演示中断：' + e.message;
  }
}

function renderResult(data) {
  const f = data.features;
  const r = f.regions || {};
  const wt = r.WT || {}, tc = r.TC || {}, et = r.ET || {};
  resultPanel.innerHTML = `
    <div class="view-tabs">
      <button class="view-tab active" data-view="axial">轴状位 Axial</button>
      <button class="view-tab" data-view="coronal">冠状位 Coronal</button>
      <button class="view-tab" data-view="sagittal">矢状位 Sagittal</button>
    </div>
    <div class="view-stage"><img id="viewImg" src="${data.images.axial}" alt="分割叠加视图"></div>
    <div class="legend">
      <div class="legend-item"><span class="legend-dot dot-ed"></span>瘤周水肿 ED</div>
      <div class="legend-item"><span class="legend-dot dot-ncr"></span>坏死核心 NCR</div>
      <div class="legend-item"><span class="legend-dot dot-et"></span>增强肿瘤 ET</div>
    </div>
    <div class="metrics-grid">
      <div class="metric wt"><small>全肿瘤 WT 体积</small><b>${wt.volume_cm3 || '-'} cm³</b></div>
      <div class="metric tc"><small>肿瘤核心 TC 体积</small><b>${tc.volume_cm3 || '-'} cm³</b></div>
      <div class="metric et"><small>增强肿瘤 ET 体积</small><b>${et.volume_cm3 || '-'} cm³</b></div>
    </div>
    <div class="report-shell">
      <div class="report-head">
        <h4>影像检查报告</h4>
        <span class="engine-tag" id="engineTag">待生成</span>
      </div>
      <button class="btn btn-ghost btn-sm ${state.deepseekAvailable ? 'btn-report-hl' : ''}" id="btnReport">
        出具影像报告${state.deepseekAvailable ? '（DeepSeek V4，约10-60秒）' : ''}
      </button>
      <div class="report-body" id="reportBody" hidden></div>
    </div>`;
  $$('.view-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.view-tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      $('#viewImg').src = data.images[tab.dataset.view];
    });
  });
  $('#btnReport').addEventListener('click', (e) => {
    e.target.classList.remove('btn-report-hl');
    generateReport(f);
  });
}

async function generateReport(features, forceLocal = false) {
  if (!state.deepseekAvailable && !forceLocal) {
    renderKeyPanel(features);
    return;
  }
  const body = $('#reportBody');
  const tag = $('#engineTag');
  const btn = $('#btnReport');
  btn.disabled = true;
  btn.textContent = '报告生成中…';
  body.hidden = false;
  body.classList.add('loading');
  body.textContent = state.deepseekAvailable
    ? 'DeepSeek V4 正在分析分割特征并撰写报告，约需 10-60 秒…'
    : '正在使用内置规则模板出具影像报告…';
  try {
    const res = await fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '报告生成失败');
    body.classList.remove('loading');
    body.textContent = data.report;
    tag.textContent = data.engine === 'local-template' ? '本地规则引擎' : 'DeepSeek V4';
  } catch (e) {
    body.classList.remove('loading');
    body.textContent = '生成失败：' + e.message;
    tag.textContent = '失败';
  } finally {
    btn.disabled = false;
    btn.textContent = '重新出具报告';
  }
}

function renderKeyPanel(features) {
  const tag = $('#engineTag');
  if (tag) tag.textContent = '未接入 AI';
  const shell = document.querySelector('.report-shell');
  if (!shell) return;
  let panel = document.getElementById('keyPanel');
  if (panel) {
    panel.hidden = false;
    return;
  }
  panel = document.createElement('div');
  panel.id = 'keyPanel';
  panel.className = 'key-panel';
  panel.innerHTML = `
    <p class="key-panel-desc">接入 <b>DeepSeek V4</b> 可获得更深入的六段式 AI 病例报告；不接入则用内置规则模板生成。Key 仅保存在本机，不会上传。</p>
    <div class="key-panel-row">
      <input type="text" id="keyInput" placeholder="sk-（从 platform.deepseek.com 获取）" spellcheck="false" autocomplete="off">
      <button class="btn btn-primary btn-sm" id="btnKeySave">保存并生成 AI 报告</button>
    </div>
    <div class="modal-msg" id="keyMsg"></div>
    <div class="key-panel-links">
      <a href="https://platform.deepseek.com" target="_blank">如何获取 API Key？</a>
      <a href="javascript:void(0)" id="btnSkipKey">跳过，用规则模板生成报告</a>
    </div>`;
  shell.appendChild(panel);
  const msg = panel.querySelector('#keyMsg');
  panel.querySelector('#btnSkipKey').addEventListener('click', () => {
    panel.hidden = true;
    generateReport(features, true);
  });
  panel.querySelector('#btnKeySave').addEventListener('click', async () => {
    const key = panel.querySelector('#keyInput').value.trim();
    if (!key) {
      msg.textContent = '请先粘贴 Key，或点击下方「跳过」用规则模板生成';
      msg.className = 'modal-msg err';
      return;
    }
    const saveBtn = panel.querySelector('#btnKeySave');
    saveBtn.disabled = true;
    saveBtn.textContent = '校验中…';
    msg.className = 'modal-msg';
    msg.textContent = '正在向 DeepSeek 校验 Key…';
    try {
      const res = await fetch('/api/set_key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '校验失败');
      state.deepseekAvailable = true;
      panel.hidden = true;
      const btn = $('#btnReport');
      if (btn) {
        btn.textContent = '出具影像报告（DeepSeek V4，约10-60秒）';
        btn.classList.add('btn-report-hl');
      }
      generateReport(features);
    } catch (e) {
      let m = e.message;
      if (m.includes('401')) m = 'Key 无效（401），请到 platform.deepseek.com 重新复制完整 Key';
      else if (m.includes('402')) m = '余额不足（402），请先到平台充值（几元即可）';
      else if (m.includes('timed out') || m.includes('Timeout')) m = '网络超时，请稍后重试';
      msg.className = 'modal-msg err';
      msg.textContent = '失败：' + m + '（也可点击「跳过」用规则模板生成）';
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = '保存并生成 AI 报告';
    }
  });
}

function refreshKeyState() {
  fetch('/api/config')
    .then((r) => r.json())
    .then((cfg) => {
      state.deepseekAvailable = !!cfg.deepseek_available;
      if (cfg.dataset_found === false) btnSample.style.opacity = 0.45;
    })
    .catch(() => {});
}

refreshKeyState();
