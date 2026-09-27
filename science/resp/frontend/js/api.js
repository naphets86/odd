// ─── API Base ────────────────────────────────────────────────────────────────
const API = window.location.hostname === 'localhost'
  ? 'http://localhost:8000/api'
  : '/api';

async function api(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const resp = await fetch(API + path, opts);
  if (resp.status === 204) return null;
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || 'Fehler');
  return data;
}

const get  = (p)    => api('GET',    p);
const post = (p, b) => api('POST',   p, b);
const patch= (p, b) => api('PATCH',  p, b);
const del  = (p)    => api('DELETE', p);

// ─── Toast ───────────────────────────────────────────────────────────────────
function toast(msg, type = 'success') {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = (type === 'success' ? '✓ ' : type === 'error' ? '✗ ' : 'ℹ ') + msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 3200);
}

// ─── Modal ───────────────────────────────────────────────────────────────────
function openModal(id) {
  document.getElementById(id).classList.add('open');
}
function closeModal(id) {
  document.getElementById(id).classList.remove('open');
  // clear form
  document.querySelectorAll(`#${id} input, #${id} select, #${id} textarea`)
    .forEach(el => el.value = '');
  const hiddenId = document.querySelector(`#${id} [name="_id"]`);
  if (hiddenId) hiddenId.value = '';
}
document.addEventListener('click', e => {
  if (e.target.classList.contains('modal-overlay')) {
    e.target.classList.remove('open');
  }
});

// ─── Navigation ──────────────────────────────────────────────────────────────
let currentPage = null;

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
  const page = document.getElementById('page-' + name);
  if (page) page.classList.add('active');
  const link = document.querySelector(`nav a[data-page="${name}"]`);
  if (link) link.classList.add('active');
  currentPage = name;
  pages[name]?.load();
}

// ─── Badge helpers ────────────────────────────────────────────────────────────
const statusBadge = s => {
  const map = { geplant: 'badge-blue', aktiv: 'badge-green', abgeschlossen: 'badge-gray', abgebrochen: 'badge-red' };
  return `<span class="badge ${map[s] || 'badge-gray'}">${s}</span>`;
};
const personTypeBadge = t => `<span class="badge badge-blue">${t}</span>`;
const materialTypeBadge = t => `<span class="badge badge-yellow">${t}</span>`;
const timeBadge = (a, u) => `<span class="badge badge-green">${a} ${u}</span>`;

// ─── Confirm Delete ────────────────────────────────────────────────────────────
function confirmDelete(msg, cb) {
  if (confirm(`Wirklich löschen?\n${msg}`)) cb();
}

// ─── Formatters ───────────────────────────────────────────────────────────────
const fmtDate = d => d ? new Date(d).toLocaleDateString('de-DE') : '—';
const fmtDateTime = d => d ? new Date(d).toLocaleString('de-DE') : '—';

// ─── Enum cache ───────────────────────────────────────────────────────────────
let ENUMS = null;
async function getEnums() {
  if (!ENUMS) ENUMS = await get('/enums');
  return ENUMS;
}

// ─── Select builder ───────────────────────────────────────────────────────────
function buildOptions(arr, selected = '') {
  return arr.map(v => `<option value="${v}" ${v === selected ? 'selected' : ''}>${v}</option>`).join('');
}
