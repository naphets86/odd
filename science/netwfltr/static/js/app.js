'use strict';

const PAGE_SIZE = 100;
let currentPage = 0;
let totalRecords = 0;
let refreshTimer = null;

// Bekannte IDs als Map: id -> DOM-Zeile
// So wird niemals die gesamte Tabelle neu gebaut
const rowMap = new Map();

const LAYER_NAMES = {
  2: 'Data Link', 3: 'Netzwerk', 4: 'Transport',
  5: 'Sitzung',  6: 'Darstellung', 7: 'Anwendung'
};

// ── Filter ────────────────────────────────────────────────
function getFilterParams() {
  return {
    protocol:  document.getElementById('f-protocol').value || 'ALL',
    date_from: document.getElementById('f-date-from').value || '',
    date_to:   document.getElementById('f-date-to').value   || '',
    time_from: document.getElementById('f-time-from').value || '',
    time_to:   document.getElementById('f-time-to').value   || '',
  };
}

function buildQuery(extra = {}) {
  const f = getFilterParams();
  const p = new URLSearchParams();
  if (f.protocol && f.protocol !== 'ALL') p.set('protocol', f.protocol);
  if (f.date_from) p.set('date_from', f.date_from);
  if (f.date_to)   p.set('date_to',   f.date_to);
  if (f.time_from) p.set('time_from', f.time_from);
  if (f.time_to)   p.set('time_to',   f.time_to);
  p.set('limit',  extra.limit  ?? PAGE_SIZE);
  p.set('offset', extra.offset ?? currentPage * PAGE_SIZE);
  return p.toString();
}

// ── Fetch ─────────────────────────────────────────────────
async function fetchTraffic() {
  try {
    const res  = await fetch('/api/traffic?' + buildQuery());
    const data = await res.json();
    totalRecords = data.total;
    updateTable(data.data);
    updatePagination();
    document.getElementById('record-count').textContent =
      totalRecords.toLocaleString('de-DE');
  } catch (e) {
    console.error('Fetch error:', e);
  }
}

async function fetchStats() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();
    document.getElementById('total-count').textContent =
      data.total.toLocaleString('de-DE');
  } catch(e) {}
}

async function loadProtocols() {
  try {
    const res    = await fetch('/api/protocols');
    const protos = await res.json();
    const sel    = document.getElementById('f-protocol');
    const cur    = sel.value;
    while (sel.options.length > 1) sel.remove(1);
    protos.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.protocol;
      opt.textContent = `${p.protocol} (${p.count.toLocaleString('de-DE')})`;
      sel.appendChild(opt);
    });
    sel.value = cur; // Auswahl behalten
  } catch(e) {}
}

// ── DOM-Diff Update — KEIN kompletter Neuaufbau ───────────
function updateTable(rows) {
  const tbody = document.getElementById('traffic-body');

  if (!rows || rows.length === 0) {
    rowMap.clear();
    tbody.innerHTML =
      '<tr class="loading-row"><td colspan="9">Keine Einträge gefunden.</td></tr>';
    return;
  }

  // Beim Seitenwechsel oder Filter-Reset: komplett neu rendern
  if (currentPage > 0 || rowMap.size === 0) {
    rowMap.clear();
    const frag = document.createDocumentFragment();
    rows.forEach(r => {
      const tr = buildRow(r, false);
      rowMap.set(r.id, tr);
      frag.appendChild(tr);
    });
    tbody.innerHTML = '';
    tbody.appendChild(frag);
    return;
  }

  // Seite 0: Diff — nur neue Zeilen oben einfügen
  const newRows = rows.filter(r => !rowMap.has(r.id));

  if (newRows.length === 0) return; // Nichts geändert → kein DOM-Touch

  // Neue Zeilen sanft oben einfügen
  const frag = document.createDocumentFragment();
  newRows.forEach(r => {
    const tr = buildRow(r, true);
    rowMap.set(r.id, tr);
    frag.appendChild(tr);
  });

  // Vor dem ersten bestehenden TR einfügen
  const firstExisting = tbody.firstChild;
  if (firstExisting && firstExisting.classList?.contains('loading-row')) {
    tbody.innerHTML = '';
    tbody.appendChild(frag);
  } else {
    tbody.insertBefore(frag, firstExisting);
  }

  // Überzählige Zeilen am Ende entfernen (max PAGE_SIZE anzeigen)
  while (tbody.rows.length > PAGE_SIZE) {
    const last = tbody.lastChild;
    if (last) {
      // ID aus rowMap entfernen
      const id = parseInt(last.dataset.id);
      if (id) rowMap.delete(id);
      tbody.removeChild(last);
    }
  }
}

// ── Zeile bauen ───────────────────────────────────────────
function buildRow(r, isNew) {
  const layer   = r.osi_layer || 3;
  const tr      = document.createElement('tr');
  tr.setAttribute('data-layer', layer);
  tr.dataset.id = r.id;
  if (isNew) tr.classList.add('new-entry');

  const srcPort = r.src_port ? `:${r.src_port}` : '';
  const dstPort = r.dst_port ? `:${r.dst_port}` : '';
  const flags   = r.flags
    ? `<span class="flag-badge">[${escHtml(r.flags)}]</span>`
    : '<span style="color:var(--text-muted)">—</span>';
  const bytes   = r.length ? r.length.toLocaleString('de-DE') : '—';
  const info    = r.info ? escHtml(r.info.substring(0, 80)) : '—';
  const port    = r.dst_port || r.src_port
    ? `<span style="color:var(--text-dim);font-family:var(--font-mono)">${r.dst_port || r.src_port}</span>`
    : '<span style="color:var(--text-muted)">—</span>';

  tr.innerHTML = `
    <td>${formatTimestamp(r.timestamp)}</td>
    <td><span class="proto-badge proto-l${layer}">${escHtml(r.protocol || '?')}</span></td>
    <td><span class="osi-badge osi-l${layer}" title="${LAYER_NAMES[layer] || ''}">${layer}</span></td>
    <td style="font-family:var(--font-mono);font-size:12px">${escHtml(r.src_ip || '—')}${escHtml(srcPort)}</td>
    <td style="font-family:var(--font-mono);font-size:12px">${escHtml(r.dst_ip || '—')}${escHtml(dstPort)}</td>
    <td>${port}</td>
    <td style="text-align:right;font-family:var(--font-mono);font-size:12px">${bytes}</td>
    <td>${flags}</td>
    <td class="col-info" title="${escHtml(r.info || '')}">${info}</td>
  `;
  return tr;
}

// ── Hilfsfunktionen ───────────────────────────────────────
function formatTimestamp(ts) {
  if (!ts) return '—';
  const d  = new Date(ts.endsWith('Z') ? ts : ts + 'Z');
  const dd = String(d.getUTCDate()).padStart(2,'0');
  const mo = String(d.getUTCMonth()+1).padStart(2,'0');
  const yy = d.getUTCFullYear();
  const hh = String(d.getUTCHours()).padStart(2,'0');
  const mm = String(d.getUTCMinutes()).padStart(2,'0');
  const ss = String(d.getUTCSeconds()).padStart(2,'0');
  return `<span class="ts-date">${dd}.${mo}.${yy}</span> <span class="ts-time">${hh}:${mm}:${ss}</span>`;
}

function escHtml(s) {
  if (!s) return '';
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function updatePagination() {
  const total = Math.max(1, Math.ceil(totalRecords / PAGE_SIZE));
  document.getElementById('page-info').textContent =
    `Seite ${currentPage + 1} / ${total}  (${totalRecords.toLocaleString('de-DE')} Einträge)`;
  document.getElementById('btn-prev').disabled = currentPage === 0;
  document.getElementById('btn-next').disabled = (currentPage + 1) * PAGE_SIZE >= totalRecords;
}

// ── Filter-Aktionen ───────────────────────────────────────
function applyFilter() {
  currentPage = 0;
  rowMap.clear();
  const f     = getFilterParams();
  const parts = [];
  if (f.protocol !== 'ALL') parts.push(`Protokoll: ${f.protocol}`);
  if (f.date_from || f.date_to) parts.push(`Datum: ${f.date_from||'…'} – ${f.date_to||'…'}`);
  if (f.time_from || f.time_to) parts.push(`Uhrzeit: ${f.time_from||'…'} – ${f.time_to||'…'}`);
  const infoEl = document.getElementById('filter-info');
  if (parts.length) {
    infoEl.textContent = 'Filter aktiv: ' + parts.join(' | ');
    infoEl.classList.add('active');
  } else {
    infoEl.textContent = 'Alle Einträge werden angezeigt';
    infoEl.classList.remove('active');
  }
  fetchTraffic();
  loadProtocols();
}

function resetFilter() {
  ['f-protocol','f-date-from','f-date-to','f-time-from','f-time-to'].forEach(id => {
    const el = document.getElementById(id);
    el.value = id === 'f-protocol' ? 'ALL' : '';
  });
  const infoEl = document.getElementById('filter-info');
  infoEl.textContent = 'Alle Einträge werden angezeigt';
  infoEl.classList.remove('active');
  currentPage = 0;
  rowMap.clear();
  fetchTraffic();
}

function nextPage() {
  if ((currentPage + 1) * PAGE_SIZE < totalRecords) {
    currentPage++;
    rowMap.clear();
    fetchTraffic();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

function prevPage() {
  if (currentPage > 0) {
    currentPage--;
    rowMap.clear();
    fetchTraffic();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

async function clearData() {
  if (!confirm('Alle Netzwerkeinträge löschen?')) return;
  await fetch('/api/traffic', { method: 'DELETE' });
  rowMap.clear();
  currentPage = 0;
  fetchTraffic();
  fetchStats();
}

// ── Auto-Refresh ──────────────────────────────────────────
function startAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(() => {
    if (document.getElementById('auto-refresh').checked && currentPage === 0) {
      fetchTraffic();
      fetchStats();
    }
  }, 5000); // 5 Sekunden — ruhiger als vorher
}

document.getElementById('auto-refresh').addEventListener('change', function() {
  if (this.checked) startAutoRefresh();
  else if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }
});

// ── Init ──────────────────────────────────────────────────
(async () => {
  await loadProtocols();
  await fetchTraffic();
  await fetchStats();
  // Protokoll-Liste alle 30s aktualisieren
  setInterval(loadProtocols, 30_000);
  startAutoRefresh();
})();
