// ─── Page: Dashboard ─────────────────────────────────────────────────────────
const pages = {};

pages.dashboard = {
  async load() {
    const [persons, materials, timeRes, projects, allocations] = await Promise.all([
      get('/persons/?limit=1000'),
      get('/materials/?limit=1000'),
      get('/time-resources/?limit=1000'),
      get('/projects/?limit=1000'),
      get('/allocations/?limit=1000'),
    ]);
    document.getElementById('stat-persons').textContent = persons.length;
    document.getElementById('stat-materials').textContent = materials.length;
    document.getElementById('stat-time').textContent = timeRes.length;
    document.getElementById('stat-projects').textContent = projects.length;
    document.getElementById('stat-allocations').textContent = allocations.length;

    // Recent projects
    const tbody = document.getElementById('dash-projects');
    tbody.innerHTML = projects.slice(0, 8).map(p => `
      <tr>
        <td>${p.name}</td>
        <td>${statusBadge(p.status)}</td>
        <td>${fmtDate(p.start_date)}</td>
        <td>${fmtDate(p.end_date)}</td>
      </tr>
    `).join('') || '<tr><td colspan="4" class="text-muted" style="text-align:center">Keine Projekte</td></tr>';
  }
};

// ─── Page: Persons ────────────────────────────────────────────────────────────
pages.persons = {
  data: [],
  async load() {
    this.data = await get('/persons/?limit=1000');
    this.render();
  },
  render() {
    const tbody = document.getElementById('persons-tbody');
    if (!this.data.length) {
      tbody.innerHTML = '<tr><td colspan="5"><div class="empty"><div class="empty-icon">👤</div><p>Noch keine Personen erfasst</p></div></td></tr>';
      return;
    }
    tbody.innerHTML = this.data.map(p => `
      <tr>
        <td><strong>${p.name}</strong></td>
        <td>${personTypeBadge(p.person_type)}</td>
        <td>${p.email || '—'}</td>
        <td>${p.department || '—'}</td>
        <td style="white-space:nowrap">
          <button class="btn-icon" onclick="editPerson(${p.id})" title="Bearbeiten">✏️</button>
          <button class="btn-icon danger" onclick="deletePerson(${p.id},'${p.name.replace(/'/g,"\\'")}') " title="Löschen">🗑️</button>
        </td>
      </tr>
    `).join('');
  },
  find(id) { return this.data.find(p => p.id === id); }
};

async function openPersonModal(person = null) {
  const enums = await getEnums();
  document.getElementById('person-type-select').innerHTML = buildOptions(enums.person_types, person?.person_type);
  if (person) {
    document.getElementById('person-modal-title').textContent = 'Person bearbeiten';
    document.getElementById('pm-id').value = person.id;
    document.getElementById('pm-name').value = person.name;
    document.getElementById('pm-email').value = person.email || '';
    document.getElementById('pm-dept').value = person.department || '';
    document.getElementById('pm-notes').value = person.notes || '';
  } else {
    document.getElementById('person-modal-title').textContent = 'Person hinzufügen';
    document.getElementById('pm-id').value = '';
  }
  openModal('person-modal');
}

async function savePerson() {
  const id = document.getElementById('pm-id').value;
  const payload = {
    name: document.getElementById('pm-name').value,
    person_type: document.getElementById('person-type-select').value,
    email: document.getElementById('pm-email').value || null,
    department: document.getElementById('pm-dept').value || null,
    notes: document.getElementById('pm-notes').value || null,
  };
  try {
    if (id) {
      await patch(`/persons/${id}`, payload);
      toast('Person aktualisiert');
    } else {
      await post('/persons/', payload);
      toast('Person hinzugefügt');
    }
    closeModal('person-modal');
    await pages.persons.load();
  } catch (e) { toast(e.message, 'error'); }
}

function editPerson(id) {
  openPersonModal(pages.persons.find(id));
}

async function deletePerson(id, name) {
  confirmDelete(name, async () => {
    try {
      await del(`/persons/${id}`);
      toast('Person gelöscht', 'info');
      await pages.persons.load();
    } catch (e) { toast(e.message, 'error'); }
  });
}

// ─── Page: Materials ──────────────────────────────────────────────────────────
pages.materials = {
  data: [],
  async load() {
    this.data = await get('/materials/?limit=1000');
    this.render();
  },
  render() {
    const tbody = document.getElementById('materials-tbody');
    if (!this.data.length) {
      tbody.innerHTML = '<tr><td colspan="5"><div class="empty"><div class="empty-icon">📦</div><p>Noch kein Material erfasst</p></div></td></tr>';
      return;
    }
    tbody.innerHTML = this.data.map(m => `
      <tr>
        <td><strong>${m.name}</strong></td>
        <td>${materialTypeBadge(m.material_type)}</td>
        <td>${m.quantity} ${m.unit}</td>
        <td>${m.location || '—'}</td>
        <td style="white-space:nowrap">
          <button class="btn-icon" onclick="editMaterial(${m.id})" title="Bearbeiten">✏️</button>
          <button class="btn-icon danger" onclick="deleteMaterial(${m.id},'${m.name.replace(/'/g,"\\'")}')">🗑️</button>
        </td>
      </tr>
    `).join('');
  },
  find(id) { return this.data.find(m => m.id === id); }
};

async function openMaterialModal(mat = null) {
  const enums = await getEnums();
  document.getElementById('material-type-select').innerHTML = buildOptions(enums.material_types, mat?.material_type);
  if (mat) {
    document.getElementById('material-modal-title').textContent = 'Material bearbeiten';
    document.getElementById('mm-id').value = mat.id;
    document.getElementById('mm-name').value = mat.name;
    document.getElementById('mm-qty').value = mat.quantity;
    document.getElementById('mm-unit').value = mat.unit;
    document.getElementById('mm-loc').value = mat.location || '';
    document.getElementById('mm-notes').value = mat.notes || '';
  } else {
    document.getElementById('material-modal-title').textContent = 'Material hinzufügen';
    document.getElementById('mm-id').value = '';
  }
  openModal('material-modal');
}

async function saveMaterial() {
  const id = document.getElementById('mm-id').value;
  const payload = {
    name: document.getElementById('mm-name').value,
    material_type: document.getElementById('material-type-select').value,
    quantity: parseFloat(document.getElementById('mm-qty').value) || 0,
    unit: document.getElementById('mm-unit').value || 'Stück',
    location: document.getElementById('mm-loc').value || null,
    notes: document.getElementById('mm-notes').value || null,
  };
  try {
    if (id) { await patch(`/materials/${id}`, payload); toast('Material aktualisiert'); }
    else     { await post('/materials/', payload);       toast('Material hinzugefügt'); }
    closeModal('material-modal');
    await pages.materials.load();
  } catch (e) { toast(e.message, 'error'); }
}

function editMaterial(id) { openMaterialModal(pages.materials.find(id)); }
async function deleteMaterial(id, name) {
  confirmDelete(name, async () => {
    try { await del(`/materials/${id}`); toast('Material gelöscht', 'info'); await pages.materials.load(); }
    catch (e) { toast(e.message, 'error'); }
  });
}

// ─── Page: Time Resources ─────────────────────────────────────────────────────
pages.time = {
  data: [],
  async load() {
    this.data = await get('/time-resources/?limit=1000');
    this.render();
  },
  render() {
    const tbody = document.getElementById('time-tbody');
    if (!this.data.length) {
      tbody.innerHTML = '<tr><td colspan="4"><div class="empty"><div class="empty-icon">⏱️</div><p>Noch keine Zeitressourcen erfasst</p></div></td></tr>';
      return;
    }
    tbody.innerHTML = this.data.map(t => `
      <tr>
        <td><strong>${t.name}</strong></td>
        <td>${timeBadge(t.amount, t.unit)}</td>
        <td>${t.notes || '—'}</td>
        <td style="white-space:nowrap">
          <button class="btn-icon" onclick="editTime(${t.id})">✏️</button>
          <button class="btn-icon danger" onclick="deleteTime(${t.id},'${t.name.replace(/'/g,"\\'")}')">🗑️</button>
        </td>
      </tr>
    `).join('');
  },
  find(id) { return this.data.find(t => t.id === id); }
};

async function openTimeModal(tr = null) {
  const enums = await getEnums();
  document.getElementById('time-unit-select').innerHTML = buildOptions(enums.time_units, tr?.unit);
  if (tr) {
    document.getElementById('time-modal-title').textContent = 'Zeitressource bearbeiten';
    document.getElementById('tm-id').value = tr.id;
    document.getElementById('tm-name').value = tr.name;
    document.getElementById('tm-amount').value = tr.amount;
    document.getElementById('tm-notes').value = tr.notes || '';
  } else {
    document.getElementById('time-modal-title').textContent = 'Zeitressource hinzufügen';
    document.getElementById('tm-id').value = '';
  }
  openModal('time-modal');
}

async function saveTime() {
  const id = document.getElementById('tm-id').value;
  const payload = {
    name: document.getElementById('tm-name').value,
    amount: parseFloat(document.getElementById('tm-amount').value),
    unit: document.getElementById('time-unit-select').value,
    notes: document.getElementById('tm-notes').value || null,
  };
  try {
    if (id) { await patch(`/time-resources/${id}`, payload); toast('Zeitressource aktualisiert'); }
    else     { await post('/time-resources/', payload);       toast('Zeitressource hinzugefügt'); }
    closeModal('time-modal');
    await pages.time.load();
  } catch (e) { toast(e.message, 'error'); }
}

function editTime(id) { openTimeModal(pages.time.find(id)); }
async function deleteTime(id, name) {
  confirmDelete(name, async () => {
    try { await del(`/time-resources/${id}`); toast('Zeitressource gelöscht', 'info'); await pages.time.load(); }
    catch (e) { toast(e.message, 'error'); }
  });
}

// ─── Page: Projects ───────────────────────────────────────────────────────────
pages.projects = {
  data: [],
  async load() {
    this.data = await get('/projects/?limit=1000');
    this.render();
  },
  render() {
    const tbody = document.getElementById('projects-tbody');
    if (!this.data.length) {
      tbody.innerHTML = '<tr><td colspan="5"><div class="empty"><div class="empty-icon">📋</div><p>Noch keine Projekte erfasst</p></div></td></tr>';
      return;
    }
    tbody.innerHTML = this.data.map(p => `
      <tr>
        <td><strong>${p.name}</strong></td>
        <td>${statusBadge(p.status)}</td>
        <td>${fmtDate(p.start_date)}</td>
        <td>${fmtDate(p.end_date)}</td>
        <td style="white-space:nowrap">
          <button class="btn-icon" onclick="editProject(${p.id})">✏️</button>
          <button class="btn-icon" onclick="viewProjectAllocations(${p.id},'${p.name.replace(/'/g,"\\'")}')">🔗</button>
          <button class="btn-icon danger" onclick="deleteProject(${p.id},'${p.name.replace(/'/g,"\\'")}')">🗑️</button>
        </td>
      </tr>
    `).join('');
  },
  find(id) { return this.data.find(p => p.id === id); }
};

async function openProjectModal(proj = null) {
  const enums = await getEnums();
  document.getElementById('proj-status-select').innerHTML = buildOptions(enums.allocation_statuses, proj?.status || 'geplant');
  if (proj) {
    document.getElementById('project-modal-title').textContent = 'Projekt bearbeiten';
    document.getElementById('pj-id').value = proj.id;
    document.getElementById('pj-name').value = proj.name;
    document.getElementById('pj-desc').value = proj.description || '';
    document.getElementById('pj-start').value = proj.start_date ? proj.start_date.slice(0, 16) : '';
    document.getElementById('pj-end').value = proj.end_date ? proj.end_date.slice(0, 16) : '';
  } else {
    document.getElementById('project-modal-title').textContent = 'Projekt erstellen';
    document.getElementById('pj-id').value = '';
  }
  openModal('project-modal');
}

async function saveProject() {
  const id = document.getElementById('pj-id').value;
  const payload = {
    name: document.getElementById('pj-name').value,
    description: document.getElementById('pj-desc').value || null,
    start_date: document.getElementById('pj-start').value || null,
    end_date: document.getElementById('pj-end').value || null,
    status: document.getElementById('proj-status-select').value,
  };
  try {
    if (id) { await patch(`/projects/${id}`, payload); toast('Projekt aktualisiert'); }
    else     { await post('/projects/', payload);       toast('Projekt erstellt'); }
    closeModal('project-modal');
    await pages.projects.load();
  } catch (e) { toast(e.message, 'error'); }
}

function editProject(id) { openProjectModal(pages.projects.find(id)); }
async function deleteProject(id, name) {
  confirmDelete(name, async () => {
    try { await del(`/projects/${id}`); toast('Projekt gelöscht', 'info'); await pages.projects.load(); }
    catch (e) { toast(e.message, 'error'); }
  });
}

function viewProjectAllocations(id, name) {
  showPage('allocations');
  document.getElementById('alloc-project-filter').value = id;
  pages.allocations.load(id);
}

// ─── Page: Allocations ────────────────────────────────────────────────────────
pages.allocations = {
  data: [],
  async load(projectFilter = null) {
    const [allocs, projects, persons, materials, timeRes] = await Promise.all([
      get('/allocations/?limit=1000' + (projectFilter ? `&project_id=${projectFilter}` : '')),
      get('/projects/?limit=1000'),
      get('/persons/?limit=1000'),
      get('/materials/?limit=1000'),
      get('/time-resources/?limit=1000'),
    ]);
    this.data = allocs;
    this._projects = projects;
    this._persons  = persons;
    this._materials = materials;
    this._time = timeRes;

    // Populate project filter
    const sel = document.getElementById('alloc-project-filter');
    const curVal = sel.value;
    sel.innerHTML = '<option value="">Alle Projekte</option>' +
      projects.map(p => `<option value="${p.id}" ${String(p.id) === String(projectFilter || curVal) ? 'selected' : ''}>${p.name}</option>`).join('');

    this.render();
  },
  render() {
    const proj = id => this._projects?.find(p => p.id === id)?.name || id;
    const pers = id => this._persons?.find(p => p.id === id)?.name || '—';
    const mat  = id => this._materials?.find(m => m.id === id)?.name || '—';
    const tr   = id => { const t = this._time?.find(t => t.id === id); return t ? `${t.amount} ${t.unit}` : '—'; };

    const tbody = document.getElementById('alloc-tbody');
    if (!this.data.length) {
      tbody.innerHTML = '<tr><td colspan="7"><div class="empty"><div class="empty-icon">🔗</div><p>Keine Zuteilungen gefunden</p></div></td></tr>';
      return;
    }
    tbody.innerHTML = this.data.map(a => `
      <tr>
        <td><strong>${proj(a.project_id)}</strong></td>
        <td>${a.person_id ? `👤 ${pers(a.person_id)}` : a.material_id ? `📦 ${mat(a.material_id)}` : a.time_resource_id ? `⏱ ${tr(a.time_resource_id)}` : '—'}</td>
        <td>${a.quantity}</td>
        <td>${statusBadge(a.status)}</td>
        <td>${fmtDate(a.start_date)}</td>
        <td>${fmtDate(a.end_date)}</td>
        <td style="white-space:nowrap">
          <button class="btn-icon" onclick="editAllocation(${a.id})">✏️</button>
          <button class="btn-icon danger" onclick="deleteAllocation(${a.id})">🗑️</button>
        </td>
      </tr>
    `).join('');
  },
  find(id) { return this.data.find(a => a.id === id); }
};

async function openAllocModal(alloc = null) {
  const enums = await getEnums();
  const [projects, persons, materials, timeRes] = await Promise.all([
    get('/projects/?limit=1000'),
    get('/persons/?limit=1000'),
    get('/materials/?limit=1000'),
    get('/time-resources/?limit=1000'),
  ]);

  document.getElementById('alloc-proj-select').innerHTML = projects.map(p =>
    `<option value="${p.id}" ${alloc?.project_id === p.id ? 'selected' : ''}>${p.name}</option>`).join('');
  document.getElementById('alloc-person-select').innerHTML = '<option value="">— keine Person —</option>' +
    persons.map(p => `<option value="${p.id}" ${alloc?.person_id === p.id ? 'selected' : ''}>${p.name} (${p.person_type})</option>`).join('');
  document.getElementById('alloc-mat-select').innerHTML = '<option value="">— kein Material —</option>' +
    materials.map(m => `<option value="${m.id}" ${alloc?.material_id === m.id ? 'selected' : ''}>${m.name} (${m.material_type})</option>`).join('');
  document.getElementById('alloc-time-select').innerHTML = '<option value="">— keine Zeit —</option>' +
    timeRes.map(t => `<option value="${t.id}" ${alloc?.time_resource_id === t.id ? 'selected' : ''}>${t.name} (${t.amount} ${t.unit})</option>`).join('');
  document.getElementById('alloc-status-select').innerHTML = buildOptions(enums.allocation_statuses, alloc?.status || 'geplant');

  if (alloc) {
    document.getElementById('alloc-modal-title').textContent = 'Zuteilung bearbeiten';
    document.getElementById('al-id').value = alloc.id;
    document.getElementById('al-qty').value = alloc.quantity;
    document.getElementById('al-start').value = alloc.start_date ? alloc.start_date.slice(0, 16) : '';
    document.getElementById('al-end').value = alloc.end_date ? alloc.end_date.slice(0, 16) : '';
    document.getElementById('al-notes').value = alloc.notes || '';
  } else {
    document.getElementById('alloc-modal-title').textContent = 'Ressource zuteilen';
    document.getElementById('al-id').value = '';
    document.getElementById('al-qty').value = '1';
  }
  openModal('alloc-modal');
}

async function saveAllocation() {
  const id = document.getElementById('al-id').value;
  const personId  = document.getElementById('alloc-person-select').value;
  const matId     = document.getElementById('alloc-mat-select').value;
  const timeId    = document.getElementById('alloc-time-select').value;

  if (!id) {
    const payload = {
      project_id: parseInt(document.getElementById('alloc-proj-select').value),
      person_id: personId ? parseInt(personId) : null,
      material_id: matId ? parseInt(matId) : null,
      time_resource_id: timeId ? parseInt(timeId) : null,
      quantity: parseFloat(document.getElementById('al-qty').value) || 1,
      start_date: document.getElementById('al-start').value || null,
      end_date: document.getElementById('al-end').value || null,
      status: document.getElementById('alloc-status-select').value,
      notes: document.getElementById('al-notes').value || null,
    };
    try { await post('/allocations/', payload); toast('Zuteilung erstellt'); }
    catch (e) { toast(e.message, 'error'); return; }
  } else {
    const payload = {
      quantity: parseFloat(document.getElementById('al-qty').value) || 1,
      start_date: document.getElementById('al-start').value || null,
      end_date: document.getElementById('al-end').value || null,
      status: document.getElementById('alloc-status-select').value,
      notes: document.getElementById('al-notes').value || null,
    };
    try { await patch(`/allocations/${id}`, payload); toast('Zuteilung aktualisiert'); }
    catch (e) { toast(e.message, 'error'); return; }
  }
  closeModal('alloc-modal');
  const pf = document.getElementById('alloc-project-filter').value;
  await pages.allocations.load(pf || null);
}

function editAllocation(id) { openAllocModal(pages.allocations.find(id)); }
async function deleteAllocation(id) {
  confirmDelete('Zuteilung', async () => {
    try {
      await del(`/allocations/${id}`);
      toast('Zuteilung gelöscht', 'info');
      const pf = document.getElementById('alloc-project-filter').value;
      await pages.allocations.load(pf || null);
    } catch (e) { toast(e.message, 'error'); }
  });
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('nav a[data-page]').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      showPage(a.dataset.page);
    });
  });

  document.getElementById('alloc-project-filter').addEventListener('change', function() {
    pages.allocations.load(this.value || null);
  });

  showPage('dashboard');
});

// ─── Page: Graph (Subgraph / Bipartiter Graph) ───────────────────────────────
pages.graph = {
  async load() {
    const projects = await get('/projects/?limit=1000');
    const opts = projects.map(p => `<option value="${p.id}">${p.name}</option>`).join('');
    document.getElementById('graph-proj-a').innerHTML = opts;
    document.getElementById('graph-proj-b').innerHTML = opts;
    // Ergebnis aus vorheriger Ansicht zurücksetzen
    document.getElementById('graph-result').style.display = 'none';
    document.getElementById('graph-viz-card').style.display = 'none';
  }
};

function _selectedIds(selectId) {
  return Array.from(document.getElementById(selectId).selectedOptions)
    .map(o => parseInt(o.value));
}

function _decisionBanner(data) {
  const cfg = {
    keep_B:       { cls: 'badge-green',  icon: '✅', text: 'Graph B ist Obermenge – Vorschlag übernehmen' },
    keep_A:       { cls: 'badge-yellow', icon: '⚠️', text: 'Graph A enthält B bereits – Vorschlag redundant' },
    keep_both:    { cls: 'badge-blue',   icon: 'ℹ️', text: 'Keine Teilmengen-Relation – beide Graphen behalten' },
    equal_keep_A: { cls: 'badge-gray',   icon: '🔁', text: 'Strukturell gleich – bestehender Graph bevorzugt' },
    equal_keep_B: { cls: 'badge-gray',   icon: '🔁', text: 'Strukturell gleich – Vorschlag hat mehr Kanten' },
  };
  const c = cfg[data.decision] || { cls: 'badge-gray', icon: '❓', text: data.decision };
  return `
    <div style="padding:.85rem;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius)">
      <div style="font-weight:700;margin-bottom:.4rem">
        ${c.icon}&nbsp;<span class="badge ${c.cls}">${data.decision}</span>&nbsp;${c.text}
      </div>
      <p style="color:var(--text-muted);font-size:.88rem;margin-bottom:.5rem">${data.message}</p>
      <div style="display:flex;gap:1.5rem;font-size:.85rem">
        <span>📊 Graph A: <strong>${data.existing_edges}</strong> Kanten</span>
        <span>📊 Graph B: <strong>${data.proposed_edges}</strong> Kanten</span>
      </div>
    </div>`;
}

function _bipartiteTable(graph, title) {
  if (!graph.n_projects || !graph.n_resources) {
    return `<div class="empty"><div class="empty-icon">🕸️</div><p>Leerer Graph</p></div>`;
  }
  const icon = k => k === 'person' ? '👤' : k === 'material' ? '📦' : '⏱️';
  const headers = graph.resources.map(r =>
    `<th style="writing-mode:vertical-lr;padding:.2rem .1rem;font-size:.7rem;white-space:nowrap">
       ${icon(r.kind)} ${r.label}
     </th>`).join('');
  const rows = graph.projects.map((proj, pi) => {
    const cells = graph.resources.map((_, ri) =>
      `<td style="text-align:center;padding:.25rem">
         ${graph.matrix[pi][ri] ? '🔵' : '<span style="opacity:.2">·</span>'}
       </td>`).join('');
    return `<tr><td style="font-weight:600;font-size:.83rem;white-space:nowrap;padding:.25rem .5rem">${proj.name}</td>${cells}</tr>`;
  }).join('');
  return `
    <p style="font-size:.78rem;font-weight:700;color:var(--text-muted);margin-bottom:.4rem">${title}</p>
    <table style="border-collapse:collapse">
      <thead><tr><th></th>${headers}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="font-size:.74rem;color:var(--text-muted);margin-top:.35rem">
      🔵 Zuteilung &nbsp;|&nbsp; ${graph.n_projects} Projekte × ${graph.n_resources} Ressourcen &nbsp;|&nbsp; ${graph.n_edges} Kanten
    </p>`;
}

async function runGraphCompare() {
  const aIds = _selectedIds('graph-proj-a');
  const bIds = _selectedIds('graph-proj-b');
  if (!aIds.length || !bIds.length) {
    toast('Bitte in beiden Listen mindestens ein Projekt auswählen.', 'error');
    return;
  }
  try {
    const data = await post('/subgraph/compare', {
      existing_project_ids: aIds,
      proposed_project_ids: bIds,
    });
    const resultDiv = document.getElementById('graph-result');
    resultDiv.innerHTML = _decisionBanner(data);
    resultDiv.style.display = 'block';

    document.getElementById('graph-viz-wrap').innerHTML =
      '<div style="display:flex;gap:2rem;flex-wrap:wrap">' +
        `<div>${_bipartiteTable(data.existing_graph, 'Graph A – Bestehend')}</div>` +
        `<div>${_bipartiteTable(data.proposed_graph, 'Graph B – Vorschlag')}</div>` +
      '</div>';
    document.getElementById('graph-viz-title').textContent =
      `Bipartite Matrizen: A (${data.existing_graph.n_edges} Kanten) vs. B (${data.proposed_graph.n_edges} Kanten)`;
    document.getElementById('graph-viz-card').style.display = 'block';

    toast(`Entscheidung: ${data.decision}`, data.is_superset ? 'success' : 'info');
  } catch (e) { toast(e.message, 'error'); }
}

async function loadRedundancy() {
  const projects = await get('/projects/?limit=1000');
  if (projects.length < 2) { toast('Mindestens 2 Projekte erforderlich.', 'error'); return; }
  try {
    const data = await post('/subgraph/redundancy', projects.map(p => [p.id]));
    const resultDiv = document.getElementById('graph-result');
    let html = `
      <div style="padding:.85rem;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius)">
        <div style="font-weight:700;margin-bottom:.5rem">🔍 ${data.summary}</div>`;
    if (!data.redundant_pairs.length) {
      html += '<p style="color:var(--text-muted);font-size:.88rem">Keine redundanten Paare gefunden.</p>';
    } else {
      data.redundant_pairs.forEach(pair => {
        const nameA = projects[pair.graph_a_index]?.name || `Gruppe ${pair.graph_a_index}`;
        const nameB = projects[pair.graph_b_index]?.name || `Gruppe ${pair.graph_b_index}`;
        html += `<div style="font-size:.87rem;margin:.3rem 0">
          <span class="badge badge-yellow">${pair.decision}</span>
          &nbsp;<strong>${nameA}</strong> ↔ <strong>${nameB}</strong> – ${pair.message}
        </div>`;
      });
    }
    if (data.optimal_graph_index !== null) {
      html += `<div style="margin-top:.5rem;font-weight:700;color:var(--success)">
        ✅ Optimaler Graph: ${projects[data.optimal_graph_index]?.name}
      </div>`;
    }
    html += '</div>';
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
    document.getElementById('graph-viz-card').style.display = 'none';
    toast('Redundanzcheck abgeschlossen.', 'info');
  } catch (e) { toast(e.message, 'error'); }
}
