'use strict';
const fs = require('fs');

function main() {
  const inPath = process.argv[2];
  const outPath = process.argv[3];
  if (!inPath || !outPath) {
    console.error('usage: node ua-tour-analyze.js <input.json> <output.json>');
    process.exit(1);
  }
  const data = JSON.parse(fs.readFileSync(inPath, 'utf8'));
  const nodes = data.nodes || [];
  const edges = data.edges || [];
  const layers = data.layers || [];

  const byId = new Map();
  nodes.forEach(n => byId.set(n.id, n));

  // adjacency
  const fanIn = new Map();
  const fanOut = new Map();
  const outAdj = new Map(); // forward edges for BFS (imports/calls)
  nodes.forEach(n => { fanIn.set(n.id, 0); fanOut.set(n.id, 0); outAdj.set(n.id, []); });

  edges.forEach(e => {
    if (!byId.has(e.source) || !byId.has(e.target)) return;
    fanOut.set(e.source, fanOut.get(e.source) + 1);
    fanIn.set(e.target, fanIn.get(e.target) + 1);
    if (e.type === 'imports' || e.type === 'calls') {
      outAdj.get(e.source).push(e.target);
    }
  });

  const name = id => (byId.get(id) || {}).name || id;
  const summ = id => (byId.get(id) || {}).summary || '';

  // A. fan-in ranking
  const fanInRanking = nodes.map(n => ({ id: n.id, fanIn: fanIn.get(n.id), name: n.name }))
    .sort((a, b) => b.fanIn - a.fanIn).slice(0, 20);

  // B. fan-out ranking
  const fanOutRanking = nodes.map(n => ({ id: n.id, fanOut: fanOut.get(n.id), name: n.name }))
    .sort((a, b) => b.fanOut - a.fanOut).slice(0, 20);

  // thresholds for entry-point heuristics
  const fanOutVals = nodes.map(n => fanOut.get(n.id)).sort((a, b) => a - b);
  const fanInVals = nodes.map(n => fanIn.get(n.id)).sort((a, b) => a - b);
  const pct = (arr, p) => arr.length ? arr[Math.min(arr.length - 1, Math.floor(arr.length * p))] : 0;
  const fanOutTop10 = pct(fanOutVals, 0.9);
  const fanInBottom25 = pct(fanInVals, 0.25);

  const entryNames = new Set(['index.ts','index.js','main.ts','main.js','app.ts','app.js','server.ts','server.js','mod.rs','main.go','main.py','main.rs','manage.py','app.py','wsgi.py','asgi.py','run.py','__main__.py','Application.java','Main.java','Program.cs','config.ru','index.php','App.swift','Application.kt','main.cpp','main.c','platform.py']);

  // C. entry point candidates
  const entryScored = nodes.map(n => {
    let score = 0;
    const fp = n.filePath || '';
    const depth = fp.split('/').length;
    if (n.type === 'document') {
      if (/(^|\/)README\.md$/i.test(fp) && depth === 1) score += 5;
      else if (/\.md$/i.test(fp) && depth === 1) score += 2;
    } else {
      if (entryNames.has(n.name)) score += 3;
      if (depth <= 2) score += 1;
      if (fanOut.get(n.id) >= fanOutTop10 && fanOutTop10 > 0) score += 1;
      if (fanIn.get(n.id) <= fanInBottom25) score += 1;
    }
    return { id: n.id, score, name: n.name, summary: n.summary };
  }).filter(e => e.score > 0).sort((a, b) => b.score - a.score).slice(0, 5);

  // D. BFS from top code entry point
  const codeEntries = entryScored.filter(e => (byId.get(e.id) || {}).type !== 'document');
  // prefer platform.py explicitly if present
  let startNode = null;
  const platform = nodes.find(n => n.id === 'file:platform.py');
  if (platform) startNode = platform.id;
  else if (codeEntries.length) startNode = codeEntries[0].id;
  else if (entryScored.length) startNode = entryScored[0].id;

  const order = [];
  const depthMap = {};
  if (startNode) {
    const q = [startNode];
    depthMap[startNode] = 0;
    while (q.length) {
      const cur = q.shift();
      order.push(cur);
      (outAdj.get(cur) || []).forEach(t => {
        if (!(t in depthMap)) { depthMap[t] = depthMap[cur] + 1; q.push(t); }
      });
    }
  }
  const byDepth = {};
  Object.keys(depthMap).forEach(id => {
    const d = depthMap[id];
    (byDepth[d] = byDepth[d] || []).push(id);
  });

  // E. non-code inventory
  const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
  nodes.forEach(n => {
    const rec = { id: n.id, name: n.name, type: n.type, summary: n.summary };
    if (n.type === 'document') nonCodeFiles.documentation.push(rec);
    else if (['service','pipeline','resource'].includes(n.type)) nonCodeFiles.infrastructure.push(rec);
    else if (['table','schema','endpoint'].includes(n.type)) nonCodeFiles.data.push(rec);
    else if (n.type === 'config') nonCodeFiles.config.push(rec);
  });

  // F. clusters via bidirectional edges
  const pairKey = (a, b) => a < b ? a + '|' + b : b + '|' + a;
  const directed = new Set();
  edges.forEach(e => {
    if (e.type === 'imports' || e.type === 'calls' || e.type === 'depends_on') {
      directed.add(e.source + '>>' + e.target);
    }
  });
  const biPairs = [];
  const seen = new Set();
  directed.forEach(d => {
    const [s, t] = d.split('>>');
    if (directed.has(t + '>>' + s)) {
      const k = pairKey(s, t);
      if (!seen.has(k)) { seen.add(k); biPairs.push([s, t]); }
    }
  });
  // build clusters by merging pairs sharing nodes
  const clusters = [];
  biPairs.forEach(([a, b]) => {
    let found = null;
    for (const c of clusters) { if (c.has(a) || c.has(b)) { found = c; break; } }
    if (found) { found.add(a); found.add(b); }
    else clusters.push(new Set([a, b]));
  });
  // count edges within cluster
  const edgeCountIn = setNodes => {
    let cnt = 0;
    edges.forEach(e => { if (setNodes.has(e.source) && setNodes.has(e.target)) cnt++; });
    return cnt;
  };
  const clustersOut = clusters
    .filter(c => c.size >= 2 && c.size <= 5)
    .map(c => ({ nodes: [...c], edgeCount: edgeCountIn(c) }))
    .sort((a, b) => b.edgeCount - a.edgeCount)
    .slice(0, 10);

  // G. layers
  const layersOut = { count: layers.length, list: layers.map(l => ({ id: l.id, name: l.name, description: l.description })) };

  // H. node summary index
  const nodeSummaryIndex = {};
  nodes.forEach(n => { nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary }; });

  const out = {
    scriptCompleted: true,
    entryPointCandidates: entryScored,
    fanInRanking,
    fanOutRanking,
    bfsTraversal: { startNode, order, depthMap, byDepth },
    nonCodeFiles,
    clusters: clustersOut,
    layers: layersOut,
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length
  };
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log('done. nodes', nodes.length, 'edges', edges.length, 'bfs reached', order.length, 'clusters', clustersOut.length);
}

try { main(); } catch (e) { console.error(e.stack || String(e)); process.exit(1); }
