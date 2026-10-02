/* Every plotted value comes from docs/data/cells.json, exported from audited CSV files. */
const COLORS = {batch1: '#4f73ae', batch2: '#dd8955', batch3: '#8b78ad'};
const LABELS = {batch1: 'Batch 1', batch2: 'Batch 2', batch3: 'Batch 3'};
const SIGNALS = {
  delta_q_logvar: 'ΔQ(V) 로그 분산',
  early_mean_QD: '평균 방전 용량 QD (Ah)',
  early_mean_IR: '평균 내부저항 IR (Ω)',
  first_c_rate: '첫 구간 C-rate'
};
const PLOT_CONFIG = {responsive: true, displaylogo: false, locale: 'ko',
  modeBarButtonsToRemove: ['lasso2d', 'select2d']};
let cells = [];

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function lifeBand(value) {
  if (value < 500) return 'short';
  if (value > 1000) return 'long';
  return 'middle';
}

function bandLabel(value) {
  return {short: '<500', middle: '500–1,000', long: '>1,000'}[lifeBand(value)];
}

function selectedRows() {
  const batches = new Set(Array.from(document.querySelectorAll('input[name="batch"]:checked'),
    input => input.value));
  const band = document.getElementById('life-band').value;
  return cells.filter(row => batches.has(row.batch) && (band === 'all' || lifeBand(row.actual) === band));
}

function number(value, digits = 0) {
  return value.toLocaleString('ko-KR', {minimumFractionDigits: digits, maximumFractionDigits: digits});
}

function commonLayout(yTitle, xTitle) {
  return {
    margin: {l: 61, r: 12, t: 13, b: 61},
    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
    font: {family: 'Arial, sans-serif', size: 11, color: '#566981'},
    xaxis: {title: {text: xTitle, font: {size: 11}}, gridcolor: '#ebf0f4', zerolinecolor: '#d5e0e9'},
    yaxis: {title: {text: yTitle, font: {size: 11}}, gridcolor: '#ebf0f4', zerolinecolor: '#d5e0e9'},
    legend: {orientation: 'h', x: 0, y: 1.13, font: {size: 11}},
    hoverlabel: {bgcolor: '#223852', font: {color: '#ffffff', size: 12}}
  };
}

function emptyPlot(id, message) {
  const element = document.getElementById(id);
  if (element.data) Plotly.purge(element);
  element.classList.add('empty');
  element.textContent = message;
}

function plot(id, traces, layout) {
  const element = document.getElementById(id);
  element.classList.remove('empty');
  return Plotly.react(element, traces, layout, PLOT_CONFIG);
}

function renderKpis(rows) {
  const evaluated = rows.filter(row => row.predicted !== null);
  const short = rows.filter(row => row.actual < 500).length;
  const mape = evaluated.length ? evaluated.reduce((sum, row) => sum + row.ape_pct, 0) / evaluated.length : null;
  document.getElementById('kpi-cells').textContent = number(rows.length);
  document.getElementById('kpi-life').textContent = rows.length ? number(median(rows.map(row => row.actual))) : '—';
  document.getElementById('kpi-short').textContent = rows.length ? number(100 * short / rows.length, 1) + '%' : '—';
  document.getElementById('kpi-mape').textContent = mape === null ? '—' : number(mape, 2) + '%';
  const chosen = Array.from(document.querySelectorAll('input[name="batch"]:checked'), input => LABELS[input.value]);
  document.getElementById('filter-summary').textContent =
    (chosen.length ? chosen.join(' · ') : '배치 미선택') + ' / ' + number(rows.length) + '셀 표시 / '
    + number(evaluated.length) + '셀 예측값 포함 · 수명 구간 필터는 실제 수명 기준';
}

function renderLife(rows) {
  if (!rows.length) return emptyPlot('life-chart', '선택한 조건에 해당하는 셀이 없습니다.');
  const traces = Object.keys(LABELS).map(batch => {
    const subset = rows.filter(row => row.batch === batch);
    return {type: 'histogram', name: LABELS[batch], x: subset.map(row => row.actual),
      xbins: {start: 0, end: 2400, size: 100}, marker: {color: COLORS[batch]}, opacity: 0.72,
      hovertemplate: LABELS[batch] + '<br>%{x}사이클 구간<br>%{y}셀<extra></extra>'};
  }).filter(trace => trace.x.length);
  const layout = commonLayout('셀 수', '실제 수명 (사이클)');
  layout.barmode = 'overlay';
  layout.xaxis.range = [0, 2400];
  plot('life-chart', traces, layout);
}

function renderSignal(rows) {
  const signal = document.getElementById('signal').value;
  const valid = rows.filter(row => Number.isFinite(row[signal]));
  if (!valid.length) return emptyPlot('signal-chart', '이 신호가 기록된 셀이 없습니다.');
  const traces = Object.keys(LABELS).map(batch => {
    const subset = valid.filter(row => row.batch === batch);
    return {type: 'scatter', mode: 'markers', name: LABELS[batch],
      x: subset.map(row => row[signal]), y: subset.map(row => row.actual),
      customdata: subset.map(row => [row.cell_id, row.policy]),
      marker: {size: 9, color: COLORS[batch], opacity: .79,
        line: {color: '#ffffff', width: 1}},
      hovertemplate: '<b>%{customdata[0]}</b><br>정책: %{customdata[1]}<br>'
        + SIGNALS[signal] + ': %{x:.3f}<br>실제 수명: %{y:.0f}사이클<extra></extra>'};
  }).filter(trace => trace.x.length);
  plot('signal-chart', traces, commonLayout('실제 수명 (사이클)', SIGNALS[signal]));
}

function renderPrediction(rows) {
  const evaluated = rows.filter(row => row.predicted !== null);
  if (!evaluated.length) return emptyPlot('prediction-chart', '선택한 범위에 Batch 2·3 평가 셀이 없습니다.');
  const high = Math.max(...evaluated.flatMap(row => [row.actual, row.predicted]));
  const bound = Math.ceil(high / 100) * 100 + 50;
  const traces = [{type: 'scatter', mode: 'lines', x: [0, bound], y: [0, bound],
    name: '오차 0', line: {color: '#96a7b8', dash: 'dash', width: 1.5}, hoverinfo: 'skip'}];
  for (const batch of ['batch2', 'batch3']) {
    const subset = evaluated.filter(row => row.batch === batch);
    if (!subset.length) continue;
    traces.push({type: 'scatter', mode: 'markers', name: LABELS[batch],
      x: subset.map(row => row.actual), y: subset.map(row => row.predicted),
      customdata: subset.map(row => [row.cell_id, row.ape_pct]),
      marker: {size: 9, color: COLORS[batch], opacity: .8,
        line: {color: '#ffffff', width: 1}},
      hovertemplate: '<b>%{customdata[0]}</b><br>실제: %{x:.0f}사이클<br>'
        + '예측: %{y:.0f}사이클<br>절대비율오차: %{customdata[1]:.1f}%<extra></extra>'});
  }
  const layout = commonLayout('예측 수명 (사이클)', '실제 수명 (사이클)');
  layout.xaxis.range = [0, bound];
  layout.yaxis.range = [0, bound];
  plot('prediction-chart', traces, layout);
}

function renderError(rows) {
  const evaluated = rows.filter(row => row.predicted !== null);
  if (!evaluated.length) return emptyPlot('error-chart', '선택한 범위에 Batch 2·3 평가 셀이 없습니다.');
  const traces = ['batch2', 'batch3'].map(batch => {
    const subset = evaluated.filter(row => row.batch === batch);
    return {type: 'box', name: LABELS[batch],
      x: subset.map(row => bandLabel(row.actual)),
      y: subset.map(row => row.predicted - row.actual),
      customdata: subset.map(row => row.cell_id),
      marker: {color: COLORS[batch], size: 5, opacity: .68},
      line: {color: COLORS[batch]}, boxpoints: 'all', jitter: .32, pointpos: 0,
      hovertemplate: '<b>%{customdata}</b><br>실제 수명 구간: %{x}<br>'
        + '예측 − 실제: %{y:.0f}사이클<extra></extra>'};
  }).filter(trace => trace.x.length);
  const layout = commonLayout('예측 − 실제 (사이클)', '실제 수명 구간');
  layout.boxmode = 'group';
  layout.xaxis.categoryorder = 'array';
  layout.xaxis.categoryarray = ['<500', '500–1,000', '>1,000'];
  layout.shapes = [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0,
    line: {color: '#8496a9', dash: 'dash', width: 1.4}}];
  plot('error-chart', traces, layout);
}

function renderTable(rows) {
  const body = document.getElementById('error-rows');
  body.replaceChildren();
  const top = rows.filter(row => row.predicted !== null)
    .sort((a, b) => Math.abs(b.predicted - b.actual) - Math.abs(a.predicted - a.actual))
    .slice(0, 8);
  if (!top.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 6;
    td.textContent = '선택한 조건에 평가 셀이 없습니다.';
    tr.appendChild(td);
    body.appendChild(tr);
    return;
  }
  for (const row of top) {
    const tr = document.createElement('tr');
    const values = [row.cell_id, LABELS[row.batch], number(row.actual), number(row.predicted),
      (row.predicted >= row.actual ? '+' : '') + number(row.predicted - row.actual),
      number(row.ape_pct, 1) + '%'];
    for (const [index, value] of values.entries()) {
      const td = document.createElement('td');
      td.textContent = value;
      if (index === 4) td.className = row.predicted >= row.actual ? 'positive' : 'negative';
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

function render() {
  const rows = selectedRows();
  renderKpis(rows);
  renderLife(rows);
  renderSignal(rows);
  renderPrediction(rows);
  renderError(rows);
  renderTable(rows);
}

async function start() {
  try {
    if (!window.Plotly) throw new Error('그래프 라이브러리를 불러오지 못했습니다. 네트워크 연결을 확인해 주세요.');
    const response = await fetch('./data/cells.json');
    if (!response.ok) throw new Error('셀 데이터 파일을 불러오지 못했습니다: HTTP ' + response.status);
    const payload = await response.json();
    cells = payload.cells;
    if (cells.length !== 124 || payload.counts.batch1 !== 41 || payload.counts.batch2 !== 43
        || payload.counts.batch3 !== 40) throw new Error('셀 수가 검증된 원자료와 다릅니다.');
    document.querySelectorAll('input[name="batch"], #life-band, #signal')
      .forEach(element => element.addEventListener('change', render));
    document.getElementById('reset-filters').addEventListener('click', () => {
      document.querySelectorAll('input[name="batch"]').forEach(input => { input.checked = true; });
      document.getElementById('life-band').value = 'all';
      document.getElementById('signal').value = 'delta_q_logvar';
      render();
    });
    render();
  } catch (error) {
    const message = document.getElementById('load-error');
    message.hidden = false;
    message.textContent = error.message;
    document.getElementById('filter-summary').textContent = '대시보드를 열 수 없습니다.';
  }
}

window.addEventListener('DOMContentLoaded', start);
