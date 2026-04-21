<template>
  <div class="page">
    <header class="topbar">
      <div>
        <h1>Heat Equation Demo</h1>
        <p>PINN 在线计算 · 时序热力图可视化</p>
      </div>
      <div class="status-chip" :class="jobStatusClass">{{ jobStatusLabel }}</div>
    </header>

    <section class="panel controls">
      <div class="controls-grid">
        <label>ν
          <input v-model.number="form.nu" type="number" step="0.01" />
        </label>
        <label>Δt
          <input v-model.number="form.dt" type="number" step="0.00001" />
        </label>
        <label>nx
          <input v-model.number="form.nx" type="number" step="1" />
        </label>
        <label>ny
          <input v-model.number="form.ny" type="number" step="1" />
        </label>
        <label>short steps
          <input v-model.number="form.short_steps" type="number" step="1" />
        </label>
        <label>long steps
          <input v-model.number="form.long_steps" type="number" step="1" />
        </label>
        <label>noise
          <input v-model.number="form.noise_level" type="number" step="0.001" />
        </label>
        <label>seed
          <input v-model.number="form.seed" type="number" step="1" />
        </label>
      </div>

      <div class="actions-row">
        <button class="btn btn-primary" @click="submitJob" :disabled="submitting">
          {{ submitting ? '提交中…' : '开始在线计算' }}
        </button>
        <button class="btn" @click="reloadAll">刷新结果</button>
        <span v-if="jobId" class="muted">job: {{ jobId }}</span>
        <span class="muted">run: {{ run }}</span>
      </div>
    </section>

    <section class="panel timeline">
      <div class="timeline-row">
        <button class="btn" @click="togglePlay">{{ playing ? '暂停' : '播放' }}</button>
        <input type="range" min="0" :max="maxT" v-model.number="t" @input="reloadFields" />
        <span class="time-label">t = {{ t }} / {{ maxT }}</span>
      </div>
    </section>

    <section class="plots-grid">
      <article class="panel plot-card">
        <h3>PINN Prediction</h3>
        <div ref="leftRef" class="chart heat"></div>
      </article>
      <article class="panel plot-card">
        <h3>Ground Truth</h3>
        <div ref="rightRef" class="chart heat"></div>
      </article>
    </section>

    <section class="plots-grid metrics-grid">
      <article class="panel plot-card">
        <h3>MSE</h3>
        <div ref="mseRef" class="chart metric"></div>
      </article>
      <article class="panel plot-card">
        <h3>RMSE</h3>
        <div ref="rmseRef" class="chart metric"></div>
      </article>
    </section>
  </div>
</template>

<script setup>
import * as echarts from 'echarts'
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const api = 'http://127.0.0.1:8000/api'
const form = ref({ model: 'pinn', equation: 'heat', nx: 101, ny: 101, nu: 1.0, dt: 1e-5, short_steps: 60, long_steps: 120, noise_level: 0.002, seed: 42 })

const run = ref('epoch_000000')
const t = ref(0)
const maxT = ref(60)
const playing = ref(false)
const submitting = ref(false)
const jobId = ref('')
const jobStatus = ref('')
let timer = null
let pollTimer = null

const leftRef = ref(null)
const rightRef = ref(null)
const mseRef = ref(null)
const rmseRef = ref(null)
let leftChart, rightChart, mseChart, rmseChart

const jobStatusLabel = computed(() => jobStatus.value || '未提交')
const jobStatusClass = computed(() => {
  if (jobStatus.value === 'success') return 'ok'
  if (jobStatus.value === 'failed') return 'bad'
  if (jobStatus.value === 'running') return 'running'
  return 'idle'
})

async function getJson(url) {
  const r = await fetch(url)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

async function postJson(url, payload) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

function renderHeatmap(chart, title, matrix) {
  if (!chart || !matrix?.length) return
  const h = matrix.length
  const w = matrix[0].length
  const data = []
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) data.push([x, y, matrix[y][x]])
  }
  const minV = Math.min(...data.map((d) => d[2]))
  const maxV = Math.max(...data.map((d) => d[2]))

  chart.setOption({
    title: { text: title, left: 8, top: 6, textStyle: { fontSize: 12, fontWeight: 500, color: '#cfd8e3' } },
    tooltip: { position: 'top' },
    grid: { top: 30, right: 34, bottom: 8, left: 8, containLabel: true },
    xAxis: { type: 'category', data: Array.from({ length: w }, (_, i) => i), axisLabel: { show: false }, axisTick: { show: false } },
    yAxis: { type: 'category', data: Array.from({ length: h }, (_, i) => i), axisLabel: { show: false }, axisTick: { show: false } },
    visualMap: {
      min: minV,
      max: maxV,
      orient: 'vertical',
      right: 2,
      top: 'middle',
      itemHeight: 140,
      text: ['high', 'low'],
      textStyle: { color: '#9fb0c8' },
      calculable: false,
      inRange: { color: ['#14213d', '#1f5aa6', '#2e8bc0', '#62b6cb', '#f4d35e', '#ee964b', '#f95738'] }
    },
    series: [{ type: 'heatmap', data, progressive: 0, animation: false }]
  })
}

function renderLine(chart, title, yName, arr = []) {
  chart?.setOption({
    title: { text: title, left: 8, top: 6, textStyle: { fontSize: 12, fontWeight: 500, color: '#cfd8e3' } },
    tooltip: { trigger: 'axis' },
    grid: { top: 30, right: 12, bottom: 24, left: 44 },
    xAxis: { type: 'category', data: arr.map((_, i) => i), name: 't', nameTextStyle: { color: '#8fa0b7' }, axisLabel: { color: '#8fa0b7' } },
    yAxis: { type: 'value', name: yName, nameTextStyle: { color: '#8fa0b7' }, axisLabel: { color: '#8fa0b7' } },
    series: [{ name: title, type: 'line', data: arr, smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#59c3c3' } }]
  })
}

async function reloadFields() {
  const [pred, gt] = await Promise.all([
    getJson(`${api}/models/pinn/${run.value}/field?kind=prediction_short&t=${t.value}`),
    getJson(`${api}/models/pinn/${run.value}/field?kind=gt_short&t=${t.value}`)
  ])
  renderHeatmap(leftChart, 'Prediction', pred.field)
  renderHeatmap(rightChart, 'Ground Truth', gt.field)
}

async function reloadMetrics() {
  const m = await getJson(`${api}/models/pinn/${run.value}/metrics`)
  renderLine(mseChart, 'MSE', 'MSE', m.short.frame_mse)
  renderLine(rmseChart, 'RMSE', 'RMSE', m.short.frame_rmse)
  maxT.value = Math.max(0, (m.short.frame_mse?.length || 1) - 1)
  if (t.value > maxT.value) t.value = maxT.value
}

async function reloadAll() {
  await Promise.all([reloadFields(), reloadMetrics()])
}

async function submitJob() {
  submitting.value = true
  try {
    const data = await postJson(`${api}/jobs`, form.value)
    jobId.value = data.job_id
    jobStatus.value = data.status
    startPolling()
  } catch (e) {
    alert(`任务提交失败: ${e.message}`)
  } finally {
    submitting.value = false
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(async () => {
    if (!jobId.value) return
    const j = await getJson(`${api}/jobs/${jobId.value}`)
    jobStatus.value = j.status
    if (j.status === 'success') {
      run.value = j.run
      clearInterval(pollTimer)
      pollTimer = null
      await reloadAll()
    }
    if (j.status === 'failed') {
      clearInterval(pollTimer)
      pollTimer = null
      alert(`任务失败: ${j.error || 'unknown error'}`)
    }
  }, 700)
}

function togglePlay() {
  playing.value = !playing.value
  if (playing.value) {
    timer = setInterval(async () => {
      t.value = t.value >= maxT.value ? 0 : t.value + 1
      await reloadFields()
    }, 260)
  } else if (timer) {
    clearInterval(timer)
    timer = null
  }
}

onMounted(() => {
  leftChart = echarts.init(leftRef.value)
  rightChart = echarts.init(rightRef.value)
  mseChart = echarts.init(mseRef.value)
  rmseChart = echarts.init(rmseRef.value)
  window.addEventListener('resize', handleResize)
})

function handleResize() {
  leftChart?.resize()
  rightChart?.resize()
  mseChart?.resize()
  rmseChart?.resize()
}

onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
  if (pollTimer) clearInterval(pollTimer)
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.page {
  min-height: 100vh;
  padding: 22px;
  color: #d8e1ec;
  background: radial-gradient(circle at 20% 20%, #1f2b3f, #0f1724 45%, #0b1220 100%);
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}

.topbar h1 {
  margin: 0;
  font-size: 24px;
  letter-spacing: 0.2px;
}

.topbar p {
  margin: 6px 0 0;
  color: #9ab0c9;
  font-size: 13px;
}

.panel {
  background: rgba(19, 27, 40, 0.86);
  border: 1px solid rgba(143, 168, 195, 0.18);
  border-radius: 12px;
  padding: 12px;
  backdrop-filter: blur(2px);
}

.controls {
  margin-bottom: 12px;
}

.controls-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: 8px;
}

label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: #a8bdd7;
}

input[type='number'],
input[type='range'] {
  width: 100%;
}

input[type='number'] {
  background: #0d1522;
  color: #d7e3ef;
  border: 1px solid #2a3b55;
  border-radius: 8px;
  padding: 7px 8px;
}

.actions-row,
.timeline-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
}

.btn {
  border: 1px solid #2f425f;
  background: #13233a;
  color: #d6e2ee;
  border-radius: 8px;
  padding: 7px 12px;
  cursor: pointer;
}

.btn:hover {
  filter: brightness(1.08);
}

.btn-primary {
  background: linear-gradient(90deg, #2962ff, #00a6ff);
  border: none;
}

.muted {
  color: #8ba2bd;
  font-size: 12px;
}

.time-label {
  font-variant-numeric: tabular-nums;
  color: #9bb0c7;
}

.plots-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-top: 12px;
}

.metrics-grid {
  margin-top: 12px;
}

.plot-card h3 {
  margin: 0 0 8px;
  font-size: 13px;
  font-weight: 600;
  color: #c7d8eb;
}

.chart.heat {
  height: 390px;
}

.chart.metric {
  height: 240px;
}

.status-chip {
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 12px;
  border: 1px solid #2d3f59;
  background: #13253d;
}

.status-chip.ok {
  color: #8ce99a;
  border-color: #2b8a3e;
}

.status-chip.bad {
  color: #ff8787;
  border-color: #c92a2a;
}

.status-chip.running {
  color: #ffd43b;
  border-color: #f08c00;
}

.status-chip.idle {
  color: #adb5bd;
  border-color: #495057;
}

@media (max-width: 1100px) {
  .controls-grid {
    grid-template-columns: repeat(2, minmax(120px, 1fr));
  }

  .plots-grid {
    grid-template-columns: 1fr;
  }

  .chart.heat {
    height: 320px;
  }
}
</style>
