<template>
  <main style="padding: 16px; font-family: Arial, sans-serif">
    <h2>在线热方程模拟平台（PINN v1）</h2>

    <section style="border:1px solid #ddd;padding:12px;margin-bottom:12px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">
      <label>nu <input v-model.number="form.nu" type="number" step="0.01" /></label>
      <label>dt <input v-model.number="form.dt" type="number" step="0.00001" /></label>
      <label>nx <input v-model.number="form.nx" type="number" step="1" /></label>
      <label>ny <input v-model.number="form.ny" type="number" step="1" /></label>
      <label>short_steps <input v-model.number="form.short_steps" type="number" step="1" /></label>
      <label>long_steps <input v-model.number="form.long_steps" type="number" step="1" /></label>
      <label>noise_level <input v-model.number="form.noise_level" type="number" step="0.001" /></label>
      <label>seed <input v-model.number="form.seed" type="number" step="1" /></label>
      <div style="grid-column:1/5;display:flex;gap:8px;align-items:center;">
        <button @click="submitJob" :disabled="submitting">{{ submitting ? '提交中...' : '开始在线计算' }}</button>
        <span>任务状态：{{ jobStatus || '未提交' }}</span>
        <span v-if="jobId">(job_id: {{ jobId }})</span>
      </div>
    </section>

    <section style="margin-bottom: 8px; display: flex; gap: 8px; align-items: center;">
      <input type="range" min="0" :max="maxT" v-model.number="t" @input="reloadFields" style="width: 420px" />
      <span>t = {{ t }} / {{ maxT }}</span>
      <button @click="togglePlay">{{ playing ? '暂停' : '播放' }}</button>
      <button @click="reloadAll">刷新结果</button>
    </section>

    <section style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px">
      <div ref="leftRef" style="height: 340px; border: 1px solid #ddd"></div>
      <div ref="rightRef" style="height: 340px; border: 1px solid #ddd"></div>
    </section>

    <section style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
      <div ref="mseRef" style="height: 260px; border: 1px solid #ddd"></div>
      <div ref="rmseRef" style="height: 260px; border: 1px solid #ddd"></div>
    </section>

    <p style="margin-top:8px;color:#666;">FNO 对比接口已预留，待你下载并接入 FNO 模型后启用。</p>
  </main>
</template>

<script setup>
import * as echarts from 'echarts'
import { onMounted, onBeforeUnmount, ref } from 'vue'

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
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.push([x, y, matrix[y][x]])
  chart.setOption({
    title: { text: title, left: 'center', textStyle: { fontSize: 12 } },
    tooltip: {},
    xAxis: { type: 'category', data: Array.from({ length: w }, (_, i) => i) },
    yAxis: { type: 'category', data: Array.from({ length: h }, (_, i) => i) },
    visualMap: { min: Math.min(...data.map(d => d[2])), max: Math.max(...data.map(d => d[2])), calculable: true },
    series: [{ type: 'heatmap', data }]
  })
}

function renderLine(chart, title, yName, arr = []) {
  chart?.setOption({
    title: { text: title, left: 'center', textStyle: { fontSize: 12 } },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: arr.map((_, i) => i), name: 'time' },
    yAxis: { type: 'value', name: yName },
    series: [{ name: title, type: 'line', data: arr, smooth: true }]
  })
}

async function reloadFields() {
  const [pred, gt] = await Promise.all([
    getJson(`${api}/models/pinn/${run.value}/field?kind=prediction_short&t=${t.value}`),
    getJson(`${api}/models/pinn/${run.value}/field?kind=gt_short&t=${t.value}`)
  ])
  renderHeatmap(leftChart, 'PINN prediction', pred.field)
  renderHeatmap(rightChart, 'Ground truth', gt.field)
}

async function reloadMetrics() {
  const m = await getJson(`${api}/models/pinn/${run.value}/metrics`)
  renderLine(mseChart, 'MSE 曲线', 'MSE', m.short.frame_mse)
  renderLine(rmseChart, 'RMSE 曲线', 'RMSE', m.short.frame_rmse)
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
    }, 250)
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
})

onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
  if (pollTimer) clearInterval(pollTimer)
})
</script>
