<template>
  <main style="padding: 16px; font-family: Arial, sans-serif">
    <h2>流体模拟可视化平台（PINN / FNO）</h2>

    <section style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px">
      <label>模型：
        <select v-model="selectedModel" @change="loadRuns">
          <option v-for="m in models" :key="m" :value="m">{{ m }}</option>
        </select>
      </label>

      <label>轮次：
        <select v-model="selectedRun" @change="reloadAll">
          <option v-for="r in runs" :key="r" :value="r">{{ r }}</option>
        </select>
      </label>

      <label>对比模型：
        <select v-model="compareModel" @change="reloadAll">
          <option value="pinn">pinn</option>
          <option value="fno">fno</option>
        </select>
      </label>
    </section>

    <section style="margin-bottom: 8px; display: flex; gap: 8px; align-items: center;">
      <input type="range" min="0" :max="maxT" v-model.number="t" @input="reloadFields" style="width: 420px" />
      <span>t = {{ t }} / {{ maxT }}</span>
      <button @click="togglePlay">{{ playing ? '暂停' : '播放' }}</button>
    </section>

    <section style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 12px">
      <div ref="leftRef" style="height: 320px; border: 1px solid #ddd"></div>
      <div ref="rightRef" style="height: 320px; border: 1px solid #ddd"></div>
      <div ref="diffRef" style="height: 320px; border: 1px solid #ddd"></div>
    </section>

    <section style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
      <div ref="mseRef" style="height: 260px; border: 1px solid #ddd"></div>
      <div ref="rmseRef" style="height: 260px; border: 1px solid #ddd"></div>
    </section>
  </main>
</template>

<script setup>
import * as echarts from 'echarts'
import { onMounted, onBeforeUnmount, ref } from 'vue'

const api = 'http://127.0.0.1:8000/api'
const models = ref([])
const runs = ref([])
const selectedModel = ref('pinn')
const compareModel = ref('fno')
const selectedRun = ref('epoch_000000')
const t = ref(0)
const maxT = ref(100)
const playing = ref(false)
let timer = null

const leftRef = ref(null)
const rightRef = ref(null)
const diffRef = ref(null)
const mseRef = ref(null)
const rmseRef = ref(null)
let leftChart, rightChart, diffChart, mseChart, rmseChart

async function getJson(url) {
  const r = await fetch(url)
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
  chart.setOption({
    title: { text: title, left: 'center', textStyle: { fontSize: 12 } },
    tooltip: {},
    xAxis: { type: 'category', data: Array.from({ length: w }, (_, i) => i) },
    yAxis: { type: 'category', data: Array.from({ length: h }, (_, i) => i) },
    visualMap: { min: Math.min(...data.map(d => d[2])), max: Math.max(...data.map(d => d[2])), calculable: true },
    series: [{ type: 'heatmap', data }]
  })
}

function renderLine(chart, title, yName, arrA = [], arrB = []) {
  chart?.setOption({
    title: { text: title, left: 'center', textStyle: { fontSize: 12 } },
    tooltip: { trigger: 'axis' },
    legend: { data: [selectedModel.value, compareModel.value], top: 20 },
    xAxis: { type: 'category', data: arrA.map((_, i) => i), name: 'time' },
    yAxis: { type: 'value', name: yName },
    series: [
      { name: selectedModel.value, type: 'line', data: arrA, smooth: true },
      { name: compareModel.value, type: 'line', data: arrB, smooth: true }
    ]
  })
}

async function loadModels() {
  const data = await getJson(`${api}/models`)
  models.value = data.models
  if (models.value.length && !models.value.includes(selectedModel.value)) selectedModel.value = models.value[0]
}

async function loadRuns() {
  const data = await getJson(`${api}/models/${selectedModel.value}/runs`)
  runs.value = data.runs
  if (runs.value.length) selectedRun.value = runs.value[runs.value.length - 1]
  await reloadAll()
}

async function reloadFields() {
  if (!selectedRun.value) return
  const [left, right, cmp] = await Promise.all([
    getJson(`${api}/models/${selectedModel.value}/${selectedRun.value}/field?kind=prediction_short&t=${t.value}`),
    getJson(`${api}/models/${selectedModel.value}/${selectedRun.value}/field?kind=gt_short&t=${t.value}`),
    getJson(`${api}/compare/${selectedRun.value}?left=${selectedModel.value}&right=${compareModel.value}&split=short&t=${t.value}`).catch(() => null)
  ])
  renderHeatmap(leftChart, `${selectedModel.value} prediction`, left.field)
  renderHeatmap(rightChart, `${selectedModel.value} ground truth`, right.field)
  renderHeatmap(diffChart, cmp ? `${selectedModel.value} - ${compareModel.value}` : 'compare (unavailable)', cmp?.diff_field || left.field.map(row => row.map(() => 0)))
}

async function reloadMetrics() {
  const a = await getJson(`${api}/models/${selectedModel.value}/${selectedRun.value}/metrics`)
  const b = await getJson(`${api}/models/${compareModel.value}/${selectedRun.value}/metrics`).catch(() => ({ short: { frame_mse: [], frame_rmse: [] } }))
  renderLine(mseChart, 'MSE 曲线', 'MSE', a.short.frame_mse, b.short.frame_mse)
  renderLine(rmseChart, 'RMSE 曲线', 'RMSE', a.short.frame_rmse, b.short.frame_rmse)
  maxT.value = Math.max(0, (a.short.frame_mse?.length || 1) - 1)
  if (t.value > maxT.value) t.value = maxT.value
}

async function reloadAll() {
  await Promise.all([reloadFields(), reloadMetrics()])
}

function togglePlay() {
  playing.value = !playing.value
  if (playing.value) {
    timer = setInterval(async () => {
      t.value = t.value >= maxT.value ? 0 : t.value + 1
      await reloadFields()
    }, 600)
  } else if (timer) {
    clearInterval(timer)
    timer = null
  }
}

onMounted(async () => {
  leftChart = echarts.init(leftRef.value)
  rightChart = echarts.init(rightRef.value)
  diffChart = echarts.init(diffRef.value)
  mseChart = echarts.init(mseRef.value)
  rmseChart = echarts.init(rmseRef.value)
  await loadModels()
  await loadRuns()
})

onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
})
</script>
