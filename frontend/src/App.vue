<template>
  <div class="page-shell">
    <div class="page-noise"></div>

    <main class="page">
      <section class="hero card">
        <div class="hero-copy">
          <p class="eyebrow">Heat Equation Laboratory</p>
          <h1>流体模拟自动化平台</h1>
          <p class="hero-text">
            统一读取 <code>web_exports</code> 下的结果集，按时间步展示 PINN、FNO 与解析真值的温度场演化，
            同时给出 RMSE / MSE 曲线，方便直接用于毕业设计演示。
          </p>
          <div class="hero-tags">
            <span class="tag">Heat Equation</span>
            <span class="tag">PINN</span>
            <span class="tag">FNO</span>
            <span class="tag">101 × 101 Grid</span>
          </div>
        </div>

        <div class="hero-side">
          <div class="formula-card">
            <div class="formula-label">控制方程</div>
            <div class="formula">u<sub>t</sub> = ν(u<sub>xx</sub> + u<sub>yy</sub>)</div>
            <div class="formula-note">
              解析解参考：<code>exp(-2π²t) sin(πx) sin(πy)</code>
            </div>
          </div>

          <div class="status-row">
            <div class="status-chip" :class="jobStatusClass">
              {{ jobStatusLabel }}
            </div>
            <div class="status-meta">当前对比切片：{{ splitLabel }}</div>
          </div>
        </div>
      </section>

      <section class="top-grid">
        <div class="left-stack">
        <article class="card control-card">
          <div class="section-head">
            <div>
              <p class="section-kicker">Result Selector</p>
              <h2>结果集与时间轴</h2>
            </div>
            <button class="ghost-btn" @click="reloadWorkspace" :disabled="loading">
              {{ loading ? '刷新中...' : '刷新结果' }}
            </button>
          </div>

          <div class="controls-grid">
            <label>
              <span>时间窗</span>
              <select v-model="view.split">
                <option value="short">Short</option>
                <option value="long">Long</option>
              </select>
            </label>

            <label>
              <span>PINN Run</span>
              <select v-model="view.demoPinnRun">
                <option v-for="run in pinnRunOptions" :key="run" :value="run">{{ run }}</option>
              </select>
            </label>
          </div>


          <div class="timeline-panel">
            <div class="timeline-top">
       <button class="play-btn" @click="togglePlay" :disabled="demoMaxT === 0">
  {{ playing ? '暂停' : '播放' }}
</button>
<div class="timeline-text">
  <strong>Frame {{ view.demoT }}</strong>
  <span>/ {{ demoMaxT }}</span>
  <small>{{ demoFrameNote }}</small>
</div>
            </div>

          <input
          v-model.number="view.demoT"
          type="range"
          min="0"
          :max="demoMaxT"
          class="slider"
          />
          </div>

          <div class="meta-strip">
            <div class="meta-item">
              <span class="meta-label">PINN</span>
              <strong>{{ view.demoPinnRun || '未找到' }}</strong>
            </div>
            <div class="meta-item">
              <span class="meta-label">演示帧数</span>
              <strong>{{ demoMaxT + 1 }}</strong>
                 </div>
          </div>
        </article>

        <section class="stat-grid stat-grid-compact stat-grid-top">
          <article v-for="item in topSummaryCards" :key="item.label" class="card stat-card">
            <span class="stat-label">{{ item.label }}</span>
            <strong class="stat-value">{{ item.value }}</strong>
            <small class="stat-note">{{ item.note }}</small>
          </article>
        </section>

        <article class="card plot-card plot-card-hero">
          <div class="plot-head">
            <h3>PINN Prediction</h3>
            <span>{{ view.demoPinnRun || '未选择' }}</span>
          </div>
          <div ref="pinnRef" class="chart heat-chart"></div>
          <div class="plot-footnote">
            <span>当前演示帧</span>
            <strong>{{ view.demoT }}</strong>
            <small>{{ splitLabel }} · 共 {{ demoMaxT + 1 }} 帧</small>
          </div>
        </article>
        </div>
        <article class="card quick-job-card">
          <div class="section-head">
            <div>
              <p class="section-kicker">PINN Preset Playback</p>
              <h2>PINN 参数化演示</h2>
            </div>
          </div>

          <div class="runtime-intro">
            <div class="runtime-intro-item">
              <span>演示方式</span>
              <strong>匹配预计算结果或触发在线 PINN 求解</strong>
            </div>
            <div class="runtime-intro-item">
              <span>结果来源</span>
              <strong>优先命中缓存；无缓存时后台训练并返回结果</strong>
            </div>
          </div>

          <div v-if="pinnPresets.length" class="preset-section">
            <div class="preset-head">
              <div>
                <p class="section-kicker">PINN Preset Runs</p>
                <h3>云端正式预设结果组</h3>
              </div>
              <small>这些参数组均来自已完成的 PINN 真实训练结果，可一键回填并映射到对应结果集。</small>
            </div>

            <div class="preset-grid">
              <button
                v-for="preset in pinnPresets"
                :key="preset.run"
                class="preset-card"
                type="button"
                @click="applyPreset(preset)"
              >
                <strong>{{ preset.title }}</strong>
                <span>{{ preset.note }}</span>
                <small>运行标识：{{ preset.run }}</small>
              </button>
            </div>
          </div>

          <div class="quick-grid runtime-core-grid">
            <label>
              <span>方程</span>
              <select v-model="quickForm.equation">
                <option value="heat">Heat Equation</option>
                <option value="poisson" disabled>Poisson (Reserved)</option>
                <option value="burgers" disabled>Burgers (Reserved)</option>
              </select>
            </label>
            <label>
              <span>ν</span>
              <input v-model.number="quickForm.nu" type="number" step="0.01" />
            </label>
            <label>
              <span>dt</span>
              <input v-model.number="quickForm.dt" type="number" step="0.00001" />
            </label>
            <label>
              <span>nx</span>
              <input v-model.number="quickForm.nx" type="number" step="1" />
            </label>
            <label>
              <span>ny</span>
              <input v-model.number="quickForm.ny" type="number" step="1" />
            </label>
            <label>
              <span>short</span>
              <input v-model.number="quickForm.short_steps" type="number" step="1" />
            </label>
            <label>
              <span>long</span>
              <input v-model.number="quickForm.long_steps" type="number" step="1" />
            </label>
            <label>
              <span>epoch</span>
              <input v-model.number="quickForm.epochs" type="number" step="1000" />
            </label>
            <label>
              <span>learning rate</span>
              <input v-model.number="quickForm.learning_rate" type="number" step="0.00001" />
            </label>
            <label>
              <span>seed</span>
              <input v-model.number="quickForm.seed" type="number" step="1" />
            </label>
          </div>

          <details class="advanced-panel">
            <summary>高级参数</summary>

            <div class="quick-grid advanced-grid">
            <label>
              <span>network</span>
              <select v-model="quickForm.network_type">
                <option value="transformer">Transformer</option>
                <option value="cnn">CNN</option>
              </select>
            </label>
            <label>
              <span>patch size</span>
              <input v-model.number="quickForm.patch_size" type="number" step="1" />
            </label>
            <label>
              <span>heads</span>
              <input v-model.number="quickForm.num_heads" type="number" step="1" />
            </label>
            <label>
              <span>layers</span>
              <input v-model.number="quickForm.num_layers" type="number" step="1" />
            </label>
            <label>
              <span>hidden</span>
              <input v-model.number="quickForm.transformer_hidden_channels" type="number" step="1" />
            </label>
            <label>
              <span>loss phy</span>
              <input v-model.number="quickForm.loss_phy_weight" type="number" step="0.1" />
            </label>
            <label>
              <span>loss data</span>
              <input v-model.number="quickForm.loss_data_weight" type="number" step="0.1" />
            </label>
            <label>
              <span>result mapping</span>
              <input :value="matchedPresetRunLabel" type="text" readonly />
            </label>
            </div>
          </details>

          <div class="runtime-summary">
            <div class="runtime-summary-item">
              <span>当前配置</span>
              <strong>{{ activePresetLabel }}</strong>
            </div>
            <div class="runtime-summary-item">
              <span>匹配状态</span>
              <strong>{{ presetMatchStatus }}</strong>
            </div>
          </div>

          <div class="solve-section">
            <button
              v-if="matchedPreset"
              class="primary-btn solve-btn solve-btn-matched"
              disabled
            >
              已匹配预计算结果 — 无需在线求解
            </button>
            <div v-if="solveState === 'running'" class="solve-btn-row">
              <button class="primary-btn solve-btn" disabled>
                求解中...
              </button>
              <button class="ghost-btn solve-cancel-btn" @click="cancelSolve">
                取消训练
              </button>
            </div>
            <button
              v-else
              class="primary-btn solve-btn"
              @click="submitSolve"
            >
              提交在线求解
            </button>
            <div v-if="solveState === 'running'" class="solve-hint">
              <template v-if="solveProgress">
                训练进度：{{ solveProgress.iteration }} / {{ solveProgress.total_epochs }} epochs
                ({{ solveProgress.pct }}%)，当前 loss {{ solveProgress.loss.toExponential(2) }}
              </template>
              <template v-else>
                后台正在启动训练进程...
              </template>
              <div class="solve-hint-refresh">刷新页面不会丢失任务进度，请放心。</div>
            </div>
            <div v-if="solveState === 'failed'" class="solve-hint solve-error">
              求解失败：{{ solveError }}
            </div>
            <div v-if="solveState === 'completed' && !matchedPreset" class="solve-hint solve-ok">
              求解完成！结果已加载至左侧演示面板。
            </div>
            <div v-if="solveState === 'idle' && !matchedPreset" class="solve-hint solve-ready">
              当前参数无缓存，点击按钮将启动后台 PINN 训练
            </div>
          </div>

        </article>
      </section>

      <section class="comparison-grid">
        <article class="card comparison-control-card">
          <div class="plot-head">
            <h3>PINN / FNO 对比控制</h3>
            <span>{{ splitLabel }}</span>
          </div>
          <div class="controls-grid comparison-controls-grid">
            <label>
              <span>PINN Run</span>
              <select v-model="view.comparePinnRun">
                <option v-for="run in epochRuns" :key="`cmp-${run}`" :value="run">{{ run }}</option>
              </select>
            </label>
            <label>
              <span>FNO Run</span>
              <select v-model="view.fnoRun">
                <option v-for="run in runs.fno" :key="`fno-${run}`" :value="run">{{ run }}</option>
              </select>
            </label>
          </div>
          <div class="timeline-panel comparison-timeline">
            <div class="timeline-top">
              <button class="play-btn" @click="toggleComparePlay" :disabled="maxT === 0">
                {{ comparePlaying ? '暂停' : '播放' }}
              </button>
              <div class="timeline-text">
                <strong>Frame {{ view.compareT }}</strong>
                <span>/ {{ maxT }}</span>
                <small>{{ compareFrameNote }}</small>
              </div>
            </div>
            <input
              v-model.number="view.compareT"
              type="range"
              min="0"
              :max="maxT"
              class="slider"
            />
          </div>
        </article>

        <article class="card plot-card">
          <div class="plot-head">
            <h3>PINN Prediction</h3>
            <span>{{ view.comparePinnRun || '未选择' }}</span>
          </div>
          <div ref="pinnCompareRef" class="chart heat-chart"></div>
        </article>

        <article class="card plot-card">
          <div class="plot-head">
            <h3>FNO Prediction</h3>
            <span>{{ view.fnoRun || '暂无结果' }}</span>
          </div>
          <div ref="fnoRef" class="chart heat-chart"></div>
        </article>

        <article class="card plot-card">
          <div class="plot-head">
            <h3>Ground Truth</h3>
            <span>解析真值</span>
          </div>
          <div ref="gtRef" class="chart heat-chart"></div>
        </article>

        <article class="card plot-card diff-card">
          <div class="plot-head">
            <h3>PINN - FNO</h3>
            <span>当前帧差异图</span>
          </div>
          <div ref="diffRef" class="chart heat-chart"></div>
          <div class="plot-footnote diff-footnote">
            <span>差异说明</span>
            <small>红色表示当前网格点上 PINN 预测值高于 FNO，蓝色表示 FNO 高于 PINN，接近白色表示两者结果接近。</small>
          </div>
        </article>
      </section>

      <section class="stat-grid stat-grid-compact stat-grid-metrics">
        <article v-for="item in bottomMetricCards" :key="item.label" class="card stat-card">
          <span class="stat-label">{{ item.label }}</span>
          <strong class="stat-value">{{ item.value }}</strong>
          <small class="stat-note">{{ item.note }}</small>
        </article>
      </section>

      <section class="metric-grid">
        <article class="card plot-card">
          <div class="plot-head">
            <h3>Frame RMSE</h3>
            <span>{{ splitLabel }}</span>
          </div>
          <div ref="rmseRef" class="chart metric-chart"></div>
        </article>

        <article class="card plot-card">
          <div class="plot-head">
            <h3>Frame MSE</h3>
            <span>{{ splitLabel }}</span>
          </div>
          <div ref="mseRef" class="chart metric-chart"></div>
        </article>
      </section>
    </main>
  </div>
</template>

<script setup>
import * as echarts from 'echarts/core'
import {
  GraphicComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import { HeatmapChart, LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'

echarts.use([
  CanvasRenderer,
  HeatmapChart,
  LineChart,
  GraphicComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
])

const api = 'http://127.0.0.1:8000/api'

const quickForm = reactive({
  equation: 'heat',
  nx: 101,
  ny: 101,
  nu: 1.0,
  dt: 1e-5,
  short_steps: 100,
  long_steps: 100,
  epochs: 500000,
  learning_rate: 1e-5,
  seed: 50976,
  patch_size: 16,
  num_heads: 4,
  num_layers: 4,
  transformer_hidden_channels: 128,
  network_type: 'transformer',
  loss_phy_weight: 1.0,
  loss_data_weight: 0.0,
})

const runs = reactive({
  pinn: [],
  fno: [],
})

const pinnPresets = ref([])

const view = reactive({
  split: 'short',
  demoPinnRun: '',
  comparePinnRun: '',
  fnoRun: '',
  demoT: 0,
  compareT: 0,
})

const loading = ref(false)
const playing = ref(false)
const comparePlaying = ref(false)
const initialized = ref(false)
const activePresetName = ref('自定义参数')

const solveState = ref('idle')
const solveJobId = ref('')
const solveRunName = ref('')
const solveError = ref('')
const solveProgress = ref(null)
let solvePollTimer = null

const meta = reactive({
  pinnDemo: null,
  pinn: null,
  fno: null,
})

const metrics = reactive({
  pinn: null,
  fno: null,
})

const fields = reactive({
  pinnMain: null,
  pinn: null,
  fno: null,
  gt: null,
  diff: null,
})

const pinnRef = ref(null)
const pinnCompareRef = ref(null)
const fnoRef = ref(null)
const gtRef = ref(null)
const diffRef = ref(null)
const rmseRef = ref(null)
const mseRef = ref(null)

let pinnChart
let pinnCompareChart
let fnoChart
let gtChart
let diffChart
let rmseChart
let mseChart
let playTimer = null
let comparePlayTimer = null

const matchedPreset = computed(
  () =>
    pinnPresets.value.find((preset) => {
      const params = preset?.params || {}
      return (
        String(params.equation) === String(quickForm.equation) &&
        Number(params.nx) === Number(quickForm.nx) &&
        Number(params.ny) === Number(quickForm.ny) &&
        Number(params.short_steps) === Number(quickForm.short_steps) &&
        Number(params.long_steps) === Number(quickForm.long_steps) &&
        Number(params.epochs) === Number(quickForm.epochs) &&
        Number(params.seed) === Number(quickForm.seed) &&
        Number(params.patch_size) === Number(quickForm.patch_size) &&
        Number(params.num_heads) === Number(quickForm.num_heads) &&
        Number(params.num_layers) === Number(quickForm.num_layers) &&
        Number(params.transformer_hidden_channels) === Number(quickForm.transformer_hidden_channels) &&
        String(params.network_type) === String(quickForm.network_type) &&
        Math.abs(Number(params.nu) - Number(quickForm.nu)) < 1e-12 &&
        Math.abs(Number(params.dt) - Number(quickForm.dt)) < 1e-12 &&
        Math.abs(Number(params.learning_rate) - Number(quickForm.learning_rate)) < 1e-12 &&
        Math.abs(Number(params.loss_phy_weight) - Number(quickForm.loss_phy_weight)) < 1e-12 &&
        Math.abs(Number(params.loss_data_weight) - Number(quickForm.loss_data_weight)) < 1e-12
      )
    }) || null,
)

const epochRuns = computed(() =>
  [...runs.pinn]
    .filter((run) => /^epoch_\d+$/i.test(run))
    .sort((left, right) => Number(left.replace(/\D/g, '')) - Number(right.replace(/\D/g, ''))),
)

const presetRuns = computed(() =>
  [...runs.pinn]
    .filter((run) => /^preset_/i.test(run))
    .sort(),
)

const pinnRunOptions = computed(() => [...presetRuns.value, ...epochRuns.value])

const jobStatusLabel = computed(() => {
  if (solveState.value === 'running') return '在线求解中'
  if (solveState.value === 'completed') return '求解完成'
  if (solveState.value === 'failed') return '求解失败'
  if (matchedPreset.value) return '预计算结果播放'
  return '等待参数匹配 / 在线求解'
})
const jobStatusClass = computed(() => {
  if (solveState.value === 'running') return 'running'
  if (solveState.value === 'completed') return 'ok'
  if (solveState.value === 'failed') return 'bad'
  if (matchedPreset.value) return 'ok'
  return 'idle'
})

const splitLabel = computed(() => (view.split === 'short' ? '短时演化' : '长时演化'))
const activePresetLabel = computed(() => {
  if (solveState.value === 'running') return '在线求解中...'
  if (solveState.value === 'completed' && solveRunName.value) return solveRunName.value
  return matchedPreset.value?.title || activePresetName.value || '自定义参数'
})
const matchedPresetRunLabel = computed(() => {
  if (solveRunName.value) return solveRunName.value
  return matchedPreset.value?.run || '未匹配到结果集'
})
const presetMatchStatus = computed(() => {
  if (solveState.value === 'running') return '正在后台训练 PINN 网络，请稍候...'
  if (solveState.value === 'completed') return `求解完成：${solveRunName.value || ''}`
  if (solveState.value === 'failed') return `求解失败：${solveError.value || '未知错误'}`
  if (matchedPreset.value) return `已匹配 ${matchedPreset.value.run}`
  return '当前参数未对应任何预计算结果，可提交在线求解'
})
const demoMaxT = computed(() => {
  const frames = currentFrameCount(meta.pinnDemo)
  return frames ? Math.max(0, frames - 1) : 0
})
const maxT = computed(() => {
  const frameCounts = []
  const pinnFrames = currentFrameCount(meta.pinn)
  const gtFrames = currentFrameCount(meta.pinn) || currentFrameCount(meta.fno)
  if (pinnFrames) frameCounts.push(pinnFrames)
  if (gtFrames) frameCounts.push(gtFrames)
  const fnoFrames = currentFrameCount(meta.fno)
  if (fnoFrames) frameCounts.push(fnoFrames)
  if (!frameCounts.length) return 0
  return Math.max(0, Math.min(...frameCounts) - 1)
})

const demoFrameNote = computed(() => `${splitLabel.value}，用于 PINN 参数化演示`)
const compareFrameNote = computed(() => `${splitLabel.value}，用于 PINN / FNO 对比播放`)

const topSummaryCards = computed(() => [
  {
    label: 'PINN Run',
    value: view.demoPinnRun || '—',
    note: '当前 PINN 演示结果集',
  },
 {
  label: '演示帧数',
  value: String(demoMaxT.value + 1),
  note: `${splitLabel.value} PINN 演示帧数`,
},
  {
    label: '参数映射',
    value: matchedPreset.value?.run || '未匹配',
    note: '当前参数对应的预计算 PINN 结果集',
  },
])

const statCards = computed(() => {
  const pinnSeries = metrics.pinn?.[view.split] || null
  const fnoSeries = metrics.fno?.[view.split] || null
  return [
    {
      label: 'PINN RMSE',
      value: formatMetric(pinnSeries?.rmse),
      note: '来自当前 PINN 结果集',
    },
    {
      label: 'FNO RMSE',
      value: formatMetric(fnoSeries?.rmse),
      note: view.fnoRun ? '来自当前 FNO 结果集' : 'FNO 结果尚未加载',
    },
    {
      label: '当前帧差异',
      value: formatMetric(currentMeanAbsoluteGap()),
      note: 'PINN 与 FNO 当前帧平均绝对差',
    },
    {
      label: '网格 / 帧数',
      value: meta.pinn ? `${meta.pinn.grid.width} × ${meta.pinn.grid.height}` : '—',
      note: `${splitLabel.value} 共 ${maxT.value + 1} 帧`,
    },
  ]
})

const bottomMetricCards = computed(() => statCards.value.filter((_, index) => index !== 2))

async function getJson(url) {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(await response.text())
  }
  return response.json()
}

async function getJsonOrNull(url) {
  try {
    return await getJson(url)
  } catch (error) {
    return null
  }
}

function sortRuns(list = []) {
  return [...list].sort((left, right) => {
    const leftNum = Number(left.replace(/\D/g, ''))
    const rightNum = Number(right.replace(/\D/g, ''))
    return rightNum - leftNum
  })
}

function findLatestCommonRun(leftRuns = [], rightRuns = []) {
  if (!leftRuns.length || !rightRuns.length) return ''
  const rightRunSet = new Set(rightRuns)
  return leftRuns.find((run) => rightRunSet.has(run)) || ''
}

function currentFrameCount(modelMeta) {
  if (!modelMeta?.frames) return 0
  return view.split === 'short' ? modelMeta.frames.short : modelMeta.frames.long
}

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  if (value === 0) return '0'
  if (Math.abs(value) < 1e-3) return value.toExponential(2)
  return value.toFixed(6)
}

function formatAxisMetric(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return ''
  if (value === 0) return '0'
  const abs = Math.abs(value)
  if (abs < 1e-4) return value.toExponential(1)
  if (abs < 1e-3) return value.toExponential(2)
  return value.toFixed(6)
}

function currentMeanAbsoluteGap() {
  if (!fields.diff?.length) return null
  let total = 0
  let count = 0
  for (const row of fields.diff) {
    for (const value of row) {
      total += Math.abs(value)
      count += 1
    }
  }
  return count ? total / count : null
}

function matrixRange(matrices) {
  let minValue = Infinity
  let maxValue = -Infinity
  for (const matrix of matrices) {
    if (!matrix?.length) continue
    for (const row of matrix) {
      for (const value of row) {
        if (value < minValue) minValue = value
        if (value > maxValue) maxValue = value
      }
    }
  }
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) return { min: 0, max: 1 }
  if (minValue === maxValue) return { min: minValue - 1, max: maxValue + 1 }
  return { min: minValue, max: maxValue }
}

function symmetricRange(matrix) {
  if (!matrix?.length) return { min: -1, max: 1 }
  let maxAbs = 0
  for (const row of matrix) {
    for (const value of row) {
      maxAbs = Math.max(maxAbs, Math.abs(value))
    }
  }
  return { min: -maxAbs || -1, max: maxAbs || 1 }
}

function computeDiff(left, right) {
  if (!left?.length || !right?.length) return null
  if (left.length !== right.length || left[0]?.length !== right[0]?.length) return null
  return left.map((row, y) => row.map((value, x) => value - right[y][x]))
}

function renderEmpty(chart, title, message) {
  chart?.setOption(
    {
      title: {
        text: title,
        left: 12,
        top: 12,
        textStyle: { color: '#17324d', fontSize: 14, fontWeight: 700 },
      },
      graphic: {
        type: 'text',
        left: 'center',
        top: 'middle',
        style: {
          text: message,
          fill: '#7b8b9f',
          fontSize: 14,
          fontWeight: 500,
        },
      },
      xAxis: { show: false },
      yAxis: { show: false },
      series: [],
    },
    true,
  )
}

function renderHeatmap(chart, title, matrix, colorStops, range, unitLabel) {
  if (!chart || !matrix?.length) {
    renderEmpty(chart, title, '暂无数据')
    return
  }
  const rows = matrix.length
  const cols = matrix[0].length
  const data = []
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      data.push([x, y, matrix[y][x]])
    }
  }

  chart.setOption(
    {
      animation: false,
      title: {
        text: title,
        left: 12,
        top: 12,
        textStyle: { color: '#17324d', fontSize: 14, fontWeight: 700 },
      },
      tooltip: {
        position: 'top',
        formatter: ({ data: item }) => `x=${item[0]}<br/>y=${item[1]}<br/>${unitLabel}: ${item[2].toFixed(6)}`,
      },
      grid: { left: 8, right: 42, top: 42, bottom: 10, containLabel: false },
      xAxis: {
        type: 'category',
        data: Array.from({ length: cols }, (_, index) => index),
        axisLabel: { show: false },
        axisTick: { show: false },
        axisLine: { show: false },
      },
      yAxis: {
        type: 'category',
        data: Array.from({ length: rows }, (_, index) => index),
        axisLabel: { show: false },
        axisTick: { show: false },
        axisLine: { show: false },
      },
      visualMap: {
        min: range.min,
        max: range.max,
        orient: 'vertical',
        right: 0,
        top: 'middle',
        itemHeight: 140,
        text: ['高', '低'],
        textStyle: { color: '#6d7e92' },
        calculable: false,
        inRange: { color: colorStops },
      },
      series: [
        {
          type: 'heatmap',
          data,
          progressive: 0,
          animation: false,
        },
      ],
    },
    true,
  )
}

function renderMetricChart(chart, title, pinnSeries, fnoSeries, colorA, colorB) {
  if (!chart) return
  const frameAxisLength = Math.max(pinnSeries?.length || 0, fnoSeries?.length || 0)
  const axisData = Array.from({ length: frameAxisLength }, (_, index) => index)

  chart.setOption(
    {
      animation: false,
      title: {
        text: title,
        left: 12,
        top: 12,
        textStyle: { color: '#17324d', fontSize: 14, fontWeight: 700 },
      },
      tooltip: { trigger: 'axis' },
      legend: {
        top: 10,
        right: 12,
        textStyle: { color: '#60758b' },
      },
      grid: { left: 66, right: 18, top: 46, bottom: 30 },
      xAxis: {
        type: 'category',
        data: axisData,
        axisLine: { lineStyle: { color: '#ced8e3' } },
        axisLabel: { color: '#60758b' },
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        splitLine: { lineStyle: { color: '#e5ebf2' } },
        axisLabel: {
          color: '#60758b',
          formatter: (value) => formatAxisMetric(value),
        },
      },
      series: [
        {
          name: 'PINN',
          type: 'line',
          data: pinnSeries || [],
          smooth: true,
          symbol: 'none',
          lineStyle: { color: colorA, width: 2.4 },
          markLine: {
            symbol: 'none',
            label: { show: false },
            lineStyle: { color: '#8b95a3', type: 'dashed' },
            data: [{ xAxis: view.compareT }],
          },
        },
        {
          name: 'FNO',
          type: 'line',
          data: fnoSeries || [],
          smooth: true,
          symbol: 'none',
          lineStyle: { color: colorB, width: 2.4 },
        },
      ],
    },
    true,
  )
}

function renderAllCharts() {
  const range = matrixRange([fields.pinn, fields.fno, fields.gt])
  renderHeatmap(
    pinnChart,
    'PINN Prediction',
    fields.pinnMain,
    ['#fff8e8', '#ffd089', '#f4a261', '#e76f51', '#b9375d'],
    range,
    'u',
  )
  renderHeatmap(
    pinnCompareChart,
    'PINN Prediction',
    fields.pinn,
    ['#fff8e8', '#ffd089', '#f4a261', '#e76f51', '#b9375d'],
    range,
    'u',
  )
  if (fields.fno) {
    renderHeatmap(
      fnoChart,
      'FNO Prediction',
      fields.fno,
      ['#f7fcfd', '#ccece6', '#66c2a4', '#238b45', '#00441b'],
      range,
      'u',
    )
  } else {
    renderEmpty(fnoChart, 'FNO Prediction', 'FNO 结果尚未导出')
  }
  renderHeatmap(
    gtChart,
    'Ground Truth',
    fields.gt,
    ['#fff8e8', '#ffd089', '#f4a261', '#e76f51', '#b9375d'],
    range,
    'u',
  )
  if (fields.diff) {
    renderHeatmap(
      diffChart,
      'PINN - FNO',
      fields.diff,
      ['#1d4e89', '#8ecae6', '#f8f9fa', '#f7b7a3', '#c1121f'],
      symmetricRange(fields.diff),
      '差值',
    )
  } else {
    renderEmpty(diffChart, 'PINN - FNO', '需要同时选择 PINN 与 FNO 结果')
  }
  renderMetricChart(
    rmseChart,
    'Frame RMSE',
    metrics.pinn?.[view.split]?.frame_rmse || [],
    metrics.fno?.[view.split]?.frame_rmse || [],
    '#d96c06',
    '#0f766e',
  )
  renderMetricChart(
    mseChart,
    'Frame MSE',
    metrics.pinn?.[view.split]?.frame_mse || [],
    metrics.fno?.[view.split]?.frame_mse || [],
    '#c2410c',
    '#0f766e',
  )
}

async function loadRuns() {
  const [pinnData, fnoData] = await Promise.all([
    getJsonOrNull(`${api}/models/pinn/runs`),
    getJsonOrNull(`${api}/models/fno/runs`),
  ])

  runs.pinn = sortRuns(pinnData?.runs || [])
  runs.fno = sortRuns(fnoData?.runs || [])
  const latestCommonRun = findLatestCommonRun(runs.pinn, runs.fno)
  const latestEpochRun = epochRuns.value[epochRuns.value.length - 1] || ''

  if (!view.demoPinnRun || !runs.pinn.includes(view.demoPinnRun)) {
    view.demoPinnRun = runs.pinn.includes('epoch_500000') ? 'epoch_500000' : (latestEpochRun || runs.pinn[0] || '')
  }
  if (!view.comparePinnRun || !epochRuns.value.includes(view.comparePinnRun)) {
    view.comparePinnRun = latestCommonRun || latestEpochRun || ''
  }
  if (!view.fnoRun || !runs.fno.includes(view.fnoRun)) {
    view.fnoRun = latestCommonRun || (runs.fno.includes(view.comparePinnRun) ? view.comparePinnRun : runs.fno[0] || '')
  }
}

async function loadPresets() {
  const data = await getJsonOrNull(`${api}/pinn/presets`)
  pinnPresets.value = data?.presets || []
  syncMatchedPresetToView()
}

function applyPreset(preset) {
  if (!preset?.params) return
  Object.assign(quickForm, preset.params)
  activePresetName.value = preset.title || preset.run || '云端预设'
}

function syncMatchedPresetToView() {
  if (!matchedPreset.value) return
  activePresetName.value = matchedPreset.value.title || matchedPreset.value.run || '云端预设'
  if (view.demoPinnRun !== matchedPreset.value.run) {
    view.demoPinnRun = matchedPreset.value.run
  }
}
async function loadPinnDemoMeta(run) {
  if (!run) {
    meta.pinnDemo = null
    return
  }

  meta.pinnDemo = await getJsonOrNull(`${api}/models/pinn/${run}/meta`)
}
async function loadMetaAndMetrics(model, run) {
  if (!run) {
    meta[model] = null
    metrics[model] = null
    return
  }
  const [metaData, metricsData] = await Promise.all([
    getJsonOrNull(`${api}/models/${model}/${run}/meta`),
    getJsonOrNull(`${api}/models/${model}/${run}/metrics`),
  ])
  meta[model] = metaData
  metrics[model] = metricsData
}

async function loadFields() {
  const fieldKind = `prediction_${view.split}`
  const gtKind = `gt_${view.split}`
  const requests = [
    view.demoPinnRun ? getJsonOrNull(`${api}/models/pinn/${view.demoPinnRun}/field?kind=${fieldKind}&t=${view.demoT}`) : null,
    view.comparePinnRun ? getJsonOrNull(`${api}/models/pinn/${view.comparePinnRun}/field?kind=${fieldKind}&t=${view.compareT}`) : null,
    view.comparePinnRun ? getJsonOrNull(`${api}/models/pinn/${view.comparePinnRun}/field?kind=${gtKind}&t=${view.compareT}`) : null,
    view.fnoRun ? getJsonOrNull(`${api}/models/fno/${view.fnoRun}/field?kind=${fieldKind}&t=${view.compareT}`) : null,
  ]

  const [pinnMainData, pinnData, gtData, fnoData] = await Promise.all(requests)
  fields.pinnMain = pinnMainData?.field || null
  fields.pinn = pinnData?.field || null
  fields.gt = gtData?.field || null
  fields.fno = fnoData?.field || null
  fields.diff = computeDiff(fields.pinn, fields.fno)
}

async function refreshView() {
  loading.value = true
  try {
    await Promise.all([loadRuns(), loadPresets()])
    syncMatchedPresetToView()
 await Promise.all([
  loadPinnDemoMeta(view.demoPinnRun),
  loadMetaAndMetrics('pinn', view.comparePinnRun),
  loadMetaAndMetrics('fno', view.fnoRun),
])

if (view.demoT > demoMaxT.value) view.demoT = demoMaxT.value
if (view.compareT > maxT.value) view.compareT = maxT.value
    await loadFields()
    renderAllCharts()
  } finally {
    loading.value = false
  }
}

async function cancelSolve() {
  if (!solveJobId.value) return
  try {
    await fetch(`${api}/pinn/jobs/${encodeURIComponent(solveJobId.value)}`, { method: 'DELETE' })
  } catch (_) {
    // best-effort, server may already be down
  }
  if (solvePollTimer) {
    clearInterval(solvePollTimer)
    solvePollTimer = null
  }
  solveState.value = 'idle'
  solveJobId.value = ''
  solveProgress.value = null
  localStorage.removeItem('pinn_solve_job_id')
  localStorage.removeItem('pinn_solve_total_epochs')
}

async function reloadWorkspace() {
  await refreshView()
}

async function submitSolve() {
  if (solveState.value === 'running') return

  solveError.value = ''
  solveRunName.value = ''

  const body = {
    equation: quickForm.equation,
    nx: quickForm.nx,
    ny: quickForm.ny,
    nu: quickForm.nu,
    dt: quickForm.dt,
    short_steps: quickForm.short_steps,
    long_steps: quickForm.long_steps,
    epochs: quickForm.epochs,
    learning_rate: quickForm.learning_rate,
    seed: quickForm.seed,
    network_type: quickForm.network_type,
    transformer_hidden_channels: quickForm.transformer_hidden_channels,
    patch_size: quickForm.patch_size,
    num_heads: quickForm.num_heads,
    num_layers: quickForm.num_layers,
    loss_phy_weight: quickForm.loss_phy_weight,
    loss_data_weight: quickForm.loss_data_weight,
  }

  try {
    const response = await fetch(`${api}/pinn/solve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!response.ok) {
      throw new Error(await response.text())
    }
    const data = await response.json()

    if (data.cached || data.status === 'completed') {
      solveState.value = 'completed'
      solveRunName.value = data.run_name
      view.demoPinnRun = data.run_name
      await refreshView()
      return
    }

    solveState.value = 'running'
    solveJobId.value = data.job_id
    localStorage.setItem('pinn_solve_job_id', data.job_id)
    localStorage.setItem('pinn_solve_total_epochs', String(quickForm.epochs))
    pollJobStatus(data.job_id)
  } catch (err) {
    solveState.value = 'failed'
    solveError.value = String(err)
  }
}

function pollJobStatus(jobId) {
  if (solvePollTimer) clearInterval(solvePollTimer)

  solvePollTimer = setInterval(async () => {
    try {
      const response = await fetch(`${api}/pinn/jobs/${encodeURIComponent(jobId)}`)
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const data = await response.json()

      if (data.status === 'completed') {
        clearInterval(solvePollTimer)
        solvePollTimer = null
        solveState.value = 'completed'
        solveRunName.value = data.run_name
        view.demoPinnRun = data.run_name
        localStorage.removeItem('pinn_solve_job_id')
        localStorage.removeItem('pinn_solve_total_epochs')
        await refreshView()
      } else if (data.status === 'failed') {
        clearInterval(solvePollTimer)
        solvePollTimer = null
        solveState.value = 'failed'
        solveError.value = data.error || '未知错误'
        localStorage.removeItem('pinn_solve_job_id')
        localStorage.removeItem('pinn_solve_total_epochs')
      } else if (data.status === 'cancelled') {
        clearInterval(solvePollTimer)
        solvePollTimer = null
        solveState.value = 'idle'
        solveJobId.value = ''
        solveProgress.value = null
        localStorage.removeItem('pinn_solve_job_id')
        localStorage.removeItem('pinn_solve_total_epochs')
      } else if (data.progress) {
        solveProgress.value = data.progress
      }
    } catch (err) {
      clearInterval(solvePollTimer)
      solvePollTimer = null
      solveState.value = 'failed'
      solveError.value = String(err)
    }
  }, 3000)
}

function togglePlay() {
  playing.value = !playing.value

  if (!playing.value) {
    if (playTimer) clearInterval(playTimer)
    playTimer = null
    return
  }

  playTimer = setInterval(() => {
    view.demoT = view.demoT >= demoMaxT.value ? 0 : view.demoT + 1
  }, 260)
}

function toggleComparePlay() {
  comparePlaying.value = !comparePlaying.value
  if (!comparePlaying.value) {
    if (comparePlayTimer) clearInterval(comparePlayTimer)
    comparePlayTimer = null
    return
  }
  comparePlayTimer = setInterval(async () => {
    view.compareT = view.compareT >= maxT.value ? 0 : view.compareT + 1
  }, 260)
}

function handleResize() {
  pinnChart?.resize()
  pinnCompareChart?.resize()
  fnoChart?.resize()
  gtChart?.resize()
  diffChart?.resize()
  rmseChart?.resize()
  mseChart?.resize()
}

watch(
  () => [view.split, view.demoPinnRun, view.comparePinnRun, view.fnoRun],
  async (_newValues, _oldValues) => {
    if (!initialized.value) return

    const oldDemoRun = _oldValues?.[1]
    if (oldDemoRun && view.demoPinnRun !== oldDemoRun) {
      view.demoT = 0
    }

    await Promise.all([
      loadPinnDemoMeta(view.demoPinnRun),
      loadMetaAndMetrics('pinn', view.comparePinnRun),
      loadMetaAndMetrics('fno', view.fnoRun),
    ])

    if (view.demoT > demoMaxT.value) view.demoT = demoMaxT.value
    if (view.compareT > maxT.value) view.compareT = maxT.value

    await loadFields()
    renderAllCharts()
  },
)

watch(
  () => [
    quickForm.equation,
    quickForm.nx,
    quickForm.ny,
    quickForm.nu,
    quickForm.dt,
    quickForm.short_steps,
    quickForm.long_steps,
    quickForm.epochs,
    quickForm.learning_rate,
    quickForm.seed,
    quickForm.patch_size,
    quickForm.num_heads,
    quickForm.num_layers,
    quickForm.transformer_hidden_channels,
    quickForm.network_type,
    quickForm.loss_phy_weight,
    quickForm.loss_data_weight,
  ],
  async () => {
    if (!initialized.value) return

    if (solveState.value !== 'running') {
      solveState.value = 'idle'
      solveRunName.value = ''
      solveError.value = ''
    }

    syncMatchedPresetToView()
  },
)

watch(
  () => [view.demoT, view.compareT],
  async () => {
    if (!initialized.value) return
    await loadFields()
    renderAllCharts()
  },
)

onMounted(async () => {
  pinnChart = echarts.init(pinnRef.value)
  pinnCompareChart = echarts.init(pinnCompareRef.value)
  fnoChart = echarts.init(fnoRef.value)
  gtChart = echarts.init(gtRef.value)
  diffChart = echarts.init(diffRef.value)
  rmseChart = echarts.init(rmseRef.value)
  mseChart = echarts.init(mseRef.value)

  window.addEventListener('resize', handleResize)
  await refreshView()
  initialized.value = true

  const savedJobId = localStorage.getItem('pinn_solve_job_id')
  if (savedJobId) {
    const totalEpochs = Number(localStorage.getItem('pinn_solve_total_epochs') || '0')
    try {
      const resp = await fetch(`${api}/pinn/jobs/${encodeURIComponent(savedJobId)}`)
      if (resp.ok) {
        const data = await resp.json()
        if (data.status === 'running') {
          solveState.value = 'running'
          solveJobId.value = savedJobId
          pollJobStatus(savedJobId)
        } else if (data.status === 'completed') {
          solveState.value = 'completed'
          solveRunName.value = data.run_name
          view.demoPinnRun = data.run_name
          localStorage.removeItem('pinn_solve_job_id')
          localStorage.removeItem('pinn_solve_total_epochs')
          await refreshView()
        } else {
          localStorage.removeItem('pinn_solve_job_id')
          localStorage.removeItem('pinn_solve_total_epochs')
        }
      }
    } catch (_) {
      // backend not available, keep localStorage for next try
    }
  }
})

onBeforeUnmount(() => {
  if (playTimer) clearInterval(playTimer)
  if (comparePlayTimer) clearInterval(comparePlayTimer)
  if (solvePollTimer) clearInterval(solvePollTimer)
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;700&display=swap');

:global(*) {
  box-sizing: border-box;
}

:global(body) {
  margin: 0;
  font-family: 'Noto Sans SC', 'Manrope', sans-serif;
  color: #17324d;
  background: #f4f1ea;
}

:global(code) {
  padding: 0.12rem 0.36rem;
  border-radius: 999px;
  background: rgba(23, 50, 77, 0.08);
  font-family: 'Manrope', monospace;
}

.page-shell {
  position: relative;
  min-height: 100vh;
  overflow: hidden;
  background:
    radial-gradient(circle at top left, rgba(255, 214, 102, 0.28), transparent 32%),
    radial-gradient(circle at top right, rgba(51, 154, 240, 0.14), transparent 24%),
    linear-gradient(180deg, #faf7f1 0%, #f2efe8 100%);
}

.page-noise {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(23, 50, 77, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(23, 50, 77, 0.03) 1px, transparent 1px);
  background-size: 32px 32px;
  mask-image: radial-gradient(circle at center, black 48%, transparent 100%);
  pointer-events: none;
}

.page {
  position: relative;
  z-index: 1;
  width: min(1400px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 28px 0 40px;
}

.card {
  border: 1px solid rgba(23, 50, 77, 0.08);
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(10px);
  box-shadow: 0 18px 48px rgba(29, 53, 87, 0.08);
  animation: rise-in 0.55s ease;
}

.hero {
  display: grid;
  grid-template-columns: 1.7fr 1fr;
  gap: 20px;
  padding: 28px;
}

.eyebrow,
.section-kicker {
  margin: 0 0 8px;
  color: #c95f1b;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.hero h1,
.section-head h2 {
  margin: 0;
  font-family: 'Manrope', sans-serif;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.hero h1 {
  font-size: clamp(32px, 4vw, 46px);
  line-height: 1.06;
}

.hero-text {
  max-width: 760px;
  margin: 16px 0 0;
  color: #526579;
  line-height: 1.75;
}

.hero-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 22px;
}

.tag {
  border-radius: 999px;
  padding: 8px 14px;
  background: #fff4dd;
  color: #9b4f16;
  font-size: 13px;
  font-weight: 700;
}

.hero-side {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.formula-card {
  border-radius: 24px;
  padding: 18px 20px;
  background: linear-gradient(145deg, #17324d, #214f73);
  color: #f6f9fc;
}

.formula-label {
  color: rgba(246, 249, 252, 0.72);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.formula {
  margin-top: 10px;
  font-family: 'Manrope', sans-serif;
  font-size: 28px;
  font-weight: 800;
}

.formula-note {
  margin-top: 12px;
  color: rgba(246, 249, 252, 0.78);
  line-height: 1.6;
}

.status-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  border-radius: 22px;
  padding: 16px 18px;
  background: #fffaf2;
}

.status-chip {
  border-radius: 999px;
  padding: 8px 14px;
  border: 1px solid #dde6f0;
  font-size: 13px;
  font-weight: 700;
}

.status-chip.ok {
  color: #0f766e;
  border-color: rgba(15, 118, 110, 0.28);
  background: rgba(15, 118, 110, 0.08);
}

.status-chip.bad {
  color: #b42318;
  border-color: rgba(180, 35, 24, 0.24);
  background: rgba(180, 35, 24, 0.08);
}

.status-chip.running {
  color: #a15c07;
  border-color: rgba(161, 92, 7, 0.22);
  background: rgba(255, 214, 102, 0.22);
}

.status-chip.idle {
  color: #5f7388;
  background: rgba(95, 115, 136, 0.08);
}

.status-meta {
  color: #60758b;
  font-size: 13px;
}

.top-grid,
.metric-grid {
  display: grid;
  grid-template-columns: 1.3fr 0.9fr;
  gap: 18px;
  margin-top: 18px;
}

.top-grid {
  align-items: start;
}

.left-stack {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.control-card,
.quick-job-card {
  padding: 22px;
}

.section-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}

.ghost-btn,
.primary-btn,
.play-btn {
  border: 0;
  border-radius: 16px;
  padding: 12px 18px;
  font-family: 'Manrope', sans-serif;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
}

.ghost-btn {
  background: #eef4f8;
  color: #1d425f;
}

.ghost-btn-small {
  padding: 10px 14px;
  border-radius: 14px;
  font-size: 13px;
}

.primary-btn {
  background: linear-gradient(135deg, #1d4ed8, #0f766e);
  color: #fff;
  box-shadow: 0 12px 24px rgba(29, 78, 216, 0.18);
}

.play-btn {
  min-width: 88px;
  background: #17324d;
  color: #fff;
}

.ghost-btn:hover,
.primary-btn:hover,
.play-btn:hover {
  transform: translateY(-1px);
}

.ghost-btn:disabled,
.primary-btn:disabled,
.play-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
  transform: none;
}

.controls-grid,
.quick-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

label {
  display: flex;
  flex-direction: column;
  gap: 7px;
  color: #5b6f82;
  font-size: 13px;
  font-weight: 600;
}

select,
input[type='number'] {
  width: 100%;
  border: 1px solid #d9e2ec;
  border-radius: 16px;
  padding: 12px 14px;
  background: #fbfcfd;
  color: #17324d;
  outline: none;
}

select:focus,
input[type='number']:focus {
  border-color: #6ea8fe;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.14);
}

.timeline-panel {
  margin-top: 18px;
  border-radius: 22px;
  padding: 16px 18px;
  background: linear-gradient(180deg, #fff9ef, #fffdf8);
  border: 1px solid #f3e5c7;
}

.epoch-browser {
  margin-top: 18px;
  border-radius: 22px;
  padding: 16px 18px;
  background: #f7fafc;
  border: 1px solid #dce7f2;
}

.epoch-browser-head,
.epoch-browser-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.epoch-browser-copy {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.epoch-browser-copy strong,
.epoch-browser-meta strong {
  color: #17324d;
  font-family: 'Manrope', sans-serif;
}

.epoch-browser-copy small,
.epoch-browser-meta span,
.epoch-browser-meta small {
  color: #6f8297;
  font-size: 12px;
}

.epoch-browser-meta {
  margin-top: 12px;
}

.timeline-top {
  display: flex;
  align-items: center;
  gap: 14px;
}

.timeline-text {
  display: flex;
  align-items: baseline;
  gap: 8px;
  color: #42586d;
}

.timeline-text strong {
  font-family: 'Manrope', sans-serif;
  color: #17324d;
}

.timeline-text small {
  margin-left: auto;
  color: #7b8b9f;
}

.slider {
  width: 100%;
  margin-top: 16px;
  accent-color: #1d4ed8;
}

.meta-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.meta-item {
  border-radius: 18px;
  padding: 14px;
  background: #f7fafc;
}

.meta-label,
.stat-label {
  display: block;
  color: #7b8b9f;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.meta-item strong {
  display: block;
  margin-top: 6px;
  font-family: 'Manrope', sans-serif;
  color: #17324d;
}

.runtime-intro {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 14px;
}

.runtime-intro-item {
  padding: 12px 14px;
  border-radius: 16px;
  background: #f7fafc;
  border: 1px solid #dbe7f3;
}

.runtime-intro-item span {
  display: block;
  color: #7b8b9f;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.runtime-intro-item strong {
  display: block;
  margin-top: 6px;
  color: #17324d;
  line-height: 1.45;
}

.runtime-core-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 14px;
}

.advanced-panel {
  margin-top: 14px;
  border: 1px solid #dbe7f3;
  border-radius: 18px;
  background: #fbfdff;
  overflow: hidden;
}

.advanced-panel summary {
  padding: 12px 16px;
  color: #35506b;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  list-style: none;
}

.advanced-panel summary::-webkit-details-marker {
  display: none;
}

.advanced-panel[open] summary {
  border-bottom: 1px solid #e3edf7;
  background: #f7fafc;
}

.advanced-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 0;
  padding: 14px 16px 16px;
}

.preset-section {
  margin-top: 16px;
  padding: 16px;
  border-radius: 20px;
  background: linear-gradient(180deg, #f8fbff, #fdfefe);
  border: 1px solid #dbe7f3;
}

.preset-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 14px;
}

.preset-head h3 {
  margin: 4px 0 0;
  font-family: 'Manrope', sans-serif;
  font-size: 17px;
  font-weight: 800;
  color: #17324d;
}

.preset-head small {
  max-width: 280px;
  color: #70859a;
  line-height: 1.6;
}

.preset-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 14px;
}

.preset-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 5px;
  padding: 12px 14px;
  border: 1px solid #dbe7f3;
  border-radius: 18px;
  background: #ffffff;
  color: #17324d;
  cursor: pointer;
  text-align: left;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.preset-card:hover {
  transform: translateY(-1px);
  border-color: #8bb5ff;
  box-shadow: 0 12px 24px rgba(23, 50, 77, 0.08);
}

.preset-card strong {
  font-family: 'Manrope', sans-serif;
  font-size: 14px;
}

.preset-card span,
.preset-card small {
  color: #60758b;
  line-height: 1.45;
}

.runtime-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.runtime-summary-item {
  border-radius: 18px;
  padding: 12px 14px;
  background: #fffaf2;
  border: 1px solid #f3e5c7;
}

.runtime-summary-item span {
  display: block;
  color: #7b8b9f;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.runtime-summary-item strong {
  display: block;
  margin-top: 6px;
  color: #17324d;
  line-height: 1.45;
}

.quick-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-top: 16px;
}

.job-inline {
  color: #5f7388;
  font-size: 13px;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 18px;
  margin-top: 18px;
}

.stat-grid-compact {
  margin-top: 0;
}

.stat-grid-top {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.stat-grid-metrics {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-top: 18px;
}

.stat-card {
  padding: 20px;
}

.stat-value {
  display: block;
  margin-top: 10px;
  font-family: 'Manrope', sans-serif;
  font-size: 28px;
  font-weight: 800;
  color: #17324d;
}

.stat-note {
  display: block;
  margin-top: 8px;
  color: #74869a;
  line-height: 1.6;
}

.prediction-layout {
  display: grid;
  gap: 18px;
  margin-top: 18px;
}

.plots-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
  margin-top: 18px;
}

.plots-grid-secondary {
  margin-top: 0;
}

.comparison-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
  margin-top: 18px;
}

.comparison-control-card {
  grid-column: 1 / -1;
  padding: 18px;
}

.comparison-timeline {
  margin-top: 12px;
}

.comparison-grid > .diff-card {
  grid-column: 2;
}

.plot-card {
  padding: 18px;
}

.plot-card-hero .heat-chart {
  height: 420px;
}

.plot-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
}

.plot-head-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.plot-head h3 {
  margin: 0;
  font-family: 'Manrope', sans-serif;
  font-size: 18px;
  font-weight: 800;
  color: #17324d;
}

.plot-head span {
  color: #7b8b9f;
  font-size: 13px;
}

.metric-pill {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  background: #fff5e8;
  border: 1px solid #f3e5c7;
  color: #9a5b12;
  font-size: 12px;
  font-weight: 700;
}

.plot-footnote {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #e7eef5;
  color: #6d8094;
  flex-wrap: wrap;
}

.plot-footnote span {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.plot-footnote strong {
  color: #17324d;
  font-family: 'Manrope', sans-serif;
}

.diff-footnote {
  align-items: flex-start;
}

.diff-footnote small {
  color: #6d8094;
  line-height: 1.65;
  max-width: 100%;
}

.solve-section {
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.solve-btn {
  width: 100%;
  justify-content: center;
}

.solve-btn-row {
  display: flex;
  gap: 10px;
}

.solve-btn-row .solve-btn {
  flex: 1;
}

.solve-cancel-btn {
  flex-shrink: 0;
  background: #fff5f5 !important;
  color: #b42318 !important;
  border: 1px solid #fdd !important;
}

.solve-hint {
  padding: 10px 14px;
  border-radius: 14px;
  background: #fff9ef;
  border: 1px solid #f3e5c7;
  color: #8a5a14;
  font-size: 13px;
  line-height: 1.55;
}

.solve-error {
  background: #fff5f5;
  border-color: #fdd;
  color: #b42318;
}

.solve-ok {
  background: #f0fdf4;
  border-color: #bbf7d0;
  color: #0f766e;
}

.solve-ready {
  background: #eff6ff;
  border-color: #bfdbfe;
  color: #1d4ed8;
}

.solve-btn-matched {
  background: #f0fdf4 !important;
  color: #0f766e !important;
  box-shadow: none !important;
  cursor: default;
  opacity: 0.85;
}

.solve-hint-refresh {
  margin-top: 6px;
  font-size: 12px;
  color: #8a5a14;
  opacity: 0.75;
}

.chart {
  width: 100%;
}

.heat-chart {
  height: 390px;
  margin-top: 14px;
}

.comparison-grid .heat-chart {
  height: 440px;
}

.diff-card .heat-chart {
  height: 300px;
}

.metric-chart {
  height: 280px;
  margin-top: 14px;
}

@keyframes rise-in {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 1180px) {
  .hero,
  .top-grid,
  .prediction-layout,
  .metric-grid,
  .plots-grid,
  .comparison-grid {
    grid-template-columns: 1fr;
  }

  .stat-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .page {
    width: min(100vw - 18px, 100%);
    padding: 14px 0 24px;
  }

  .hero,
  .control-card,
  .quick-job-card,
  .plot-card,
  .stat-card {
    padding: 18px;
    border-radius: 22px;
  }

  .controls-grid,
  .quick-grid,
  .meta-strip,
  .stat-grid,
  .preset-grid,
  .runtime-summary,
  .runtime-core-grid,
  .advanced-grid,
  .runtime-intro {
    grid-template-columns: 1fr;
  }

  .hero h1 {
    font-size: 30px;
  }

  .formula {
    font-size: 24px;
  }

  .timeline-top,
  .epoch-browser-head,
  .epoch-browser-meta,
  .preset-head,
  .quick-actions,
  .status-row {
    flex-direction: column;
    align-items: flex-start;
  }

  .plot-head,
  .plot-head-meta,
  .plot-footnote {
    align-items: flex-start;
  }

  .timeline-text {
    flex-wrap: wrap;
  }

  .heat-chart {
    height: 310px;
  }

  .plot-card-hero .heat-chart {
    height: 340px;
  }

  .comparison-grid .heat-chart {
    height: 320px;
  }

  .diff-card .heat-chart {
    height: 260px;
  }
}
</style>


