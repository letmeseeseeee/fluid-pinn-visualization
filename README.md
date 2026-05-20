# 流体模拟自动化平台

本项目围绕二维热方程构建了一套面向毕业设计展示的自动化平台。平台以原始 PINN 热方程求解模块为主求解链路，以 FNO 结果为离线对照链路，统一组织结果导出、接口访问、前端展示和在线求解流程。

## 项目定位

- **系统主线**：统一结果管理、接口访问、网页展示、参数化回放与在线求解。
- **学术支撑**：在同一热方程任务下完成 PINN 与 FNO 的结果对比分析。
- **交互方式**：用户填写参数后，系统匹配预计算结果并播放；无缓存时自动启动后台训练。

## 核心功能

| 功能 | 说明 |
|------|------|
| PINN 参数化演示 | 填写 16 个超参数，匹配 4 组预设或 500+ 个历史快照，热力图播放 |
| PINN / FNO 对比分析 | 独立时间轴下同步展示 PINN、FNO、真值、差异热力图 + RMSE/MSE 曲线 |
| 在线求解链路 | 参数无缓存时自动后台训练，实时进度反馈，完成后自动加载 |
| 取消训练 | 训练中可随时取消，进程即时终止 |
| 页面刷新恢复 | 训练中途刷新页面，自动恢复任务状态和进度 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3 + Vite + ECharts |
| 后端 | FastAPI + Uvicorn |
| 训练 | PyTorch (PINN) + neuraloperator (FNO) |
| 数据 | NumPy (.npy) + JSON |

## 项目结构

```text
graduate_project/
├── backend/
│   ├── app.py                    # FastAPI 主应用（11 个接口）
│   ├── pinn_original_runner.py   # PINN 训练执行器（缓存匹配 + 进程管理）
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.vue               # Vue 3 单文件组件（~2100 行）
│   │   └── main.js
│   ├── package.json
│   └── vite.config.js
├── Hea0.1.19.py                  # 原始 PINN 训练脚本
├── train_fno_neuralop.py         # FNO 训练脚本
├── web_export.py                 # 结果导出标准化模块
├── web_exports/                  # 统一结果存储（pinn/ + fno/）
├── docs/
│   ├── project_manual.md         # 项目说明书（8000+ 字）
│   └── thesis_draft.md           # 论文草稿
├── start_all.bat                 # 一键启动
└── stop_all.bat                  # 一键停止
```

## 结果目录规范

每个结果目录（`web_exports/{model}/{run}/`）统一包含：

- `prediction_short.npy` / `gt_short.npy` / `diff_short.npy` — 短时窗口数据
- `prediction_long.npy` / `gt_long.npy` / `diff_long.npy` — 长时窗口数据
- `meta.json` — 元数据（网格、帧数、超参数）
- `metrics.json` — 误差指标（总 MSE/RMSE + 逐帧序列）

## 一键启动

双击 `start_all.bat`，自动安装依赖并启动后端（8000）和前端（5173）。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/models` | 列出模型 |
| GET | `/api/models/{model}/runs` | 列出结果集 |
| GET | `/api/pinn/presets` | 列出预计算结果组 |
| GET | `/api/models/{model}/{run}/meta` | 获取元数据 |
| GET | `/api/models/{model}/{run}/metrics` | 获取误差指标 |
| GET | `/api/models/{model}/{run}/field` | 获取单帧热力图 |
| GET | `/api/compare/{epoch}` | PINN/FNO 对比数据 |
| POST | `/api/pinn/solve` | 提交在线求解 |
| GET | `/api/pinn/jobs/{job_id}` | 查询任务状态与进度 |
| DELETE | `/api/pinn/jobs/{job_id}` | 取消训练任务 |

## 在线求解链路

```
用户修改参数 → 点击"提交在线求解"
  → POST /api/pinn/solve
    → find_cached_pinn_run() 扫描全部历史结果
    → 命中？直接返回
    → 未命中？spawn 后台线程 → subprocess: python Hea0.1.19.py
      → 训练循环 → export_web_bundle() → web_exports/
  → 前端每 3 秒轮询进度
  → 完成 → 自动加载结果至演示面板
```

## 论文信息

- 论文题目：流体模拟自动化平台
- 真实验证对象：二维热方程
- 学术性来源：PINN 与 FNO 的统一任务对比分析
- 系统设计亮点：非侵入式接入、统一结果标准、参数化回放、在线求解闭环
