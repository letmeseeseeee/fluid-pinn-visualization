# 流体模拟自动化平台（Heat Equation Online Demo）

当前版本目标：
- 只做**热方程**；
- 第一版只开放 **PINN** 在线任务流程；
- FNO 先保留对比接口占位，后续再接入。

## 1) 你现在可以做什么

1. 前端输入参数，提交在线仿真任务；
2. 后端在几秒内生成结果数据并写入统一导出目录；
3. 前端展示热力图时间步播放 + MSE/RMSE 曲线。

## 2) 统一导出格式

在线任务成功后会落盘：

- `web_exports/pinn/epoch_xxxxxx/prediction_short.npy`
- `web_exports/pinn/epoch_xxxxxx/gt_short.npy`
- `web_exports/pinn/epoch_xxxxxx/diff_short.npy`
- `web_exports/pinn/epoch_xxxxxx/prediction_long.npy`
- `web_exports/pinn/epoch_xxxxxx/gt_long.npy`
- `web_exports/pinn/epoch_xxxxxx/diff_long.npy`
- `web_exports/pinn/epoch_xxxxxx/meta.json`
- `web_exports/pinn/epoch_xxxxxx/metrics.json`

## 3) 后端启动

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

## 4) 前端启动

```bash
cd frontend
npm install
npm run dev
```

## 5) 参数范围说明（第一版）

后端对参数进行约束：
- `nx, ny: [21, 201]`
- `nu: [0.01, 2.0]`
- `dt: (0, 1e-3]`
- `short_steps: [10, 300]`
- `long_steps: [20, 600]`

并加入二维显式离散稳定性条件检查（避免发散和超时）。

## 6) FNO 后续接入说明

当前已保留 `/api/compare` 占位。后续你下载并接入 FNO 后，把结果按同一导出格式写入：
- `web_exports/fno/epoch_xxxxxx/...`
即可与 PINN 自动对比展示。

## 7) Windows 一键启动（推荐你答辩演示时使用）

在项目根目录双击：
- `start_all.bat`

它会自动：
1. 安装/检查后端依赖
2. 安装前端依赖（首次）
3. 启动后端（8000）
4. 启动前端并自动打开浏览器（5173）

> 以后项目代码怎么改，前后端入口不变的话，都可以继续用这个脚本快速预览。

## 8) 云计算服务思路（可行）

这个思路完全可行：
- 前端保留本地交互和展示；
- 计算任务放到云端（GPU/CPU服务）；
- 当前本地后端接口以后可逐步替换成云端API；
- 导出格式仍保持 `web_exports` 的统一结构，便于本地/云端对齐。
