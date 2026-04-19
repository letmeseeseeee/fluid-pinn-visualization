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
