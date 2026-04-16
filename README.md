# 流体模拟自动化平台（PINN/FNO 可视化）

本仓库当前在原有 `Hea0.1.19.py` 基础上新增了三层能力：

1. **统一导出格式**（`web_exports/...`）
2. **FastAPI 后端接口**（`backend/`）
3. **Vue 3 前端可视化骨架**（`frontend/`）

## 1) 统一导出格式

`Hea0.1.19.py` 的 `test()` 已接入导出逻辑，会导出到：

- `web_exports/pinn/epoch_xxxxxx/prediction_short.npy`
- `web_exports/pinn/epoch_xxxxxx/gt_short.npy`
- `web_exports/pinn/epoch_xxxxxx/diff_short.npy`
- `web_exports/pinn/epoch_xxxxxx/prediction_long.npy`
- `web_exports/pinn/epoch_xxxxxx/gt_long.npy`
- `web_exports/pinn/epoch_xxxxxx/diff_long.npy`
- `web_exports/pinn/epoch_xxxxxx/meta.json`
- `web_exports/pinn/epoch_xxxxxx/metrics.json`

未来接入 FNO 时，只需同样写到 `web_exports/fno/epoch_xxxxxx/` 即可复用前后端。

## 2) 后端（FastAPI）

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

接口：
- `GET /api/models`
- `GET /api/models/{model}/runs`
- `GET /api/models/{model}/{epoch}/meta`
- `GET /api/models/{model}/{epoch}/metrics`
- `GET /api/models/{model}/{epoch}/field?kind=prediction_short&t=0`
- `GET /api/compare/{epoch}?left=pinn&right=fno&split=short&t=0`

## 3) 前端（Vue3 + ECharts）

```bash
cd frontend
npm install
npm run dev
```

功能骨架：
- PINN/FNO 模型切换
- 左右热力图对比
- 误差热力图
- 时间步滑块 + 播放
- MSE / RMSE 曲线

## 4) 现有代码入口提示

当前训练和测试主入口仍然是：
- `python Hea0.1.19.py`

`test.py` 只是打印 torch 版本，不参与主流程。
