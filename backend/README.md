# Backend (FastAPI)

## 作用

后端负责统一结果访问与参数预设读取，不再承担在线求解任务调度。当前接口服务于两类前端场景：

- PINN 参数化结果回放
- PINN / FNO 历史结果对比展示

## 启动

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

## 主要接口

- `GET /api/models`
- `GET /api/models/{model}/runs`
- `GET /api/models/{model}/{run}/meta`
- `GET /api/models/{model}/{run}/metrics`
- `GET /api/models/{model}/{run}/field?kind=prediction_short&t=0`
- `GET /api/compare/{epoch}?left=pinn&right=fno&t=0&split=short`
- `GET /api/pinn/presets`

## 说明

- `GET /api/pinn/presets` 返回 PINN 预计算结果组及其参数，用于参数回填与结果映射。
- 平台不再通过后端触发新的求解任务，所有参数化演示均基于已导出的正式结果。
- FNO 结果以离线对照数据形式提供，用于统一展示与实验比较。
