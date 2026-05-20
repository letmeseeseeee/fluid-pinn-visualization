from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from web_export import ExportPayload, export_prediction_bundle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class HeatBasisFamily:
    grid_size: int
    max_mode: int
    nu: float
    dt: float
    device: torch.device

    def __post_init__(self) -> None:
        x = torch.linspace(0.0, 1.0, self.grid_size, device=self.device)
        y = torch.linspace(0.0, 1.0, self.grid_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        self.coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)

        modes = []
        basis = []
        eigenvalues = []
        coeff_scale = []
        for m in range(1, self.max_mode + 1):
            for n in range(1, self.max_mode + 1):
                modes.append((m, n))
                basis.append(torch.sin(math.pi * m * grid_x) * torch.sin(math.pi * n * grid_y))
                eigenvalues.append(self.nu * (math.pi ** 2) * (m * m + n * n))
                coeff_scale.append(1.0 / (m + n))

        self.modes = modes
        self.basis = torch.stack(basis, dim=0)
        self.eigenvalues = torch.tensor(eigenvalues, dtype=torch.float32, device=self.device)
        self.coeff_scale = torch.tensor(coeff_scale, dtype=torch.float32, device=self.device)

    def sample_batch(self, batch_size: int, t_max: float) -> tuple[torch.Tensor, torch.Tensor]:
        coeffs = torch.randn(batch_size, len(self.modes), device=self.device) * self.coeff_scale
        times = torch.rand(batch_size, 1, device=self.device) * t_max
        decay_now = torch.exp(-times * self.eigenvalues.unsqueeze(0))
        decay_next = torch.exp(-(times + self.dt) * self.eigenvalues.unsqueeze(0))
        current = torch.einsum("bk,khw->bhw", coeffs * decay_now, self.basis)
        target = torch.einsum("bk,khw->bhw", coeffs * decay_next, self.basis)
        return current, target

    def sample_rollout_batch(self, batch_size: int, rollout_steps: int, t_max: float) -> torch.Tensor:
        max_start = max(0.0, t_max - rollout_steps * self.dt)
        coeffs = torch.randn(batch_size, len(self.modes), device=self.device) * self.coeff_scale
        start_times = torch.rand(batch_size, 1, device=self.device) * max_start
        offsets = torch.arange(rollout_steps + 1, device=self.device, dtype=torch.float32).view(1, -1, 1)
        times = start_times.unsqueeze(1) + offsets * self.dt
        decay = torch.exp(-times * self.eigenvalues.view(1, 1, -1))
        sequence = torch.einsum("btk,khw->bthw", coeffs.unsqueeze(1) * decay, self.basis)
        return sequence

    def exact_trajectory(self, steps: int) -> np.ndarray:
        basis11 = self.basis[self.modes.index((1, 1))]
        eigen = self.eigenvalues[self.modes.index((1, 1))]
        times = torch.arange(steps + 1, device=self.device, dtype=torch.float32) * self.dt
        decay = torch.exp(-times * eigen)
        seq = decay[:, None, None] * basis11[None, ...]
        return seq.detach().cpu().numpy().astype(np.float32)


def build_model(args: argparse.Namespace):
    try:
        from neuralop.models import FNO
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "neuraloperator is required. Install it with `pip install neuraloperator`."
        ) from exc

    return FNO(
        n_modes=(args.fno_modes, args.fno_modes),
        hidden_channels=args.width,
        in_channels=3 if args.use_coord_channels else 1,
        out_channels=1,
        n_layers=args.depth,
        positional_embedding=None if args.use_coord_channels else "grid",
    )


def make_model_input(current: torch.Tensor, coords: torch.Tensor | None, use_coord_channels: bool) -> torch.Tensor:
    if not use_coord_channels:
        return current.unsqueeze(1)
    if coords is None:
        raise ValueError("coords are required when use_coord_channels=True")
    return torch.cat([current.unsqueeze(1), coords], dim=1)


def step_model(model, current: torch.Tensor, coords: torch.Tensor | None, args: argparse.Namespace) -> torch.Tensor:
    model_input = make_model_input(current, coords, args.use_coord_channels)
    delta = model(model_input).squeeze(1)
    next_field = current + delta if args.residual_output else delta
    next_field[:, 0, :] = 0.0
    next_field[:, -1, :] = 0.0
    next_field[:, :, 0] = 0.0
    next_field[:, :, -1] = 0.0
    return next_field


def rollout(model, family: HeatBasisFamily, initial: np.ndarray, steps: int) -> np.ndarray:
    field = torch.tensor(initial, dtype=torch.float32, device=family.device).unsqueeze(0)
    coords = family.coords.expand(field.shape[0], -1, -1, -1) if family.coords is not None else None
    outputs = [field[0].detach().cpu().numpy()]
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            field = step_model(model, field, coords, family.args)
            outputs.append(field[0].detach().cpu().numpy())
    return np.stack(outputs, axis=0).astype(np.float32)


def export_run(model, family: HeatBasisFamily, root_dir: Path, step: int, args: argparse.Namespace) -> Path:
    gt_short = family.exact_trajectory(args.short_steps)
    gt_long = family.exact_trajectory(args.long_steps)
    pred_short = rollout(model, family, gt_short[0], args.short_steps)
    pred_long = rollout(model, family, gt_long[0], args.long_steps)
    payload = ExportPayload(
        prediction_short=pred_short,
        gt_short=gt_short,
        prediction_long=pred_long,
        gt_long=gt_long,
        model_name="fno",
        epoch=step,
        dt=family.dt,
        dx=1.0 / (family.grid_size - 1),
        dy=1.0 / (family.grid_size - 1),
        extra_meta={
            "nu": family.nu,
            "architecture": "neuraloperator.FNO",
            "modes": args.fno_modes,
            "width": args.width,
            "depth": args.depth,
            "basis_modes": args.basis_modes,
            "rollout_steps": args.rollout_steps,
            "residual_output": args.residual_output,
            "use_coord_channels": args.use_coord_channels,
            "source": "train_fno_neuralop.py",
        },
    )
    return export_prediction_bundle(root_dir=root_dir, payload=payload)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    root_dir = Path(args.root_dir).resolve()
    ckpt_dir = root_dir / "output" / "fno"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    family = HeatBasisFamily(
        grid_size=args.grid_size,
        max_mode=args.basis_modes,
        nu=args.nu,
        dt=args.dt,
        device=device,
    )
    family.args = args
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_steps)
    loss_meter = 0.0
    start = time.time()

    print(
        f"[fno-official] device={device} train_steps={args.train_steps} "
        f"batch_size={args.batch_size} modes={args.fno_modes} width={args.width} depth={args.depth} "
        f"rollout={args.rollout_steps} residual={args.residual_output} coords={args.use_coord_channels}",
        flush=True,
    )
    for step in range(1, args.train_steps + 1):
        model.train()
        rollout_steps = max(1, int(args.rollout_steps))
        if rollout_steps == 1:
            current, target = family.sample_batch(args.batch_size, t_max=args.long_steps * args.dt)
            coords = family.coords.expand(args.batch_size, -1, -1, -1) if args.use_coord_channels else None
            pred = step_model(model, current, coords, args)
            loss = F.mse_loss(pred, target)
        else:
            sequence = family.sample_rollout_batch(args.batch_size, rollout_steps, t_max=args.long_steps * args.dt)
            coords = family.coords.expand(args.batch_size, -1, -1, -1) if args.use_coord_channels else None
            field = sequence[:, 0]
            losses = []
            for offset in range(1, rollout_steps + 1):
                field = step_model(model, field, coords, args)
                losses.append(F.mse_loss(field, sequence[:, offset]))
            loss = torch.stack(losses).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_meter += float(loss.item())
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start
            avg_loss = loss_meter / max(1, args.log_every if step > 1 else 1)
            print(
                f"[fno-official] step={step:06d}/{args.train_steps:06d} "
                f"loss={avg_loss:.6e} lr={scheduler.get_last_lr()[0]:.3e} elapsed={elapsed:.1f}s",
                flush=True,
            )
            loss_meter = 0.0

        if step % args.export_every == 0 or step == args.train_steps:
            out_dir = export_run(model, family, root_dir, step, args)
            ckpt_path = ckpt_dir / f"fno_heat_step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[fno-official] exported {out_dir}", flush=True)
            print(f"[fno-official] checkpoint {ckpt_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an official neuraloperator FNO model for the 2D heat equation.")
    parser.add_argument("--root-dir", default=".", help="Project root containing web_export.py")
    parser.add_argument("--device", default="", help="Training device, e.g. cuda or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--short-steps", type=int, default=100)
    parser.add_argument("--long-steps", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=108000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--basis-modes", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--use-coord-channels", action="store_true")
    parser.add_argument("--no-residual-output", dest="residual_output", action="store_false")
    parser.set_defaults(residual_output=True)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--export-every", type=int, default=12000)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
