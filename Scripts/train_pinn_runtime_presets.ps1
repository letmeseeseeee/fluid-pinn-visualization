$ErrorActionPreference = "Stop"

$root = "F:\graduate_project"
$python = Join-Path $root ".venv-cuda\Scripts\python.exe"
$script = Join-Path $root "Hea0.1.19.py"
$logDir = Join-Path $root "output\preset_runtime_logs"
$summaryDir = Join-Path $root "output\preset_runtime_summaries"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null

$presets = @(
    @{
        RunName = "preset_low_nu"
        Nu = "0.5"
        Dt = "1e-5"
        Epochs = "12000"
    },
    @{
        RunName = "preset_base"
        Nu = "1.0"
        Dt = "1e-5"
        Epochs = "12000"
    },
    @{
        RunName = "preset_high_nu"
        Nu = "1.5"
        Dt = "1e-5"
        Epochs = "12000"
    },
    @{
        RunName = "preset_small_dt"
        Nu = "1.0"
        Dt = "5e-6"
        Epochs = "12000"
    }
)

Push-Location $root
try {
    foreach ($preset in $presets) {
        $runName = $preset.RunName
        $logPath = Join-Path $logDir "$runName.log"
        $summaryPath = Join-Path $summaryDir "$runName.summary.txt"
        $resultDir = Join-Path $root "web_exports\pinn\$runName"

        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start $runName" | Set-Content -Path $logPath -Encoding UTF8
        $cmdLine = "`"$python`" `"$script`" --equation heat --nx 101 --ny 101 --nu $($preset.Nu) --dt $($preset.Dt) --short-steps 100 --long-steps 100 --epochs $($preset.Epochs) --learning-rate 1e-5 --seed 50976 --run-name $runName --export-interval 12000 --run-initial-test >> `"$logPath`" 2>&1"
        cmd /c $cmdLine
        if ($LASTEXITCODE -ne 0) {
            throw "Preset $runName failed with exit code $LASTEXITCODE"
        }

        if (-not (Test-Path (Join-Path $resultDir "metrics.json"))) {
            throw "Preset $runName finished but metrics.json is missing"
        }

        & $python -c @"
import json
from pathlib import Path
root = Path(r"$resultDir")
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
summary = []
summary.append(f"run={root.name}")
summary.append(f"epoch={meta.get('epoch')}")
summary.append(f"nu={meta.get('extra',{}).get('runtime_config',{}).get('nu')}")
summary.append(f"dt={meta['grid']['dt']}")
summary.append(f"short_rmse={metrics['short']['rmse']}")
summary.append(f"short_mse={metrics['short']['mse']}")
summary.append(f"long_rmse={metrics['long']['rmse']}")
summary.append(f"long_mse={metrics['long']['mse']}")
Path(r"$summaryPath").write_text("\n".join(summary), encoding="utf-8")
print("\n".join(summary))
"@
    }
}
finally {
    Pop-Location
}
