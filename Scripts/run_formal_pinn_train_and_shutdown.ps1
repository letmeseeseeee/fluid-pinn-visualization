$ErrorActionPreference = "Stop"

$root = "F:\graduate_project"
$python = Join-Path $root ".venv-cuda\Scripts\python.exe"
$logsDir = Join-Path $root "output\train_logs"
$runName = "formal_pinn_local_rtx3050_epoch108000_safeexport"
$logPath = Join-Path $logsDir "$runName.log"
$summaryPath = Join-Path $logsDir "$runName.summary.txt"
$resultDir = Join-Path $root "web_exports\pinn\$runName"

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$cmd = @(
    $python
    (Join-Path $root "Hea0.1.19.py")
    "--equation", "heat"
    "--nx", "101"
    "--ny", "101"
    "--nu", "1.0"
    "--dt", "1e-5"
    "--short-steps", "100"
    "--long-steps", "100"
    "--epochs", "108000"
    "--learning-rate", "1e-5"
    "--seed", "50976"
    "--run-name", $runName
    "--export-interval", "12000"
    "--run-initial-test"
)

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start formal training: $runName" | Set-Content -Path $logPath -Encoding UTF8
"command: $($cmd -join ' ')" | Add-Content -Path $logPath -Encoding UTF8

Push-Location $root
try {
    $pythonScript = Join-Path $root "Hea0.1.19.py"
    $cmdLine = "`"$python`" `"$pythonScript`" --equation heat --nx 101 --ny 101 --nu 1.0 --dt 1e-5 --short-steps 100 --long-steps 100 --epochs 108000 --learning-rate 1e-5 --seed 50976 --run-name $runName --export-interval 12000 --run-initial-test >> `"$logPath`" 2>&1"
    cmd /c $cmdLine
    $nativeExitCode = $LASTEXITCODE

    if ($nativeExitCode -ne 0) {
        throw "Native training process exited with code $nativeExitCode"
    }

    if (-not (Test-Path (Join-Path $resultDir "metrics.json"))) {
        throw "Training finished but metrics.json was not found in $resultDir"
    }

    & $python -c @"
import json
from pathlib import Path
root = Path(r"$resultDir")
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
summary = []
summary.append(f"run={root.name}")
summary.append(f"model={meta.get('model')}")
summary.append(f"epoch={meta.get('epoch')}")
summary.append(f"grid={meta['grid']['width']}x{meta['grid']['height']}")
summary.append(f"dt={meta['grid']['dt']}")
summary.append(f"short_rmse={metrics['short']['rmse']}")
summary.append(f"short_mse={metrics['short']['mse']}")
summary.append(f"long_rmse={metrics['long']['rmse']}")
summary.append(f"long_mse={metrics['long']['mse']}")
summary.append(f"frames_short={len(metrics['short']['frame_rmse'])}")
summary.append(f"frames_long={len(metrics['long']['frame_rmse'])}")
Path(r"$summaryPath").write_text("\n".join(summary), encoding="utf-8")
print("\n".join(summary))
"@ 1>> $logPath 2>> $logPath

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] training completed successfully, shutdown in 120 seconds." | Add-Content -Path $logPath -Encoding UTF8
    shutdown /s /t 120 /c "PINN正式训练已完成，系统将在120秒后自动关机。若需取消，请运行 shutdown /a"
}
catch {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] training failed: $($_.Exception.Message)" | Add-Content -Path $logPath -Encoding UTF8
    throw
}
finally {
    Pop-Location
}
