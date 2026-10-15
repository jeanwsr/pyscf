param(
    [int]$Repeats = 20,
    [switch]$ThreadJitter
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$tempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$runRoot = Join-Path $tempRoot 'pyscf-cabs-stress'
$venv = Join-Path $runRoot '.venv'

Remove-Item -LiteralPath $runRoot -Recurse -Force -ErrorAction SilentlyContinue
python -m venv $venv
if ($LASTEXITCODE -ne 0) { throw 'Unable to create stress venv' }
$python = Join-Path $venv 'Scripts\python.exe'

$wheels = @(Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'dist') -Filter 'pyscf-*.whl' -File)
if ($wheels.Count -ne 1) { throw "Expected exactly one wheel, found $($wheels.Count)" }
& $python -m pip install $wheels[0].FullName 'pytest<9' 2>&1 | Select-Object -Last 5
if ($LASTEXITCODE -ne 0) { throw 'Stress venv setup failed' }

# Wheels do not ship tests. Overlay only the mp test directory onto the
# installed package so production modules and DLLs still come from the wheel.
$installedPackage = Join-Path $venv 'Lib\site-packages\pyscf'
$src = Join-Path $RepoRoot 'pyscf\mp\test'
$dst = Join-Path $installedPackage 'mp\test'
New-Item -ItemType Directory -Path $dst -Force | Out-Null
Copy-Item -Path (Join-Path $src '*') -Destination $dst -Recurse -Force
foreach ($d in @($installedPackage, (Join-Path $installedPackage 'mp'), $dst)) {
    $marker = Join-Path $d '__init__.py'
    if (-not (Test-Path -LiteralPath $marker)) {
        New-Item -ItemType File -Path $marker | Out-Null
    }
}

$pyscfConfig = Join-Path $runRoot '.pyscf_conf.py'
@(
    'pbc_tools_pbc_fft_engine = "NUMPY+BLAS"',
    'scf_hf_SCF_mute_chkfile = True'
) | Set-Content -LiteralPath $pyscfConfig -Encoding utf8
$env:PYSCF_CONFIG_FILE = $pyscfConfig
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYSCF_EXT_PATH -ErrorAction SilentlyContinue
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'

$sitePackages = Split-Path $installedPackage -Parent
Push-Location $sitePackages
$fails = 0
$summary = @()
try {
    for ($i = 1; $i -le $Repeats; $i++) {
        if ($ThreadJitter) {
            $n = Get-Random -Minimum 1 -Maximum 5
            $env:OMP_NUM_THREADS = "$n"
            $env:OPENBLAS_NUM_THREADS = "$n"
        }
        else {
            $env:OMP_NUM_THREADS = '1'
            $env:OPENBLAS_NUM_THREADS = '1'
        }
        $log = New-TemporaryFile
        & $python -m pytest (Join-Path $dst 'test_cabs.py') -q > $log 2>&1
        $exitCode = $LASTEXITCODE
        $status = 'ok'
        if ($exitCode -ne 0) {
            $status = 'FAIL'
            $fails++
            Get-Content -LiteralPath $log
        }
        Remove-Item -LiteralPath $log -Force
        $line = "iter $i/$Repeats [threads=$($env:OMP_NUM_THREADS)]: $status"
        Write-Host $line
        $summary += $line
    }
}
finally {
    Pop-Location
    Remove-Item Env:PYSCF_CONFIG_FILE -ErrorAction SilentlyContinue
}

$report = @('## CABS stress summary', '') + $summary + @('', "Failed iterations: $fails / $Repeats")
if ($env:GITHUB_STEP_SUMMARY) {
    $report | Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Encoding utf8
}
Write-Host "stress: $fails/$Repeats failed"
if ($fails -gt 0) { throw "$fails of $Repeats stress iterations failed" }
