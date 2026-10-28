# OpenMP scalability runs of the particles sample (Step 1).
#
# Strong scaling: fixed problem size N, increasing number of threads T
#   speedup S(T) = T(1) / T(T), efficiency E(T) = S(T) / T          [Lez01 - Scalabilita', Amdahl]
# Weak scaling: N grows linearly with T (N = WeakBase * T)
#   ideal: constant time per frame                                   [Lez01 - Gustafson]
#
# Every run appends one CSV line (see particles.cpp, printReport) to the output file.
#
# Example:
#   .\scaling_omp.ps1 -Exe ..\..\..\build-particles\particles.exe -N 262144 -Out omp_scaling.csv

param(
    [string]   $Exe       = "particles.exe",
    [int]      $N         = 262144,
    [int]      $Iters     = 100,
    [int[]]    $Threads   = @(1, 2, 4, 8, 16),
    [string[]] $Schedules = @("static", "dynamic:64", "guided:16"),
    [int]      $WeakBase  = 16384,
    [string]   $Out       = "omp_scaling.csv"
)

$header = $null
$rows   = @()

function Invoke-Run([string] $kind, [string[]] $runArgs) {
    $lines = & $Exe -benchmark -backend=omp @runArgs
    if ($LASTEXITCODE -ne 0) { throw "run failed: $Exe $runArgs" }
    if (-not $script:header) {
        $script:header = ($lines | Where-Object { $_ -like "CSV_HEADER,*" } | Select-Object -First 1) -replace "^CSV_HEADER,", "scaling,"
    }
    $row = ($lines | Where-Object { $_ -like "CSV,*" } | Select-Object -First 1) -replace "^CSV,", "$kind,"
    Write-Host $row
    $script:rows += $row
}

foreach ($s in $Schedules) {
    foreach ($t in $Threads) {
        Invoke-Run "strong" @("-n=$N", "-i=$Iters", "-threads=$t", "-schedule=$s")
    }
}

foreach ($t in $Threads) {
    Invoke-Run "weak" @("-n=$($WeakBase * $t)", "-i=$Iters", "-threads=$t", "-schedule=$($Schedules[0])")
}

Set-Content -Path $Out -Value (@($header) + $rows) -Encoding utf8
Write-Host "Wrote $($rows.Count) runs to $Out"
