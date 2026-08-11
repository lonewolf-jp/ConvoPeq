# 一時検証スクリプト（CI スクリプトの FAIL 状況一括確認用）
$ErrorActionPreference = 'Continue'
$scripts = @(
    'isr-verify-v5-retire-authority-lane.ps1',
    'isr-verify-publication-single-path.ps1',
    'isr-verify-facade-bypass.ps1',
    'isr-verify-self-contained-world.ps1',
    'check-authority-boundary.ps1'
)
foreach ($s in $scripts) {
    Write-Host "===== $s ====="
    pwsh -NoProfile -ExecutionPolicy Bypass -File ".github/scripts/$s" 2>&1 | Select-Object -Last 8
    Write-Host "EXIT=$LASTEXITCODE"
}
