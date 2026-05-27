param(
    [string]$SharedMetricsPath,
    [string]$SplitMetricsPath,
    [string]$OutputPath = (Join-Path $PSScriptRoot "..\..\doc\work\ISR_Shared_EpochDomain_Shared_vs_Split_Comparison_2026-05-27.md")
)

$ErrorActionPreference = 'Stop'

function Read-Metrics {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    $raw = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $null
    }

    return $raw | ConvertFrom-Json
}

function Format-MetricValue {
    param($Value)

    if ($null -eq $Value) {
        return '未測定'
    }

    return [string]$Value
}

$shared = Read-Metrics -Path $SharedMetricsPath
$split = Read-Metrics -Path $SplitMetricsPath

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('# ISR Shared vs Split Comparison') | Out-Null
$lines.Add('') | Out-Null
$lines.Add(('作成日: {0}' -f (Get-Date -Format 'yyyy-MM-dd'))) | Out-Null
$lines.Add('対象: R10 latency / jitter / reclaim burst 比較表') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('---') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('## 1. 比較軸') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('| 軸 | shared | split | 判定コメント |') | Out-Null
$lines.Add('| --- | --- | --- | --- |') | Out-Null
$lines.Add(('| latency | {0} | {1} | 5分窓の中央値 / P95 を比較 |' -f (Format-MetricValue $shared.latencyMs), (Format-MetricValue $split.latencyMs))) | Out-Null
$lines.Add(('| jitter | {0} | {1} | callback jitter を比較 |' -f (Format-MetricValue $shared.jitterMs), (Format-MetricValue $split.jitterMs))) | Out-Null
$lines.Add(('| reclaim burst | {0} | {1} | retire burst のピーク / 継続時間を比較 |' -f (Format-MetricValue $shared.reclaimBurst), (Format-MetricValue $split.reclaimBurst))) | Out-Null
$lines.Add(('| shutdown drain | {0} | {1} | bounded completion への影響を比較 |' -f (Format-MetricValue $shared.shutdownDrainMs), (Format-MetricValue $split.shutdownDrainMs))) | Out-Null
$lines.Add('') | Out-Null
$lines.Add('---') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('## 2. 判定') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('| 判定 | 条件 |') | Out-Null
$lines.Add('| --- | --- |') | Out-Null
$lines.Add('| Go(shared継続) | shared が split に対して全軸で劣後しない |') | Out-Null
$lines.Add('| Go(split移行) | split が 1軸以上で安定性優位、他軸が許容内 |') | Out-Null
$lines.Add('| No-Go | いずれの方式も A1 / A2 / A4 を満たさない |') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('---') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('## 3. 判定記録') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('- 日時:') | Out-Null
$lines.Add('- 判定:') | Out-Null
$lines.Add('- 理由:') | Out-Null
$lines.Add('- 追跡チケット:') | Out-Null

Set-Content -LiteralPath $OutputPath -Value ($lines -join "`n") -Encoding UTF8
Write-Host "[PASS] comparison written: $OutputPath"
