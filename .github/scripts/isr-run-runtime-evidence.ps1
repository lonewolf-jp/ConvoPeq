param(
    # 既定は合成しない。合成が必要な場合だけ明示的に有効化し、
    # 生成物には provenance=synthesized / status=not_measured を付ける。
    [switch]$AllowSynthesis,
    # アプリを動かす時間。実測 artifact（HB 証跡・retire・shutdown）は
    # この時間内の publish から生成される。
    [int]$RunMilliseconds = 12000
)

$ErrorActionPreference = 'Stop'

# ★ B-11: アプリの実測出力は run 単位のディレクトリへ書く。
#   evidence/ 直下を削除・上書きしないため、generate / seed の実測証跡と衝突しない。
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceRoot = Join-Path $repoRoot "evidence"

$runtimeRunId = [string]$env:CONVO_ISR_RUNTIME_RUN_ID
if ([string]::IsNullOrWhiteSpace($runtimeRunId)) {
    throw "CONVO_ISR_RUNTIME_RUN_ID is required for strict runtime evidence mode."
}

$runRoot = Join-Path (Join-Path $evidenceRoot "runs") $runtimeRunId
# アプリは current_path()/evidence に書くため、run 直下に evidence/ を作る。
$appEvidenceRoot = Join-Path $runRoot "evidence"
if (-not (Test-Path $appEvidenceRoot)) {
    New-Item -ItemType Directory -Path $appEvidenceRoot -Force | Out-Null
}

# 同時実行の排他。既にロックがあれば、同じ runId へ二重に書かない。
$lockPath = Join-Path $runRoot ".lock"
try {
    $lockStream = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew,
                                          [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
}
catch {
    throw "Another runtime evidence run is already writing to $runRoot. Refusing to overwrite."
}

try {
    $buildRoot = Join-Path $repoRoot "build"
    if (-not (Test-Path $buildRoot)) {
        throw "Build directory not found: $buildRoot"
    }

    $allCandidates = Get-ChildItem -Path $buildRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'ConvoPeq*.exe' }

    if ($null -eq $allCandidates -or $allCandidates.Count -eq 0) {
        throw "No runnable ConvoPeq executable found under build/. strict runtime evidence mode requires executable output."
    }

    # Debug ビルドを優先する。Release は exportEvidence の isRelease 分岐で manifest のみを
    # 書き出すため、artifact を要求する検証には足りない。
    $debugCandidates = $allCandidates | Where-Object { $_.FullName -match '[\\/]Debug[\\/]' }
    if ($debugCandidates) {
        $exe = ($debugCandidates | Sort-Object LastWriteTimeUtc -Descending)[0].FullName
    }
    else {
        $exe = ($allCandidates | Sort-Object LastWriteTimeUtc -Descending)[0].FullName
        Write-Host "[WARN] No Debug build of ConvoPeq found under build/. Using latest executable: $exe (live evidence mode works in Release)."
    }

    Write-Host "[INFO] Strict runtime evidence mode: launching $exe (cwd=$runRoot, runMs=$RunMilliseconds)"
    # ★ B-11: アプリを実際に動かし、HB 証跡・retire・shutdown を実測で生成させる。
    #   以前の --verify-runtime-evidence はエンジンを起動せず、Release では manifest だけを
    #   書いていた（measured=0）。
    $process = Start-Process -FilePath $exe `
        -ArgumentList "--cli-run", "--cli-exit-ms", "$RunMilliseconds", "--verify-runtime-evidence-live" `
        -WorkingDirectory $runRoot -PassThru -WindowStyle Hidden
    # アプリは終了時に evidence を書き出す。実測モードは実行時間 + 終了処理
    # （この環境では約 30 秒）を待つ必要があるため、余裕を持って待つ。
    $waitSeconds = [math]::Max(60, [math]::Ceiling($RunMilliseconds / 1000.0) + 90)
    try {
        Wait-Process -Id $process.Id -Timeout $waitSeconds -ErrorAction Stop
    }
    catch {
        Write-Host "[WARN] Runtime process did not exit in time. Forcing stop after evidence collection window."
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }

    $manifestPath = Join-Path $appEvidenceRoot "evidence_manifest.json"
    if (-not (Test-Path $manifestPath)) {
        throw "Runtime evidence manifest not found after execution: $manifestPath"
    }

    $manifestRaw = Get-Content -Path $manifestPath -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($manifestRaw)) {
        throw "Runtime evidence manifest is empty: $manifestPath"
    }

    $manifest = $manifestRaw | ConvertFrom-Json
    if ($null -eq $manifest) {
        throw "Failed to parse runtime evidence manifest: $manifestPath"
    }

    if (-not ($manifest.PSObject.Properties.Name -contains "generationMode")) {
        throw "Manifest missing generationMode field"
    }
    if ([string]$manifest.generationMode -ne "runtime") {
        throw "Strict runtime evidence requires generationMode=runtime (actual=$($manifest.generationMode))"
    }
    if (-not ($manifest.PSObject.Properties.Name -contains "runtimeRunId")) {
        throw "Manifest missing runtimeRunId field"
    }
    if ([string]$manifest.runtimeRunId -ne $runtimeRunId) {
        throw "runtimeRunId mismatch. manifest=$($manifest.runtimeRunId) expected=$runtimeRunId"
    }

    $nowNs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() * 1000000

    # 合成 artifact は「実測ではない」ことが分かる形でのみ作る。
    function New-SynthesizedArtifactObject {
        param(
            [Parameter(Mandatory = $true)][string]$ArtifactName,
            [Parameter(Mandatory = $true)][long]$GeneratedAtNs,
            [Parameter(Mandatory = $true)][string]$RunId
        )

        $schema = switch ($ArtifactName) {
            "closure_graph.json"          { "closure_graph_v1" }
            "mutation_fault_trace.json"   { "mutation_fault_trace_v1" }
            "payload_tier_report.json"    { "payload_tier_report_v1" }
            "hb_graph_trace.json"         { "hb_trace_v1" }
            "hb_violation_report.json"    { "hb_violation_report_v1" }
            "shutdown_trace.json"         { "shutdown_trace_v1" }
            "retire_latency_report.json"  { "retire_latency_report_v1" }
            "retire_timeline.json"        { "retire_timeline_v2" }
            "runtime_budget_report.json"  { "runtime_budget_report_v1" }
            "recovery_trace.json"         { "recovery_trace_v1" }
            default                       { "unknown_v1" }
        }

        return @{
            artifact        = $ArtifactName
            schema          = $schema
            provenance      = "synthesized"
            status          = "not_measured"
            measured        = $false
            synthesizedBy   = "isr-run-runtime-evidence.ps1"
            sourceRunId     = $RunId
            generatedAtNs   = $GeneratedAtNs
            runId           = $RunId
            reason          = "the application did not emit this artifact in this run"
        }
    }

    $requiredArtifacts = @(
        "closure_graph.json",
        "mutation_fault_trace.json",
        "payload_tier_report.json",
        "hb_graph_trace.json",
        "hb_violation_report.json",
        "shutdown_trace.json",
        "retire_latency_report.json",
        "retire_timeline.json",
        "runtime_budget_report.json",
        "recovery_trace.json"
    )

    $missing = @()
    foreach ($requiredArtifact in $requiredArtifacts) {
        $requiredPath = Join-Path $appEvidenceRoot $requiredArtifact
        if (Test-Path $requiredPath) {
            continue
        }

        if (-not $AllowSynthesis) {
            $missing += $requiredArtifact
            continue
        }

        Write-Host "[WARN] Synthesizing missing artifact (not a measurement): $requiredArtifact"
        New-SynthesizedArtifactObject -ArtifactName $requiredArtifact -GeneratedAtNs $nowNs -RunId $runtimeRunId |
            ConvertTo-Json -Depth 100 | Set-Content -Path $requiredPath -Encoding UTF8
    }

    if ($missing.Count -gt 0) {
        throw ("Runtime evidence is incomplete. Missing measured artifacts: " + ($missing -join ", ") +
               ". Re-run with -AllowSynthesis only when a synthesized (not measured) artifact is acceptable.")
    }

    # manifest と実体を一致させる。既存 artifact の provenance は書き換えない。
    $artifactFiles = @(Get-ChildItem -Path $appEvidenceRoot -File -Filter "*.json" |
                       Where-Object { $_.Name -ne "evidence_manifest.json" })
    $artifactNames = @($artifactFiles | ForEach-Object { $_.Name })
    $artifactDetail = @()

    foreach ($artifactFile in $artifactFiles) {
        $provenance = "unknown"
        $status = "unknown"
        $measured = $false
        try {
            $parsed = Get-Content -Path $artifactFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($parsed.PSObject.Properties.Name -contains "provenance") { $provenance = [string]$parsed.provenance }
            if ($parsed.PSObject.Properties.Name -contains "status")     { $status = [string]$parsed.status }
            $measured = ($provenance -eq "runtime" -and $status -ne "not_measured" -and $status -ne "not_run")
        }
        catch {
            $provenance = "unparsable"
        }

        $artifactDetail += @{
            name       = $artifactFile.Name
            provenance = $provenance
            status     = $status
            measured   = $measured
        }
    }

    $manifest | Add-Member -NotePropertyName artifacts -NotePropertyValue $artifactNames -Force
    $manifest | Add-Member -NotePropertyName artifactDetail -NotePropertyValue $artifactDetail -Force
    $manifest | Add-Member -NotePropertyName evidenceRunRoot -NotePropertyValue ("runs/" + $runtimeRunId + "/evidence") -Force
    if (-not ($manifest.PSObject.Properties.Name -contains "generatedAtNs") -or [long]$manifest.generatedAtNs -le 0) {
        $manifest | Add-Member -NotePropertyName generatedAtNs -NotePropertyValue $nowNs -Force
    }

    $existingGenerators = @()
    if ($manifest.PSObject.Properties.Name -contains "generator" -and
        -not [string]::IsNullOrWhiteSpace([string]$manifest.generator)) {
        $existingGenerators += [string]$manifest.generator
    }
    $existingGenerators += "isr-run-runtime-evidence.ps1"
    $manifest | Add-Member -NotePropertyName additionalGenerators -NotePropertyValue $existingGenerators -Force

    $manifest | ConvertTo-Json -Depth 100 | Set-Content -Path $manifestPath -Encoding UTF8

    $measuredCount = @($artifactDetail | Where-Object { $_.measured }).Count
    Write-Host "[PASS] Runtime evidence collected (runId=$runtimeRunId, artifacts=$($artifactNames.Count), measured=$measuredCount)"
    Write-Host "[INFO] Evidence root: $runRoot"
}
finally {
    $lockStream.Close()
    $lockStream.Dispose()
    Remove-Item -Path $lockPath -Force -ErrorAction SilentlyContinue
}
