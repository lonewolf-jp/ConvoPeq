Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$srcDir = Join-Path $repoRoot "src"
$script:PrintedRgFallbackInfo = $false

if (-not (Test-Path $srcDir)) {
    Write-Error "Target directory not found: $srcDir"
    exit 2
}

function Invoke-RgLines {
    param(
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string[]]$Targets
    )

    $rgAvailable = $null -ne (Get-Command rg -ErrorAction SilentlyContinue)
    $results = @()
    foreach ($target in $Targets) {
        if (-not (Test-Path -LiteralPath $target)) {
            continue
        }

        if ($rgAvailable) {
            $out = & rg -n --hidden -e $Pattern -- $target 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $results += $out
            }
            continue
        }

        $candidateFiles = @()
        if (Test-Path -LiteralPath $target -PathType Container) {
            $candidateFiles = Get-ChildItem -LiteralPath $target -Recurse -File -ErrorAction SilentlyContinue
        }
        else {
            $candidateFiles = Get-Item -LiteralPath $target -ErrorAction SilentlyContinue
        }

        foreach ($file in $candidateFiles) {
            $matches = Select-String -LiteralPath $file.FullName -Pattern $Pattern -CaseSensitive -ErrorAction SilentlyContinue
            foreach ($match in $matches) {
                $results += ("{0}:{1}:{2}" -f $match.Path, $match.LineNumber, $match.Line)
            }
        }
    }

    if ((-not $rgAvailable) -and (-not $script:PrintedRgFallbackInfo)) {
        Write-Host "INFO: 'rg' not found. Falling back to Select-String scan."
        $script:PrintedRgFallbackInfo = $true
    }

    return $results
}

$violations = New-Object System.Collections.Generic.List[object]
$warnings = New-Object System.Collections.Generic.List[object]

function Add-Failure {
    param([string]$RuleId, [string]$Message, [string]$Line)
    $violations.Add([PSCustomObject]@{ Rule = $RuleId; Message = $Message; Line = $Line }) | Out-Null
}

function Add-Warn {
    param([string]$RuleId, [string]$Message, [string]$Line)
    $warnings.Add([PSCustomObject]@{ Rule = $RuleId; Message = $Message; Line = $Line }) | Out-Null
}

# ------------------------------------------------------------
# list.md 15.1 危険コメント
# ------------------------------------------------------------
$dangerCommentPattern = '(//|/\*).*\b(TODO|FIXME|workaround|quick\s+fix|just\s+for\s+now|temporary)\b'
$dangerMatches = Invoke-RgLines -Pattern $dangerCommentPattern -Targets @($srcDir)
foreach ($m in $dangerMatches) {
    Add-Failure -RuleId '15.1' -Message 'Danger comment token detected' -Line $m
}

# ------------------------------------------------------------
# list.md 2.x RT safety checks (主要RTファイル)
# ------------------------------------------------------------
$rtFiles = @(
    (Join-Path $srcDir 'audioengine\AudioEngine.Processing.AudioBlock.cpp'),
    (Join-Path $srcDir 'audioengine\AudioEngine.Processing.BlockDouble.cpp'),
    (Join-Path $srcDir 'audioengine\AudioEngine.Processing.DSPCoreDouble.cpp'),
    (Join-Path $srcDir 'audioengine\AudioEngine.Processing.DSPCoreFloat.cpp'),
    (Join-Path $srcDir 'audioengine\AudioEngine.Processing.DSPCoreIO.cpp'),
    (Join-Path $srcDir 'convolver\ConvolverProcessor.Runtime.cpp'),
    (Join-Path $srcDir 'eqprocessor\EQProcessor.Processing.cpp')
)

$rtAllocPattern = 'new\b|delete\b|malloc|calloc|realloc|free\b|_aligned_malloc|mkl_malloc|mkl_free|setSize\s*\(|resize\s*\(|reserve\s*\(|push_back\s*\(|emplace_back\s*\('
$rtAllocMatches = Invoke-RgLines -Pattern $rtAllocPattern -Targets $rtFiles
foreach ($m in $rtAllocMatches) {
    # コメントによる誤検知を除外（実装行のみ）
    if ($m -notmatch '://|//|/\*') {
        Add-Failure -RuleId '2.1' -Message 'Allocation-like token in RT file' -Line $m
    }
}

$rtLockPattern = 'std::mutex|std::lock_guard|std::unique_lock|std::shared_mutex|std::condition_variable|CriticalSection|ScopedLock|WaitableEvent|std::future|std::promise|std::async'
$rtLockMatches = Invoke-RgLines -Pattern $rtLockPattern -Targets $rtFiles
foreach ($m in $rtLockMatches) {
    Add-Failure -RuleId '2.2' -Message 'Lock/blocking token in RT file' -Line $m
}

$rtExPattern = '\bthrow\b|\btry\b|\bcatch\b|__try|__except'
$rtExMatches = Invoke-RgLines -Pattern $rtExPattern -Targets $rtFiles
foreach ($m in $rtExMatches) {
    Add-Failure -RuleId '2.3' -Message 'Exception token in RT file' -Line $m
}

$rtIoPattern = 'Logger::writeToLog|\bDBG\s*\(|printf\s*\(|std::cout|MessageManager|File::|std::ifstream|std::ofstream'
$rtIoMatches = Invoke-RgLines -Pattern $rtIoPattern -Targets $rtFiles
foreach ($m in $rtIoMatches) {
    Add-Failure -RuleId '2.5' -Message 'Logging/I-O token in RT file' -Line $m
}

# ------------------------------------------------------------
# list.md 3.1 dot-call atomic 禁止（既存スクリプトを再利用）
# ------------------------------------------------------------
$dotCallScript = Join-Path $PSScriptRoot 'check-src-atomic-dotcall.ps1'
& powershell -NoProfile -ExecutionPolicy Bypass -File $dotCallScript
if ($LASTEXITCODE -ne 0) {
    Add-Failure -RuleId '3.1' -Message 'Strict atomic dot-call scan failed' -Line $dotCallScript
}

# ------------------------------------------------------------
# list.md 7.1 delete/free 監査（deferred reclaim は許可）
# ------------------------------------------------------------
$ownershipPattern = '\bdelete\b|\bfree\s*\('
$ownershipMatches = Invoke-RgLines -Pattern $ownershipPattern -Targets @($srcDir)
foreach ($m in $ownershipMatches) {
    # '= delete;' (copy/move禁止宣言) は対象外
    if ($m -match '=\s*delete\s*;') {
        continue
    }

    # コメント行のヒットは対象外
    if ($m -match ':\d+:\s*(//|/\*)') {
        continue
    }

    $allowed = (
        $m -match 'enqueueDeferredDeleteNonRt' -or
        $m -match 'retireDSP\(' -or
        $m -match 'retireStereoConvolver' -or
        $m -match 'unique_ptr' -or
        $m -match '直接 delete 禁止' -or
        $m -match 'static\s+void\s+delete[A-Za-z0-9_]*\s*\(' -or
        $m -match 'ISRRTExecution\.cpp:\d+:\s*delete\[\]\s+buf;' -or
        $m -match 'AudioEngine\.Commit\.cpp:\d+:\s*delete\s+static_cast<AudioEngine::PublicationIntent\*>\(ptr\);' -or
        $m -match 'AudioEngine\.Commit\.cpp:\d+:\s*delete\s+intent;'
    )

    if (-not $allowed) {
        Add-Warn -RuleId '7.1' -Message 'delete/free token needs manual ownership review' -Line $m
    }
}

# ------------------------------------------------------------
# list.md 1.1.5 UI staging direct setter 監査（warning）
# ------------------------------------------------------------
$audioEngineHeader = Join-Path $srcDir 'audioengine\AudioEngine.h'
$uiMutatePattern = 'uiConvolverProcessor\.set[A-Za-z0-9_]+\('
$uiMutateMatches = Invoke-RgLines -Pattern $uiMutatePattern -Targets @($audioEngineHeader)
foreach ($m in $uiMutateMatches) {
    $isSanctionedWrapper = (
        $m -match 'uiConvolverProcessor\.setMix\(' -or
        $m -match 'uiConvolverProcessor\.setSmoothingTime\('
    )

    if ($isSanctionedWrapper) {
        continue
    }

    Add-Warn -RuleId '1.1.5' -Message 'UI staging setter detected; ensure rebuild/snapshot path is used' -Line $m
}

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
Write-Host "list.md compliance summary"
Write-Host "  Failures: $($violations.Count)"
Write-Host "  Warnings: $($warnings.Count)"

if ($warnings.Count -gt 0) {
    Write-Host ""
    Write-Host "Warnings (manual review):"
    foreach ($w in $warnings) {
        Write-Host "  [$($w.Rule)] $($w.Message) :: $($w.Line)"
    }
}

if ($violations.Count -gt 0) {
    Write-Host ""
    Write-Host "Failures:"
    foreach ($v in $violations) {
        Write-Host "  [$($v.Rule)] $($v.Message) :: $($v.Line)"
    }
    exit 1
}

Write-Host "list.md compliance checks passed (with possible warnings)."
exit 0
