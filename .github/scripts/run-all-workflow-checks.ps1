# Run all .github/workflows/ script checks locally
# This replicates what the CI workflows execute.

$ErrorActionPreference = 'Stop'
$failed = $false

Write-Host "===== list-compliance.yml ====="
try {
    & "$PSScriptRoot\check-list-compliance.ps1"
    if ($LASTEXITCODE -ne 0) { throw "check-list-compliance.ps1 failed (exit=$LASTEXITCODE)" }
    Write-Host "PASS: check-list-compliance.ps1"
} catch {
    Write-Host "FAIL: check-list-compliance.ps1 : $_"
    $failed = $true
}

try {
    & "$PSScriptRoot\check-src-size-mul-cast.ps1"
    if ($LASTEXITCODE -ne 0) { throw "check-src-size-mul-cast.ps1 failed (exit=$LASTEXITCODE)" }
    Write-Host "PASS: check-src-size-mul-cast.ps1"
} catch {
    Write-Host "FAIL: check-src-size-mul-cast.ps1 : $_"
    $failed = $true
}

Write-Host "`n===== audioengine-lint.yml ====="
try {
    & "$PSScriptRoot\check-audioengine-lint.ps1"
    if ($LASTEXITCODE -ne 0) { throw "check-audioengine-lint.ps1 failed (exit=$LASTEXITCODE)" }
    Write-Host "PASS: check-audioengine-lint.ps1"
} catch {
    Write-Host "FAIL: check-audioengine-lint.ps1 : $_"
    $failed = $true
}

Write-Host "`n===== isr-authority-compliance.yml - Static checks ====="

# P1 Phase1-B: Legacy PublicationIntent
$legacyErrors = @()
if (Select-String -Path "$PSScriptRoot\..\..\src\**\*" -Pattern "struct PublicationIntent|struct PublicationLog" -Quiet) {
    $legacyErrors += "PublicationIntent/PublicationLog struct still exists"
}
$legacyFuncs = @("appendPublicationIntentForCommit", "drainPublicationLogForShutdown", "hasPublicationLogPending")
foreach ($f in $legacyFuncs) {
    if (Select-String -Path "$PSScriptRoot\..\..\src\**\*" -Pattern "\b$f\s*\(" -Quiet) {
        $legacyErrors += "Legacy function $f still exists"
    }
}
if ($legacyErrors.Count -gt 0) {
    Write-Host "FAIL: Phase1-B legacy checks"
    foreach ($e in $legacyErrors) { Write-Host "  $e" }
    $failed = $true
} else {
    Write-Host "PASS: Phase1-B legacy PublicationIntent checks"
}

# P14: Partial publication
if (Select-String -Path "$PSScriptRoot\..\..\src\**\*" -Pattern "publish\(generation|publish\(dsp" -Quiet) {
    Write-Host "FAIL: P14 partial publication detected"
    $failed = $true
} else {
    Write-Host "PASS: P14 partial publication check"
}

# P3: Direct EpochDomain::enqueueRetire (new code only - full scan as approximation)
$epochErrors = @()
$epochFiles = Select-String -Path "$PSScriptRoot\..\..\src\**\*" -Pattern "EpochDomain.*enqueueRetire" -SimpleMatch
foreach ($f in $epochFiles) {
    if ($f.Line -notmatch "Coordinator") {
        $epochErrors += "$($f.Filename):$($f.LineNumber) - $($f.Line.Trim())"
    }
}
if ($epochErrors.Count -gt 0) {
    Write-Host "FAIL: Direct EpochDomain::enqueueRetire (not through Coordinator)"
    foreach ($e in $epochErrors) { Write-Host "  $e" }
    $failed = $true
} else {
    Write-Host "PASS: P3 no direct EpochDomain::enqueueRetire"
}

# Audit reports
$auditFailed = $false
$auditDir = "$PSScriptRoot\..\audit"
$reports = @("p11_audit.md", "p12_audit.md", "p14_audit.md", "p15_audit.md")
$requiredFields = @("AUDIT_RESULT", "AUDIT_DATE", "AUDITOR", "CHECKED_SYMBOLS", "FINDINGS", "SEARCH_COMMANDS")
foreach ($report in $reports) {
    $path = "$auditDir\$report"
    if (-not (Test-Path $path)) {
        Write-Host "FAIL: $report not found"
        $auditFailed = $true
        continue
    }
    $content = Get-Content $path -Raw
    foreach ($field in $requiredFields) {
        if ($content -notmatch "${field}:") {
            Write-Host "FAIL: $report missing $field"
            $auditFailed = $true
        }
    }
}
if ($auditFailed) {
    Write-Host "FAIL: Audit report verification"
    $failed = $true
} else {
    Write-Host "PASS: Audit report verification"
}

Write-Host "`n===== isr-verification.yml - Python tools check ====="
$toolDir = "$PSScriptRoot\..\..\tools"

# Verifier entries: [0]=script name, [1]=arguments (empty string if none)
$verifiers = @(
    ,@("coverage_verifier.py", "")
    ,@("runtime_graph_authority_verifier.py", "--mode baseline")
    ,@("capture_session_id_verifier.py", "")
    ,@("identity_authority_verifier.py", "")
    ,@("engine_runtime_authority_verifier.py", "")
    ,@("non_authoritative_observe_verifier.py", "")
    ,@("retire_authority_verifier.py", "")
    ,@("snapshot_authority_usage_verifier.py", "")
    ,@("authority_source_count_verifier.py", "")
    ,@("publication_authority_verifier.py", "")
    ,@("generate_publication_manifest.py", "--verify --repo-root REPO_ROOT_PLACEHOLDER")
    ,@("detect_publication_mutation.py", "")
    ,@("retire_ordering_verifier.py", "")
    ,@("authority_inventory_verifier.py", "")
    ,@("authority_duplication_verifier.py", "")
    ,@("projection_origin_verifier.py", "")
    ,@("diagnostic_field_verifier.py", "")
)
foreach ($entry in $verifiers) {
    $v = $entry[0]
    $args = $entry[1]
    $vPath = "$toolDir\$v"
    if (-not (Test-Path $vPath)) {
        Write-Host "WARN: $v not found (optional)"
        continue
    }
    try {
        # Resolve REPO_ROOT_PLACEHOLDER to actual path
        $resolvedArgs = $args -replace "REPO_ROOT_PLACEHOLDER", (Resolve-Path "$PSScriptRoot\..\..")
        if ([string]::IsNullOrWhiteSpace($resolvedArgs)) {
            $result = & python "$vPath" 2>&1
        } else {
            $result = & python "$vPath" $resolvedArgs.Split(" ") 2>&1
        }
        if ($LASTEXITCODE -eq 0) {
            Write-Host "PASS: $v"
        } else {
            Write-Host "FAIL: $v $args (exit=$LASTEXITCODE)"
            $result | ForEach-Object { Write-Host "  $_" }
            $failed = $true
        }
    } catch {
        Write-Host "FAIL: $v threw exception: $_"
        $failed = $true
    }
}

# Check isr-8_1-close-policy.json
$policyPath = "$PSScriptRoot\..\isr-8_1-close-policy.json"
if (Test-Path $policyPath) {
    try {
        $policy = Get-Content $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($policy.schema -ne "isr_8_1_close_policy_v1") {
            Write-Host "FAIL: Unexpected schema: $($policy.schema)"
            $failed = $true
        } else {
            foreach ($field in @("owner","issue","rationale","expiry")) {
                if ([string]::IsNullOrWhiteSpace("$($policy.$field)")) {
                    Write-Host "FAIL: Missing field: $field"
                    $failed = $true
                }
            }
            if (-not $failed) { Write-Host "PASS: isr-8_1-close-policy.json" }
        }
    } catch {
        Write-Host "FAIL: isr-8_1-close-policy.json parse error: $_"
        $failed = $true
    }
} else {
    Write-Host "WARN: isr-8_1-close-policy.json not found (optional for local dev)"
}

Write-Host "`n===== Build & test check (isr-authority-compliance.yml) ====="
$buildDir = "$PSScriptRoot\..\..\build"
if (Test-Path "$buildDir\build.ninja") {
    # ISR Bridge Runtime 原則: 環境依存の失敗は早期に検出し、明確なエラーメッセージを表示する
    # VS dev command prompt が有効か確認（LIB 環境変数に VC のパスが含まれているか）
    $vsEnvActive = $false
    $libEnv = [Environment]::GetEnvironmentVariable("LIB", "Process")
    if ($libEnv -and $libEnv -match "Microsoft Visual Studio") {
        $vsEnvActive = $true
    }

    $testTargets = @("ISRSemanticValidationTests", "PartialPublicationRejectTests")
    foreach ($t in $testTargets) {
        $testExe = "$buildDir\Debug\$t.exe"
        $alreadyBuilt = Test-Path $testExe

        if ($alreadyBuilt -and -not $vsEnvActive) {
            # 既にビルド済みの場合は環境なしでも実行可能
            Write-Host "INFO: $t already built (skipping build, running existing)"
            & $testExe 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "PASS: $t execution"
            } else {
                Write-Host "FAIL: $t execution (exit=$LASTEXITCODE)"
                $failed = $true
            }
        } elseif ($vsEnvActive) {
            try {
                & cmake --build $buildDir --config Debug --target $t 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "PASS: Build target $t"
                    if (Test-Path $testExe) {
                        & $testExe 2>&1 | Out-Null
                        if ($LASTEXITCODE -eq 0) {
                            Write-Host "  PASS: $t execution"
                        } else {
                            Write-Host "  FAIL: $t execution (exit=$LASTEXITCODE)"
                            $failed = $true
                        }
                    }
                } else {
                    Write-Host "FAIL: Build target $t (exit=$LASTEXITCODE)"
                    $failed = $true
                }
            } catch {
                Write-Host "FAIL: Build target $t threw: $_"
                $failed = $true
            }
        } else {
            # ISR Bridge 原則: 前提未達は skip として扱い、環境セットアップ方法を案内する
            Write-Host "SKIP: $t - VS dev command prompt not detected."
            Write-Host "  To build and run tests, use:"
            Write-Host '    call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64'
            Write-Host "  Or use the VS Code task: 'Debug Build (cmd env retry)'"
        }
    }
} else {
    Write-Host "SKIP: Build directory not configured (run cmake configure first)"
    Write-Host "  Run: cmake -S . -B build -G 'Ninja Multi-Config' -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl"
}

if ($failed) {
    Write-Host "`n===== RESULT: SOME CHECKS FAILED ====="
    exit 1
}
Write-Host "`n===== RESULT: ALL CHECKS PASSED ====="
