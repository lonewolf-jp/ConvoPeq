# Test script for audit report verification

Write-Host "Testing audit report verification (P11, P12, P14, P15)..."
$failed = $false
$reports = @("p11_audit.md", "p12_audit.md", "p14_audit.md", "p15_audit.md")
$requiredFields = @("AUDIT_RESULT", "AUDIT_DATE", "AUDITOR", "CHECKED_SYMBOLS", "FINDINGS", "SEARCH_COMMANDS")

foreach ($report in $reports) {
    $path = ".github/audit/$report"
    if (-not (Test-Path $path)) {
        Write-Host "error::$report not found"
        $failed = $true
        continue
    }

    $content = Get-Content $path -Raw

    # 必須フィールドの存在確認
    foreach ($field in $requiredFields) {
        if ($content -notmatch "${field}:") {
            Write-Host "error::$report missing $field"
            $failed = $true
        }
    }

    # CHECKED_SYMBOLS が空でないこと
    if ($content -match "CHECKED_SYMBOLS:\s*$") {
        Write-Host "error::$report CHECKED_SYMBOLS is empty"
        $failed = $true
    }

    # SEARCH_COMMANDS が空でないこと
    if ($content -match "SEARCH_COMMANDS:\s*$") {
        Write-Host "error::$report SEARCH_COMMANDS is empty"
        $failed = $true
    }

    # AUDIT_RESULT が PASS であること
    if ($content -notmatch "AUDIT_RESULT: PASS") {
        Write-Host "error::$report AUDIT_RESULT is not PASS"
        $failed = $true
    }

    Write-Host "$report audit fields verified."
}

if ($failed) {
    Write-Host "One or more audit reports failed verification."
    exit 1
}
Write-Host "All audit reports passed verification."

Write-Host ""
Write-Host "Testing P15 tool outputs (non-empty)..."
$failed = $false
$outputs = @("p15_serena.txt", "p15_codegraph.txt")
foreach ($output in $outputs) {
    $path = ".github/audit/$output"
    if (-not (Test-Path $path)) {
        Write-Host "error::$output not found"
        $failed = $true
        continue
    }
    $item = Get-Item $path
    if ($item.Length -eq 0) {
        Write-Host "error::$output is empty"
        $failed = $true
    } else {
        Write-Host "$output size: $($item.Length) bytes"
    }
}
if ($failed) {
    Write-Host "P15 tool outputs verification failed."
    exit 1
}
Write-Host "All P15 tool outputs verified."

Write-Host ""
Write-Host "Testing P1 Phase1-B audit (optional)..."
$path = ".github/audit/p1_phase1b_audit.md"
if (Test-Path $path) {
    $content = Get-Content $path -Raw
    if ($content -notmatch "AUDIT_RESULT: PASS") {
        Write-Host "warning::P1 Phase1-B audit result is not PASS"
    } else {
        Write-Host "P1 Phase1-B audit verified."
    }
} else {
    Write-Host "P1 Phase1-B audit not yet created (Phase1-B not started)."
}
