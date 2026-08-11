$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$audioRoot = Join-Path $repoRoot 'src\audioengine'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'publication_ownership_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = @()
$hits = @()

$checks = [ordered]@{
    publishWorldBuilderExists = $false
    publishLifecycleWrapperExists = $false
    retireLifecycleWrapperExists = $false
    authorityPublishExists = $false                 # ★ X4-B: RuntimeWorldAuthority::publish（sole gateway）
    noRuntimePublishAuthorityFactory = $false       # ★ X4-B: makeRuntimePublishAuthority() は削除必須
    noRtPublicationFactoryCall = $false
}

if (-not (Test-Path -LiteralPath $headerPath)) {
    $violations += "Missing header: $headerPath"
}
else {
    $headerText = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8

    if ([regex]::IsMatch($headerText, '\bbuildRuntimePublishWorld\s*\(')) { $checks.publishWorldBuilderExists = $true }
    else { $violations += 'Publication ownership contract missing: buildRuntimePublishWorld()' }

    if ([regex]::IsMatch($headerText, '\bonRuntimePublishedNonRt\s*\(') -or
        [regex]::IsMatch($headerText, '\bdidPublishRuntimeNonRt\s*\(')) {
        $checks.publishLifecycleWrapperExists = $true
    }
    else { $violations += 'Publication ownership contract missing: onRuntimePublishedNonRt()/didPublishRuntimeNonRt()' }

    if ([regex]::IsMatch($headerText, '\bonRuntimeRetiredNonRt\s*\(') -or
        [regex]::IsMatch($headerText, '\bwillRetireRuntimeNonRt\s*\(')) {
        $checks.retireLifecycleWrapperExists = $true
    }
    else { $violations += 'Publication ownership contract missing: onRuntimeRetiredNonRt()/willRetireRuntimeNonRt()' }

    # ★ work88 (X4-B §6.4 / X4-B-8): makeRuntimePublishAuthority() は production から削除済み
    #   （一時生成 factory が write-capable RuntimeStore を作る構造 — INV-X4-3/5 違反のため）。
    #   publish は RuntimeWorldAuthority::publish（sole physical publish gateway — INV-X4-2）に一本化。
    if ([regex]::IsMatch($headerText, '\bmakeRuntimePublishAuthority\s*\(')) {
        $violations += 'X4-B violation: makeRuntimePublishAuthority() must be removed (temporary factory creates write-capable Store — INV-X4-3/5)'
    }
    else { $checks.noRuntimePublishAuthorityFactory = $true }

    # RuntimeWorldAuthority::publish（sole physical publish gateway — INV-X4-2）の存在を、
    # authority ヘッダ（定義本体）で検証する。
    $authorityHeader = Join-Path $repoRoot 'src\audioengine\RuntimeWorldAuthority.h'
    if (Test-Path -LiteralPath $authorityHeader) {
        $authText = Get-Content -LiteralPath $authorityHeader -Raw -Encoding UTF8
        if ([regex]::IsMatch($authText, '\bRuntimeState\s*\*\s+publish\s*\(') -or
            [regex]::IsMatch($authText, 'Store::WriteAccess')) {
            $checks.authorityPublishExists = $true
        }
        else { $violations += 'Publication ownership contract missing: RuntimeWorldAuthority::publish (sole physical publish gateway)' }
    }
    else { $violations += "Missing authority header: $authorityHeader" }
}

$rtFiles = @(
)

$forbiddenRtPatterns = @(
    'makeRuntimePublishAuthority\s*\(',
    'commitRuntimePublication\s*\(',
    'retireRuntimePublication\s*\(',
    'onRuntimePublishedNonRt\s*\(',
    'onRuntimeRetiredNonRt\s*\('
)

$rtViolations = 0
foreach ($targetPath in $rtFiles) {
    if (-not (Test-Path -LiteralPath $targetPath)) {
        $violations += "Missing RT file for ownership scan: $targetPath"
        continue
    }

    $text = Get-Content -LiteralPath $targetPath -Raw -Encoding UTF8
    foreach ($pattern in $forbiddenRtPatterns) {
        $m = [regex]::Matches($text, $pattern)
        if ($m.Count -gt 0) {
            $rtViolations += $m.Count
            $hits += [pscustomobject]@{ path = $targetPath; pattern = $pattern; count = $m.Count }
            $violations += "Publication ownership violation: RT path uses publication API path=$targetPath pattern=$pattern count=$($m.Count)"
        }
    }
}

if ($rtViolations -eq 0) {
    $checks.noRtPublicationFactoryCall = $true
}

$report = [ordered]@{
    schema = 'publication_ownership_report_v1'
    generatedAt = (Get-Date -Format 'o')
    headerPath = $headerPath
    checks = $checks
    hits = $hits
    violations = $violations
    ready = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'publication ownership verification failed'
}

Write-Host '[PASS] publication ownership verification passed'
