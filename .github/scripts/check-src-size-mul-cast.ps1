Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$srcDir = Join-Path $repoRoot "src"

if (-not (Test-Path -LiteralPath $srcDir)) {
    Write-Error "Target directory not found: $srcDir"
    exit 2
}

# CodeQL cpp/integer-multiplication-cast-to-long の再発防止（src限定）
# 例: static_cast<size_t>(a * b)
$pattern = 'static_cast\s*<\s*(?:std::)?size_t\s*>\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*\)'

$matches = @()
$rg = Get-Command rg -ErrorAction SilentlyContinue

if ($null -ne $rg) {
    $out = & rg -n --pcre2 -e $pattern -- $srcDir 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) {
        $matches = @($out)
    }
}
else {
    $files = Get-ChildItem -LiteralPath $srcDir -Recurse -File -Include *.h, *.hpp, *.hh, *.cpp, *.cc, *.cxx
    foreach ($file in $files) {
        $lineHits = Select-String -LiteralPath $file.FullName -Pattern $pattern -CaseSensitive -ErrorAction SilentlyContinue
        foreach ($hit in $lineHits) {
            $matches += ("{0}:{1}:{2}" -f $hit.Path, $hit.LineNumber, $hit.Line.Trim())
        }
    }
    Write-Host "INFO: 'rg' not found. Falling back to Select-String scan."
}

if ($matches.Count -gt 0) {
    Write-Host "Detected forbidden size cast multiplication pattern(s) in src:"
    foreach ($m in $matches) {
        Write-Host "  $m"
    }
    Write-Host ""
    Write-Host "Policy: avoid static_cast<size_t>(a * b). Promote before multiply, e.g. static_cast<size_t>(a) * static_cast<size_t>(b)."
    exit 1
}

Write-Host "size-mul-cast guard passed (src only)."
exit 0
