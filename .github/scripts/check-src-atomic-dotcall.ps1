Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$targetDir = Join-Path $repoRoot "src"

if (-not (Test-Path $targetDir)) {
    Write-Error "Target directory not found: $targetDir"
    exit 2
}

$pattern = "\.(load|store|exchange)\s*\("
$extensions = @("*.h", "*.hpp", "*.hh", "*.cpp", "*.cxx", "*.cc")

function Remove-CommentsFromLine {
    param(
        [AllowEmptyString()][string]$Line,
        [Parameter(Mandatory = $true)][ref]$InBlockComment
    )

    $result = ""
    $i = 0
    while ($i -lt $Line.Length) {
        if ($InBlockComment.Value) {
            $end = $Line.IndexOf("*/", $i)
            if ($end -lt 0) {
                return $result
            }
            $InBlockComment.Value = $false
            $i = $end + 2
            continue
        }

        if ($i + 1 -lt $Line.Length) {
            $pair = $Line.Substring($i, 2)
            if ($pair -eq "//") {
                return $result
            }
            if ($pair -eq "/*") {
                $InBlockComment.Value = $true
                $i += 2
                continue
            }
        }

        $result += $Line[$i]
        $i++
    }

    return $result
}

$sourceFiles = foreach ($ext in $extensions) {
    Get-ChildItem -Path $targetDir -Recurse -File -Filter $ext
}

$violations = @()

foreach ($file in $sourceFiles) {
    $fullPath = [System.IO.Path]::GetFullPath($file.FullName)
    $repoPath = [System.IO.Path]::GetFullPath($repoRoot)
    if ($fullPath.StartsWith($repoPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relativePath = $fullPath.Substring($repoPath.Length).TrimStart([char[]]@('\', '/'))
    }
    else {
        $relativePath = $fullPath
    }
    $relativePath = $relativePath.Replace('\\', '/')
    $lines = Get-Content -Path $file.FullName -Encoding UTF8
    $inBlockComment = $false

    for ($index = 0; $index -lt $lines.Length; $index++) {
        $lineNumber = $index + 1
        $line = $lines[$index]
        $codeOnly = Remove-CommentsFromLine -Line $line -InBlockComment ([ref]$inBlockComment)

        if ([string]::IsNullOrWhiteSpace($codeOnly)) {
            continue
        }

        if ([System.Text.RegularExpressions.Regex]::IsMatch($codeOnly, $pattern)) {
            $violations += [PSCustomObject]@{
                File    = $relativePath
                Line    = $lineNumber
                Snippet = $line.Trim()
            }
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "Strict atomic dot-call violations detected: $($violations.Count)"
    Write-Host ""

    $grouped = $violations | Group-Object File
    foreach ($group in $grouped) {
        Write-Host $group.Name
        foreach ($item in $group.Group) {
            Write-Host "  $($item.Line): $($item.Snippet)"
        }
        Write-Host ""
    }

    Write-Host "Policy: use convo::consumeAtomic / publishAtomic / exchangeAtomic helpers instead of direct .load/.store/.exchange calls."
    exit 1
}

Write-Host "Strict atomic dot-call scan passed (src/**/*.h,*.hpp,*.cpp,*.cxx,*.cc)."
exit 0
