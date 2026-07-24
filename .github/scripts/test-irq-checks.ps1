# Test ISR authority compliance checks locally
$ErrorActionPreference = "Stop"
$sourceRoot = "C:\VSC_Project\ConvoPeq\src"

Write-Host "=== P1 Phase1-B: PublicationIntent remnants ==="
$errors = @()
$files = Get-ChildItem -Path $sourceRoot -Recurse -File -Include "*.h","*.hpp","*.cpp","*.cxx","*.cc"
foreach ($file in $files) {
    $content = Get-Content -Path $file.FullName -Raw -ErrorAction SilentlyContinue
    if ($content -and ($content -match "struct PublicationIntent|struct PublicationLog")) {
        $errors += "PublicationIntent/PublicationLog struct found in $($file.FullName)"
    }
}
if ($errors.Count -gt 0) {
    foreach ($e in $errors) { Write-Host "ERROR: $e" }
    exit 1
}
Write-Host "PASS: All PublicationIntent/PublicationLog remnants removed."

Write-Host "`n=== P14: Partial publication interfaces ==="
foreach ($file in $files) {
    $content = Get-Content -Path $file.FullName -Raw -ErrorAction SilentlyContinue
    if ($content -and ($content -match "publish\(generation|publish\(dsp")) {
        Write-Host "ERROR: Partial publication interface detected in $($file.FullName)"
        exit 1
    }
}
Write-Host "PASS: No partial publication interfaces detected."

Write-Host "`n=== All ISR authority static checks passed ==="
exit 0
