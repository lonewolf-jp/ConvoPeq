param(
    [Parameter(Mandatory=$true)]
    [string]$LogPath
)

$lines = Get-Content $LogPath

# XRUN timeline
$xrunLines = $lines | Select-String '\[XRUN'
$f = $xrunLines[0]
$l = $xrunLines[-1]
$fu = if ($f -match 'Us=(\d+)') { [int]$matches[1] } else { 0 }
$lu = if ($l -match 'Us=(\d+)') { [int]$matches[1] } else { 0 }
$spanSec = ($lu - $fu) / 1000000.0
$spanMin = $spanSec / 60.0
$density = $xrunLines.Count / $spanMin

Write-Host "=== XRUN OVERVIEW ==="
Write-Host "Total XRUNs: $($xrunLines.Count)"
Write-Host "First XRUN Us: $fu"
Write-Host "Last XRUN Us: $lu"
Write-Host "XRUN span: $($spanSec.ToString('F1'))s ($($spanMin.ToString('F1'))min)"
Write-Host "XRUN density: $($density.ToString('F1'))/min"

# Key events timeline
Write-Host "`n=== KEY EVENTS TIMELINE ==="
$keys = @('BUILD_PHASE', 'CBSUMMARY', 'PageFault surge', 'COEFF_AUTH', 'ADAPTIVE_SWITCH', '[PUBLISH]', '[DSP_TIMING')
$specialXrun = '32.', '11.0', '10.3'

foreach ($line in $lines) {
    $keep = $false
    foreach ($k in $keys) {
        if ($line.Contains($k)) { $keep = $true; break }
    }
    if (-not $keep) {
        foreach ($s in $specialXrun) {
            if ($line.Contains('[XRUN') -and $line.Contains($s)) { $keep = $true; break }
        }
    }
    if ($keep) {
        write-Host "  $line"
    }
}

# Timeline sequences: CBSUMMARY intervalMax values
Write-Host "`n=== CBSUMMARY intervalMax TREND ==="
$cbLines = $lines | Select-String '\[CBSUMMARY\]'
$i = 0
foreach ($cb in $cbLines) {
    if ($cb -match 'intervalMax=([\d.]+)ms') {
        $val = [double]$matches[1]
        $seq = if ($cb -match 'Seq=(\d+)') { $matches[1] } else { '?' }
        $mark = if ($val -ge 9) { ' <-- HIGH' } elseif ($val -ge 8) { ' <-- elevated' } else { '' }
        Write-Host "  Seq=$seq intervalMax=$($val.ToString('F3'))ms$mark"
        $i++
    }
}
Write-Host "  Total CBSUMMARY entries: $i"
