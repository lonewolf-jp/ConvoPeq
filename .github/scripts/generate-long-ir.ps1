param(
    [string]$OutputPath = "c:\VSC_Project\ConvoPeq\sampledata\synthetic_long_ir_20s.wav",
    [int]$SampleRate = 48000,
    [double]$DurationSec = 20.0,
    [int]$Channels = 2,
    [int]$Seed = 20260523
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($SampleRate -le 0) { throw "SampleRate must be > 0." }
if ($DurationSec -le 0) { throw "DurationSec must be > 0." }
if ($Channels -lt 1 -or $Channels -gt 2) { throw "Channels must be 1 or 2." }

$totalFrames = [int][Math]::Floor($SampleRate * $DurationSec)
if ($totalFrames -lt 1) { throw "Duration too short." }

$dataBytes = [int64]$totalFrames * [int64]$Channels * 2
if ($dataBytes -gt [int64]::MaxValue) { throw "Data size overflow." }

$dir = Split-Path -Path $OutputPath -Parent
if (-not [string]::IsNullOrWhiteSpace($dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

$fs = [System.IO.File]::Open($OutputPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
try {
    $bw = New-Object System.IO.BinaryWriter($fs, [System.Text.Encoding]::ASCII, $false)
    try {
        $byteRate = $SampleRate * $Channels * 2
        $blockAlign = $Channels * 2
        $riffSize = 36 + [int]$dataBytes

        # RIFF header
        $bw.Write([System.Text.Encoding]::ASCII.GetBytes("RIFF"))
        $bw.Write([int]$riffSize)
        $bw.Write([System.Text.Encoding]::ASCII.GetBytes("WAVE"))

        # fmt chunk (PCM 16-bit)
        $bw.Write([System.Text.Encoding]::ASCII.GetBytes("fmt "))
        $bw.Write([int]16)
        $bw.Write([int16]1)
        $bw.Write([int16]$Channels)
        $bw.Write([int]$SampleRate)
        $bw.Write([int]$byteRate)
        $bw.Write([int16]$blockAlign)
        $bw.Write([int16]16)

        # data chunk
        $bw.Write([System.Text.Encoding]::ASCII.GetBytes("data"))
        $bw.Write([int]$dataBytes)

        $rng = [System.Random]::new($Seed)

        # 反射パターン（秒）
        $reflections = @(0.003, 0.007, 0.013, 0.021, 0.034, 0.055, 0.089, 0.144, 0.233)
        $reflectionGain = @(0.58, 0.44, 0.34, 0.27, 0.21, 0.16, 0.12, 0.09, 0.06)

        $leftOffset = 0
        $rightOffset = [int]($SampleRate * 0.00035) # 0.35msの軽微な左右差

        for ($n = 0; $n -lt $totalFrames; $n++) {
            $t = $n / [double]$SampleRate

            # 20秒で十分減衰するような包絡
            $env = [Math]::Exp(-3.2 * $t)

            $x = 0.0
            if ($n -eq 0) { $x = 1.0 }

            for ($i = 0; $i -lt $reflections.Count; $i++) {
                $ri = [int]([Math]::Round($reflections[$i] * $SampleRate))
                if ($n -eq $ri) {
                    $x += $reflectionGain[$i]
                }
            }

            # 長尺テスト用途の弱いノイズテール
            $noise = (($rng.NextDouble() * 2.0) - 1.0) * 0.03 * $env

            $left = $x + $noise
            $right = $x + ((($rng.NextDouble() * 2.0) - 1.0) * 0.03 * $env)

            # 軽いチャンネル差（時間差由来の擬似効果）
            if ($n -eq $leftOffset) { $left += 0.02 }
            if ($n -eq $rightOffset) { $right += 0.02 }

            if ($left -gt 1.0) { $left = 1.0 }
            if ($left -lt -1.0) { $left = -1.0 }
            if ($right -gt 1.0) { $right = 1.0 }
            if ($right -lt -1.0) { $right = -1.0 }

            $l16 = [int16][Math]::Round($left * 32767.0)
            $r16 = [int16][Math]::Round($right * 32767.0)

            if ($Channels -eq 1) {
                $bw.Write($l16)
            }
            else {
                $bw.Write($l16)
                $bw.Write($r16)
            }
        }

        $bw.Flush()
    }
    finally {
        $bw.Dispose()
    }
}
finally {
    $fs.Dispose()
}

Write-Output "Generated: $OutputPath"
Write-Output "SampleRate=$SampleRate, DurationSec=$DurationSec, Channels=$Channels, Frames=$totalFrames"
