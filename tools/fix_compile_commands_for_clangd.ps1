param(
    [string]$InputFile = "compile_commands.json",
    [string]$OutputFile = ""
)

if (-not $OutputFile) { $OutputFile = $InputFile }

$json = Get-Content $InputFile -Encoding UTF8 -Raw | ConvertFrom-Json

$count = 0
foreach ($entry in $json) {
    $cmd = $entry.command
    $orig = $cmd

    # -external:I<path> → -isystem<path>
    $cmd = $cmd -replace '-external:I', '-isystem'
    # -external:W0 → remove
    $cmd = $cmd -replace '-external:W0', ''
    # -Qstd:c++20 → -std=c++20
    $cmd = $cmd -replace '-Qstd:c\+\+20', '-std=c++20'
    # -Qstd:c++17 → -std=c++17
    $cmd = $cmd -replace '-Qstd:c\+\+17', '-std=c++17'
    # -MDd → remove (MSVC debug runtime, not needed for analysis)
    $cmd = $cmd -replace '-MDd', ''

    # Collapse multiple spaces
    $cmd = $cmd -replace '\s+', ' '

    $entry.command = $cmd
    if ($orig -ne $cmd) { $count++ }
}

$json | ConvertTo-Json -Depth 10 | Set-Content $OutputFile -Encoding UTF8
Write-Output "Fixed $count entries. Written to $OutputFile"
