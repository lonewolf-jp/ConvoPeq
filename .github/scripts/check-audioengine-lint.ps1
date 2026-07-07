Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$targetDir = Join-Path $repoRoot "src\audioengine"

if (-not (Test-Path $targetDir)) {
    Write-Error "Target directory not found: $targetDir"
    exit 2
}

$rules = @(
    @{
        Id          = "LINT-AE-001"
        Description = "No zero-arg getRuntimeGraphState()/getEngineRuntimeState() in src/audioengine/*.cpp"
        Pattern     = "getRuntimeGraphState\(\)|getEngineRuntimeState\(\)"
    },
    @{
        Id          = "LINT-AE-002"
        Description = "No single-arg resolveCurrent/resolveFading DSP runtime publish calls in src/audioengine/*.cpp"
        Pattern     = "resolveCurrentDSPFromRuntimePublish\(\s*[^,\)]*\)|resolveFadingDSPFromRuntimePublish\(\s*[^,\)]*\)"
    },
    @{
        Id          = "LINT-AE-003"
        Description = "No direct crossfade atomic loads in src/audioengine/*.cpp"
        Pattern     = "dspCrossfadePending\.load\(|dspCrossfadeUseDryAsOld\.load\(|dspCrossfadeStartDelayBlocks\.load\("
    }
)

$warningRule = @{
    Id          = "LINT-AE-004"
    Description = "Warn when getRuntimePublishWorld() is called 3+ times in the same src/audioengine/*.cpp function"
    Pattern     = "getRuntimePublishWorld\(\)"
}

$strictRtFiles = @(
    "src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp",
    "src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp",
    "src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp",
    "src/convolver/ConvolverProcessor.Runtime.cpp"
)

$sourceRoot = Join-Path $repoRoot "src"
$sourceExtensions = @("*.h", "*.hpp", "*.hh", "*.cpp", "*.cxx", "*.cc")

if (-not (Test-Path $sourceRoot)) {
    Write-Error "Source root not found: $sourceRoot"
    exit 2
}

function Get-RelativePathCompat {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $rootFull = [System.IO.Path]::GetFullPath($Root)
    $pathFull = [System.IO.Path]::GetFullPath($Path)
    $getRelativePath = [System.IO.Path].GetMethod("GetRelativePath", [Type[]]@([string], [string]))
    if ($null -ne $getRelativePath) {
        return [System.IO.Path]::GetRelativePath($rootFull, $pathFull).Replace('\\', '/')
    }

    if ($pathFull.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $pathFull.Substring($rootFull.Length).TrimStart([char[]]@([char]'\', [char]'/')).Replace('\\', '/')
    }

    return $pathFull.Replace('\\', '/')
}

function Get-LineNumberFromIndex {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Content,
        [Parameter(Mandatory = $true)]
        [int]$Index
    )

    return 1 + [System.Text.RegularExpressions.Regex]::Matches($Content.Substring(0, $Index), "`n").Count
}

function Get-LineText {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Lines,
        [Parameter(Mandatory = $true)]
        [int]$LineNumber
    )

    if ($LineNumber -ge 1 -and $LineNumber -le $Lines.Length) {
        return $Lines[$LineNumber - 1].Trim()
    }

    return ""
}

function Add-RegexViolations {
    param(
        [Parameter(Mandatory = $true)]
        [ref]$Violations,
        [Parameter(Mandatory = $true)]
        [string]$Content,
        [Parameter(Mandatory = $true)]
        [object[]]$Lines,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,
        [Parameter(Mandatory = $true)]
        [string]$Pattern,
        [Parameter(Mandatory = $true)]
        [string]$RuleId,
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [scriptblock]$Filter = { param($lineText, $lineNumber, $match) return $true }
    )

    $regexMatches = [System.Text.RegularExpressions.Regex]::Matches(
        $Content,
        $Pattern,
        [System.Text.RegularExpressions.RegexOptions]::None
    )

    foreach ($match in $regexMatches) {
        $line = Get-LineNumberFromIndex -Content $Content -Index $match.Index
        $lineText = Get-LineText -Lines $Lines -LineNumber $line
        if (-not (& $Filter $lineText $line $match)) {
            continue
        }

        $Violations.Value += [PSCustomObject]@{
            Rule        = $RuleId
            Description = $Description
            File        = $RelativePath
            Line        = $line
            Snippet     = $lineText
        }
    }
}

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

function Test-IsLikelyFunctionSignature {
    param(
        [string]$Line
    )

    if ([string]::IsNullOrWhiteSpace($Line)) { return $false }

    $trimmed = $Line.Trim()
    if ($trimmed.StartsWith("//") -or $trimmed.StartsWith("/*") -or $trimmed.StartsWith("*") -or $trimmed.StartsWith("#")) { return $false }
    if ($trimmed -match '^(if|for|while|switch|catch|else|do|namespace|class|struct|enum)\b') { return $false }
    if ($trimmed.EndsWith(";")) { return $false }

    return ($trimmed -match '\)\s*(const\b|noexcept\b|override\b|final\b|\{|$)')
}

function Get-FunctionScopedWorldWarnings {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Content,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,
        [Parameter(Mandatory = $true)]
        [string]$Pattern,
        [Parameter(Mandatory = $true)]
        [string]$RuleId,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    $results = @()
    if ([string]::IsNullOrEmpty($Content)) {
        return $results
    }

    $Lines = @($Content -split "`r?`n")
    if ($Lines.Count -eq 0) {
        return $results
    }

    $braceDepth = 0
    $pendingSignatureLine = $null
    $currentFunction = $null

    for ($index = 0; $index -lt $Lines.Length; $index++) {
        $lineNumber = $index + 1
        $line = $Lines[$index]

        if ($braceDepth -eq 0 -and $null -eq $currentFunction -and (Test-IsLikelyFunctionSignature -Line $line)) {
            $pendingSignatureLine = $lineNumber
        }

        if ($null -ne $pendingSignatureLine -and $null -eq $currentFunction -and $braceDepth -eq 0 -and $line.Contains("{")) {
            $currentFunction = [PSCustomObject]@{
                StartLine = $pendingSignatureLine
                StartText = ($Lines[$pendingSignatureLine - 1]).Trim()
                Count     = 0
            }
            $pendingSignatureLine = $null
        }

        if ($null -ne $currentFunction) {
            $regexMatches = [System.Text.RegularExpressions.Regex]::Matches(
                $line,
                $Pattern,
                [System.Text.RegularExpressions.RegexOptions]::None
            )
            $currentFunction.Count += $regexMatches.Count
        }

        foreach ($ch in $line.ToCharArray()) {
            switch ($ch) {
                '{' { $braceDepth++ }
                '}' {
                    $braceDepth--
                    if ($braceDepth -lt 0) { $braceDepth = 0 }
                    if ($braceDepth -eq 0 -and $null -ne $currentFunction) {
                        if ($currentFunction.Count -ge 3) {
                            $results += [PSCustomObject]@{
                                Rule        = $RuleId
                                Description = $Description
                                File        = $RelativePath
                                Line        = $currentFunction.StartLine
                                Snippet     = $currentFunction.StartText
                                Count       = $currentFunction.Count
                            }
                        }
                        $currentFunction = $null
                    }
                }
            }
        }
    }

    return $results
}

function Get-SwitchEngineOrderViolations {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Content,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $results = @()
    $signaturePattern = "void\\s+ConvolverProcessor::switchEngineOnMessageThread\\s*\\(\\s*StereoConvolver\\*\\s+newEngine\\s*\\)\\s*noexcept"
    $signatureMatch = [System.Text.RegularExpressions.Regex]::Match($Content, $signaturePattern)
    if (-not $signatureMatch.Success) {
        return $results
    }

    $bodyStart = $Content.IndexOf('{', $signatureMatch.Index + $signatureMatch.Length)
    if ($bodyStart -lt 0) {
        return $results
    }

    $depth = 0
    $bodyEnd = -1
    for ($index = $bodyStart; $index -lt $Content.Length; $index++) {
        switch ($Content[$index]) {
            '{' { $depth++ }
            '}' {
                $depth--
                if ($depth -eq 0) {
                    $bodyEnd = $index
                    break
                }
            }
        }
    }

    if ($bodyEnd -lt 0) {
        return $results
    }

    $body = $Content.Substring($bodyStart, $bodyEnd - $bodyStart + 1)
    $advanceMatch = [System.Text.RegularExpressions.Regex]::Match($body, "advanceEpoch\\s*\\(")
    $retireMatch = [System.Text.RegularExpressions.Regex]::Match($body, "retireStereoConvolver\\s*\\(")

    if (-not $advanceMatch.Success -or -not $retireMatch.Success -or $advanceMatch.Index -gt $retireMatch.Index) {
        $line = Get-LineNumberFromIndex -Content $Content -Index $signatureMatch.Index
        $results += [PSCustomObject]@{
            Rule        = "LINT-AE-007"
            Description = "switchEngineOnMessageThread() must call advanceEpoch() before retireStereoConvolver()"
            File        = $RelativePath
            Line        = $line
            Snippet     = "void ConvolverProcessor::switchEngineOnMessageThread(StereoConvolver* newEngine) noexcept"
        }
    }

    return $results
}

function Add-RequiredPatternViolation {
    param(
        [Parameter(Mandatory = $true)]
        [ref]$Violations,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,
        [Parameter(Mandatory = $true)]
        [string]$Pattern,
        [Parameter(Mandatory = $true)]
        [string]$RuleId,
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [Parameter(Mandatory = $true)]
        [string]$Snippet
    )

    if (-not (Test-Path $FilePath)) {
        return
    }

    $content = Get-Content -Path $FilePath -Raw -Encoding UTF8
    if ([string]::IsNullOrEmpty($content)) {
        return
    }

    if (-not [System.Text.RegularExpressions.Regex]::IsMatch($content, $Pattern)) {
        $Violations.Value += [PSCustomObject]@{
            Rule        = $RuleId
            Description = $Description
            File        = $RelativePath
            Line        = 1
            Snippet     = $Snippet
        }
    }
}

$cppFiles = Get-ChildItem -Path $targetDir -Recurse -File -Filter "*.cpp"
$violations = @()
$warnings = @()

foreach ($file in $cppFiles) {
    $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    if ([string]::IsNullOrEmpty($content)) {
        continue
    }

    $lines = @($content -split "`r?`n")
    $relativePath = Get-RelativePathCompat -Root $repoRoot -Path $file.FullName

    foreach ($rule in $rules) {
        Add-RegexViolations `
            -Violations ([ref]$violations) `
            -Content $content `
            -Lines $lines `
            -RelativePath $relativePath `
            -Pattern $rule.Pattern `
            -RuleId $rule.Id `
            -Description $rule.Description
    }

    if ($strictRtFiles -contains $relativePath) {
        Add-RegexViolations `
            -Violations ([ref]$violations) `
            -Content $content `
            -Lines $lines `
            -RelativePath $relativePath `
            -Pattern "publishAtomic\(|exchangeAtomic\(" `
            -RuleId "LINT-AE-005" `
            -Description "No publishAtomic()/exchangeAtomic() in strict RT processing sources"

        Add-RegexViolations `
            -Violations ([ref]$violations) `
            -Content $content `
            -Lines $lines `
            -RelativePath $relativePath `
            -Pattern "fetch_add\(" `
            -RuleId "LINT-AE-008" `
            -Description "fetch_add() in strict RT processing sources must be explicitly marked RT-RESTRICTED" `
            -Filter { param($lineText, $lineNumber, $match) return ($lineText -notmatch "RT-RESTRICTED") }
    }

    $warnings += Get-FunctionScopedWorldWarnings `
        -Content $content `
        -RelativePath $relativePath `
        -Pattern $warningRule.Pattern `
        -RuleId $warningRule.Id `
        -Description $warningRule.Description
}

$commentKeywordPattern = "//[^\r\n]*\b(TODO|FIXME|quick\s+fix|workaround|just\s+for\s+now|temporary)\b|/\*[\s\S]*?\b(TODO|FIXME|quick\s+fix|workaround|just\s+for\s+now|temporary)\b[\s\S]*?\*/"
$sourceFiles = foreach ($ext in $sourceExtensions) {
    Get-ChildItem -Path $sourceRoot -Recurse -File -Filter $ext
}

foreach ($file in $sourceFiles) {
    $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    if ([string]::IsNullOrEmpty($content)) {
        continue
    }

    $lines = @($content -split "`r?`n")
    $relativeSourcePath = Get-RelativePathCompat -Root $repoRoot -Path $file.FullName

    Add-RegexViolations `
        -Violations ([ref]$violations) `
        -Content $content `
        -Lines $lines `
        -RelativePath $relativeSourcePath `
        -Pattern $commentKeywordPattern `
        -RuleId "LINT-AE-009" `
        -Description "No dangerous AI-generated comment keywords in src/** (TODO/FIXME/quick fix/workaround/just for now/temporary)" `
        -Filter { param($lineText, $lineNumber, $match) return ($lineText -notmatch 'NOLINT\(danger-comment\)') }

    $inBlockComment = $false
    for ($lineIndex = 0; $lineIndex -lt $lines.Length; $lineIndex++) {
        $lineText = $lines[$lineIndex]
        $codeOnly = Remove-CommentsFromLine -Line $lineText -InBlockComment ([ref]$inBlockComment)
        if ([string]::IsNullOrWhiteSpace($codeOnly)) {
            continue
        }

        if ([System.Text.RegularExpressions.Regex]::IsMatch($codeOnly, "\b(thread_local|mutable)\b")) {
            # Allow mutable on standard mutex types (const-correct thread safety pattern)
            $allowedMutablePattern = '\bmutable\s+(std::)?(mutex|shared_mutex|recursive_mutex)\b'
            if ([System.Text.RegularExpressions.Regex]::IsMatch($codeOnly, $allowedMutablePattern)) {
                # Legitimate const-correct mutex pattern, skip
            }
            elseif ($lineText -match 'NOLINT\(thread-local\)' -and $lineText -match 'RT-SAFE:') {
                # NOLINT(thread-local) + RT-SAFE: intentinally reviewed thread-local cache.
                # ISR rule: allowed only when the comment documents WHY it is RT-safe
                # (POD type, const, no destructor, single initialization, etc.)
            }
            else {
                $violations += [PSCustomObject]@{
                    Rule        = "LINT-AE-011"
                    Description = "No mutable/thread_local tokens in src/** code (rule 4.1.5 / 15.2)"
                    File        = $relativeSourcePath
                    Line        = $lineIndex + 1
                    Snippet     = $lineText.Trim()
                }
            }
        }

        if ([System.Text.RegularExpressions.Regex]::IsMatch($codeOnly, "\breclaimAllIgnoringEpoch\s*\(")) {
            $normalizedSourcePath = ($relativeSourcePath -replace '\\', '/')
            if ($normalizedSourcePath -ne "src/DeferredDeletionQueue.h") {
                $violations += [PSCustomObject]@{
                    Rule        = "LINT-AE-012"
                    Description = "reclaimAllIgnoringEpoch() is shutdown-only and must not be called from src/**"
                    File        = $relativeSourcePath
                    Line        = $lineIndex + 1
                    Snippet     = $lineText.Trim()
                }
            }
        }

        if ([System.Text.RegularExpressions.Regex]::IsMatch($codeOnly, "\bconst_cast\s*<")) {
            $violations += [PSCustomObject]@{
                Rule        = "LINT-AE-013"
                Description = "No const_cast usage in src/** code (rule 15.2.2 const removal prohibition)"
                File        = $relativeSourcePath
                Line        = $lineIndex + 1
                Snippet     = $lineText.Trim()
            }
        }

        $hasSuspiciousCStylePtrCast = [System.Text.RegularExpressions.Regex]::IsMatch(
            $codeOnly,
            '\(\s*(?![^\)]*\bconst\b)[^\)]*\*\s*\)\s*[^;]+'
        )
        $hasConstToken = [System.Text.RegularExpressions.Regex]::IsMatch($codeOnly, '\bconst\b')
        $hasCppCastSyntax = [System.Text.RegularExpressions.Regex]::IsMatch(
            $codeOnly,
            '\b(?:reinterpret_cast|static_cast|const_cast|dynamic_cast)\s*<'
        )
        $castsStringLiteralToNonConstPtr = [System.Text.RegularExpressions.Regex]::IsMatch(
            $codeOnly,
            '\(\s*(?![^\)]*\bconst\b)[^\)]*\*\s*\)\s*L?"'
        )
        if ((( $hasSuspiciousCStylePtrCast -and $hasConstToken) -or $castsStringLiteralToNonConstPtr) -and -not $hasCppCastSyntax) {
            $violations += [PSCustomObject]@{
                Rule        = "LINT-AE-014"
                Description = "No suspicious C-style cast that may strip const in src/** code (rule 15.2.2)"
                File        = $relativeSourcePath
                Line        = $lineIndex + 1
                Snippet     = $lineText.Trim()
            }
        }
    }
}

$runtimeCommandQueuePath = Join-Path $repoRoot "src\audioengine\RuntimeCommandQueue.h"
if (Test-Path $runtimeCommandQueuePath) {
    $queueContent = Get-Content -Path $runtimeCommandQueuePath -Raw -Encoding UTF8
    $queueLines = @($queueContent -split "`r?`n")
    Add-RegexViolations `
        -Violations ([ref]$violations) `
        -Content $queueContent `
        -Lines $queueLines `
        -RelativePath (Get-RelativePathCompat -Root $repoRoot -Path $runtimeCommandQueuePath) `
        -Pattern "std::mutex|lock_guard|unique_lock" `
        -RuleId "LINT-AE-006" `
        -Description "RuntimeCommandQueue must remain lock-free (no mutex/lock_guard/unique_lock)"
}

$loadPipelinePath = Join-Path $repoRoot "src\convolver\ConvolverProcessor.LoadPipeline.cpp"
if (Test-Path $loadPipelinePath) {
    $loadPipelineContent = Get-Content -Path $loadPipelinePath -Raw -Encoding UTF8
    $violations += Get-SwitchEngineOrderViolations `
        -Content $loadPipelineContent `
        -RelativePath (Get-RelativePathCompat -Root $repoRoot -Path $loadPipelinePath)
}

$releaseResourcesPath = Join-Path $repoRoot "src\audioengine\AudioEngine.Processing.ReleaseResources.cpp"
Add-RequiredPatternViolation `
    -Violations ([ref]$violations) `
    -FilePath $releaseResourcesPath `
    -RelativePath (Get-RelativePathCompat -Root $repoRoot -Path $releaseResourcesPath) `
    -Pattern "drainDeferredRetireQueues\(\s*true\s*\)\s*;" `
    -RuleId "LINT-AE-010" `
    -Description "releaseResources() must call drainDeferredRetireQueues(true) during shutdown drain phase" `
    -Snippet "drainDeferredRetireQueues(true);"

$ctorDtorPath = Join-Path $repoRoot "src\audioengine\AudioEngine.CtorDtor.cpp"
Add-RequiredPatternViolation `
    -Violations ([ref]$violations) `
    -FilePath $ctorDtorPath `
    -RelativePath (Get-RelativePathCompat -Root $repoRoot -Path $ctorDtorPath) `
    -Pattern "drainDeferredRetireQueues\(\s*true\s*\)\s*;" `
    -RuleId "LINT-AE-010" `
    -Description "~AudioEngine() must call drainDeferredRetireQueues(true) during shutdown drain phase" `
    -Snippet "drainDeferredRetireQueues(true);"

if ($violations.Count -gt 0) {
    Write-Host "AudioEngine lint violations detected: $($violations.Count)"
    Write-Host ""

    $grouped = $violations | Group-Object Rule
    foreach ($group in $grouped) {
        Write-Host "[$($group.Name)]"
        foreach ($item in $group.Group) {
            Write-Host "  $($item.File):$($item.Line)"
            if ($item.Snippet) {
                Write-Host "    $($item.Snippet)"
            }
        }
        Write-Host ""
    }

    exit 1
}

if ($warnings.Count -gt 0) {
    Write-Host "AudioEngine lint warnings detected: $($warnings.Count)"
    Write-Host ""

    foreach ($warning in $warnings) {
        Write-Host "[$($warning.Rule)] $($warning.File):$($warning.Line)"
        Write-Host "  occurrences: $($warning.Count)"
        if ($warning.Snippet) {
            Write-Host "  $($warning.Snippet)"
        }
        Write-Host ""
    }
}

Write-Host "AudioEngine lint passed (LINT-AE-001/002/003/005/006/007/008/009/010/011/012/013/014)."
exit 0
