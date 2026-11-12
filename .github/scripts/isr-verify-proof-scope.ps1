param(
    # 検証対象リポジトリのルート。省略時はこのスクリプトから 2 つ上のディレクトリ。
    # テスト時はフィクスチャのルートを渡して、同じ検査器を別ツリーへ向ける。
    [string]$Root
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Root)) {
    $repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
}
else {
    $repoRoot = [System.IO.Path]::GetFullPath($Root)
}

#-----------------------------------------------------------------------------
# C++ の字句を 1 回走査して、コメント範囲と文字列・文字リテラル範囲を求める。
#   - 文字列の中の // や /* はコメントにしない
#   - コメントの中の引用符は文字列にしない
#   - raw string（R"delim( ... )delim"）
#   - 数値の桁区切り（50'000'000）と文字リテラル（'}' / L'}' / u8'a'）を区別する
#-----------------------------------------------------------------------------
function Get-LexicalRanges {
    param([string]$Text)

    $comments = New-Object System.Collections.Generic.List[object]
    $strings = New-Object System.Collections.Generic.List[object]
    $n = $Text.Length
    $i = 0
    # 1 文字のバックスラッシュ。PowerShell のシングルクォートでは '\' が 1 文字になる。
    # 比較の見た目で迷わないよう [char] で明示する。
    $backslash = [char]'\'

    # 直前のトークンが「数値の途中」かどうかの判定に使う。
    function Test-NumericSeparator {
        param([string]$Src, [int]$Pos)

        if ($Pos -le 0 -or $Pos + 1 -ge $Src.Length) { return $false }
        $prev = $Src[$Pos - 1]
        $next = $Src[$Pos + 1]
        if ($prev -notmatch '[0-9a-fA-F]') { return $false }
        if ($next -notmatch '[0-9a-fA-F]') { return $false }

        # 文字リテラルの接頭辞（L / u / U / u8）が直前に付いている場合は区切りではない。
        $j = $Pos - 1
        $prefix = ''
        while ($j -ge 0 -and ($Pos - $j) -le 2 -and $Src[$j] -match '[A-Za-z0-9]') {
            $prefix = $Src[$j] + $prefix
            if ($prefix -in @('L', 'u', 'U', 'u8')) { return $false }
            $j--
        }
        return $true
    }

    while ($i -lt $n) {
        $c = $Text[$i]
        $n1 = if ($i + 1 -lt $n) { $Text[$i + 1] } else { [char]0 }

        # --- line comment（行末のバックスラッシュは次の行へ継続する: C++ の行連結）
        if ($c -eq '/' -and $n1 -eq '/') {
            $start = $i
            while ($i -lt $n) {
                if ($Text[$i] -eq "`n") {
                    $j = $i - 1
                    if ($j -ge $start -and $Text[$j] -eq "`r") { $j-- }
                    if ($j -ge $start -and $Text[$j] -eq $backslash) { $i++; continue }
                    break
                }
                $i++
            }
            $comments.Add([PSCustomObject]@{ Start = $start; End = $i })
            continue
        }

        # --- block comment
        if ($c -eq '/' -and $n1 -eq '*') {
            $start = $i
            $i += 2
            while ($i + 1 -lt $n -and -not ($Text[$i] -eq '*' -and $Text[$i + 1] -eq '/')) { $i++ }
            $i = [Math]::Min($i + 2, $n)
            $comments.Add([PSCustomObject]@{ Start = $start; End = $i })
            continue
        }

        # --- raw string literal: [u8|u|U|L]R"delim( ... )delim"
        $rawQuoteIndex = -1
        if ($c -eq 'R' -and $n1 -eq '"') {
            $rawQuoteIndex = $i + 2
        }
        elseif (($c -eq 'u' -or $c -eq 'U' -or $c -eq 'L') -and $i + 2 -lt $n) {
            if ($c -eq 'u' -and $n1 -eq '8' -and $i + 3 -lt $n -and $Text[$i + 2] -eq 'R' -and $Text[$i + 3] -eq '"') {
                $rawQuoteIndex = $i + 4
            }
            elseif ($n1 -eq 'R' -and $Text[$i + 2] -eq '"') {
                $rawQuoteIndex = $i + 3
            }
        }

        if ($rawQuoteIndex -ge 0) {
            $start = $i
            $j = $rawQuoteIndex
            $delim = ''
            while ($j -lt $n -and $Text[$j] -ne '(' -and $delim.Length -le 16) { $delim += $Text[$j]; $j++ }
            if ($j -lt $n -and $Text[$j] -eq '(') {
                $close = ')' + $delim + '"'
                $endIdx = $Text.IndexOf($close, $j, [System.StringComparison]::Ordinal)
                if ($endIdx -lt 0) { $endIdx = $n - 1 }
                $end = [Math]::Min($endIdx + $close.Length, $n)
                $strings.Add([PSCustomObject]@{ Start = $start; End = $end })
                $i = $end
                continue
            }
        }

        # --- ordinary string literal
        if ($c -eq '"') {
            $start = $i
            $i++
            while ($i -lt $n) {
                $ch = $Text[$i]
                if ($ch -eq $backslash) {
                    # 行継続（バックスラッシュ + LF / CRLF / CR）はリテラルの内側として読み飛ばす
                    $nextCh = if ($i + 1 -lt $n) { $Text[$i + 1] } else { [char]0 }
                    if ($nextCh -eq "`r" -and ($i + 2) -lt $n -and $Text[$i + 2] -eq "`n") { $i += 3; continue }
                    if ($nextCh -eq "`n" -or $nextCh -eq "`r") { $i += 2; continue }
                    $i += 2
                    continue
                }
                if ($ch -eq '"') { $i++; break }
                if ($ch -eq "`n" -or $ch -eq "`r") { break }
                $i++
            }
            $strings.Add([PSCustomObject]@{ Start = $start; End = [Math]::Min($i, $n) })
            continue
        }

        # --- character literal (unless it is a numeric digit separator)
        if ($c -eq "'") {
            if (Test-NumericSeparator -Src $Text -Pos $i) {
                $i++
                continue
            }
            $start = $i
            $i++
            # 文字リテラルのエスケープ判定。通常文字列とは別名にして、片側だけの回帰を
            # 変異対照で切り分けられるようにする（意味は同じ $backslash）。
            $charEscape = $backslash
            while ($i -lt $n) {
                $ch = $Text[$i]
                if ($ch -eq $charEscape) { $i += 2; continue }
                if ($ch -eq "'") { $i++; break }
                if ($ch -eq "`n" -or $ch -eq "`r") { break }
                $i++
            }
            $strings.Add([PSCustomObject]@{ Start = $start; End = [Math]::Min($i, $n) })
            continue
        }

        $i++
    }

    return [PSCustomObject]@{
        Comments = $comments
        Strings  = $strings
    }
}

function New-BlankedCopy {
    param([string]$Text, $Ranges)

    $chars = $Text.ToCharArray()
    foreach ($r in $Ranges) {
        for ($k = $r.Start; $k -lt $r.End -and $k -lt $chars.Length; $k++) {
            if ($chars[$k] -ne "`n" -and $chars[$k] -ne "`r") { $chars[$k] = ' ' }
        }
    }
    return (-join $chars)
}

# コメントだけを空白化する（文字列は残す。条件式 `mode == "Release"` を判定するため）
function Get-CodeText {
    param([string]$Text)
    return (New-BlankedCopy -Text $Text -Ranges (Get-LexicalRanges -Text $Text).Comments)
}

# コメントと文字列の両方を空白化する（トークン検査用）
function Get-MaskedText {
    param([string]$Text)
    $lex = Get-LexicalRanges -Text $Text
    $withoutComments = New-BlankedCopy -Text $Text -Ranges $lex.Comments
    return (New-BlankedCopy -Text $withoutComments -Ranges $lex.Strings)
}

#-----------------------------------------------------------------------------
# 関数本体を波括弧対応で切り出す。文字列・文字リテラル内の波括弧は数えない。
#-----------------------------------------------------------------------------
function Get-FunctionSpan {
    param(
        [string]$CodeText,
        [string]$Signature
    )

    # 署名は文字列リテラルの中に現れることもあるため、文字列範囲の外の最初の一致を使う。
    $signatureRanges = (Get-LexicalRanges -Text $CodeText).Strings
    $m = $null
    foreach ($candidate in [System.Text.RegularExpressions.Regex]::Matches(
                 $CodeText, $Signature + '\s*(?:noexcept\s*)?\{')) {
        $insideString = $false
        foreach ($r in $signatureRanges) {
            if ($candidate.Index -ge $r.Start -and $candidate.Index -lt $r.End) { $insideString = $true; break }
        }
        if (-not $insideString) { $m = $candidate; break }
    }
    if ($null -eq $m) {
        return $null
    }

    $openIndex = $CodeText.IndexOf('{', $m.Index + $m.Length - 1)
    if ($openIndex -lt 0) { return $null }

    $stringRanges = (Get-LexicalRanges -Text $CodeText).Strings
    $rangeEnds = @{}
    foreach ($r in $stringRanges) { $rangeEnds[$r.Start] = $r.End }

    $depth = 0
    $i = $openIndex
    while ($i -lt $CodeText.Length) {
        if ($rangeEnds.ContainsKey($i)) {
            $i = $rangeEnds[$i]
            continue
        }
        $ch = $CodeText[$i]
        if ($ch -eq '{') { $depth++ }
        elseif ($ch -eq '}') {
            $depth--
            if ($depth -eq 0) {
                return [PSCustomObject]@{ Start = $openIndex; End = $i + 1 }
            }
        }
        $i++
    }
    return $null
}

function Get-FunctionBody {
    param(
        [string]$CodeText,
        [string]$Signature
    )

    $span = Get-FunctionSpan -CodeText $CodeText -Signature $Signature
    if ($null -eq $span) { return $null }
    return $CodeText.Substring($span.Start, $span.End - $span.Start)
}

function Get-FunctionBodyMasked {
    param(
        [string]$CodeText,
        [string]$MaskedText,
        [string]$Signature
    )

    $span = Get-FunctionSpan -CodeText $CodeText -Signature $Signature
    if ($null -eq $span) { return $null }
    return $MaskedText.Substring($span.Start, $span.End - $span.Start)
}

#-----------------------------------------------------------------------------
# 条件付き早期 return を列挙する
#-----------------------------------------------------------------------------
function Get-ConditionalReturns {
    param(
        [string]$CodeBody,
        [string]$MaskedBody
    )

    if ([string]::IsNullOrEmpty($MaskedBody)) { $MaskedBody = $CodeBody }

    # 構造（if (...) { return; }）は文字列を伏せた本文で探し、条件式の文字列だけを
    # 元の本文から取り出す。文字列に書かれた偽のガードを拾わないため。
    $results = @()
    $pattern = 'if\s*\((?<cond>[^{};]*?)\)\s*\{\s*return\s*;\s*\}'
    foreach ($m in [System.Text.RegularExpressions.Regex]::Matches($MaskedBody, $pattern)) {
        $condGroup = $m.Groups['cond']
        $condText = $CodeBody.Substring($condGroup.Index, $condGroup.Length)
        $results += [PSCustomObject]@{
            Condition = $condText
            Index     = $m.Index
            Text      = $m.Value
        }
    }
    return $results
}

function Remove-OuterParens {
    param([string]$Text)

    $t = $Text.Trim()
    while ($t.Length -ge 2 -and $t.StartsWith('(') -and $t.EndsWith(')')) {
        $depth = 0
        $enclosesWhole = $true
        for ($k = 0; $k -lt $t.Length; $k++) {
            if ($t[$k] -eq '(') { $depth++ }
            elseif ($t[$k] -eq ')') {
                $depth--
                if ($depth -eq 0 -and $k -lt ($t.Length - 1)) { $enclosesWhole = $false; break }
            }
        }
        if (-not $enclosesWhole) { break }
        $t = $t.Substring(1, $t.Length - 2).Trim()
    }
    return $t
}

# Release ガードの条件式が意図した形かを判定する。
#   force 実装:   mode == "Release" && ! isForceEvidenceEmission()   （項の順序は不問）
#   旧実装:       mode == "Release"
# 次の形は受け付けない:
#   - "||" で結ばれた条件（強制時にも抑制してしまう）
#   - "!" で全体を反転した条件（!(mode == "Release") など）
#   - 項が 3 つ以上ある条件
function Test-ReleaseGuardCondition {
    param(
        [string]$Condition,
        [bool]$ForceSwitchExists
    )

    $releaseTerm = 'mode == "Release"'
    $forceTerm = '! isForceEvidenceEmission()'

    $c = ($Condition -replace '\s+', ' ')
    $c = $c -replace '!\s*', '! '
    $c = $c.Trim()

    if ($c -match '\|\|') { return $false }

    if (-not $ForceSwitchExists) {
        return ((Remove-OuterParens -Text $c) -eq $releaseTerm)
    }

    $terms = @($c -split '&&' | ForEach-Object { Remove-OuterParens -Text $_.Trim() })
    if ($terms.Count -ne 2) { return $false }
    return (($terms -contains $releaseTerm) -and ($terms -contains $forceTerm))
}

#-----------------------------------------------------------------------------
# 検証本体。エラーメッセージの配列を返す（空配列 = PASS）
#-----------------------------------------------------------------------------
function Test-ProofScopeContract {
    param([string]$RepoRoot)

    $errors = New-Object System.Collections.Generic.List[string]

    $evidenceExporterPath = Join-Path $RepoRoot "src\audioengine\ISREvidenceExporter.cpp"
    $debugRuntimePath = Join-Path $RepoRoot "src\audioengine\ISRDebugRuntime.cpp"

    if (-not (Test-Path $evidenceExporterPath)) {
        $errors.Add("ISREvidenceExporter.cpp not found: $evidenceExporterPath")
        return ($errors | ForEach-Object { $_ })
    }
    if (-not (Test-Path $debugRuntimePath)) {
        $errors.Add("ISRDebugRuntime.cpp not found: $debugRuntimePath")
        return ($errors | ForEach-Object { $_ })
    }

    $exporterCode = Get-CodeText (Get-Content -Path $evidenceExporterPath -Raw -Encoding UTF8)
    $runtimeCode = Get-CodeText (Get-Content -Path $debugRuntimePath -Raw -Encoding UTF8)
    $runtimeMasked = Get-MaskedText (Get-Content -Path $debugRuntimePath -Raw -Encoding UTF8)

    #--- R25-1: proofLevel の割り当て（Release=off / Debug=partial / それ以外=full）
    if ($exporterCode -notmatch 'if \(mode == "Release"\)\s*\{\s*return "off";\s*\}') {
        $errors.Add("Proof scope contract violation: Release must map to proofLevel=off")
    }
    if ($exporterCode -notmatch 'if \(mode == "Debug"\)\s*\{\s*return "partial";\s*\}') {
        $errors.Add("Proof scope contract violation: Debug must map to proofLevel=partial")
    }
    if ($exporterCode -notmatch 'return "full";') {
        $errors.Add("Proof scope contract violation: CI/default must map to proofLevel=full")
    }

    #--- R25-2: Release では minimal evidence のみ
    if ($exporterCode -notmatch 'if \(isRelease\)\s*\{') {
        $errors.Add("Release evidence scope violation: isRelease block not found")
    }
    if ($exporterCode -notmatch 'minimalEvidence') {
        $errors.Add("Release evidence scope violation: minimalEvidence marker not found")
    }
    if ($exporterCode -notmatch 'if \(isRelease\)[\s\S]*?return;') {
        $errors.Add("Release evidence scope violation: early return for release path not found")
    }

    #--- R25-3: 強制証跡モード（B-11）の有無
    $hasForceEvidenceSwitch = $runtimeMasked -match 'isForceEvidenceEmission\s*\(\s*\)'

    $emitCIArtifacts = Get-FunctionBody -CodeText $runtimeCode -Signature 'void\s+DebugRuntime::emitCIArtifacts\s*\(\s*\)'
    $emitCIArtifactsMasked = Get-FunctionBodyMasked -CodeText $runtimeCode -MaskedText $runtimeMasked `
        -Signature 'void\s+DebugRuntime::emitCIArtifacts\s*\(\s*\)'
    if ($null -eq $emitCIArtifacts) {
        $errors.Add("DebugRuntime contract violation: emitCIArtifacts() body not found")
    }
    else {
        $returns = @(Get-ConditionalReturns -CodeBody $emitCIArtifacts -MaskedBody $emitCIArtifactsMasked)
        $releaseReturns = @($returns | Where-Object { Test-ReleaseGuardCondition -Condition $_.Condition -ForceSwitchExists $hasForceEvidenceSwitch })

        if ($releaseReturns.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: Release early-return not found or its condition is wrong in emitCIArtifacts()")
        }

        # 呼び出しの有無は文字列を伏せた本文で見る（文字列内の "emitHBTrace();" を拾わない）。
        $traceMatch = [System.Text.RegularExpressions.Regex]::Match($emitCIArtifactsMasked, 'emitHBTrace\s*\(\s*\)\s*;')
        if (-not $traceMatch.Success) {
            $errors.Add("DebugRuntime contract violation: emitCIArtifacts() does not emit the HB trace")
        }
        elseif ($releaseReturns.Count -gt 0) {
            $firstReleaseReturn = ($releaseReturns | Sort-Object Index | Select-Object -First 1).Index
            if ($traceMatch.Index -lt $firstReleaseReturn) {
                $errors.Add("DebugRuntime contract violation: emitHBTrace() runs before the Release early-return")
            }
        }
    }

    #--- R25-4: emitHBTrace() 自身の Release 抑制（トレース発行より前にあること）
    $emitHBTrace = Get-FunctionBody -CodeText $runtimeCode -Signature 'void\s+DebugRuntime::emitHBTrace\s*\(\s*\)'
    $emitHBTraceMasked = Get-FunctionBodyMasked -CodeText $runtimeCode -MaskedText $runtimeMasked `
        -Signature 'void\s+DebugRuntime::emitHBTrace\s*\(\s*\)'
    if ($null -eq $emitHBTrace) {
        $errors.Add("DebugRuntime contract violation: emitHBTrace() body not found")
    }
    else {
        $traceReturns = @(Get-ConditionalReturns -CodeBody $emitHBTrace -MaskedBody $emitHBTraceMasked)
        $traceReleaseReturns = @($traceReturns | Where-Object { Test-ReleaseGuardCondition -Condition $_.Condition -ForceSwitchExists $hasForceEvidenceSwitch })
        if ($traceReleaseReturns.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: emitHBTrace() lost its Release suppression or its condition is wrong")
        }
        else {
            $emitMatch = [System.Text.RegularExpressions.Regex]::Match($emitHBTraceMasked, 'emitTrace\s*\(')
            $firstTraceReturn = ($traceReleaseReturns | Sort-Object Index | Select-Object -First 1).Index
            if ($emitMatch.Success -and $emitMatch.Index -lt $firstTraceReturn) {
                $errors.Add("DebugRuntime contract violation: emitHBTrace() emits the trace before the Release suppression")
            }
        }
    }

    #--- R25-5: A(D1) の最終違反レポート生成。関数本体の中で emitViolationReport を呼ぶこと。
    #   ソース全体の後方一致だと、別関数の呼び出しで PASS してしまうため本体に限定する。
    $emitFinalMasked = Get-FunctionBodyMasked -CodeText $runtimeCode -MaskedText $runtimeMasked `
        -Signature 'void\s+DebugRuntime::emitFinalViolationReport\s*\(\s*\)'
    if ($null -eq $emitFinalMasked) {
        $errors.Add("DebugRuntime contract violation: emitFinalViolationReport() not found")
    }
    elseif ($emitFinalMasked -notmatch 'emitViolationReport\s*\(') {
        $errors.Add("DebugRuntime contract violation: emitFinalViolationReport() does not emit the violation report")
    }

    return ($errors | ForEach-Object { $_ })
}

$problems = @(Test-ProofScopeContract -RepoRoot $repoRoot)
if ($problems.Count -gt 0) {
    foreach ($p in $problems) {
        Write-Host "[FAIL] $p"
    }
    exit 1
}

Write-Host "[PASS] Proof scope verification (R25 + B-11 force-evidence)"
exit 0
