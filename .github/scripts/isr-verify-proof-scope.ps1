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
# プリプロセッサ条件の走査。`#if 0` などで除外される領域を空白化し、条件次第で
# 含まれるか決まらない領域は「曖昧」として記録する。
# 無効化された定義を関数本体として選ばないために使う。
#-----------------------------------------------------------------------------
function Get-BranchState {
    param([string]$Condition)

    # 完全な定数だけを受理する（"1 && 0" のような式を有効分岐と誤認しない）。
    $c = $Condition.Trim()
    if ($c -cmatch '^0+$' -or $c -ceq 'false') { return 'out' }
    if ($c -cmatch '^1+$' -or $c -ceq 'true') { return 'in' }
    return 'unknown'
}

function Get-PreprocessorRegions {
    param([string]$Text)

    $disabled = New-Object System.Collections.Generic.List[object]
    $conditional = New-Object System.Collections.Generic.List[object]
    if ($Text.Length -eq 0) {
        return [pscustomobject]@{ Disabled = $disabled; Conditional = $conditional }
    }

    # 行ごとに a=有効 / d=無効 / c=条件次第 を割り当てる
    $states = New-Object 'char[]' $Text.Length
    for ($k = 0; $k -lt $states.Length; $k++) { $states[$k] = [char]'a' }

    # 指令はコメント・文字列リテラルの外でのみ認識する。同じ長さの伏せ字を作り、
    # 行の切り出しだけを伏せ字から行う（位置は元テキストと一致する）。
    $masked = Get-MaskedText -Text $Text

    $frames = New-Object System.Collections.Generic.List[object]
    $lineStart = 0
    while ($lineStart -le $masked.Length) {
        $nl = $masked.IndexOf("`n", $lineStart)
        if ($nl -lt 0) { $nl = $masked.Length }
        $line = $masked.Substring($lineStart, $nl - $lineStart).TrimEnd("`r")

        $lineState = 'a'
        foreach ($f in $frames) {
            if ($f.State -eq 'out') { $lineState = 'd'; break }
            if ($f.State -eq 'unknown') { $lineState = 'c' }
        }
        for ($k = $lineStart; $k -lt $nl -and $k -lt $states.Length; $k++) {
            $states[$k] = [char]$lineState
        }

        $m = [System.Text.RegularExpressions.Regex]::Match(
            $line, '^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$')
        if ($m.Success) {
            $directive = $m.Groups[1].Value
            $argument = $m.Groups[2].Value.Trim()
            if ($directive -ceq 'if') {
                $state = Get-BranchState -Condition $argument
                $frame = [pscustomobject]@{ State = $state; Taken = 'none' }
                if ($state -eq 'in') { $frame.Taken = 'true' }
                elseif ($state -eq 'unknown') { $frame.Taken = 'unknown' }
                $frames.Add($frame)
            }
            elseif ($directive -ceq 'ifdef' -or $directive -ceq 'ifndef') {
                $frames.Add([pscustomobject]@{ State = 'unknown'; Taken = 'unknown' })
            }
            elseif ($directive -ceq 'elif') {
                if ($frames.Count -gt 0) {
                    $frame = $frames[$frames.Count - 1]
                    if ($frame.Taken -eq 'unknown') { $frame.State = 'unknown' }
                    elseif ($frame.Taken -eq 'true') { $frame.State = 'out' }
                    else {
                        $frame.State = Get-BranchState -Condition $argument
                        if ($frame.State -eq 'in') { $frame.Taken = 'true' }
                        elseif ($frame.State -eq 'unknown') { $frame.Taken = 'unknown' }
                    }
                }
            }
            elseif ($directive -ceq 'else') {
                if ($frames.Count -gt 0) {
                    $frame = $frames[$frames.Count - 1]
                    if ($frame.Taken -eq 'unknown') { $frame.State = 'unknown' }
                    elseif ($frame.Taken -eq 'true') { $frame.State = 'out' }
                    else { $frame.State = 'in'; $frame.Taken = 'true' }
                }
            }
            elseif ($directive -ceq 'endif') {
                if ($frames.Count -gt 0) { $frames.RemoveAt($frames.Count - 1) }
            }
        }

        if ($nl -ge $Text.Length) { break }
        $lineStart = $nl + 1
    }

    $runState = $states[0]
    $runStart = 0
    for ($k = 1; $k -le $states.Length; $k++) {
        $changed = ($k -ge $states.Length) -or ($states[$k] -ne $runState)
        if (-not $changed) { continue }
        if ($runState -ne [char]'a') {
            $range = [pscustomobject]@{ Start = $runStart; End = $k }
            if ($runState -eq [char]'d') { $disabled.Add($range) } else { $conditional.Add($range) }
        }
        if ($k -lt $states.Length) {
            $runState = $states[$k]
            $runStart = $k
        }
    }

    return [pscustomobject]@{ Disabled = $disabled; Conditional = $conditional }
}

# 無効化された領域を空白化する（文字位置は変えない）。
function Get-EnabledText {
    param(
        [string]$Text,
        $DisabledRanges
    )

    if ($null -eq $DisabledRanges -or $DisabledRanges.Count -eq 0) { return $Text }
    return (New-BlankedCopy -Text $Text -Ranges $DisabledRanges)
}

# 行頭のプリプロセッサ指令（#define の置換リストを含む）を空白化する。長さと改行位置は変えない。
# 未使用マクロの置換リストが「ガード」や「呼び出し」として数えられるのを防ぐ。
# Reference は行構造と行継続の判定に使う元テキスト。伏せ字を渡す場合、文字列の空白化で
# 末尾のバックスラッシュが消えるため、行の判定だけは元テキストで行う。
function Get-DirectiveBlankedText {
    param([string]$Text, [string]$Reference = $Text)

    if ($Text.Length -eq 0) { return $Text }
    $chars = $Text.ToCharArray()
    $n = $Text.Length
    $i = 0
    while ($i -lt $n) {
        $lineEnd = $Reference.IndexOf("`n", $i)
        if ($lineEnd -lt 0) { $lineEnd = $n }
        $line = $Reference.Substring($i, $lineEnd - $i)
        if ($line -match '^\s*#') {
            # 行継続（末尾のバックスラッシュ）が続く限り、次の行も置換リストの一部。
            $blankEnd = $lineEnd
            $check = $line.TrimEnd("`r")
            while ($check.EndsWith('\') -and $blankEnd -lt $n) {
                $nextStart = $blankEnd + 1
                $nextEnd = $Reference.IndexOf("`n", $nextStart)
                if ($nextEnd -lt 0) { $nextEnd = $n }
                $blankEnd = $nextEnd
                $check = $Reference.Substring($nextStart, $nextEnd - $nextStart).TrimEnd("`r")
            }
            for ($k = $i; $k -lt $blankEnd -and $k -lt $n; $k++) {
                if ($chars[$k] -ne "`n" -and $chars[$k] -ne "`r") { $chars[$k] = ' ' }
            }
            $i = $blankEnd + 1
            continue
        }
        $i = $lineEnd + 1
    }
    return (-join $chars)
}

#-----------------------------------------------------------------------------
# 関数本体を波括弧対応で切り出す。文字列・文字リテラル内の波括弧は数えない。
# 同じシグネチャの定義が複数ある場合に備えて、見つかった定義をすべて返す。
#-----------------------------------------------------------------------------
function Get-FunctionSpans {
    param(
        [string]$CodeText,
        [string]$Signature
    )

    $stringRanges = (Get-LexicalRanges -Text $CodeText).Strings
    $rangeEnds = @{}
    foreach ($r in $stringRanges) { $rangeEnds[$r.Start] = $r.End }

    $spans = @()
    foreach ($candidate in [System.Text.RegularExpressions.Regex]::Matches(
                 $CodeText, $Signature + '\s*(?:noexcept\s*)?\{')) {

        # 署名は文字列リテラルの中に現れることもあるため、文字列範囲の外だけを見る。
        $insideString = $false
        foreach ($r in $stringRanges) {
            if ($candidate.Index -ge $r.Start -and $candidate.Index -lt $r.End) { $insideString = $true; break }
        }
        if ($insideString) { continue }

        $openIndex = $CodeText.IndexOf('{', $candidate.Index + $candidate.Length - 1)
        if ($openIndex -lt 0) { continue }

        $depth = 0
        $i = $openIndex
        $end = -1
        while ($i -lt $CodeText.Length) {
            if ($rangeEnds.ContainsKey($i)) { $i = $rangeEnds[$i]; continue }
            $ch = $CodeText[$i]
            if ($ch -eq '{') { $depth++ }
            elseif ($ch -eq '}') {
                $depth--
                if ($depth -eq 0) { $end = $i + 1; break }
            }
            $i++
        }
        if ($end -lt 0) { continue }

        $spans += [pscustomobject]@{ SignatureIndex = $candidate.Index; Start = $openIndex; End = $end }
    }
    return $spans
}

# 定義が 1 つに定まるときだけ本体を返す。無効化された定義は、呼ぶ前に
# Get-PreprocessorRegions で空白化しておくこと。条件次第で含まれる定義や、
# 有効な定義が複数ある場合は Problem に理由を入れて本体を返さない。
function Resolve-FunctionBody {
    param(
        [string]$CodeText,
        [string]$MaskedText,
        $ConditionalRanges,
        [string]$Signature
    )

    # 署名探索と本体の文走査からは、プリプロセッサ指令（#define の置換リストを含む）を除く。
    # 位置は変わらないため、本体の切り出しと条件領域の計算は元テキストで行う。
    $spans = @(Get-FunctionSpans -CodeText (Get-DirectiveBlankedText -Text $CodeText) -Signature $Signature)
    if ($spans.Count -eq 0) {
        return [pscustomobject]@{ Code = $null; Masked = $null; Conditional = @(); Problem = 'notfound' }
    }

    foreach ($s in $spans) {
        foreach ($r in $ConditionalRanges) {
            if ($s.SignatureIndex -ge $r.Start -and $s.SignatureIndex -lt $r.End) {
                return [pscustomobject]@{ Code = $null; Masked = $null; Conditional = @(); Problem = 'conditional' }
            }
        }
    }

    if ($spans.Count -gt 1) {
        return [pscustomobject]@{ Code = $null; Masked = $null; Conditional = @(); Problem = 'ambiguous' }
    }

    $s = $spans[0]
    $bodyCode = $CodeText.Substring($s.Start, $s.End - $s.Start)
    $bodyMasked = $MaskedText.Substring($s.Start, $s.End - $s.Start)

    # 本体の内側にある条件コンパイルも記録する。必須のガードや呼び出しがその中に
    # ある場合、どのビルドで有効かを決められないため呼び出し側で FAIL にする。
    $bodyConditional = (Get-PreprocessorRegions -Text $bodyCode).Conditional

    return [pscustomobject]@{
        Code        = (Get-DirectiveBlankedText -Text $bodyCode)
        Masked      = (Get-DirectiveBlankedText -Text $bodyMasked -Reference $bodyCode)
        Conditional = $bodyConditional
        Problem     = $null
    }
}

# 指定位置が条件コンパイル下（有効・無効を確定できない領域）にあるか。
function Test-InRanges {
    param(
        [int]$Index,
        $Ranges
    )

    foreach ($r in $Ranges) {
        if ($Index -ge $r.Start -and $Index -lt $r.End) { return $true }
    }
    return $false
}

function Add-BodyProblem {
    param(
        $Errors,
        [string]$FunctionName,
        [string]$Problem
    )

    if ($Problem -ceq 'notfound') {
        $Errors.Add("DebugRuntime contract violation: $FunctionName() body not found")
    }
    elseif ($Problem -ceq 'conditional') {
        $Errors.Add("DebugRuntime contract violation: $FunctionName() is defined inside a conditional compilation block; the active definition cannot be determined")
    }
    elseif ($Problem -ceq 'ambiguous') {
        $Errors.Add("DebugRuntime contract violation: $FunctionName() has more than one active definition")
    }
}

#-----------------------------------------------------------------------------
# 波括弧ブロックを列挙し、そのブロックを導入した構文（if / for / else など）を記録する。
# 条件付きブロックの中でしか実行されない呼び出しを見つけるために使う。
#-----------------------------------------------------------------------------
function Get-BraceFrames {
    param([string]$MaskedBody)

    $frames = New-Object System.Collections.Generic.List[object]
    $stack = New-Object System.Collections.Generic.List[object]
    $lastDelimiter = 0
    $parenDepth = 0
    for ($i = 0; $i -lt $MaskedBody.Length; $i++) {
        $ch = $MaskedBody[$i]
        if ($ch -eq '(') { $parenDepth++; continue }
        if ($ch -eq ')') { if ($parenDepth -gt 0) { $parenDepth-- }; continue }
        if ($ch -eq ';' -and $parenDepth -eq 0) { $lastDelimiter = $i; continue }

        if ($ch -eq '{') {
            $prefix = $MaskedBody.Substring($lastDelimiter, $i - $lastDelimiter)
            $kind = 'block'
            if ($prefix -match '(?s)\b(if|else|for|while|switch|do|try|catch)\b[^{}]*$') {
                $kind = $Matches[1]
            }
            $stack.Add([pscustomobject]@{ Start = $i; End = -1; Kind = $kind })
            $lastDelimiter = $i
        }
        elseif ($ch -eq '}') {
            if ($stack.Count -gt 0) {
                $frame = $stack[$stack.Count - 1]
                $frame.End = $i
                $frames.Add($frame)
                $stack.RemoveAt($stack.Count - 1)
            }
            $lastDelimiter = $i
        }
    }
    return $frames
}

# 波括弧ブロックを導入した条件式を取り出す。位置は伏せ字版で取り、文字列を残した
# 本文から切り出す（条件式の中の文字列を比較するため）。
function Get-FrameCondition {
    param(
        [string]$CodeBody,
        [string]$MaskedBody,
        [int]$Start
    )

    $i = $Start - 1
    while ($i -ge 0 -and [char]::IsWhiteSpace($MaskedBody[$i])) { $i-- }
    if ($i -lt 0 -or $MaskedBody[$i] -ne ')') { return '' }

    $depth = 0
    $j = $i
    while ($j -ge 0) {
        if ($MaskedBody[$j] -eq ')') { $depth++ }
        elseif ($MaskedBody[$j] -eq '(') {
            $depth--
            if ($depth -eq 0) { break }
        }
        $j--
    }
    if ($j -lt 0) { return '' }
    return $CodeBody.Substring($j + 1, $i - $j - 1)
}

# 指定位置を含み、条件・ループで導入されたブロックを返す（裸のブロックは除く）。
function Get-ConditionalFramesContaining {
    param(
        [string]$CodeBody,
        [string]$MaskedBody,
        [int]$Index
    )

    $result = @()
    foreach ($f in (Get-BraceFrames -MaskedBody $MaskedBody)) {
        if ($f.Start -lt $Index -and $f.End -gt $Index -and $f.Kind -cne 'block') {
            $condition = Get-FrameCondition -CodeBody $CodeBody -MaskedBody $MaskedBody -Start $f.Start
            $result += [pscustomobject]@{
                Kind      = $f.Kind
                Start     = $f.Start
                End       = $f.End
                Condition = $condition
            }
        }
    }
    return $result
}

# Release でしか成立しない条件か。`mode == "Release"` を含み、強制証跡の抜け道
# （isForceEvidenceEmission）を持たない条件は、Release 以外では実行されない。
function Test-ReleaseOnlyCondition {
    param([string]$Condition)

    if ([string]::IsNullOrWhiteSpace($Condition)) { return $false }
    $c = $Condition -replace '\s+', ' '
    if ($c -cnotmatch 'mode\s*==\s*"Release"') { return $false }
    if ($c -cmatch 'isForceEvidenceEmission') { return $false }
    return $true
}

# 定数で偽と書かれた条件か（if (false) / if (0) の中は実行されない）。
function Test-NeverTrueCondition {
    param([string]$Condition)

    if ([string]::IsNullOrWhiteSpace($Condition)) { return $false }
    $c = $Condition.Trim()
    return ($c -cmatch '^(false|0+)$')
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
    # Depth は関数本体の開き波括弧を 1 としたときの深さ。トップレベル文だけを
    # ガードとして数えるために使う（実行されないブロック内のガードを除く）。
    $results = @()
    $pattern = 'if\s*\((?<cond>[^{};]*?)\)\s*\{\s*return\s*;\s*\}'
    foreach ($m in [System.Text.RegularExpressions.Regex]::Matches($MaskedBody, $pattern)) {
        $depth = 0
        for ($k = 0; $k -lt $m.Index; $k++) {
            if ($MaskedBody[$k] -eq '{') { $depth++ }
            elseif ($MaskedBody[$k] -eq '}') { $depth-- }
        }
        $condGroup = $m.Groups['cond']
        $condText = $CodeBody.Substring($condGroup.Index, $condGroup.Length)
        $results += [PSCustomObject]@{
            Condition = $condText
            Index     = $m.Index
            Depth     = $depth
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
#   - 文字列リテラルの綴りが違う条件（mode == "release" は実行時に一致しない）
# 文字列リテラルの比較は大文字小文字を区別する（-ceq / -ccontains）。
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

    if ($c -cmatch '\|\|') { return $false }

    if (-not $ForceSwitchExists) {
        return ((Remove-OuterParens -Text $c) -ceq $releaseTerm)
    }

    $terms = @($c -split '&&' | ForEach-Object { Remove-OuterParens -Text $_.Trim() })
    if ($terms.Count -ne 2) { return $false }
    return (($terms -ccontains $releaseTerm) -and ($terms -ccontains $forceTerm))
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

    $exporterRaw = Get-Content -Path $evidenceExporterPath -Raw -Encoding UTF8
    $runtimeRaw = Get-Content -Path $debugRuntimePath -Raw -Encoding UTF8

    # `#if 0` などで無効化された定義を本体として選ばない。条件次第で含まれる定義は、
    # 有効な定義を決められないため後段で曖昧として扱う。
    $exporterRegions = Get-PreprocessorRegions -Text $exporterRaw
    $runtimeRegions = Get-PreprocessorRegions -Text $runtimeRaw
    $exporterEnabled = Get-EnabledText -Text $exporterRaw -DisabledRanges $exporterRegions.Disabled
    $runtimeEnabled = Get-EnabledText -Text $runtimeRaw -DisabledRanges $runtimeRegions.Disabled

    $exporterCode = Get-CodeText $exporterEnabled
    $runtimeCode = Get-CodeText $runtimeEnabled
    $runtimeMasked = Get-MaskedText $runtimeEnabled
    $runtimeConditional = $runtimeRegions.Conditional

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

    $emitCIArtifacts = Resolve-FunctionBody -CodeText $runtimeCode -MaskedText $runtimeMasked `
        -ConditionalRanges $runtimeConditional -Signature 'void\s+DebugRuntime::emitCIArtifacts\s*\(\s*\)'
    if ($null -ne $emitCIArtifacts.Problem) {
        Add-BodyProblem -Errors $errors -FunctionName 'emitCIArtifacts' -Problem $emitCIArtifacts.Problem
    }
    else {
        $ciReturns = @(Get-ConditionalReturns -CodeBody $emitCIArtifacts.Code -MaskedBody $emitCIArtifacts.Masked)
        $ciReleaseAll = @($ciReturns | Where-Object { Test-ReleaseGuardCondition -Condition $_.Condition -ForceSwitchExists $hasForceEvidenceSwitch })
        # 条件コンパイル下のガードは、どのビルドで有効かを決められないため数えない。
        $ciReleaseFixed = @($ciReleaseAll | Where-Object { -not (Test-InRanges -Index $_.Index -Ranges $emitCIArtifacts.Conditional) })
        $ciReleaseTop = @($ciReleaseFixed | Where-Object { $_.Depth -eq 1 })

        if ($ciReleaseAll.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: Release early-return not found or its condition is wrong in emitCIArtifacts()")
        }
        elseif ($ciReleaseFixed.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: the Release early-return of emitCIArtifacts() is inside conditional compilation; the guard cannot be guaranteed")
        }
        elseif ($ciReleaseTop.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: the Release early-return of emitCIArtifacts() is nested inside another block; it must be a top-level statement")
        }

        # 呼び出しの有無は文字列を伏せた本文で見る（文字列内の "emitHBTrace();" を拾わない）。
        $traceMatches = @([System.Text.RegularExpressions.Regex]::Matches($emitCIArtifacts.Masked, 'emitHBTrace\s*\(\s*\)\s*;'))
        $traceFixed = @($traceMatches | Where-Object { -not (Test-InRanges -Index $_.Index -Ranges $emitCIArtifacts.Conditional) })
        if ($traceMatches.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: emitCIArtifacts() does not emit the HB trace")
        }
        elseif ($traceFixed.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: emitCIArtifacts() calls emitHBTrace() only inside conditional compilation; CI and Debug paths must call it")
        }
        else {
            $traceMatch = $traceFixed[0]
            if ($ciReleaseTop.Count -gt 0) {
                $firstReleaseReturn = ($ciReleaseTop | Sort-Object Index | Select-Object -First 1).Index
                if ($traceMatch.Index -lt $firstReleaseReturn) {
                    $errors.Add("DebugRuntime contract violation: emitHBTrace() runs before the Release early-return")
                }
            }

            # CI / Debug で必ず実行されること。Release でしか成立しない条件や、
            # 定数で偽の条件の中でしか呼ばれない場合、Release 以外で trace が出ない。
            $traceEnclosing = @(Get-ConditionalFramesContaining -CodeBody $emitCIArtifacts.Code `
                    -MaskedBody $emitCIArtifacts.Masked -Index $traceMatch.Index)
            $releaseOnly = @($traceEnclosing | Where-Object { Test-ReleaseOnlyCondition -Condition $_.Condition })
            $neverTrue = @($traceEnclosing | Where-Object { Test-NeverTrueCondition -Condition $_.Condition })
            if ($releaseOnly.Count -gt 0) {
                $errors.Add("DebugRuntime contract violation: emitHBTrace() is only reachable inside a Release-only conditional block; CI and Debug paths must call it")
            }
            elseif ($neverTrue.Count -gt 0) {
                $errors.Add("DebugRuntime contract violation: emitHBTrace() is inside a block whose condition is never true")
            }
        }
    }

    #--- R25-4: emitHBTrace() 自身の Release 抑制（トレース発行より前にあること）
    $emitHBTrace = Resolve-FunctionBody -CodeText $runtimeCode -MaskedText $runtimeMasked `
        -ConditionalRanges $runtimeConditional -Signature 'void\s+DebugRuntime::emitHBTrace\s*\(\s*\)'
    if ($null -ne $emitHBTrace.Problem) {
        Add-BodyProblem -Errors $errors -FunctionName 'emitHBTrace' -Problem $emitHBTrace.Problem
    }
    else {
        $traceReturns = @(Get-ConditionalReturns -CodeBody $emitHBTrace.Code -MaskedBody $emitHBTrace.Masked)
        $traceReleaseAll = @($traceReturns | Where-Object { Test-ReleaseGuardCondition -Condition $_.Condition -ForceSwitchExists $hasForceEvidenceSwitch })
        $traceReleaseFixed = @($traceReleaseAll | Where-Object { -not (Test-InRanges -Index $_.Index -Ranges $emitHBTrace.Conditional) })
        $traceReleaseTop = @($traceReleaseFixed | Where-Object { $_.Depth -eq 1 })

        if ($traceReleaseAll.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: emitHBTrace() lost its Release suppression or its condition is wrong")
        }
        elseif ($traceReleaseFixed.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: the Release suppression of emitHBTrace() is inside conditional compilation; the guard cannot be guaranteed")
        }
        elseif ($traceReleaseTop.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: the Release suppression of emitHBTrace() is nested inside another block; it must be a top-level statement")
        }

        $emitMatches = @([System.Text.RegularExpressions.Regex]::Matches($emitHBTrace.Masked, 'emitTrace\s*\('))
        $emitFixed = @($emitMatches | Where-Object { -not (Test-InRanges -Index $_.Index -Ranges $emitHBTrace.Conditional) })
        if ($emitMatches.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: emitHBTrace() does not emit the trace")
        }
        elseif ($emitFixed.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: emitHBTrace() emits the trace only inside conditional compilation")
        }
        elseif ($traceReleaseTop.Count -gt 0) {
            $firstTraceReturn = ($traceReleaseTop | Sort-Object Index | Select-Object -First 1).Index
            if ($emitFixed[0].Index -lt $firstTraceReturn) {
                $errors.Add("DebugRuntime contract violation: emitHBTrace() emits the trace before the Release suppression")
            }
        }
    }

    #--- R25-5: A(D1) の最終違反レポート生成。関数本体の中で emitViolationReport を呼ぶこと。
    #   ソース全体の後方一致だと、別関数の呼び出しで PASS してしまうため本体に限定する。
    $emitFinal = Resolve-FunctionBody -CodeText $runtimeCode -MaskedText $runtimeMasked `
        -ConditionalRanges $runtimeConditional -Signature 'void\s+DebugRuntime::emitFinalViolationReport\s*\(\s*\)'
    if ($null -ne $emitFinal.Problem) {
        Add-BodyProblem -Errors $errors -FunctionName 'emitFinalViolationReport' -Problem $emitFinal.Problem
    }
    else {
        $finalMatches = @([System.Text.RegularExpressions.Regex]::Matches($emitFinal.Masked, 'emitViolationReport\s*\('))
        $finalFixed = @($finalMatches | Where-Object { -not (Test-InRanges -Index $_.Index -Ranges $emitFinal.Conditional) })
        if ($finalMatches.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: emitFinalViolationReport() does not emit the violation report")
        }
        elseif ($finalFixed.Count -eq 0) {
            $errors.Add("DebugRuntime contract violation: emitFinalViolationReport() emits the violation report only inside conditional compilation")
        }
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
