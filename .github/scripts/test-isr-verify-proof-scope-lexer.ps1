#requires -Version 7.0
<#
.SYNOPSIS
    Lexical boundary tests for isr-verify-proof-scope.ps1.

.DESCRIPTION
    Loads the scanner functions from the verifier and exercises the boundary cases that
    the fixture-based self-test cannot observe directly:

      - comment markers inside normal, raw and prefixed-raw string literals
      - an escaped quote followed by // and /* inside the same literal
      - LF and CRLF line continuations (one literal range, start/end positions checked)
      - a comment marker inside a continued literal (must not be removed)
      - character literals vs numeric digit separators (with and without prefixes)
      - mutation controls: breaking the escape check or the CRLF skip must make the
        corresponding boundary case fail, so the tests are demonstrably discriminating
#>
$ErrorActionPreference = 'Stop'
$src = Get-Content -Path (Join-Path $PSScriptRoot 'isr-verify-proof-scope.ps1') -Raw -Encoding UTF8
$ast = [System.Management.Automation.Language.Parser]::ParseInput($src, [ref]$null, [ref]$null)
$funcs = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)
foreach ($f in $funcs) { Invoke-Expression $f.Extent.Text }

$fail = 0

function Check([string]$Name, [bool]$Ok, [string]$Detail) {
    if ($Ok) { Write-Host "[PASS] $Name" }
    else { Write-Host "[FAIL] $Name -- $Detail"; $script:fail++ }
}

function Get-StringRangesOf([string]$Text) {
    # 要素 0 件でも配列として返す（ToList の生 List を @() で包むと環境により失敗するため ToArray を使う）
    return (Get-LexicalRanges -Text $Text).Strings.ToArray()
}

# --- 1. "//" inside a normal string literal must not start a comment
$code1 = @'
const char* url = "https://example.com";
emitHBTrace();   // trailing comment
'@
$out1 = Get-CodeText $code1
Check "line-comment-inside-string" ($out1 -match 'emitHBTrace\(\);') "code after the string was blanked: [$($out1 -replace "`r?`n", ' | ')]"
Check "line-comment-inside-string (string kept)" ($out1 -match 'https://example\.com') "string was blanked"
Check "real-trailing-comment-blanked" ($out1 -notmatch 'trailing comment') "comment was kept"

# --- 2. "/*" inside a normal string literal must not start a block comment
$code2 = @'
const char* glob = "/* not a comment */";
emitHBTrace();
'@
$out2 = Get-CodeText $code2
Check "block-comment-inside-string" ($out2 -match 'emitHBTrace\(\);') "code after the string was blanked: [$($out2 -replace "`r?`n", ' | ')]"

# --- 3. comment markers inside a raw string literal
$code3 = @'
const char* doc = R"hb(/* not a comment */ // nor this)hb";
emitHBTrace();
'@
$out3 = Get-CodeText $code3
Check "raw-string-comment-markers" ($out3 -match 'emitHBTrace\(\);') "code after the raw string was blanked: [$($out3 -replace "`r?`n", ' | ')]"

# --- 3b. prefixed raw string literal
$code3b = @'
const char* doc = u8R"hb(/* not a comment */ // nor this )hb";
emitHBTrace();
'@
$out3b = Get-CodeText $code3b
Check "prefixed-raw-string" ($out3b -match 'emitHBTrace\(\);') "code after the prefixed raw string was blanked: [$($out3b -replace "`r?`n", ' | ')]"

# --- 4. character literal holding a brace is a literal, not code
$code4 = "const char closing = '}';`nemitHBTrace();`n"
$ranges4 = @(Get-StringRangesOf $code4)
$charRange = $ranges4 | Where-Object { $code4.Substring($_.Start, $_.End - $_.Start) -eq "'}'" }
Check "char-literal-brace-is-literal" ($null -ne $charRange) "no char-literal range found for '}'"
Check "char-literal-brace-brace-hidden" ($null -ne $charRange -and $code4.IndexOf('}') -ge $charRange.Start -and $code4.IndexOf('}') -lt $charRange.End) "the brace was treated as code"

# --- 5. numeric digit separator is not a character literal
$code5 = "constexpr long long k = 50'000'000;`nemitHBTrace();`n"
$ranges5 = @(Get-StringRangesOf $code5)
Check "digit-separator-not-literal" ($ranges5.Count -eq 0) "digit separator was scanned as a character literal"
$out5 = Get-CodeText $code5
Check "digit-separator-code-kept" ($out5 -match "50'000'000" -and $out5 -match 'emitHBTrace\(\);') "code was altered: [$($out5 -replace "`r?`n", ' | ')]"

# --- 5b. escaped single quote inside a character literal is one literal range
$code5b = "const char quote = '\'';`nemitHBTrace();`n"
$ranges5b = @(Get-StringRangesOf $code5b)
$charOpen = $code5b.IndexOf("'")
$charLiteral = "'\''"   # 4 文字: 引用符, バックスラッシュ, 引用符, 引用符
$slice5b = if ($ranges5b.Count -eq 1) { $code5b.Substring($ranges5b[0].Start, $ranges5b[0].End - $ranges5b[0].Start) } else { '' }
Check "char-literal-escaped-quote-single-range" ($ranges5b.Count -eq 1) "expected 1 literal range, got $($ranges5b.Count)"
Check "char-literal-escaped-quote-slice" ($slice5b -eq $charLiteral) "range slice [$slice5b] != [$charLiteral]"
Check "char-literal-escaped-quote-start" ($ranges5b.Count -eq 1 -and $ranges5b[0].Start -eq $charOpen) "range start mismatch"
Check "char-literal-escaped-quote-end" ($ranges5b.Count -eq 1 -and $ranges5b[0].End -eq ($charOpen + $charLiteral.Length)) "range end mismatch (expected $($charLiteral.Length) chars from the opening quote)"
Check "char-literal-escaped-quote-code-kept" ((Get-CodeText $code5b) -match 'emitHBTrace\(\);') "code after the literal was blanked"

# --- 6. prefixed character literal (L'}') is still a literal
$code6 = "const wchar_t w = L'}';`nemitHBTrace();`n"
$ranges6 = @(Get-StringRangesOf $code6)
$wRange = $ranges6 | Where-Object { $code6.Substring($_.Start, $_.End - $_.Start) -eq "'}'" }
Check "prefixed-char-literal" ($null -ne $wRange) "L'}' was treated as a digit separator"

# --- 7. an apostrophe inside a comment must not open a char literal
$code7 = "// don't break`nemitHBTrace();`n"
$ranges7 = @(Get-StringRangesOf $code7)
Check "apostrophe-in-comment" ($ranges7.Count -eq 0) "apostrophe inside a comment opened a literal"
$out7 = Get-CodeText $code7
Check "apostrophe-in-comment-code-kept" ($out7 -match 'emitHBTrace\(\);') "code after the comment was blanked"

# --- 8. an apostrophe inside a string must not open a char literal
$code8 = "const char* s = `"it's fine`";`nemitHBTrace();`n"
$out8 = Get-CodeText $code8
Check "apostrophe-in-string" ($out8 -match 'emitHBTrace\(\);') "code after the string was blanked: [$($out8 -replace "`r?`n", ' | ')]"

# --- 9. escaped quote followed by // and /* inside the same string literal
$code9 = 'const char* s = "escaped quote: \" // still inside /* here */";' + "`nemitHBTrace();`n"
$ranges9 = @(Get-StringRangesOf $code9)
$openQuote = $code9.IndexOf('"')
$closeQuote = $code9.LastIndexOf('"')
$expected9 = $code9.Substring($openQuote, $closeQuote - $openQuote + 1)
$slice9 = if ($ranges9.Count -eq 1) { $code9.Substring($ranges9[0].Start, $ranges9[0].End - $ranges9[0].Start) } else { '' }
Check "escaped-quote-single-range" ($ranges9.Count -eq 1) "expected 1 literal range, got $($ranges9.Count)"
Check "escaped-quote-range-covers-literal" ($slice9 -eq $expected9) "range slice [$slice9] != literal [$expected9]"
Check "escaped-quote-range-start" ($ranges9.Count -eq 1 -and $ranges9[0].Start -eq $openQuote) "range start mismatch"
Check "escaped-quote-range-end" ($ranges9.Count -eq 1 -and $ranges9[0].End -eq ($closeQuote + 1)) "range end mismatch"
Check "escaped-quote-comment-marker-inside" ($ranges9.Count -eq 1 -and $code9.IndexOf('//') -gt $ranges9[0].Start -and $code9.IndexOf('//') -lt $ranges9[0].End) "'//' is not inside the literal range"
$out9 = Get-CodeText $code9
Check "escaped-quote-code-kept" ($out9 -match 'emitHBTrace\(\);' -and $out9 -match 'still inside') "code or string content was removed: [$($out9 -replace "`r?`n", ' | ')]"

# --- 10. LF line continuation is one literal range
$code10 = 'const char* s = "abc\' + "`ndef" + '";' + "`nemitHBTrace();`n"
$ranges10 = @(Get-StringRangesOf $code10)
$slice10 = if ($ranges10.Count -eq 1) { $code10.Substring($ranges10[0].Start, $ranges10[0].End - $ranges10[0].Start) } else { '' }
Check "lf-continuation-single-range" ($ranges10.Count -eq 1) "expected 1 literal range, got $($ranges10.Count)"
Check "lf-continuation-range-spans-lines" ($slice10.Contains("`n") -and $slice10.StartsWith('"abc') -and $slice10.EndsWith('def"')) "range does not span the continued literal: [$($slice10 -replace "`r?`n", ' | ')]"
Check "lf-continuation-code-kept" ((Get-CodeText $code10) -match 'emitHBTrace\(\);') "code after the literal was blanked"

# --- 11. CRLF line continuation is one literal range
$code11 = 'const char* s = "abc\' + "`r`ndef" + '";' + "`r`nemitHBTrace();`r`n"
$ranges11 = @(Get-StringRangesOf $code11)
$slice11 = if ($ranges11.Count -eq 1) { $code11.Substring($ranges11[0].Start, $ranges11[0].End - $ranges11[0].Start) } else { '' }
Check "crlf-continuation-single-range" ($ranges11.Count -eq 1) "expected 1 literal range, got $($ranges11.Count)"
Check "crlf-continuation-range-spans-lines" ($slice11.Contains("`r`n") -and $slice11.StartsWith('"abc') -and $slice11.EndsWith('def"')) "range does not span the continued literal: [$($slice11 -replace "`r?`n", ' | ')]"
Check "crlf-continuation-code-kept" ((Get-CodeText $code11) -match 'emitHBTrace\(\);') "code after the literal was blanked"

# --- 12. comment markers inside a continued literal are not removed
$code12 = 'const char* s = "abc\' + "`n// not a comment" + '";' + "`nemitHBTrace();   // real comment`n"
$ranges12 = @(Get-StringRangesOf $code12)
$out12 = Get-CodeText $code12
Check "continued-literal-single-range" ($ranges12.Count -eq 1) "expected 1 literal range, got $($ranges12.Count)"
Check "continued-literal-comment-marker-kept" ($out12 -match '// not a comment') "the comment marker inside the literal was removed"
Check "continued-literal-real-comment-blanked" ($out12 -notmatch 'real comment') "the real trailing comment was kept"
Check "continued-literal-code-kept" ($out12 -match 'emitHBTrace\(\);') "code after the literal was blanked"

# --- 13. mutation controls: the boundary cases must be sensitive to a broken scanner
$fnText = ($funcs | Where-Object { $_.Name -eq 'Get-LexicalRanges' }).Extent.Text

$brokenEscape = $fnText.Replace('-eq $backslash', "-eq [char]'|'")
if ($brokenEscape -eq $fnText) {
    Check "mutation-escape-detection-observable" $false "could not mutate the backslash initialiser"
}
else {
    Invoke-Expression $brokenEscape
    $codeM = 'const char* s = "escaped quote: \" // still inside";' + "`nemitHBTrace();`n"
    $rM = @((Get-LexicalRanges -Text $codeM).Strings.ToArray())
    $sliceM = if ($rM.Count -eq 1) { $codeM.Substring($rM[0].Start, $rM[0].End - $rM[0].Start) } else { '' }
    $expectedM = '"escaped quote: \" // still inside"'
    $detected = -not ($rM.Count -eq 1 -and $sliceM -eq $expectedM)
    Check "mutation-escape-detection-observable" $detected "breaking the escape check did not change the result (case is not discriminating)"
    Invoke-Expression $fnText
}

$brokenCrlf = $fnText.Replace('$i += 3; continue', '$i += 2; continue')
if ($brokenCrlf -eq $fnText) {
    Check "mutation-crlf-continuation-observable" $false "could not mutate the CRLF skip"
}
else {
    # 変異の前に、正しい走査器がこの入力を受け入れること（基準線）を確認する。
    $codeN = 'const char* s = "abc\' + "`r`ndef" + '";' + "`r`nemitHBTrace();`r`n"
    $rBase = @((Get-LexicalRanges -Text $codeN).Strings.ToArray())
    $sliceBase = if ($rBase.Count -eq 1) { $codeN.Substring($rBase[0].Start, $rBase[0].End - $rBase[0].Start) } else { '' }
    Check "crlf-mutation-baseline" ($rBase.Count -eq 1 -and $sliceBase.Contains("`r`n")) "baseline scanner does not accept the mutated input (test would be vacuous)"

    Invoke-Expression $brokenCrlf
    $rN = @((Get-LexicalRanges -Text $codeN).Strings.ToArray())
    $sliceN = if ($rN.Count -eq 1) { $codeN.Substring($rN[0].Start, $rN[0].End - $rN[0].Start) } else { '' }
    $detectedN = -not ($rN.Count -eq 1 -and $sliceN.Contains("`r`n"))
    Check "mutation-crlf-continuation-observable" $detectedN "breaking the CRLF skip did not change the result (case is not discriminating)"
    Invoke-Expression $fnText
}

$brokenCharEscape = $fnText.Replace('-eq $charEscape', "-eq [char]'|'")
if ($brokenCharEscape -eq $fnText) {
    Check "mutation-char-escape-detection-observable" $false "could not mutate the character-literal escape check"
}
else {
    Invoke-Expression $brokenCharEscape
    $codeC = "const char quote = '\'';`nemitHBTrace();`n"
    $rC = @((Get-LexicalRanges -Text $codeC).Strings.ToArray())
    $sliceC = if ($rC.Count -eq 1) { $codeC.Substring($rC[0].Start, $rC[0].End - $rC[0].Start) } else { '' }
    $detectedC = -not ($rC.Count -eq 1 -and $sliceC -eq "'\''")
    Check "mutation-char-escape-detection-observable" $detectedC "breaking the character-literal escape check did not change the result (case is not discriminating)"
    Invoke-Expression $fnText
}

# --- 13. a line comment continued with a backslash keeps the next line commented
$code13 = "// disabled \" + "`n" + "if (mode == `"Release`") { return; }" + "`nemitHBTrace();`n"
$out13 = Get-CodeText $code13
Check "comment-continuation-hides-guard" ($out13 -notmatch 'mode ==') "the continued comment line was treated as code"
Check "comment-continuation-code-kept" ($out13 -match 'emitHBTrace\(\);') "code after the comment was blanked"

# --- 13b. same with CRLF splicing
$code13b = "// disabled \" + "`r`n" + "if (mode == `"Release`") { return; }" + "`r`nemitHBTrace();`r`n"
$out13b = Get-CodeText $code13b
Check "comment-continuation-crlf-hides-guard" ($out13b -notmatch 'mode ==') "the continued CRLF comment line was treated as code"
Check "comment-continuation-crlf-code-kept" ($out13b -match 'emitHBTrace\(\);') "code after the CRLF comment was blanked"

# --- 13c. a comment that ends without a backslash must not swallow the next line
$code13c = "// note`nif (mode == `"Release`") { return; }`nemitHBTrace();`n"
$out13c = Get-CodeText $code13c
Check "comment-without-continuation-keeps-code" ($out13c -match 'mode ==') "a non-continued comment swallowed the next line"

if ($fail -gt 0) {
    Write-Host ""
    Write-Host "scanner probe: $fail check(s) failed"
    exit 1
}
Write-Host ""
Write-Host "scanner probe: all checks passed"
exit 0
