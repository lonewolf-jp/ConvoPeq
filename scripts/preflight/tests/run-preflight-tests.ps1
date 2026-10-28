#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Preflight (A/B/C) 回帰テスト。
.DESCRIPTION
    Owner 承認 2026-10-28（P1）で追加した修正の回帰テスト。

    方針:
      * テスト用入力は一時ディレクトリにのみ作成する。
      * 正本リポジトリ・保全対象・既存の成果物を変更しない。
      * 失敗注入は「一時ディレクトリの欠落・不正値」に対してのみ行う。

    検証する契約:
      1. LF/CRLF と内容一致・不一致の組み合わせ
      2. ファイルの存在・不在
      3. Dropbox metadata の注入あり／なし／不正値
      4. A/B の正常終了・異常終了・起動失敗
      5. 必須出力キーの欠落・不正値
      6. 非 Git ディレクトリで b_exit=5
      7. orchestrate が検証失敗を成功に変換しない
      8. 既存の正本が正常に検証される
#>
[CmdletBinding()]
param(
    [string] $RepositoryRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- パス解決 ---------------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
}
else {
    $RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
}
$guardScript = (Resolve-Path (Join-Path $PSScriptRoot '..\verify_canonical_source.ps1')).Path
$gitScript   = (Resolve-Path (Join-Path $PSScriptRoot '..\verify_git_state.ps1')).Path
$aggScript   = (Resolve-Path (Join-Path $PSScriptRoot '..\aggregate_preflight.ps1')).Path

# --- 結果集計 ---------------------------------------------------------------
$script:PassCount = 0
$script:Failures = New-Object System.Collections.Generic.List[string]

function Assert-Equal {
    param([string]$Name, $Expected, $Actual)
    if ("$Expected" -eq "$Actual") {
        $script:PassCount++
    }
    else {
        $script:Failures.Add("$Name : expected=[$Expected] actual=[$Actual]")
    }
}

function Assert-True {
    param([string]$Name, [bool]$Condition, [string]$Detail = '')
    if ($Condition) { $script:PassCount++ }
    else { $script:Failures.Add("$Name $Detail") }
}

function Get-MapValue {
    param($Map, [string]$Key)
    if ($Map.ContainsKey($Key)) { return "$($Map[$Key])" }
    return '<missing>'
}

function Invoke-PreflightScript {
    param([string]$Path, [string[]]$Arguments)
    $raw = & pwsh -NoProfile -File $Path @Arguments 2>&1
    $code = $LASTEXITCODE
    $text = ($raw | Out-String)
    $map = @{}
    foreach ($line in ($text -split "`r?`n")) {
        $idx = $line.IndexOf('=')
        if ($idx -gt 0) { $map[$line.Substring(0, $idx).Trim()] = $line.Substring($idx + 1).Trim() }
    }
    return [pscustomobject]@{ Exit = $code; Map = $map; Text = $text }
}

function New-Fixture {
    param([string]$Path, [string]$Eol, [string]$Stamp)
    $body = "# Project Extract & Source Code: ConvoPeq{0}{0}> Generated: {1}{0}{0}fixture body{0}" -f $Eol, $Stamp
    [System.IO.File]::WriteAllText($Path, $body, (New-Object System.Text.UTF8Encoding($false)))
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hex = -join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') })
    }
    finally {
        $sha.Dispose()
    }
    return [pscustomobject]@{ Sha = $hex; Size = [long] $bytes.Length }
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) ('preflight-tests-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work -Force | Out-Null
$stamp = '2026-10-28 12:00:00'
$lfFile = Join-Path $work 'canonical-lf.md'
$crlfFile = Join-Path $work 'canonical-crlf.md'
$lf = New-Fixture -Path $lfFile -Eol "`n" -Stamp $stamp
$crlf = New-Fixture -Path $crlfFile -Eol "`r`n" -Stamp $stamp

try {
    # =========================================================================
    # 1) A: LF / CRLF と内容一致・不一致
    # =========================================================================
    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $lfFile, '-BaselineSha256', $lf.Sha, '-BaselineSize', "$($lf.Size)",
        '-BaselineGenerated', $stamp, '-DropboxSize', "$($lf.Size)")
    Assert-Equal 'A/LF 一致 -> exit' 0 $r.Exit
    Assert-Equal 'A/LF 一致 -> eol_class' 'NORMALIZED_LF' (Get-MapValue $r.Map 'eol_class')
    Assert-Equal 'A/LF 一致 -> expected_eol_class' 'NORMALIZED_LF' (Get-MapValue $r.Map 'expected_eol_class')

    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $crlfFile, '-BaselineSha256', $crlf.Sha, '-BaselineSize', "$($crlf.Size)",
        '-BaselineGenerated', $stamp, '-DropboxSize', "$($crlf.Size)")
    Assert-Equal 'A/CRLF は既定(LF)では EOL_ANOMALY -> exit' 6 $r.Exit
    Assert-Equal 'A/CRLF -> eol_class' 'NORMALIZED_CRLF' (Get-MapValue $r.Map 'eol_class')

    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $crlfFile, '-BaselineSha256', $crlf.Sha, '-BaselineSize', "$($crlf.Size)",
        '-BaselineGenerated', $stamp, '-DropboxSize', "$($crlf.Size)", '-ExpectedEolClass', 'NORMALIZED_CRLF')
    Assert-Equal 'A/CRLF + 期待値 CRLF -> exit' 0 $r.Exit

    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $lfFile, '-BaselineSha256', ('0' * 64), '-BaselineSize', "$($lf.Size)",
        '-BaselineGenerated', $stamp, '-DropboxSize', "$($lf.Size)")
    Assert-Equal 'A/内容不一致 -> exit' 1 $r.Exit
    Assert-Equal 'A/内容不一致 -> identity_hash_W' "FAIL:$($lf.Sha)" (Get-MapValue $r.Map 'identity_hash_W')

    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $lfFile, '-BaselineSha256', $lf.Sha, '-BaselineSize', "$($lf.Size + 1)",
        '-BaselineGenerated', $stamp, '-DropboxSize', "$($lf.Size)")
    Assert-Equal 'A/size 不一致 -> exit' 1 $r.Exit

    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $lfFile, '-BaselineSha256', $lf.Sha, '-BaselineSize', "$($lf.Size)",
        '-BaselineGenerated', '1999-01-01 00:00:00', '-DropboxSize', "$($lf.Size)")
    Assert-Equal 'A/provenance 不一致 -> exit' 4 $r.Exit

    # =========================================================================
    # 2) A: 存在・不在
    # =========================================================================
    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', (Join-Path $work 'absent.md'), '-BaselineSha256', $lf.Sha,
        '-BaselineSize', "$($lf.Size)", '-BaselineGenerated', $stamp, '-DropboxSize', "$($lf.Size)")
    Assert-Equal 'A/ファイル不在 -> exit' 3 $r.Exit

    # =========================================================================
    # 3) A: Dropbox metadata の注入あり／なし／不正値
    # =========================================================================
    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $lfFile, '-BaselineSha256', $lf.Sha, '-BaselineSize', "$($lf.Size)",
        '-BaselineGenerated', $stamp)
    Assert-Equal 'A/Dropbox 未注入 -> exit' 2 $r.Exit
    Assert-Equal 'A/Dropbox 未注入 -> dropbox_available' 'false' (Get-MapValue $r.Map 'dropbox_available')

    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $lfFile, '-BaselineSha256', $lf.Sha, '-BaselineSize', "$($lf.Size)",
        '-BaselineGenerated', $stamp, '-DropboxSize', "$($lf.Size)")
    Assert-Equal 'A/Dropbox 注入(size) -> exit' 0 $r.Exit
    Assert-Equal 'A/Dropbox 注入 -> method' 'injected' (Get-MapValue $r.Map 'dropbox_method')
    Assert-Equal 'A/Dropbox 注入 -> size' "$($lf.Size)" (Get-MapValue $r.Map 'dropbox_size')

    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $lfFile, '-BaselineSha256', $lf.Sha, '-BaselineSize', "$($lf.Size)",
        '-BaselineGenerated', $stamp, '-DropboxSize', '0', '-DropboxAvailable', 'true')
    Assert-Equal 'A/Dropbox available=true & size=0 -> exit(fail-closed)' 2 $r.Exit
    Assert-Equal 'A/Dropbox available=true & size=0 -> method' 'injected-invalid-size' (Get-MapValue $r.Map 'dropbox_method')

    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $lfFile, '-BaselineSha256', $lf.Sha, '-BaselineSize', "$($lf.Size)",
        '-BaselineGenerated', $stamp, '-DropboxAvailable', 'notabool')
    Assert-Equal 'A/Dropbox 不正文字列 -> exit(fail-closed)' 2 $r.Exit
    Assert-Equal 'A/Dropbox 不正文字列 -> method' 'injected-invalid-available' (Get-MapValue $r.Map 'dropbox_method')

    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $lfFile, '-BaselineSha256', $lf.Sha, '-BaselineSize', "$($lf.Size)",
        '-BaselineGenerated', $stamp, '-DropboxSize', '-1')
    Assert-Equal 'A/Dropbox 負値 -> exit(fail-closed)' 2 $r.Exit

    # =========================================================================
    # 決定的な検証用リポジトリ（クリーン + detached HEAD）
    #   正本の作業ツリーは未コミット変更を持つことがあるため、
    #   「clean な repo」の期待値は一時リポジトリで固定する。
    # =========================================================================
    $canonicalMd = Join-Path $RepositoryRoot 'ConvoPeq.md'
    $cleanRepo = Join-Path $work 'clean-repo'
    New-Item -ItemType Directory -Path $cleanRepo -Force | Out-Null
    Copy-Item -LiteralPath $canonicalMd -Destination (Join-Path $cleanRepo 'ConvoPeq.md') -Force
    $null = & git -C $cleanRepo init --quiet 2>&1
    $null = & git -C $cleanRepo -c user.email=t@example.invalid -c user.name=t add ConvoPeq.md 2>&1
    $null = & git -C $cleanRepo -c user.email=t@example.invalid -c user.name=t commit --quiet -m init 2>&1
    $null = & git -C $cleanRepo checkout --quiet --detach HEAD 2>&1
    $cleanMd = Join-Path $cleanRepo 'ConvoPeq.md'
    $cleanBytes = [System.IO.File]::ReadAllBytes($cleanMd)
    $cleanShaEngine = [System.Security.Cryptography.SHA256]::Create()
    try { $cleanSha = -join ($cleanShaEngine.ComputeHash($cleanBytes) | ForEach-Object { $_.ToString('x2') }) }
    finally { $cleanShaEngine.Dispose() }
    $cleanSize = [long] $cleanBytes.Length
    $cleanStamp = (Select-String -LiteralPath $cleanMd -Pattern '^> Generated: (.*)$' |
        Select-Object -First 1).Matches.Groups[1].Value.Trim()

    # =========================================================================
    # 4/6) B: 正常・非 Git・detached・起動失敗
    # =========================================================================
    $r = Invoke-PreflightScript $gitScript @('-RepositoryRoot', $cleanRepo)
    Assert-Equal 'B/クリーン detached repo -> exit' 0 $r.Exit
    Assert-Equal 'B/クリーン detached repo -> b_exit' 0 (Get-MapValue $r.Map 'b_exit')
    Assert-Equal 'B/クリーン detached repo -> branch は空' '' (Get-MapValue $r.Map 'branch')

    # 正本リポジトリは作業ツリーの状態に依存するため、契約値と整合だけを検証する
    $r = Invoke-PreflightScript $gitScript @('-RepositoryRoot', $RepositoryRoot)
    Assert-True 'B/正本リポジトリ -> exit は契約値(0/5)' ($r.Exit -in @(0, 5)) "actual=$($r.Exit)"
    Assert-True 'B/正本リポジトリ -> git_dir_type が ABSENT でない' ((Get-MapValue $r.Map 'git_dir_type') -ne 'ABSENT')
    $dirtyLines = @(& git -C $RepositoryRoot status --porcelain 2>&1 |
        Where-Object { $_ -and -not "$_".StartsWith('??') })
    if ($dirtyLines.Count -gt 0) {
        Assert-Equal 'B/正本リポジトリ -> 未コミット変更ありは 5' 5 $r.Exit
    }
    else {
        Assert-Equal 'B/正本リポジトリ -> クリーンなら 0' 0 $r.Exit
    }

    $r = Invoke-PreflightScript $gitScript @('-RepositoryRoot', $work)
    Assert-Equal 'B/非 Git -> exit' 5 $r.Exit
    Assert-Equal 'B/非 Git -> b_exit' 5 (Get-MapValue $r.Map 'b_exit')
    Assert-Equal 'B/非 Git -> git_dir_type' 'ABSENT' (Get-MapValue $r.Map 'git_dir_type')
    Assert-Equal 'B/非 Git -> diagnostics' 'NOT_A_GIT_REPOSITORY' (Get-MapValue $r.Map 'diagnostics')

    $missingRoot = Join-Path $work 'no-such-dir'
    $r = Invoke-PreflightScript $gitScript @('-RepositoryRoot', $missingRoot)
    Assert-True 'B/存在しないルートは例外終了(exit 1)か契約値のいずれか' ($r.Exit -in @(1, 5)) "actual=$($r.Exit)"

    # =========================================================================
    # 5/7) C: orchestrate と注入
    # =========================================================================
    # 8) 既存の正本 ConvoPeq.md が正常に検証される（git 状態に依存しない）
    $r = Invoke-PreflightScript $guardScript @(
        '-CanonicalPath', $canonicalMd,
        '-BaselineSha256', (Get-FileHash -LiteralPath $canonicalMd -Algorithm SHA256).Hash.ToLowerInvariant(),
        '-BaselineSize', "$((Get-Item -LiteralPath $canonicalMd).Length)",
        '-BaselineGenerated', (Select-String -LiteralPath $canonicalMd -Pattern '^> Generated: (.*)$' |
            Select-Object -First 1).Matches.Groups[1].Value.Trim(),
        '-DropboxSize', "$((Get-Item -LiteralPath $canonicalMd).Length)")
    Assert-Equal 'A/正本 ConvoPeq.md -> exit' 0 $r.Exit
    Assert-Equal 'A/正本 ConvoPeq.md -> eol_class' 'NORMALIZED_LF' (Get-MapValue $r.Map 'eol_class')

    # orchestrate は決定的な clean repo に対して実行する
    $baseArgs = @(
        '-RepositoryRoot', $cleanRepo, '-CanonicalPath', $cleanMd,
        '-BaselineSha256', $cleanSha, '-BaselineSize', "$cleanSize",
        '-BaselineGenerated', $cleanStamp)
    $r = Invoke-PreflightScript $aggScript ($baseArgs + @('-Orchestrate', '-DropboxSize', "$cleanSize"))
    Assert-Equal 'C/orchestrate 正常 -> exit' 0 $r.Exit
    Assert-Equal 'C/orchestrate 正常 -> result' 'CANONICAL_OK' (Get-MapValue $r.Map 'result')

    $r = Invoke-PreflightScript $aggScript ($baseArgs + @('-Orchestrate'))
    Assert-Equal 'C/orchestrate Dropbox なし -> exit' 2 $r.Exit
    Assert-Equal 'C/orchestrate Dropbox なし -> result' 'UNVERIFIED_DROPBOX' (Get-MapValue $r.Map 'result')

    $r = Invoke-PreflightScript $aggScript @(
        '-RepositoryRoot', $work, '-CanonicalPath', $cleanMd,
        '-BaselineSha256', $cleanSha, '-BaselineSize', "$cleanSize",
        '-BaselineGenerated', $cleanStamp, '-Orchestrate', '-DropboxSize', "$cleanSize")
    Assert-Equal 'C/orchestrate 非 Git -> exit' 5 $r.Exit
    Assert-True 'C/orchestrate 非 Git -> result が CANONICAL_OK でない' ((Get-MapValue $r.Map 'result') -ne 'CANONICAL_OK')

    $lonelyDir = Join-Path $work 'lonely'
    New-Item -ItemType Directory -Path $lonelyDir -Force | Out-Null
    $lonelyAgg = Join-Path $lonelyDir 'aggregate_preflight.ps1'
    Copy-Item -LiteralPath $aggScript -Destination $lonelyAgg -Force
    $r = Invoke-PreflightScript $lonelyAgg ($baseArgs + @('-Orchestrate', '-DropboxSize', "$cleanSize"))
    Assert-Equal 'C/A/B 不在 -> exit' 70 $r.Exit
    Assert-Equal 'C/A/B 不在 -> result' 'HARNESS_ERROR' (Get-MapValue $r.Map 'result')

    $r = Invoke-PreflightScript $aggScript ($baseArgs + @('-GuardStdout', '', '-PreflightStdout', ''))
    Assert-Equal 'C/注入 空 -> exit' 70 $r.Exit
    Assert-Equal 'C/注入 空 -> result' 'HARNESS_ERROR' (Get-MapValue $r.Map 'result')

    $guardPartial = "native_exit=0`ndiagnostics="
    $gitFull = "staged=0`ntracked_modified=0`nuntracked=0`nstash=0`ngit_dir_type=DIRECTORY`nb_exit=0`ndiagnostics="
    $r = Invoke-PreflightScript $aggScript ($baseArgs + @('-GuardStdout', $guardPartial, '-PreflightStdout', $gitFull))
    Assert-Equal 'C/注入 A の必須キー欠落 -> exit' 70 $r.Exit

    $guardBadExit = "native_exit=9`nidentity_hash_W=OK:x`nidentity_size_W=OK:1`neol_class=NORMALIZED_LF`nprovenance_generated=OK:x`ndropbox_available=true`ndiagnostics="
    $r = Invoke-PreflightScript $aggScript ($baseArgs + @('-GuardStdout', $guardBadExit, '-PreflightStdout', $gitFull))
    Assert-Equal 'C/注入 契約外 exit -> exit' 70 $r.Exit

    $guardFull = "native_exit=0`nidentity_hash_W=OK:x`nidentity_size_W=OK:1`neol_class=NORMALIZED_LF`nprovenance_generated=OK:x`ndropbox_available=true`ndiagnostics="
    $r = Invoke-PreflightScript $aggScript ($baseArgs + @('-GuardStdout', $guardFull, '-PreflightStdout', $gitFull))
    Assert-Equal 'C/注入 正常 -> exit' 0 $r.Exit
    Assert-Equal 'C/注入 正常 -> result' 'CANONICAL_OK' (Get-MapValue $r.Map 'result')
}
finally {
    if (Test-Path -LiteralPath $work) {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# --- 結果出力 ---------------------------------------------------------------
$total = $script:PassCount + $script:Failures.Count
Write-Host "preflight tests: checks=$total passed=$($script:PassCount) failed=$($script:Failures.Count)"
if ($script:Failures.Count -gt 0) {
    foreach ($f in $script:Failures) { Write-Host "[FAIL] $f" }
    exit 1
}
Write-Host '[PASS] preflight regression tests passed'
exit 0
