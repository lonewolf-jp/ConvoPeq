#!/usr/bin/env pwsh
<#
.SYNOPSIS
    [C] Final Aggregator — Phase III implementation.

.DESCRIPTION
    C 層は Final Aggregator であり、唯一の semantic decision point である
    (Owner §5)。以下を唯一の責務とする:

      1. A の観測結果を取得する。
      2. B の観測結果を取得する。
      3. A/B から failure set を構築する。
      4. precedence 3 > 2 > 1 > 6 > 4 > 5 > 0 を **唯一ここで** 適用する。
      5. primary / reasons / diagnostics を生成する。
      6. CANONICAL_OK を **唯一ここで** 宣言する。
      7. final exit を **唯一ここで** 返す。

    A/B は precedence を適用しない。A は native_exit 0..6（5 を含まない）を返し、
    B は b_exit 0/5（互換性の補助表現）を返す。CLEAN_STATE_DEVIATION は
    **C が B の staged / tracked_modified count から生成**する（Owner §3）。

.NOTES
    Contract reference: D-1R-2BU §2.3 / §3.2 / §3.3 / §4.4 / §5.3 / §6.5 / §6.6 / §6.7
    Owner GO: D-1R-2BV Owner 回答 §6 Phase III
#>

[CmdletBinding()]
param(
    # --- C が A/B を起動するための execution parameters（Owner §6） ---
    [Parameter(Mandatory = $true)][string] $RepositoryRoot,

    [Parameter(Mandatory = $true)][string] $CanonicalPath,

    [Parameter(Mandatory = $true)][string] $BaselineSha256,

    [Parameter(Mandatory = $true)][long]   $BaselineSize,

    [Parameter(Mandatory = $true)][string] $BaselineGenerated,

    [string] $DropboxPath = '/ConvoPeq.md',

    [string] $DropboxRev = '',

    # C は実装済み A/B の观测結果を注入文字列として受け取る。
    # これは Owner §6 の「A -> C : A result」「B -> C : B result」に該当する。
    # A へ Git-state を渡す経路は本 layer を経由しない（B -> A は禁止）。
    [string] $GuardStdout = '',

    [string] $PreflightStdout = '',

    # Orchestration mode: C が A / B を起動して観測結果を取得する。
    # Owner §7 は C の process spawn（workflow orchestration）を許可している。
    # B -> A のデータ経路は存在せず、A へ Git-state を渡さない。
    [switch] $Orchestrate,

    # Owner 承認 2026-10-28 (P1-1 / P1-2): A へそのまま転送する。
    # EOL 正常形の指定と Dropbox metadata の注入を C からも行えるようにする。
    [ValidateSet('NORMALIZED_LF', 'NORMALIZED_CRLF')]
    [string] $ExpectedEolClass = 'NORMALIZED_LF',

    [long] $DropboxSize = 0,

    [string] $DropboxAvailable = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# =============================================================================
# C 層の final exit（Owner §4: 0..6 全 7 値）
# =============================================================================
$script:FINAL_EXIT = [ordered]@{
    CANONICAL_OK          = 0
    IDENTITY_MISMATCH     = 1
    UNVERIFIED_DROPBOX    = 2
    LOCAL_MISSING         = 3
    PROVENANCE_FAILURE    = 4
    CLEAN_STATE_DEVIATION = 5
    EOL_ANOMALY           = 6
}

# =============================================================================
# 唯一の precedence 表（Owner §4: 3 > 2 > 1 > 6 > 4 > 5 > 0）
# 配列の添字が precedence rank。rank が小さいほど優先度が高い。
# =============================================================================
$script:PRECEDENCE_ORDER = @(
    'LOCAL_MISSING',          # rank 0, exit 3
    'UNVERIFIED_DROPBOX',     # rank 1, exit 2
    'IDENTITY_MISMATCH',      # rank 2, exit 1
    'EOL_ANOMALY',            # rank 3, exit 6
    'PROVENANCE_FAILURE',     # rank 4, exit 4
    'CLEAN_STATE_DEVIATION'   # rank 5, exit 5
)

# A の native_exit -> failure name（A は 5 を返さない）
$script:A_EXIT_TO_FAILURE = @{
    1 = 'IDENTITY_MISMATCH'
    2 = 'UNVERIFIED_DROPBOX'
    3 = 'LOCAL_MISSING'
    4 = 'PROVENANCE_FAILURE'
    6 = 'EOL_ANOMALY'
}

# =============================================================================
# ハーネス異常（Owner 承認 2026-10-28 / P1-3）
# =============================================================================
# 意味的な exit 0..6 とは区別する。A/B の起動失敗・異常終了・必須出力の欠落は
# 「検証できていない」状態であり、CANONICAL_OK を返してはならない。
$script:HARNESS_ERROR_EXIT = 70
$script:GIT_DISTINGUISHED_EXITS = @(0, 5)
$script:GUARD_ALLOWED_EXITS = @(0, 1, 2, 3, 4, 6)

function New-HarnessError {
    param([string[]] $Reasons)
    return @{
        Result      = 'HARNESS_ERROR'
        Exit        = $script:HARNESS_ERROR_EXIT
        Primary     = 'HARNESS_ERROR'
        Reasons     = ($Reasons -join ',')
        Diagnostics = 'HARNESS_ERROR'
    }
}

function Assert-PreflightOutput {
    <#
    .SYNOPSIS
        A/B の出力が契約を満たすか検証し、違反理由を返す（空なら正常）。
    #>
    param(
        [hashtable] $Map,
        [string]    $Layer,
        [string]    $ExitKey,
        [int[]]     $AllowedExits,
        [string[]]  $RequiredKeys
    )

    $errors = New-Object System.Collections.Generic.List[string]

    if ($null -eq $Map -or $Map.Count -eq 0) {
        $errors.Add("$Layer`:output is empty")
        return $errors.ToArray()
    }
    if (-not $Map.ContainsKey($ExitKey)) {
        $errors.Add("$Layer`:missing $ExitKey")
    }
    foreach ($k in $RequiredKeys) {
        if (-not $Map.ContainsKey($k)) {
            $errors.Add("$Layer`:missing $k")
        }
    }

    if ($Map.ContainsKey($ExitKey)) {
        $parsed = 0
        if (-not [int]::TryParse([string] $Map[$ExitKey], [ref] $parsed)) {
            $errors.Add("$Layer`:$ExitKey is not an integer ($($Map[$ExitKey]))")
        }
        elseif ($AllowedExits -notcontains $parsed) {
            $errors.Add("$Layer`:$ExitKey out of contract ($parsed)")
        }
    }

    return $errors.ToArray()
}

function ConvertFrom-PreflightSchema {
    <#
    .SYNOPSIS
        B の stdout を key=value 形式から hashtable へ変換する。
    #>
    [OutputType([hashtable])]
    param([string] $Stdout)

    $map = @{}
    foreach ($line in ($Stdout -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $idx = $line.IndexOf('=')
        if ($idx -lt 0) { continue }
        $key = $line.Substring(0, $idx).Trim()
        $val = $line.Substring($idx + 1).Trim()
        $map[$key] = $val
    }
    return $map
}

function Get-CleanStateFailure {
    <#
    .SYNOPSIS
        B の観測 count から CLEAN_STATE_DEVIATION を生成する（semantic decision は C のみ）。
    .DESCRIPTION
        Owner §3: 「CLEAN_STATE_DEVIATION failure : C が B の staged / tracked_modified
        観測から生成」。B の b_exit は参照しない（互換性の補助表現に留める）。
        untracked / stash は exit 5 の条件にしない（Owner §4）。
    #>
    [OutputType([bool])]
    param(
        [int] $Staged,
        [int] $TrackedModified
    )

    if (($Staged -gt 0) -or ($TrackedModified -gt 0)) { return $true }
    return $false
}

function Resolve-Precedence {
    <#
    .SYNOPSIS
        唯一の precedence 適用主体。
    .DESCRIPTION
        Owner 不変則:
          primary  = highest-priority failure
          reasons  = all detected failures
          diagnostics = all non-failure diagnostics
          exit     = code(primary)
    #>
    [OutputType([hashtable])]
    param(
        [string[]] $Failures,
        [string[]] $Diagnostics = @()
    )

    $detected = @()
    foreach ($name in $script:PRECEDENCE_ORDER) {
        if ($Failures -contains $name) { $detected += $name }
    }

    if ($detected.Count -eq 0) {
        return @{
            Result     = 'CANONICAL_OK'
            Exit       = $script:FINAL_EXIT.CANONICAL_OK
            Primary    = ''
            Reasons    = ''
            Diagnostics = ($Diagnostics -join ',')
        }
    }

    $primary = $detected[0]
    return @{
        Result     = $primary
        Exit       = $script:FINAL_EXIT[$primary]
        Primary    = $primary
        Reasons    = ($detected -join ',')
        Diagnostics = ($Diagnostics -join ',')
    }
}

function Invoke-FinalAggregator {
    <#
    .SYNOPSIS
        C 層の entry point。唯一の semantic decision point。
    #>
    [OutputType([hashtable])]
    param(
        [hashtable] $GuardMap,
        [hashtable] $GitMap
    )

    $failures = New-Object System.Collections.Generic.List[string]
    $diagnostics = New-Object System.Collections.Generic.List[string]

    # --- A の観測を failure へ反映 -------------------------------------------
    $aNative = 0
    if ($GuardMap.ContainsKey('native_exit')) {
        [void][int]::TryParse([string] $GuardMap['native_exit'], [ref] $aNative)
    }
    if ($script:A_EXIT_TO_FAILURE.ContainsKey($aNative)) {
        $failures.Add($script:A_EXIT_TO_FAILURE[$aNative])
    }
    if ($GuardMap.ContainsKey('diagnostics') -and $GuardMap['diagnostics']) {
        $diagnostics.Add([string] $GuardMap['diagnostics'])
    }

    # --- B の観測から CLEAN_STATE_DEVIATION を生成（semantic decision は C のみ）
    $staged = 0
    $tracked = 0
    if ($GitMap.ContainsKey('staged')) { [void][int]::TryParse([string] $GitMap['staged'], [ref] $staged) }
    if ($GitMap.ContainsKey('tracked_modified')) { [void][int]::TryParse([string] $GitMap['tracked_modified'], [ref] $tracked) }
    if (Get-CleanStateFailure -Staged $staged -TrackedModified $tracked) {
        $failures.Add('CLEAN_STATE_DEVIATION')
    }
    if ($GitMap.ContainsKey('diagnostics') -and $GitMap['diagnostics']) {
        $diagnostics.Add([string] $GitMap['diagnostics'])
    }

    # --- untracked / stash は diagnostic-only（Owner §4） ---------------------
    if ($GitMap.ContainsKey('untracked') -and $GitMap['untracked'] -and [int] $GitMap['untracked'] -gt 0) {
        $diagnostics.Add("UNTRACKED:$($GitMap['untracked'])")
    }
    if ($GitMap.ContainsKey('stash') -and $GitMap['stash'] -and [int] $GitMap['stash'] -gt 0) {
        $diagnostics.Add("STASH:$($GitMap['stash'])")
    }
    # identity_B は diagnostic/provenance であり exit 判定に参加しない（Owner §11）
    if ($GitMap.ContainsKey('identity_B_relation') -and $GitMap['identity_B_relation'] -eq 'CONTENT_DIFFERENCE') {
        $diagnostics.Add('BLOB_DIAGNOSTIC_DIVERGENCE')
    }
    if ($GitMap.ContainsKey('identity_B_relation') -and $GitMap['identity_B_relation'] -eq 'MISSING') {
        $diagnostics.Add('BLOB_UNAVAILABLE')
    }

    # --- 非 Git ディレクトリは「clean と証明できていない」として failure にする --
    # Owner 承認 2026-10-28 (P1-4): B は b_exit=5 を返すが、C は counts しか見ないため
    # ここで ABSENT を拾わないと 0/0 で clean と誤判定してしまう（fail-closed）。
    if ($GitMap.ContainsKey('git_dir_type') -and "$($GitMap['git_dir_type'])" -eq 'ABSENT') {
        if (-not $failures.Contains('CLEAN_STATE_DEVIATION')) { $failures.Add('CLEAN_STATE_DEVIATION') }
        if (-not $diagnostics.Contains('NOT_A_GIT_REPOSITORY')) { $diagnostics.Add('NOT_A_GIT_REPOSITORY') }
    }

    # --- 唯一の precedence 適用 ---------------------------------------------
    return Resolve-Precedence -Failures ([string[]] $failures) -Diagnostics ([string[]] $diagnostics)
}

# =============================================================================
# Entry point: A / B を起動し、観測結果をメモリ上で統合する。
# temp file は使用しない（Owner BU §10.2）。
# =============================================================================
$harnessErrors = New-Object System.Collections.Generic.List[string]
$aExit = $null
$bExit = $null

if ($Orchestrate) {
    $guardScript = Join-Path $PSScriptRoot 'verify_canonical_source.ps1'
    $gitScript   = Join-Path $PSScriptRoot 'verify_git_state.ps1'

    # --- 起動前の存在確認（不在はハーネス異常。CANONICAL_OK にしない） -------
    foreach ($pair in @(@{ Name = 'A'; Path = $guardScript }, @{ Name = 'B'; Path = $gitScript })) {
        if (-not (Test-Path -LiteralPath $pair.Path -PathType Leaf)) {
            $harnessErrors.Add("$($pair.Name):script not found ($($pair.Path))")
        }
    }
    $pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($null -eq $pwshCmd) {
        $harnessErrors.Add('harness:pwsh not found')
    }

    if ($harnessErrors.Count -eq 0) {
        # --- A を起動（C -> A : execution parameters のみ） -----------------
        $aOut = & $pwshCmd.Source -NoProfile -File $guardScript `
            -CanonicalPath     $CanonicalPath `
            -BaselineSha256    $BaselineSha256 `
            -BaselineSize      $BaselineSize `
            -BaselineGenerated $BaselineGenerated `
            -DropboxPath       $DropboxPath `
            -DropboxRev        $DropboxRev `
            -ExpectedEolClass  $ExpectedEolClass `
            -DropboxSize       $DropboxSize `
            -DropboxAvailable  $DropboxAvailable
        $aExit = $LASTEXITCODE
        $aText = ($aOut | Out-String)

        # --- B を起動（C -> B : execution parameters のみ） -----------------
        $bOut = & $pwshCmd.Source -NoProfile -File $gitScript -RepositoryRoot $RepositoryRoot
        $bExit = $LASTEXITCODE
        $bText = ($bOut | Out-String)

        if ($script:GUARD_ALLOWED_EXITS -notcontains [int] $aExit) {
            $harnessErrors.Add("A:child exit out of contract ($aExit)")
        }
        if ($script:GIT_DISTINGUISHED_EXITS -notcontains [int] $bExit) {
            $harnessErrors.Add("B:child exit out of contract ($bExit)")
        }

        $guardMap = ConvertFrom-PreflightSchema -Stdout $aText
        $gitMap   = ConvertFrom-PreflightSchema -Stdout $bText
    }
    else {
        $guardMap = @{}
        $gitMap   = @{}
    }
} else {
    # A / B の観測結果はメモリ上で保持し、temp file を経由しない。
    $guardMap = ConvertFrom-PreflightSchema -Stdout $GuardStdout
    $gitMap   = ConvertFrom-PreflightSchema -Stdout $PreflightStdout
}

# --- 出力契約の検証（注入モード・起動モードの双方に適用） -------------------
foreach ($msg in (Assert-PreflightOutput -Map $guardMap -Layer 'A' -ExitKey 'native_exit' `
        -AllowedExits $script:GUARD_ALLOWED_EXITS `
        -RequiredKeys @('identity_hash_W', 'identity_size_W', 'eol_class', 'provenance_generated', 'dropbox_available'))) {
    $harnessErrors.Add($msg)
}
foreach ($msg in (Assert-PreflightOutput -Map $gitMap -Layer 'B' -ExitKey 'b_exit' `
        -AllowedExits $script:GIT_DISTINGUISHED_EXITS `
        -RequiredKeys @('staged', 'tracked_modified', 'untracked', 'stash', 'git_dir_type'))) {
    $harnessErrors.Add($msg)
}

# --- 子プロセスの exit と報告値の一致（起動モードのみ） ---------------------
if ($Orchestrate -and $harnessErrors.Count -eq 0) {
    if ("$($guardMap['native_exit'])" -ne "$aExit") {
        $harnessErrors.Add("A:reported native_exit ($($guardMap['native_exit'])) != child exit ($aExit)")
    }
    if ("$($gitMap['b_exit'])" -ne "$bExit") {
        $harnessErrors.Add("B:reported b_exit ($($gitMap['b_exit'])) != child exit ($bExit)")
    }
}

if ($harnessErrors.Count -gt 0) {
    Write-Host "[ERROR] preflight harness error: $($harnessErrors -join ' | ')"
    $final = New-HarnessError -Reasons ([string[]] $harnessErrors)
}
else {
    $final = Invoke-FinalAggregator -GuardMap $guardMap -GitMap $gitMap
}

Write-Output ("result={0}"     -f $final.Result)
Write-Output ("final_exit={0}" -f $final.Exit)
Write-Output ("primary={0}"    -f $final.Primary)
Write-Output ("reasons={0}"    -f $final.Reasons)
Write-Output ("diagnostics={0}" -f $final.Diagnostics)

exit ([int] $final.Exit)