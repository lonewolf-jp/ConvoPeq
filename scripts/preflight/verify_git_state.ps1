#!/usr/bin/env pwsh
<#
.SYNOPSIS
    [B] Git-State Preflight — Phase II implementation.

.DESCRIPTION
    B 層は Git-State Preflight である。Owner 契約により以下を満たす:
      - Owner 承認済み read-only Git allow-list と argument grammar のみを使用
      - VCS mutation / object write を行わない
      - precedence を適用しない（semantic decision は C のみ）
      - CLEAN_STATE_DEVIATION を生成しない（B は count のみ観測する）
      - A を呼び出さない / A にデータを送らない     (Owner §5: B -> A は禁止)
      - native exit は 0 または 5 のみ               (Owner §3: 5 は補助表現)

    Owner は git cat-file を read-only blob access に限定し、
    write mode を受け付けないことを明示している。

.NOTES
    Contract reference: D-1R-2BU §2.2 / §3.2 / §5.2 / §6.3 / §6.4 / §7
    Owner GO: D-1R-2BV Owner 回答 §6 Phase II
#>

[CmdletBinding()]
param(
    # --- C から B へ渡す execution parameters のみ（Owner §6） ---
    [Parameter(Mandatory = $true)]
    [string] $RepositoryRoot,

    [string] $CanonicalRef = 'HEAD:ConvoPeq.md',

    [string[]] $RemoteRefs = @('origin/main', 'origin/bright-falcon')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# =============================================================================
# B 層の exit（Owner §3: b_exit は互換性上の補助表現。precedence には参加しない）
# =============================================================================
$script:PREFLIGHT_EXIT = [ordered]@{
    CLEAN = 0
    DIRTY = 5
}

# =============================================================================
# Owner 承認済み read-only Git allow-list（Owner §6 / BU §6.3）
# deny-list の command は一切この配列に入れない。
# =============================================================================
$script:GIT_READONLY_ALLOWED = @(
    @('status', '--porcelain'),
    @('rev-parse'),
    @('ls-files', '--others'),
    @('stash', 'list'),
    @('cat-file', 'blob'),        # Owner §6: read-only blob access に限定
    @('branch', '--show-current')
)

# Owner が全面禁止した mutation command（Owner §6 / BU §6.4 B-M1）
$script:GIT_MUTATION_DENIED = @(
    'checkout', 'add', 'commit', 'push', 'reset', 'rebase', 'merge',
    'cherry-pick', 'revert', 'clean', 'rm', 'mv', 'update-ref', 'symbolic-ref',
    'cat-file', 'hash-object', 'commit-tree', 'mktree', 'worktree', 'filter-branch'
)

function Invoke-GitReadOnly {
    <#
    .SYNOPSIS
        allow-list に含まれる git read-only command のみを実行する。
    .DESCRIPTION
        deny-list に該当する command は、この関数を経由しないことを保証する。
        引数の grammar も検証し、write option を拒否する。
    #>
    [OutputType([string[]])]
    param(
        [Parameter(Mandatory = $true)][string[]] $GitArgs
    )

    if ($GitArgs.Count -eq 0) {
        throw "git argument grammar violation: empty argument list"
    }

    $sub = $GitArgs[0]

    # --- deny-list 検査（Owner §6） -------------------------------------------
    if ($script:GIT_MUTATION_DENIED -contains $sub) {
        # cat-file は read-only blob を許可するため、後段で個別に扱う
        if ($sub -ne 'cat-file') {
            throw "VCS mutation command denied: git $sub"
        }
    }

    # --- allow-list 検査 -----------------------------------------------------
    $matched = $false
    foreach ($allowed in $script:GIT_READONLY_ALLOWED) {
        if ($allowed[0] -ne $sub) { continue }
        # git cat-file は 'blob' subcommand のみ許可（Owner §6）
        if ($sub -eq 'cat-file') {
            if ($GitArgs.Count -ge 2 -and $GitArgs[1] -eq 'blob') {
                $matched = $true
            }
            continue
        }
        $matched = $true
    }
    if (-not $matched) {
        throw "git command not in read-only allow-list: git $sub"
    }

    # --- write option の拒否 -------------------------------------------------
    foreach ($a in $GitArgs) {
        if ($a -eq '-w' -or $a -eq '--write') {
            throw "git object write option denied: $a"
        }
        if ($a -like '--*' -and ($a -match 'write|batch')) {
            throw "git write-mode option denied: $a"
        }
    }

    # --- 実行 ---------------------------------------------------------------
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = 'git'
    foreach ($a in $GitArgs) { [void] $psi.ArgumentList.Add($a) }
    $psi.WorkingDirectory = $RepositoryRoot
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false

    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()

    if ($proc.ExitCode -ne 0) {
        throw "git $($GitArgs -join ' ') failed (exit=$($proc.ExitCode)): $stderr"
    }
    $lines = @($stdout -split "`r?`n" | Where-Object { $_ -ne '' -and $null -ne $_ })
    # 戻り値を PowerShell の pipeline 自動展開から完全に守るため、
    # PSCustomObject の単一オブジェクトとして返す。呼び出し側は
    # $r.Lines / $r.Count を使う。スカラー切り詰めも配列化も発生しない。
    return [PSCustomObject]@{
        Lines = [string[]] $lines
        Count = [int] $lines.Count
    }
}

function Invoke-GitReadOnlyBinary {
    <#
    .SYNOPSIS
        git read-only command の stdout を binary bytes として取得する。
    .DESCRIPTION
        text 変換（改行・encoding）による byte fidelity の毀損を避けるため、
        StandardOutput を Stream として読む。blob 取得専用。
        allow-list / deny-list / write-option 検査は Invoke-GitReadOnly と同じ。
    #>
    [OutputType([byte[]])]
    param(
        [Parameter(Mandatory = $true)][string[]] $GitArgs
    )

    if ($GitArgs.Count -eq 0) {
        throw "git argument grammar violation: empty argument list"
    }
    $sub = $GitArgs[0]
    if ($sub -ne 'cat-file') {
        throw "binary mode is restricted to 'git cat-file blob' only (given: $sub)"
    }
    if ($GitArgs.Count -lt 3 -or $GitArgs[1] -ne 'blob') {
        throw "git cat-file binary mode requires the 'blob' subcommand"
    }
    foreach ($a in $GitArgs) {
        if ($a -eq '-w' -or $a -eq '--write' -or $a -like '--batch*') {
            throw "git object write option denied: $a"
        }
    }

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = 'git'
    foreach ($a in $GitArgs) { [void] $psi.ArgumentList.Add($a) }
    $psi.WorkingDirectory = $RepositoryRoot
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false

    $proc = [System.Diagnostics.Process]::Start($psi)
    $outStream = $proc.StandardOutput.BaseStream
    $ms = [System.IO.MemoryStream]::new()
    try {
        $outStream.CopyTo($ms)
        $stderr = $proc.StandardError.ReadToEnd()
        $proc.WaitForExit()
        if ($proc.ExitCode -ne 0) {
            throw "git $($GitArgs -join ' ') failed (exit=$($proc.ExitCode)): $stderr"
        }
        return [byte[]] $ms.ToArray()
    } finally {
        $ms.Dispose()
    }
}

function Get-FirstLine {
    <#
    .SYNOPSIS
        read-only 結果の先頭行を安全に取り出す。
    .DESCRIPTION
        detached HEAD では `git branch --show-current` が 0 行を返す。
        Set-StrictMode 3.0 以降は空配列への添字アクセスが例外になるため、
        ここで明示的に空文字へ落とす（Owner 承認 2026-10-28 / P1-4 の回帰）。
    #>
    [OutputType([string])]
    param(
        [AllowNull()][string[]] $Lines
    )

    if ($null -eq $Lines -or $Lines.Count -eq 0) {
        return ''
    }
    return "$($Lines[0])"
}

function Resolve-RepositoryRoot {
    <#
    .SYNOPSIS
        worktree root を検出し、worktree pointer file を正しく処理する。
    .DESCRIPTION
        本リポジトリは linked worktree であり、.git はディレクトリではなく
        "gitdir: <path>" 形式の pointer file である（Owner §6 Phase II）。
        git CLI の worktree-awareness に任せ、B 側は固有ロジックを持たない。
    #>
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory = $true)][string] $Root
    )

    $gitPath = Join-Path $Root '.git'
    $pointerFile = $false
    $gitDirType = 'ABSENT'

    if (Test-Path -LiteralPath $gitPath -PathType Leaf) {
        $pointerFile = $true
        $gitDirType = 'POINTER_FILE'
        $pointer = (Get-Content -LiteralPath $gitPath -TotalCount 1 -ErrorAction Stop)
        if ($pointer -notmatch '^gitdir:\s*(.+)$') {
            throw "invalid .git pointer file (expected 'gitdir: <path>')"
        }
    } elseif (Test-Path -LiteralPath $gitPath -PathType Container) {
        $gitDirType = 'DIRECTORY'
    }

    $commonDir = Get-FirstLine -Lines (Invoke-GitReadOnly -GitArgs @('rev-parse', '--git-common-dir')).Lines
    $gitDir    = Get-FirstLine -Lines (Invoke-GitReadOnly -GitArgs @('rev-parse', '--git-dir')).Lines
    $toplevel  = Get-FirstLine -Lines (Invoke-GitReadOnly -GitArgs @('rev-parse', '--show-toplevel')).Lines

    return @{
        Root            = $Root
        PointerFile     = $pointerFile
        GitDirType      = $gitDirType
        GitDir          = $gitDir
        CommonDir       = $commonDir
        Toplevel        = $toplevel
    }
}

function Get-GitStateObservation {
    <#
    .SYNOPSIS
        B 層の entry point。Git-state を観測のみする。
    .DESCRIPTION
        precedence を適用せず、CLEAN_STATE_DEVIATION を生成しない。
        CLEAN_STATE_DEVIATION は C が B の count から生成する（Owner §3）。
    #>
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory = $true)][string] $RepositoryRoot,
        [string] $CanonicalRef = 'HEAD:ConvoPeq.md',
        [string[]] $RemoteRefs = @('origin/main', 'origin/bright-falcon')
    )

    # --- 非 Git ディレクトリは契約どおり b_exit=5（Owner 承認 2026-10-28 / P1-4） ---
    # 例外終了（exit 1）は契約外であり、呼び出し側の分岐を壊す。
    # 「契約上の検証結果」として 5 を返し、想定外の例外は再送出する。
    try {
        $null = Invoke-GitReadOnly -GitArgs @('rev-parse', '--git-dir')
    }
    catch {
        if ("$($_.Exception.Message)" -match 'not a git repository') {
            return [ordered]@{
                'staged'              = 0
                'tracked_modified'    = 0
                'untracked'           = 0
                'stash'               = 0
                'head'                = 'NA'
                'branch'              = 'NA'
                'git_dir_type'        = 'ABSENT'
                'git_common_dir'      = 'NA'
                'identity_hash_B'     = 'NA'
                'identity_size_B'     = 'NA'
                'identity_B_relation' = 'MISSING'
                'b_exit'              = $script:PREFLIGHT_EXIT.DIRTY
                'diagnostics'         = 'NOT_A_GIT_REPOSITORY'
            }
        }
        throw
    }

    # --- repository root / worktree pointer（Owner §6 Phase II） -------------
    $repo = Resolve-RepositoryRoot -Root $RepositoryRoot
    if ($repo.PointerFile) {
        $repoDiagnostics = 'WORKTREE_POINTER_FILE'
    } else {
        $repoDiagnostics = ''
    }

    # --- status --porcelain から staged / tracked_modified / untracked -------
    $porcelain = (Invoke-GitReadOnly -GitArgs @('status', '--porcelain')).Lines

    $staged          = 0
    $trackedModified = 0
    $untracked       = 0
    foreach ($line in $porcelain) {
        if ($line.Length -lt 2) { continue }
        $indexStatus = $line[0]
        $workStatus  = $line[1]
        # '??' は untracked
        if ($indexStatus -eq '?' -and $workStatus -eq '?') {
            $untracked++
            continue
        }
        if ($indexStatus -ne ' ' -and $indexStatus -ne '?') { $staged++ }
        if ($workStatus -ne ' ' -and $workStatus -ne '?') { $trackedModified++ }
    }

    # --- stash ---------------------------------------------------------------
    $stash = [int] (Invoke-GitReadOnly -GitArgs @('stash', 'list')).Count

    # --- HEAD / branch / remotes ---------------------------------------------
    $head   = Get-FirstLine -Lines (Invoke-GitReadOnly -GitArgs @('rev-parse', 'HEAD')).Lines
    $branch = Get-FirstLine -Lines (Invoke-GitReadOnly -GitArgs @('branch', '--show-current')).Lines

    $remoteMap = [ordered]@{}
    foreach ($r in $RemoteRefs) {
        $key = $r -replace '[^A-Za-z0-9]', '_'
        try {
            $remoteMap[$key] = Get-FirstLine -Lines (Invoke-GitReadOnly -GitArgs @('rev-parse', $r)).Lines
        } catch {
            $remoteMap[$key] = 'NA'
        }
    }

    # --- identity_B（read-only cat-file blob、Owner §6） ----------------------
    # Owner 契約: identity_B は canonical identity_W の判定軸ではない。
    # relation は次の 3 値を取り、exit 判定には一切参加しない:
    #   NORMALIZED_DIFFERENCE : worktree を LF 正規化した結果が blob と一致
    #   CONTENT_DIFFERENCE   : LF 正規化では説明できない差
    #   MISSING              : blob を取得できない
    $hashB = 'NA'
    $sizeB = 'NA'
    $relation = 'MISSING'
    try {
        # blob を binary として直接取得し、text 変換による byte 毀損を避ける
        $blobBytes = Invoke-GitReadOnlyBinary -GitArgs @('cat-file', 'blob', $CanonicalRef)

        $shaCalc = [System.Security.Cryptography.SHA256]::Create()
        try {
            $blobSha = -join ($shaCalc.ComputeHash($blobBytes) | ForEach-Object { $_.ToString('x2') })
        } finally {
            $shaCalc.Dispose()
        }
        $hashB = $blobSha
        $sizeB = [long] $blobBytes.Length

        # worktree bytes を LF 正規化して比較する（C3: blob を identity と比較しない）
        $canonicalFile = Join-Path $RepositoryRoot 'ConvoPeq.md'
        if (Test-Path -LiteralPath $canonicalFile -PathType Leaf) {
            $fsW = [System.IO.File]::Open(
                $canonicalFile,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read,
                [System.IO.FileShare]::ReadWrite)
            try {
                $wt = New-Object byte[] $fsW.Length
                $rd = 0
                while ($rd -lt $fsW.Length) {
                    $n = $fsW.Read($wt, $rd, ($fsW.Length - $rd))
                    if ($n -le 0) { break }
                    $rd += $n
                }
                if ($rd -lt $fsW.Length) { $wt = $wt[0..($rd - 1)] }
            } finally {
                $fsW.Dispose()
            }
            $wtLf = [System.Text.RegularExpressions.Regex]::Replace(
                [System.Text.Encoding]::UTF8.GetString($wt), "`r`n", "`n")
            $wtLfBytes = [System.Text.Encoding]::UTF8.GetBytes($wtLf)
            $shaNorm = [System.Security.Cryptography.SHA256]::Create()
            try {
                $wtLfSha = -join ($shaNorm.ComputeHash($wtLfBytes) | ForEach-Object { $_.ToString('x2') })
            } finally {
                $shaNorm.Dispose()
            }
            if ($wtLfSha -eq $blobSha) {
                $relation = 'NORMALIZED_DIFFERENCE'
            } else {
                $relation = 'CONTENT_DIFFERENCE'
            }
        } else {
            $relation = 'MISSING'
        }
    } catch {
        $hashB = 'NA'
        $sizeB = 'NA'
        $relation = 'MISSING'
    }

    # --- b_exit は互換性上の補助表現（Owner §3） ------------------------------
    $bExit = $script:PREFLIGHT_EXIT.CLEAN
    if ($staged -gt 0 -or $trackedModified -gt 0) {
        $bExit = $script:PREFLIGHT_EXIT.DIRTY
    }

    $diagnostics = @($repoDiagnostics) | Where-Object { $_ -ne '' }

    return [ordered]@{
        'staged'                 = $staged
        'tracked_modified'       = $trackedModified
        'untracked'              = $untracked
        'stash'                  = $stash
        'head'                   = $head
        'branch'                 = $branch
        'git_dir_type'           = $repo.GitDirType
        'git_common_dir'         = $repo.CommonDir
        'identity_hash_B'        = $hashB
        'identity_size_B'        = $sizeB
        'identity_B_relation'    = $relation
        'b_exit'                 = $bExit
        'diagnostics'            = ($diagnostics -join ',')
    }
}

# =============================================================================
# Entry point
# =============================================================================
$obs = Get-GitStateObservation `
    -RepositoryRoot $RepositoryRoot `
    -CanonicalRef   $CanonicalRef `
    -RemoteRefs     $RemoteRefs

foreach ($key in $obs.Keys) {
    Write-Output ("{0}={1}" -f $key, $obs[$key])
}
foreach ($key in @('origin_main', 'origin_bright_falcon')) {
    if ($obs.Contains($key)) {
        Write-Output ("{0}={1}" -f $key, $obs[$key])
    }
}

exit ([int] $obs['b_exit'])