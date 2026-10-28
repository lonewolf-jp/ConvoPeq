#!/usr/bin/env pwsh
<#
.SYNOPSIS
    [A] Canonical Source Guard — Phase I implementation.

.DESCRIPTION
    A 層は Canonical Source Guard である。Owner 契約により以下を満たす:
      - Git-state を一切参照・受領しない            (Owner §5)
      - child process を生成しない                  (Owner §7)
      - filesystem write / temp file を行わない    (Owner BQ 回答 §3)
      - Dropbox access は read-only GET に限定      (Owner §5)
      - CANONICAL_OK を判定しない                  (Owner §3)
      - native exit は 0,1,2,3,4,6 のみ          (Owner §4)

    A は観測のみを行い、意味的判定（precedence / CANONICAL_OK / final exit）は
    C（Final Aggregator）の責務である。

.NOTES
    Contract reference: D-1R-2BU §2.1 / §3.1 / §5.1 / §6.1 / §6.2
    Owner GO: D-1R-2BV Owner 回答 §5 / §6 Phase I
#>

[CmdletBinding()]
param(
    # --- C から A へ渡す execution parameters のみ（Owner §6） ---
    # Git-state 関連のパラメータは存在しない（Owner §5: B -> A は禁止）
    [Parameter(Mandatory = $true)]
    [string] $CanonicalPath,

    [Parameter(Mandatory = $true)]
    [string] $BaselineSha256,

    [Parameter(Mandatory = $true)]
    [long]   $BaselineSize,

    [Parameter(Mandatory = $true)]
    [string] $BaselineGenerated,

    # Dropbox canonical の metadata 検証に使う identity。
    # Owner が提示した Dropbox metadata が未提供の場合は unavailable 扱いとする。
    [string] $DropboxPath = '/ConvoPeq.md',

    # Owner 提供 Dropbox rev（存在する場合は size 整合の補助情報）
    [string] $DropboxRev = '',

    # Owner 承認 2026-10-28 (P1-1): EOL 判定の正常形を呼び出し側が指定できる。
    # 既定は .gitattributes の `*.md text eol=lf` に合わせて LF。
    # CRLF を正常形とする運用に切り替える場合のみ NORMALIZED_CRLF を渡す。
    [ValidateSet('NORMALIZED_LF', 'NORMALIZED_CRLF')]
    [string] $ExpectedEolClass = 'NORMALIZED_LF',

    # Owner 承認 2026-10-28 (P1-2): Dropbox metadata は呼び出し側から注入できる。
    # ローカル同期フォルダの存在を必須条件にしない。
    # -DropboxSize が 1 以上なら「取得済み」、未指定(0)なら未取得として fail-closed。
    [long] $DropboxSize = 0,

    # 'true' / 'false' を明示注入する場合に使用（空文字なら DropboxSize から判定）
    [string] $DropboxAvailable = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# =============================================================================
# A 層 native exit 定数（Owner §4: 0,1,2,3,4,6 のみ。5 は存在しない）
# =============================================================================
$script:GUARD_NATIVE_EXIT = [ordered]@{
    OK                 = 0
    IDENTITY_MISMATCH  = 1
    UNVERIFIED_DROPBOX = 2
    LOCAL_MISSING      = 3
    PROVENANCE_FAILURE = 4
    EOL_ANOMALY        = 6
}

# EOL 正式 6 分類（Owner §2: E-7 は追加しない）
$script:EOL_CLASS_CR   = 'NORMALIZED_CRLF'
$script:EOL_CLASS_LF   = 'NORMALIZED_LF'
$script:EOL_CLASS_MAC  = 'OLD_MAC_EOL'
$script:EOL_CLASS_M_CR = 'MIXED_CRLF_BARE_CR'
$script:EOL_CLASS_M_LF = 'MIXED_CRLF_BARE_LF'
$script:EOL_CLASS_MIX  = 'MIXED_OTHER_EOL'
# Owner G-11 CLOSED: UNCLASSIFIED_NO_EOL は 6 分類のいずれにも該当しない
# 入力異常であり、eol_class の第7値ではない。native exit 6 に統合し、
# 詳細は diagnostics の reason として保持する。
$script:EOL_CLASS_NONE  = 'UNCLASSIFIED'
$script:EOL_REASON_NONE = 'EOL_REASON:UNCLASSIFIED_NO_EOL'

function Get-CanonicalSha256 {
    <#
    .SYNOPSIS
        worktree bytes の SHA-256 と byte size を算出する。
    .DESCRIPTION
        git を経由しない。読み取り専用ストリームのみを使用する。
        Owner C-1 / Owner §5「A は Git-state を一切参照しない」に適合。
    #>
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory = $true)][string] $Path
    )

    # read-only FileStream（write mode ではない）
    $stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $hashBytes = $sha.ComputeHash($stream)
        } finally {
            $sha.Dispose()
        }
        $hex = -join ($hashBytes | ForEach-Object { $_.ToString('x2') })
        return @{ Sha256 = $hex; Size = [long] $stream.Length }
    } finally {
        $stream.Dispose()
    }
}

function Get-EolClassification {
    <#
    .SYNOPSIS
        bytes の EOL 分類を正式 6 分類で判定する。
    .DESCRIPTION
        単一パス走査。bare_cr / bare_lf / crlf を数えて Owner §2 の表で分類する。
        3 値すべてが 0 の場合は UNCLASSIFIED（6 分類のいずれにも該当しない）と
        返し、native exit 6（EOL_ANOMALY）に統合する（Owner G-11 CLOSED）。
    #>
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory = $true)][byte[]] $Bytes
    )

    $bareCr = 0
    $bareLf = 0
    $crlf   = 0

    $n = $Bytes.Length
    $i = 0
    while ($i -lt $n) {
        $b = $Bytes[$i]
        if ($b -eq 0x0D) {
            if (($i + 1) -lt $n -and $Bytes[$i + 1] -eq 0x0A) {
                $crlf++
                $i += 2
                continue
            }
            $bareCr++
            $i++
            continue
        }
        if ($b -eq 0x0A) {
            $bareLf++
            $i++
            continue
        }
        $i++
    }

    if ($bareCr -eq 0 -and $bareLf -eq 0 -and $crlf -gt 0) {
        $class = $script:EOL_CLASS_CR
    } elseif ($bareCr -eq 0 -and $bareLf -gt 0 -and $crlf -eq 0) {
        $class = $script:EOL_CLASS_LF
    } elseif ($bareCr -gt 0 -and $bareLf -eq 0 -and $crlf -eq 0) {
        $class = $script:EOL_CLASS_MAC
    } elseif ($bareCr -gt 0 -and $bareLf -eq 0 -and $crlf -gt 0) {
        $class = $script:EOL_CLASS_M_CR
    } elseif ($bareCr -eq 0 -and $bareLf -gt 0 -and $crlf -gt 0) {
        $class = $script:EOL_CLASS_M_LF
    } elseif ($bareCr -gt 0 -and $bareLf -gt 0 -and $crlf -gt 0) {
        $class = $script:EOL_CLASS_MIX
    } else {
        # bare CR = bare LF = CRLF = 0
        $class = $script:EOL_CLASS_NONE
    }

    return @{
        Class  = $class
        BareCr = $bareCr
        BareLf = $bareLf
        Crlf   = $crlf
    }
}

function Get-GeneratedStamp {
    <#
    .SYNOPSIS
        先頭領域から "Generated: YYYY-MM-DD HH:MM:SS" を抽出する。
    .DESCRIPTION
        canonical は "<title>\n\n> Generated: ...\n\n" 形式のため先頭 20 行で足りる。
        UNPARSEABLE は provenance failure（native exit 4）として扱う。
    #>
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory = $true)][byte[]] $Bytes
    )

    # BOM を考慮しつつ先頭 4096 byte を UTF-8 として解釈する
    $take = [Math]::Min(4096, $Bytes.Length)
    $head = [byte[]]::new($take)
    [Array]::Copy($Bytes, 0, $head, 0, $take)
    $text = [System.Text.Encoding]::UTF8.GetString($head)

    $m = [regex]::Match($text, '(?m)^>\s*Generated:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$')
    if (-not $m.Success) {
        return @{ Status = 'UNPARSEABLE'; Stamp = '' }
    }
    return @{ Status = 'OK'; Stamp = $m.Groups[1].Value }
}

function Get-DropboxMetadata {
    <#
    .SYNOPSIS
        Dropbox canonical の metadata を read-only で解決する。
    .DESCRIPTION
        Owner §5: A の Dropbox access は read-only GET に限定する。
        Owner 承認 2026-10-28 (P1-2): ローカル同期フォルダを必須条件にせず、
        呼び出し側（MCP/HTTP で取得した metadata）からの注入を第一経路とする。
        解決できない場合は unavailable とし、CANONICAL_OK を permit しない。
    #>
    [OutputType([hashtable])]
    param(
        [string] $Path = '/ConvoPeq.md',
        [string] $Rev   = '',
        [long]   $InjectedSize = 0,
        [string] $InjectedAvailable = ''
    )

    $base = @{
        Path     = $Path
        Rev      = $Rev
        Modified = $null
        Size     = $null
        Method   = 'injected'
    }

    # --- 1) 明示的な available フラグ（最優先） --------------------------------
    if (-not [string]::IsNullOrWhiteSpace($InjectedAvailable)) {
        $flag = $false
        if (-not [bool]::TryParse($InjectedAvailable, [ref] $flag)) {
            # 不正値は「取得済み」と誤認させない（fail-closed）
            $base.Available = $false
            $base.Method = 'injected-invalid-available'
            return $base
        }
        if (-not $flag) {
            $base.Available = $false
            $base.Method = 'injected-unavailable'
            return $base
        }
        if ($InjectedSize -le 0) {
            # available=true だが size が無い/不正 → 検証に使えない
            $base.Available = $false
            $base.Method = 'injected-invalid-size'
            return $base
        }
        $base.Available = $true
        $base.Size = [long] $InjectedSize
        return $base
    }

    # --- 2) size の注入 -------------------------------------------------------
    if ($InjectedSize -lt 0) {
        $base.Available = $false
        $base.Method = 'injected-invalid-size'
        return $base
    }
    if ($InjectedSize -gt 0) {
        $base.Available = $true
        $base.Size = [long] $InjectedSize
        return $base
    }

    # --- 3) 後方互換: ローカル同期フォルダの probe ----------------------------
    $base.Method = 'local-sync-folder-probe'
    $dropboxRoot = Join-Path $env:USERPROFILE 'Dropbox'
    if (Test-Path -LiteralPath $dropboxRoot) {
        $candidate = Join-Path $dropboxRoot ($Path -replace '^/', '')
        if (Test-Path -LiteralPath $candidate) {
            $base.Available = $true
            $base.Size = [long] (Get-Item -LiteralPath $candidate).Length
            return $base
        }
    }

    $base.Available = $false
    return $base
}

function Invoke-CanonicalSourceGuard {
    <#
    .SYNOPSIS
        A 層の entry point。観測のみを行い、意味的判定は行わない。
    .OUTPUTS
        A の観測結果（stdout schema。BU §5.1）。
        result / primary / reasons は出力しない（Owner §3: C のみが宣言）。
    #>
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory = $true)][string] $CanonicalPath,
        [Parameter(Mandatory = $true)][string] $BaselineSha256,
        [Parameter(Mandatory = $true)][long]   $BaselineSize,
        [Parameter(Mandatory = $true)][string] $BaselineGenerated,
        [string] $DropboxPath = '/ConvoPeq.md',
        [string] $DropboxRev  = ''
    )

    $diagnostics = New-Object System.Collections.Generic.List[string]
    $expectedSha = $BaselineSha256.ToLowerInvariant()

    # --- Dropbox metadata（read-only GET のみ） -------------------------------
    $dropbox = Get-DropboxMetadata -Path $DropboxPath -Rev $DropboxRev `
        -InjectedSize $DropboxSize -InjectedAvailable $DropboxAvailable

    # --- local canonical の存在確認 ------------------------------------------
    if (-not (Test-Path -LiteralPath $CanonicalPath -PathType Leaf)) {
        return [ordered]@{
            'identity_hash_W'          = 'FAIL:NA'
            'identity_size_W'          = 'FAIL:NA'
            'eol_class'                = 'NA'
            'eol_bare_cr'              = 'NA'
            'eol_bare_lf'              = 'NA'
            'eol_crlf'                 = 'NA'
            'provenance_generated'     = 'UNPARSEABLE:NA'
            'dropbox_available'        = ([string] $dropbox.Available).ToLowerInvariant()
            'dropbox_rev'              = $dropbox.Rev
            'dropbox_size'             = $(if ($null -ne $dropbox.Size) { "$($dropbox.Size)" } else { 'NA' })
            'dropbox_method'           = "$($dropbox.Method)"
            'dropbox_modified'         = 'NA'
            'native_exit'              = $script:GUARD_NATIVE_EXIT.LOCAL_MISSING
            'diagnostics'              = ''
        }
    }

    # --- SHA-256 / byte size（git を経由しない） ------------------------------
    $shaResult = Get-CanonicalSha256 -Path $CanonicalPath

    $shaOk = ($shaResult.Sha256 -eq $expectedSha)
    $sizeOk = ($shaResult.Size -eq $BaselineSize)

    # --- EOL classification（正式 6 分類 + UNCLASSIFIED） ---------------------
    $fs = [System.IO.File]::Open(
        $CanonicalPath,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite)
    try {
        $all = New-Object byte[] $fs.Length
        $read = 0
        while ($read -lt $fs.Length) {
            $n = $fs.Read($all, $read, ($fs.Length - $read))
            if ($n -le 0) { break }
            $read += $n
        }
        if ($read -lt $fs.Length) { $all = $all[0..($read - 1)] }
    } finally {
        $fs.Dispose()
    }

    $eol = Get-EolClassification -Bytes $all
    # Owner 承認 2026-10-28 (P1-1): 正常形は呼び出し側の指定に従う（既定 NORMALIZED_LF）。
    # EOL 判定は「内容の一致判定」とは独立であり、混同しない。
    $eolMatches = ($eol.Class -eq $ExpectedEolClass)
    if (-not $eolMatches) {
        $diagnostics.Add("EOL_CLASS:$($eol.Class)")
        $diagnostics.Add("EOL_EXPECTED:$ExpectedEolClass")
    }
    if ($eol.Class -eq $script:EOL_CLASS_NONE) {
        # Owner G-11 CLOSED: 第7分類にはせず exit 6 に統合し reason を保持する
        $diagnostics.Add($script:EOL_REASON_NONE)
    }

    # --- provenance（Generated:） --------------------------------------------
    $gen = Get-GeneratedStamp -Bytes $all
    $genOk = ($gen.Status -eq 'OK' -and $gen.Stamp -eq $BaselineGenerated)

    # --- native exit の決定（意味的 precedence ではない） ---------------------
    # Owner §4: A は自分の観測範囲の failure だけを表現する。
    $nativeExit = $script:GUARD_NATIVE_EXIT.OK
    if ($dropbox.Available -ne $true) {
        # Owner §3: Dropbox unavailable → exit 2、CANONICAL_OK を permit しない
        $nativeExit = $script:GUARD_NATIVE_EXIT.UNVERIFIED_DROPBOX
    } elseif (-not ($shaOk -and $sizeOk)) {
        # Owner §3: identity_W mismatch は EOL anomaly より上位（precedence 1 < 6）
        $nativeExit = $script:GUARD_NATIVE_EXIT.IDENTITY_MISMATCH
    } elseif (-not $genOk) {
        $nativeExit = $script:GUARD_NATIVE_EXIT.PROVENANCE_FAILURE
    } elseif (-not $eolMatches) {
        $nativeExit = $script:GUARD_NATIVE_EXIT.EOL_ANOMALY
    }

    return [ordered]@{
        'identity_hash_W'      = ("{0}:{1}" -f $(if ($shaOk) { 'OK' } else { 'FAIL' }), $shaResult.Sha256)
        'identity_size_W'      = ("{0}:{1}" -f $(if ($sizeOk) { 'OK' } else { 'FAIL' }), $shaResult.Size)
        'eol_class'            = $eol.Class
        'expected_eol_class'   = $ExpectedEolClass
        'eol_bare_cr'          = $eol.BareCr
        'eol_bare_lf'          = $eol.BareLf
        'eol_crlf'             = $eol.Crlf
        'provenance_generated' = ("{0}:{1}" -f $(if ($genOk) { 'OK' } else { 'FAIL' }), $(if ($gen.Stamp) { $gen.Stamp } else { 'NA' }))
        'dropbox_available'    = ([string] $dropbox.Available).ToLowerInvariant()
        'dropbox_rev'          = $dropbox.Rev
        'dropbox_size'         = $(if ($null -ne $dropbox.Size) { "$($dropbox.Size)" } else { 'NA' })
        'dropbox_method'       = "$($dropbox.Method)"
        'dropbox_modified'     = 'NA'
        'native_exit'          = $nativeExit
        'diagnostics'          = ($diagnostics -join ',')
    }
}

# =============================================================================
# Entry point: A 層の観測結果を出力する（stdout のみ。redirect を伴わない）
# =============================================================================
$result = Invoke-CanonicalSourceGuard `
    -CanonicalPath      $CanonicalPath `
    -BaselineSha256     $BaselineSha256 `
    -BaselineSize       $BaselineSize `
    -BaselineGenerated  $BaselineGenerated `
    -DropboxPath        $DropboxPath `
    -DropboxRev         $DropboxRev

foreach ($key in $result.Keys) {
    Write-Output ("{0}={1}" -f $key, $result[$key])
}

exit ([int] $result['native_exit'])