# check-authority-boundary.ps1
# ★ P0-5: Publication Authority Audit (compile-time主防御の補助監査)
#
# 以下の直接利用を警告 (CI Fail にはしない):
#   - makeRuntimePublicationCoordinator(
#   - RuntimePublicationCoordinator::acquireWriteAccess(
#   - publishWorld(
#   - ::acquireWriteAccess(
#
# friend増殖監査 (CI Fail):
#   - friend RuntimePublicationOrchestrator|PublicationExecutor|RuntimePublicationStateOwner 以外の friend 追加
#
# 戻り値: 0=警告なし, 1=違反あり (ただしCI Failにはしないgrep警告)

param(
    [string]$RepoRoot = (Resolve-Path "$PSScriptRoot/../..")
)

$exitCode = 0

Write-Host "=== P0-5: Authority Boundary Audit ===" -ForegroundColor Cyan

# 監査対象ファイル
$engineHeader = Join-Path $RepoRoot "src/audioengine/AudioEngine.h"
$coordinatorHeader = Join-Path $RepoRoot "src/core/RuntimePublicationCoordinator.h"
$orchestratorHeader = Join-Path $RepoRoot "src/audioengine/RuntimePublicationOrchestrator.h"

# ── 1. Coordinator生成の直接利用を警告 ──
Write-Host "`n[1/3] makeRuntimePublicationCoordinator() 直接利用:" -ForegroundColor Yellow
$coordCalls = Select-String -Path "$RepoRoot\src\**\*.h", "$RepoRoot\src\**\*.cpp" `
    -Pattern 'makeRuntimePublicationCoordinator\(' `
    -SimpleMatch `
    -ErrorAction SilentlyContinue | `
    Where-Object { $_.Path -notlike "*\JUCE\*" -and $_.Path -notlike "*\r8brain-free-src\*" }

if ($coordCalls) {
    Write-Host "  WARNING: 以下のファイルで直接利用されています (将来 Orchestrator 経由に変更推奨):" -ForegroundColor Yellow
    $coordCalls | ForEach-Object {
        Write-Host "    $($_.Path):$($_.LineNumber)"
    }
}
else {
    Write-Host "  OK: 直接利用は検出されませんでした" -ForegroundColor Green
}

# ── 2. Coordinator::acquireWriteAccess の直接利用を警告 ──
Write-Host "`n[2/3] RuntimePublicationCoordinator 直接API利用:" -ForegroundColor Yellow
$directApiCalls = Select-String -Path "$RepoRoot\src\**\*.h", "$RepoRoot\src\**\*.cpp" `
    -Pattern '(RuntimePublicationCoordinator::acquireWriteAccess|::acquireWriteAccess\()' `
    -ErrorAction SilentlyContinue | `
    Where-Object { $_.Path -notlike "*\JUCE\*" -and $_.Path -notlike "*\r8brain-free-src\*" }

if ($directApiCalls) {
    Write-Host "  WARNING: 以下のファイルで直接API利用が検出されました:" -ForegroundColor Yellow
    $directApiCalls | ForEach-Object {
        Write-Host "    $($_.Path):$($_.LineNumber): $($_.Line.Trim())"
    }
}
else {
    Write-Host "  OK: 直接API利用は検出されませんでした" -ForegroundColor Green
}

# ── 3. friend増殖監査 ──
Write-Host "`n[3/3] friend 宣言の監査 (許可リスト照合):" -ForegroundColor Yellow
$allowedFriends = @(
    'RuntimePublicationOrchestrator',
    'PublicationExecutor',
    'RuntimePublicationStateOwner',
    'RuntimeBuilder',
    # ★ BuilderToken: RuntimeState の BuilderToken 経由構築を許可 (AudioEngine, RuntimeBuilder)
    'AudioEngine',
    # ★ DSPTransition: クロスフェード遷移中に RuntimeWorld 内部へのアクセスが必要
    'DSPTransition',
    # ★ NoiseShaperLearner / EQEditProcessor: AudioEngine 内部状態へのアクセスが必要
    'NoiseShaperLearner',
    'EQEditProcessor',
    # ★ RuntimePublicationCoordinator<World,Handle,Bridge>: CRTP テンプレート friend パターン
    'RuntimePublicationCoordinator'
)

$engineFriendLines = Select-String -Path $engineHeader -Pattern 'friend class' -ErrorAction SilentlyContinue
$coordFriendLines = Select-String -Path $coordinatorHeader -Pattern 'friend class' -ErrorAction SilentlyContinue

$allFriendLines = @()
if ($engineFriendLines) { $allFriendLines += $engineFriendLines }
if ($coordFriendLines) { $allFriendLines += $coordFriendLines }

# ★ Practical Stable: コメント行（// または /*）のヒットは誤検出として除外
$allFriendLines = $allFriendLines | Where-Object {
    $line = $_.Line.Trim()
    $line -notmatch '^\s*(//|/\*|\*)'
}

$violations = @()
foreach ($line in $allFriendLines) {
    $found = $false
    foreach ($allowed in $allowedFriends) {
        if ($line.Line -match $allowed) {
            $found = $true
            break
        }
    }
    if (-not $found) {
        $violations += $line
    }
}

if ($violations.Count -gt 0) {
    Write-Host "  FAIL: 許可されていない friend 宣言が検出されました!" -ForegroundColor Red
    $violations | ForEach-Object {
        Write-Host "    $($_.Path):$($_.LineNumber): $($_.Line.Trim())"
    }
    $exitCode = 1
}
else {
    Write-Host "  OK: friend 宣言は許可リスト内に収まっています" -ForegroundColor Green
}

Write-Host "`n=== Audit Complete ===" -ForegroundColor Cyan
exit $exitCode
