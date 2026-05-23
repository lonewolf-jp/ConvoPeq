# ISR Rebuild Admission 運用手順（S-DEF-03）

作成日: 2026-05-23
対象系列: `S-DEF-03`（Finalize defer timeout による `REBUILD_FORCED_DISPATCH`）

---

## 1. 目的

`DeferredFinalizeAware` が長時間解放されず、timeout により強制発行へ移行する事象を、
**5分窓で早期検知し一次切り分けする**ための最短手順。

---

## 2. 監視KPI（5分窓）

- `forced_dispatch_count`
  - 条件: `event=REBUILD_FORCED_DISPATCH` かつ `reason=deferred_finalize_rebuild_requested`
- `forced_dispatch_ratio`
  - 分母: `reason=deferred_finalize_rebuild_requested`
  - 分子: 上記 `forced_dispatch_count`

推奨初期閾値（運用調整前の仮置き）:

- 回数: `forced_dispatch_count > 5`
- 比率: `forced_dispatch_ratio > 10%`

---

## 3. 抽出コマンド（PowerShell）

### 3.0 共通維持の統合抽出（短文化版）

```powershell
$SDefPattern = 'event=REBUILD_DEFERRED.*reason=deferred_structural_due|event=REBUILD_DISPATCHED.*reason=deferred_structural_rebuild_requested|event=REBUILD_DEFERRED.*reason=deferred_finalize_ready|event=REBUILD_DISPATCHED.*reason=deferred_finalize_rebuild_requested|event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested|event=REBUILD_REQUESTED.*reason=deferred_finalize_rebuild_requested'

# まず S-DEF-* を一括抽出（時系列確認）
Select-String -Path <windowLog> -Pattern $SDefPattern | Sort-Object LineNumber
```

### 3.1 回数監視

```powershell
$WarnForced = 5
$forcedPattern = 'event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested'
$forced = (Select-String -Path <windowLog> -Pattern $forcedPattern).Count
if ($forced -gt $WarnForced) {
   Write-Warning "S-DEF-03 threshold exceeded: forced_dispatch_count=$forced (> $WarnForced)"
} else {
   Write-Output "OK: S-DEF-03 forced_dispatch_count=$forced (<= $WarnForced)"
}
```

### 3.2 発生率監視

```powershell
$WarnRatio = 10.0
$forcedPattern = 'event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested'
$forced = (Select-String -Path <windowLog> -Pattern $forcedPattern).Count
$finalizeReq = (Select-String -Path <windowLog> -Pattern 'reason=deferred_finalize_rebuild_requested').Count
if ($finalizeReq -gt 0) {
   $ratio = [math]::Round(($forced / $finalizeReq) * 100, 2)
   if ($ratio -gt $WarnRatio) {
      Write-Warning "S-DEF-03 threshold exceeded: forced_dispatch_ratio=${ratio}% (> ${WarnRatio}%)"
   } else {
      Write-Output "OK: S-DEF-03 forced_dispatch_ratio=${ratio}% (<= ${WarnRatio}%)"
   }
}
```

### 3.3 8.1 定例実行プリセット（8.6）への相互参照

Runbook から直接たどるため、8.1 再検証は以下のプリセットを実行例として固定する。

- 参照先（実行プリセット正本）:
  - `doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md`（`8.6 8.1 定例実行プリセット（運用固定）`）

実行例（窓分割）:

```powershell
# 窓A: finalize defer / MustExecute 観測
powershell -NoProfile -ExecutionPolicy Bypass -File .github/scripts/isr-8_1-cli-run.ps1 -ProbeFinalizeAware -ExitMs 9000 -ProbeDelayMs 1400

# 窓B: forced dispatch 観測
powershell -NoProfile -ExecutionPolicy Bypass -File .github/scripts/isr-8_1-cli-run.ps1 -ProbeFinalizeAware -ExitMs 12000 -ProbeDelayMs 1800 -ProbeIrReloadStorm

# 窓C: UI burst 抑制補強
powershell -NoProfile -ExecutionPolicy Bypass -File .github/scripts/isr-8_1-cli-run.ps1 -ProbeFinalizeAware -ExitMs 8000 -ProbeIntentBurst 120
```

運用ルール:

1. 判定は単一窓 `readyToClose8_1` ではなく、8.1-1〜8.1-4 の文言充足で行う。
2. 記録の正本は `doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md` とする。
3. 本Runbookで閾値超過を検知した場合は、上記3窓を再実行して再現性を確認する。

---

## 4. 一次切り分け（5分で実施）

1. `S-DEF-*` の統合抽出結果を確認（`timerCallback` 単一入口の時系列）
2. `forced_dispatch_count` が閾値超過か確認
3. 同時間帯で以下の件数を並べる
   - `reason=deferred_finalize_ready`
   - `reason=deferred_finalize_rebuild_requested`
   - `event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested`
4. 判断
   - `deferred_finalize_ready` が低く、`forced_dispatch` が高い
     - finalize ready 条件に到達しづらい可能性
   - `deferred_finalize_rebuild_requested` 自体が高い
     - finalize 系要求が過多（上流入口の発火頻度）

---

## 5. コード到達先（調査ジャンプ用）

- timeout 閾値定義
  - `src/audioengine/AudioEngine.Timer.cpp:244`
- defer開始時刻の記録/参照
  - `src/audioengine/AudioEngine.Timer.cpp:249,252`
- timeout 経路での強制 dispatch ログ
  - `src/audioengine/AudioEngine.Timer.cpp:291`
- finalize ready 経路ログ
  - `src/audioengine/AudioEngine.Timer.cpp:305`
- `submitRebuildIntent(...MustExecute)` 発行
  - `src/audioengine/AudioEngine.Timer.cpp:315`
- timeout 時刻保持メンバ
  - `src/audioengine/AudioEngine.h:1140`
- telemetry 文字列表現（event/reason）
  - `src/audioengine/AudioEngine.h:1536,1573`
- shutdown 時刻リセット
  - `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:58`

---

## 6. エスカレーション条件

- 連続3窓（15分）で `forced_dispatch_count` 超過
- 1窓で `forced_dispatch_ratio >= 30%`
- かつ `Release` ビルド済み環境で再現

上記のいずれかを満たす場合、`AudioEngine.Timer.cpp` の finalize ready 判定群と
上流入口（`DeferredFinalizeAware` を立てる箇所）を重点レビューする。

---

## 7. 運用記録欄（統一テンプレート）

| 日時 | 窓 | 値 | 判断 | 対応 |
| --- | --- | --- | --- | --- |
| YYYY-MM-DD HH:mm | 直近5分 | `forced_dispatch_count=<n>`, `forced_dispatch_ratio=<x>%` | 例: 閾値内 / 閾値超過 | 例: 継続監視 / 一次切り分け / エスカレーション |
| 2026-05-23 10:00 | 直近5分 | `forced_dispatch_count=2`, `forced_dispatch_ratio=4.00%` | 閾値内 | 監視継続（ダミー記録） |

記録ルール:

- 日時はローカル時刻で統一（24時間表記）。
- 「値」は Runbook のKPI名をそのまま使う（省略しない）。
- 「判断」は閾値との比較結果を明記する（主観表現のみは禁止）。
- 「対応」は実施アクションを1行で残す（未対応なら `保留` と理由）。

---

## 8. 参照元（正本）

- `doc/work/ISR_Rebuild_Admission_Phase0_実装タスク分解_2026-05-23.md`
  - `3.7 S-* 系列ID対応 ログ抽出テンプレート（運用定型）`
- `doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md`
