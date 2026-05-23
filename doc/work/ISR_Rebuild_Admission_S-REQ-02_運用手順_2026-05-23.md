# ISR Rebuild Admission 運用手順（S-REQ-02）

作成日: 2026-05-23
対象系列: `S-REQ-02`（`PendingDuplicate` による重複抑止）

---

## 1. 目的

`requestRebuild(double,int)` の重複抑止が過剰化していないかを、
**5分窓で早期検知し一次切り分けする**ための最短手順。

---

## 2. 監視KPI（5分窓）

- `pending_duplicate_ratio`
  - 分子: `reason=pending_duplicate`
  - 分母: `reason=requestRebuild_sr_bs`

推奨初期閾値（運用調整前の仮置き）:

- 比率: `pending_duplicate_ratio > 40%`

---

## 3. 抽出コマンド（PowerShell）

### 3.1 PendingDuplicate比率監視

```powershell
$WarnRatio = 40.0
$pending = (Select-String -Path <windowLog> -Pattern 'reason=pending_duplicate').Count
$requested = (Select-String -Path <windowLog> -Pattern 'reason=requestRebuild_sr_bs').Count

if ($requested -gt 0) {
  $ratio = [math]::Round(($pending / $requested) * 100, 2)
   if ($ratio -gt $WarnRatio) {
    Write-Warning "S-REQ-02 threshold exceeded: pending_duplicate_ratio=${ratio}% (> ${WarnRatio}%)"
   } else {
    Write-Output "OK: S-REQ-02 pending_duplicate_ratio=${ratio}% (<= ${WarnRatio}%)"
   }
}
```

### 3.2 PendingDuplicate件数監視（任意）

```powershell
$WarnPendingCount = 200
$pending = (Select-String -Path <windowLog> -Pattern 'reason=pending_duplicate').Count
if ($pending -gt $WarnPendingCount) {
  Write-Warning "S-REQ-02 threshold exceeded: pending_duplicate_count=$pending (> $WarnPendingCount)"
} else {
  Write-Output "OK: S-REQ-02 pending_duplicate_count=$pending (<= $WarnPendingCount)"
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

1. `pending_duplicate_ratio` が閾値超過か確認
2. 同時間帯で以下の件数を並べる
   - `reason=requestRebuild_sr_bs`
   - `reason=task_queued`
   - `reason=pending_duplicate`
3. 判断
   - `pending_duplicate` 高止まり
     - 既存 pendingTask が詰まり気味（queue 消費遅延・処理時間増加）
   - `pending_duplicate` 低位かつ `task_queued` 優勢
     - 重複抑止は過剰化していない可能性が高い

---

## 5. コード到達先（調査ジャンプ用）

- queue投入成功ログ
  - `src/audioengine/AudioEngine.RebuildDispatch.cpp`（`requestRebuild(sr,bs): task queued`）
- `pending_duplicate` マージログ
  - `src/audioengine/AudioEngine.RebuildDispatch.cpp`（`BLOCKED duplicate pending task`）
- pending duplicate カウンタ更新
  - `src/audioengine/AudioEngine.RebuildDispatch.cpp`（`debugRebuildDispatchBlockedPendingDuplicateCount`）
- telemetry reason 文字列表現
  - `src/audioengine/AudioEngine.h`（`pending_duplicate`）
- diagnostics 集計値公開
  - `src/audioengine/AudioEngine.h`（rebuild diagnostics）

---

## 6. エスカレーション条件

- 連続3窓（15分）で `pending_duplicate_ratio` 超過
- 1窓で `pending_duplicate_ratio >= 60%`
- かつ `task_queued` が有意に減少（通常帯比で半減目安）

上記のいずれかを満たす場合、`requestRebuild(double,int)` の重複判定条件と
上流発火（同一要求連打）を重点レビューする。

---

## 7. 運用記録欄（統一テンプレート）

| 日時 | 窓 | 値 | 判断 | 対応 |
| --- | --- | --- | --- | --- |
| YYYY-MM-DD HH:mm | 直近5分 | `pending_duplicate_ratio=<x>%`, `pending_duplicate_count=<n>` | 例: 閾値内 / 閾値超過 | 例: 継続監視 / 一次切り分け / エスカレーション |
| 2026-05-23 10:00 | 直近5分 | `pending_duplicate_ratio=18.00%`, `pending_duplicate_count=42` | 閾値内 | 監視継続（ダミー記録） |

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
