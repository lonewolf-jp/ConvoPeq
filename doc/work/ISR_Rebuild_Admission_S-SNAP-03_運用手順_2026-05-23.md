# ISR Rebuild Admission 運用手順（S-SNAP-03）

作成日: 2026-05-23
対象系列: `S-SNAP-03`（NonMT snapshot command の queued / buffer full）

---

## 1. 目的

NonMT 経路の snapshot command が `buffer_full` で取りこぼしていないかを、
**5分窓で早期検知し一次切り分けする**ための最短手順。

---

## 2. 監視KPI（5分窓）

- `buffer_full_non_mt_count`
  - 条件: `reason=snapshot_command_buffer_full_non_mt`
- `buffer_full_non_mt_ratio`
  - 分子: `reason=snapshot_command_buffer_full_non_mt`
  - 分母: `reason=snapshot_command_queued_non_mt` + `reason=snapshot_command_buffer_full_non_mt`

推奨初期閾値（運用調整前の仮置き）:

- 回数: `buffer_full_non_mt_count > 20`
- 比率: `buffer_full_non_mt_ratio > 15%`

---

## 3. 抽出コマンド（PowerShell）

### 3.1 回数監視

```powershell
$N = 20
$count = (Select-String -Path <windowLog> -Pattern 'reason=snapshot_command_buffer_full_non_mt').Count
if ($count -gt $N) {
   Write-Warning "S-SNAP-03 threshold exceeded: buffer_full_non_mt=$count (> $N)"
} else {
   Write-Output "OK: S-SNAP-03 buffer_full_non_mt=$count (<= $N)"
}
```

### 3.2 比率監視

```powershell
$WarnRatio = 15.0
$queued = (Select-String -Path <windowLog> -Pattern 'reason=snapshot_command_queued_non_mt').Count
$full = (Select-String -Path <windowLog> -Pattern 'reason=snapshot_command_buffer_full_non_mt').Count
$total = $queued + $full

if ($total -gt 0) {
   $ratio = [math]::Round(($full / $total) * 100, 2)
   if ($ratio -gt $WarnRatio) {
      Write-Warning "S-SNAP-03 threshold exceeded: buffer_full_non_mt_ratio=${ratio}% (> ${WarnRatio}%)"
   } else {
      Write-Output "OK: S-SNAP-03 buffer_full_non_mt_ratio=${ratio}% (<= ${WarnRatio}%)"
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

1. `buffer_full_non_mt_count` または `buffer_full_non_mt_ratio` が閾値超過か確認
2. 同時間帯で以下の件数を並べる
   - `reason=enqueue_snapshot_command`
   - `reason=snapshot_command_queued_non_mt`
   - `reason=snapshot_command_buffer_full_non_mt`
   - `reason=snapshot_intent_debounced`（MT側抑止）
3. 判断
   - `buffer_full_non_mt` のみ増加
     - NonMT 入口バースト過多、または command buffer 容量不足
   - `queued_non_mt` も同時増加
     - 負荷増加局面だが疎通は継続。比率優先で監視

---

## 5. コード到達先（調査ジャンプ用）

- snapshot command enqueue 入口
  - `src/audioengine/AudioEngine.Init.cpp:85,90`
- MT側 debounce 抑止
  - `src/audioengine/AudioEngine.Init.cpp:137`
- MT側 buffer full / queued
  - `src/audioengine/AudioEngine.Init.cpp:153,166`
- NonMT側 buffer full / queued
  - `src/audioengine/AudioEngine.Init.cpp:182,192`
- telemetry reason 文字列表現
  - `src/audioengine/AudioEngine.h:1578,1579`
- 主要呼び出し元（例）
  - `src/audioengine/AudioEngine.UIEvents.cpp:148`
  - `src/audioengine/AudioEngine.Timer.cpp:163`

---

## 6. エスカレーション条件

- 連続3窓（15分）で `buffer_full_non_mt_count` 超過
- 1窓で `buffer_full_non_mt_ratio >= 30%`
- かつ NonMT 側操作で再現性がある

上記のいずれかを満たす場合、`enqueueSnapshotCommand()` の NonMT 分岐と
command buffer 容量/消費遅延を重点レビューする。

---

## 7. 運用記録欄（統一テンプレート）

| 日時 | 窓 | 値 | 判断 | 対応 |
| --- | --- | --- | --- | --- |
| YYYY-MM-DD HH:mm | 直近5分 | `buffer_full_non_mt_count=<n>`, `buffer_full_non_mt_ratio=<x>%` | 例: 閾値内 / 閾値超過 | 例: 継続監視 / 一次切り分け / エスカレーション |
| 2026-05-23 10:00 | 直近5分 | `buffer_full_non_mt_count=3`, `buffer_full_non_mt_ratio=6.00%` | 閾値内 | 監視継続（ダミー記録） |

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
