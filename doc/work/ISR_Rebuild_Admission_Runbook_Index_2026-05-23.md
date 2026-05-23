# ISR Rebuild Admission Runbook Index

作成日: 2026-05-23
目的: 系列別Runbookを単一入口で参照し、監視オペレーションを統一する。

---

## 1. Runbook一覧

| 系列ID | 監視対象 | Runbook |
| --- | --- | --- |
| `S-REQ-02` | 重複抑止（`pending_duplicate`） | `doc/work/ISR_Rebuild_Admission_S-REQ-02_運用手順_2026-05-23.md` |
| `S-SNAP-03` | NonMT snapshot command 成否（queued/full） | `doc/work/ISR_Rebuild_Admission_S-SNAP-03_運用手順_2026-05-23.md` |
| `S-DEF-03` | finalize defer timeout 強制発行（`REBUILD_FORCED_DISPATCH`） | `doc/work/ISR_Rebuild_Admission_S-DEF-03_運用手順_2026-05-23.md` |

---

## 2. 日次チェック順（推奨）

1. `S-DEF-03` を確認（timeout 強制発行の常態化を最優先で検知）
2. `S-REQ-02` を確認（重複抑止率の異常増加を検知）
3. `S-SNAP-03` を確認（NonMT バッファあふれを検知）

運用統一ルール: `S-DEF-*` は `timerCallback` 単一入口の共通維持系列として、**統合抽出（S-DEF-01/02/03 一括）を先に実施してから系列別カウントする**。

---

## 3. 共通エスカレーション基準

- 連続3窓（15分）で各Runbookの警告閾値を超過
- 1窓で重大閾値（各Runbookのエスカレーション条件）を超過
- 同時に `Release` 構成で再現

上記のいずれかを満たす場合、系列別Runbookの「コード到達先」に従って該当分岐をレビューする。

---

## 4. 運用記録の更新手順（統一）

1. 5分窓で対象系列のKPIを抽出する。
2. 該当Runbookの `## 7. 運用記録欄（統一テンプレート）` に1行追記する。
   - 列順は必ず `日時 / 窓 / 値 / 判断 / 対応` を維持する。
3. 閾値超過時は「判断」に超過内容を明記し、「対応」に一次切り分けまたはエスカレーションを記録する。
4. エスカレーション時は、同一行にレビュー対象ファイル（例: `AudioEngine.Timer.cpp`）を追記する。

記録フォーマット（共通）:

`YYYY-MM-DD HH:mm | 直近5分 | KPI値 | 閾値内/超過 | 実施対応`

---

## 5. 運用記録欄への導線

- `S-REQ-02` 記録欄:
  - `doc/work/ISR_Rebuild_Admission_S-REQ-02_運用手順_2026-05-23.md`（`## 7`）
- `S-SNAP-03` 記録欄:
  - `doc/work/ISR_Rebuild_Admission_S-SNAP-03_運用手順_2026-05-23.md`（`## 7`）
- `S-DEF-03` 記録欄:
  - `doc/work/ISR_Rebuild_Admission_S-DEF-03_運用手順_2026-05-23.md`（`## 7`）

---

## 6. 正本参照

- 抽出/閾値テンプレート正本:
  - `doc/work/ISR_Rebuild_Admission_Phase0_実装タスク分解_2026-05-23.md`（3.7）
- 受入/フェーズ基準:
  - `doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md`
- 受入クローズ判定の正本:
  - `doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md`
- R11〜R25監査の正本:
  - `doc/work/R11-R25_Closed判定監査表_2026-05-21.md`

---

## 7. 日次オペ最短手順（入口ファイル完結版）

### 7.1 毎日まずやること（5分窓）

1. `S-REQ-02` で `pending_duplicate_ratio` を確認する。
2. `S-SNAP-03` で `buffer_full_non_mt_ratio` を確認する。
3. `S-DEF-03` で `forced_dispatch_ratio` を確認する。

### 7.2 閾値超過時（再現確認）

1. 該当Runbookの `3.3 8.1 定例実行プリセット（8.6）への相互参照` を開く。
2. 窓A/B/Cを同日で連続実行し、再現性を確認する。
3. 抽出結果を `ConvoPeq.log` から保存し、Runbook `## 7` 記録欄へ追記する。

### 7.3 記録と同期の優先順

1. 判定記録は `doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md` を正本とする。
2. 判定差異があれば `クローズログ → 最終計画書 → 監査表注記` の順で同期する。

### 7.4 すぐ開く先（ショートカット）

- `S-REQ-02`: `doc/work/ISR_Rebuild_Admission_S-REQ-02_運用手順_2026-05-23.md`
- `S-SNAP-03`: `doc/work/ISR_Rebuild_Admission_S-SNAP-03_運用手順_2026-05-23.md`
- `S-DEF-03`: `doc/work/ISR_Rebuild_Admission_S-DEF-03_運用手順_2026-05-23.md`
- 実行プリセット正本（8.6）: `doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md`
