# STG-11-D17 Fresh Discovery — Failure-State / Fallback / Recovery Semantic Integrity Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D17_FRESH-DISCOVERY_20261001.md`
- Work item: **STG-11-D17** — failure-state／fallback／recovery 意味的一貫性監査
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20261001-054723).md` と同一。
  D11 マーカー 8 件・FRESH を作業開始時に確認。過去資料・記憶・旧行番号は未使用。
  D12〜D16 の結論の流用なし）
- Commit / push: **禁止**

---

## 0. 判定

```
NO-GO — no concrete defect
```

Loader／IR／resample／FFT／builder／validation／publish／rebuild／cancel／
retry／fallback／placeholder／old-valid／pending／dirty／error／recovery／
UI-visible state の failure-state transition を実コードで監査した結果、
D17 GO 条件（§3 の 7 項目）をすべて満たす concrete defect は存在しない。
Implementation / Gate は作成しない。

D1〜D11 の未 commit 変更・D12〜D17 監査成果物はすべて保持する。

---

## 1. Failure-state transition map（§2）

| 経路 | request→pending | failure→state | old-valid | flag/error | next request |
| --- | --- | --- | --- | --- | --- |
| Loader decode/alloc/read/cancel | isLoading＋clearLastError | error＋flags false＋部分 object retire／RAII | 維持（swap なし） | error 設定、開始時に clear | 可能 |
| callAsync 失敗 | — | raw 解放＋newConv retire＋error dispatch | 維持 | error 設定 | 可能 |
| runSynchronously 失敗 | — | newConv retire | 維持 | — | 可能 |
| applyNewState 例外 | — | flag reset（isLoading/isRebuilding） | 維持 | reset | 可能 |
| stale generation commit | — | swap なし・metadata なし（destructor retire） | 維持 | flags false | 可能 |
| rebuild obsolete（多 phase） | — | 破棄（publish なし） | 維持 | — | 可能 |
| prepare 例外 | — | `rollbackPrepareFailure` | 維持 | — | 可能 |
| prepare publish 失敗 | CallerDestroy（`destroyRolledBackDSP`） | 物理解放＋slot null | 維持 | — | 可能 |

## 2. Typical-defect audit（§4 A–H）

- **A（stale pending）**：全 failure path で flag（isLoading/isRebuilding）が
  false に戻る。`callAsync` 失敗時も atomic のみ整合維持。該当なし。
- **B（stale error）**：`lastError` は次回 load 開始時に clear。成功時は error を
  設定しない。`clearLastError` caller は load-start の 1 箇所のみであり、
  「成功後に failure state が残る」のは次回 load までの正規表示（last-error
  semantics）であるため欠陥ではない。該当なし。
- **C（ownership 不一致）**：builder success→publish 拒否時は CallerDestroy 契約で
  物理解放。destructor が非所有 retire なし。該当なし。
- **D（old-valid retention）**：全 failure で swap 前に return／破棄されるため
  old valid が残る。placeholder は初回のみ・bypass 付き。該当なし。
- **E（fallback 解除）**：bypass 遷移は dry/wet の両系で完結（dry 不可時は
  wet-only 減衰で遷移進行）。解除不能の証拠なし。
- **F（retry semantics）**：`rebuildPendingAfterLoad` は exchange-once＋失敗時復元。
  RetryScheduler は enum-only dispatch（冪等 merge）。重複・上書きの証拠なし。
- **G（cancel semantics）**：cancel は `threadShouldExit` により error 報告から除外
  （`run()` の条件分岐）。cancel の failure 記録なし。
- **H（stale completion 副作用）**：currency guard 外の state mutation は
  idempotent 処理（UI 通知・latency・再要求）のみ。該当なし。

## 3. Authority / ISR audit（§5）

RT 待機／alloc／判断の追加なし。authority 単一性維持。observer 観測のみ。
validation 失敗の publish なし。convo wrapper のみ。fallback の RT authority 化なし。
（変更を行わないため不変）

## 4. なぜ NO-GO か

D17 GO 条件 7 項目に照らし、「既存 guard で完全に中和されていない」
「具体的なユーザー影響がある」を満たす候補が存在しない。
最も近い B（stale error）は last-error semantics の正規表示であり、
G（cancel）は明示的に除外設計であることを実証した。

## 5. 推奨

- D6 NO-GO・R8 flake・R4（P3）・dead code 一括整理は引き続き未解決として残る。

## 6. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**
- D1〜D11 の未 commit 変更・D12〜D17 監査成果物はすべて保持
- Implementation / Gate は作成しない（NO-GO のため）
