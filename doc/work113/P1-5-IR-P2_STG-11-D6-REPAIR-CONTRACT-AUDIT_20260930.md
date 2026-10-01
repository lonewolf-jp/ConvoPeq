# STG-11-D6 Fresh Discovery — Repair Contract Audit (第2版)

- Document: `doc/work113/P1-5-IR-P2_STG-11-D6-REPAIR-CONTRACT-AUDIT_20260930.md`
  （初版の D6-1 選定 NO-GO 報告は本ファイルに統合・置換する。
  初版ファイルは保持するが、本第2版が最新の governing record である）
- Work item: **STG-11-D6 Fresh Discovery** — 既存候補リストを起点にしない新規 concrete defect 発見
- Date: 2026-09-30
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20260930-132503).md` と同一。authority 再確認の繰り返しは Owner 指示により省略し、FRESH であることを作業開始時に確認済み）
- Commit / push: **禁止**

---

## 0. 判定

```
NO-GO（concrete defect の証明なし。実装に進まない）
```

Fresh invariant / ownership-graph / failure-path / shutdown-path / RT call-chain /
state-container の 6 監査（Owner §10）を実コードから実施した結果、
「実害がコードから証明でき、現在の作業ツリーを壊さずに修正できる」新規 defect は存在しなかった。
初版 NO-GO とは異なり、本第2版は既存 inventory の結論を再利用せず、
すべての候補を実コードから再検証している（§2〜§7）。

D1〜D5 の未 commit 状態・D5 の READY FOR COMMIT はそのまま保持する。
Implementation / Gate は作成しない。

---

## 1. Fresh ownership-graph audit

retire ownership の生成→終端までの全経路を実コードで追跡した。

### 1.1 `Shutdown` / `QueueFull` リターンの到達性（§4.A/C）

`ISRRetireRouter::enqueueWithRetry`（`ISRRetireRouter.cpp:303-384`）の全 return 経路を実読：

- Stage 1/2 の `enqueueRetire`（4-arg）は `Success` / `QueuePressure` のみを返す
  （`:239-272`。`false` → `QueuePressure` に写像し、`Shutdown` は返さない）。
- Stage 2 の `break` 条件（`!= QueuePressure`）は到達不能な dead defense。
- Stage 3 は `QueuePressure` / `QueueFull` でのみ到達し、Q → E → T の順で
  ownership transfer のみで return する。T（growable）は常に受領。
- したがって direct-router 経路の `enqueueWithRetry` は `Shutdown` / `QueueFull` を返さない。

`Shutdown` を返す producer は実コード上存在しない
（`DeferredDeletionQueue` に Shutdown 概念なし。
`shutdownReclaim` → `terminalReclaim` は常に true）。
帰結：

- `ISRRetireRouter::retire()`（void）の非 Success 沈黙経路は到達不能。欠陥ではない。
- `enqueueDeferredDeleteNonRt` の `false`（`AudioEngine.h:4508`）は到達不能な dead defense。
  `ConvolverProcessor.Lifecycle.cpp:59,72` と `AudioEngine.h:3838,3854` の bool 破棄は
  contract 上の wart だが、**実害なし**（failure が発生しない）。
  `EQCacheManager::storeNewMap`（`Cache.cpp:40`）の fallback 処理は正しいが発火しない。
- 将来 `Shutdown` を返す producer が追加された場合は本評価が覆る。
  その際は bool 破棄 4 箇所が一斉に leak となる（記録のみ。推測による先行修正はしない）。

### 1.2 Q/E 格納時の seqId=generation=0

`enqueueWithRetry` Stage 3 と `quarantineRetireSink` は seqId/generation に 0 を渡すが、
drain の safety 判定は `epoch` のみ使用（`RetireQuarantineStore.h:103-107`）。
両 field は metadata であり provenance に影響しない。D1 の epoch provenance 問題と異なり無害。

### 1.3 `retireRT` / `releaseRT` / `enqueueRetireEpochBounded`

- `releaseRT` の caller 0 件（`RefCountedDeferred` は DEPRECATED）。
- `enqueueRetireEpochBounded` の caller 0 件（decl＋def のみ）。
- いずれも dead code。到達不能のため欠陥ではない。

### 1.4 所有権保存の結論

全 failure path（enqueue failure / fallback / retry / quarantine promotion /
shutdown / terminal / emergency drain）で 0-release / 2-release になる到達可能経路なし。
`m_retireSink == nullptr` 経路は ctor で必ず設定されるため到達不能。

---

## 2. Fresh failure-path audit（§4.A）

`bool` / `[[nodiscard]]` / `Result` の repository-wide 確認（RT 到達域＋retire 域）：

| 経路 | 結果 |
| --- | --- |
| `enqueueDeferredDeleteNonRt` の bool 破棄 4 箇所 | §1.1 のとおり failure 到達不能。実害なし |
| `enqueueDeferredDeleteWithFallback` の `false`（Shutdown のみ） | caller が ownership 保持する契約どおり。`Shutdown` は coordinator 経由でのみ到達可能だが、当該 caller（EQ setter）は `false` を処理する。実害なし |
| `retireDSPHandleForRuntime` の bool（`RebuildDispatch.cpp:983`） | `false` 時に `destroyDSPCoreNode` 直接破棄（if-else）。過去の二重解放は修正済み（コメントに記録）。現行正しい |
| `observe` 系の drop | 文書化 policy（条件付き drop）どおり＋counter。実害なし |
| `quarantine` 系の drop | fallback ring＋drop counter（Critical 駆動）。実害なし |

「operation fails → caller continues → state is changed anyway」型の経路なし。

---

## 3. Fresh state/container consistency audit（§4.B）

- `switchImmediate` / `startFade` / `completeFade` / `retireCurrentAndTarget`：
  いずれも `enqueueWithRetry`＋`quarantineRetireSink` fallback。同一 obligation の
  複数 container 同時存在なし。
- Q/E/T の stage 移送は ownership-transfer 成立でのみ return（D1/D2 契約のまま）。
- `resetFadeStateAndRetireTarget`（raw enqueue＋bool 破棄）は production caller 0 件の
  dead code（D2 で確定済み。今回も再確認）。dead code の修正は Owner 方針により行わない。
- DSPHandle 台帳の誤用パターン（`retireDSPHandleForRuntime` 単独使用）は
  production caller が `DSPLifetimeManager` のみに限定されていることを再確認
  （`RebuildDispatch.cpp:983` は正規の if-else 処理）。
- P3 型の double representation は Publish / Retire / Snapshot / Runtime /
  Crossfade のいずれにも検出なし。

---

## 4. Fresh shutdown-path audit（§4.D）

- reconfigure/terminal 境界（D167-2）は維持。reconfigure は close/epch 進行なし。
- terminal pipeline：graceful（5s・epoch 前進＋reclaim 駆動）→ final drain →
  EmergencyDrain（要求時のみ）→ Q/E/T 強制 drain（audio 停止後）。
  各段の ownership 終端を確認。具体的な欠落なし。
- shutdown flag publish 前後・callback 最終 iteration・epoch reader active・
  deferred/quarantine 残存のいずれにも race 欠陥の証拠なし。
  （audio 停止は JUCE 契約＋harness の join により drain に先行する）
- `mmcssShutdownRequested` の device reopen 競合：新 thread は未登録のため
  revert が no-op になる。無害。

---

## 5. Fresh RT call-chain audit（§5 — 間接 chain を含む）

- Audio callback（`AudioBlock` / `BlockDouble`）の直接 backend token：0 件。
- lazy init：`constexpr` のみ＋診断 static（guard 内）。`ensureThreadFloatingPointEnvironment`
 （MXCSR）・MMCSS（D5）・affinity（D4）以外の初回 OS 呼び出しなし。
- `logger` / `String` / `refcount` / `mutex` / `CV` / `file I/O` / `JUCE backend` /
  `destruction` の audio 到達：D3〜D5 で整理済みの 3 TU 以外に検出なし。
  `Commit` 経路の logging は caller 全件 NonRT。
- 「OS API だから危険」とは判定していない（Owner §5）。
  RT 到達＋具体的 blocking/allocation/ownership/backend 依存＋call path の
  3 点が揃った候補なし。

---

## 6. Intermittent の fault-injection 検討（§6）

R8 / OBS-5 / OBS-D4-3 の signature（STG-6 area の access violation、
STG-8-D2b の timing assert）について race window / invariant / log signature を再評価したが、
fault injection で deterministic に再現できる controlled 条件を特定できなかった。
procdump 12 回試行でも crash dump 未取得（D3 Gate 記録）。
「書けない」ことを elimination 根拠にはしないが、**injection 設計の起点となる
具体的 window も特定できない**ため、現時点で repair 対象にできない。
観測継続（再現時の dump 取得が条件）。

---

## 7. D5-R1 / D5-R2 の再評価（初版 §4 を維持・補強）

- **R1**：別問題としての記録を維持。実害の証明なしのため D5 再実装なし。
- **R2**：writer overlap の具体的 path なし（join 先行）を再確認。D5 defect としない。

---

## 8. Contract 判定

| 確認項目 | 結果 |
| --- | --- |
| authority | PASS（D1〜D5 反映確認済み） |
| §4.A error-return 無視 | 到達可能な欠陥なし（§1.1、§2） |
| §4.B state/container 二重表現 | 検出なし（§3） |
| §4.C ownership conservation | 全経路で保存（§1） |
| §4.D shutdown race | 証拠なし（§4） |
| §5 RT 間接 chain | 残存なし（§5） |
| §6 intermittent | injection 起点の特定なし（§6） |
| 新 authority / queue / worker の必要性 | なし |
| invariant impact | なし（変更を行わない） |

```
NO-GO（concrete defect の証明なし）
```

---

## 9. 推奨（D6-1 ではない。次工程の投入候補）

| 候補 | 理由 | 規模 |
| --- | --- | --- |
| 将来 `Shutdown` を返す producer が追加された場合の bool 破棄 4 箇所 | §1.1。latent contract wart。現時点で実害なし | 変更なし（記録のみ） |
| R8 intermittent の観測継続 | §6。再現時 dump が条件 | 観測継続 |
| R4 Candidate C（P3） | 大型再設計級 | 別 STG |
| dead code 一括整理 | 到達不能を確認済みの 5 件超。単独 STG ではなく一括で | 別機会 |
| N4 × STG-8 非交差確認 | 体系確認。大型調査 | 別機会 |

---

## 10. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**
- D1〜D5 の未 commit 変更・D5 の READY FOR COMMIT・初版 D6 discovery 成果物はすべて保持
- Implementation / Gate は作成しない（NO-GO のため）
- D6-2 へは進まない。Owner の指示を待つ
