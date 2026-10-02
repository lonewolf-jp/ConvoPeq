# STG-11-D14 Fresh Discovery — Ownership / Authority / RT / Failure / Recovery Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D14_FRESH-DISCOVERY_20261002.md`
- Work item: **STG-11-D14** — D13 と重複しない領域の concrete defect 探索
- Date: 2026-10-02
- Authority: リポジトリルート `ConvoPeq.md`（監査 AI が参照する `ConvoPeq(20261002-115916).md` と同一ファイル。
  ファイル名の違いを理由に別世代として扱わない。Owner 確認済みの同一最新統合ソース）
  - baseline commit: `d49d05bb96cab9b5b3429b843f2c7892b02d2edc`（D13 docs）
  - `Generated: 2026-10-02 20:56:11` / 5,928,120 B
  - SHA-256: `41E27B57205FB50E6F9CAA76EFE6E9F199A3E6571481FF87ED94ADDDFC82E52F`
  - `--check`: `NEWER_SRC_COUNT = 0` / **FRESH**（調査開始時に確認）
  - 過去資料・記憶・旧行番号は根拠として使用していない。結論はすべて現行ソースから導出。
- 本 work item は **read-only Fresh Discovery**。**production / test / build ファイルの変更は一切行っていない。**
- 本書は **未 commit**（Owner の判断待ち）

---

## 0. 判定

```
STG-11-D14 — NO-GO（concrete defect なし）
```

D13 と同じ領域を繰り返さず、Ownership / Lifetime、Publication / Authority、
RT boundary、Failure / Overflow、Atomic / synchronization、Recovery / obligation を
現行ソースで直接確認した。Owner の 7 条件（具体的入力 / state transition /
guard 欠落 / failure mechanism / concrete harm / invariant 違反 / 再現 test）を
すべて満たす新規 defect は存在しない。

---

## 1. 監査範囲と方法

| # | 面 | 確認した内容 | 結果 |
| --- | --- | --- | --- |
| 1 | Ownership / Lifetime（retire 経路） | retire 3 経路の所有権処理 | §2.1 NO-GO |
| 2 | Ownership / Lifetime（shutdown 順序） | destructor の停止順序と deferred disposition | §2.2 NO-GO |
| 3 | Publication / Authority（validator bypass） | requires 節と Bridge 実装の列挙 | §2.3 NO-GO |
| 4 | Publication / Authority（receipt timeout） | timeout 時の所有権扱い | §2.4 NO-GO |
| 5 | RT boundary（allocation / lock） | DSPCore process ファイル走査 | §2.5 NO-GO |
| 6 | Failure / Overflow（queue / registry） | full 時の動作 | §2.6 NO-GO |
| 7 | Atomic / synchronization | raw API と wrapper 規約 | §2.7 NO-GO |
| 8 | Recovery / obligation（deferred identity） | obligation identity と二重解決の有無 | §2.8 NO-GO |
| 9 | State / Numeric（D13 外の残差分） | learning mode enum、convolver state tree 経路 | §2.9 NO-GO |

---

## 2. 追跡結果

### 2.1 retire 3 経路の所有権処理

`AudioEngine.h:3832-3861` の Bridge retire 群は次のとおり分離されている。

- `retirePublishedRuntimeWorldNonRt`（`:3832`）: PublishedDomain のみ。`DeletionEntryType::World`。
- `retireRejectedRuntimeWorldNonRt`（`:3848`）: 非 Published のみ。`DeletionEntryType::Generic`。
- いずれも null-safe、deferred delete（unseal → dtor → aligned_free）経由。
  直接破壊ではなく EBR 破壊権へ委譲する。

Coordinator の reject 経路（`RuntimePublicationCoordinator.h:118-126`）は
`retireRejectedRuntimeWorldNonRt` のみを呼び、published 側には触れない。
二重 retire / use-after-retire の構造は無い。

### 2.2 shutdown 順序と deferred disposition

`AudioEngine::~AudioEngine`（`CtorDtor.cpp:103-`）は次の順序を固定している。

1. `StopAcceptingWork` + `Releasing` + `cancelPendingUpdate` + bridge `requestShutdown`
2. `StopAudio`（`stopTimer`）
3. `StopWorkers`（RetryScheduler → CoordinatorLoop join → rebuild thread stop）

deferred slot の shutdown 時 disposition は
`RuntimePublicationOrchestrator.cpp:613-` `clearDeferredForShutdown` が担い、
slot の DSP を EBR 経由で disposition する（D162-2-G1 で再有効化済み）。
member teardown 時の allocator mismatch（D162-2-F）も解決済みである。

残存 ownership の放置や順序違反は確認できない。

### 2.3 validator bypass（requires 節）

`RuntimePublicationCoordinator.h:118` の `if constexpr (requires ...)` は
Bridge 型に method が無い場合に validation を省略する構造だが、
production の Bridge 型は `RuntimePublicationBridge`（`AudioEngine.h:3801`、method あり）のみである。
live の bypass は無い（D13 §2.5 と同一結論を現行 tree で再確認）。

### 2.4 receipt timeout 時の所有権扱い

`AudioEngine.h:4968-5002` `commitRuntimePublication` は receipt を最大 250ms 待つが、
timeout しても所有権は enqueue 時点で Transferred のままであり、
呼び出し元は world / DSP を破棄しない（`:4985-4989`）。
timeout を failure と誤解して rollback すると二重所有になるため、
現行の扱い（timeout ≠ failure）が正しいことがコード上に明記されている
（work88 X2 §6.2）。shutdown 中の timeout も既存設計どおりである。

### 2.5 RT boundary

`DSPCoreIO.cpp` / `DSPCoreFloat.cpp` / `DSPCoreDouble.cpp` に
`new` / `malloc` / `std::mutex` / `std::lock_guard` / `std::scoped_lock` は無い
（D13 §2.8 と同一結論を現行 tree で再確認）。

### 2.6 queue overflow / silent loss

`PendingPublishRegistry::registerPublish`（`RuntimeWorldAuthority.h:43-49`）は
cursor の modulo 64 で上書きする構造だが、lookup は seqId 照合であり、
古い entry の上書きは「registry の fallback が効かない」場合に
`executePublish` の `hasOwner` 側で吸収される設計である
（`RuntimePublishExecutor.h:42-45` の `owner ? owner : registry.lookup : payload`）。
SnapshotCoordinator の queueFull 時 quarantine と retire routing は D1 / D2 の範囲であり、
本 R で新規の証拠は無い。D19 の failure boundary 走査も concrete harm なしで確定済みである。

### 2.7 atomic 規約

setter / restore / learning ファイルに raw `.store()` / `.load()` /
`.exchange()` は無い。すべて `convo::publishAtomic` / `convo::consumeAtomic` /
`convo::fetchAddAtomic` 経由である（D13 §2.7 と同一結論を現行 tree で再確認）。

### 2.8 deferred retry obligation の identity と二重解決

`RuntimePublicationOrchestrator.h:324-360` は obligation identity を
`(generation, recoveryObligationId)` と定義し、generation 単独ではないことを明記している
（recovery が同一 generation を再利用して別 payload を発行するため）。
terminal（Accepted / StaleDiscard / Expired / ShutdownDiscard）でのみ
`invalidateDeferredObligation` を呼び、re-drive（retention）では呼ばない。
async 解決（`onPublishCommitted`）と sync 解決の single-winner は CAS によることが
コード上に記録されている（D105-R5-9）。read-only の静的追跡で二重解決の穴は確認できない。

### 2.9 State / Numeric の残差分（D13 外）

- `pendingLearningMode`（`NoiseShaperLearningMode`: 0..5）は session / XML の
  復元対象ではない（`StateIO.cpp` / `DeviceSettings.cpp` に参照なし）。
  `setNoiseShaperLearningMode`（`Learning.cpp:305-311`）は型付き enum のみを受け、
  consumer の `getAdaptiveCoeffBankIndex` も `getAdaptiveCoeffBankForIndex` の clamp を経由する。
  外部入力からの到達経路が無い。
- `setConvolverStateTree`（`Parameters.cpp:793-803`）は
  `uiConvolverProcessor.setState` 経由であり、D8 / D9 の setter guard が適用される。
  filter mode 系は AudioEngine 側の管轄であり convolver child に含まれない。
  D7 の inline guard との不整合は無い。

---

## 3. D7〜D13 の再発確認（該当なし）

本 R の追跡で既存 repair の再発は無い。

---

## 4. カバレッジの留保（誠実に記載）

本 R は read-only の静的追跡であり、次の動的側面は検証していない。

- timing-sensitive な race（既存の soak / T1-T4 measurement が担当する領域）
- 実デバイスでの shutdown / drain の挙動
- D135 / D162-2 系の証明（コード上の gate と test が存在し、本 R はその再証明を行わない）

これらは「未検証の残存」ではなく「既存の検証資産が担当する領域」であり、
本 R がその資産を無効化する証拠は何も見つけていない。

---

## 5. 本 D14 の作業記録（read-only 遵守）

```text
変更ファイル: 0 件（production / test / build / CMakeLists.txt / ConvoPeq.md すべて無変更）
追加ファイル: 本書 1 件（未 commit。Owner の判断待ち）

実施した操作:
  - 読み取り（git show / rg / Read）: 多数
  - ビルド / テスト: なし（ソース未変更のため不要）
  - reset / clean / rebase / amend / squash / force push: いずれも不使用

main checkout (C:\VSC_Project\ConvoPeq): 未改変（HEAD = bda43034）
作業 worktree: HEAD = d49d05bb（= origin/main）、clean、未追跡 1 件（本書のみ）
```
