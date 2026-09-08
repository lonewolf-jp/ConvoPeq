# D152-R2〜D154-F2 統合報告書 — T3c Atomic Backend 事実訂正・運用ストレス検証・RT Affinity 監査

**Date:** 2026-08-31 (+09:00)
**Type:** 統合報告書（本セッション成果物 9 文書 + ストレス検証記録の統合）
**Production source:** 0 変更 / Test source（リポジトリ内）: 0 変更 / CMake: 0 / Build（production）: 0 — worktree は D154 検証時と同一（`git diff --stat`: 7 files, +2798/−460）
**統合対象:**

| 種別 | ファイル |
|---|---|
| 規範（本セッションで新設） | `evidence/D152R2_T3C_ATOMIC_BACKEND_FACT_RESTATEMENT.md` |
| 規範（訂正適用） | `evidence/D152R1_T3C_ATOMIC_BACKEND_CORRECTION.md` / `doc/work88/D152R1_D153R_BACKEND_FIX_REPORT.md` |
| 監査 | `evidence/D154R2_ATOMIC_BACKEND_REAUDIT.md` / `evidence/D152R2_RT_AFFINITY_AUDIT.md` / `evidence/D154-F2_DSPHANDLE_COMMENT_AUDIT.md` |
| ストレス検証 | `evidence/D152R2_ST1_AV_STRESS_200.md` + `evidence/st1/`（driver・build log・実行 log ×4） |
| 作業報告 | `doc/work88/D152R2_ATOMIC_BACKEND_FACT_RESTATEMENT_REPORT.md` / `D154R2_REAUDIT_REPORT.md` / `D152R2_ST1_AV_STRESS_REPORT.md` / `D152R2_RT_AFFINITY_AUDIT_REPORT.md` / `D154-F2_DSPHANDLE_COMMENT_AUDIT_REPORT.md` |

---

## 0. 総括判定（先出し）

| 工程 | 内容 | 判定 |
|---|---|---|
| [1] D152-R2 | atomic backend の事実訂正（仕様/evidence 文言のみ） | **確定** |
| [2] D154-R2 | 軽量 read-only 再監査（R2-01〜R2-08） | **PASS（全項）** |
| [3] ST-1 | AV stress 200（lock-pool backend 下の contention / lifecycle pressure） | **PASS**（Debug/Release 200/200・カウンタ整合違反 0） |
| [4] RT affinity audit | lifecycle W の ISR/AudioThread 非接触再確認 | **PASS**（RT パス参照 0 件実測） |
| [5] D154-F2 | DSPHandle 既存コメント問題の別トラック監査 | **監査完了** — comment-only patch 仕様確定・**await user go** |

**T3c = close candidate**（ST-1 PASS + RT affinity PASS により、D154-F1 の「lock-free ではないが NonRT-only なので許容」が現行 topology の実測として維持）。

---

## 1. 確定した技術的事実（本セッションの核）

本ツールチェーンの **MSVC STL 実物ソース**（実コンパイル使用版 14.51.36231、および 14.52.36615 の両方）で検証した事実:

| # | 事実 | 実測根拠 |
|---|---|---|
| 1 | `std::atomic<16B by-value>`（`RecoveryLifecycleWord` も `DSPHandle` も同型）は **lock-pool（`_Guard _Lock{_Spinlock}` per-object spinlock）で実装される**。by-value 16B 専用特殊化は両 STL バージョンに**存在しない**（検索 0 件） | 14.51.36231 `include/atomic:530` / 14.52.36615 `:527-660`（ジェネリック locking 版・store/load/exchange/CAS 全て spinlock 排他下の memcmp/memcpy） |
| 2 | `is_lock_free()==false` は**保身ではなく正確な報告**。`atomic::is_lock_free()` は `_Is_always_lock_free<sizeof(_Ty)>`（compile-time 定数）を返し、実装も実際に lock-pool であるため値は一致する | `:2133-2138`（14.52.36615） |
| 3 | 「lock-free using 16-byte intrinsics」特殊化（`_Atomic_storage<_Ty&, 16>`）は **atomic_ref 用の参照形（`_Ty&`）のみ**。参照形だけが `__iso_volatile_load16/store16` + `_InterlockedCompareExchange128`（真の lock-free）を使用する | 14.51.36231 `:1120` / 14.52.36615 `:1102` + `atomic_ref : _Choose_atomic_base_t<_Ty, _Ty&>`（`:2276`） |
| 4 | D154-F1 プローブ実測と 3 系統で整合: `is_lock_free()==false`・exe 内 cmpxchg16b ゼロ・CAS 意味論（原子的相互排他 + フルバリア）は lock-pool 下でも保持 | evidence/D154_T3C_BUILD_GATE_VERIFICATION.md |

**歴史的経緯**: D152 §2.3「MSVC の 16B atomic は lock pool」→ D152-R1 がこれを撤回して「lock-free」と確定（誤り）→ D154-F1 実証 → **D152-R2 で D152 §2.3 が正しかったことを復元**し、混同の本体（atomic vs atomic_ref）を特定。誤りが 2 回対抗方向に発生したため、以後の監査は「runtime `is_lock_free()` 値を実装経路と突き合わせる」ことを判定規律とする（新教訓）。

**訂正された教訓（制度化）**: 「MSVC では `std::atomic<T>`（by-value）と `std::atomic_ref<T>`（参照形）は 16B で異なる実装経路を持つ。by-value = lock-pool（false が正確）、参照形 = intrinsics lock-free（true）。両者を混同して実装経路を記述しない。」

---

## 2. [1] D152-R2 — 仕様/evidence 文言訂正（確定）

### 2.1 変更範囲

- **不変**: backend 選択（`std::atomic<RecoveryLifecycleWord>` 維持）・intrinsic wrapper 不採用・T1〜T7 意味論・W フィールド構成・liveCount/delivery authority・NT-1〜NT-5・V1〜V4/V6〜V10。
- **訂正**: D152-R1 §1（lock-free 事実記述 → lock-pool 訂正 + 全体撤回）、§2（16B 実装経路行）、§3（load 整合性の根拠 → spinlock 排他下コピー）、§4（ctor 検証仕様の事実記述 + assert 対象範囲明確化）、§5（V5）、教訓条項。

### 2.2 主要確定内容（ユーザー指示 5 項目の反映）

1. **§1**: 「MSVC x64 の `std::atomic<16B>` は runtime lock-free」を**撤回**。atomic_ref の `_Atomic_storage<_Ty&, 16>` と `std::atomic<_Ty>` を混同しない。
2. **§2**: backend = `std::atomic<RecoveryLifecycleWord>` のまま維持。intrinsic wrapper 切替は**今回行わない** — 理由: T3c の W 接触者は CoordinatorLoop / RebuildThread で ISR ではない。
3. **§4**: ctor 検証 — `is_lock_free()==false` を異常扱いしない・MSVC は前例同様 runtime 値を記録するだけ・非 MSVC `assert(ok)` は非 MSVC x64（alignas(16) で真の lock-free）が対象範囲。
4. **§5（V5 最終形）**:

```text
intrinsic direct use                              = 0
std::atomic<RecoveryLifecycleWord> storage        = 1
runtime is_lock_free() probe                      = 1
expected MSVC result                              = false
```

runtime verification の「成功」を `is_lock_free()==true` と定義**しない**。

5. **安全性要件の明示**（D150 §12 既存承認の引用・新判断なし）:

```text
T3c の安全性要件は lock-free 性ではなく、
16B lifecycle state の原子的 CAS semantics と
非RT affinity を満たすことである。
```

根拠: D150 §12（evidence/D150_T3C_LIFECYCLE_WORD_REPROOF.md:215）「実 call graph で W 触达者 = {CL, RebuildThread}、実 audio callback 非接触。よって lock-pool でも RT 制約は違反しない」。

### 2.3 適用した文書変更

| ファイル | 変更 |
|---|---|
| `evidence/D152R2_T3C_ATOMIC_BACKEND_FACT_RESTATEMENT.md` | 新設（規範本文 §0〜§7） |
| `evidence/D152R1_T3C_ATOMIC_BACKEND_CORRECTION.md` | 冒頭訂正バナー + §1 全体撤回（取り消し線で履歴保持）+ §2/§3/§4/§5 訂正 |
| `doc/work88/D152R1_D153R_BACKEND_FIX_REPORT.md` | 訂正バナー + 「撤回」項の撤回戻し + V5 行更新 |

---

## 3. [2] D154-R2 — 軽量 read-only 再監査（PASS）

| ID | 確認内容 | 結果 | 実測 |
|---|---|---|---|
| R2-01 | 「`std::atomic<16B>` = lock-free」誤記述が D152-R1 から消滅 | **PASS** | 残存言及は全て【D152-R2 訂正】文脈・取り消し線内。規範誤記述 0 |
| R2-02 | atomic（by-value）/ atomic_ref（参照形）の混同なし | **PASS** | §1 事実表（両 STL バージョン実物検証） |
| R2-03 | backend = `std::atomic<RecoveryLifecycleWord>` のまま | **PASS** | `_InterlockedCompareExchange128` in src/audioengine = 0・atomic<W> 宣言 1（h:386） |
| R2-04 | `compare_exchange_strong` 全文 CAS semantics 不変 | **PASS** | lifecycle CAS 4 サイト（cpp:942/1096/1140/1423）同一形式 acq_rel |
| R2-05 | T1〜T7 意味論変更なし | **PASS** | D154 以降ソース変更 0（diff stat 同一・ソース mtime 全て D154 evidence より旧） |
| R2-06 | V5 = intrinsic 0 / atomic<W> 1 / runtime probe 1 / expected false | **PASS** | h:386（宣言）・h:421（probe）・D152-R2 §4 |
| R2-07 | `is_lock_free()==false` を T3c failure と誤定義していない | **PASS** | 規範・実コード（MSVC 分岐 `(void)ok;` 記録のみ）不変。ソースコメント残存は D154-F2 登録済み residual（behavior 0） |
| R2-08 | D154-F1 結論と矛盾しない | **PASS** | D154-F1 実測を規範へ昇格・安全性根拠は D150 §12 に限定 |

**既知 residual（非ブロッキング・D154-F2 に登録済み）**: S-1〜S-4（ISRRuntimePublicationCoordinator.h コメント 4 箇所）・S-5〜S-6（ISRDSPHandle.cpp/.h コメント）・S-7（static_assert メッセージ文言）。

---

## 4. [3] ST-1 — AV stress 200（PASS）

### 4.1 実施方法

- **Part A**: D154 検証済み `ISRSemanticValidationTests.exe`（T1〜T10 / NT-1〜NT-5 / RLOE C1〜C16 / R21 T1〜T3 — カウンタ整合は内部 assert で per-iteration 検証）を Debug/Release 各 **200 反復**。
- **Part B**: リポジトリ外ストレスドライバ `evidence/st1/D152R2_ST1_AVStress200.cpp`（**D154 production obj を再リンク — production は byte-identical**、CMake 未登録・`evidence/st1/st1_build.bat` で vcvarsall x64 + MSVC 14.51.36231）。1 サイクル = admission + T5 coalesce storm ×16（同キー再提出・coalesced 厳密検証）+ **T1/T2 実 2 スレッド競合 8ms**（postSignal ∥ adjudicate）+ T6 delivery CAS churn ×64 + T3 terminal teardown。**200 サイクル × 2 構成 × 2 回**。

### 4.2 計測値（200 サイクル集計）

| 構成 | run | elapsed | coalesced | exhausted | droppedTerminal | droppedStale | droppedInvalid | saturated | liveCount 最終 |
|---|---|---|---|---|---|---|---|---|---|
| Debug | 1 | 1845 ms | 15,937 | **200** | 25,401,916 | 488,875 | **0** | 358,133 | **0** |
| Debug | 2 | 1843 ms | 15,901 | **200** | 18,354,203 | 787,171 | **0** | 358,121 | **0** |
| Release | 1 | 1826 ms | 15,950 | **200** | 102,449,288 | 626,834 | **0** | 2,540,614 | **0** |
| Release | 2 | 1816 ms | 15,964 | **200** | 107,384,926 | 453,571 | **0** | 1,482,532 | **0** |

### 4.3 チェック項目別判定

| チェック項目 | 結果 |
|---|---|
| `liveCount_ < 0` / `> capacity(32)` | **0 件**（全フェーズ境界チェック・最終 0） |
| double terminal transition | **0 件**（2 回目 resolve 常に false） |
| droppedTerminal / droppedStale / saturated | **期待意味論どおり**（terminal 化後の storm 投入・id 消滅後の観測・pending>=K の inert 投入のみ計上・単調増加・過剰計上ゲート違反 0） |
| droppedInvalid | **0（厳密一致）** |
| `recoveryRetryExhaustedCount` | **over-count 0**（== Failed 終端数と厳密一致: 200/200 サイクルで 1 サイクル 1 件） |
| runtime `is_lock_free()` | **false** を両構成で記録（D152-R2 §4 expected = false の実行時証跡） |

**読み取り**: lock-pool（spinlock）backend 下で数千万回規模の lifecycle CAS 競合（実 2 スレッド storm × 200 サイクル × 2 構成）を負荷しても全不変条件が維持。coalesced ≈ 79.7/サイクル = 16 storm + churn 分で契約整合。

### 4.4 driver 設計訂正（production は契約どおり・契約実証として記録）

| # | 初版の誤った期待 | 実際の契約挙動（実測で確定） |
|---|---|---|
| F-1 | terminal 直後に delivery==None | **delivery は ObligationState と独立**（h:356）。resolve / adjudicate は delivery を保持し、None-ing は非 terminal drain CAS のみ（cpp:1140-1146） |
| F-2 | resolve が必ず勝ち旧 terminal 語が残る | 競合下で adjudicate が先に K=4 到達 → ResolvedFailed 化し得る。その後の同キー resubmit は**合法的に新規 obligation 再承認**（terminal → resubmit NEW・G-4.3-T T7 semantic）し、terminal slot を再利用して上書きし得る。driver は子 id を transport pop から捕捉し teardown で双方 terminal 化・liveCount 基線復帰を検証 |

---

## 5. [4] RT affinity audit — PASS

### 5.1 W アクセス点の全列挙（20 サイト・coordinator 内部に限定的）

| # | 位置 | 関数 | 操作 |
|---|---|---|---|
| A1 | cpp:902-972 | `submitRecoveryRequest`（coalesce/tryInsert 呼び出し元） | load + CAS |
| A2 | cpp:1081-1105 | `postRecoveryFailureSignal`（T1） | load + CAS（pending++） |
| A3 | cpp:1121-1150 | `adjudicateRecoveryFailureSignals`（T2） | load + CAS（drain/terminalize） |
| A4 | cpp:1031-1062 + h:512/520 | `resolveRecoveryObligation` → `table.resolve`（T3 / Completion Authority） | load + CAS（Live→terminal） |
| A5 | cpp:1415-1430 | `casDelivery`（T6/T7） | load + CAS（delivery） |
| A6 | h:437/465/482 | `table.findByKey` / `tryInsert` | load + CAS（Live 公開） |
| A7 | cpp:1164/1196/1215/1225 | `peekLifecycleForTest` / resolve 内部 | load（test-only / A4 内部） |
| A8 | cpp:1463 | `redriveDeferredRecoveryObligations` スキャン | load |

### 5.2 入口別 thread affinity（enclose 関数 + スレッド起動点で実測）

| 入口 | スレッド | 根拠 |
|---|---|---|
| rebuildThreadLoop 内 postSignal ×4（RebuildDispatch:1006/1033/1091/1115）+ enqueuePublicationIntentForRuntimeCommit（Route A） | **RebuildThread** | 専用スレッド本体（:824） |
| runCoordinatorPhase（adjudicate / redrive / processIntent → QuarantineIntentHandler → submitRecoveryRequest） | **CoordinatorLoop** | `juce::Thread("ConvoPeq.CoordinatorLoop")`・ソース明記「Non-RT only, never RT」（ISRCoordinatorLoop.cpp:8） |
| Orchestrator Route C | CoordinatorLoop / RebuildThread | RebuildDispatch:912-919 / Threading:240/299 |
| shutdown close（discardRecoveryRequestsOnShutdown） | join 後の単一スレッド | Threading:249 → RebuildDispatch:810 |

### 5.3 Audio/RT パスの非接触（実測 0 件）

- RT 処理系 8 ファイル（AudioBlock / DSPCoreFloat / DSPCoreDouble / DSPCoreIO / BlockDouble / DSPCoreLifecycle / AudioEngineProcessor / Reader）: recovery API 参照 **0 件**（HIT_COUNT=0）
- `EQProcessor.h:466` / ConvolverProcessor の `retireCoordinator_` ポインタ: prepareToPlay での配線のみ・**deref 0 件** — RT callback から coordinator メソッドを一切呼ばない
- Timer 経路にも recovery API 呼び出し 0 件（旧 timerCallback cadence は CoordinatorLoop に移設済み）

**帰結**: D154-F1 の安全性評価前提（W access = CL + RebuildThread、ISR 非接触）が現行 topology の実測として維持。**変質リスク注記**: 将来 RT callback 内から `retireCoordinator_` を deref する変更が入った場合、D152 §2 の `_InterlockedCompareExchange128` wrapper 案（真の lock-free）への切替が必須。

---

## 6. [5] D154-F2 — DSPHandle 既存コメント問題（別トラック・監査完了・await go）

**原則**: `コメント訂正のみ ≠ std::atomic<DSPHandle> の実装変更`。T3c トラックと混在しない。

### 6.1 誤コメント在庫（comment-only patch 対象・7 グループ）

| # | 位置 | 誤記述 | 訂正方針 |
|---|---|---|---|
| S-5a | `ISRDSPHandle.cpp:16-20` | 「STL の保身的判定。実際は CMPXCHG16B で lock-free」 | 「MSVC では 16B by-value atomic は lock-pool（spinlock）実装・false は正確な報告。CAS 意味論は保持・全消費者 NonRT ゆえ許容」 |
| S-6a | `ISRDSPHandle.h:22` | 「atomic<DSPHandle> が CMPXCHG16B を使用」 | alignas(16) の目的を「将来の lock-free 切替備え + 整列コピー効率」に訂正 |
| S-6b | `ISRDSPHandle.h:204-205` | 「must be lock-free … (CMPXCHG16B, Haswell+/AVX2)」 | 「原子的 CAS semantics が要件・MSVC は lock-pool・NonRT affinity」に訂正 |
| S-6c | `ISRDSPHandle.h:212` | static_assert メッセージ「uses CMPXCHG16B on x64」 | メッセージ文言のみ（assert 自体維持） |
| S-6d | `ISRDSPHandle.h:214-218` | （runtime 検証記述は概ね正しい） | 「MSVC では false が期待値（正確な報告）」を追記 |
| S-1〜S-4 | `ISRRuntimePublicationCoordinator.h:79-83 / 98 / 413-417 / 423` | T3c 側同一誤り | 同一 patch で訂正 |
| S-7 | `ISRRuntimePublicationCoordinator.h:93 / 96-97` | static_assert メッセージ中の「CMPXCHG16B」 | 文言清掃（assert 維持） |

### 6.2 `std::atomic<DSPHandle>` の安全性確認（本監査で実測）

消費者: `releaseResources`（MessageThread）・`timerCallback` / `onHealthEvent` / `retirePublishedDSP`（Timer = MessageThread）・Orchestrator（RebuildThread/CL）— **RT callback からのアクセス 0 件**。lock-pool backend は T3c（D150 §12）と同型の NonRT 根拠で安全。既存コメントの「全操作は NonRT」の主張自体は正しく、誤りは lock-free 実装経路の記述のみ。

### 6.3 LatencySnapshot 前例の再評価（訂正不要）

`ConvolverProcessor.StateAndUI.cpp:960-966`: `is_lock_free()` false を DBG 記録し「implementation-provided atomic semantics で継続」と明記 — **当初から正しい**。以後これを「CMPXCHG16B lock-free の根拠」として引用しない（D152-R1 で既に撤回済みの引用形）。

### 6.4 patch 実施時の運用（await user go）

1. 7 グループの comment-only 編集（コード・静的 assert の論理は不変）
2. `python output_sourcecode_markdown.py` で ConvoPeq.md 再生成
3. git diff が comment 行のみであることの確認

---

## 7. 現在の状態と次アクション

```text
D154 = PASS ─ D154-F1（follow-up required）
                 │
                 ▼ 完了
        [1] D152-R2 確定 ─ [2] D154-R2 PASS ─ [3] ST-1 PASS ─ [4] RT affinity PASS
                                                                    │
                                                          T3c close candidate
                                                                    │
        [5] D154-F2 comment-only patch（7 グループ）── await user go
                 （適用後: ConvoPeq.md 再生成 → diff comment-only 確認）
```

- **未確定事項（本セッションで全て解消）**: D152-R1 の lock-free 誤記述（解消）、ST-1 未実施（実施・PASS）、RT affinity 未検証（実測・PASS）、D154-F2 スコープ（監査完了）。
- **保留事項**: D154-F2 patch 適用のみ。T3c close 判定はユーザー判断。
- **禁止継続項目**: T3c 再実装 / intrinsic wrapper 切替 / CMake 変更 / 無目的な 40/40 再実行 / DSPHandle 実装変更 — 変更不要と確定。

## 8. 参照（統合元ファイル一覧）

- 規範: `evidence/D152R2_T3C_ATOMIC_BACKEND_FACT_RESTATEMENT.md`（§0-§7）
- 監査: `evidence/D154R2_ATOMIC_BACKEND_REAUDIT.md` / `evidence/D152R2_RT_AFFINITY_AUDIT.md` / `evidence/D154-F2_DSPHANDLE_COMMENT_AUDIT.md`
- ST-1: `evidence/D152R2_ST1_AV_STRESS_200.md` / `evidence/st1/D152R2_ST1_AVStress200.cpp` / `evidence/st1/st1_build.{bat,log}` / `evidence/st1/D152R2_ST1_partA_semantic200.log` / `evidence/st1/D152R2_ST1_{debug,release}_run2.log`
- 訂正適用先: `evidence/D152R1_T3C_ATOMIC_BACKEND_CORRECTION.md` / `doc/work88/D152R1_D153R_BACKEND_FIX_REPORT.md`
- 個別報告: `doc/work88/D152R2_ATOMIC_BACKEND_FACT_RESTATEMENT_REPORT.md` / `D154R2_REAUDIT_REPORT.md` / `D152R2_ST1_AV_STRESS_REPORT.md` / `D152R2_RT_AFFINITY_AUDIT_REPORT.md` / `D154-F2_DSPHANDLE_COMMENT_AUDIT_REPORT.md`
- 引用規範: `evidence/D150_T3C_LIFECYCLE_WORD_REPROOF.md`（§2.3・§12 lock-pool 許容）/ `evidence/D154_T3C_BUILD_GATE_VERIFICATION.md`（D154-F1）
