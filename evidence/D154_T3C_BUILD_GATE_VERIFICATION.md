# D154 — T3c Build Gate + Post-Implementation Verification

**Date:** 2026-08-31 (+09:00)
**Type:** build gate (G1-G4) + implementation verification (G5/D154). Source changes during gates: 0（ゲート中に追加したものは tools/ の診断補助のみ）。

```text
G1 Debug build    : PASS  T3C_CMAKE_EXIT=0 / error-scan 0 / 239 targets   (evidence/T3C_Debug_BUILD_LOG.txt)
G2 Debug CTest    : PASS  100% tests passed out of 40 / T3C_CTEST_EXIT=0  (evidence/T3C_Debug_CTEST_LOG.txt)
G3 Release build  : PASS  T3C_CMAKE_EXIT=0 / error-scan 0 / 553 targets   (evidence/T3C_Release_BUILD_LOG.txt)
G4 Release CTest  : PASS  100% tests passed out of 40 / T3C_CTEST_EXIT=0  (evidence/T3C_Release_CTEST_LOG.txt)
```

- #21 `ISRSemanticValidationRejects`（= ISRSemanticValidationTests.exe・**NT-1..5 を含む全 main アサート**）両構成 Passed。main は全テスト成功時のみ return 0 → NT-1..5 の invariant は Debug/Release 双方で成立。
- AudioEngineHarness 18.47s Passed（lost-wake 回帰なし）。
- 既存 40/40 と T3c 後の 40/40 は**別時点の実績として分離記録**: 本ゲートが T3c 実装後の初全通過。

## 1. V1-V10 再実測（実装後・本ターン）

| V | target | 実測 | 判定 |
|---|---|---|---|
| V1 | production `markTransientFailure(` 呼び出し 0 | 宣言 h:589 + 定義 cpp:1153 のみ、呼び出し **0** | PASS |
| V2 | slot への `.delivery =` plain write 0 | **0**（`desired.delivery` ローカルのみ） | PASS |
| V3 | 旧 atomic field 0 | **0** | PASS |
| V4 | id/state 直 atomic アクセス 0 | **0**（D153-I 実測・ソース無変化） | PASS |
| V5 | intrinsic 直接 0 / atomic<W> 1 / runtime 検証 1 | **0 / 1（h:386）/ 1（h:421）** | PASS |
| V6 | W + alignas + static_assert | 実装済（h:84-96・5 点） | PASS |
| V7 | adjudicate production 1 | Threading.cpp:272 + TEST-ONLY ラッパ内 + tests | PASS |
| V8 | postSignal production 6 | **6**（RebuildDispatch×4 + Orchestrator×2） | PASS |
| V9 | CTest | **RUN: 40/40 × 2（Debug/Release）** | PASS |
| V10 | 旧 counter/delivery 構造 0 | **0** | PASS |

意図しないソース変更: git status = T3c 対象 6 ファイル + ConvoPeq.md（再生成）+ tools/ 診断補助（新規・untracked）のみ。既存 G-4.x/F6 変更の revert なし。

## 2. runtime lock-free 検証 — **発見事項 D154-F1（要仕様訂正・安全性影響なし）**

指示「単なる compile 成功ではなく runtime verification が実際に通ったことを記録」に応え、レイアウト同一の診断プローブ（tools/t3c_lockfree_probe.cpp・production/test 外）を同一ツールチェーンでビルド・実行し、加えて逆アセンブルとソース交差アセンブリで実装経路を確定した。

**実測（MSVC 14.51 / VS2026 18.9.2 x64 /O2）**:
```text
sizeof=16 alignof=16 atomic_alignof=16 is_lock_free=0
cas_success=1 cas_fail_updates_comparand=1 (observed id=7)
```
- disasm（evidence/T3C_PROBE_DISASM.txt）: `cmpxchg16b` **0 箇所**、`F0 48 0F C7` バイト **0**。
- /FAcs（build/t3c_probe.asm）: `std::_Atomic_storage<RecoveryLifecycleWord,16>::compare_exchange_strong` の本体に **`_Guard _Lock{_Spinlock}`** — **16B `std::atomic<T>` の CAS はロックプール（smtx スピンロック）経由**。
- 原因の特定: STL の「lock-free using 16-byte intrinsics」16B 特殊化は **`_Atomic_storage<_Ty&, 16>`（atomic_ref 用・`_Ty&` 形）** であり、`std::atomic<T>`（`_Ty` 形）は汎用ロックプール経路。D152-R1 §1 が引用したコメントは atomic_ref 側の話だった。
- 帰結: **`is_lock_free()==false` は「保身的」ではなくこのツールチェーンでの正確な報告**。ISRDSPHandle.cpp:16-22 の「実際は CMPXCHG16B で lock-free に動作」という既存コメントも**同一の誤りを含む（既存問題・T3c 範囲外）**。

**安全性評価（T3c に影響なし）**:
1. 正しさ: ロックプールでも 16B RMW は真の原子的相互排他 + フルバリア。CAS 意味論（成功/失敗時 comparand 更新）はプローブで実証、NT-5 の実 2 スレッド競合も Debug/Release 通過。
2. RT 適合: W の触达者 = CoordinatorLoop + RebuildThread（実 ISR 非接触は D150 §2.3/D153 で実測済）。**D150 §2 は「lock-pool でも許容（触达者が ISR でないこと）」を事前承認済み** — 承認範囲内。
3. 性能: smtx 非競合スピンは ~ns 級、32 slot/tick の advisory load 増分は 1ms tick に対して無視可能。ネスト保持なし（adjudicate の CAS は resolve 呼び出し前に完了 — 同一スレッドで直列）。
4. ゲート停止条件「16B atomic runtime verification failure」は**非発火**（ctor の MSVC 分岐は設計どおり記録のみで失敗せず、全テスト通過）。

**要 follow-up（D152-R2 — 文言のみ・設計不変）**:
- D152-R1 §1/§2 の「MSVC x64 で 16B atomic は runtime lock-free（CMPXCHG16B）」記述を撤回し、「`std::atomic<W>` 16B はロックプール（is_lock_free()==false が正確）。NonRT 専用接触のため D150 §2 承認により許容」と訂正。
- table ctor コメント（h:423「actual path is CMPXCHG16B」）を同一事実へ修正（次の編集ウィンドウで可・機能変更なし）。
- 真の lock-free が必要なら D152 §2 の明示 `_InterlockedCompareExchange128` wrapper 案へ切替可（設計は既に仕様化済み・要ユーザー判断）。
- 付随: ISRDSPHandle.cpp:16-22 / ISRDSPHandle.h:22-25 の既存コメントも同一誤り — 別トラックで修正推奨。

## 3. 重点トレース（D154 指示項目）

1. **T5 contention**: cpp:938-947 — CAS 失敗→`w`=現在値→`state!=Live` のときのみ tryInsert 経路、Live のうちは再試行。**「CAS 失敗→tryInsert→ΔL+1」誤経路なし**（G43 T1/T7/T10・C16・NT 系が両構成通過）。
2. **T4 publication**: h:469-494 — payload 6 書込→全文 CAS（Live 公開）→liveCount++（勝者のみ）。呼び出し側に Live 後 payload 書込なし。
3. **terminal ownership**: h:515-524 — `liveCount_−1` は CAS 成功ブロック内のみ。adjudicate 枯渇も resolveRecoveryObligation の true 返却（=勝者）経由（cpp:1132）。
4. **exhaustion telemetry**: cpp:1132-1133 — `recoveryRetryExhaustedCount_++` は resolve 成功者のみ。T-R18-5/T-R20-3/T-R21-1/T-P2-5/NT-3/NT-4 の正確値アサートが両構成通過 = 過計上なし。
5. lifecycle 全 writer 6 箇所 = 全て `compare_exchange_strong`（D153-I §2-9 トレース、ソース無変化）。

## 4. 判定

```text
D154 = PASS（G1-G4 全通過・V1-V10 全 PASS・重点トレース 5/5）
  + 発見事項 D154-F1（仕様文言の事実誤認・安全性影響なし・D152-R2 で訂正予定）

→ T3c 実装ゲートは事実上解除。ConvoPeq.md は 21:03:41 に再生成済みで、
   次の監査 baseline = T3c 実装反映後ソース。
```
