# D154 — T3c Build Gate（Work Report）

```text
G1 Debug build PASS → G2 Debug CTest 40/40 PASS → G3 Release build PASS → G4 Release CTest 40/40 PASS
詳細: evidence/D154_T3C_BUILD_GATE_VERIFICATION.md
```

**Status: 4 ゲート全 PASS + D154 検証 PASS。T3c 実装ゲート事実上解除。ただし発見事項 D154-F1（仕様文言の事実誤認・安全性影響なし）を D152-R2 で訂正要。**

## ゲート結果
- Debug build: T3C_CMAKE_EXIT=0（239 ターゲット、エラー 0）→ Debug CTest: **40/40**（T3C_CTEST_EXIT=0、#21=NT-1..5 含む全 main アサート通過、AudioEngineHarness 18.47s Passed）。
- Release build: T3C_CMAKE_EXIT=0（553 ターゲット）→ Release CTest: **40/40**（T3C_CTEST_EXIT=0）。
- 既存 40/40（G-4.3-T-R 時点）と本ゲートの 40/40 は別時点実績として分離記録。**T3c 実装後の初全通過**。
- V1-V10 再実測: 全 PASS（V9 は RUN に更新: 40/40×2）。意図しないソース変更なし（変更= T3c 対象 6 ファイル + tools 診断補助のみ、既存 G-4.x 変更の revert なし）。

## 発見事項 D154-F1（重要・要 follow-up）
「runtime lock-free が実際に通ったこと」の記録要求に応え、レイアウト同一プローブ + disasm + /FAcs で実装経路を確定:
- **`std::atomic<16B>` の CAS はロックプール（`_Guard _Lock{_Spinlock}`）経由。`is_lock_free()==false` は保身ではなく正確な報告。exe 内に cmpxchg16b ゼロ。**
- 原因: STL の「lock-free 16-byte intrinsics」特殊化は **atomic_ref 用（`_Ty&` 形）**。`std::atomic<T>` ではない。D152-R1 §1 の引用対象が誤りだった。
- **安全性影響なし**: ロックプールでも 16B RMW は真の原子的相互排他+フルバリア（CAS 意味論はプローブ実証、NT-5 実 2 スレッド競合も両構成通過）。W 触达者は CL+RebuildThread のみ（実 ISR 非接触）で、**D150 §2 の「lock-pool 許容」事前承認の範囲内**。ゲート停止条件「runtime verification failure」は非発火（ctor の MSVC 分岐は設計どおり記録のみ、全テスト通過）。
- **follow-up（D152-R2・文言のみ）**: ①D152-R1 §1/§2 の lock-free 記述撤回→「ロックプール・is_lock_free==false が正確・NonRT 専用ゆえ許容」②table ctor コメント（h:423）修正③真の lock-free が必要なら D152 §2 の明示 `_InterlockedCompareExchange128` wrapper 案へ切替可（仕様済み・要判断）④付随: ISRDSPHandle.cpp:16-22 の既存コメントも同一誤り（別トラック）。

## 重点トレース（全 PASS）
T5 contention 再試行（失敗→現在値再評価→Live なら retry、terminal のみ tryInsert — 誤 NEW 化なし）/ T4 payload→CAS→liveCount++ / terminal −1 と exhausted++ は CAS 勝者のみ（正確値アサート群が両構成通過）/ lifecycle 全 writer=compare_exchange_strong のみ。

## baseline 更新
ConvoPeq.md を **21:03:41** に再生成（RecoveryLifecycleWord×26 / postRecoveryFailureSignal×20 / adjudicateRecoveryFailureSignals×10 マーカー確認）。次監査の baseline は T3c 反映後ソース。

**次アクション候補**: D152-R2（文言訂正 + ctor コメント修正の小型パッチ、read-only 監査不要と判断するなら即適用可）→ 任意で AV stress 200 等の運用検証。
