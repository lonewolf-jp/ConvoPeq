# D154-R2 — T3c Atomic Backend 軽量再監査（read-only・R2-01〜R2-08）

**Date:** 2026-08-31 (+09:00)
**Type:** read-only 再監査。**Production source: 0 変更 / Test source: 0 変更 / CMake: 0 / build: 0 / CTest: 0。**
**対象:** D152-R2（evidence/D152R2_T3C_ATOMIC_BACKEND_FACT_RESTATEMENT.md）適用後の整合確認。
**Baseline:** git worktree = D154 検証時と同一（`git diff --stat`: 7 files, +2798/−460）。ソース mtime（ISRRuntimePublicationCoordinator.cpp 20:06:44 / .h 19:56:05 / ISRSemanticValidationTests.cpp 20:16:39）は全て D154 evidence (21:04:48) より旧 — **D154 以降ソース変更 0**。

---

## 判定（先出し）: **PASS（R2-01〜R2-08 全項）** — 既知 residual 2 件は D154-F2 別トラック登録済み・非ブロッキング

| ID | 確認内容 | 結果 | 実測根拠 |
|---|---|---|---|
| R2-01 | `std::atomic<16B>` = lock-free の誤記述が D152-R1 から消滅 | **PASS** | D152-R1 evidence + work88 report の残存 lock-free/CMPXCHG 言及は全て【D152-R2 訂正】ブロック・取り消し線・訂正後文の文脈内（grep -v 訂正/撤回/~~ で規範誤記述 0）。§4 見出し「runtime lock-free 検証」は節題の継承名称のみで、本文は訂正済み（非ブロッキング） |
| R2-02 | atomic（by-value）/ atomic_ref（参照形）の MSVC STL 特殊化混同なし | **PASS** | D152-R2 §1: by-value 16B = ジェネリック locking 版（`_Atomic_storage<_Ty, 16>` 特殊化は存在せず・`_Guard _Lock{_Spinlock}`）/ 参照形 16B = `_Atomic_storage<_Ty&, 16>`（atomic_ref 用・intrinsics lock-free）— 本ツールチェーン MSVC STL 14.52.36615 実物で検証済み |
| R2-03 | backend は `std::atomic<RecoveryLifecycleWord>` のまま | **PASS** | `_InterlockedCompareExchange128` in src/audioengine = **0**。`std::atomic<RecoveryLifecycleWord> lifecycle` メンバ宣言 = **1**（ISRRuntimePublicationCoordinator.h:386）。intrinsic wrapper 切替なし |
| R2-04 | `compare_exchange_strong` による全文 CAS semantics 不変 | **PASS** | lifecycle 全文 CAS 4 サイト（cpp:942 coalesce / 1096 terminal resolve / 1140 adjudication / 1423 delivery）— 全て `compare_exchange_strong(w, desired, std::memory_order_acq_rel)`、同一形式維持。pendingRecoveryAdmission_.state の CAS 4 サイトは durable transport slot の別ドメイン（不変） |
| R2-05 | T1〜T7 の意味論変更なし | **PASS** | D154 以降のソース変更 0（R2-05 baseline 実測：diff stat 同一・mtime 全て D154 evidence より旧）。意味論は D154 Debug/Release CTest 40/40 時点から不変 |
| R2-06 | V5 = intrinsic 0 / atomic<W> 1 / runtime probe 1 / expected MSVC false | **PASS** | intrinsic 0 / atomic<W> 宣言 1（h:386）/ runtime probe 1（h:421 `test.is_lock_free()` — h:414 はコメント言及）/ expected MSVC result = false は D152-R2 §4 に明記。「成功 ≠ is_lock_free()==true」定義済み |
| R2-07 | `is_lock_free()==false` を T3c failure と誤定義していない | **PASS**（residual 登録済み） | 規範（D152-R2 §3-1/§4）: false は正確な報告・記録のみ・成功定義に非拘束。実コード MSVC 分岐 `(void)ok;`（記録のみ）不変。residual: ソースコメント h:413-417/h:423 に旧文言「conservative / CMPXCHG16B-based」残存 = S-3/S-4 として D152-R2 §6 で D154-F2 に登録済み（behavior 影響 0） |
| R2-08 | D154-F1 の結論と矛盾しない | **PASS** | D152-R2 は D154-F1 の 3 実測（lock-pool CAS・`is_lock_free()==false` 正確・exe 内 cmpxchg16b ゼロ）を規範へ昇格したもの。安全性根拠は D150 §12 既存承認（W 触达者 = {CL, RebuildThread}・audio callback 非接触 → lock-pool 許容）に限定・新たな安全性判断なし |

---

## 既知 residual（D154-F2 別トラック — 本監査では非ブロッキング）

| # | 位置 | 内容 | 処置 |
|---|---|---|---|
| S-1〜S-4 | ISRRuntimePublicationCoordinator.h:79-83 / 98 / 413-417 / 423 | T3c 側誤コメント（lock-free 16B 特殊化・「保身的 false」・CMPXCHG16B-based 記述） | D154-F2 comment-only patch（実装変更不要・await go） |
| S-5〜S-6 | ISRDSPHandle.cpp:16-20 / ISRDSPHandle.h:204-218 | DSPHandle 側同一誤り（`std::atomic<DSPHandle>` も by-value 16B = lock-pool であり CMPXCHG16B lock-free 記述は不正確） | D154-F2 本体（ユーザー指示の別トラック） |
| S-7 | ISRRuntimePublicationCoordinator.h:93 / 96-97 | static_assert メッセージ文言中の「CMPXCHG16B」 | 文言清掃候補（コード自体は維持） |

## 監査手順の記録

- grep（MSYS2）: D152-R1 evidence/work88 report の残余 lock-free 言及の文脈分類（訂正文脈除外 grep → 規範誤記述 0）
- grep: src/audioengine の `_InterlockedCompareExchange128`（0）/ `std::atomic<RecoveryLifecycleWord> lifecycle`（1）/ `compare_exchange_strong` サイト列挙（lifecycle 4 + durable transport 4）
- git: `git diff --stat` が D154 baseline と一致・ソース mtime < D154 evidence mtime
- 文書照合: D152-R2 全節と D154-F1 結論の対読

**判定: D154-R2 = PASS。T3c は仕様・実装・backend とも D152-R2 訂正後の規範に整合。次工程 = ST-1（AV stress 200）。**
