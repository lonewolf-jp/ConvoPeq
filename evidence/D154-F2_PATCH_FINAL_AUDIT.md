# D154-F2 — DSPHandle / T3c residual comment correction（final audit report）

**Date:** 2026-08-31 (+09:00)
**Type:** comment-only patch 適用 + diff-only audit。**Build: NOT RUN / CTest: NOT RUN（コメントのみ変更のため不要 — D154 40/40 ×2 + ST-1 200×2 の実証済み）**
**仕様:** D152-R2 確定事実への統一 — MSVC x64 の `std::atomic<16B>` by-value = lock-pool / `is_lock_free()==false` が正確 / atomic_ref の 16B lock-free 経路とは別物。**lock-free 性そのものを要件にしない**（要求 = 16B lifecycle state の原子的 CAS semantics + NonRT affinity）。

---

## 0. 判定（先出し）: **PASS — 7 グループ + sweep 発見 1 行を訂正。実行コード・assert 条件・実装は 0 変更。T3c 正式 close 条件充足**

---

## 1. 適用内容（7 グループ + 追加 1 行）

### 1.1 指示対象 7 グループ

| # | 位置 | 訂正内容 |
|---|---|---|
| S-1 | `ISRRuntimePublicationCoordinator.h:79-83`（W バックエンドコメント） | 「MSVC x64 provides a lock-free 16-byte specialization (SSE / cmpxchg16b)」→「STL lock-pool backend（`_Atomic_storage` spinlock・CAS = spinlock 排他下の全 16 バイト比較/コピー = 真の原子的相互排他 + フルバリア）・`is_lock_free()==false` は **ACCURATE report**（D152-R2 §1）・ctor で runtime 値を記録・false が期待値」 |
| S-2 | `ISRRuntimePublicationCoordinator.h:98`（教訓コメント） | 「D152-R1 §1 教訓: … 保身的 false が仕様」→「D152-R2 教訓: … atomic<16B by-value> は lock-pool 実装で false が正確」 |
| S-3 | `ISRRuntimePublicationCoordinator.h:413-417`（ctor コメント） | 「MSVC reports is_lock_free() conservatively … CMPXCHG16B-based」→「lock-pool backend・false is an ACCURATE report (not an anomaly — MSVC branch records only)」 |
| S-4 | `ISRRuntimePublicationCoordinator.h:423`（inline コメント） | 「conservative false is expected; actual path is CMPXCHG16B」→「false is the expected, accurate lock-pool report (D152-R2 §3); value recorded only」 |
| S-5a | `ISRDSPHandle.cpp:12, 16-20`（ctor コメント） | 「STL の保宅的判定。実際は CMPXCHG16B で lock-free に動作」→「lock-pool（spinlock）実装を正確に反映して false（保身ではない — D152-R2 確定事実）。CAS 意味論は lock-pool でも保持。runtime 値の記録のみ（異常扱いしない）」。見出しも「ロックフリー性を検証」→「runtime is_lock_free() 値を記録検証」 |
| S-6a〜S-6d | `ISRDSPHandle.h:22-26 / 204-206 / 212 / 218-220` | 「CMPXCHG16B を使用するには 16B アライメントが必要 / must be lock-free」→「将来の lock-free 切替（atomic_ref 参照形 / wrapper 案）への静的保証 + 原子的 CAS semantics。MSVC は lock-pool・false が正確・全消費者 NonRT」。static_assert メッセージ 2 件も同様に文言訂正（assert 条件自体は不変） |
| S-7 | `ISRRuntimePublicationCoordinator.h:93 / 96-97`（static_assert メッセージ） | メッセージ中の「(CMPXCHG16B)」→「(lock-free switch readiness)」×2（assert 条件は不変） |

### 1.2 sweep 発見分（本 patch 内で追加訂正・同 comment-only）

| 位置 | 内容 | 判定 |
|---|---|---|
| `ISRRuntimePublicationCoordinator.h:451` | tryInsert コメント内「acq_rel; x64 = lock cmpxchg16b」— 全域 sweep（手順 6）で発見 | 「acq_rel; MSVC = lock-pool CAS, D152-R2」に訂正。**2 行構成を維持**（+1 行が出ると下流行番号参照 h:593 等がずれるため） |
| `ISRShutdown.h:75` | 「sizeof(BlockingReasonStats) = 32 > 16 (x64 HW atomic limit: CMPXCHG16B)」 | **訂正不要・維持** — これは「x64 の HW 原子幅上限が 16B であり 32B 構造体は atomic 化不可」という正しい事実記述（実装経路の誤認ではない） |

## 2. diff-only audit（手順 1〜6 の実測）

| 手順 | 結果 |
|---|---|
| 1. `git diff --check` | **EXIT=0（whitespace clean）** |
| 2. `git diff --stat` / `--numstat` | 全体 9 files（D154 baseline 7 files + ISRDSPHandle.{cpp,h} が本 patch で新規追加）。`ISRDSPHandle.cpp` **6/6**・`ISRDSPHandle.h` **11/11** — 追加/削除同数 = 行数保持置換。`ISRRuntimePublicationCoordinator.h` は 289/83（うち T3c 実装分は D153/D154 baseline に含まれる既存 diff。本 patch 分は削除行と同数の置換で行数不変） |
| 3-4. 対象 diff の全行精査 | **全て `//`・`/* *` コメント行または static_assert メッセージ文字列のみ**。非コメント diff 行を抽出（grep で `^[+-]` から `//`・`*`・`"` 開始を除外）した結果、現れた行は全て T3c 実装（D153/D154 baseline・既に審査済み）の既存差分であり、本 patch によるコード追記は **0 行** |
| 5. 実行コード無変更 | static_assert の条件式（`sizeof`/`alignof`/`is_trivially_copyable_v`/`is_standard_layout_v`）・`std::atomic<DSPHandle>` 実装・`std::atomic<RecoveryLifecycleWord>` 実装・`_InterlockedCompareExchange128`（0 件のまま）・CMake・テストコード = **すべて無変更** |
| 6. 誤記 sweep（src 全域） | `CMPXCHG16B\|cmpxchg16b\|保身的\|保宅的\|lock-free 16-byte` → production では ISRShutdown.h:75 の正しい HW 上限記述 1 件のみ残存（訂正不要）。誤記 **0 件** |

**行アンカー維持確認**: h:386（lifecycle 宣言）・h:401（kMax=4）・h:593（peekLifecycleForTest）は全て patch 前と同一行 — 下流コメントの行番号参照を壊さない comment-only 編集を実現。

## 3. ConvoPeq.md 再生成（手順 7〜8）

- **再生成**: `python output_sourcecode_markdown.py` → `Generated: 2026-08-31 23:39:12`（GEN_EXIT=0）
- **手順 8（誤記再侵入確認）**: 再生成後の ConvoPeq.md で `保身的|保宅的|CMPXCHG16B|cmpxchg16b` を grep → **誤記 0 件**（唯一の残存は ISRShutdown.h 由来の正しい HW 上限記述 1 件）
- **新記述の反映確認**: 「lock-pool backend / ACCURATE report / D152-R2」の新文言が `:51041/:51054-55/:57769-71/:58104` 等に正しく取り込み済み（lock-pool 記述 11 件）

## 4. 判定

**D154-F2 = PASS。**

- 7 グループ + sweep 発見 1 行の計 12 箇所を、D152-R2 確定事実の記述に統一
- 実行コード・assert 条件・backend 実装・CMake・テスト = 0 変更（diff 実測）
- 行数保持により既存の行番号参照体系を維持
- ConvoPeq.md は 23:39:12 baseline として再生成・誤記再侵入なし
- 40/40 再実行は不要（指示どおり未実施 — D154 Debug/Release 40/40 + ST-1 200×2 の実証で十分）

**次工程: T3c Close Audit（既存証拠の統合による close 条件確認 — 再設計は行わない）**
