# D152-R2 — T3c Atomic Backend Fact Restatement (spec/evidence 文言訂正のみ)

**Date:** 2026-08-31 (+09:00)
**Type:** 仕様/evidence の文言訂正のみ。**Production source: 0 変更 / Test source: 0 変更 / CMake: 0 / build: 0 / CTest: 0。**
**Baseline:** git worktree = D154 検証時と同一（`git diff --stat`: 7 files, +2798/−460 — T3c 実装 D152→D154 分）。本訂正によるソース diff 追加は**ゼロ**。
**目的:** D154-F1 が実証した「D152-R1 の atomic backend に関する事実認識の誤り」を、D152-R1 規範文書から撤回・訂正する。T1〜T7 の状態遷移仕様・権限不変条件・backend 選択は**一切変更しない**。

---

## 0. 変更範囲の宣言

```text
不変（本訂正で触れない）:
  RecoveryLifecycleWord フィールド構成 / pending / adjudicated / delivery
  T1〜T7 の expected/new 遷移・T5 coalesce retry・T4 payload-before-Live
  backend 選択 = std::atomic<RecoveryLifecycleWord>（维持 — §2）
  liveCount authority / delivery authority / markTransientFailure 3分割
  CoordinatorLoop affinity（jassert 不採用）/ telemetry 条件 / NT-1〜NT-5
  V1〜V4・V6〜V10（V5 の内容のみ §4 で更新）

変更（本訂正の対象 — 文言のみ）:
  D152-R1 §1 の lock-free 事実記述（撤回 → lock-pool 訂正）
  D152-R1 §2 の 16B 実装経路記述（撤回 — backend 選択自体は不変）
  D152-R1 §3 の load 整合性の根拠（「整列 SSE 単一アクセス」→ spinlock 排他下コピー）
  D152-R1 §4 の ctor 検証仕様の事実記述（保身/正確の訂正・assert 対象範囲明確化）
  D152-R1 §5 の V5 条件（expected MSVC result = false を追加）
  D152-R1 の「教訓」条項（atomic/atomic_ref 混同に基づくため訂正）
```

**production source の扱い**: D154-F1 follow-up ②（table ctor コメント h:423 修正）および付随 ④（ISRDSPHandle コメント）は source への文言反映を伴うため、本訂正では**実施しない**。D154-F2（別トラック・comment-only patch）へ切り出し済み。本訂正時点で production source に残存する誤コメントの一覧は §6。

---

## 1. D152-R1 §1 の撤回と訂正事実

**撤回**: D152-R1 §1「MSVC x64 の `std::atomic<16B>` は runtime lock-free（load/store = 整列 16B SSE 単一アクセス、CAS = `lock cmpxchg16b`）」— この記述は**事実誤認**であった。D154-F1 の layout 同一プローブ + disasm + /FAcs が否定（exe 内に cmpxchg16b ゼロ・`is_lock_free()==false`）。

**訂正事実（本ツールチェーンの MSVC STL 14.52.36615 実物ソースで検証・GitHub 本家 raw と一致）**:

| # | 事実 | 実測根拠（MSVC 14.52.36615 `include/atomic`） |
|---|---|---|
| 1 | `std::atomic<RecoveryLifecycleWord>`（by-value・16B）は**ジェネリック locking 版 `_Atomic_storage`** を実体化する。store/load/exchange/CAS は全て `_Guard _Lock{_Spinlock}`（per-object spinlock `long`）による排他下のコピー/比較で実装 | by-value 16B 専用特殊化は**存在しない**（`_Atomic_storage<_Ty, 16>` の検索 = 0 件）。実在特殊化は size 1/2/4/8 と参照形 16 のみ。ジェネリック版（line 527-）のコメント「Locking version used when hardware has no atomic operations for sizeof(_Ty)」 |
| 2 | `is_lock_free()==false` は**保身ではなく正確な報告**。`atomic::is_lock_free()` は `_Is_always_lock_free<sizeof(_Ty)>`（compile-time 定数）を返すが、16B by-value の実装も実際に lock-pool であるため値は実装と一致する | `is_lock_free()` 実装（line 2133-2138）+ ジェネリック locking 版の CAS 実体（spinlock 下 memcmp/memcpy） |
| 3 | D152-R1 §1 が引用した「`_Atomic_storage<_Ty&, 16> { // lock-free using 16-byte intrinsics }`」は **atomic_ref 用の参照形（`_Ty&`）特殊化**であり、`std::atomic<T>` ではない。参照形のみ `__iso_volatile_load16/store16` + `_InterlockedCompareExchange128`（真の lock-free）を使用する | `_Atomic_storage<_Ty&, 16>`（line 1102-）+ `atomic_ref : _Choose_atomic_base_t<_Ty, _Ty&>`（line 2276）。atomic_ref::is_always_lock_free = `sizeof(_Ty) <= 2*sizeof(void*) && pow2` → 16B で true |
| 4 | D154-F1 プローブ実測と整合: `is_lock_free()==false`、exe 内 cmpxchg16b ゼロ、CAS は lock-pool 経由で真の原子的相互排他 + フルバリア | evidence/D154_T3C_BUILD_GATE_VERIFICATION.md（D154-F1） |

**訂正された教訓（D152-R1 §1 教訓の置換）**:

- 旧教訓（撤回）: 「`is_always_lock_free == false` だから lock-free ではない、という論理を導入しない」（MSVC 16B は実際 lock-free という前提）
- **新教訓**: 「MSVC では、**`std::atomic<T>`（by-value）と `std::atomic_ref<T>`（参照形）は 16B で異なる実装経路を持つ**。by-value 16B = lock-pool（`is_lock_free()==false` が**正確**）、参照形 16B = intrinsics lock-free（`is_always_lock_free==true`）。両者を混同して実装経路を記述しないこと。判定は runtime `is_lock_free()` 値を実装経路と突き合わせて行う」

---

## 2. D152-R1 §2 の訂正（backend 選択は不変・実装経路記述のみ撤回）

| 項目 | D152-R2 での確定 |
|---|---|
| W | `alignas(16) RecoveryLifecycleWord`（不変） |
| atomic owner | `std::atomic<RecoveryLifecycleWord> lifecycle;` — **維持**（変更しない） |
| 16B 実装 | **MSVC STL の lock-pool（spinlock）経路**。CAS は spinlock 排他下の全 16 バイト比較/コピー = 真の原子的相互排他 + フルバリア。`is_lock_free()==false` は正確な報告 |
| intrinsic wrapper への切替 | **今回行わない**（D152 §2 の明示 `_InterlockedCompareExchange128` wrapper 案は将来の切替候補として仕様済みのまま維持） |
| 切替しない理由 | D154 で確認された通り、T3c の W への接触者は **CoordinatorLoop / RebuildThread** であり ISR ではない（D150 §2.3 実測・audio callback 0 hit）。lock-pool は NonRT affinity 下で RT 制約に違反しない（D150 §12 既存承認・§5 引用） |
| `is_always_lock_free` | 判定根拠・static_assert とも不使用（D152 §1 から不変） |
| ordering | 標準 `memory_order` API（acquire load / acq_rel CAS）— 不変 |

**設計意図の維持**: 「16B 全文を単一 atomic ownership domain として扱う」「commit は全文 CAS のみ」は不変。変更されるのは「実装経路の事実記述」のみで、backend 選択・T1〜T7 意味論は一切変更しない。

---

## 3. D152-R1 §4 の訂正（runtime `is_lock_free()` 検証仕様）

`RecoveryAdmissionTable` ctor の検証仕様は**コード構造のまま不変**（DSPHandle 前例の `#if defined(_MSC_VER) (void)ok; #else assert(ok); #endif` 分岐）。以下の事実記述を訂正する:

1. **`is_lock_free()==false` を異常扱いしない。** MSVC では 16B by-value atomic の `is_lock_free()` は lock-pool 実装を正確に反映して false を返す（§1 事実 #2）。D152-R1 §4 の擬似コードコメント「保身的 false を返し得る / 実際は CMPXCHG16B で lock-free」は**撤回**する。
2. **MSVC では前例同様、runtime 値を記録するだけ。** `(void)ok;`（記録のみ）は D152-R1 §4 から不変 — ただしその根拠を「保身への回避」から「false は期待値であり異常でない」へ訂正。
3. **非 MSVC の `assert(ok)` は現仕様の対象範囲を明確化。** Clang/GCC x64 では alignas(16) により 16B atomic が真の lock-free（`cmpxchg16b`）で実装され `is_lock_free()==true` が期待されるため、**非 MSVC x64 ツールチェーンでのみ** assert が妥当である。現行 spec の対象は MSVC toolchain であり、MSVC では assert は発火しない設計（記録のみ）であることを明記する。
4. 追加 static_assert `alignof(std::atomic<W>) >= 16`（h:96-97）は**維持**（alignas(16) は lock-pool backend でも整列コピーの効率と将来の切替（atomic_ref 参照形は alignas 必須）を保証する）。メッセージ文言中の「CMPXCHG16B」は実装経路を示すものではない点に留意（文言の清掃候補は §6 / D154-F2）。

---

## 4. V5 更新（D152-R1 §5 の置換）

```text
intrinsic direct use                              = 0
std::atomic<RecoveryLifecycleWord> storage        = 1
runtime is_lock_free() probe                      = 1
expected MSVC result                              = false
```

**重要**: runtime verification の「成功」を `is_lock_free()==true` と定義しない。probe の役割は runtime 値の記録であり、MSVC での期待値は `false` である（§3-1）。V5 判定は上記 4 項目の計数一致をもって PASS とする。

V1〜V4・V6〜V10 は D152/D153 のまま不変。

---

## 5. T3c 安全性要件の明示（D150 既存承認の引用）

```text
T3c の安全性要件は lock-free 性ではなく、
16B lifecycle state の原子的 CAS semantics と
非RT affinity を満たすことである。
```

この要件の根拠は**既存の承認内容**であり、本訂正で新たな安全性判断を追加するものではない:

- **D150 §12（evidence/D150_T3C_LIFECYCLE_WORD_REPROOF.md:213-215）**: 「MSVC STL は 16B を lock pool（critical section）で実装（`is_always_lock_free==false`）」を前提に、「lock-pool 可否の判定（短絡しない）: 実 call graph で W 触达者 = {CL, RebuildThread}、実 audio callback 非接触（§2.3 実測 0 hit）。**よって lock-pool でも RT 制約は違反しない**」— lock-pool 許容の事前承認。
- **D154-F1 実測**: lock-pool backend 下で 16B RMW は真の原子的相互排他 + フルバリア（CAS 意味論プローブ実証・NT-5 実 2 スレッド競合両構成通過・Debug/Release CTest 40/40）。

したがって現行 `std::atomic<RecoveryLifecycleWord>` backend は D150 承認条件（非RT affinity）を満たしたまま維持される。真の lock-free（atomic_ref 参照形相当 / `_InterlockedCompareExchange128` wrapper）が将来必要になるのは「W への ISR/AudioThread 接触が新設される場合」に限られる（その場合の切替は D152 §2 wrapper 案を適用）。

---

## 6. production source 残存誤コメントの一覧（D154-F2 別トラックへ）

本訂正は production source を変更しないため、以下の誤コメントが**現行ソースに残存する**。全て comment-only で訂正可能（実装変更不要）であり、**D154-F2 — Existing DSPHandle atomic-backend comment audit（T3c 側残存分を含む comment cleanup track）**として切り出し済み:

| # | 位置 | 誤記述内容 |
|---|---|---|
| S-1 | `ISRRuntimePublicationCoordinator.h:79-83` | 「MSVC x64 provides a lock-free 16-byte specialization (load/store = aligned 16B SSE access; CAS = lock cmpxchg16b)」— 実装経路の誤記述（正 = lock-pool） |
| S-2 | `ISRRuntimePublicationCoordinator.h:98` | 「MSVC では保身的 false が仕様」— false は正確な報告（保身ではない） |
| S-3 | `ISRRuntimePublicationCoordinator.h:413-417` | ctor コメント「MSVC reports is_lock_free() conservatively ... while the implementation is CMPXCHG16B-based」— 同一誤り |
| S-4 | `ISRRuntimePublicationCoordinator.h:423` | inline コメント「conservative false is expected; actual path is CMPXCHG16B」— D154-F1 follow-up ② の対象 |
| S-5 | `ISRDSPHandle.cpp:16-20` | 「STL の保身的判定。実際は InterlockedCompareExchange128 (CMPXCHG16B) で lock-free に動作」— `std::atomic<DSPHandle>`（by-value 16B）にも同一誤りが適用される（D154-F1 付随 ④） |
| S-6 | `ISRDSPHandle.h:204-205 / 214-218` | 「std::atomic<DSPHandle> must be lock-free ... (CMPXCHG16B, Haswell+ / AVX2)」— 同一誤り |
| S-7 | `ISRRuntimePublicationCoordinator.h:93 / 96-97` static_assert メッセージ | メッセージ文言中の「CMPXCHG16B」（コード自体は維持・文言清掃候補） |

**historical 記録の扱い**: D153 系 report/evidence が引用する「CMPXCHG16B lock-free」記述は当時の監査記録として**改変しない**（D152-R2 本文で supersede 済み）。規範は本 D152-R2 + 訂正済み D152-R1 evidence。

---

## 7. D154-R2 への引き継ぎ（照合点）

| D154-R2 | 照合内容 | 本訂正の該当節 |
|---|---|---|
| R2-01 | 「`std::atomic<16B>` = lock-free」誤記述が D152-R1 から消滅 | §1 + D152-R1 evidence/work88 report の訂正 |
| R2-02 | atomic（by-value）/ atomic_ref（参照形）の MSVC STL 特殊化混同なし | §1 事実 #1/#3 |
| R2-03 | backend = `std::atomic<RecoveryLifecycleWord>` のまま | §2 |
| R2-04 | `compare_exchange_strong` 全文 CAS semantics 不変 | §0 不変リスト（実コード 0 変更で担保） |
| R2-05 | T1〜T7 意味論変更なし | §0 不変リスト（実コード 0 変更で担保） |
| R2-06 | V5 = intrinsic 0 / atomic<W> 1 / runtime probe 1 / expected false | §4 |
| R2-07 | `is_lock_free()==false` を T3c failure と誤定義していない | §3-1 / §4 |
| R2-08 | D154-F1 結論と矛盾しない | 全節（D154-F1 実測を規範へ昇格） |

**判定: D152-R2 = 仕様訂正確定。D154-R2（軽量 read-only 再監査）へ。**
