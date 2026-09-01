# D152-R1 — T3c Atomic Backend Specification Correction (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only specification correction. **Production source: 0 / Test source: 0 / CMake: 0 / build / CTest: 非着手。**
**基準:** ConvoPeq.md `Generated: 2026-08-31 15:32:58`（mtime 再実測: src より新しいもの 0 件、git worktree は D153 時点と同一・本修正によるソース変更なし）。
**目的:** D153-D（単一ブロッカー）を解消する。D152 §2/§2.3/§2.4/§11-V5/§12/§13-R4 の**アトミックバックエンド記述のみ**を訂正し、T1〜T7 の状態遷移仕様・権限不変条件・テスト仕様は**一切変更しない**。

> **【2026-08-31 D152-R2 訂正 — 重要】** D154-F1 の実証（layout 同一プローブ + disasm + /FAcs: exe 内 cmpxchg16b ゼロ・`is_lock_free()==false` は正確な報告）により、本文書 §1・§2・§3・§4・§5 の一部は**撤回・訂正済み**。本ツールチェーン（MSVC STL 14.52.36615）では `std::atomic<RecoveryLifecycleWord>`（by-value 16B）は **lock-pool（spinlock）で実装され lock-free ではない**。D152-R1 が引用した「lock-free using 16-byte intrinsics」特殊化（`_Atomic_storage<_Ty&, 16>`）は **atomic_ref 用の参照形**であり `std::atomic<T>` ではない（混同が D152-R1 の誤りの本体）。規範は **evidence/D152R2_T3C_ATOMIC_BACKEND_FACT_RESTATEMENT.md**。訂正済み箇所には【D152-R2 訂正】印を付す。

---

## 0. 変更範囲の宣言（指示 §3 準守）

```text
不変（本修正で触れない）:
  RecoveryLifecycleWord フィールド構成 / pending / adjudicated / delivery
  T1〜T7 の expected/new 遷移・T5 coalesce retry・T4 payload-before-Live
  liveCount authority / delivery authority / markTransientFailure 3分割
  CoordinatorLoop affinity（jassert 不採用）/ telemetry 条件 / NT-1〜NT-5
  V1〜V10 の基本構造（V5 の内容のみ下記 §5 で更新）

変更（本修正の対象）:
  D152 §1 include ブロック / §2 wrapper 定義 / §2.3 棄却根拠 / §2.4・§3 の load 規律の根拠
  §11 V5 条件 / §12 file 一覧（intrin.h 追加の削除・runtime 検証の追加）/ §13 R4
```

---

## 1. D152 §2.3 の撤回と訂正事実 —【D152-R2 訂正: 本節全体を撤回】

> **【D152-R2 訂正】** 本節（D152 §2.3 の撤回・一次資料 3 系統表・旧教訓）は**全体を撤回**する。
> 実際には **D152 §2.3「MSVC の 16B atomic は lock pool」が std::atomic（by-value）については正しかった**。誤りだったのは、MSVC STL の lock-free 16B 特殊化（`_Atomic_storage<_Ty&, 16>` = **atomic_ref 用参照形**）を `std::atomic<T>`（by-value）と**混同した D152-R1 §1 自身**である。
> 本ツールチェーン（MSVC STL 14.52.36615 実物ソースで検証）: by-value 16B 専用特殊化は存在せず、`std::atomic<RecoveryLifecycleWord>` はジェネリック locking 版 `_Atomic_storage`（store/load/exchange/CAS 全て `_Guard _Lock{_Spinlock}` per-object spinlock）を実体化する。`is_lock_free()==false` は**保身ではなく正確な報告**。詳細は evidence/D152R2_T3C_ATOMIC_BACKEND_FACT_RESTATEMENT.md §1。

~~**撤回**: 「MSVC STL は 16B を lock pool で実装する → `std::atomic<16B>` は却下」— この記述は**事実誤認**であった。~~

~~**訂正事実（一次資料 3 系統で整合）**:（旧 3 系統表 — atomic_ref 参照形の引用を by-value atomic の実装と誤認したため撤回。原文は git 履歴参照）~~

~~**教訓（仕様へ制度化）**: 「`is_always_lock_free == false` だから lock-free ではない」という論理を**以後いかなる仕様にも導入しない**…~~

**【D152-R2 訂正後の教訓（置換）】**: 「MSVC では `std::atomic<T>`（by-value）と `std::atomic_ref<T>`（参照形）は 16B で異なる実装経路を持つ（by-value = lock-pool・`is_lock_free()==false` が正確 / 参照形 = intrinsics lock-free・`is_always_lock_free==true`）。両者を混同して実装経路を記述しないこと。」

---

## 2. 確定案: (a) `std::atomic<RecoveryLifecycleWord>`

| 項目 | D152-R1 での確定 |
|---|---|
| W | `alignas(16) RecoveryLifecycleWord`（フィールド構成は D152 §1 と同一・不変） |
| atomic owner | `std::atomic<RecoveryLifecycleWord> lifecycle;`（LogicalRecoveryObligation の唯一所有権メンバ） |
| 16B 実装 | 【D152-R2 訂正】 ~~MSVC x64 の 16B 専用 atomic 特殊化（load/store=整列 16B SSE 単一アクセス、CAS/exchange=`lock cmpxchg16b`）を std::atomic 経由で使用~~ → **MSVC STL の lock-pool（spinlock `_Guard _Lock{_Spinlock}`）経路**。CAS は spinlock 排他下の全 16 バイト比較/コピー = 真の原子的相互排他 + フルバリア。`is_lock_free()==false` は正確な報告 |
| `is_lock_free()` | **runtime 検証対象**（§4 の検証コード・DSPHandle 前例パターン） |
| `is_always_lock_free` | **判定根拠として使用しない**（static_assert にも載せない — D152 §1 は既に含んでおらず不変） |
| intrinsic 直接使用 | production 0（`_InterlockedCompareExchange128` を自前呼出ししない。STL 内部使用は可） |
| ordering | `std::atomic` の標準 `memory_order` API（acquire load / acq_rel CAS / liveCount helper は現行維持） |
| 既存前例 | `std::atomic<DSPHandle>`（ISRDSPHandle.h:222-223 / .cpp:12-27）+ `std::atomic<LatencySnapshot>` 実行時検査（ConvolverProcessor.StateAndUI.cpp:962） |
| wrapper | 明示 `_InterlockedCompareExchange128` wrapper は**採用しない**（D152 §2 の `casLifecycle`/`loadLifecycleAdvisory` 自由関数は廃止。std::atomic メンバ関数を直接使用） |
| V5 | §5 の新条件へ更新 |

**設計意図の維持**: 「16B 全文を単一 atomic ownership domain として扱う」「commit は全文 CAS のみ」は D152 と同一。変更されるのはバックエンド（自前 intrinsic → 標準 `std::atomic<T>`）のみ。

---

## 3. 擬似コードの API マッピング（T1〜T7 の意味論は不変）

D152 §5 の各遷移は以下の呼び出し対応で**そのまま成立**（expected/new・retry・liveCount・telemetry の論理は一切変更なし）:

```text
D152 §2 の記述            → D152-R1 (a) の実API
loadLifecycleAdvisory(w)  → slot.lifecycle.load(memory_order_acquire)   // 判断経路
                            slot.lifecycle.load(memory_order_relaxed)   // telemetry/scan 候補絞り込み
casLifecycle(exp,des,dst) → slot.lifecycle.compare_exchange_strong(exp, des,
                                       memory_order::acq_rel)           // commit（失敗時 exp=現在値 — 標準保証）
```

- `compare_exchange_strong` の失敗時 expected 更新は C++ 標準の保証であり、D152 §2 の「失敗時再 load 不要 retry 規則」はそのまま成立。
- CAS の acq_rel は x64 では【D152-R2 訂正】~~`lock cmpxchg16b`（フルバリア）~~ **lock-pool spinlock 排他 + フルバリア**が実現 — D152 §2.4 の順序結論（追加 fence 不要）は不変。
- **load の整合性（根拠更新 —【D152-R2 訂正】）**: ~~alignas(16) により MSVC の 16B load は単一の整列 SSE アクセスとして発行され…~~ → lock-pool backend では load も同一 spinlock の排他下で実行される（`_Guard _Lock{_Spinlock}; _TVal _Local(_Storage);` — ジェネリック locking 版の実装）。**load は spinlock 排他により分断されない整合スナップショット**となる（CAS と同一ロックで直列化 — atomic_ref 参照形の SSE 単一アクセスに依存する記述は撤回）。
- したがって §3 の規律の**根拠を「torn 許容」から「torn は発生しないが、commit 権限は依然 CAS のみ」へ更新**する。実務上の帰結は同一: load 結果だけで遷移を commit してはならない（TOCTOU 防止 — load と CAS の間に他スレッド遷移が介入し得るため）。偽成功防止の本体は全文比較 CAS であり、これは不変。

---

## 4. runtime lock-free 検証（新設仕様・DSPHandle 前例の継承）

`RecoveryAdmissionTable` にコンストラクタを追加し、初期化時 1 回検証（ISRDSPHandle.cpp:12-27 と同一パターン）:

```cpp
template <std::size_t Capacity>
class RecoveryAdmissionTable {
public:
    RecoveryAdmissionTable() noexcept {
        static const bool isLockFree = [] {
            std::atomic<RecoveryLifecycleWord> test{};
            const bool ok = test.is_lock_free();
            // 【D152-R2 訂正】MSVC: is_lock_free()==false は保身ではなく lock-pool 実装の正確な報告。
            //   （旧記述「保身的 false を返し得る / 実際は CMPXCHG16B で lock-free」は撤回）
            //   Debug で ok==false は MSVC では期待値（異常ではない）。alignas(16) 崩れは
            //   static_assert(alignof(std::atomic<W>)>=16) が compile-time で検出する。
            jassert(ok || isDebugBreakDisabledForMsvc16b());   // 下記注参照 — 実装は前例の #if 分岐を踏襲
            return ok;
        }();
        (void)isLockFree;
    }
    ...
```

**実装注（前例厳密踏襲 —【D152-R2 訂正】）**: ISRDSPHandle.cpp:21-26 の `#if defined(_MSC_VER) (void)ok; #else assert(ok); #endif` 分岐を**そのまま踏襲**する（MSVC では記録のみ、非 MSVC では assert）。上記 jassert 行は説明用の疑似であり、D152-R1 の規範は前例コードそのものとする。Debug/Release 両構成で実行（前例と同じく ctor 1 回）。

**【D152-R2 訂正 — assert 対象範囲の明確化】**:
1. `is_lock_free()==false` は MSVC では lock-pool 実装の**正確な報告**であり、異常扱いしない。
2. MSVC 分岐 `(void)ok;`（記録のみ）は不変 — 根拠を「保身への回避」から「false は期待値」へ訂正。
3. 非 MSVC の `assert(ok)` は、**非 MSVC x64（Clang/GCC）では alignas(16) により 16B atomic が真の lock-free（cmpxchg16b）で実装され `is_lock_free()==true` が期待される**ことに依拠する。現行 spec の対象 toolchain は MSVC のみであり、MSVC では assert は発火しない設計（記録のみ）。

**追加 static_assert（D152 §1 の 4 点に加えて）**:
```cpp
static_assert(alignof(std::atomic<RecoveryLifecycleWord>) >= 16,
    "atomic<W> must keep 16-byte alignment for CMPXCHG16B (ISRDSPHandle.h:212 前例)");
```
（`is_always_lock_free` の assert は**追加しない** — §1 教訓。）

---

## 5. V5 更新（D153-R-03 対応）

```text
旧 V5: `_InterlockedCompareExchange128` 直接使用 = 1（wrapper 定義）
新 V5【D152-R2 訂正で最終確定】:
  (i)  intrinsic direct use                              = **0**（production 直接使用禁止）
  (ii) std::atomic<RecoveryLifecycleWord> storage        = **1**（lifecycle メンバ宣言のみ）
  (iii) runtime is_lock_free() probe                     = **1**（RecoveryAdmissionTable ctor・前例パターン）
  (iv)  expected MSVC result                             = **false**（正確な報告 — §4 訂正参照）
```

**【D152-R2 訂正】** runtime verification の「成功」を `is_lock_free()==true` と定義**しない**。probe は runtime 値の記録であり、MSVC での期待値は `false` である。V5 判定は (i)-(iv) の計数一致をもって PASS。

V1〜V4・V6〜V10 は D152/D153 のまま不変。

---

## 6. §12 file 一覧・§13 リスクの追随

- **§12 変更**: ISRRuntimePublicationCoordinator.h の変更欄から「`<intrin.h>` 追加・`_M_X64` #error ガード」を**削除**（不要化）。追加項目: RecoveryAdmissionTable ctor（runtime 検証）+ alignof static_assert。§1 の include ブロックは現行 include（`<atomic>`/`<cstdint>`/`<type_traits>` 既存）のまま新規追加なし。
- **§13 R4 更新**: 「`_M_X64` ガード」→「x64 前提はプロジェクト既存（ISRDSPHandle.h:205 'x64 ABI is assumed throughout ISR'）。ARM64 移植時は std::atomic 16B の `_acq/_rel` 変種相当の検証を再設計 — 本パッチ範囲外」。
- **§13 R1〜R3/R5〜R7 不変**。

---

## 7. D153-R への引き継ぎ（照合点）

| D153-R | 照合内容 | 本仕様の該当節 |
|---|---|---|
| R-01 | 「16B=lock pool」記述の消滅・一次資料整合・is_always_lock_free/runtime 混同なし | §1 |
| R-02 | W → std::atomic<W> → 全文 CAS commit の連鎖と T1〜T7 意味論不変 | §2/§3 |
| R-03 | V5 新条件（直接使用 0 / atomic 宣言 1 / runtime 検証 1） | §5 |
| R-04 | baseline（ConvoPeq 15:32:58 / newer 0 / worktree 無変化） | 本ヘッダ |

**判定: D152-R1 = 仕様訂正确定。D153-R（限定再監査）へ。**
