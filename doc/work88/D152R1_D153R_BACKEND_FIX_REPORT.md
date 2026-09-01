# D152-R1 + D153-R — T3c Atomic Backend 訂正・限定再監査（Work Report）

```text
Production/Test/CMake changes: 0 / Build: NOT RUN / CTest: NOT RUN
基準: ConvoPeq.md 15:32:58（newer src 0・git worktree は D153 時点と同一 = 本作業によるソース変更 0）
詳細: evidence/D152R1_T3C_ATOMIC_BACKEND_CORRECTION.md / evidence/D153R_T3C_BACKEND_REAUDIT.md
```

**Status: D152-R1 仕様訂正确定 → D153-R = PASS。ブロッカー D153-D 解消。次工程（T3c production implementation）の監査上の障害はなし。**

> **【2026-08-31 D152-R2 訂正】** D154-F1 の実証（exe 内 cmpxchg16b ゼロ・`is_lock_free()==false` は正確な報告）により、下記「D152-R1 の確定内容」のうち atomic backend の**事実記述は撤回・訂正済み**。本ツールチェーン（MSVC STL 14.52.36615）では `std::atomic<RecoveryLifecycleWord>`（by-value 16B）は **lock-pool（spinlock）で実装**され lock-free ではない。D152-R1 が引用した「lock-free 16B 特殊化」は **atomic_ref 用参照形（`_Atomic_storage<_Ty&, 16>`）**であり `std::atomic<T>` ではない（混同が誤りの本体）。**backend 選択（std::atomic<RecoveryLifecycleWord> 維持・intrinsic wrapper 不採用）と T1〜T7 意味論は D152-R2 でも不変** — lock-pool 許容は D150 §12 の既存承認の範囲内。規範: evidence/D152R2_T3C_ATOMIC_BACKEND_FACT_RESTATEMENT.md。

## D152-R1 の確定内容（変更はバックエンドのみ）

- **~~撤回~~【D152-R2 で本項全体を撤回】**: ~~D152 §2.3「MSVC の 16B atomic は lock pool」は事実誤認。一次資料 3 系統で訂正 — ①MSVC STL ソース `_Atomic_storage<_Ty&,16> // lock-free using 16-byte intrinsics`（load=`__iso_volatile_load16`、CAS=`_InterlockedCompareExchange128`）、②`_Is_always_lock_free<=8` は**保身的 compile-time 定数**にすぎない、③プロジェクト前例 ISRDSPHandle.cpp:16-22 + 実ストレージ `std::atomic<DSPHandle>`（h:222-223）+ 第 2 前例 `std::atomic<LatencySnapshot>` 実行時検査（ConvolverProcessor.StateAndUI.cpp:962）。~~ 【D152-R2 訂正】D152 §2.3 の lock pool 記述は std::atomic（by-value）については**正しかった**。D152-R1 は atomic_ref 参照形の特殊化を `std::atomic<T>` と混同した。`is_lock_free()==false` は保身ではなく正確な報告。
- **確定案 (a)**: `std::atomic<RecoveryLifecycleWord>`（alignas(16)）を lifecycle の唯一所有権メンバに。ordering は標準 memory_order API（acquire load / acq_rel CAS）。明示 intrinsic wrapper は**不採用**。`is_always_lock_free` は判定根拠・static_assert とも使用禁止（教訓制度化）。runtime `is_lock_free()` 検証を RecoveryAdmissionTable ctor に追加（DSPHandle の `#if _MSC_VER` パターン厳密踏襲）。static_assert に `alignof(std::atomic<W>)>=16` を追加。
- **不変宣言**: W フィールド構成、T1〜T7 の expected/new・retry・liveCount gate、T5 coalesce retry、T4 payload-before-Live、delivery/liveCount authority、markTransientFailure 3 分割、CL affinity（jassert 不採用）、telemetry、NT-1..5、V1-V10 基本構造。§3 の commit 規律は「load は整合スナップショット（整列 16B SSE 単一アクセス・Intel SDM 8.1.1）だが commit 権限は TOCTOU 防止のため CAS のみ」に根拠更新、実務規則は同一。
- **V5 更新**: intrinsic 直接使用 0 / `std::atomic<RecoveryLifecycleWord>` 宣言 1 / runtime 検証 1。【D152-R2 訂正】**expected MSVC result = false** を追加（「成功」を `is_lock_free()==true` とは定義しない）。

## D153-R 再監査結果: **PASS（R-01..R-04）**

- R-01: 「lock pool」記述は撤回文脈 2 箇所のみ・規範記述残存 0。is_always_lock_free/runtime 混同なし。一次資料引用は全て現行ソース・取得済み STL ソースと一致。
- R-02: W→std::atomic<W>→全文 CAS commit の連鎖確認。API マッピングは呼び出し形の置換のみで T1〜T7 の意味論不変。失敗時 expected 更新は標準保証。禁止構造（CAS 後の別 atomic 更新）不在のまま。
- R-03: 新 V5 条件が要求どおり。
- R-04: baseline 再実測（15:32:58 / newer 0 / worktree 無変化）。
- 軽微注記（非ブロッキング）: §4 の jassert 行は説明用疑似 — 規範は前例 `#if` 分岐。実装後監査で踏襲照合項目に追加。

## 次工程（指示順序）

```text
D153-R PASS（現在地）
   ↓
T3c production implementation   ← 次の指示まで凍結継続
   ↓
implementation audit（V1-V10 + 前例踏襲照合）
   ↓
Debug build → Debug CTest → Release build → Release CTest
   ↓
D154 相当の実装後検証
```

**本ターンでは実装・build・CTest を行わない。`RecoveryLifecycleWord` の std::atomic 化を含むソース変更は次ターン以降の指示まで凍結。**
