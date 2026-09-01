# D153-R — T3c Atomic Backend 限定再監査（read-only）

```text
Production source changes: 0
Test source changes:       0
CMake changes:             0
Build:                     NOT RUN
CTest:                     NOT RUN

Source baseline (R-04):
  ConvoPeq.md = Generated: 2026-08-31 15:32:58 / src newer = 0（ctx_execute 実測）
  git worktree(src) = 5 files M — D153 時点と同一（セッション開始時既存の G-4.x/F6 変更）。
    D152-R1 / D153-R によるソース変更 0（本監査の書込は evidence/doc のみ）。
```

**照合対象**: D152-R1（`evidence/D152R1_T3C_ATOMIC_BACKEND_CORRECTION.md`）の修正節のみ。D152 本体の T1〜T7・権限不変条件・テスト仕様は照合対象外（不変宣言済み）。

---

## R-01 — 事実整合: **PASS**

| 検査 | 結果 |
|---|---|
| 「16B atomic = lock pool」記述の残存 | **なし**。D152-R1 内 "lock pool" 出現は 2 箇所のみ = ①撤回宣言（§1:29「事実誤認であった」）②照合チェックリスト項目名（§7）。規範記述としての残存有 0 |
| MSVC 16B atomic の説明と一次資料の整合 | **整合**。STL ソース `_Atomic_storage<_Ty&,16> // lock-free using 16-byte intrinsics`・load=`__iso_volatile_load16`・CAS=`_InterlockedCompareExchange128`（本セッション raw 取得）と D152-R1 §1 表が一致。プロジェクト前例 ISRDSPHandle.cpp:16-22（保身的判定／実動作 CMPXCHG16B）+ 実ストレージ h:222-223 + 第 2 前例 ConvolverProcessor.StateAndUI.cpp:962 を引用 |
| `is_always_lock_free` と runtime `is_lock_free()` の混同 | **なし**。§1 教訓で「compile-time 定数=保身的／runtime 検証=判定根拠」を分離し、§2 表で「is_always_lock_free は判定根拠に使用しない・static_assert にも載せない」、§4 で runtime 検証を規範化。追加 assert は `alignof(std::atomic<W>)>=16`（整列保証）のみで混同なし |

## R-02 — 実装方式整合: **PASS**

```text
RecoveryLifecycleWord（フィールド構成不変）
   ↓
std::atomic<RecoveryLifecycleWord> lifecycle（宣言 1 箇所）
   ↓
compare_exchange_strong（全文 16B 比較・acq_rel）= 唯一の commit 手段
```
- §3 の API マッピングは**呼び出し形の置換のみ**で、D152 §5 の T1〜T7 の expected/new・分岐・retry・liveCount gate・telemetry 条件を 1 文字も変更しないことを対照確認（`casLifecycle→compare_exchange_strong`、`loadLifecycleAdvisory→load(acquire/relaxed)`）。
- 「失敗時 expected 更新」は C++ 標準保証（compare_exchange_strong）で、D152 §2 の再 load 不要 retry 規則は維持。
- 「CAS のみ commit」規律: §3 で根拠を「torn 許容」から「整列 16B load は整合スナップショットだが、commit 権限は TOCTOU 防止のため依然 CAS のみ」へ更新 — **実務規則は同一**（load 単独で遷移を commit しない）。x64 の整列 16B load/store 原子性は Intel SDM Vol.3A 8.1.1 に基づき、DSPHandle が同一前提で実運用中 — 前例整合。
- 禁止構造（CAS 成功後に別 atomic 更新で意味を完成）の不在: 不変（T2/T3/T4 の liveCount/telemetry はいずれも CAS 勝者 gate — D152 §5.8 と同一）。

## R-03 — V5 更新: **PASS**

D152-R1 §5 の新 V5（(i) intrinsic 直接使用 0 / (ii) `std::atomic<RecoveryLifecycleWord>` 宣言 1 / (iii) runtime `is_lock_free()` 検証 1 箇所）が D153-R-03 の要求条件と一致。V1〜V4・V6〜V10 不変確認。

## R-04 — source baseline: **PASS**（上記ヘッダ実測）

## 軽微注記（非ブロッキング）

- D152-R1 §4 の jassert 行は「説明用疑似」であることを明記済みで、規範は ISRDSPHandle.cpp:21-26 の `#if defined(_MSC_VER)` 分岐踏襲。実装者は §4 末尾の「実装注（前例厳密踏襲）」を優先すること。D153（実装後監査）でこの踏襲を照合項目に含める。

---

## 判定

```text
D153-R = PASS（R-01..R-04 全項目）
  → D153 の単一ブロッカー D153-D は解消。
  → 次工程: T3c production implementation（D152 + D152-R1 準拠）
     → implementation audit（D153 実装後版・V1-V10 + 前例踏襲照合）
     → Debug build → Debug CTest → Release build → Release CTest → D154 相当実装後検証。
本監査時点では依然として実装・build・CTest 非実施（凍結は次ターン以降の指示まで継続）。
```
