# D152-R2 — T3c Atomic Backend 事実訂正（Work Report）

```text
Production source changes: 0 / Test source changes: 0 / CMake changes: 0 / Build: NOT RUN / CTest: NOT RUN
変更対象 = 仕様/evidence 文言のみ（D152-R1 evidence + work88 report）
基準: git worktree は D154 検証時と同一（7 files, +2798/−460 — 本作業によるソース diff 追加 0）
詳細: evidence/D152R2_T3C_ATOMIC_BACKEND_FACT_RESTATEMENT.md
```

**Status: D152-R2 確定。D152-R1 の「`std::atomic<16B>` は runtime lock-free」記述を撤回し、lock-pool（spinlock）実装の事実に訂正。backend 選択（`std::atomic<RecoveryLifecycleWord>` 維持）・T1〜T7 意味論は不変。次工程 = D154-R2（軽量 read-only 再監査）。**

## 背景 — D154-F1 の再発防止

D152-R1 は D152 §2.3（lock pool 記述）を撤回して「MSVC x64 の `std::atomic<16B>` は lock-free」と確定したが、D154-F1 の実証（layout 同一プローブ + disasm + /FAcs）により**この確定自体が誤り**だった。誤りの本体は **atomic_ref 用参照形特殊化（`_Atomic_storage<_Ty&, 16>` // lock-free using 16-byte intrinsics）を `std::atomic<T>`（by-value）と混同**した点。D152-R2 でこの事実認識を訂正する（T3c 自体の再実装・backend 切替は行わない）。

## 本ツールチェーンでの実測（MSVC STL 14.52.36615 実物 `include/atomic`）

| 事実 | 実測 |
|---|---|
| by-value 16B 専用特殊化 | **存在しない**（`_Atomic_storage<_Ty, 16>` 検索 = 0 件）。`std::atomic<RecoveryLifecycleWord>` はジェネリック locking 版 `_Atomic_storage`（line 527-）を実体化 |
| locking 版の実装 | store/load/exchange/CAS 全て `_Guard _Lock{_Spinlock}`（per-object spinlock `long`）。CAS = spinlock 排他下の全 16 バイト memcmp/memcpy = 真の原子的相互排他 + フルバリア |
| `is_lock_free()` | `_Is_always_lock_free<sizeof(_Ty)>`（compile-time 定数）を返す → 16B で **false**。実装も lock-pool なので **false は正確な報告（保身ではない）** |
| lock-free 16B 特殊化 | `_Atomic_storage<_Ty&, 16>`（line 1102-）= **atomic_ref 用参照形のみ**（`__iso_volatile_load16/store16` + `_InterlockedCompareExchange128`）。`atomic_ref : _Choose_atomic_base_t<_Ty, _Ty&>`（line 2276）経由で参照形が選択される |
| D154-F1 プローブ整合 | `is_lock_free()==false`・exe 内 cmpxchg16b ゼロ — 3 系統（本 STL・GitHub 本家・disasm）で整合 |

## 確定内容（ユーザー指示 5 項目の反映）

1. **§1 訂正**: 「MSVC x64 の `std::atomic<16B>` は runtime lock-free」を**撤回**。`std::atomic<RecoveryLifecycleWord>` は本 toolchain で `is_lock_free()==false`・CAS は STL lock-pool/spinlock 経由。atomic_ref の `_Atomic_storage<_Ty&, 16>` と `std::atomic<_Ty>` を混同しない。
2. **§2 維持**: backend は `std::atomic<RecoveryLifecycleWord>` のまま。**intrinsic wrapper への切替は今回行わない**。理由: D154 で確認された通り T3c の W への接触者は CoordinatorLoop / RebuildThread で ISR ではない（D150 §2.3 実測・audio callback 0 hit）。
3. **§4 訂正**: ctor 検証仕様の事実記述を訂正 — `is_lock_free()==false` を異常扱いしない・MSVC は前例同様 runtime 値を記録するだけ・非 MSVC `assert(ok)` は非 MSVC x64（alignas(16) で真の lock-free）が対象範囲であることを明確化。
4. **§5（V5）更新**: `intrinsic direct use = 0 / std::atomic<RecoveryLifecycleWord> storage = 1 / runtime is_lock_free() probe = 1 / expected MSVC result = false`。runtime verification の「成功」を `is_lock_free()==true` と定義しない。
5. **安全性要件の明示**: 「T3c の安全性要件は lock-free 性ではなく、16B lifecycle state の原子的 CAS semantics と非RT affinity を満たすことである」— 根拠は **D150 §12 の既存承認**（「実 call graph で W 触达者 = {CL, RebuildThread}、実 audio callback 非接触。よって lock-pool でも RT 制約は違反しない」— evidence/D150_T3C_LIFECYCLE_WORD_REPROOF.md:215）であり、新たな安全性判断は追加していない。

## 適用した文書変更

| ファイル | 変更 |
|---|---|
| `evidence/D152R2_T3C_ATOMIC_BACKEND_FACT_RESTATEMENT.md` | **新設**（本訂正の規範本文） |
| `evidence/D152R1_T3C_ATOMIC_BACKEND_CORRECTION.md` | 冒頭に D152-R2 訂正バナー + §1 全体撤回（旧 3 系統表・旧教訓を取り消し線で保持）+ §2 16B 実装行訂正 + §3 load 根拠訂正 + §4 疑似コードコメント・assert 範囲訂正 + §5 V5 訂正 |
| `doc/work88/D152R1_D153R_BACKEND_FIX_REPORT.md` | Status 直下に D152-R2 訂正バナー + 「撤回」項を撤回戻し（D152 §2.3 は正しかった）+ V5 行に expected false 追加 |

## production source 残存誤コメント（D154-F2 別トラックへ切り出し）

本訂正は production source を変更しないため、以下が現行ソースに残存する（全て comment-only 訂正可能・実装変更不要）:

- S-1 `ISRRuntimePublicationCoordinator.h:79-83`（lock-free 16B 特殊化の誤記述）
- S-2 `ISRRuntimePublicationCoordinator.h:98`（「保身的 false」）
- S-3 `ISRRuntimePublicationCoordinator.h:413-417`（ctor コメント・CMPXCHG16B-based 誤記）
- S-4 `ISRRuntimePublicationCoordinator.h:423`（inline コメント・D154-F1 follow-up ② 対象）
- S-5 `ISRDSPHandle.cpp:16-20` / S-6 `ISRDSPHandle.h:204-218`（DSPHandle 側同一誤り = ユーザー指示の D154-F2 本体）
- S-7 `ISRRuntimePublicationCoordinator.h:93/96-97` static_assert メッセージ文言（コード自体は維持）

historical 記録（D153 系 report/evidence）は当時の監査記録として改変しない（D152-R2 が規範として supersede）。

## 次工程（指示順序）

```text
D152-R2 確定（現在地）
   ↓
D154-R2 軽量 read-only 再監査（R2-01〜R2-08）
   ↓
ST-1 AV stress 200（lock-pool backend 下の contention / lifecycle pressure）
   ↓
RT affinity audit（lifecycle W が ISR/AudioThread に触れていないことの再確認）
   ↓ PASS → T3c close candidate / FAIL → STOP・新規監査
   ↓
D154-F2（DSPHandle + T3c 側残存誤コメントの comment-only 別トラック）
```
