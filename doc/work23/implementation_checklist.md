# 実装チェックリスト

**計画書**: doc/work23/bug_fix_plan.md v6
**作成日**: 2026-06-08

---

## 凡例

| 記号 | 意味 |
| --- | --- |
| 🔴 | 未着手 |
| 🟡 | 作業中 |
| ✅ | 完了 |
| ❌ | 問題あり（要対応） |

---

## Phase 1: BUG-01 OutputFilter

**ファイル**: DSPCoreDouble.cpp, DSPCoreFloat.cpp
**変更**: `if (!convIsLast)` 削除、process() 常時呼び出し

- [ ] 🔴 1-1. DSPCoreDouble.cpp の該当箇所を読取り・確認
- [ ] 🔴 1-2. DSPCoreDouble.cpp に修正適用
- [ ] 🔴 1-3. DSPCoreFloat.cpp の該当箇所を読取り・確認
- [ ] 🔴 1-4. DSPCoreFloat.cpp に修正適用
- [ ] 🔴 1-5. get_errors でエラーゼロ確認
- [ ] 🔴 1-6. Build_CMakeTools (Debug) 成功確認
- [ ] 🔴 1-7. Strict Atomic Dot-Call Scan 通過確認

---

## Phase 2: BUG-02 ringWrite

**ファイル**: MKLNonUniformConvolver.cpp
**変更**: overflow ブランチの二重更新行削除

- [ ] 🔴 2-1. MKLNonUniformConvolver.cpp 該当箇所を読取り・確認
- [ ] 🔴 2-2. 修正適用（1行削除/コメント化）
- [ ] 🔴 2-3. get_errors でエラーゼロ確認
- [ ] 🔴 2-4. Build_CMakeTools (Debug) 成功確認

---

## Phase 3: BUG-03 DeferredDeletionQueue

**ファイル**: DeferredDeletionQueue.h
**変更**: CAS成功後に `++deqPos` 追加

- [ ] 🔴 3-1. DeferredDeletionQueue.h 該当箇所を読取り・確認
- [ ] 🔴 3-2. 修正適用（++deqPos 追加）
- [ ] 🔴 3-3. get_errors でエラーゼロ確認
- [ ] 🔴 3-4. Build_CMakeTools (Debug) 成功確認
- [ ] 🔴 3-5. Strict Atomic Dot-Call Scan 通過確認

---

## Phase 4: BUG-04 softClipBlockAVX2

**ファイル**: DSPCoreDouble.cpp
**変更**: store前に入力値を退避

- [ ] 🔴 4-1. DSPCoreDouble.cpp 該当箇所を読取り・確認
- [ ] 🔴 4-2. 修正適用（nextPrev 退避追加）
- [ ] 🔴 4-3. get_errors でエラーゼロ確認
- [ ] 🔴 4-4. Build_CMakeTools (Debug) 成功確認
- [ ] 🔴 4-5. Strict Atomic Dot-Call Scan 通過確認

---

## Phase 5: BUG-05 kIdleEpoch

**ファイル**: SafeStateSwapper.h
**変更**: コメント UINT64_MAX → 0 に修正

- [ ] 🔴 5-1. SafeStateSwapper.h 該当箇所を読取り・確認
- [ ] 🔴 5-2. 修正適用（コメント修正）
- [ ] 🔴 5-3. get_errors でエラーゼロ確認

---

## 全体検証

- [ ] 🔴 G-1. Build_CMakeTools (Debug) 成功
- [ ] 🔴 G-2. Build_CMakeTools (Release) 成功
- [ ] 🔴 G-3. Strict Atomic Dot-Call Scan 通過
- [ ] 🔴 G-4. work21 EpochDomain CI Gate 通過
- [ ] 🔴 G-5. CLI Smoke Test 通過
- [ ] 🔴 G-6. CodeGraph Incremental Index 更新
