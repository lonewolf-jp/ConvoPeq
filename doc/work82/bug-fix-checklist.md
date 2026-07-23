# bug.md 実装チェックリスト

**作成日**: 2026-07-23
**ベース**: `doc\work82\bug-fix-plan.md` v6

---

## P0: 即時修正（2件）

- [x] **CVMD-005**: 完全 bypass 時の UI 表示誤り
  - [x] `MainWindow.cpp:1261-1266` に `orderModeBox.addItem("Bypass", 5)` を追加
  - [x] `MainWindow.cpp:1190-1199` の表示ロジックで `eqBypassed && convBypassed` 時に `modeId = 5` を設定
  - [x] `MainWindow.cpp:1350-1373` の `orderModeBoxChanged()` を `switch` 文に変更し、`case 5` で bypass フラグ設定
  - [x] プロジェクト全体で `getSelectedId()` の参照箇所を検索し、ID=5 追加による更新漏れがないことを確認

- [x] **H-5**: CpuFeatureCheck の FMA ビット位置誤り + XSAVE チェック
  - [x] `CpuFeatureCheck.cpp:38-60` の Method 2 を書き直し
  - [x] Step 0: `CPUID(0)` で `maxLeaf >= 7` を確認
  - [x] Step 1: `CPUID(1)` で OSXSAVE + AVX + FMA を確認
  - [x] Step 2: `CPUID(7)` で AVX2 を確認
  - [x] Step 3: `XGETBV` で YMM 保存を確認
  - [x] `#else` ブランチを `return false` に変更

---

## P1: 次回リリース前（3件）

- [x] **CVMD-007**: cachedTailLength のスレッド安全性
  - [x] `AudioEngineProcessor.h:45` を `std::atomic<double>` に変更
  - [x] `AudioEngineProcessor.cpp:24-26` の読み込みを `consumeAtomic(..., relaxed)` に変更
  - [x] `AudioEngineProcessor.cpp:40-50` の書き込みを `publishAtomic(..., relaxed)` に変更

- [x] **M-3**: DeferredRetireFallbackQueue の totalPushCount_ 未実装
  - [x] `ISRRetireOverflowRing.h` の `push()` 内で `totalPushCount_.fetch_add(1, relaxed)` を追加

- [x] **Bug5**: prepareSingleStage の noexcept 見直し
  - [x] `CustomInputOversampler.cpp:380` 内で `makeAlignedArray` を `makeAlignedArray_nothrow` に変更
  - [x] `prepareStage()` を `bool` 戻り値に変更し、内部で `makeAlignedArray_nothrow` を使用
  - [x] `prepareSingleStage` で `prepareStage` の戻り値を確認し、失敗時は `return false`

---

## P2: リファクタリング（3件）

- [x] **Bug4**: lastError のスレッド安全性
  - [x] `ConvolverProcessor.h` に `lastErrorMutex`, `setLastError()`, `clearLastError()` を追加
  - [x] `LoadPipeline.cpp:47` の `lastError.clear()` を `clearLastError()` に変更
  - [x] `LoadPipeline.cpp:538` の直接代入を `setLastError()` に変更
  - [x] `StateAndUI.cpp:378` の直接代入を `setLastError()` に変更
  - [x] `Rebuild.cpp:133` の `lastError.clear()` は `ConvolverProcessor::lastError` を参照（`IncrementalRebuildJob::lastError` とは別物）のためそのまま（mutex 不要）

- [x] **Bug12**: progressCallback のマーシャリング
  - [x] `AllpassDesigner.cpp:399-401` で `callAsync` に変更
  - [ ] 呼び出し側で `SafePointer` を使用する API 契約をドキュメント化

- [x] **Bug14**: CacheManager の一時ファイル残存
  - [x] `CacheManager.cpp` の `save()` で `flush()` の戻り値を確認
  - [x] flush 失敗時に `temp.deleteFile()` を呼び出す

---

## 実装ログ

| 日時 | 項目 | 状態 | 備考 |
|------|------|------|------|
| 2026-07-23 | CVMD-005 | ✅ 完了 | ComboBox ID=5 追加、表示ロジック修正、switch文変更 |
| 2026-07-23 | H-5 | ✅ 完了 | Method 2 書き直し（MaxLeaf+OSXSAVE+AVX+FMA+AVX2+XGETBV） |
| 2026-07-23 | CVMD-007 | ✅ 完了 | cachedTailLength を atomic に変更（relaxed） |
| 2026-07-23 | M-3 | ✅ 完了 | totalPushCount fetch_add 追加 |
| 2026-07-23 | Bug5 | ✅ 完了 | makeAlignedArray_nothrow に変更 |
| 2026-07-23 | Bug4 | ✅ 完了 | lastError に mutex 保護、setLastError/clearLastError 追加 |
| 2026-07-23 | Bug12 | ✅ 完了 | callAsync で Message Thread にマーシャリング |
| 2026-07-23 | Bug14 | ✅ 完了 | flush 失敗検査追加 |
