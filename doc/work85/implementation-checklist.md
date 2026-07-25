# ConvoPeq バグ改修 実装チェックリスト

作成日: 2026-07-25
ベース文書: `doc/work85/remediation-plan.md`

---

凡例: ✅ 完了 | 🔄 作業中 | ⬜ 未着手 | ❌ 問題あり

---

## Phase 1: P1-High（即時対応）— Week 1

### T2. B01: build.bat SHIFT 解析方式
**優先度: P1-High** | **種別: 🔧** | **工数: 小**

- [ ] 1. `for %%A` ループを `SHIFT` 解析方式に置き換え
- [ ] 2. `=ON` 強制付与を削除
- [ ] 3. ヘッダーコメントの `=ON` 自動付与に関する記述を修正
- [ ] 4. テスト: `build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` の動作確認
- [ ] 5. テスト: `build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の従来動作維持確認
- [ ] 6. テスト: `build.bat Release icx clean` の動作確認

### T3. G03: FlagResetter 修正
**優先度: P1-High** | **種別: 🛠** | **工数: 小**

- [ ] 1. `ConvolverProcessor.LoaderThread.cpp` の確認
- [ ] 2. `if (!success && !t.threadShouldExit())` → `if (!success)` に変更
- [ ] 3. `callAsync` 失敗時のコメント更新
- [ ] 4. ビルド確認
- [ ] 5. テスト: IRロード中に `signalThreadShouldExit()` → `isLoadingIR()==false` 確認

### T1. A01: `_mm256_zeroupper()` 欠如
**優先度: P1-High** | **種別: 🛠⚡** | **工数: 中（16ファイル）**

#### Step 1: Compiler Option（icx）
- [ ] 1. CMakeLists.txt に icx `-mvzeroupper` を追加（CXX限定＋AVX2有効時）
- [ ] 2. icx ビルド確認 + `objdump -d` で ABI境界への vzeroupper 挿入確認

#### Step 2: MSVC 明示的挿入
- [ ] 3. `DSPCoreDouble.cpp` — AVX→legacy SSE 境界に `_mm256_zeroupper()` 追加
- [ ] 4. `DSPCoreFloat.cpp` — 同上
- [ ] 5. `DSPCoreIO.cpp` — 同上
- [ ] 6. `EQProcessor.Processing.cpp` — 同上
- [ ] 7. `MKLNonUniformConvolver.cpp` — 同上
- [ ] 8. `CustomInputOversampler.cpp` — 同上
- [ ] 9. `TruePeakDetector.cpp` — 同上
- [ ] 10. `LoudnessMeter.cpp` — 同上
- [ ] 11. `ConvolverProcessor.Runtime.cpp` — 同上
- [ ] 12. `ConvolverProcessor.LoaderThread.cpp` — 同上
- [ ] 13. `AudioEngine.EQResponse.cpp` — 同上
- [ ] 14. `SpectrumAnalyzerComponent.cpp` — 同上
- [ ] 15. `LatticeNoiseShaper.h` — 同上
- [ ] 16. `InputBitDepthTransform.h` — 同上
- [ ] 17. `DspNumericPolicy.h` — 同上（`__m256d` 戻り値関数は対象外）
- [ ] 18. `dsp/math/FastTanhApprox.h` — 同上（`__m256d` 戻り値関数は対象外）
- [ ] 19. MSVC Release ビルド確認 + `dumpbin /disasm` で確認

### Phase 1 統合テスト
- [ ] 1. MSVC Release/Debug ビルド
- [ ] 2. icx Release/Debug ビルド
- [ ] 3. CTest 全テスト通過
- [ ] 4. Clang-Tidy 有効時警告ゼロ

---

## Phase 2: P2-Medium（次期対応）— Week 2

### T4. A02: NoiseShaperLearner スタック→ヒープ
**優先度: P2-Medium** | **種別: 🛠** | **工数: 小**

- [ ] 1. `NoiseShaperLearner.cpp` の `buildTrainingSegments()` を確認
- [ ] 2. `recentLeft[34816]` → `makeAlignedArray<double>(kRecentSampleRequest)`
- [ ] 3. `recentRight[34816]` → `makeAlignedArray<double>(kRecentSampleRequest)`
- [ ] 4. `bad_alloc` 時の catch とログ出力を追加
- [ ] 5. ビルド確認

### T5. B02: Clang-Tidy 引数フォーマット修正
**優先度: P2-Medium** | **種別: 🔧** | **工数: 小**

- [ ] 1. CMakeLists.txt の Clang-Tidy 設定を確認
- [ ] 2. マルチラインクォートを個別引数に分割
- [ ] 3. ビルド確認

### T6. B03: icx + ASan CRT 競合修正
**優先度: P2-Medium** | **種別: 🔧** | **工数: 小**

- [ ] 1. CMakeLists.txt の `ENABLE_ASAN` ブロックを確認
- [ ] 2. IntelLLVM 用の CRT 切替（`/MT` → `/MD`）を追加
- [ ] 3. `compile_commands.json` で `/MT` が残っていないことを確認
- [ ] 4. icx + ENABLE_ASAN=ON のビルド確認

### T8. G07: makeEngineRuntimeState フォールバック
**優先度: P2-Medium** | **種別: 🛠** | **工数: 小**

- [ ] 1. `AudioEngine.h` の `makeEngineRuntimeState()` を確認
- [ ] 2. `runtimeWorld == nullptr` 時のコメントを追記
- [ ] 3. ビルド確認

### Phase 2 統合テスト
- [ ] 1. MSVC/icx ビルド
- [ ] 2. CTest 全テスト通過

---

## Phase 3: P3-Low（予防・品質）— Week 3

### T10. A03: push() 戻り値改善
**優先度: P3-Low** | **種別: 🛠** | **工数: 小**

- [ ] 1. `LockFreeAudioRingBuffer::push()` の戻り値を `int` に変更
- [ ] 2. 全呼出側で戻り値を処理（ログ/XRUN/Telemetry）
- [ ] 3. 最後に `[[nodiscard]]` を追加
- [ ] 4. ビルド確認

### T11. A04: AllpassDesigner デッドコード修正
**優先度: P3-Low** | **種別: 🛠** | **工数: 小**

- [ ] 1. `AllpassDesigner.cpp` の `std::min(0.45*sampleRate, 0.499*sampleRate)` を確認
- [ ] 2. `kMaxAllpassFrequencyHz = 20000.0` を使用した式に修正
- [ ] 3. ビルド確認

### T13. B04: icx テストターゲットリンクパス
**優先度: P3-Low** | **種別: 🔧** | **工数: 小**

- [ ] 1. CMakeLists.txt で `_INTEL_COMPILER_ROOT` のテストターゲット反映を追加
- [ ] 2. ビルド確認

### Phase 3 統合テスト
- [ ] 1. MSVC/icx ビルド
- [ ] 2. CTest 全テスト通過

---

## Phase 4: P4-Design（設計検討事項）— Week 4

### T7. G02: retireImmediateDuringShutdown
**優先度: P4-Design** | **種別: 🛠** | **工数: 小**

- [ ] 1. `EQProcessor.Core.cpp` で `(void)retireEQStateDeferred(oldState)` を確認
- [ ] 2. `retireImmediateDuringShutdown()` を実装（Runtime停止完了後専用）
- [ ] 3. `enqueueDeferredDeleteWithFallback` から同期deleteを削除
- [ ] 4. 呼出元を2パターン（通常時/Shutdown時）に分岐
- [ ] 5. jassert による防御追加
- [ ] 6. ビルド確認

### T9. A05: makeAlignedArrayZero 分離
**優先度: P4-Design** | **種別: 🛠** | **工数: 小**

- [ ] 1. `AlignedAllocation.h` に `makeAlignedArrayZero()` を追加
- [ ] 2. T4 (A02) の呼出側を `makeAlignedArrayZero` に変更
- [ ] 3. ビルド確認

### T12. A06: alignas(64) 追加
**優先度: P4-Design** | **種別: 🛠** | **工数: 小**

- [ ] 1. `NoiseShaperLearner.h` の `AudioSegment` に `alignas(64)` 追加
- [ ] 2. `static_assert(alignof(AudioSegment) == 64)` 追加
- [ ] 3. `static_assert(sizeof(AudioSegment) % 64 == 0)` 追加
- [ ] 4. ビルド確認

### Phase 4 統合テスト＋全Phaseリグレッション
- [ ] 1. MSVC/icx Release/Debug ビルド
- [ ] 2. icx + ENABLE_ASAN=ON ビルド
- [ ] 3. CTest 全テスト通過
- [ ] 4. Clang-Tidy 有効時警告ゼロ

---

## 未確定・未決定事項（別設計検討）

### T14. C05: IRConverter リサンプルフォールバック
**ステータス: 別設計検討（時期未定）**

- [ ] SRC ライブラリ選定（libsamplerate / libsoxr / IPP）
- [ ] 設計判断後、convertFile に実装
