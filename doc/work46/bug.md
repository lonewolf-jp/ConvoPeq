# ConvoPeq バグ修正レポート（実装可能な最終版）

> **修正状況（2026-06-18）**: Bug #1 と #3 は以下の通り修正済み。詳細は `doc/work47/resolution_report.md` 参照。

本レポートは、`ConvoPeq.md` のソースコード解析に基づき、**修正が必須と判断された 5 件のバグ**について、**即座に実装可能な修正コード**を提供するものです。
各バグには **ファイル名・関数名・該当行（近似）** を明示し、**Before/After の差分** を示します。

---

## 🔥 優先度：最高（即時修正必須）

---

### バグ #1: AVX2 デシメーションフィルタのメモリ範囲外読み出し

| 項目 | 内容 |
|:---|:---|
| **ファイル** | `src/CustomInputOversampler.cpp` |
| **関数** | `CustomInputOversampler::prepareStage` |
| **影響** | クラッシュ / 意図しないノイズ発生（未定義動作） |
| **再現条件** | オーバーサンプリング率 2x/4x/8x で、特定のブロックサイズ・フィルタタップ数で発生 |
| **ステータス** | ✅ **2026-06-18 修正済み** |

**🔍 根本原因**
`loadStride2` が `ptr[-6]` までアクセスするにもかかわらず、`historyDownKeep` の計算でこのオフセットが考慮されていなかった。
その結果、`globalMinConvIdx` が 0 になるケースで、バッファ先頭より前（index -6 まで）のメモリを読み出す可能性があった。

**📝 修正コード**

```cpp
// Before (prepareStage)
stage.historyDownKeep = juce::jmax(stage.centerTap, stage.convParity + ((stage.convCount - 1) << 1));

// After
stage.historyDownKeep = juce::jmax(stage.centerTap, stage.convParity + ((stage.convCount - 1) << 1) + 6);
```

> **備考**: 当初提案の `globalMinConvIdx` に `-6` を追加する方法では Stage 0 の AVX2 パスが完全無効化されるため、`historyDownKeep` にマージンを追加する方法を採用した。

**🔍 根本原因**
`decimateStage` 内の `globalMinConvIdx` 計算において、`loadStride2` がアクセスする最大負方向オフセット（`-6`）が考慮されていません。バッファ先頭（index 0）より前のメモリを読み込む可能性があります。

**📝 修正コード**

```cpp
// Before (L162-163 付近)
const int globalMinConvIdx = keep - stage.convParity - ((stage.convCount - 1) << 1);
const int globalMaxConvIdx = baseMax - stage.convParity;

// After
const int globalMinConvIdx = keep - stage.convParity - ((stage.convCount - 1) << 1) - 6; // ← -6 を追加
const int globalMaxConvIdx = baseMax - stage.convParity;
```

---

### バグ #2: ノイズシェーパー出力レベルの異常減衰（-1dB ロス）

| 項目 | 内容 |
|:---|:---|
| **ファイル** | `src/FixedNoiseShaper.h` / `src/Fixed15TapNoiseShaper.h` |
| **関数** | `processStereoBlock` |
| **影響** | 出力レベルが意図せず約 -1dB 減衰（音質劣化） |
| **再現条件** | ディザリング有効時（bitDepth > 0）に常時発生 |

**🔍 根本原因**
`headroom`（0.891）を乗算した値を量子化し、そのまま出力している。量子化後に元の振幅に復元する処理が欠落している。

**📝 修正コード**

```cpp
// Before (Fixed15TapNoiseShaper.h L145-148 付近)
for (int i = 0; i < numSamples; ++i) {
    double error = 0.0;
    dataL[i] = processSample(dataL[i] * headroom, 0, error);
    // ...
}
if (dataR != nullptr) {
    for (int i = 0; i < numSamples; ++i) {
        double error = 0.0;
        dataR[i] = processSample(dataR[i] * headroom, 1, error);
    }
}

// After
for (int i = 0; i < numSamples; ++i) {
    double error = 0.0;
    dataL[i] = processSample(dataL[i] * headroom, 0, error) / headroom; // ← / headroom を追加
    // ...
}
if (dataR != nullptr) {
    for (int i = 0; i < numSamples; ++i) {
        double error = 0.0;
        dataR[i] = processSample(dataR[i] * headroom, 1, error) / headroom; // ← / headroom を追加
    }
}
```

> **注意**: `FixedNoiseShaper.h` の `processStereoBlock` も同一の修正が必要です。

---

### バグ #3: CMake ASan 設定位置誤り（ビルド構成エラー）

| 項目 | 内容 |
|:---|:---|
| **ファイル** | `CMakeLists.txt` |
| **影響** | `-DENABLE_ASAN=ON` 時に CMake 構成が失敗し、ビルド開始不能 |
| **再現条件** | ASan 有効化時（手動で `-DENABLE_ASAN=ON` を指定した場合） |
| **ステータス** | ✅ **2026-06-18 修正済み** |

**🔍 根本原因**
`juce_add_gui_app(ConvoPeq ...)` でターゲットが定義される前に `target_compile_options(ConvoPeq ...)` が呼ばれている。

**📝 修正内容**

ASan ブロックをファイル先頭から削除し、コンパイラ固有設定セクションの直後（`juce_add_gui_app` より後）に移動。

```cmake
# CMakeLists.txt 冒頭: ASan ブロックを削除（ターゲット未定義エラー防止）
#============================================================================
# CMakeLists.txt  ── v0.5.0 (JUCE 8.0.12 / VS Code + MSVC + icx + Windows 11)
# ...

# ターゲット定義後、コンパイラ固有設定直後に ASan ブロックを配置
#------------------------------------------------------------
# AddressSanitizer (ASan) オプション（ターゲット定義後でなければならない）
#------------------------------------------------------------
option(ENABLE_ASAN "Enable AddressSanitizer (Debug only)" OFF)
if(ENABLE_ASAN)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE /fsanitize=address)
        target_link_options(ConvoPeq PRIVATE /fsanitize=address)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE -fsanitize=address)
        target_link_options(ConvoPeq PRIVATE -fsanitize=address)
    endif()
endif()
```

        target_compile_options(ConvoPeq PRIVATE /fsanitize=address)
        target_link_options(ConvoPeq PRIVATE /fsanitize=address)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE -fsanitize=address)
        target_link_options(ConvoPeq PRIVATE -fsanitize=address)
    endif()
endif()

```

---

### バグ #4: `/MP1` によるシングルスレッドビルド強制（ビルド時間劣化）

| 項目 | 内容 |
|:---|:---|
| **ファイル** | `CMakeLists.txt` |
| **影響** | ビルド時間が著しく増大（開発効率低下） |
| **再現条件** | MSVC ビルド時に常時発生 |

**🔍 根本原因**
`/MP1` は並列コンパイル数を 1 に固定する。JUCE プロジェクトの規模ではビルド時間が数倍に増加する。

**📝 修正コード**

```cmake
# Before (MSVC 固有設定セクション)
target_compile_options(ConvoPeq PRIVATE
    /utf-8
    /W4
    /MP1        # ← 問題のフラグ
    /EHsc
    # ...
)

# After
target_compile_options(ConvoPeq PRIVATE
    /utf-8
    /W4
    /MP         # ← 自動並列化（コア数に応じて調整）
    /EHsc
    # ...
)
```

> **補足**: メモリ使用量を厳密に制御したい場合は `/MP4` など明示的な値も検討可能ですが、開発環境では `/MP`（自動）が最も実用的です。

---

## ⚠️ 優先度：高（早めの修正推奨）

---

### バグ #5: `rebuildAllIRsSynchronous` の Nullptr リスク

| 項目 | 内容 |
|:---|:---|
| **ファイル** | `src/convolver/ConvolverProcessor.Rebuild.cpp` |
| **関数** | `ConvolverProcessor::rebuildAllIRsSynchronous` |
| **影響** | 特定の呼び出し経路でクラッシュの可能性 |
| **再現条件** | `rebuildJob` が `nullptr` の状態で同関数が呼ばれた場合 |

**🔍 根本原因**
関数冒頭で `if (rebuildJob) rebuildJob->reset();` と削除処理は行うが、`nullptr` の場合に新規生成するロジックが欠落している。

**📝 修正コード**

```cpp
// Before (ConvolverProcessor.Rebuild.cpp L114-130 付近)
void ConvolverProcessor::rebuildAllIRsSynchronous(std::function<bool()> shouldCancel) {
    [[maybe_unused]] auto stageToString = ...;

    const IRState* state = acquireIRState();
    if (state && state->ir && state->ir->getNumSamples() > 0 && state->sampleRate > 0.0) {
        if (shouldCancel && shouldCancel()) {
            if (rebuildJob)
                rebuildJob->reset();  // ← ここでは nullptr チェック済みだが...
            releaseIRState(state);
            return;
        }

        auto runRebuildPath = [&]() {
            // ... リビルド処理 ...
        };

        runRebuildPath();
        // ...
    }

    if (rebuildJob)
        rebuildJob->reset();  // ← ここでも nullptr チェック済み

    releaseIRState(state);
}

// After
void ConvolverProcessor::rebuildAllIRsSynchronous(std::function<bool()> shouldCancel) {
    // ★ 追加: インスタンスがなければ新規生成
    if (!rebuildJob) {
        rebuildJob = std::make_unique<IncrementalRebuildJob>();
    }

    [[maybe_unused]] auto stageToString = ...;

    const IRState* state = acquireIRState();
    if (state && state->ir && state->ir->getNumSamples() > 0 && state->sampleRate > 0.0) {
        if (shouldCancel && shouldCancel()) {
            rebuildJob->reset();  // ← 安全に呼び出せる
            releaseIRState(state);
            return;
        }

        auto runRebuildPath = [&]() {
            // ... リビルド処理 ...
        };

        runRebuildPath();
    }

    rebuildJob->reset();  // ← 安全に呼び出せる
    releaseIRState(state);
}
```

---

## 修正適用後の動作確認項目

| バグ ID | 確認テスト |
|:---|:---|
| #1 | オーバーサンプリング 2x/4x/8x で長時間動作させ、クラッシュやノイズが発生しないことを確認 |
| #2 | ノイズシェーパー有効時（16bit/24bit）の出力レベルが、バイパス時と一致することを RMS 測定で確認 |
| #3 | `build.bat Debug clean` および ASan 有効ビルドが正常に完了することを確認 |
| #4 | ビルド時間が `/MP1` 時と比較して短縮されていることを確認（実測） |
| #5 | `rebuildAllIRsSynchronous` を初回呼び出し時にクラッシュしないことを確認 |

---

## 補足：検証済みの「非バグ」項目

前回ご指摘いただいた以下の項目は、再検証の結果 **現状のコードで問題ありません**。

| 項目 | 判定 | 理由 |
|:---|:---|:---|
| ① `outputMakeupDb` 読み込み固定化 | ❌ 非バグ | `sanitizeFiniteClamped` の min=0.0, max=12.0 は正しい |
| ③ `LockFreeAudioRingBuffer` の精度低下 | ❌ 非バグ | アナライザー描画用バッファであり音声経路外 |
| ⑧ `DeferredDeletionQueue` スキャンロジック | ❌ 非バグ | エポックは必ず進行するため無限ループしない |

---

ご不明な点や、追加の確認事項がございましたら、どうぞお知らせください。
本修正案は、コードベースに直接適用可能な状態でご用意しております。
