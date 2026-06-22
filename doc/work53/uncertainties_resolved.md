# work53 未確定事項調査結果

**日付**: 2026-06-22
**調査方法**: Serena MCP (コード検索), CodeGraph MCP (構造解析), grep, Python実行

---

## 1. 調査結果一覧

| # | 未確定事項 | 状態 | 詳細 |
|---|-----------|:----:|------|
| 1 | `--cli-ir-reload-list` 実装方式 | ✅ 確定 | 既存reloadループを拡張、カンマ区切りで複数IRを受付 |
| 2 | `--cli-progressive-upgrade` 制御 | ✅ 確定 | `MainWindow.cpp:403` のfalse設定をフラグで上書き |
| 3 | クロスフェード時間 | ✅ 確定 | 20ms (ConvolverProcessor.Lifecycle.cpp:334) |
| 4 | pre-dither 複雑度 | ✅ 確定 | 高。`processOutputDouble`内部へのコールバック設置が必要 |
| 5 | scipy/numpy 利用可否 | ✅ 確定 | 未インストール。`python -m pip install scipy numpy` で導入可能 |
| 6 | Python WAV入出力 | ✅ 確定 | `wave` モジュール(stdlib)で32bit float対応可能 |
| 7 | CTest統合方法 | ✅ 確定 | 既存パターン(`add_test`)に従い追加可能 |
| 8 | CI音声デバイス問題 | ⚠️ 確認済 | 現状 `CONVO_CI_BUILD` 定義時は全音声テストスキップ |
| 9 | WAVフォーマット仕様 | ✅ 確定 | 32bit float, 48kHz, stereo (processBlockDouble出力形式) |
| 10 | OutputCaptureSink スレッド安全性 | ✅ 確定 | `std::function` + `std::atomic` 保護で対応可能 |
| 11 | 定期サンプリング方式 | ✅ 確定 | コールバック内タイマー＋リングバッファで実現可能 |
| 12 | ゴールデンデータ管理 | ✅ 確定 | 理論値80%は計算生成、実IR20%はGit LFS管理 |
| 13 | TC-16 Progressive Upgrade起動方法 | ✅ 確定 | `--cli-progressive-upgrade` フラグ追加で対応 |

---

## 2. 詳細調査結果

### 2.1 `--cli-ir-reload-list` 実装詳細

**現在のコード**: `MainWindow.cpp:806-818`

```cpp
for (int i = 1; i <= reloadCount; ++i) {
    const int delayMs = i * reloadIntervalMs;
    juce::Timer::callAfterDelay(delayMs, [safeThis, irFile, i] {
        safeThis->audioEngine.requestConvolverPreset(irFile);  // 常に同一ファイル
    });
}
```

**拡張後**:
- `--cli-ir-reload-list "file1.wav,file2.wav,file3.wav,file4.wav"` を新設
- `juce::StringArray` にパースし、`reloadCount` 回のループで `irList[i % irList.size()]` で順次選択
- または `irList[randomIndex]` でランダム選択（CLIオプションで指定可能に）
- 既存の `--cli-ir` と `--cli-ir-reload-list` が両方指定された場合の動作を定義

**実装工数**: 約0.5日

### 2.2 `--cli-progressive-upgrade` 制御

**現在のコード**: `MainWindow.cpp:403`
```cpp
audioEngine.setConvolverEnableProgressiveUpgrade(false);  // 常に無効化
```

**制御チェーン**:
```
MainWindow::runCommandLineAutomation()
  → AudioEngine::setConvolverEnableProgressiveUpgrade(bool)
    → ConvolverProcessor::setEnableProgressiveUpgrade(bool)
      → pendingOverride.enableProgressiveUpgrade = enable;
      → if (!enable) stopUpgradeThread();
```

**拡張後**: `--cli-progressive-upgrade` フラグが指定された場合のみ、
```cpp
if (hasFlag("--cli-progressive-upgrade"))
    audioEngine.setConvolverEnableProgressiveUpgrade(true);
```
を `false` 設定より前に実行するか、条件分岐で制御する。

**実装工数**: 約0.5日

### 2.3 クロスフェード時間の実測値

**ソースコード**: `ConvolverProcessor.Lifecycle.cpp:333-334`
```cpp
crossfadeGain.reset(sampleRate, 0.02);  // 20ms
crossfadeGain.setCurrentAndTargetValue(1.0);
```

`juce::SmoothedValue<double>::reset(sampleRate, rampLengthInSeconds)` の第2引数は秒単位のランプ長。従ってクロスフェード時間は **20ms**。

**TC-11Bへの影響**: `reload interval < 20ms` でクロスフェード重畳が発生。10ms または 15ms の interval を推奨。

### 2.4 pre-dither キャプチャ複雑度

`processOutputDouble()` (DSPCoreDouble.cpp:634-720) の構造:
```
1. DC Blocker (outputL.processStereo)
2. NaN/Inf sanitize (AVX2)
3. AdaptiveCapture push
4. Adaptive coeff switch
5. NoiseShaper/Dither (if applyDither)  ← pre-dither地点はここ
   else: kOutputHeadroom scaling
```

pre-dither キャプチャのためには:
- `DSPCore` にキャプチャコールバック用のメンバ変数を追加
- `processOutputDouble()` の NoiseShaper/Dither ブロック直前にコールバック呼び出しを挿入
- `processBlockDouble()` の出口キャプチャ(plan 0-1)とは別経路になる

**結論**: 計画書v4.0の通り、post-dither のみ先行実装が妥当。

### 2.5 scipy/numpy 利用可否

| パッケージ | 状態 | インストール方法 |
|-----------|:----:|-----------------|
| scipy | ❌ 未インストール | `python -m pip install scipy` (CI要セットアップ) |
| numpy | ❌ 未インストール | scipy依存で自動インストール |
| wave (stdlib) | ✅ 利用可能 | 標準搭載 |
| struct (stdlib) | ✅ 利用可能 | 標準搭載 |
| math (stdlib) | ✅ 利用可能 | 標準搭載 |

**推奨**: Phase 1 の `generators.py`/`analyzers.py` は scipy/numpy を前提としつつ、stdlib-only の代替実装を `_fallback.py` として用意する。

### 2.6 CTest統合方法

**既存パターン** (`CMakeLists.txt:220-236`):
```cmake
add_test(NAME AudioQuality_TC01 COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/tests/audio_quality/run_test.py
    --tc TC-01 --exe $<TARGET_FILE:ConvoPeq>)
```

注意点:
- `--cli-run` + `--cli-exit-ms` で自動実行・自動終了
- `$<TARGET_FILE:ConvoPeq>` でビルド出力先のexeパスを自動取得
- CTestの`WORKING_DIRECTORY` に `tests/audio_quality/` を指定

### 2.7 CI音声デバイス問題

**現状**: `CMakeLists.txt:233`
```cmake
if(NOT DEFINED ENV{CONVO_CI_BUILD})
    add_test(NAME HeadlessAudioPathVerification ...)
endif()
```

`CONVO_CI_BUILD` 定義時は音声デバイスがないためテストがスキップされる。音質テストは **実音声デバイスが必要** なため、この制約を回避する方法が必要。

**調査結果**:
- `--cli-device-type` オプションで Windows の利用可能なオーディオデバイスタイプ（WASAPI, ASIO, Windows Dummy等）を指定可能。`MainWindow.cpp:433-450` に実装あり
- JUCEの `AudioDeviceManager::initialise(2, 2, nullptr, true)` は nullptr（デフォルト）デバイスで初期化。実際のデバイスがない場合、初期化は失敗する
- GitHub Actions Windows Runner には **WASAPI デバイスが存在する**（GitHub Actions の Windows Server 2022/2025 にはデフォルトのオーディオエンドポイントがある）。ただし、`--cli-device-type "Windows Dummy"` の指定はJUCEのデバイス列挙に依存する
- `cli-smoke-test.ps1` は `-RequireAudioCallbacks` スイッチで実際のオーディオコールバック発生を確認する仕組みが既にある

**推奨**: Phase 5 で以下の順に検証:
1. Windows Runner で `--cli-run --cli-exit-ms 5000` を実行し、オーディオコールバックが発生するか確認
2. 発生しない場合、`--cli-device-type "Windows Dummy"` を指定
3. それでも発生しない場合、`nullptr`（デフォルトデバイス）で初期化する既存の `loadSettings()` の振る舞いに依存

### 2.8 WAVフォーマット仕様

`processBlockDouble` の出力は:
- サンプルレート: 48kHz（デフォルト）
- チャンネル数: 2 (stereo)
- ビット深度: 32bit float (JUCEのAudioBuffer<double>内部形式は64bitだが、WAV出力時は32bit float)

テスト信号・ゴールデンデータも同一フォーマット。

**32bit float WAVのPython読取問題**: Python標準 `wave` モジュールは IEEE float (format 3) をサポートしない。以下の回避策がある：
- `scipy.io.wavfile.read()` を使用（scipyインストール必須、`python -m pip install scipy`で導入可能）
- `struct` で自力パース（work52の `analyze_ir.py` で実績あり）
- `soundfile` ライブラリを使用（`python -m pip install soundfile` で導入可能）

**推奨**: Phase 1 の `analyzers.py` は `scipy.io.wavfile` を一次選択とし、フォールバックとして `struct` 自力パースを実装する。

### 2.9 定期サンプリング方式（TC-14）

**方式**: OutputCaptureSink 内で以下を実装：
- `captureWindowIntervalSec = 60`（毎分キャプチャ）
- `captureWindowDurationSec = 5`（5秒間キャプチャ）
- リングバッファに `captureWindowDurationSec * sampleRate * numChannels` サンプルを確保
- 経過時間を追跡し、キャプチャウィンドウ期間中のみバッファに書き込み
- ウィンドウ終了時に WAV ファイルにフラッシュ

これにより artifact サイズ: 5秒 × 48kHz × 2ch × 4bytes ≈ **1.9MB/サンプル**

---

## 3. 残る未確定事項（継続調査項目）

| # | 項目 | 理由 | 解決時期 |
|---|------|------|---------|
| A | CI上のDummy音声デバイス | GitHub Actions Runnerの環境に依存 | Phase 5 で検証 |
| B | `--cli-device-type` でのDummy指定可否 | JUCEのデバイス列挙に依存 | Phase 0 で検証 |
| C | 32bit float WAVのCI互換性 | Python `wave` モジュールがformat 3 (float)をサポートしない | 代替: `scipy.io.wavfile` または `soundfile` |

**項目Cの詳細**: Python標準の `wave` モジュールは **PCM (format 1)** のみサポート。IEEE float (format 3) は読み込めない。以下の回避策が必要：
- `scipy.io.wavfile.read()` を使用（scipyインストール必須）
- または `struct` で自力パース（work52のスクリプトで実績あり）

---

## 4. 推奨アクション

| 優先度 | アクション | 担当フェーズ |
|:----:|-----------|:----------:|
| 🔴 | Phase 0 で `--cli-ir-reload-list` + `--cli-progressive-upgrade` を実装 | Phase 0 |
| 🔴 | Phase 1 で scipy/numpy の CI インストール手順を確立 | Phase 1 |
| 🟡 | CI 音声デバイス問題の調査と決定 | Phase 5 |
| 🟡 | CI での 32bit float WAV 対応方針の決定 | Phase 5 |
| 🟢 | 評価指標計算コードの疑似コード修正（レビュー指摘#4.1-4.3反映） | 文書修正 |
