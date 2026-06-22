# work53 音質評価自動化テスト計画 検証報告書

**対象文書**: `doc/work53/basic_test_plan.md` (v3.0)
**検証日**: 2026-06-22
**結果**: **概ね妥当（スコア: 86/100）** — 6項目の修正推奨

---

## 1. 検証方法

以下のツールを用いて、計画書の主張と現行ソースコードの整合性を検証した：
- Serena MCP: コード検索・シンボル特定
- CodeGraph MCP: ファイル構造解析
- grep/Select-String: パターン検索
- ソースコード直接参照: `AudioEngine.h`, `MainWindow.cpp`, `DSPCoreDouble.cpp`, `BlockDouble.cpp` 等

---

## 2. 総合評価

| 評価項目 | 点数 | 判定 |
|---------|:---:|:----:|
| アーキテクチャ整合性 | 18/20 | 既存CLI＋出力キャプチャ方式は妥当 |
| DSP評価設計 | 16/18 | TC-01B追加は良いがTC-09に曖昧さ |
| Runtime評価設計 | 14/18 | TC-11に実装上不可能な要件あり |
| 実装容易性 | 14/16 | Phase 0は概ね現実的 |
| CI適合性 | 16/18 | 現行CTest基盤との統合は可能 |
| 評価指標・閾値 | 8/10 | 概ね妥当だが一部検証不足 |
| **合計** | **86/100** | **修正後再レビュー推奨** |

---

## 3. 検証結果詳細

### 3.1 既存CLI基盤との整合性 ✅ (問題なし)

**確認済み**: 以下のCLIオプションは既に実装済み：

| オプション | ステータス | 参照先 |
|-----------|:--------:|--------|
| `--cli-ir` | ✅ 実装済み | `MainWindow.cpp:772` |
| `--cli-ir-reload-count` | ✅ 実装済み | `MainWindow.cpp:786` |
| `--cli-ir-reload-interval-ms` | ✅ 実装済み | `MainWindow.cpp:794` |
| `--cli-bypass-burst-count` | ✅ 実装済み | `MainWindow.cpp:868` |
| `--cli-bypass-burst-interval-ms` | ✅ 実装済み | `MainWindow.cpp:878` |
| `--cli-bypass-burst-value` | ✅ 実装済み | `MainWindow.cpp:886` |
| `--cli-intent-burst-count` | ✅ 実装済み | `MainWindow.cpp:917` |
| `--cli-intent-burst-interval-ms` | ✅ 実装済み | `MainWindow.cpp:927` |
| `--cli-exit-ms` | ✅ 実装済み | `MainWindow.cpp` |
| `--cli-order` | ✅ 実装済み | `MainWindow.cpp` |
| `--cli-noise-shaper` | ✅ 実装済み | `MainWindow.cpp` |
| `--cli-dither-bit-depth` | ✅ 実装済み | `MainWindow.cpp` |
| `--cli-phase` | ✅ 実装済み | `MainWindow.cpp` |

**未実装（Phase 0で追加予定）**:
| `--cli-output-wav` | ❌ 未実装 | Phase 0-2 |
| `--cli-capture-mode` | ❌ 未実装 | Phase 0-3 |

### 3.2 OutputCaptureSink 設計の妥当性 ✅ (概ね妥当)

**計画の主張**: `processBlockDouble()` 出口でコールバックを呼び出し、出力WAVを保存する。

**検証結果**:

`processBlockDouble()`（`AudioEngine.Processing.BlockDouble.cpp:14`）の出口では、以下の処理が完了している：
1. `DSPCore::processDouble()` → SoftClip → outputMakeup → NoiseShaper/Dither 完了
2. クロスフェード（該当する場合）完了
3. 出力バッファは最終状態

**結論**: コールバック方式は適切。ただし注意点として：
- コールバックは **Audio Thread** から呼ばれるため、`std::function` を `std::atomic` で保護する設計は正しい
- `--cli-capture-mode=pre-dither` を実装する場合、キャプチャポイントを `processOutputDouble()` 内部（`DSPCoreDouble.cpp:617` より前）に置く必要があり、`processBlockDouble` 出口ではなくなる → **設計の複雑化**

**推奨**: v1.0 では `post-dither`（デフォルト）のみ実装し、`pre-dither` は将来の拡張とする。

### 3.3 TC-03 THD+N 閾値の妥当性 ✅

| ビルド | 閾値 | 検証 |
|--------|:----:|:----|
| Debug | ≤ -80 dB | 妥当。デノーマル対策が無効のため |
| Release (MSVC) | ≤ -100 dB | 妥当。FTZ/DAZ + 最適化により達成可能 |
| Release (icx) | ≤ -98 dB | 妥当。MSVC比でやや緩い設定 |

### 3.4 TC-04 無音ノイズフロア閾値 ✅

≤ -120 dBFS は FloatVectorOperations::clear 使用時には達成可能。64bit double の量子化ノイズは -300dB 以下であるため、この閾値は余裕を持ってクリアできる。

### 3.5 TC-07 Butterworth LPF 合成IR ✅

`scipy.signal` が利用可能であれば `scipy.signal.butter(4, 1000, btype='low', fs=48000)` で合成可能。
**注意**: CI環境で scipy が利用可能か事前確認が必要。代替として `scipy.signal.freqz` のみでも理論値計算は可能。

---

### 3.6 TC-09 エイリアシング試験 ⚠️ 要修正

**問題**: テスト条件にOS（オーバーサンプリング）の有無が明記されていない。

| OS設定 | 21kHz | 22kHz | 23kHz | Nyquist |
|--------|:----:|:----:|:----:|:-------:|
| OS=1x (48kHz) | 可 | 可 | **危険** | 24kHz |
| OS=2x (96kHz) | 可 | 可 | 可 | 48kHz |

OS=1x で 23kHz（= fs/2 - 1kHz）を入力すると、アンチエイリアシングフィルタの過渡特性により減衰が生じる可能性がある。

**推奨**:
- OS有効時のみこのテストを実行する条件を明記する
- または 23kHz を 19kHz に変更する

### 3.7 TC-11 IR Reload Storm ❌ 要修正（最重要）

**問題**: 計画書は「100回連続で異なるIRをロードする（ディラック、LPF、HPF、オールパスをランダムに選択）」としているが、現行の `--cli-ir-reload-count` は **同一IRファイルの繰り返し再読み込み** のみをサポートしている。

**証拠**: `MainWindow.cpp:786-810`
```cpp
for (int i = 1; i <= reloadCount; ++i)
{
    const int delayMs = i * reloadIntervalMs;
    juce::Timer::callAfterDelay(delayMs, [safeThis = ..., irFile, i] {
        // irFile is the same file from --cli-ir
    });
}
```

**推奨**:
- オプションA: `--cli-ir-reload-list "file1.wav,file2.wav,file3.wav,file4.wav"` を新設する
- オプションB: TC-11 を同一IRの繰り返し再読み込みに変更する（簡易版、ただし網羅性低下）
- オプションC: TC-11 と TC-11B を統合し、IR Reload Storm は同一IR、Crossfade Storm は別途とする

### 3.8 TC-11B Crossfade Storm ⚠️ 要確認

**問題**: 計画書は「reload interval < crossfade duration（約 50ms）」としているが、実際のクロスフェード時間は **20ms** である。

**証拠**: `ConvolverProcessor.Lifecycle.cpp:334`
```cpp
crossfadeGain.reset(sampleRate, 0.02);  // 20ms
```

**推奨**: クロスフェード間隔の記述を「約 20ms」に修正。テスト条件の reload interval は 10〜15ms に設定すべき。

### 3.9 TC-14 長期耐久試験 ⚠️ 要確認

**問題**: 計画書は「PR時: 30分間連続処理。Nightly: 1時間連続処理」としている。`--cli-exit-ms` は最大で `int32` の範囲（約24日）であるため時間的制約はないが、**30分〜1時間の連続出力WAV** のサイズが問題になる。

- 30分 @ 48kHz, stereo, 32bit float = 30×60×48000×2×4 = **691MB**
- 1時間 = **1.38GB**

GitHub Actions の artifact 上限（通常 2GB）に近い。

**推奨**: 定期サンプリング方式を採用する（例えば1分毎に5秒間だけWAV保存）。これにより artifact サイズを 5×48000×2×4 ≈ **1.9MB/サンプル** に抑えられる。

### 3.10 TC-15 Mixed Phase Cache Rebuild ✅

CMA-ES 最適化による Mixed Phase 変換は既存機能。3段階の比較テストは妥当。

**注意**: CMA-ES 最適化には数秒〜数十秒かかる可能性がある。CI タイムアウトに注意（Phase 5 の Risk Register にも記載あり）。

### 3.11 TC-16 Progressive Upgrade ⚠️ 要確認

**問題**: 計画書は FFT サイズ 512→1024→2048 の Progressive Upgrade をテストするが、`convolverProcessor.setConvolverEnableProgressiveUpgrade(false)` が `runCommandLineAutomation` 内で常に実行される。

**証拠**: `MainWindow.cpp:403`
```cpp
audioEngine.setConvolverEnableProgressiveUpgrade(false);
```

**結論**: Progressive Upgrade 試験のためには、この設定を CLI オプションで上書きできるようにする必要がある。例えば `--cli-progressive-upgrade` フラグを新設する。または、TC-16 用のCLIオプションを Phase 0 で追加する。

---

## 4. 評価指標・計算方法の検証

### 4.1 周波数応答RMS誤差 ⚠️

`scipy.signal.freqz` の出力は複素数ゲインであり、`log10(abs(H))` は dB 値にならない。計画書の疑似コードは誤り。
```python
# 誤: error_db = 20 * log10(abs(H[mask])) - 20 * log10(abs(theoretical_response[mask]))
# 正: error_db = 20 * log10(abs(H_measured[mask]) / abs(H_theoretical[mask]))
```

### 4.2 THD+N 計算 ⚠️

基本波インデックス検索の範囲が固定値（±10ビン）なのは周波数分解能に依存する。`periodogram` のデフォルト窓（Hanning）を使う場合、メインローブ幅は約 2bin のため、5ビン程度で十分。ただし、窓関数の指定がないため `scipy.signal.periodogram` のデフォルト（Hanningではなく矩形窓？）に依存することになる。

**推奨**: `scipy.signal.periodogram(data, fs=sr, window='flattop')` のように明示的に窓関数を指定する。Flattop窓は振幅精度が高い。

### 4.3 非調和成分エネルギー ⚠️

高調波除去の範囲が narrow（2〜5次）に設定されている。実際の非調和成分は 5次を超える高域にも現れる可能性がある。TC-05A〜TC-05D の対象帯域が 20-200Hz であることを考慮すると、5次までの除去で十分だが、その旨を明確に文書化すべき。

---

## 5. 修正推奨項目サマリ

| # | 重要度 | 該当箇所 | 問題 | 推奨修正 |
|---|:----:|---------|------|---------|
| 1 | **高** | TC-11 | 異なるIRのロードが現行CLIで不可能 | CLIオプション新設またはテスト内容変更 |
| 2 | **高** | TC-11B | クロスフェード時間が20ms（計画書は50ms） | 50ms→20msに修正 |
| 3 | **中** | TC-16 | Progressive Upgrade が常時無効化される | `--cli-progressive-upgrade` フラグ追加 |
| 4 | **中** | Phase 0-3 | pre-dither モードは実装複雑度大 | 当面 post-dither のみ実装 |
| 5 | **低** | TC-09 | OS有無の条件が未定義 | OS有効時に限定または周波数変更 |
| 6 | **低** | TC-14 | 長時間WAVのartifact容量超過 | 定期サンプリング方式に変更 |

---

## 6. 総評

本テスト計画は **ConvoPeq の音質評価自動化** として現実的で十分な網羅性を持ち、特に v2.0 からのレビュー指摘（実IR追加、Crossfade Storm、Mixed Phase試験）を反映している点は高く評価できる。

現行アーキテクチャとの整合性は概ね良好で、Phase 0（OutputCaptureSink）の最小実装から段階的に構築可能な設計になっている。

**最大の課題**は TC-11 の「異なるIRをロードする」要件が現行CLIで実現不可能な点である。この修正が最も優先度が高い。

上記6項目の修正後、95点以上の計画書になりうる。修正版の再レビューを推奨する。
