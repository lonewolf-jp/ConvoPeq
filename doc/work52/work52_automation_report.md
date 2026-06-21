# work52 自動テスト環境構築 報告書

## 概要

ConvoPeq の「ジジジジ」ノイズ問題（Conv→Peq モード）の原因特定のため、完全自動テスト環境を構築した。

## 改修内容

### 1. 内蔵テストトーンジェネレータ（新規）

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`

**概要**: オーディオ処理チェーンにテスト信号注入機構を追加。

```
g_diagInjectTone = false  # 通常時: 無効（パフォーマンス影響ゼロ）
     ↓ true（capture開始で自動有効化）
diagFillTestTone(alignedL, alignedR, channelSamples)
     ↓
40Hz サイン波、振幅 0.5（-6dBFS）
2.5Hz 振幅変調（200ms ON / 200ms OFF のビートパターン）
     ↓
alignedL/alignedR を上書き → Convolver へ入力
```

**有効化条件**: `diagStartCapture()` が呼ばれると自動で `g_diagInjectTone = true` に設定される。

### 2. Convolver出力キャプチャ（既存・更新）

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`

**概要**: Convolver 処理直後の出力を `C:\TEMP\conv_output_l.raw` にキャプチャ。

- 形式: raw double (64-bit IEEE 754), モノラル, 48kHz
- ファイル名: `C:\TEMP\conv_output_l.raw`
- 書き込みタイミング: 各オーディオブロック処理完了後（Convolver → EQ → 出力フィルタの前）
- 終了: ファイルサイズが 48kB を超えると自動停止（1秒分）

### 3. 完全自動テストスクリプト（更新）

**ファイル**: `tools/diagnostics/run_conv_diag.ps1`

**仕組み**:
1. 古いキャプチャファイル削除
2. ConvoPeq 起動（CLI自動化モード）
   - `--cli-ir "IRファイル" --cli-order Conv->Peq --cli-run --cli-exit-ms 15000`
3. ConvoPeq が起動 → オーディオコールバック開始
4. 内蔵テストトーンが自動生成・注入（外部WAV不要）
5. Convolver 出力が自動キャプチャ
6. 15秒後に ConvoPeq 自動終了
7. Python 解析スクリプトがキャプチャファイルを評価

### 4. 解析スクリプト（既存・互換性確認済み）

**ファイル**: `tools/diagnostics/analyze_conv_output.py`

**評価基準**:
| 項目 | PASS | WARN | FAIL |
|------|------|------|------|
| DCオフセット | < 1e-6 | < 1e-4 | >= 1e-3 |
| ピーク振幅 | < 0.99 | < 0.999 | >= 1.0 |
| ブロック境界不連続性 | < 0.01 | < 0.1 | >= 0.1 |
| 高域ノイズフロア | < 0.001 | - | - |

## テスト実行手順

### 自動テスト（推奨）

```powershell
# ターミナルで以下を実行
cd C:\VSC_Project\ConvoPeq
$env:PYTHONUTF8="1"
.\tools\diagnostics\run_conv_diag.ps1
```

### 手動テスト

```powershell
# 1. 古いキャプチャを削除
Remove-Item C:\TEMP\conv_output_l.raw -ErrorAction SilentlyContinue

# 2. ConvoPeq 起動
.\build\ConvoPeq_artefacts\Debug\ConvoPeq.exe `
    --cli-ir "C:\Users\user\Documents\conv_filter\impulse.wav" `
    --cli-order Conv->Peq --cli-noise-shaper Psychoacoustic `
    --cli-dither-bit-depth 0 --cli-run --cli-exit-ms 30000

# 3. 解析（ConvoPeq終了後）
$env:PYTHONUTF8="1"; python tools\diagnostics\analyze_conv_output.py --raw C:\TEMP\conv_output_l.raw --sr 48000
```

## 信号経路

```
テストトーン (40Hz, -6dBFS, 2.5Hz beat)
  → [processInputDouble] → diagFillTestTone() が alignedL/R を上書き
  → (procesInputDouble出力) → alignedL/R (L/R独立)
  → [Convolver::process()] → Conv→Peq順
     → Add(block, alignedL, alignedR)
     → Get(block, convOutL, convOutR)
  → ★キャプチャ: convOutL → C:\TEMP\conv_output_l.raw ← ここを解析
  → [EQProcessor::processBlockDouble()] → 6-band PEQ
  → [outputFilter.process()] → UltraHighRateDCBlocker (3Hz IIR HPF)
  → [input makeup gain]
  → [SoftClip] → [NoiseShaper] → [output makeup gain]
  → 出力
```

## 期待される結果

| シナリオ | 期待される判定 | 説明 |
|----------|---------------|------|
| 正常動作 | PASS | Convolver 出力が純粋な畳み込み結果 |
| パーティション境界グリッチ | WARN/FAIL | ブロック境界で不連続が検出される |
| IR共振問題 | WARN | 40Hz 以外の周波数成分が異常に大きい |
| DC Blocker影響 | PASS | DCオフセットは検出されないはず |
| 無音（処理未実行） | INFO(silent) | キャプチャファイルが未作成または空 |

## 注意事項

1. **CLI自動動作の前提条件**:
   - ConvoPeq が `--cli-run` フラグでオーディオデバイスを自動起動すること
   - オーディオコールバックが定期的に fire すること
   - オーディオ入力端子に何も接続されていなくてもコールバックは継続すること
   - **これらが満たされない場合、テストトーンは生成されない**

2. **キャプチャ停止条件**:
   - `diagWriteCapture()` はファイルサイズが 48kB（6000 doubles = 0.125秒）を超えると停止
   - 短いキャプチャでも十分な分析は可能

3. **既存キャプチャファイル**:
   - 実行前に `C:\TEMP\conv_output_l.raw` が存在する場合は削除すること
   - 自動スクリプトが自動削除する

## 今後の展開

### テスト実行後、判定に応じた次のステップ

**PASS（Convolver正常）**:
- 問題は Convolver より後段（EQ / outputFilter / SoftClip / NoiseShaper）にある
- 各段階の出力を順次キャプチャして原因を特定する

**FAIL（Convolver異常）**:
- IR ファイル自体に問題がある可能性
- IRConverter の処理（resample / DC Blocker / Tukey窓 / scale factor）を確認
- MKLNonUniformConvolver の implementation を検証
- FFT パーティション境界処理を確認

**WARN（軽微な異常）**:
- ブロック境界の不連続がパーティションサイズと一致するか確認
- ring buffer underflow が発生していないか確認

---

**作成日**: 2026-06-03
**ステータス**: 自動テスト環境構築完了、実行待ち
