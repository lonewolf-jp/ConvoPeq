# ConvoPeq 音質評価自動化 改修計画書 v7.0

**ドキュメントバージョン**: 7.0（コード監査反映版）
**策定日**: 2026-07-06
**ベース**: v6.3+（2026-06-22）+ コード監査結果（2026-07-06）
**ステータス**: **コード監査完了・全未確定事項解決 — Phase 0 即時開始**
**対象バージョン**: ConvoPeq v0.5.3 → v1.0 (QA Phase)

---

## 0. v6.3+ からの変更点

| # | 変更項目 | v6.3+ 記載 | v7.0 修正内容 | 重要度 |
|---|---------|-----------|--------------|--------|
| 0-1 | TC-01B 参照ファイル | `reference/room_correction.wav` | **`sampledata/impulse_room_correction.wav`**（REW+rePhase生成、実測IR、48kHz/32bit float/0.172秒） | ❌ 修正必須 |
| 0-2 | TC-30 API | `getRecoveryHistory()` 前提（未調査） | **コード監査で未実装確定 → Phase 0-6 をプロトタイプ実装に昇格** | ❌ 修正必須 |
| 0-3 | 総テストケース数 | **全31件** | **全34件**（TC-05A〜D がカテゴリ表から欠落していた） | ⚠️ 修正推奨 |
| 0-4 | カテゴリ表「ノイズ」 | 3件（TC-04, TC-04A, TC-23） | **7件**（+ TC-05A〜D） | ⚠️ 修正推奨 |
| 0-5 | CLIオプション一覧 | 5新規のみ記載 | **全25既存＋5新規の完全リファレンス**（2.3節拡充） | ⚠️ 修正推奨 |
| 0-6 | TC-24 FFT条件 | 262144 固定 | **デフォルト262144、ベンチマーク後変更可能**と明確化 | ⚠️ 修正推奨 |
| 0-7 | OutputCaptureSink | `AudioBlock<double>&` | **`AudioBuffer<double>&`**（processBlockDoubleの実引数型に合わせる） | ℹ️ 修正推奨 |
| 0-8 | Phase 1/2 工数 | 40+50=90日 | 既存Pythonツール活用で**25〜30%削減余地** | ℹ️ 参考情報 |
| 0-9 | 成功基準 | 34/31テストケース | **全34テストケース** | ⚠️ 修正推奨 |

---

## 1. 改訂経緯

| 版 | 主要内容 | 判定 |
|---|---------|------|
| v1.0〜v6.3 | 初版〜最終調整（v6.3+ までの全履歴） | 継承 |
| **v6.3+** | **コード監査完了、Phase 0 着手可能と確定（2026-06-22）** | **凍結** |
| **v7.0** | **コード監査結果反映版（2026-07-06）**<br>- TC-01B参照パス修正<br>- TC-30 APIプロトタイプ実装に昇格<br>- 全CLIオプション棚卸し反映<br>- 既存Pythonツール活用方針追加 | **本版を実装開始版とする** |

---

## 2. 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CI (GitHub Actions)                              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Stage: Build (MSVC/icx, Debug/Release)                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Stage: Audio Quality Tests (CTest)                          │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │  Python Test Orchestrator (run_quality_tests.py)        │ │ │
│  │  │  1. Generate test signals (numpy)                      │ │ │
│  │  │  2. Launch ConvoPeq.exe with --cli-* options           │ │ │
│  │  │  3. OutputCaptureSink 経由で最終出力を取得              │ │ │
│  │  │  4. Analyze vs theoretical/golden                      │ │ │
│  │  │  5. NaN/Inf/Denormal/DC 異常検査                       │ │ │
│  │  │  6. Report JUnit XML + HTML (RMAA互換)                │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 既存CLIの活用

`MainWindow::runCommandLineAutomation()` をテストハーネスから呼び出す形で利用する。新たな `ConvoPeqCLI` 実行可能ファイルは作成しない。

### 2.2 OutputCaptureSink 設計

`processBlockDouble()` の出口（**OutputFilter適用後・Crossfade完了後**）にコールバックを設置し、最終出力をキャプチャする。

```cpp
// AudioEngine.h に追加
using OutputCaptureCallback = std::function<void(const juce::AudioBuffer<double>&)>;
void setOutputCaptureCallback(OutputCaptureCallback cb) noexcept;

// AudioEngine.Processing.BlockDouble.cpp の processBlockDouble() 出口（L470直後）で呼び出し
if (outputCaptureCallback) {
    outputCaptureCallback(buffer);  // ★ AudioBuffer<double>& （AudioBlockではない）
}
```

**挿入箇所確定**: `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` 内、`processBlockDouble()` の DSP 処理完了直後（`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ブロックの直前、L470 ライン直後）。
この位置であれば OutputFilter 適用後・Crossfade完了後の最終出力を正しくキャプチャできる。

### 2.3 CLIオプション完全リファレンス（全30オプション）

#### 既存CLIオプション（25個） — すべて `runCommandLineAutomation()` で解析済み

| # | オプション | タイプ | 説明 | 計画での使用 |
|---|-----------|--------|------|-------------|
| 1 | `--cli-run` | フラグ | 自動化起動検出 | - |
| 2 | `--cli-start-learning` | フラグ | NoiseShaper学習開始 | - |
| 3 | `--cli-resume-learning` | フラグ | 学習再開 | - |
| 4 | `--cli-ir` | 値 | IRファイル指定 | TC-11, TC-12 |
| 5 | `--cli-device-type` | 値 | ASIO/WASAPI選択 | - |
| 6 | `--cli-buffer-samples` | 値 | バッファサイズ | - |
| 7 | `--cli-sample-rate-hz` | 値 | サンプルレート | - |
| 8 | `--cli-phase` | 値 | Mixed Phaseモード | TC-02, TC-15 |
| 9 | `--cli-order` | 値 | 処理順序（Conv/PEQ/Conv→PEQ/PEQ→Conv） | TC-15 |
| 10 | `--cli-dither-bit-depth` | 値 | ディザービット深度 | TC-04 |
| 11 | `--cli-noise-shaper` | 値 | ノイズシェイパータイプ | TC-04 |
| 12 | `--cli-post-load-dither-bit-depth` | 値 | 読込後ディザー深度 | TC-11 |
| 13 | `--cli-post-load-delay-ms` | 値 | 読込後ディザー適用遅延 | TC-11 |
| 14 | `--cli-ir-reload-count` | 値 | IRリロード回数 | TC-11, TC-11B |
| 15 | `--cli-ir-reload-interval-ms` | 値 | リロード間隔（ms） | TC-11, TC-11B |
| 16 | `--cli-bypass-burst-count` | 値 | バイパスバースト回数 | TC-12 |
| 17 | `--cli-bypass-burst-interval-ms` | 値 | バイパス間隔（ms） | TC-12 |
| 18 | **`--cli-bypass-burst-value`** | 値 | **バイパス値（0/1）** | **TC-12で明示的に使用** |
| 19 | `--cli-intent-burst-count` | 値 | Intentバースト回数 | TC-13 |
| 20 | `--cli-intent-burst-interval-ms` | 値 | Intent間隔（ms） | TC-13 |
| 21 | `--cli-target-ir-sec` | 値 | ターゲットIR長（秒） | TC-27 |
| 22 | `--cli-debounce-ms` | 値 | リビルドデバウンス（ms） | TC-11/11B |
| 23 | `--cli-f1-hz` | 値 | Mixed Phase遷移開始Hz | TC-15 |
| 24 | `--cli-f2-hz` | 値 | Mixed Phase遷移終了Hz | TC-15 |
| 25 | `--cli-exit-ms` | 値 | 自動終了遅延（ms） | 全テスト |

#### 新規追加CLIオプション（5個） — Phase 0で実装

| # | オプション | タイプ | 説明 | 実装場所 | 難易度 |
|---|-----------|--------|------|---------|--------|
| 26 | `--cli-output-wav` | 値 | 出力WAV保存先パス | MainWindow.cpp | 低 (2h) |
| 27 | `--cli-capture-mode` | 値 | `none` / `post-dither` | MainWindow.cpp | 低 (1h) |
| 28 | `--cli-dump-filter-coeffs` | フラグ | OutputFilter係数をJSON出力 | MainWindow.cpp + OutputFilter.h | 低 (2h) |
| 29 | `--cli-ir-reload-list` | 値 | 複数IRをカンマ区切りで指定 | MainWindow.cpp | 中 (4h) |
| 30 | `--cli-progressive-upgrade` | フラグ | Progressive Upgrade有効化 | MainWindow.cpp | 低 (1h) |

---

## 3. テストケース一覧（全34件）

### 3.1 カテゴリ別構成

| カテゴリ | テストケース | 数 | 対応RMAA |
|---------|-------------|----|----------|
| **周波数応答/位相** | TC-01, TC-01B, TC-02, TC-21 | 4 | Frequency Response / Phase |
| **ノイズ/リニアリティ** | TC-04, TC-04A, TC-23, **TC-05A, TC-05B, TC-05C, TC-05D** | **7** | Noise Level / Low-Level Linearity |
| **歪み（THD+N / IMD）** | TC-03, TC-17, TC-18, TC-24 | 4 | THD / IMD (AES17) |
| **フィルタ/エイリアシング** | TC-06, TC-07, TC-09, TC-28 | 4 | (補助) |
| **モード切替/過渡応答** | TC-08, TC-10 | 2 | (補助) |
| **ISR Runtime / Crossfade** | TC-11, TC-11B, TC-12, TC-13, TC-14, TC-25 | 6 | (ストレス) |
| **Mixed Phase / Upgrade** | TC-15, TC-16 | 2 | (機能固有) |
| **ConvoPeq固有** | TC-26, TC-27, TC-29A, TC-29B, TC-30 | 5 | (ISR専用) |
| **合計** | | **34** | |

### 3.2 各テストケース詳細

#### TC-01: 周波数応答（ディラック）

| 項目 | 内容 |
|------|------|
| 入力 | 20Hz-20kHz 対数スイープ、-6dBFS、10秒、48kHz |
| IR | ディラックIR（合成） — `tools/diagnostics/create_dirac_ir.py` で生成可能 |
| 評価 | 振幅スペクトルRMS誤差（20Hz-20kHz） |
| 閾値 | ≤ 0.05 dB |

#### TC-01B: 周波数応答（実IR）

| 項目 | 内容 |
|------|------|
| 入力 | 20Hz-20kHz 対数スイープ、-6dBFS、10秒、48kHz |
| IR | ルーム補正IR（実測） — **`sampledata/impulse_room_correction.wav`**（REW+rePhase生成、48kHz/32bit float/0.172秒） |
| 評価 | 振幅スペクトルRMS誤差（ゴールデン出力との比較） |
| 閾値 | ≤ 0.1 dB |

#### TC-02: 周波数応答（Mixed Phase）

| 項目 | 内容 |
|------|------|
| 入力 | 20Hz-20kHz 対数スイープ、-6dBFS、10秒、48kHz |
| IR | ディラックIR（合成） |
| 評価 | 振幅スペクトルRMS誤差（Mixed Phaseでもフラットを維持） |
| 閾値 | ≤ 0.05 dB |
| CLI | `--cli-phase` で Mixed Phase モード指定 |

#### TC-03: 低域THD+N（複数周波数）

| 項目 | 内容 |
|------|------|
| 入力 | 20, 40, 50, 80, 100Hz 定常正弦波、-6dBFS、3秒、48kHz |
| IR | ディラックIR |
| 評価 | 各周波数における THD+N |
| 閾値 | Debug: ≤ -80dB / Release: ≤ -100dB |

#### TC-04: ノイズフロア（Unweighted）

| 項目 | 内容 |
|------|------|
| 入力 | 無音（全ゼロ）、3秒、48kHz |
| IR | ディラックIR |
| 評価 | 20-200Hz 帯域RMSノイズレベル（フラット特性） |
| 閾値 | ≤ -120 dBFS |

#### TC-04A: ノイズフロア（A-weighted）

| 項目 | 内容 |
|------|------|
| 入力 | 無音（全ゼロ）、3秒、48kHz |
| IR | ディラックIR |
| 評価 | A特性重み付けフィルタ適用後のRMSレベル |
| 閾値 | ≤ -110 dBFS（RMAA "Excellent" 相当） |

#### TC-05A〜D: 低域ノイズ（拡充版）

| 項目 | TC-05A | TC-05B | TC-05C | TC-05D |
|------|--------|--------|--------|--------|
| 入力 | 50Hz 正弦波 | 40Hz 正弦波 | 80Hz 正弦波 | 50Hz 正弦波 |
| IR | ディラック | ディラック | ディラック | **ルーム補正IR** |
| 評価 | 非調和成分エネルギー | 同左 | 同左 | 同左 |
| 閾値 | ≤ -90dBFS | ≤ -90dBFS | ≤ -90dBFS | ≤ -85dBFS |

#### TC-06: スペクトログラムクリーン性

| 項目 | 内容 |
|------|------|
| 入力 | 20-200Hz 対数スイープ、-6dBFS、5秒、48kHz |
| IR | ディラックIR |
| 評価 | スイープライン外のアーティファクト総エネルギー |
| 閾値 | ≤ -60 dB（相対） |

#### TC-07: フィルタ特性（LPF）

| 項目 | 内容 |
|------|------|
| 入力 | 20-20kHz 対数スイープ、-6dBFS、10秒、48kHz |
| IR | Butterworth LPF 1kHz IR（合成） — `tools/diagnostics/create_test_irs.py` で生成可能 |
| 評価 | カットオフ周波数・減衰量の理論値誤差 |
| 閾値 | ≤ 0.2 dB |

#### TC-08: モード切替過渡応答

| 項目 | 内容 |
|------|------|
| 入力 | 50Hz 定常正弦波、-6dBFS、3秒、48kHz |
| IR | ディラックIR |
| 評価 | EQバイパス切り替え時の出力ジャンプ量 |
| 閾値 | ≤ -90dBFS |

#### TC-09: オーバーサンプリングエイリアシング

| 項目 | 内容 |
|------|------|
| 入力 | 21, 22, 23kHz 定常正弦波、-6dBFS、3秒、48kHz |
| IR | ディラックIR |
| 評価 | エイリアシング成分（10kHz以下） |
| 閾値 | ≤ -120 dBFS |
| 条件 | **OS=2x/4x/8x 有効時のみ実行** |

#### TC-10: バイパスA/B比較（原因切り分け）

| 項目 | 内容 |
|------|------|
| 入力 | 50Hz 定常正弦波、-6dBFS、3秒、48kHz |
| IR | ディラックIR |
| 評価 | 各モジュール有効/無効時のノイズフロア差 |
| 目的 | 特定モジュールのみ異常上昇を検出 |

#### TC-11: IR Reload Storm

| 項目 | 内容 |
|------|------|
| 方法 | `--cli-ir-reload-list "dirac.wav,lpf.wav,hpf.wav,allpass.wav"` で100回連続IRロード |
| 評価 | クラッシュしないこと + 出力NaN/Infなし |
| 閾値 | NaN/Infカウント == 0 |

#### TC-11B: Crossfade Storm

| 項目 | 内容 |
|------|------|
| 方法 | reload interval = 10ms（クロスフェード20ms未満）で50回連続IRロード |
| 評価 | クラッシュしないこと + 出力NaN/Infなし + HealthMonitor Violation==0 |
| 閾値 | NaN/Infカウント == 0 / Violation Count == 0 |

#### TC-12: Bypass Burst

| 項目 | 内容 |
|------|------|
| 方法 | `--cli-bypass-burst-count 1000 --cli-bypass-burst-interval-ms 10 --cli-bypass-burst-value 0` |
| 評価 | 出力NaN/Infなし + HealthMonitor Violation==0 |
| 閾値 | NaN/Infカウント == 0 / Violation Count == 0 |

#### TC-13: Publication Saturation

| 項目 | 内容 |
|------|------|
| 方法 | `--cli-intent-burst-count 500 --cli-intent-burst-interval-ms 5` |
| 評価 | Publication Reject Count が一定値以下 |
| 閾値 | Reject Count ≤ 10 |

#### TC-14: 長期耐久試験

| 項目 | 内容 |
|------|------|
| 方法 | PR時: 30分 / Nightly: 1時間 連続処理（定期サンプリング方式） |
| 評価 | 出力NaN/Inf/Denormal/DC異常なし + HealthMonitor Violation==0 |
| 閾値 | 全項目クリア |

#### TC-15: Mixed Phase Cache Rebuild

| 項目 | 内容 |
|------|------|
| 方法 | 最適化前 → 最適化後 → キャッシュ再読込後 の応答比較 |
| CLI | `--cli-phase`, `--cli-f1-hz`, `--cli-f2-hz`, `--cli-order` |
| 評価 | 振幅スペクトルRMS誤差 / 群遅延RMS誤差 |
| 閾値 | 振幅 ≤ 0.1dB / 位相 ≤ 1° |

#### TC-16: Progressive Upgrade

| 項目 | 内容 |
|------|------|
| 方法 | `--cli-progressive-upgrade` 指定で L1(512)→L2(1024)→L3(2048) 移行 |
| 評価 | 移行前後で出力差分のピーク / RMS |
| 閾値 | ピーク ≤ -60dBFS / RMS ≤ -80dBFS |
| **実装状況** | `ProgressiveUpgradeThread` は完全実装済み。本オプションは `setConvolverEnableProgressiveUpgrade(true)` の呼び出しのみ |

#### TC-17: SMPTE IMD

| 項目 | 内容 |
|------|------|
| 入力 | 60Hz + 7kHz、振幅比 4:1、-6dBFS、3秒、48kHz |
| IR | ディラックIR |
| 評価 | 7kHz周囲のサイドバンドパワー / キャリアパワー |
| 閾値 | ≤ -90dB (Debug: -80dB) |

#### TC-18: CCIF IMD

| 項目 | 内容 |
|------|------|
| 入力 | 19kHz + 20kHz、振幅比 1:1、-6dBFS、3秒、48kHz |
| IR | ディラックIR |
| 評価 | 差周波成分（1kHz）パワー / 入力総パワー |
| 閾値 | ≤ -100dB (Debug: -90dB) |

#### TC-21: 高精度位相応答

| 項目 | 内容 |
|------|------|
| 入力 | 20Hz-20kHz 対数スイープ |
| IR | オールパスIR（2次, ρ=0.7, θ=45°） — `tools/diagnostics/create_test_irs.py` で生成可能 |
| 評価 | 理論位相曲線との位相誤差 / 群遅延誤差 |
| 閾値 | 位相 ≤ ±1° / 群遅延 ≤ ±0.1 サンプル |

#### TC-23: Low-Level Linearity（旧Dynamic Range）

| 項目 | 内容 |
|------|------|
| 入力 | 1kHz 正弦波、-60, -80, -100, -120 dBFS（各3秒） |
| IR | ディラックIR |
| 評価 | 出力レベルの理論値（入力レベル）との誤差 |
| 閾値 | -60dB: ±0.01dB / -80dB: ±0.05dB / -100dB: ±0.1dB / -120dB: ±1dB |

#### TC-24: IMD（AES17準拠）

| 項目 | 内容 |
|------|------|
| 入力ペア | 40+400, 63+630, 80+800, 50+500, 100+1k, 200+2k, 500+5k, 1k+10k |
| 解析帯域 | 20Hz〜20kHz（DCおよびf1/f2を除外） |
| FFT条件 | **サイズ: 262144（デフォルト） / 窓: Blackman-Harris / オーバーラップ: 75%** |
| 代替FFT（高速） | 65536, Hanning, 50%（Phase 1-6 のベンチマーク結果により採用可能） |
| 実装方針 | **Python (numpy/scipy) で実装** — コードベースのC++ FFT実装に依存しない |
| 評価 | 帯域内全残留成分（差周波＋和周波＋高次IMD）の総パワー |
| 閾値 | ≤ -90dB (Debug: -80dB) |

#### TC-25: Crossfade Integrity

| 項目 | 内容 |
|------|------|
| 条件 | 1ms, 5ms, 10ms, 20ms, 50ms の5条件 |
| 参照信号 | IR-B（LPF 1kHz）単独処理出力 |
| 測定信号 | クロスフェード完了後の出力 |
| 評価1（クロスフェード中） | ピークゲインジャンプ |
| 評価2（クロスフェード完了後） | Null Test（差分RMS） |
| 閾値（完了後） | **同一ビルド: ≤ -120dBFS / 異ビルド: ≤ -100dBFS** |

#### TC-26: Publication Latency（Telemetry）

| 項目 | 内容 |
|------|------|
| ステータス | **Pass/Fail テストではなく Telemetry（情報収集のみ）** |
| 測定 | IRサイズ別（Small/Medium/Large）のレイテンシ（平均, p95, p99, 最大）を記録 |
| CI判定 | **行わない**（CI環境の負荷変動による誤検出を防止） |

#### TC-27: Rebuild Consistency

| 項目 | 内容 |
|------|------|
| 方法 | 同一IRを100回再構築し、1vs2, 1vs10, 1vs50, 1vs100 を比較 |
| **主評価（Pass/Fail）** | **波形比較（RMS / Peak）** |
| **副次評価（診断用）** | SHA256ハッシュ（Pass/Fail非対象） |
| 閾値（同一ビルド） | RMS ≤ -140dBFS / Peak ≤ -130dBFS |
| 閾値（異ビルド） | RMS ≤ -120dBFS / Peak ≤ -110dBFS |

#### TC-28: OutputFilter Verification

| 項目 | 内容 |
|------|------|
| 方法 | `--cli-dump-filter-coeffs` で係数をJSON出力 → Pythonで理論値生成 |
| 実装 | `OutputFilter` に const getter 追加（privateメンバ `hcCoeff`, `lcCoeff`, `hpfCoeff`, `lpCoeff` の読み出し用） |
| 測定ポイント | HPF: 5, 10, 20, 40Hz / LPF: 10k, 19k, 22kHz |
| 評価 | `|actual_db(f) - expected_db(f)|` の最大値 |
| 閾値 | 全ポイントで **≤ 0.5 dB** |

#### TC-29A: NaN/Inf Robustness（全異常）

| 項目 | 内容 |
|------|------|
| 入力 | 全サンプルNaN, 全サンプル+Inf, 全サンプル-Inf（各1秒） |
| **仕様** | **非有限値→0.0** に置き換えられることを確認 |
| 評価 | 出力NaN/Infカウント == 0 / 出力RMS ≤ -140dBFS |

#### TC-29B: Sparse NaN Injection

| 項目 | 内容 |
|------|------|
| 入力 | 1kHz正弦波に100個のランダムNaNを注入（3秒） |
| 評価 | 出力NaN/Infカウント == 0 / RMS誤差 ≤ ±0.5dB |

#### TC-30: Runtime Recovery Verification

| 項目 | 内容 |
|------|------|
| 前提API | **`RuntimeHealthMonitor::getRecoveryHistory()`** — **Phase 0-6 でプロトタイプ実装** |
| 実装詳細 | → 4.1節 0-6 参照 |
| 方法 | IR Reload Storm（10ms間隔×50回）を実行 |
| 評価1 | 最終HealthState == `Healthy` |
| 評価2 | `getRecoveryHistory()` に `result != None` のイベントが少なくとも1件存在 |
| 評価3 | 復帰後の出力信号に異常（NaN/Inf/DC）がない |

---

## 4. フェーズ別実施計画

### Phase 0: CLI拡張 + OutputCaptureSink + RecoveryHistory API（工数: 7日）

| # | タスク | 成果物 | 確認項目 | 備考 |
|---|-------|--------|---------|------|
| 0-1 | `AudioEngine::setOutputCaptureCallback()` 実装 | `AudioEngine.h/cpp` | `processBlockDouble()` L470直後でコールバック呼び出し | 引数型は `AudioBuffer<double>&` |
| 0-2 | `--cli-output-wav` / `--cli-capture-mode` 実装 | `MainWindow.cpp` | 出力WAV保存、post-ditherモード | WAV出力実装（libsndfile or JUCE WAV） |
| 0-3 | `--cli-dump-filter-coeffs` 実装 | `MainWindow.cpp` + `OutputFilter.h` | OutputFilter係数のJSON出力 | `OutputFilter` に const getter 追加 |
| 0-4 | `--cli-ir-reload-list` 実装 | `MainWindow.cpp` | 複数IRをカンマ区切りで指定→逐次ロード | 既存の`--cli-ir` + `--cli-ir-reload-count` の拡張 |
| 0-5 | `--cli-progressive-upgrade` 実装 | `MainWindow.cpp` | `setConvolverEnableProgressiveUpgrade(true)` 呼び出し | 実装量最小 |
| 0-6 | **`RuntimeHealthMonitor::getRecoveryHistory()` 実装** | `RuntimeHealthMonitor.h/cpp` | リングバッファ履歴API | **v6.3+ から昇格**（設計レビュー→実装）|

**タスク 0-6 実装詳細**:

```cpp
// RuntimeHealthMonitor.h に追加
struct RecoveryEvent {
    uint64_t        timestampUs;
    RecoveryAction  action;
    RecoveryOutcome result;   // 初期値 None
};
static constexpr size_t kRecoveryHistoryCapacity = 64;

// 記録（m_actionCallback 呼び出し直前で挿入）
void recordRecoveryAction(RecoveryAction action) noexcept {
    const uint32_t idx = convo::fetchAddAtomic(
        m_recoveryHistoryWriteIndex, uint32_t{1},
        std::memory_order_relaxed) % kRecoveryHistoryCapacity;
    m_recoveryHistory[idx] = {
        getCurrentTimeUs(), action, RecoveryOutcome::None
    };
}

// 公開API
[[nodiscard]] std::span<const RecoveryEvent> getRecoveryHistory() const noexcept {
    return { m_recoveryHistory, kRecoveryHistoryCapacity };
}
```

### Phase 1: Pythonテストオーケストレーター（工数: 30〜40日）

**v7.0 での変更**: 既存Pythonツール（`tools/diagnostics/` の10本）を積極活用することで、新規実装工数を削減する。

| # | タスク | 成果物 | 既存ツール活用 | 備考 |
|---|-------|--------|---------------|------|
| 1-1 | テスト信号生成モジュール | `generators.py` | ✅ `create_dirac_ir.py` / `create_test_irs.py` / `generate_test_signal.py` をラップ | 対数スイープ、定常正弦波、無音、マルチトーン、Sparse NaN |
| 1-2 | 解析モジュール | `analyzers.py` | ✅ `compare_raw.py` / `compare_dirac.py` / `analyze_ir.py` を参照 | FFT、THD+N、IMD（AES17）、帯域別RMS、NaN/Inf検出 |
| 1-3 | CLIラッパー | `cli_runner.py` | - | `subprocess.run([ConvoPeq.exe, ...])` |
| 1-4 | 理論値計算ユーティリティ | `golden_calculator.py` | ✅ `analyze_compare.py` 参照 | フィルタ応答、ディラック応答、THD基準 |
| 1-5 | テストケース実装 | `test_core.py` | ✅ `analyze_conv_output.py` (403行) 参照 | TC-01〜TC-10, TC-17〜TC-24, TC-28, TC-29A, TC-29B |
| 1-6 | FFTサイズベンチマーク | 決定レポート | ✅ `analyze_ir.py` (221行) 参照 | 65536 / 131072 / 262144 の3候補を比較 |
| 1-7 | 閾値調整 | `test_config.yaml` | - | Debug/Release MSVC、CI環境での調整 |

**既存ツール活用による工数削減効果**:
- 信号生成: 約30%（3ツール合計209行を流用）
- 解析処理: 約25%（`analyze_conv_output.py` 403行の関数群をインポート）
- IR分析: 約40%（`analyze_ir.py` 221行を参照実装として利用）
- **全体: 28〜33日相当の新規実装で従来の40日分をカバー可能**

### Phase 2: Runtime試験（工数: 40〜50日）

| # | タスク | 成果物 | 備考 |
|---|-------|--------|------|
| 2-1 | ISR Runtime試験実装（TC-11, TC-11B, TC-12, TC-13, TC-14） | `test_runtime.py` | `soak_test_fault_injection.py` のパターンを参照可能 |
| 2-2 | Crossfade試験実装（TC-25: 5条件） | `test_runtime.py` | - |
| 2-3 | Publication/Rebuild試験実装（TC-26 Telemetry, TC-27多世代比較） | `test_runtime.py` | - |
| 2-4 | Recovery試験実装（TC-30 + `getRecoveryHistory()` API） | `test_runtime.py` | Phase 0-6 で実装済みAPIを利用 |
| 2-5 | Mixed Phase / Progressive Upgrade試験（TC-15, TC-16） | `test_mixed_phase.py` | - |
| 2-6 | 閾値調整（icx/異コンパイラ） | `test_config.yaml` | - |

### Phase 3: RMAAレポート + CI統合（工数: 20〜25日）

| # | タスク | 成果物 | 備考 |
|---|-------|--------|------|
| 3-1 | HTMLレポートジェネレーター（RMAA互換） | `report_generator.py` | `analyze_ir.py` のレポート機能を参照 |
| 3-2 | 品質ランク表示（Excellent/Very Good/Good/Poor） | 同上 | CI判定とは独立 |
| 3-3 | JSONデータ出力（Telemetry含む） | `report.json` | - |
| 3-4 | GitHub Actions workflow（PR/Nightly分離） | `.github/workflows/audio_quality.yml` | - |
| 3-5 | CTest連携（`add_test`） | `CMakeLists.txt` | - |
| 3-6 | PRコメント自動投稿 | GitHub API連携 | - |

---

## 5. 工数総括

| フェーズ | v6.3+ 工数 | v7.0 工数 | 差分理由 |
|---------|-----------|----------|---------|
| Phase 0: CLI拡張 | 5日 | **7日** | TC-30 API プロトタイプ実装追加（+2日）|
| Phase 1: DSP品質試験 | 40日 | **30〜40日** | 既存Pythonツール活用（-0〜10日）|
| Phase 2: Runtime試験 | 50日 | **40〜50日** | 同上（-0〜10日）|
| Phase 3: レポート/CI | 25日 | **20〜25日** | 同上（-0〜5日）|
| 予備（調整・安定化） | 30〜70日 | **30〜70日** | 変動なし |
| **合計（推奨管理レンジ）** | **180〜220日** | **127〜192日** | **中央値 160 人日（約20〜24週間）** |

---

## 6. 実装前チェックリスト

| # | 確認項目 | ステータス | 備考 |
|---|---------|-----------|------|
| 1 | `--cli-dump-filter-coeffs` が `OutputFilter` の生係数を JSON 出力できるか | ☐ | Phase 0-3 で確認。`OutputFilter` の private メンバに const getter 追加が必要 |
| 2 | `OutputCaptureSink` を `processBlockDouble()` の最終出口に設置できるか | ☐ | Phase 0-1 で確認。**L470直後**に確定 |
| 3 | TC-27 の「波形比較主評価・SHA256診断用」方針をチーム合意済みか | ☐ | 事前合意 |
| 4 | TC-24 の FFT 条件を Python で再現可能か | ☐ | Phase 1-6 でベンチマーク。**numpy/scipy で実装（Windows .venv にインストール済）** |
| 5 | TC-25 の「同一ビルド/異ビルド」区別を CI で実現可能か | ☐ | Phase 3-4 で設計。GitHub Actions artifacts 保存要対応 |
| 6 | **`getRecoveryHistory()` API の Phase 0 プロトタイプ実装完了** | ☐ | **Phase 0-6 で実装。v6.3+ から昇格（設計レビュー→実装）** |
| 7 | TC-01B の参照IRが `sampledata/impulse_room_correction.wav` で利用可能か | ☐ | **実測IR確認済み**（48kHz/32bit float/0.172秒/REW+rePhase）|
| 8 | TC-30 の Recovery イベント記録が `getRecoveryHistory()` 経由で可能か | ☐ | Phase 0-6 で確認。リングバッファ64エントリ固定 |

---

## 7. リスク評価と軽減策

| リスク | 影響度 | 軽減策 | 備考 |
|--------|-------|--------|------|
| TC-25 Null -120dBFS が実装差で不成立 | 中 | 同一ビルド/異ビルドで閾値分離済み | v6.3+ 継承 |
| TC-27 SHA256 がコンパイラ更新で変化 | 低 | Pass/Failから診断用に格下げ済み | v6.3+ 継承 |
| TC-24 FFT 262144 が CI 時間を圧迫 | 中 | Phase 1-6 でベンチマーク後、最適サイズを採用。**65536 代替あり** | 軽減策強化 |
| **TC-30 API 実装が Phase 0 に間に合わない** | **中→低** | **v7.0 でプロトタイプ実装に昇格済み（約50行）。Phase 0 内で完了保証** | **v6.3+ から改善** |
| CI環境での浮動小数点再現性 | 中 | Debug/Release/icx/MSVC で別閾値を設定 | v6.3+ 継承 |
| **TC-01B 参照IRの品質不足** | **低** | **`sampledata/impulse_room_correction.wav` 実在確認済み** | **v6.3+ から改善** |
| Phase 1 Python依存関係不足 | 低 | **numpy/scipy Windows .venv にインストール済み** | **v7.0 で確認** |
| Phase 1 工数超過（新規実装予測誤差） | 中 | 既存Pythonツール10本（合計1,373行）を活用。`analyze_conv_output.py`(403行)・`analyze_ir.py`(221行) の流用で緩和 | v7.0 改善 |

---

## 8. 成功基準

| 基準 | 内容 |
|------|------|
| Phase 0 完了 | 新CLIオプション5種 + RecoveryHistory API が全て動作し、`--cli-output-wav` で出力WAVが取得できる |
| Phase 1 完了 | TC-01〜TC-29 が CI 環境（Debug/Release MSVC）で全パス |
| Phase 2 完了 | TC-11〜TC-30 が CI 環境（Debug/Release MSVC/icx）で全パス |
| Phase 3 完了 | HTMLレポートが生成され、PRコメントに自動投稿される |
| **最終** | **全34テストケースが週次フルテストでパスし、長期運用が確立される** |

---

## 9. 承認

| 役割 | 名前 | 承認日 | 署名 |
|------|------|--------|------|
| プロジェクトマネージャ | | | |
| テクニカルリード | | | |
| QAリード | | | |

---

## 付録A: コード監査で使用したツール

| カテゴリ | ツール | バージョン | 用途 |
|---------|-------|-----------|------|
| **WSL検索** | grep (GNU) | - | 全文検索 |
| | ripgrep (rg) | 15.1.0 | 高速検索（`-g` glob対応） |
| | ast-grep | 0.44.0 | 構造検索（ASTパターンマッチ） |
| | fd (fdfind) | - | ファイル検索 |
| | fzf | - | ファジーファインダー |
| | sed (GNU) | 4.9 | ストリーム編集 |
| | awk (GNU) | 5.3.2 | テキスト処理 |
| **MCP** | AiDex MCP | 2.2.2 | コード検索・セマンティック検索 |
| | Serena MCP | 1.5.3 | プロジェクト管理・コード探索 |
| **CLI** | semble | - | 意味検索（`semble search "query" <path>`）|
| | cocoindex-code (ccc) | - | インデックス作成・意味検索 |
| | graphify | 0.9.7 | 知識グラフ操作 |

## 付録B: 既存Pythonツール活用マップ

```
tools/diagnostics/                    Phase 1 での活用
├── create_dirac_ir.py      (23行) → generators.py でラップ
├── create_test_irs.py      (89行) → generators.py でラップ
├── generate_test_signal.py (97行) → generators.py に統合
├── compare_raw.py          (37行) → analyzers.py 参照実装
├── compare_dirac.py        (33行) → analyzers.py RMS誤差計算
├── compare_all_irs.py      (94行) → analyzers.py FFT分析
├── analyze_compare.py      (49行) → analyzers.py フィルタ応答
├── analyze_conv_output.py (403行) → analyzers.py メイン参照
├── analyze_ir.py          (221行) → analyzers.py / report_generator.py 参照
├── analyze_verify.py      (154行) → 補助検証
├── check_build.py          (18行) → CI連携
└── compare_input_vs_conv.py (33行)→ 補助
tools/
└── soak_test_fault_injection.py (122行) → test_runtime.py 参照
```

---

**本計画（v7.0）をコード監査結果反映版とし、Phase 0 の実装を直ちに開始する。**
