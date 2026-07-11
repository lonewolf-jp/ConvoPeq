# ConvoPeq 音質評価自動化 改修計画書 v6.3+

**ドキュメントバージョン**: 6.3+（実装開始最終確定版）  
**策定日**: 2026-06-22  
**ステータス**: **正式凍結 — Phase 0 即時開始**  
**対象バージョン**: ConvoPeq v0.5.3 → v1.0 (QA Phase)

---

## 1. 改訂経緯

本計画は、以下の議論を経て最終化された。

| 版 | 主要内容 | 判定 |
|---|---------|------|
| v1.0 | 初版（オフラインCLI新規実装＋processOffline()） | 設計不適合 |
| v2.0 | 既存CLI活用＋OutputCaptureSink導入 | 改善 |
| v3.0 | ISR Runtime試験（TC-11〜TC-14）追加 | 拡充 |
| v4.0 | RMAA統合（TC-17〜TC-21） | 統合 |
| v5.0 | TC-23修正（Dynamic Range→Low-Level Linearity）、TC-25〜27追加 | 修正 |
| v6.0 | TC-28〜30追加、工数100人日 | 拡充 |
| v6.1 | TC-26 Telemetry化、TC-29B追加 | 現実化 |
| v6.2 | TC-24 AES17方式、TC-27多世代比較、TC-30履歴API | 最終調整 |
| v6.3 | TC-24 FFT条件固定、TC-25 Null閾値分離、TC-27波形比較主体 | 確定 |
| **v6.3+** | **コード監査完了、Phase 0 着手可能と確定** | **凍結** |

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

`MainWindow::runCommandLineAutomation()` をテストハーネスから呼び出す形で利用する【4†L380-L400】。新たな `ConvoPeqCLI` 実行可能ファイルは作成しない。

### 2.2 OutputCaptureSink 設計

`processBlockDouble()` の出口（**OutputFilter適用後・Crossfade完了後**）にコールバックを設置し、最終出力をキャプチャする【5†L14-L30】。

```cpp
// AudioEngine.h に追加
void setOutputCaptureCallback(std::function<void(const juce::dsp::AudioBlock<double>&)> callback);

// AudioEngine.Processing.BlockDouble.cpp の processBlockDouble() 出口で呼び出し
if (outputCaptureCallback) {
    outputCaptureCallback(block);
}
```

### 2.3 新規CLIオプション（Phase 0で実装）

| オプション | 説明 | 実装場所 |
|-----------|------|---------|
| `--cli-output-wav` | 出力WAV保存先を指定 | `MainWindow::runCommandLineAutomation()` |
| `--cli-capture-mode` | `none` / `post-dither`（デフォルト） | 同上 |
| `--cli-dump-filter-coeffs` | OutputFilter係数をJSON出力 | 同上 |
| `--cli-ir-reload-list` | 複数IRをカンマ区切りで指定 | 同上 |
| `--cli-progressive-upgrade` | Progressive Upgradeを有効化 | 同上 |

---

## 3. テストケース一覧（全31件）

### 3.1 カテゴリ別構成

| カテゴリ | テストケース | 数 | 対応RMAA |
|---------|-------------|----|----------|
| **周波数応答/位相** | TC-01, TC-01B, TC-02, TC-21 | 4 | Frequency Response / Phase |
| **ノイズ/リニアリティ** | TC-04, TC-04A, TC-23 | 3 | Noise Level / Low-Level Linearity |
| **歪み（THD+N / IMD）** | TC-03, TC-17, TC-18, TC-24 | 4 | THD / IMD (AES17) |
| **フィルタ/エイリアシング** | TC-06, TC-07, TC-09, TC-28 | 4 | (補助) |
| **モード切替/過渡応答** | TC-08, TC-10 | 2 | (補助) |
| **ISR Runtime / Crossfade** | TC-11, TC-11B, TC-12, TC-13, TC-14, TC-25 | 6 | (ストレス) |
| **Mixed Phase / Upgrade** | TC-15, TC-16 | 2 | (機能固有) |
| **ConvoPeq固有** | TC-26, TC-27, TC-29A, TC-29B, TC-30 | 5 | (ISR専用) |
| **合計** | | **31** | |

### 3.2 各テストケース詳細

#### TC-01: 周波数応答（ディラック）

| 項目 | 内容 |
|------|------|
| 入力 | 20Hz-20kHz 対数スイープ、-6dBFS、10秒、48kHz |
| IR | ディラックIR（合成） |
| 評価 | 振幅スペクトルRMS誤差（20Hz-20kHz） |
| 閾値 | ≤ 0.05 dB |

#### TC-01B: 周波数応答（実IR）

| 項目 | 内容 |
|------|------|
| 入力 | 20Hz-20kHz 対数スイープ、-6dBFS、10秒、48kHz |
| IR | ルーム補正IR（実測、`reference/room_correction.wav`） |
| 評価 | 振幅スペクトルRMS誤差（ゴールデン出力との比較） |
| 閾値 | ≤ 0.1 dB |

#### TC-02: 周波数応答（Mixed Phase）

| 項目 | 内容 |
|------|------|
| 入力 | 20Hz-20kHz 対数スイープ、-6dBFS、10秒、48kHz |
| IR | ディラックIR（合成） |
| 評価 | 振幅スペクトルRMS誤差（Mixed Phaseでもフラットを維持） |
| 閾値 | ≤ 0.05 dB |

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
| IR | ディラック | ディラック | ディラック | ルーム補正IR |
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
| IR | Butterworth LPF 1kHz IR（合成） |
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
| 方法 | `--cli-bypass-burst-count 1000 --cli-bypass-burst-interval-ms 10` |
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
| 評価 | 振幅スペクトルRMS誤差 / 群遅延RMS誤差 |
| 閾値 | 振幅 ≤ 0.1dB / 位相 ≤ 1° |

#### TC-16: Progressive Upgrade

| 項目 | 内容 |
|------|------|
| 方法 | `--cli-progressive-upgrade` 指定で L1(512)→L2(1024)→L3(2048) 移行 |
| 評価 | 移行前後で出力差分のピーク / RMS |
| 閾値 | ピーク ≤ -60dBFS / RMS ≤ -80dBFS |

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
| 閾值 | ≤ -100dB (Debug: -90dB) |

#### TC-21: 高精度位相応答

| 項目 | 内容 |
|------|------|
| 入力 | 20Hz-20kHz 対数スイープ |
| IR | オールパスIR（2次, ρ=0.7, θ=45°） |
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
| FFT条件 | **サイズ: 262144 / 窓: Blackman-Harris / オーバーラップ: 75%** |
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
| 前提API | **`RuntimeHealthMonitor::getRecoveryHistory()`**（`RecoveryEvent { timestamp, action, result }` のリスト） |
| 方法 | IR Reload Storm（10ms間隔×50回）を実行 |
| 評価1 | 最終HealthState == `Healthy` |
| 評価2 | `getRecoveryHistory()` に `result == executed` のイベントが少なくとも1件存在 |
| 評価3 | 復帰後の出力信号に異常（NaN/Inf/DC）がない |

---

## 4. フェーズ別実施計画

### Phase 0: CLI拡張 + OutputCaptureSink（工数: 5日）

| # | タスク | 成果物 | 確認項目 |
|---|-------|--------|---------|
| 0-1 | `AudioEngine::setOutputCaptureCallback()` 実装 | `AudioEngine.h/cpp` | `processBlockDouble()` 出口でコールバック呼び出し |
| 0-2 | `--cli-output-wav` / `--cli-capture-mode` 実装 | `MainWindow.cpp` | 出力WAV保存、post-ditherモード |
| 0-3 | `--cli-dump-filter-coeffs` 実装 | `MainWindow.cpp` | OutputFilter係数のJSON出力 |
| 0-4 | `--cli-ir-reload-list` 実装 | `MainWindow.cpp` | 複数IRのカンマ区切り指定 |
| 0-5 | `--cli-progressive-upgrade` 実装 | `MainWindow.cpp` | Progressive Upgrade有効化フラグ |
| 0-6 | `RuntimeHealthMonitor::getRecoveryHistory()` 設計レビュー | 設計書 | Phase B前にAPI確定 |

### Phase 1: Pythonテストオーケストレーター（工数: 40日）

| # | タスク | 成果物 |
|---|-------|--------|
| 1-1 | テスト信号生成モジュール | `generators.py`（対数スイープ、定常正弦波、無音、マルチトーン、Sparse NaN） |
| 1-2 | 解析モジュール | `analyzers.py`（FFT、THD+N、IMD（AES17）、帯域別RMS、スペクトログラム、NaN/Inf検出） |
| 1-3 | CLIラッパー | `cli_runner.py`（`ConvoPeq.exe` 起動、出力WAV読込） |
| 1-4 | 理論値計算ユーティリティ | `golden_calculator.py`（フィルタ応答、ディラック応答、THD基準） |
| 1-5 | テストケース実装（TC-01〜TC-10, TC-17〜TC-24, TC-28, TC-29A, TC-29B） | `test_core.py` |
| 1-6 | FFTサイズベンチマーク（65536/131072/262144） | 最終FFTサイズ決定 |
| 1-7 | 閾値調整（Debug/Release MSVC） | `test_config.yaml` |

### Phase 2: Runtime試験（工数: 50日）

| # | タスク | 成果物 |
|---|-------|--------|
| 2-1 | ISR Runtime試験実装（TC-11, TC-11B, TC-12, TC-13, TC-14） | `test_runtime.py` |
| 2-2 | Crossfade試験実装（TC-25: 5条件） | `test_runtime.py` |
| 2-3 | Publication/Rebuild試験実装（TC-26 Telemetry, TC-27多世代比較） | `test_runtime.py` |
| 2-4 | Recovery試験実装（TC-30 + `getRecoveryHistory()` API） | `test_runtime.py` |
| 2-5 | Mixed Phase / Progressive Upgrade試験（TC-15, TC-16） | `test_mixed_phase.py` |
| 2-6 | 閾値調整（icx/異コンパイラ） | `test_config.yaml` |

### Phase 3: RMAAレポート + CI統合（工数: 25日）

| # | タスク | 成果物 |
|---|-------|--------|
| 3-1 | HTMLレポートジェネレーター（RMAA互換） | `report_generator.py` |
| 3-2 | 品質ランク表示（Excellent/Very Good/Good/Poor） | 同上（CI判定とは独立） |
| 3-3 | JSONデータ出力（Telemetry含む） | `report.json` |
| 3-4 | GitHub Actions workflow（PR/Nightly分離） | `.github/workflows/audio_quality.yml` |
| 3-5 | CTest連携（`add_test`） | `CMakeLists.txt` |
| 3-6 | PRコメント自動投稿 | GitHub API連携 |

---

## 5. 工数総括

| フェーズ | 工数（人日） | 備考 |
|---------|-------------|------|
| Phase 0: CLI拡張 | 5 | OutputCaptureSink + 新CLIオプション |
| Phase 1: DSP品質試験 | 40 | TC-24 FFTベンチマーク含む |
| Phase 2: Runtime試験 | 50 | TC-25/27/30の安定化含む |
| Phase 3: レポート/CI | 25 | HTML + GitHub連携 |
| 予備（TC-25/27/30調整、CI安定化） | 30〜70 | 実績に応じて変動 |
| **合計（推奨管理レンジ）** | **180〜220** | **中央値 200 人日（約25〜30週間）** |

---

## 6. 実装前チェックリスト

| # | 確認項目 | ステータス | 備考 |
|---|---------|-----------|------|
| 1 | `--cli-dump-filter-coeffs` が `OutputFilter` の生係数を JSON 出力できるか | ☐ | Phase 0 で確認 |
| 2 | `OutputCaptureSink` を `processBlockDouble()` の最終出口に設置できるか | ☐ | Phase 0 で確認 |
| 3 | TC-27 の「波形比較主評価・SHA256診断用」方針をチーム合意済みか | ☐ | 事前合意 |
| 4 | TC-24 の FFT 条件を Python で再現可能か | ☐ | Phase 1 でベンチマーク |
| 5 | TC-25 の「同一ビルド/異ビルド」区別を CI で実現可能か | ☐ | Phase 0 で設計 |
| 6 | TC-30 API（`getRecoveryHistory()`）の設計を Phase B 前にレビュー済みか | ☐ | Phase 0 で設計レビュー |

---

## 7. リスク評価と軽減策

| リスク | 影響度 | 軽減策 |
|--------|-------|--------|
| TC-25 Null -120dBFS が実装差で不成立 | 中 | 同一ビルド/異ビルドで閾値分離済み |
| TC-27 SHA256 がコンパイラ更新で変化 | 低 | Pass/Failから診断用に格下げ済み |
| TC-24 FFT 262144 が CI 時間を圧迫 | 中 | Phase 1 でベンチマーク後、最適サイズを採用 |
| TC-30 API 設計が複雑化 | 中 | Phase 0 で設計レビューを先行実施 |
| CI環境での浮動小数点再現性 | 中 | Debug/Release/icx/MSVC で別閾値を設定 |

---

## 8. 成功基準

| 基準 | 内容 |
|------|------|
| Phase 0 完了 | 新CLIオプション5種が全て動作し、`--cli-output-wav` で出力WAVが取得できる |
| Phase 1 完了 | TC-01〜TC-29 が CI 環境（Debug/Release MSVC）で全パス |
| Phase 2 完了 | TC-11〜TC-30 が CI 環境（Debug/Release MSVC/icx）で全パス |
| Phase 3 完了 | HTMLレポートが生成され、PRコメントに自動投稿される |
| 最終 | 全31テストケースが週次フルテストでパスし、長期運用が確立される |

---

## 9. 承認

| 役割 | 名前 | 承認日 | 署名 |
|------|------|--------|------|
| プロジェクトマネージャ | | | |
| テクニカルリード | | | |
| QAリード | | | |

---

**本計画（v6.3+）を「実装開始最終確定版」とし、Phase 0 の実装を直ちに開始する。**