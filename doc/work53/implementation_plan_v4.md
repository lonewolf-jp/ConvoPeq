# ConvoPeq 音質評価自動化 改修計画書 v4.0

**ドキュメントバージョン**: 4.0
**ステータス**: 確定版（レビュー反映済み・未確定事項調査完了）
**対象バージョン**: v0.5.3 → v1.0 (QA Phase)
**作成日**: 2026-06-22

---

## 0. 策定経緯

本計画書は以下の検証・レビューを経て確定した：

| フェーズ | 内容 | 成果物 |
|---------|------|--------|
| v3.0提出 | 音質評価自動化テスト計画 原案 | `basic_test_plan.md` |
| レビュー#1 | ソースコード突合検証（6項目指摘） | `review_report.md`（86/100点） |
| 回答 | 全6項目の即時受諾・対応方針表明 | ユーザー回答 |
| 未確定事項調査 | 13項目確定 + 3項目継続調査 | `uncertainties_resolved.md` |
| **v4.0** | **全指摘反映・確定版** | **本書** |

---

## 1. 是正項目一覧（v3.0からの変更点）

| # | 項目 | 種別 | 変更内容 | 工数影響 |
|---|------|------|---------|---------|
| 1 | `--cli-ir-reload-list` 新設 | 新機能 | カンマ区切り複数IR指定でリロード時に順次/ランダム選択 | +0.5日 |
| 2 | TC-11B クロスフェード時間修正 | 文書修正 | 50ms → 20ms（ソースコード確認値） | なし |
| 3 | `--cli-progressive-upgrade` 新設 | 新機能 | Progressive Upgrade 有効化フラグ | +0.5日 |
| 4 | pre-dither モード延期 | スコープ縮小 | v1.0 では `none` / `post-dither` のみ実装 | -0.5日 |
| 5 | TC-09 OS条件明記 | 文書修正 | OS=2x以上が有効な場合のみ実行 | なし |
| 6 | TC-14 定期サンプリング方式 | 設計変更 | 1分毎に5秒間キャプチャ、artifact抑制 | +1.0日 |
| 7 | 評価指標計算コード修正 | 文書修正 | THD+N窓関数明示、周波数応答計算式修正 | なし |
| **純増工数** | | | | **+1.5日** |

---

## 2. 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CI (GitHub Actions / ローカル実行)                │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Stage: Build (MSVC/icx, Debug/Release)                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Stage: Audio Quality Tests (CTest/Python)                  │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │  Python Test Orchestrator (tests/audio_quality/)       │ │ │
│  │  │  1. generators.py - テスト信号生成（numpy + stdlib）    │ │ │
│  │  │  2. cli_runner.py - ConvoPeq.exe 起動＋--cli-* 制御    │ │ │
│  │  │  3. OutputCaptureSink → processBlockDouble 出口キャプチャ│ │ │
│  │  │  4. analyzers.py - FFT/THD+N/スペクトログラム/異常検出  │ │ │
│  │  │  5. golden_calculator.py - 理論値との比較             │ │ │
│  │  │  6. Report JUnit XML                                  │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 OutputCaptureSink の設計（確定）

**方針**: `AudioEngine` に `setOutputCaptureCallback(std::function<void(const juce::AudioBuffer<double>&)>)` を追加し、`processBlockDouble()` の出口（全処理完了後）でコールバックを呼び出す。

**v4.0 確定仕様**:

| 項目 | 仕様 |
|------|------|
| コールバック型 | `std::function<void(const juce::AudioBuffer<double>&)>` |
| 呼び出し位置 | `processBlockDouble()` 末尾（クロスフェード完了後） |
| スレッド安全性 | `std::atomic` でコールバック有効/無効を保護 |
| デフォルト | コールバック未登録（nullptr）→ 何もしない |
| キャプチャモード | `post-dither`（デフォルト、NoiseShaper適用後をキャプチャ） |
| `pre-dither` | v1.0 では未実装（将来拡張） |

**`--cli-capture-mode` の拡張性**:
- `none` - キャプチャ無効
- `post-dither` - 全処理完了後をキャプチャ（デフォルト）
- `pre-dither` - 受け付けるが `post-dither` と同義（警告表示、将来実装予定）

### 2.2 既存CLIの活用

`MainWindow::runCommandLineAutomation()` をテストハーネスから呼び出す形で利用する。新規CLI実行可能ファイルは作成しない。

### 2.3 各CLIオプションの状態（v4.0）

| オプション | 状態 | 備考 |
|-----------|:----:|------|
| `--cli-run` | ✅ 既存 | 自動処理開始 |
| `--cli-ir` | ✅ 既存 | IRファイル指定 |
| `--cli-ir-reload-count` | ✅ 既存 | IRリロード回数 |
| `--cli-ir-reload-interval-ms` | ✅ 既存 | リロード間隔(ms) |
| **`--cli-ir-reload-list`** | **🔧 追加** | **カンマ区切り複数IR** |
| `--cli-bypass-burst-count` | ✅ 既存 | バイパスバースト回数 |
| `--cli-bypass-burst-interval-ms` | ✅ 既存 | バースト間隔(ms) |
| `--cli-bypass-burst-value` | ✅ 既存 | バイパス値(0/1) |
| `--cli-intent-burst-count` | ✅ 既存 | Intentバースト回数 |
| `--cli-intent-burst-interval-ms` | ✅ 既存 | Intentバースト間隔(ms) |
| **`--cli-output-wav`** | **🔧 追加** | **出力WAVファイルパス** |
| **`--cli-capture-mode`** | **🔧 追加** | **none/post-dither** |
| **`--cli-progressive-upgrade`** | **🔧 追加** | **Progressive Upgrade有効化** |
| `--cli-exit-ms` | ✅ 既存 | 自動終了時間(ms) |
| `--cli-order` | ✅ 既存 | 処理順序 |
| `--cli-noise-shaper` | ✅ 既存 | ノイズシェイパータイプ |
| `--cli-dither-bit-depth` | ✅ 既存 | ディザービット深度 |
| `--cli-phase` | ✅ 既存 | 位相モード |
| `--cli-device-type` | ✅ 既存 | オーディオデバイスタイプ |

---

## 3. フェーズ構成（確定版）

### Phase 0: CLI拡張 + OutputCaptureSink（工数: 5日）

| タスク | 説明 | 成果物 |
|--------|------|--------|
| 0-1 | `AudioEngine` に `OutputCaptureSink` 機能を実装。`setOutputCaptureCallback()` 追加、`processBlockDouble()` 出口でコールバック | `AudioEngine.h/cpp` |
| 0-2 | `--cli-output-wav` オプション追加。指定パスに WAV 出力 | `MainWindow.cpp` |
| 0-3 | `--cli-capture-mode` 追加（`none`/`post-dither`の2モード） | `MainWindow.cpp` |
| 0-4 | **🔧 `--cli-ir-reload-list` 追加**。カンマ区切り複数IR | `MainWindow.cpp` |
| 0-5 | **🔧 `--cli-progressive-upgrade` 追加**。Progressive Upgrade有効化 | `MainWindow.cpp` |
| 0-6 | **🔧 定期キャプチャ機能実装**（TC-14用: 1分毎5秒間キャプチャ） | `MainWindow.cpp` + OutputCaptureSink |
| 0-7 | テストハーネステンプレート作成 | `cli_runner.py` |

**受け入れ基準**:
- `ConvoPeq.exe --cli-output-wav out.wav --cli-capture-mode post-dither` で出力WAV生成
- `--cli-ir-reload-list "a.wav,b.wav,c.wav" --cli-ir-reload-count 10` で複数IRロード
- `--cli-progressive-upgrade` 指定時のみ Progressive Upgrade 有効化

### Phase 1: Pythonテストオーケストレーター（工数: 4日）

| タスク | 説明 | 成果物 |
|--------|------|--------|
| 1-1 | テスト信号生成モジュール | `generators.py` |
| 1-2 | 解析モジュール（scipy優先 + structフォールバック） | `analyzers.py` |
| 1-3 | CLIラッパー | `cli_runner.py` |
| 1-4 | 理論値計算ユーティリティ | `golden_calculator.py` |
| 1-5 | pytestフィクスチャ | `conftest.py` |
| 1-6 | ビルド構成別設定 | `config/test_config.yaml` |

**scipy依存性**: 推奨（`python -m pip install scipy numpy`）。フォールバックとして stdlib（`wave`, `struct`, `math`）のみでも動作する代替実装を用意。

**WAV入出力**:
- `scipy.io.wavfile` を優先使用（32bit float対応）
- フォールバック: `struct` 自力パース（work52の `analyze_ir.py` で実績あり）

### Phase 2: DSP品質試験（工数: 9日）

| ID | テストケース名 | 入力信号 | IR | 評価指標 | 許容誤差 | 優先度 |
|----|---------------|----------|----|---------|----------|--------|
| TC-01 | 周波数応答（ディラック） | 20-20k 対数スイープ | ディラックIR | 振幅スペクトルRMS誤差 | ≤ 0.05 dB | 高 |
| TC-01B | 周波数応答（実IR） | 20-20k 対数スイープ | ルーム補正IR | 振幅スペクトルRMS誤差（ゴールデン比較） | ≤ 0.1 dB | 高 |
| TC-02 | 周波数応答（Mixed Phase） | 20-20k 対数スイープ | ディラックIR | 振幅スペクトルRMS誤差 | ≤ 0.05 dB | 高 |
| TC-03 | 低域THD+N | 20,40,50,80,100Hz 正弦波 | ディラックIR | THD+N | Debug:-80dB / Release:-100dB | 高 |
| TC-04 | 無音ノイズフロア | 無音 | ディラックIR | 20-200Hz RMS | ≤ -120 dBFS | 中 |
| TC-05A | 低域ノイズ（50Hz） | 50Hz正弦波 | ディラックIR | 非調和成分エネルギー | ≤ -90 dBFS | 高 |
| TC-05B | 低域ノイズ（40Hz） | 40Hz正弦波 | ディラックIR | 非調和成分エネルギー | ≤ -90 dBFS | 高 |
| TC-05C | 低域ノイズ（80Hz） | 80Hz正弦波 | ディラックIR | 非調和成分エネルギー | ≤ -90 dBFS | 高 |
| TC-05D | 低域ノイズ（実IR） | 50Hz正弦波 | ルーム補正IR | 非調和成分エネルギー（ゴールデン差分） | ≤ -85 dBFS | 高 |
| TC-06 | スペクトログラムクリーン性 | 20-200Hz 対数スイープ | ディラックIR | アーティファクトエネルギー比 | ≤ -60 dB | 高 |
| TC-07 | フィルタ特性（LPF） | 20-20k 対数スイープ | Butterworth LPF 1kHz | カットオフ・減衰量理論値誤差 | ≤ 0.2 dB | 中 |
| TC-08 | モード切替過渡応答 | 50Hz正弦波 | ディラックIR | 出力振幅ジャンプ量 | ≤ -90 dBFS | 中 |
| **TC-09** | **オーバーサンプリングエイリアシング** | **21/22/23kHz 正弦波** | **ディラックIR** | **エイリアシング成分** | **≤ -120 dBFS** | **中** |
| TC-10 | バイパスA/B比較 | 50Hz正弦波 | ディラックIR | ノイズフロア差 | 特定モジュール検出 | 高 |

**TC-09 注記**: 本テストは **オーバーサンプリング（2x/4x/8x）が有効な設定でのみ実行する。OS=1x（48kHz）では 23kHz がナイキスト周波数（24kHz）に近すぎるため実施しない。**

### Phase 3: ISR Runtime試験（工数: 7日）

| ID | テストケース名 | 内容 | 評価指標 | 優先度 |
|----|---------------|------|---------|--------|
| **TC-11** | **IR Reload Storm** | **`--cli-ir-reload-list` で指定された複数IRを100回連続リロード** | クラッシュなし、NaN/Inf/DC異常なし | 高 |
| **TC-11B** | **Crossfade Storm** | **reload interval < crossfade duration（約20ms）の条件で50回連続IRリロード。interval=10ms推奨** | クラッシュなし、NaN/Inf/DC異常なし | 高 |
| TC-12 | Bypass Burst | 1000回バイパス切替（`--cli-bypass-burst-count 1000 --cli-bypass-burst-interval-ms 10`） | NaN/Inf/DC異常なし | 高 |
| TC-13 | Publication Saturation | 500 Intent 投入（`--cli-intent-burst-count 500 --cli-intent-burst-interval-ms 5`） | Reject Count ≤ 10 | 中 |
| **TC-14** | **長期耐久試験** | **PR時:30分 / Nightly:1時間。定期サンプリング方式（1分毎5秒間キャプチャ）でartifact抑制** | 1分毎サンプリングでNaN/Inf/DC異常なし、メモリ増加なし | 高 |

**全Runtime試験に共通の異常検査**:
- `NaN` が含まれていないこと
- `Inf` が含まれていないこと
- デノーマル（絶対値 < 1e-20）が多数含まれていないこと
- DCオフセット（全サンプル平均）が 1e-6 未満であること

### Phase 4: Mixed Phase / Progressive Upgrade試験（工数: 3日）

| ID | テストケース名 | 内容 | 評価指標 | 優先度 |
|----|---------------|------|---------|--------|
| TC-15 | Mixed Phase Cache Rebuild | 3段階比較（最適化前/後/キャッシュ再読込） | 振幅スペクトルRMS ≤ 0.1 dB | 高 |
| **TC-16** | **Progressive Upgrade** | **`--cli-progressive-upgrade` フラグ指定必須。FFT 512→1024→2048 ステップ経由** | 各ステップ移行前後出力差分 ≤ -60 dBFS | 中 |

### Phase 5: CI統合と運用準備（工数: 3日）

| タスク | 説明 | 成果物 |
|--------|------|--------|
| 5-1 | GitHub Actions workflow 作成（PR時全テスト / Nightly全テスト+1h耐久） | `.github/workflows/audio_quality.yml` |
| 5-2 | CTest統合（`add_test` + Pythonテスト呼び出し） | `CMakeLists.txt` |
| 5-3 | 許容誤差の統計的調整 | `test_config.yaml` |
| 5-4 | テスト失敗時 artifact アップロード | workflow設定 |
| 5-5 | JUnit XML 出力 | Pythonテストレポーター |
| 5-6 | **CI音声デバイス確認**: Windows Runner で `--cli-run --cli-exit-ms 5000` の動作検証 | CI検証 |

---

## 4. 実装スケジュール

| 週 | 実施フェーズ | マイルストーン |
|----|-------------|---------------|
| 1 | Phase 0 (CLI拡張 + OutputCaptureSink) | `--cli-output-wav` + `--cli-ir-reload-list` + `--cli-progressive-upgrade` が動作 |
| 2 | Phase 1 (Pythonオーケストレーター) | `pytest` でテストが収集・実行可能 |
| 3-4 | Phase 2 (DSP品質試験 TC-01〜TC-10) | TC-01〜TC-10 全パス |
| 5 | Phase 3 (ISR Runtime試験 TC-11〜TC-14) | TC-11〜TC-14 全パス |
| 5 | Phase 4 (Mixed Phase試験 TC-15〜TC-16) | TC-15〜TC-16 全パス |
| 6 | Phase 5 (CI統合と微調整) | PR時に自動テスト実行 |

**総工数**: 31 人日（約 4.5 週間）

---

## 5. ディレクトリ構成（最終形）

```
ConvoPeq/
├── src/...
├── tools/diagnostics/          # work52 diagnostics（維持）
├── tests/audio_quality/
│   ├── __init__.py
│   ├── conftest.py             # pytest fixtures
│   ├── generators.py           # テスト信号生成（numpy / stdlib）
│   ├── analyzers.py            # FFT/THD/スペクトログラム/異常検出
│   ├── golden_calculator.py    # 理論値計算（scipy / 手動実装）
│   ├── cli_runner.py           # ConvoPeq.exe 起動ラッパー
│   ├── test_core.py            # TC-01〜TC-10
│   ├── test_runtime.py         # TC-11〜TC-14
│   ├── test_mixed_phase.py     # TC-15〜TC-16
│   ├── test_regression.py      # ゴールデンデータ比較
│   ├── data/reference/         # Git LFS: 実測IR
│   │   ├── room_correction.wav
│   │   ├── cabinet_ir.wav
│   │   └── long_reverb.wav
│   ├── data/golden/            # Git LFS: ゴールデン出力（最小限）
│   ├── scripts/                # 補助スクリプト
│   └── config/test_config.yaml # 許容誤差設定
├── .github/workflows/
│   ├── audio_quality_pr.yml    # PR時: 全テスト（短縮版）
│   └── audio_quality_nightly.yml # Nightly: 全テスト + 1h耐久
└── CMakeLists.txt              # add_test 追加
```

---

## 6. リスク管理（v4.0更新）

| リスク | 影響度 | 軽減策 | 状態 |
|--------|:-----:|--------|:----:|
| CI環境で音声デバイスがない | 高 | `--cli-device-type` 指定 + Windows WASAPI 利用。GitHub Actions Runner実機で Phase 5 検証 | **調査済** |
| scipy/numpy未インストール | 中 | stdlibフォールバック実装を `analyzers.py` に用意 | **対応済** |
| 32bit float WAVがPythonで読めない | 中 | `scipy.io.wavfile`優先 + `struct`自力パースのフォールバック | **対応済** |
| `--cli-ir-reload-list` 実装工数超過 | 低 | 既存リロードループの拡張のみ。+0.5日 | **見積済** |
| TC-14 長期耐久のartifact容量超過 | 低 | 定期サンプリング方式（1分毎5秒）で 1.9MB/サンプルに抑制 | **対応済** |
| Progressive Upgrade テストが時間切れ | 中 | CMA-ES最適化のイテレーション数をテスト用に削減するオプション | **計画済** |

---

## 7. 残課題（Phase 5以降）

| # | 項目 | 内容 | 解決条件 |
|---|------|------|---------|
| R1 | CI音声デバイス実機確認 | GitHub Actions Windows Runner で音声コールバックの有無確認 | Phase 5 実機検証 |
| R2 | pre-dither キャプチャ | OutputCaptureSink の pre-dither モード実装 | 将来拡張 |
| R3 | 許容誤差の統計的調整 | 初回CI実行結果に基づく閾値調整 | Phase 5-3 |
| R4 | Slack通知 | テスト失敗通知（オプション） | Phase 5-6 |

---

## 8. 評価指標計算方法（修正版）

### 8.1 周波数応答RMS誤差（TC-01, TC-01B, TC-02, TC-07）

```python
def compute_frequency_response_rms_error(output_wav, theoretical_response, f_min=20, f_max=20000):
    sr, H_measured = read_wav_and_compute_freqz(output_wav)  # scipy.signal.freqz
    f = theoretical_response.frequencies  # 対応する周波数軸
    mask = (f >= f_min) & (f <= f_max)
    error_db = 20 * log10(abs(H_measured[mask]) / abs(theoretical_response[mask]))
    return sqrt(mean(error_db**2))
```

### 8.2 THD+N（TC-03）

```python
def compute_thdn(output_wav, fundamental_freq):
    sr, data = read_wav(output_wav)
    f, Pxx = scipy.signal.periodogram(data, fs=sr, window='flattop')  # 窓関数明示指定
    search_range = max(2, int(20 * len(f) / sr))  # 約20Hzの動的範囲
    fund_idx = argmax(Pxx[int(fundamental_freq * len(f) / sr - search_range):
                          int(fundamental_freq * len(f) / sr + search_range)]) + offset
    fund_power = Pxx[fund_idx]
    harmonic_indices = [fund_idx * n for n in range(2, 11) if fund_idx * n < len(Pxx)]
    noise_mask = np.ones(len(Pxx), dtype=bool)
    noise_mask[fund_idx] = False
    for h in harmonic_indices:
        noise_mask[h] = False
    noise_power = np.sum(Pxx[noise_mask])
    thdn = 10 * log10((noise_power + sum(Pxx[harmonic_indices])) / fund_power)
    return thdn
```

### 8.3 非調和成分エネルギー（TC-05A〜D）

```python
def compute_non_harmonic_energy(output_wav, fundamental_freq, f_min=20, f_max=200):
    sr, data = read_wav(output_wav)
    f, Pxx = scipy.signal.periodogram(data, fs=sr, window='flattop')
    fund_idx = argmax(Pxx[int(fundamental_freq * len(f) / sr - search_range):
                          int(fundamental_freq * len(f) / sr + search_range)]) + offset
    harmonic_indices = [fund_idx * n for n in range(2, 6) if fund_idx * n < len(Pxx)]  # 2〜5次
    mask = (f >= f_min) & (f <= f_max)
    mask[fund_idx] = False
    for h in harmonic_indices:
        mask[h] = False
    total_energy = np.sum(Pxx[mask])
    return 10 * log10(total_energy + 1e-18)
```

---

## 9. テスト結果解釈ガイド

| 結果 | 意味 | アクション |
|:----:|------|-----------|
| ✅ PASS | 品質基準を満たしている | 次のテストへ |
| ⚠️ WARN | 許容誤差に近いか、閾値超過1項目 | 出力WAV・スペクトル確認。Regressionの可能性 |
| ❌ FAIL | 品質基準未達 | CI artifact から出力WAV+診断ログ取得。原因調査 |
| 🔴 CRASH | プロセスクラッシュ | スタックトレース・アサーションログ確認。RuntimeHealthMonitor診断 |

---

## 10. 承認

| 役割 | 名前 | 承認日 | 署名 |
|------|------|--------|------|
| プロジェクトマネージャ | | | |
| テクニカルリード | | | |
| QAリード | | | |

---

*本計画書（v4.0）は、指摘事項6項目・未確定事項13項目を全て反映した確定版です。*
*レビュースコア: 86/100（v3.0）→ **94/100（v4.0目標）***
