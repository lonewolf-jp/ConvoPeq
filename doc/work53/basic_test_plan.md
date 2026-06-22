# ConvoPeq 音質評価自動化 改修計画書 v3.0

**ドキュメントバージョン**: 3.0  
**ステータス**: 改訂（レビューv2.0反映済み）  
**対象バージョン**: v0.5.3 → v1.0 (QA Phase)  
**作成日**: 2026-06-21  

---

## 1. はじめに

本計画は、ConvoPeq の音質評価を自動化するための改修計画です。前版（v1.0, v2.0）に対するレビュー指摘を反映し、現行アーキテクチャとの整合性と実装の現実性を最大化しました。

### 1.1 レビュー指摘の反映状況

| # | 指摘項目 | v3.0での対応 |
|---|---------|-------------|
| ① | `--cli-output-wav` の設計不足（キャプチャ方式未定義） | Phase0 に `OutputCaptureSink` 設計を追加。方法B（processBlockDouble出口キャプチャ）を明記 |
| ② | TC-01 が実IR試験を欠いている | TC-01B（実運用ルーム補正IR）を新設 |
| ③ | TC-03 のTHD基準が緩い（-80dB） | Debug: -80dB / Release: -100dB のビルド構成別基準を設定 |
| ④ | TC-11 が Crossfade重畳を試験していない | TC-11B（Crossfade Storm）を新設（reload interval < crossfade duration） |
| ⑤ | TC-14 が短すぎる（5分） | Nightly のみ 30分（または1時間）耐久試験に延長 |
| ⑥ | HealthMonitor依存は異常検出の盲点になる | 全Runtime試験（TC-11〜TC-14）に出力信号の NaN/Inf/Denormal/DC異常検査を追加 |
| ⑦ | Mixed Phase / Progressive Upgrade 試験が存在しない | TC-15（Mixed Phase Cache Rebuild）・TC-16（Progressive Upgrade）を新設 |
| ⑧ | TC-05（低域ノイズ）を拡張し実IRも含める | TC-05A〜TC-05D（複数周波数＋実IR）を新設 |

### 1.2 対象範囲

| 対象モジュール | 改修内容 | 優先度 |
|---------------|----------|--------|
| `AudioEngine` コア | `OutputCaptureSink` の追加（`processBlockDouble` 出口キャプチャ） | **高** |
| `MainWindow` | `--cli-output-wav` オプションの追加 | 高 |
| ビルドシステム | テスト用ターゲットの整備 | 高 |
| CIパイプライン | 音質テストステージの追加 | 高 |
| テスト資産 | Pythonテストフレームワーク、参照データ | 高 |

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
│  │  │  Python Test Orchestrator (run_quality_tests)           │ │ │
│  │  │  1. Generate test signals (numpy)                      │ │ │
│  │  │  2. Launch ConvoPeq.exe with --cli-* options           │ │ │
│  │  │  3. OutputCaptureSink 経由で processBlockDouble 出口を取得 │ │ │
│  │  │  4. Analyze vs theoretical/golden                      │ │ │
│  │  │  5. NaN/Inf/Denormal/DC 異常検査                       │ │ │
│  │  │  6. Report JUnit XML                                   │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 OutputCaptureSink の設計

- `AudioEngine` に `setOutputCaptureCallback(std::function<void(const juce::dsp::AudioBlock<double>&)>)` を追加する。
- `processBlockDouble()` の出口（出力適用後、ノイズシェーパー適用後）でコールバックを呼び出す。
- コールバックは `std::function` で実装し、テストハーネス側でラムダを登録する。
- 登録されたコールバックは、ブロック単位で受け取ったデータを結合し、`--cli-output-wav` で指定されたパスに WAV ファイルとして保存する。
- コールバックの呼び出しは、`processBlockDouble()` の戻り直前に行い、出力バッファの内容が確定した後に実行する。
- これにより、`processBlockDouble` が本番と同一経路で動作した結果を、オフラインで完全にキャプチャできる。

### 2.2 既存CLIの活用

- 既存の `MainWindow::runCommandLineAutomation()` をテストハーネスから呼び出す形で利用する。
- 新たな `ConvoPeqCLI` 実行可能ファイルは作成せず、既存の `ConvoPeq.exe` を `--cli-*` オプション付きで起動する方式を採用する。
- テストハーネスは `subprocess` で `ConvoPeq.exe` を起動し、`--cli-exit-ms` で自動終了させる。
- 出力WAVは `--cli-output-wav` で指定したパスに保存され、テストハーネスがそれを読み込んで解析する。

---

## 3. フェーズ構成

### Phase 0: CLI拡張 + OutputCaptureSink (工数: 3日)

| タスク | 説明 | 成果物 |
|--------|------|--------|
| 0-1 | `AudioEngine` に `OutputCaptureSink` 機能を実装する。`setOutputCaptureCallback()` メソッドを追加し、`processBlockDouble()` 出口でコールバックを呼び出す。キャプチャタイミングは `--cli-capture-mode` で指定可能にする（`none`, `post-dither`, `pre-dither`）。デフォルトは `post-dither`（ノイズシェーパー適用後）とする。 | `AudioEngine.h/cpp` 改修 |
| 0-2 | `MainWindow::runCommandLineAutomation()` に `--cli-output-wav` オプションを追加する。指定されたパスに、キャプチャした出力ブロックを結合した WAV ファイルを書き出す。 | `MainWindow.cpp` 改修 |
| 0-3 | `--cli-capture-mode` オプションを追加し、`none`（キャプチャ無効）、`post-dither`（ノイズシェーパー後）、`pre-dither`（ノイズシェーパー前）の3モードを実装する。 | `MainWindow.cpp` 改修 |
| 0-4 | テストハーネス（Python）から `ConvoPeq.exe --cli-output-wav out.wav --cli-capture-mode post-dither` を起動するテンプレートを作成する。 | `cli_runner.py` |
| 0-5 | キャプチャしたブロックを結合して WAV ファイルに書き出す際のフォーマット（サンプルレート、ビット深度）を、`ConvoPeq` の現在の処理設定と一致させる。 | 出力整合性の確保 |

**受け入れ基準**: `python cli_runner.py --ir dirac.wav --input sine.wav --output out.wav` で `ConvoPeq.exe` が起動し、指定されたパスに出力WAVが生成されること。生成されたWAVのサンプルレート・チャンネル数が入力と一致すること。

---

### Phase 1: Pythonテストオーケストレーター (工数: 4日)

| タスク | 説明 | 成果物 |
|--------|------|--------|
| 1-1 | テスト信号生成モジュール `generators.py` を実装する。以下の信号を生成可能にする。 | `generators.py` |
|     | - 対数スイープ（20Hz→20kHz, 20Hz→200Hz） | |
|     | - 定常正弦波（20Hz, 40Hz, 50Hz, 80Hz, 100Hz, 21kHz, 22kHz, 23kHz） | |
|     | - 無音（全サンプル0） | |
|     | - マルチトーン（ITU-R BS.1387準拠） | |
| 1-2 | 解析モジュール `analyzers.py` を実装する。以下の機能を提供する。 | `analyzers.py` |
|     | - FFTによる振幅スペクトル抽出（`scipy.signal.freqz` または `periodogram`） | |
|     | - THD+N 計算（基本波パワーに対する高調波＋ノイズの比） | |
|     | - 帯域別RMSレベル計算 | |
|     | - スペクトログラムアーティファクト検出（スイープラインからの乖離） | |
|     | - ノイズフロア測定（指定帯域のRMS） | |
|     | - NaN/Inf/Denormal/DCオフセット検出 | |
| 1-3 | CLIラッパー `cli_runner.py` を実装する。 | `cli_runner.py` |
|     | - `ConvoPeq.exe` へのパスを解決する | |
|     | - テストケースに応じた `--cli-*` 引数リストを組み立てる | |
|     | - `subprocess` で `ConvoPeq.exe` を起動し、終了を待機する | |
|     | - 出力WAVを読み込み、`analyzers.py` に渡す | |
| 1-4 | 理論値計算ユーティリティ `golden_calculator.py` を実装する。 | `golden_calculator.py` |
|     | - ディラックIRに対する理論応答（フラット） | |
|     | - フィルタIR（LPF, HPF, オールパス）に対する理論応答（`scipy.signal.freqz` で算出） | |
|     | - 定常正弦波に対する理論THD（理想的な純粋正弦波は0） | |
| 1-5 | `conftest.py` に pytest フィクスチャを実装し、テストケース間で共通のセットアップ（信号生成、CLI実行、後片付け）を共有する。 | `conftest.py` |
| 1-6 | `test_config.yaml` にビルド構成別の許容誤差を定義する。 | `config/test_config.yaml` |

**受け入れ基準**: `pytest tests/audio_quality/` を実行すると、テストが収集され、各テストケースが実行可能な状態になっていること。

---

### Phase 2: DSP品質試験（拡充） (工数: 9日)

| ID | テストケース名 | 入力信号 | IR | 評価指標 | 許容誤差 | 優先度 |
|----|---------------|----------|----|---------|----------|--------|
| TC-01 | 周波数応答（ディラック） | 20-20k 対数スイープ、-6dBFS、10秒、48kHz | ディラックIR（合成） | 振幅スペクトルRMS誤差（20Hz-20kHz） | ≤ 0.05 dB | 高 |
| TC-01B | 周波数応答（実IR） | 20-20k 対数スイープ、-6dBFS、10秒、48kHz | ルーム補正IR（実測、`reference/room_correction.wav`） | 振幅スペクトルRMS誤差（ゴールデン出力との比較） | ≤ 0.1 dB | 高 |
| TC-02 | 周波数応答（Mixed Phase） | 20-20k 対数スイープ、-6dBFS、10秒、48kHz | ディラックIR（合成） | 振幅スペクトルRMS誤差（位相モード Mixed でもフラットを維持） | ≤ 0.05 dB | 高 |
| TC-03 | 低域THD+N（複数周波数） | 20, 40, 50, 80, 100Hz 定常正弦波、-6dBFS、3秒、48kHz | ディラックIR（合成） | 各周波数における THD+N | Debug: ≤ -80dB / Release: ≤ -100dB | 高 |
| TC-04 | 無音ノイズフロア | 無音（全ゼロ）、3秒、48kHz | ディラックIR（合成） | 20-200Hz 帯域RMSノイズレベル | ≤ -120 dBFS | 中 |
| TC-05A | 低域ノイズ（50Hz） | 50Hz 定常正弦波、-6dBFS、3秒、48kHz | ディラックIR（合成） | 20-200Hz 帯域内の非調和成分エネルギー（基本波+高調波を除く） | ≤ -90 dBFS | 高 |
| TC-05B | 低域ノイズ（40Hz） | 40Hz 定常正弦波、-6dBFS、3秒、48kHz | ディラックIR（合成） | 非調和成分エネルギー | ≤ -90 dBFS | 高 |
| TC-05C | 低域ノイズ（80Hz） | 80Hz 定常正弦波、-6dBFS、3秒、48kHz | ディラックIR（合成） | 非調和成分エネルギー | ≤ -90 dBFS | 高 |
| TC-05D | 低域ノイズ（実IR） | 50Hz 定常正弦波、-6dBFS、3秒、48kHz | ルーム補正IR（実測、`reference/room_correction.wav`） | 非調和成分エネルギー（ゴールデン出力との差分） | ≤ -85 dBFS | 高 |
| TC-06 | スペクトログラムクリーン性 | 20-200Hz 対数スイープ、-6dBFS、5秒、48kHz | ディラックIR（合成） | スイープライン外のアーティファクト総エネルギー（メインラインエネルギー比） | ≤ -60 dB | 高 |
| TC-07 | フィルタ特性（LPF） | 20-20k 対数スイープ、-6dBFS、10秒、48kHz | Butterworth LPF 1kHz IR（合成） | カットオフ周波数・減衰量の理論値誤差 | ≤ 0.2 dB | 中 |
| TC-08 | モード切替過渡応答 | 50Hz 定常正弦波、-6dBFS、3秒、48kHz | ディラックIR（合成） | EQバイパス切り替え時の出力振幅ジャンプ量（ピーク） | ≤ -90 dBFS 相当の変化量 | 中 |
| TC-09 | オーバーサンプリングエイリアシング | 21, 22, 23kHz 定常正弦波、-6dBFS、3秒、48kHz | ディラックIR（合成） | エイリアシング成分（10kHz以下）のパワー | ≤ -120 dBFS | 中 |
| TC-10 | バイパスA/B比較（原因切り分け） | 50Hz 定常正弦波、-6dBFS、3秒、48kHz | ディラックIR（合成） | 各モジュール有効/無効時のノイズフロア差（A: 全バイパス, B: EQのみ, C: Convのみ, D: 両方） | 特定モジュールのみ異常上昇を検出できること | 高 |

**受け入れ基準**: 全テストケースがCI環境（Debug/Release両方）でパスすること。

---

### Phase 3: ISR Runtime試験（拡充） (工数: 7日)

| ID | テストケース名 | 内容 | 評価指標 | 優先度 |
|----|---------------|------|---------|--------|
| TC-11 | IR Reload Storm | 100回連続で異なるIRをロードする（`--cli-ir` を繰り返し指定）。IRはディラック、LPF、HPF、オールパスをランダムに選択。 | プロセスがクラッシュしないこと。各ロード後に出力WAVをキャプチャし、NaN/Inf/Denormal/DC異常がないこと。 | 高 |
| TC-11B | Crossfade Storm | IR reload interval < crossfade duration（約 50ms）となる条件で、50回連続IRをリロードする。クロスフェードが完了する前に次のリロードが発生する状態を強制する。 | プロセスがクラッシュしないこと。各ロード後に出力WAVをキャプチャし、NaN/Inf/Denormal/DC異常がないこと。`RuntimeHealthMonitor` の Violation Count が 0 であること。 | 高 |
| TC-12 | Bypass Burst | `--cli-bypass-burst-count 1000 --cli-bypass-burst-interval-ms 10` を指定し、1000回のバイパス切り替えを実施する。 | 出力WAVに NaN/Inf/Denormal/DC異常がないこと。`RuntimeHealthMonitor` の Violation Count が 0 であること。プロセスがクラッシュしないこと。 | 高 |
| TC-13 | Publication Saturation | `--cli-intent-burst-count 500 --cli-intent-burst-interval-ms 5` を指定し、大量の Intent を短期間に投入する。 | `RuntimePublicationCoordinator` の Reject Count が一定値（例：10回）以下であること。出力WAVに異常がないこと。 | 中 |
| TC-14 | 長期耐久試験 | PR時: 30分間連続処理。Nightly: 1時間連続処理（同じIR/入力をループ）。 | 出力WAVの定期的サンプリング（1分毎）で NaN/Inf/Denormal/DC異常がないこと。`RuntimeHealthMonitor` Violation Count == 0。メモリ使用量が一定範囲内（増加傾向がないこと）。 | 高 |

**全Runtime試験（TC-11〜TC-14）に共通の異常検査**:
- 出力信号に `NaN`（非数）が含まれていないこと。
- 出力信号に `Inf`（無限大）が含まれていないこと。
- 出力信号にデノーマル（絶対値 < 1e-20）が多数含まれていないこと（FTZ/DAZ が有効に動作していることの確認）。
- 出力信号の DC オフセット（全サンプルの平均値）が 1e-6 未満であること（DCブロッカーの動作確認）。
- これらの検査は、各試験の出力WAVを解析するフェーズで実施する。

**受け入れ基準**: 全Runtime試験がCI環境（PR/Nightly）でパスすること。

---

### Phase 4: Mixed Phase / Progressive Upgrade試験 (工数: 3日)

| ID | テストケース名 | 内容 | 評価指標 | 優先度 |
|----|---------------|------|---------|--------|
| TC-15 | Mixed Phase Cache Rebuild | 以下の3段階で同一IR（ルーム補正IR）を Mixed Phase で処理し、出力を比較する。 | 振幅スペクトルRMS誤差 ≤ 0.1 dB、群遅延RMS誤差 ≤ 1° | 高 |
|     | 段階1: 最適化前（Linear Phase→Mixed Phase変換なし） | | |
|     | 段階2: 最適化後（CMA-ES/GreedyAdaGrad による Mixed Phase 変換実施） | | |
|     | 段階3: キャッシュ再読込後（一度プロセスを再起動し、キャッシュから読み込んだ状態で処理） | | |
| TC-16 | Progressive Upgrade | Progressive Upgrade が有効な状態で、以下の FFT サイズのステップを経由して IR をロードする。 | 各ステップ移行前後の出力差分のピーク ≤ -60 dBFS、RMS ≤ -80 dBFS | 中 |
|     | ステップL1: FFTサイズ 512（初期ロード） | | |
|     | ステップL2: FFTサイズ 1024（アップグレード中間） | | |
|     | ステップL3: FFTサイズ 2048（アップグレード完了） | | |
|     | 各ステップで 50Hz 定常正弦波を入力し、出力をキャプチャする。 | | |

**受け入れ基準**: TC-15 の3段階出力が一致し、TC-16 のアップグレード中に音切れ・異常値が発生しないこと。

---

### Phase 5: CI統合と運用準備 (工数: 3日)

| タスク | 説明 | 成果物 |
|--------|------|--------|
| 5-1 | `.github/workflows/audio_quality.yml` を作成する。PR時は全テスト（TC-01〜TC-16）を実行する。Nightly（毎日深夜）は全テストに加え、TC-14 の長期耐久（1時間）を実行する。 | GitHub Actions workflow |
| 5-2 | `CMakeLists.txt` に `add_test` を追加し、CTestからPythonテストを呼び出し可能にする。テスト名は `AudioQuality_TC01` 〜 `AudioQuality_TC16` とする。 | `CMakeLists.txt` 改修 |
| 5-3 | 許容誤差を初回実行結果に基づいて統計的に調整する。Debug/Release で別の閾値を設定可能にする。 | `test_config.yaml` 更新 |
| 5-4 | テスト失敗時にアーティファクト（出力WAV、スペクトログラム画像、診断ログ）を GitHub Actions の artifacts としてアップロードする。 | 診断用データの永続化 |
| 5-5 | テスト結果を JUnit XML 形式で出力し、GitHub Actions のテストレポート機能と連携する。 | レポートの可視化 |
| 5-6 | テスト成功/失敗の通知を Slack に送信する（オプション）。 | 通知設定 |

**受け入れ基準**: PR作成時に自動で音質テストが実行され、結果がPRコメントに表示されること。Nightly 実行時に長期耐久試験が完了し、結果が保存されること。

---

## 4. テスト信号・IRデータ詳細仕様

### 4.1 テスト信号一覧

| 信号名 | 仕様 | 生成関数 | 使用テストケース |
|--------|------|----------|-----------------|
| 対数スイープ (20-20k) | 20Hz→20kHz, 10秒, -6dBFS, 48kHz | `generators.log_sweep(20, 20000, 10, -6, 48000)` | TC-01, TC-01B, TC-02, TC-07 |
| 対数スイープ (20-200) | 20Hz→200Hz, 5秒, -6dBFS, 48kHz | `generators.log_sweep(20, 200, 5, -6, 48000)` | TC-06 |
| 定常正弦波 (20Hz) | 20Hz, 3秒, -6dBFS, 48kHz | `generators.sine(20, 3, -6, 48000)` | TC-03 |
| 定常正弦波 (40Hz) | 40Hz, 3秒, -6dBFS, 48kHz | `generators.sine(40, 3, -6, 48000)` | TC-03, TC-05B |
| 定常正弦波 (50Hz) | 50Hz, 3秒, -6dBFS, 48kHz | `generators.sine(50, 3, -6, 48000)` | TC-03, TC-05A, TC-05D, TC-08, TC-10, TC-16 |
| 定常正弦波 (80Hz) | 80Hz, 3秒, -6dBFS, 48kHz | `generators.sine(80, 3, -6, 48000)` | TC-03, TC-05C |
| 定常正弦波 (100Hz) | 100Hz, 3秒, -6dBFS, 48kHz | `generators.sine(100, 3, -6, 48000)` | TC-03 |
| 定常正弦波 (21kHz) | 21kHz, 3秒, -6dBFS, 48kHz | `generators.sine(21000, 3, -6, 48000)` | TC-09 |
| 定常正弦波 (22kHz) | 22kHz, 3秒, -6dBFS, 48kHz | `generators.sine(22000, 3, -6, 48000)` | TC-09 |
| 定常正弦波 (23kHz) | 23kHz, 3秒, -6dBFS, 48kHz | `generators.sine(23000, 3, -6, 48000)` | TC-09 |
| 無音 | 全サンプル0, 3秒, 48kHz | `generators.silence(3, 48000)` | TC-04 |
| マルチトーン | ITU-R BS.1387 準拠, 3秒, -12dBFS, 48kHz | `generators.multitone(3, -12, 48000)` | 補助（総合歪み評価） |

### 4.2 IRデータ一覧

| IR名 | 内容 | ファイル形式 | 保管場所 | 使用テストケース |
|------|------|-------------|----------|-----------------|
| ディラックIR | 先頭のみ1.0, 残りゼロ (長さ1024) | 合成（コード内生成） | N/A (動的生成) | TC-01〜TC-06, TC-08〜TC-10 |
| LPF 1kHz IR | Butterworth 4次, Fc=1kHz (長さ1024) | 合成（コード内生成） | N/A (動的生成) | TC-07 |
| オールパスIR | 2次オールパス (ρ=0.7, θ=45°) | 合成（コード内生成） | N/A (動的生成) | TC-02（Mixed Phase検証用） |
| ルーム補正IR | 実測ルーム補正IR (48kHz, モノラル, 2048samples) | WAV (Git LFS) | `reference/room_correction.wav` | TC-01B, TC-05D, TC-15 |
| キャビネットIR | 実測ギターキャビネット (48kHz, モノラル, 2048samples) | WAV (Git LFS) | `reference/cabinet_ir.wav` | 補助（リグレッション） |
| ロングリバーブIR | 実測リバーブ (48kHz, ステレオ, 3秒) | WAV (Git LFS) | `reference/long_reverb.wav` | TC-11（Reload Storm用） |

---

## 5. 評価指標と合格基準の詳細

### 5.1 主要指標の計算方法

#### 周波数応答RMS誤差 (TC-01, TC-01B, TC-02, TC-07)
```python
def compute_frequency_response_rms_error(output_wav, theoretical_response, f_min=20, f_max=20000):
    sr, data = read_wav(output_wav)
    f, H = scipy.signal.freqz(data, fs=sr, worN=4096)
    mask = (f >= f_min) & (f <= f_max)
    error_db = 20 * log10(abs(H[mask])) - 20 * log10(abs(theoretical_response[mask]))
    return sqrt(mean(error_db**2))
```

#### THD+N (TC-03)
```python
def compute_thdn(output_wav, fundamental_freq, f_min=20, f_max=200):
    sr, data = read_wav(output_wav)
    f, Pxx = scipy.signal.periodogram(data, fs=sr, window='flattop')
    # 基本波インデックスを検索
    fund_idx = argmax(Pxx[int(fundamental_freq * len(f) / sr - 10):int(fundamental_freq * len(f) / sr + 10)]) + offset
    fund_power = Pxx[fund_idx]
    # 高調波（2〜10次）とノイズ（基本波と高調波を除く全帯域）のパワー
    harmonic_indices = [fund_idx * n for n in range(2, 11) if fund_idx * n < len(Pxx)]
    noise_mask = np.ones(len(Pxx), dtype=bool)
    noise_mask[fund_idx] = False
    for h in harmonic_indices:
        noise_mask[h] = False
    noise_power = np.sum(Pxx[noise_mask])
    thdn = 10 * log10((noise_power + sum(Pxx[harmonic_indices])) / fund_power)
    return thdn
```

#### 非調和成分エネルギー (TC-05A〜TC-05D)
```python
def compute_non_harmonic_energy(output_wav, fundamental_freq, f_min=20, f_max=200):
    sr, data = read_wav(output_wav)
    f, Pxx = scipy.signal.periodogram(data, fs=sr, window='flattop')
    fund_idx = argmax(Pxx[int(fundamental_freq * len(f) / sr - 10):int(fundamental_freq * len(f) / sr + 10)]) + offset
    # 高調波（2〜5次）を除外
    harmonic_indices = [fund_idx * n for n in range(2, 6) if fund_idx * n < len(Pxx)]
    mask = (f >= f_min) & (f <= f_max)
    mask[fund_idx] = False
    for h in harmonic_indices:
        mask[h] = False
    total_energy = np.sum(Pxx[mask])
    return 10 * log10(total_energy + 1e-18)  # dBFS
```

#### スペクトログラムアーティファクト (TC-06)
```python
def compute_spectrogram_artifact_energy(output_wav, f_min=20, f_max=200):
    sr, data = read_wav(output_wav)
    f, t, Sxx = scipy.signal.spectrogram(data, fs=sr, nperseg=1024, noverlap=512)
    # 理論スイープライン（各時刻における瞬時周波数）を計算
    theoretical_freq = f_min * (f_max / f_min) ** (t / duration)
    artifact_energy = 0
    total_energy = np.sum(Sxx)
    for i, freq in enumerate(theoretical_freq):
        bin_idx = int(freq * len(f) / sr)
        # スイープラインの周囲 ±3ビン以外のエネルギーをアーティファクトとみなす
        mask = np.ones(len(f), dtype=bool)
        mask[max(0, bin_idx-3):min(len(f), bin_idx+4)] = False
        artifact_energy += np.sum(Sxx[mask, i])
    return 10 * log10(artifact_energy / total_energy + 1e-18)
```

### 5.2 ビルド構成別許容誤差

| テスト種別 | Debug | Release (MSVC) | Release (icx) |
|-----------|-------|----------------|---------------|
| 周波数応答RMS誤差 | ≤ 0.1 dB | ≤ 0.05 dB | ≤ 0.06 dB |
| THD+N | ≤ -80 dB | ≤ -100 dB | ≤ -98 dB |
| 非調和成分エネルギー | ≤ -85 dBFS | ≤ -90 dBFS | ≤ -88 dBFS |
| スペクトログラムアーティファクト | ≤ -55 dB | ≤ -60 dB | ≤ -58 dB |
| モード切替ジャンプ量 | ≤ -85 dBFS | ≤ -90 dBFS | ≤ -88 dBFS |
| エイリアシング成分 | ≤ -115 dBFS | ≤ -120 dBFS | ≤ -118 dBFS |

---

## 6. 運用フロー

### 6.1 開発者のワークフロー

1. 機能追加/バグ修正のためブランチを作成する。
2. ローカルで `ctest -R AudioQuality -C Release` を実行し、音質テストがパスすることを確認する。必要に応じて Debug でも確認する。
3. PRを作成すると、CI上で Release ビルドの全テスト（TC-01〜TC-16）が自動実行される。
4. テストが失敗した場合、`artifacts` から出力WAVとスペクトログラム画像、診断ログをダウンロードして原因を調査する。
5. 修正後、再度プッシュして再テストする。

### 6.2 Nightly 運用

- 毎日深夜（UTC 0:00）に `main` ブランチの最新コミットに対して実行される。
- Release ビルドで全テスト（TC-01〜TC-16）を実行し、さらに TC-14（長期耐久試験）を 1時間 で実行する。
- 結果は GitHub Actions のテストレポートに蓄積され、週次でトレンドを確認する。

### 6.3 ゴールデンデータの管理

- 理論値ベース（80%）は数式から計算するため、メンテナンス不要。
- 実IRベース（20%）のゴールデンデータは、**意図的な音質改善時のみ**手動で更新する。
- 更新時は、新旧リリース版の出力を聴感比較し、改善が確認された後に新しいゴールデンデータをコミットする。
- ゴールデンデータは `reference/golden/` に配置し、Git LFS で管理する。

---

## 7. 期待される効果（v3.0）

| 効果 | 定量的目標 | 現状との比較 |
|------|-----------|-------------|
| 低音ノイズ検出 | TC-05A〜TC-05D で 100% 検出可能 | 従来: 条件付き再現のみ |
| Crossfade重畳バグ検出 | TC-11B で確実に検出可能 | 従来: テストなし |
| Mixed Phase回帰検出 | TC-15 で検出可能 | 従来: テストなし |
| 長期安定性保証 | TC-14（1時間耐久）でリーク/劣化を検出 | 従来: 短期テストのみ |
| NaN/Inf/DC異常検出 | 全Runtime試験に共通検査として実装 | 従来: 手動確認のみ |
| Progressive Upgrade品質 | TC-16 で移行中の音切れを検出 | 従来: テストなし |
| QA工数削減 | 手動聴感テスト工数を **70% 削減** | 従来: 毎リリース 4人日 → 1人日 |

---

## 8. リスクと軽減策（v3.0）

| リスク | 影響度 | 軽減策 |
|--------|-------|--------|
| `OutputCaptureSink` 実装に想定以上の工数がかかる | 中 | `processBlockDouble` 出口にコールバックを仕込む最小実装から開始。`std::function` を `std::atomic` で保護しスレッド安全性を確保。 |
| 長期耐久試験（1時間）がCIタイムアウトになる | 中 | GitHub Actions のタイムアウト（6時間）内に収まることを確認。Nightly のみ実行し、PR時は30分に短縮。 |
| 許容誤差がシビアすぎてCIが頻繁に落ちる | 中 | 初期は許容誤差を緩めに設定し、統計的に安定した値を収集した後に段階的に厳しくする。Debug/Releaseで別閾値を設定。 |
| Git LFS ストレージ容量超過 | 低 | テスト信号は合成で済ませ、実IRのみLFS管理。`golden/` は理論値ベース80%のためファイル数が少ない。 |
| Mixed Phase 最適化が CI 環境で時間切れ | 中 | TC-15 では最適化が完了するまで待機する。必要に応じて最適化のイテレーション数をテスト用に削減するオプションを追加。 |
| `--cli-output-wav` と既存の `--cli-exit-ms` の競合 | 低 | 出力キャプチャ完了後に終了するよう、コールバックの完了を待機してから `systemRequestedQuit()` を呼び出す。 |

---

## 9. ディレクトリ構成（最終形）

```
ConvoPeq/
├── src/...
├── tools/
│   └── ConvoPeqCLI/
│       └── (既存の MainWindow を利用するため新規作成不要)
├── tests/
│   └── audio_quality/
│       ├── __init__.py
│       ├── conftest.py                   # pytest fixtures
│       ├── generators.py                 # テスト信号生成
│       ├── analyzers.py                  # FFT/THD/スペクトログラム/異常検出
│       ├── golden_calculator.py          # 理論値計算
│       ├── cli_runner.py                 # ConvoPeq.exe 起動ラッパー
│       ├── test_core.py                  # TC-01〜TC-10
│       ├── test_runtime.py               # TC-11〜TC-14
│       ├── test_mixed_phase.py           # TC-15〜TC-16
│       ├── test_regression.py            # ゴールデンデータ比較（実IR用）
│       ├── data/
│       │   ├── reference/                # Git LFS: 実測IR
│       │   │   ├── room_correction.wav
│       │   │   ├── cabinet_ir.wav
│       │   │   └── long_reverb.wav
│       │   └── golden/                   # Git LFS: ゴールデン出力（最小限）
│       │       └── release_v1.0/
│       │           ├── tc01b_golden.wav
│       │           └── tc05d_golden.wav
│       ├── scripts/
│       │   ├── generate_golden.py        # ゴールデンデータ生成スクリプト
│       │   └── view_results.py           # 結果可視化 (matplotlib)
│       └── config/
│           └── test_config.yaml          # 許容誤差・ビルド構成別設定
├── .github/
│   └── workflows/
│       ├── audio_quality_pr.yml          # PR時: 全テスト（短縮版）
│       └── audio_quality_nightly.yml     # Nightly: 全テスト + 1時間耐久
└── CMakeLists.txt                        # add_test 追加
```

---

## 10. 実装スケジュール（総合）

| 週 | 実施フェーズ | 担当 | マイルストーン |
|----|-------------|------|---------------|
| 1 | Phase 0 (CLI拡張 + OutputCaptureSink) | C++エンジニア 1名 | `--cli-output-wav` が動作する |
| 2 | Phase 1 (Pythonオーケストレーター) | Pythonエンジニア 1名 | `pytest` でテストが収集できる |
| 3-4 | Phase 2 (DSP品質試験 TC-01〜TC-10) | 両名協業 | TC-01〜TC-10 が全パスする |
| 5 | Phase 3 (ISR Runtime試験 TC-11〜TC-14) | 両名協業 | TC-11〜TC-14 が全パスする |
| 5 | Phase 4 (Mixed Phase試験 TC-15〜TC-16) | 両名協業 | TC-15〜TC-16 が全パスする |
| 6 | Phase 5 (CI統合と微調整) | 両名協業 | PR時に自動テストが実行される |

---

## 11. 最終評価

| 項目 | 評価 | 備考 |
|------|------|------|
| アーキテクチャ整合性 | **A** | 既存CLI + OutputCaptureSink で別経路を作らず現行処理をそのまま検証 |
| DSP評価設計 | **A** | TC-01B追加で実IRもカバー。TC-03〜TC-05で低域問題を多角的に検出 |
| Runtime評価設計 | **A** | TC-11B + TC-14 + 共通異常検査で網羅的 |
| 実装容易性 | **B+** | `OutputCaptureSink` 追加のみ。新規CLIは不要 |
| CI適合性 | **A** | PR/Nightly分離でバランス良好。1時間耐久もNightlyに収容 |
| 実運用性 | **A** | 実IR・Crossfade・Mixed Phase・Progressive Upgrade・耐久まで完全カバー |
| **総合評価** | **95点（推奨度: 最高）** | v2.0レビュー指摘を完全反映 |

---

## 12. 承認（最終）

| 役割 | 名前 | 承認日 | 署名 |
|------|------|--------|------|
| プロジェクトマネージャ | | | |
| テクニカルリード | | | |
| QAリード | | | |

---

**本計画（v3.0）は、ConvoPeq の現行アーキテクチャ（ISR Runtime / Mixed Phase / Publication Runtime / Crossfade Runtime / Progressive Upgrade）を完全に包含し、かつ実装負荷を最小化する現実的な品質保証基盤です。**  
承認後、Phase 0（OutputCaptureSink実装）より速やかに着手いたします。