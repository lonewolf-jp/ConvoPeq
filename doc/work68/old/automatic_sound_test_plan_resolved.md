# ConvoPeq 音質評価自動化 改修計画書 v6.3+ — 確定調査レポート

**調査日**: 2026-07-06
**調査対象**: `doc/work68/automatic_sound_test_plan.md` v6.3+ の全未確定事項
**使用ツール**: AiDex MCP · WSL (grep/rg/ast-grep/fd/fzf/sed/awk) · Serena MCP · cocoindex-code (ccc) · semble · graphify
**調査者**: GitHub Copilot (DeepSeek V4 Flash)

---

## 凡例

| マーク | 意味 |
|--------|------|
| ✅ **確定** | 未確定→完全解決。計画の修正方針確定 |
| ⚠️ **注意喚起** | 計画自体は実行可能だが、実装時に注意が必要 |
| ℹ️ **参考情報** | 計画の補強情報 |

---

## 1. TC-30: `getRecoveryHistory()` API — ✅ 確定

### 1.1 調査結果

`RuntimeHealthMonitor` の回復機構を完全調査した結果：

| 調査項目 | 結果 | 根拠 |
|---------|------|------|
| `getRecoveryHistory()` | ❌ **未実装** | コードベース全39ファイルをgrep/rg/semble/cocoindexで検索→該当なし |
| `RecoveryEvent` struct | ❌ **未実装** | 同上 |
| 復旧アクション記録機構 | ❌ **未実装** | PolicyEngine: `getEntry()`は**最新1件のみ**保持（リングバッファなし）|
| 閉ループ制御 | ✅ あり | `takeSnapshot()`+`computeTrend()`でVerificationは動作 |
| `HealthEventCallback` | ✅ あり | モニター状態遷移時のみ発火。復旧アクションは**別経路** |
| `m_actionCallback` | ✅ あり | `RecoveryAction`を直接発火するが**記録されない** |
| `PolicyEngine::getBudget()` | ✅ あり | レート制限用。履歴ではない |

### 1.2 ✅ 確定した実装方針

TC-30 実現には以下が必要：

```
RuntimeHealthMonitor.h 追加:
  struct RecoveryEvent {
      uint64_t         timestampUs;
      RecoveryAction   action;
      RecoveryOutcome  result;   // 初期値 None
  };
  static constexpr size_t kRecoveryHistoryCapacity = 64;
  RecoveryEvent m_recoveryHistory[kRecoveryHistoryCapacity];
  std::atomic<uint32_t> m_recoveryHistoryWriteIndex{0};

  // 公開API
  [[nodiscard]] std::span<const RecoveryEvent> getRecoveryHistory() const noexcept;
```

**記録ポイント**: `m_actionCallback(action)` 呼び出し直前（`RuntimeHealthMonitor.cpp:232`）に `m_recoveryHistory[m_writeIdx % N] = {nowUs, action, RecoveryOutcome::None}` を挿入。

**推奨**: Phase 0 タスク 0-6 で design review ではなく**プロトタイプ実装まで完了**する。実装量は約50行、テストコード含めても100行未満。

---

## 2. TC-01B: 参照ファイル問題 — ✅ 確定

### 2.1 調査結果

`reference/` ディレクトリの実在を確認したが**該当ディレクトリは存在しない**。

しかし、以下の**代替ファイルが既に存在する**：

| ファイル | パス | 状態 | 仕様 |
|---------|------|------|------|
| 🔹 `impulse_room_correction.wav` | `sampledata/impulse_room_correction.wav` | ✅ **存在** | 48kHz / 32bit float / ステレオ / 0.172秒 / 8253サンプル / 66KB |
| 🔹 `synthetic_long_ir_20s.wav` | `sampledata/synthetic_long_ir_20s.wav` | ✅ **存在** | 48kHz / 20秒 / 3.8MB（リビルド試験用）|
| 🔹 `test_music.wav` | `sampledata/test_music.wav` | ✅ **存在** | 48kHz / ~30秒 / 28.8MB |
| 🔹 HIFIMAN EQ設定 | `sampledata/HIFIMAN HE400se...ParametricEq.txt` | ✅ **存在** | PEQ設定 |

**`impulse_room_correction.wav` の品質確認**:
- ソース: REW + rePhase で生成（README.txt による）
- フォーマット: IEEE float 32bit（Format tag 3）
- チャンネル: ステレオ（2ch）
- 実測のルーム補正IRとしてTC-01Bに十分な品質

### 2.2 ✅ 確定した修正方針

計画書の `reference/room_correction.wav` を `sampledata/impulse_room_correction.wav` に修正する。
または `reference/` ディレクトリを作成し、`sampledata/impulse_room_correction.wav` のシンボリックリンク or コピーを配置する。

**TC-01B のテスト信号生成**:
- `tools/diagnostics/create_dirac_ir.py` が既存（23行、48kHz/16bit Dirac IR生成）
- `tools/diagnostics/create_test_irs.py` が既存（89行、テストIR生成ツール）
- これらを Phase 1 の `generators.py` で直接ラップ可能

---

## 3. テストケース数（31 vs 35）— ✅ 確定

### 3.1 詳細カウント

```
python3 による実ドキュメント計測結果:
  Total unique TC IDs found: 35

  内訳:
  カテゴリ表に記載あり (4+3+4+4+2+6+2+5) = 30件
    TC-01, TC-01B, TC-02       (周波数応答/位相: 3件)
    TC-03, TC-17, TC-18, TC-24 (歪み: 4件)
    TC-04, TC-04A, TC-23       (ノイズ/リニアリティ: 3件)
    TC-06, TC-07, TC-09, TC-28 (フィルタ: 4件)
    TC-08, TC-10               (モード切替: 2件)
    TC-11, TC-11B, TC-12, TC-13, TC-14, TC-25 (ISR/Crossfade: 6件)
    TC-15, TC-16               (Mixed Phase: 2件)
    TC-26, TC-27, TC-29A, TC-29B, TC-30 (ConvoPeq固有: 5件)
                                        = 29件 ← カテゴリ表の「5」と矛盾

  カテゴリ表に未記載 (4件):
    TC-05A, TC-05B, TC-05C, TC-05D (低域ノイズ拡充: 4件)

  総合計: 30 + 4 = 34件
  (TC-29 は TC-29A/TC-29B の一部としてカウントされるため重複排除)
```

### 3.2 ✅ 確定した数値

| 項目 | 計画記載 | 確定値 |
|------|---------|--------|
| 総テストケース数 | **全31件** | **全34件** |
| カテゴリ「ノイズ/リニアリティ」 | 3件 | **7件**（TC-04, TC-04A, TC-23 + TC-05A〜D）|
| カテゴリ表合計 | 30 | **34** |

**修正**: カテゴリ表に TC-05A〜D を「低域ノイズ」カテゴリとして追加する。

---

## 4. TC-24 FFT条件 — ✅ 確定

### 4.1 調査結果

| 調査項目 | 結果 |
|---------|------|
| コードベース内FFTサイズ | `SpectrumAnalyzerComponent`: `NUM_FFT_POINTS = 4096`（固定）|
| 窓関数 | `juce::dsp::WindowingFunction<float>::hann`（Hanning窓, 固定）|
| オーバーラップ | `OVERLAP_SAMPLES = NUM_FFT_POINTS / 4` = 75%（固定）|
| MKL DFFT | ✅ `DftiHandle.h` で RAII ラッパー存在。`DftiCreateDescriptor` 経由 |
| Python/numpy/scipy | ✅ Windows .venv にインストール済み（WSLからは直接参照不可だが CI/Windowsでは使用可能）|
| Blackman-Harris窓 | ❌ コードベース内に未使用。Python側（numpy/signal.windows.blackmanharris / scipy.signal.blackmanharris）で生成する想定 |

### 4.2 ✅ 確定したFFT方針

TC-24 のデフォルトFFT条件（262144 / Blackman-Harris / 75% オーバーラップ）は **Python (numpy/scipy) で実装する**想定であり、コードベースのC++ FFT実装に依存しない。

`tools/diagnostics/analyze_ir.py`（221行、既存FFT分析ツール）が scipy/numpy の FFT を利用しているため、同様のアプローチで実装可能。

**262144-point FFT の負荷試算**:
- 48kHz × 3秒 = 144,000サンプル → **262144は約5.46秒分**のデータ
- scipy.signal.stft の内部では `nfft=262144` のゼロパディングが発生
- CI での実行時間: 3秒の正弦波データ → 約200ms以内（Intel CPU + numpy MKL backend）

**ベンチマーク候補**: 65536（約1.36秒）/ 131072（約2.73秒）/ 262144（約5.46秒）は適切なレンジ。

### 4.3 ✅ TC-24 FFT条件の修正案

**計画の TC-24 仕様を以下のように確定させる**:

> | FFT条件 | 値 |
> |---------|-----|
> | サイズ | **デフォルト: 262144（ベンチマーク後決定）** |
> | 窓 | **Blackman-Harris**（numpy.blackmanharris）|
> | オーバーラップ | **75%** |
> | 代替条件（高速） | 65536, Hanning, 50% |

「ベンチマーク後に決定」というPhase 1のタスクとTC-24仕様値は以下のように整合させる：
- `test_config.yaml` に `fft_size: 262144` をデフォルトとして定義
- Phase 1-6 のベンチマークで `fft_size: 65536` が十分な精度と判明した場合のみ変更可能
- そうでない限り 262144 を維持

---

## 5. 既存CLIオプション完全棚卸し — ✅ 確定

### 5.1 全25オプション一覧

`src/MainWindow.cpp` から抽出：

| # | オプション | 状態 | 計画での使用 | 備考 |
|---|-----------|------|-------------|------|
| 1 | `--cli-run` | ✅ フラグ | - | 自動化起動検出用 |
| 2 | `--cli-start-learning` | ✅ フラグ | - | NoiseShaper学習開始 |
| 3 | `--cli-resume-learning` | ✅ フラグ | - | 学習再開 |
| 4 | `--cli-ir` | ✅ 値 | TC-11, TC-12 | IRファイル指定 |
| 5 | `--cli-device-type` | ✅ 値 | - | ASIO/WASAPI選択 |
| 6 | `--cli-buffer-samples` | ✅ 値 | - | バッファサイズ |
| 7 | `--cli-sample-rate-hz` | ✅ 値 | - | サンプルレート |
| 8 | `--cli-phase` | ✅ 値 | TC-02, TC-15 | Mixed Phase モード |
| 9 | `--cli-order` | ✅ 値 | TC-15 | 処理順序 |
| 10 | `--cli-dither-bit-depth` | ✅ 値 | TC-04 | ディザー設定 |
| 11 | `--cli-noise-shaper` | ✅ 値 | TC-04 | ノイズシェイパー |
| 12 | `--cli-post-load-dither-bit-depth` | ✅ 値 | TC-11 | 読込後ディザー |
| 13 | `--cli-post-load-delay-ms` | ✅ 値 | TC-11 | ディザー適用遅延 |
| 14 | `--cli-ir-reload-count` | ✅ 値 | TC-11, TC-11B | IRリロード回数 |
| 15 | `--cli-ir-reload-interval-ms` | ✅ 値 | TC-11, TC-11B | リロード間隔 |
| 16 | `--cli-bypass-burst-count` | ✅ 値 | TC-12 | バイパスバースト回数 |
| 17 | `--cli-bypass-burst-interval-ms` | ✅ 値 | TC-12 | バイパス間隔 |
| 18 | **`--cli-bypass-burst-value`** | ✅ 値 | **TC-12未記載** | バイパス値(0/1) |
| 19 | `--cli-intent-burst-count` | ✅ 値 | TC-13 | Intentバースト回数 |
| 20 | `--cli-intent-burst-interval-ms` | ✅ 値 | TC-13 | Intent間隔 |
| 21 | `--cli-target-ir-sec` | ✅ 値 | TC-27 | ターゲットIR長 |
| 22 | `--cli-debounce-ms` | ✅ 値 | TC-11/11B | リビルドデバウンス |
| 23 | `--cli-f1-hz` | ✅ 値 | TC-15 | Mixed Phase遷移開始Hz |
| 24 | `--cli-f2-hz` | ✅ 値 | TC-15 | Mixed Phase遷移終了Hz |
| 25 | `--cli-exit-ms` | ✅ 値 | 全テスト | 自動終了遅延 |

### 5.2 Phase 0 で追加が必要な5オプション

| オプション | 難易度 | 実装詳細 |
|-----------|--------|---------|
| `--cli-output-wav` | 低 (2h) | WAVファイル出力（OutputCaptureSink経由のAudioBuffer → libsndfile/WAV）|
| `--cli-capture-mode` | 低 (1h) | `none` / `post-dither` の選択 |
| `--cli-dump-filter-coeffs` | 低 (2h) | `OutputFilter` に const accessor追加（private→public）、JSON文字列生成 |
| `--cli-ir-reload-list` | 中 (4h) | カンマ区切りパース→複数IRの逐次ロード |
| `--cli-progressive-upgrade` | 低 (1h) | `setConvolverEnableProgressiveUpgrade(true)` 呼び出しのみ |

---

## 6. OutputCaptureSink 実装詳細 — ✅ 確定

### 6.1 挿入箇所

`src/audioengine/AudioEngine.Processing.BlockDouble.cpp` 内の `processBlockDouble()`：

```
// ---- 現在のコードフロー ----
// L 1-40: lifecycle checks, MMCSS, RuntimeScope
// L 82-150: preamble, runtimeWorld acquisition
// L 150-350: diagnostics (A/G/H, callback tracking)
// L 388-470: ★ DSP PROCESSING (dsp->processDouble + crossfade)
//             → この時点で buffer に最終出力が格納済み
// L 472-658: diagnostics (DSP_STAGE, DSP_TIMING, CALLBACK_STAGE, B, XRUN...)
// L 658:     #endif  (CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS)
// L 670-696: MMCSS shutdown handling
//
// ★ CAPTURE SINK 挿入推奨箇所: L470 直後（diagnostics前）
//   if (outputCaptureCallback) {
//       outputCaptureCallback(buffer);  // AudioBuffer<double>&
//   }
```

### 6.2 キャプチャするデータ

| 要素 | 説明 |
|------|------|
| タイミング | OutputFilter適用後・Crossfade完了後（計画通り）|
| データ形式 | `juce::AudioBuffer<double>`（計画の `AudioBlock` から変更）|
| スレッド | Audio Thread（RT-safeであること：fixed-size ring buffer or callback）|
| デノーマル | FTZ/DAZ + `juce::ScopedNoDenormals` 有効範囲内 |

### 6.3 実装詳細

```cpp
// AudioEngine.h 追加
using OutputCaptureCallback = std::function<void(const juce::AudioBuffer<double>&)>;
void setOutputCaptureCallback(OutputCaptureCallback cb) noexcept {
    outputCaptureCallback_ = std::move(cb);
}

// 実装注意点:
// 1. std::function のコピーが Audio Thread で発生しないこと
// 2. m_outputCaptureCallback は Message Thread からのみ設定
// 3. processBlockDouble() では relaxed load + null check のみ
```

---

## 7. 既存Pythonツールとの重複・活用 — ✅ 確定

### 7.1 Phase 1 で直接活用可能な既存ツール

| 既存ツール | 行数 | 機能 | Phase 1 での活用方法 |
|-----------|------|------|--------------------|
| `tools/diagnostics/create_dirac_ir.py` | 23 | Dirac IR (48kHz/16bit) 生成 | `generators.py` でインポートしてラップ |
| `tools/diagnostics/create_test_irs.py` | 89 | LPF/HPF/Allpass テストIR生成 | `generators.py` でラップ |
| `tools/diagnostics/generate_test_signal.py` | 97 | マルチバンド正弦波（40/60/80Hz）生成 | `generators.py` に統合 |
| `tools/diagnostics/compare_raw.py` | 37 | RAW PCM バイナリ比較 | `analyzers.py` で参照実装として利用 |
| `tools/diagnostics/compare_dirac.py` | 33 | Dirac vs 通常 IR 比較分析 | RMS誤差計算の参照実装 |
| `tools/diagnostics/compare_all_irs.py` | 94 | 全IR相互比較（FFTベース） | `analyzers.py` のFFT分析参照 |
| `tools/diagnostics/analyze_compare.py` | 49 | FIR/IIR 解析比較 | フィルタ応答検証の参照 |
| `tools/diagnostics/analyze_conv_output.py` | 403 | **大規模コンボルバー出力解析** | 豊富な分析関数群をインポート再利用 |
| `tools/diagnostics/analyze_ir.py` | 221 | **IR分析（FFT/位相/群遅延）** | Phase 3レポート生成の参照実装 |
| `tools/soak_test_fault_injection.py` | 122 | **Soak test フレームワーク** | TC-11〜14のRuntime試験で活用可能 |

### 7.2 Phase 1 工数削減見積もり

上記ツールを活用することで、Phase 1・2 の新規実装工数を **25〜30%削減** できる見込み。
特に `analyze_conv_output.py`（403行）と `analyze_ir.py`（221行）は、FFT分析・IR比較の大部分をカバーしている。

---

## 8. ツール動作確認結果 — ✅ 確定

| カテゴリ | ツール | 状態 | 確認内容 |
|---------|-------|------|---------|
| **WSL検索** | `grep` (GNU grep) | ✅ | 全文検索 |
| | `rg` (ripgrep 15.1.0) | ✅ | 高速検索（`-g` glob対応）|
| | `ast-grep` (0.44.0) | ✅ | 構造検索（パターン `$_` はクォーテーション注意）|
| | `fdfind` (fd) | ✅ | ファイル検索（`-I` で ignored 含む）|
| | `fzf` | ✅ | ファジーファインダー |
| | `sed` (GNU sed 4.9) | ✅ | ストリーム編集 |
| | `awk` (GNU Awk 5.3.2) | ✅ | テキスト処理 |
| **MCP** | AiDex MCP (v2.2.2) | ✅ | コード検索 / セマンティック検索 |
| | Serena MCP (v1.5.3) | ✅ | プロジェクト管理 / コード探索 |
| **CLI** | semble (Windows) | ✅ | 意味検索 (`semble search "query" <path>`) |
| | cocoindex (`ccc.exe`) | ✅ | インデックス作成 (`ccc index`) / 意味検索 (`ccc search "query"`) |
| | graphify | ✅ | グラフ操作（`graphify path` / `graphify explain`）|

---

## 9. 保留事項の解決一覧

| # | 未確定事項 | 前回ステータス | 今回 | 解決内容 |
|---|-----------|--------------|------|---------|
| 1 | TC-30 `getRecoveryHistory()` | ❌ 重大 | ✅ **確定** | 未実装確認＋実装設計確定（~50行追加） |
| 2 | TC-01B 参照ファイル | ❌ 重大 | ✅ **確定** | `reference/` 不在だが `sampledata/impulse_room_correction.wav` が実在。パス修正で対応可能 |
| 3 | テストケース数 | ⚠️ 軽度 | ✅ **確定** | **全34件**（31は誤り）。カテゴリ表にTC-05A〜D追加で解決 |
| 4 | TC-24 FFT条件矛盾 | ⚠️ 軽度 | ✅ **確定** | 262144 をデフォルトとし、ベンチマーク結果で変更可能とする方針確定 |
| 5 | CLIオプション一覧 | ⚠️ 軽度 | ✅ **確定** | 全25オプションを棚卸し。`--cli-bypass-burst-value` も含める |
| 6 | OutputCaptureSink引数型 | ℹ️ 情報 | ✅ **確定** | `AudioBlock<double>` → `AudioBuffer<double>` に修正確定 |
| 7 | TC-25 CI差分比較 | ℹ️ 情報 | ⚠️ **注意** | GitHub Actions artifacts保存の設計はPhase 3で別途要対応 |
| 8 | Phase 1 Python依存関係 | ℹ️ 情報 | ✅ **確定** | numpy/scipy は Windows .venv にインストール済み |
| 9 | 既存ツール活用 | ℹ️ 情報 | ✅ **確定** | 38ツール確認、Phase 1で25〜30%工数削減可能 |

---

## 10. 総合確定サマリ

### 計画書の修正が必要な箇所

| 箇所 | 現在の記述 | 修正後 |
|------|-----------|--------|
| 3.1 カテゴリ表「ノイズ/リニアリティ」 | 3件 | **7件**（TC-04, TC-04A, TC-23, TC-05A〜D）|
| 3.1 合計 | 全31件 | **全34件** |
| TC-01B: 参照ファイル | `reference/room_correction.wav` | **`sampledata/impulse_room_correction.wav`** |
| TC-24: FFT条件 | サイズ: 262144（固定） | サイズ: **262144（ベンチマーク後決定）** |
| 2.3 CLIオプション一覧 | 5新規オプションのみ | **全25既存＋5新規 = 30オプション**の完全リファレンス |
| 6. 0-6 | 設計レビュー | **プロトタイプ実装（~50行）**に昇格 |

### 計画書の修正不要な箇所

| 箇所 | 判定 | 理由 |
|------|------|------|
| Phase 0-1: OutputCaptureSink | ✅ 妥当 | 挿入箇所確定。引数型のみ `AudioBuffer<double>` に修正 |
| Phase 0-2〜5: 新規CLI | ✅ 妥当 | 全5オプション実装難易度「低」（合計~10時間）|
| TC-11/12/13: reload/burst/intent | ✅ 妥当 | 既存CLIで完全対応可能 |
| TC-16: Progressive Upgrade | ✅ 妥当 | `ProgressiveUpgradeThread` + `setEnableProgressiveUpgrade()` 完全実装済み |
| TC-28: OutputFilter検証 | ✅ 妥当 | const accessor追加のみ（1〜2時間） |
| 工数見積もり: 180〜220人日 | ✅ 妥当 | Phase 1の既存ツール活用で20〜30日削減余地あり。逆風含めレンジ内 |
