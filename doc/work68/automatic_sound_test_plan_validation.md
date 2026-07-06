# ConvoPeq 音質評価自動化 改修計画書 v6.3+ 妥当性検証レポート

**検証日**: 2026-07-06
**検証対象**: `doc/work68/automatic_sound_test_plan.md` v6.3+
**検証方法**: ソースコード全件調査（AiDex MCP, WSL grep/rg, Serena MCP, 実装ファイル直接レビュー）

---

## 凡例

| 記号 | 意味 |
|------|------|
| ✅ 問題なし | 計画内容と実態が一致、または実装可能 |
| ⚠️ 軽度の問題 | 修正推奨だが計画自体は実行可能 |
| ❌ 重大な問題 | 計画の修正が必須 |

---

## 1. Phase 0: CLI拡張 + OutputCaptureSink の検証

### 1-1 ✅ OutputCaptureSink (`setOutputCaptureCallback()`)

**計画**: `AudioEngine.h` に `setOutputCaptureCallback()` を追加し、`processBlockDouble()` の出口（OutputFilter適用後・Crossfade完了後）にコールバックを設置する。

**現状確認**:
- ✅ `OutputCaptureSink` / `setOutputCaptureCallback` / `outputCaptureCallback` はコードベースに**存在しない** → Phase 0 での新規実装が必要。
- ✅ `processBlockDouble()` (`src/audioengine/AudioEngine.Processing.BlockDouble.cpp`) は存在し、`juce::AudioBuffer<double>& buffer` を最終出力として保持。関数末尾で全処理が完了した状態のバッファを取得可能。
- ✅ クロスフェードミキシングは `dsp->processDouble()` 呼び出し**後**に行われる（L388-455）。OutputFilter は `dsp->processDouble()` 内部 (`DSPCoreDouble.cpp:503`) で適用される。したがって、`processBlockDouble()` 末尾でのキャプチャは「OutputFilter適用後・Crossfade完了後」を正しく捉えられる。

**注意点**:
- 計画は `juce::dsp::AudioBlock<double>&` をコールバック引数としているが、`processBlockDouble()` が扱うのは `juce::AudioBuffer<double>&` である。実際の実装では `AudioBlock` から `AudioBuffer` への変換、または `AudioBuffer` ベースのコールバックとする必要がある。

### 1-2 ✅ 新規CLIオプション

| オプション | 現状 | 備考 |
|-----------|------|------|
| `--cli-output-wav` | ❌ 未実装 | Phase 0 で実装が必要 |
| `--cli-capture-mode` | ❌ 未実装 | Phase 0 で実装が必要 |
| `--cli-dump-filter-coeffs` | ❌ 未実装 | Phase 0 で実装が必要 |
| `--cli-ir-reload-list` | ❌ 未実装 | Phase 0 で実装が必要 |
| `--cli-progressive-upgrade` | ❌ 未実装 | Phase 0 で実装が必要 |

### 1-3 ✅ 既存CLIオプションの確認（計画で使用）

| オプション | 現状 | 計画での使用 |
|-----------|------|-------------|
| `--cli-ir` | ✅ 実装済み | TC-11 |
| `--cli-ir-reload-count` | ✅ 実装済み | TC-11, TC-11B |
| `--cli-ir-reload-interval-ms` | ✅ 実装済み | TC-11, TC-11B |
| `--cli-bypass-burst-count` | ✅ 実装済み | TC-12 |
| `--cli-bypass-burst-interval-ms` | ✅ 実装済み | TC-12 |
| `--cli-intent-burst-count` | ✅ 実装済み | TC-13 |
| `--cli-intent-burst-interval-ms` | ✅ 実装済み | TC-13 |
| `--cli-exit-ms` | ✅ 実装済み | 全テスト |

### 1-4 ⚠️ `--cli-bypass-burst-value` の記載漏れ

**既存CLI** には `--cli-bypass-burst-value` オプションが存在する（TC-12 でバイパス値の指定に使用可能）が、計画書のCLIオプション一覧に記載がない。TC-12 のテスト手順で言及しておくことが望ましい。

### 1-5 ✅ `setConvolverEnableProgressiveUpgrade()` / `ProgressiveUpgradeThread`

- ✅ `AudioEngine::setConvolverEnableProgressiveUpgrade()` → `ConvolverProcessor::setEnableProgressiveUpgrade()` の委譲パスが確立済み
- ✅ `ProgressiveUpgradeThread` は完全実装済み（`ProgressiveUpgradeThread.h`, `ProgressiveUpgradeThread.cpp`）
- ✅ `runCommandLineAutomation()` 内で `setConvolverEnableProgressiveUpgrade(false)` は呼ばれている

---

## 2. ❌ TC-30: Runtime Recovery Verification — 要API新規実装（重大）

**計画**: `RuntimeHealthMonitor::getRecoveryHistory()` が `RecoveryEvent { timestamp, action, result }` のリストを返すことを前提としている。

**現状**:
| 要素 | 状態 | 詳細 |
|------|------|------|
| `getRecoveryHistory()` | ❌ **存在しない** | コードベースの全ファイルを検索したが該当なし |
| `RecoveryEvent` struct | ❌ **存在しない** | 同上 |
| `RecoveryAction` enum | ✅ 存在する | 6レベル階層 (Observe/Throttle/Recover/Restore/Safe/Critical) — `RuntimePolicyEngine.h:51` |
| `RecoveryOutcome` enum | ✅ 存在する | 4状態 (None/Improving/Recovered/Stalled/Worsening) — `RuntimePolicyEngine.h:61` |
| `HealthEvent` struct | ✅ 存在する | timestampUs, severity, eventCode, value, slot を保持 — `RuntimeHealthMonitor.h:18` |
| `HealthEventCallback` | ✅ 存在する | `std::function<void(const HealthEvent&)>` — `RuntimeHealthMonitor.h:98` |
| `RuntimeHealthMonitor::tick()` | ✅ 存在する | 16種類のチェック関数を呼び出す |
| Recovery Action callback | ✅ 存在する | `setActionCallback(RecoveryActionCallback)` — `RuntimeHealthMonitor.h:152` |

**影響**: TC-30 の実装には以下が必要:
1. 新しい型 `RecoveryEvent` (timestamp, RecoveryAction, RecoveryOutcome/result) の定義
2. イベントを記録するリングバッファまたは履歴コンテナ
3. `getRecoveryHistory()` 公開API
4. CIテストからの呼び出し可能なインターフェース

**推奨**: Phase 0 のタスク 0-6「`RuntimeHealthMonitor::getRecoveryHistory()` 設計レビュー」は必須。Phase 0 内でプロトタイプ実装まで行い、Phase B のリスクを低減すべき。

---

## 3. ❌ TC-01B: 参照ファイル不足（重大）

**計画**: `reference/room_correction.wav`（ルーム補正IR、実測）を使用する。

**現状**:
- ❌ `reference/` ディレクトリが**存在しない**
- ❌ `room_correction.wav` が**存在しない**
- ❌ プロジェクト内に測定済みIRファイルは一切存在しない（scipy テストデータを除く）

**影響**:
- TC-01B は実行不可能。代替案として以下が必要:
  a) 実測IRを別途用意して配置
  b) 合成IR（疑似ルーム補正IR）を使用するよう計画を変更
  c) TC-01B をスキップまたは削除

**推奨**: `reference/` ディレクトリをプロジェクトルートに作成し、少なくとも合成代替IRを同梱することをPhase 1に含める。

---

## 4. ❌ テストケース数の矛盾（軽度の誤りだが要修正）

**計画**: 「テストケース一覧（全31件）」と記載。

**実際の内訳（詳細セクションから）**:
| # | テストケース | カテゴリ |
|---|-------------|---------|
| 1 | TC-01 | 周波数応答 |
| 2 | TC-01B | 周波数応答 |
| 3 | TC-02 | 周波数応答 |
| 4 | TC-03 | 歪み |
| 5 | TC-04 | ノイズ |
| 6 | TC-04A | ノイズ |
| 7 | **TC-05A** | **低域ノイズ（詳細セクションのみ）** |
| 8 | **TC-05B** | **低域ノイズ（詳細セクションのみ）** |
| 9 | **TC-05C** | **低域ノイズ（詳細セクションのみ）** |
| 10 | **TC-05D** | **低域ノイズ（詳細セクションのみ）** |
| 11 | TC-06 | フィルタ |
| 12 | TC-07 | フィルタ |
| 13 | TC-08 | モード切替 |
| 14 | TC-09 | エイリアシング |
| 15 | TC-10 | バイパス |
| 16 | TC-11 | ISR |
| 17 | TC-11B | ISR |
| 18 | TC-12 | ISR |
| 19 | TC-13 | ISR |
| 20 | TC-14 | ISR |
| 21 | TC-15 | Mixed Phase |
| 22 | TC-16 | Progressive Upgrade |
| 23 | TC-17 | IMD |
| 24 | TC-18 | IMD |
| 25 | TC-21 | 位相 |
| 26 | TC-23 | リニアリティ |
| 27 | TC-24 | IMD |
| 28 | TC-25 | Crossfade |
| 29 | TC-26 | Telemetry |
| 30 | TC-27 | Rebuild |
| 31 | TC-28 | OutputFilter |
| 32 | TC-29A | NaN/Inf |
| 33 | TC-29B | NaN/Inf |
| 34 | TC-30 | Recovery |
| **実際の合計** | **34件** | |

**カテゴリ表の集計** (4+3+4+4+2+6+2+5=30) にも TC-05A〜D が計上されておらず、30件となっている。

**修正案**: 合計値を **34件** に修正し、カテゴリ表に TC-05A〜D を「ノイズ/リニアリティ」または新カテゴリとして追加する。

---

## 5. TC-24 FFT条件の自己矛盾（軽度）

**計画**:
- FFT条件: **サイズ: 262144** / 窓: Blackman-Harris / オーバーラップ: 75%
- Phase 1 タスク 1-6: 「**FFTサイズベンチマーク（65536/131072/262144）** → 最終FFTサイズ決定」

**問題点**: TC-24 の仕様にハードコードされた `262144` と、Phase 1 のベンチマーク後に「最適サイズを採用」する記述が矛盾する。

**推奨**: TC-24 のFFTサイズを「ベンチマーク決定（候補: 65536/131072/262144）」とし、確定後に仕様を凍結する。

---

## 6. TC-28: OutputFilter係数ダンプの実装容易性（参考情報）

**計画**: `--cli-dump-filter-coeffs` で OutputFilter の係数を JSON 出力。

**技術評価**:
- ✅ `convo::OutputFilter` クラスは完全実装済み
- ✅ `BiquadCoeff` 構造体は public フィールド（b0, b1, b2, a1, a2）を持つ
- ✅ 全係数は `prepare()` で事前計算済み
- ⚠️ 係数は private メンバ（`hcCoeff`, `lcCoeff`, `hpfCoeff`, `lpCoeff`）→ const アクセサ追加が必要
- ✅ 実装難易度: 低（1〜2時間）

---

## 7. TC-25: Crossfade Integrity 閾値分離（参考情報）

**計画**: 同一ビルド ≤ -120dBFS / 異ビルド ≤ -100dBFS

**評価**:
- ✅ 閾値分離の設計は適切。`--cli-exit-ms` と出力WAVの組み合わせで実現可能。
- ⚠️ CIでの「異ビルド比較」には前回ビルド成果物の保存・参照が必要。GitHub Actions の artifacts 永続化 or キャッシュ戦略を Phase 3 で設計する必要がある。

---

## 8. 既存CLIオプションと計画の対応関係（補足）

**計画で触れられていないが既存の有用なCLIオプション**:

| CLIオプション | 用途 | 計画との関係 |
|--------------|------|-------------|
| `--cli-bypass-burst-value` | バイパス値（0/1）指定 | TC-12 で明示的に使用可能 |
| `--cli-phase` | Mixed Phase モード設定 | TC-02, TC-15 に関連 |
| `--cli-order` | 処理順序（Conv/PEQ/Conv→PEQ/PEQ→Conv） | 計画のアーキテクチャ図はConv→PEQのみ記載 |
| `--cli-target-ir-sec` | ターゲットIR長 | TC-27 に関連 |
| `--cli-debounce-ms` | リビルドデバウンス時間 | TC-11/11B に関連 |
| `--cli-dither-bit-depth` | ディザービット深度設定 | ノイズ試験に関連 |
| `--cli-f1-hz` / `--cli-f2-hz` | Mixed Phase Transition Hz | TC-15 に関連 |

**推奨**: 計画書のCLIオプション一覧（2.3節）に既存オプションも含めて完全なリファレンスとすることで、テスト実装時の混乱を防げる。

---

## 9. 既存のPythonテストツールとの重複

**計画ではPhase 1で新規Pythonモジュールを実装するが、以下の既存ツールが参考になる**:

| 既存ツール | 用途 | 計画との関係 |
|-----------|------|-------------|
| `tools/diagnostics/create_dirac_ir.py` | ディラックIR生成 | TC-01等で参照可能 |
| `tools/diagnostics/create_test_irs.py` | テストIR生成 | Phase 1 generators.py の参考 |
| `tools/diagnostics/generate_test_signal.py` | テスト信号生成 | Phase 1 generators.py の参考 |
| `tools/diagnostics/compare_raw.py` | RAW PCM比較 | Phase 1 analyzers.py の参考 |
| `tools/diagnostics/compare_dirac.py` | Dirac応答比較 | Phase 1 analyzers.py の参考 |
| `tools/diagnostics/analyze_compare.py` | 分析比較 | Phase 1 解析ロジックの参考 |

**推奨**: 既存ツールをラップ/リファクタリングすることでPhase 1の工数を削減できる可能性がある。

---

## 10. 重大度サマリ

| # | 項目 | 重大度 | 対応推奨 |
|---|------|--------|---------|
| 1 | TC-30: `getRecoveryHistory()` 未実装 | **❌ 重大** | Phase 0 でAPI設計＋プロトタイプ実装必須 |
| 2 | TC-01B: `reference/room_correction.wav` 不在 | **❌ 重大** | 参照ファイル作成、または合成IRに代替 |
| 3 | テストケース数の矛盾（31→34） | **⚠️ 軽度** | カテゴリ表と合計値を修正 |
| 4 | TC-24 FFT条件の自己矛盾 | **⚠️ 軽度** | 確定後に仕様凍結するよう注釈追加 |
| 5 | `--cli-bypass-burst-value` 未記載 | **⚠️ 軽度** | CLI一覧に追加推奨 |
| 6 | 既存CLIオプションの完全リファレンス不在 | **⚠️ 軽度** | 2.3節に拡充推奨 |
| 7 | 既存Pythonツールの活用機会 | **ℹ️ 情報** | Phase 1 工数削減の可能性 |
| 8 | `processBlockDouble()` の引数型の不一致 | **ℹ️ 情報** | 実装時に `AudioBlock`→`AudioBuffer` に修正 |
| 9 | TC-25 CI差分比較のストレージ設計 | **ℹ️ 情報** | Phase 3 で設計必要 |

---

## 11. 総評

`doc/work68/automatic_sound_test_plan.md` v6.3+ は、コードベースの現状を正確に把握した上で作成されており、**全体として実行可能な計画である**。特に以下の点が高く評価される:

- ✅ **Phase 0 と Phase 1〜3 の分離**: CLI拡張とコアAPIを先行させる設計は適切
- ✅ **既存CLIオプションの完全活用**: IR reload storm、bypass burst、intent burst など既存機能を最大限活用
- ✅ **Telemetry分離（TC-26）**: CI負荷変動の影響を考慮した設計判断
- ✅ **SHA256 診断用格下げ**: ビルド再現性問題への適切な対処
- ✅ **閾値のビルド種別分離**: Debug/Release/MSVC/icx で別閾値の設定

**修正が必要な重大項目（2件）**:
1. **TC-30** の `getRecoveryHistory()` API — 現状未実装のため、Phase 0 での先行実装または Phase B へのリスケジュールが必要
2. **TC-01B** の `reference/room_correction.wav` — 実測データまたは合成代替データの準備が必要

**修正推奨の軽微項目（3件）**:
1. テストケース数の「全31件」→「全34件」への修正
2. TC-24 FFT条件とベンチマーク方針の整合性向上
3. CLIオプション一覧の完全化

これらを修正することで、計画の実効性はさらに高まる。
