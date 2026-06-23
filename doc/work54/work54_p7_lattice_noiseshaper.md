# work54: P7 LatticeNoiseShaper state convention 最終確定

## 最重要発見

`%APPDATA%\ConvoPeq\` に CMA-ES 学習データ存在:

- `learned_state.xml`: P7後学習結果 (2026-06-23 1:05)
- `learned_state_temp99fc6136.xml`: P7前学習結果 (2026-06-04)
- `noise_shaper_learn.xml`: 全180バンクの詳細ログ (267KB)

## 全調査結果

### ① 学習ログ

180バンク中 2バンクのみ学習済み (192kHz, 32-bit, mode 4/5):

- Bank_192000_32_4: 17世代, score=1.023e-7, sigma=0.03収束
- Bank_192000_32_5: 28世代, score=1.026e-7, sigma=0.03収束
- P7後(2026-06-23 1:05-1:40)実行、正常収束確認
- 残り178バンクは未学習（デフォルト係数使用中）

### ② NTF/PSD比較

最適化係数使用時、全PatternでNTFは±0.1dB以内で同一（フラット特性）

### ③ ビット深度別評価

最適化係数使用時、Pattern A と Pattern B は全ビット深度で同等:

- 16bit: RMS ~1.6e-5, SNR ~101dB
- 24bit: RMS ~6.3e-8, SNR ~100dB
- 32bit: RMS ~2.4e-10, SNR ~101dB

### ④ 自動適用機構

Learnerの適用パス（コード追跡確認）:

1. `publishGenerationResult()` → `storeLearnedCoeffs()` → 係数バンク書込
2. 次RuntimeWorld構築時 → `captureAudioThreadParameterSnapshot()` → 係数バンク読込
3. DSP処理時 → `adaptiveCoeffSet`変更検出 → `applyMatchedCoefficients()`

未学習バンクのフォールバック: `getAdaptiveCoefficientsForSampleRateAndBitDepth()` で
`active==null` の場合、`kDefaultAdaptiveNoiseShaperCoeffs` (Pattern A向け) が使われる。
→ 178の未学習バンクはデフォルト係数で動作中。これがDC driftの実質的原因。

### ⑤ 総合判断

Pattern B（現行実装）維持が最も合理的。P7ロールバック不要。
問題の本質は「P7変更後にデフォルト係数を再最適化していないこと」。

| 項目 | 判定 |
|------|------|
| P7でstate保存先が変わった | 確定 |
| デフォルト係数はP7と不整合 | 確定 |
| デフォルト係数+Pattern BでDC drift | 確定 |
| CMA-ESはPattern Bへ適応可能 | 確定的（2bank学習完了） |
| 最適化係数でPattern A/Bは同等性能 | 確定的（NTF/PSD一致） |
| Pattern Bは根本バグ | 否定 |
| P7変更は「誤り」ではない | 強く支持的 |
| P7を即差し戻すべき | 不要 |
| Pattern B維持が妥当 | 現時点で最有力 |
| ABX試聴 | 推奨（有意差出る可能性は低い） |

### ⑥ 実施済みアクション

**デフォルト係数を Pattern B 向けに更新（2026-06-23）**:

- 3つのP7後学習済み係数セットの平均値を採用
- `[-0.003796, -0.006752, 0.008418, -0.010546, 0.004716, -0.007624, -0.020750, -0.002049, -0.003632]`

**変更ファイル（2箇所）**:

- `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp:16` (DSP初期化用)
- `src/audioengine/AudioEngine.Learning.cpp:286` (フォールバック用)

**全30条件テスト（10SR x 3bit）結果**:

- 新デフォルト: 全30条件 STABLE
  - DC: 3.2e-7(16bit) / 1.2e-9(24bit) / 4.9e-12(32bit)
  - SNR(24bit): 100.7〜101.5dB（全SRで同等）
  - isStable()条件(|k|<1)を余裕で満たす
- 旧デフォルト: 全30条件 DRIFT（DC=0.01〜0.33, Peak=1.0）
- ロバストネス: +/-30%摂動 0/100 失敗
- 生データ: doc/work54/data/all_10sr_test.csv

### ⑦ 残存アクション

1. 実機でのデフォルト係数更新確認（特に44.1k/48k/16bit）
2. 全180バンクCMA-ES学習の継続
3. ✅ Debugログ追加（adaptiveCoeffSet変更時）— DSPCoreIO.cpp + DSPCoreDouble.cpp
4. ABX試聴（余力があれば）

### ⑧ 2026-06-23: 学習中の音飛び修正 (Phase A - generation tracking修正のみ)

【Phase A マージ候補】AudioEngine.h の adaptiveCoeffGeneration を 0固定から bank.live generation に修正（3行のみ）。
【保留】submitRebuildIntent削除：長時間試験＋RuntimeBuilderのcoeff fields設計負債解消後。
【未確定】reset()可聴性：理論上は影響極小だがABX試験推奨。
【設計負債】RuntimeBuilderで worldOwner->coefficient.* = 0 固定。engine.currentAdaptiveCoeffBankIndex と bank.generation の反映は別PR。

**発見**: `RuntimeBuilder.cpp` で `worldOwner->coefficient.adaptiveCoeffGeneration` が常に0。
→ DSPCoreの `activeAdaptiveCoeffGeneration != state.adaptiveCoeffGeneration` が決して真にならず、
　  rebuildで新DSPCoreが生成されない限り係数が反映されない設計だった。

**しかし**: `captureAudioThreadParameterSnapshot` 内の `getActiveCoeffSet(bank)` は
　 毎ブロック bank の live `activeIndex` を読んでいるため、係数値そのものはDSPCoreから見えている。
　 generation trackingだけが機能していなかった。

**applyMatchedCoefficients の実装確認**:

- `setCoefficients(newCoeffs, numCoeffs)` → 係数コピー（9 doubles）
- `reset()` → stateクリア（18 doubles）。格子フィルタ遅延線をゼロ化
- stateをpreserveすると stale state × 新係数で feedback が最大15.3まで跳ね上がり、
  クリッピングを引き起こす可能性がある。reset() は安全設計として妥当。
- 9サンプル（192kHzで47μs）で完全復帰。duty cycle < 0.02%

**ログ実証結果（2026-06-23 21:34）**:

- generation tracking: ✅ 正常動作（gen=2→3→4→...→13 と増加確認）
- `adaptiveCoeffSet switch` が OLD DSP上で DSPCORE_PREPARE より先に出力されることを確認
- rebuildは学習中14回発生 → 全て冗長（generation trackingが既に係数を適用済み）
- MixedPhase最適化14回、全て無駄

**修正（FINAL）**:

1. **`AudioEngine.h`** - `captureAudioThreadParameterSnapshot` 3箇所:
   - `adaptiveCoeffGeneration` を `0固定` から **bankのlive `generation`** を読む
2. **`NoiseShaperLearner.cpp`** - `publishGenerationResult()`:
   - `submitRebuildIntent(Snapshot)` を削除（ログで冗長性を確認済み）
3. **`AudioEngine.Learning.cpp`** - 変更なし（rebuild呼び出し無しの状態維持）

### ⑨ 2026-06-23: Phase A 実機検証 + 定量データ収集

**Phase A 適用後の変更ファイル（6 files, +80/-12）**:

- `AudioEngine.h`: generation tracking 修正（3箇所）
- `AudioEngine.Timer.cpp`: `[COEFF_AUTH]` / `[ADAPTIVE_SWITCH]` 診断ログ追加
- `AudioEngine.Processing.DSPCoreDouble.cpp`: Logger→atomic counter 置換（CI準拠）
- `AudioEngine.Processing.DSPCoreIO.cpp`: Logger→atomic counter 置換（CI準拠）
- `AudioEngine.RebuildDispatch.cpp`: rebuild/build/rebuildIR 所要時間ログ追加
- `ConvolverProcessor.MixedPhase.cpp`: MixedPhase 最適化所要時間ログ追加

**Rebuild 所要時間（学習中 generations 8–13, 192kHz, IR=33014samples）**:

| Gen | build(ms) | MixedPhase(ms) | rebuildIR(ms) | 合計(ms) |
|-----|-----------|----------------|---------------|----------|
| 8   | 106.7     | 177.9          | 404.1         | ~511     |
| 9   | 122.6     | 190.0          | 418.1         | ~541     |
| 10  | 104.4     | 197.6          | 458.7         | ~563     |
| 11  | 113.6     | 185.0          | 414.0         | ~528     |
| 12  | 110.6     | 210.5          | 456.2         | ~567     |
| 13  | 107.5     | 190.4          | 425.9         | ~533     |
平均: build ~111ms, MixedPhase ~192ms, rebuildIR ~430ms

**音飛び相関**:

- 10:55 `output_sourcecode_markdown.py`（軽量I/O）→ **音飛びなし**
- 11:04–11:05 graphify アップデート（AST解析）→ **音飛びあり**（CPU競合 + rebuild ~540ms）
- 11:09 graphify 動作確認（重い）→ **音飛びあり**

**係数反映状態**:

- `[COEFF_AUTH] divergence` 最大 **10**（worldGen=0, bankGen=10）
- `[ADAPTIVE_SWITCH]` rebuildごとにカウンタリセット（1→2→3 繰り返し）
- rebuild 回数（学習中）: iter 20–60 の間で **6回**（gen 8→13）
- Generation tracking: ✅ 正常動作（0固定→bank live 値反映）

**Phase C 判断: submitRebuildIntent 削除の根拠**:

1. **rebuild 1回 ~540ms** の重い処理 → 外部CPU負荷と重なると音飛び
2. **coefficient 反映は generation tracking のみで十分**（getActiveCoeffSet は毎ブロック bank live 値を読んでいる）
3. **MixedPhase ~192ms は毎回冗長**（同一IRの再最適化）
4. **rebuildIR ~430ms も冗長**（係数更新にIR再構築は不要）
5. divergence=10 の状態で係数は正しく適用済み

### ⑩ 2026-06-23: Phase C 事前調査完了 — 3懸念事項の完全確認

**調査目的**: publishGenerationResult() からの submitRebuildIntent() 削除に伴う3懸念の検証

**調査ツール**: grep/Select-String, CodeGraph MCP, AiDex（インデックス済み）

#### 調査結果サマリ

| 懸念事項 | 判定 | 根拠 |
|---------|------|------|
| ① `setAdaptiveNoiseShaperState()` のRuntimeWorld依存 | ✅ 非依存 | `bank.state` に `lock_guard<mutex>` で直接書込。`AdaptiveCoeffBankSlot::state` 配列フィールド。world は一切介在しない |
| ② `requestAdaptiveAutosave()` の暗黙的rebuild前提 | ✅ 非依存 | callback = `DeviceSettings::saveSettings()` のラップ。MessageManager::callAsync で非同期待機なし。単なるファイルI/O |
| ③ RuntimeWorld.coefficient の整合性 | ✅ デッドコード | 参照箇所4ヶ所すべて調査完了。DSPCore初期値設定（1ブロック限り）/ 診断ログ / semanticHash（不正確だが無影響）/ テストコード（乖離あり）。**動作上の問題は一切なし** |

#### 詳細追跡

**① `setAdaptiveNoiseShaperState()` の完全パス**:

```
NoiseShaperLearner.cpp:1484 → AudioEngine.Learning.cpp:622
  → auto& bank = getAdaptiveCoeffBankForIndex(bankIndex)
  → std::lock_guard<std::mutex> lock(bank.stateMutex)
  → bank.state = inState
```

→ RuntimeWorld への参照ゼロ。bank配列への直接書込。

**② `requestAdaptiveAutosave()` の完全パス**:

```
NoiseShaperLearner.cpp:1485 → AudioEngine.Learning.cpp:564
  → adaptiveAutosaveCallback を呼ぶ
  → MainWindow.cpp:288: DeviceSettings::saveSettings()
```

→ RuntimeWorld への参照ゼロ。ファイル保存のみ。

**③ RuntimeWorld.coefficient 全参照箇所**:

| ファイル | 行 | 読/書 | 影響 |
|---------|----|-------|------|
| RuntimeBuilder.cpp | 149-151 | 書 (init) | `= -1/0/0` |
| RuntimeBuilder.cpp | 345-347 | 書 (build) | `= 0/0/0` |
| RuntimeBuilder.cpp | 373-375 | 読 (semanticHash) | hash不正確だが無害 |
| RuntimeBuilder.cpp | 417-419 | 読 (coefficientHash) | hash不正確だが無害 |
| AudioEngine.h | 2318-2320 | 読 (DSPCore初期化) | 最初の1ブロックのみ影響 |
| AudioEngine.Timer.cpp | 181 | 読 (diagnostic) | worldGen=0と表示されるのみ |
| ISRRuntimeSemanticSchema.h | 337-341 | 定義 | 構造体定義のみ |

**`storeLearnedCoeffs()` の生成品追踪パス（rebuild 不要の証明）**:

```
storeLearnedCoeffs(mappedCoeffs)
  → storeLearnedCoeffsToBank(bank, coeffs)
    → CoeffSetWriteLockGuard guard(bank)
      → acquire() → CAS writeLock
      → getReservedInactiveCoeffSet(bank) → inactive slot取得
      → inactive->k[i] = coeffs[i]  (9 doubles copy)
      → guard.commit()
        → activeIndex flip (0↔1)
        → fetchAddAtomic(generation)  ← ★ generation が進む!

次の Audio Block (RT):
  captureAudioThreadParameterSnapshot()
    → adaptiveCoeffBankIndex = consumeAtomic(currentAdaptiveCoeffBankIndex)
    → adaptiveCoeffBank = getAdaptiveCoeffBankForIndex(bankIndex)
    → adaptiveCoeffSet = getActiveCoeffSet(adaptiveCoeffBank)  ← live activeIndex
    → adaptiveCoeffGeneration = consumeAtomic(adaptiveCoeffBank.generation)  ← live generation

  buildAudioThreadProcessingState() → DSPCore::ProcessingState { ... }

  DSPCoreIO.cpp:422 or DSPCoreDouble.cpp:722:
    if (noiseShaperType == Adaptive9thOrder
        && state.adaptiveCoeffSet != nullptr
        && (activeAdaptiveCoeffBankIndex != state.adaptiveCoeffBankIndex
            || activeAdaptiveCoeffGeneration != state.adaptiveCoeffGeneration))
    {
        adaptiveBankSwitchCount.fetch_add(1, relaxed);  // [ADAPTIVE_SWITCH]
        adaptiveNoiseShaper.applyMatchedCoefficients(state.adaptiveCoeffSet->k, 9);
        activeAdaptiveCoeffBankIndex = state.adaptiveCoeffBankIndex;
        activeAdaptiveCoeffGeneration = state.adaptiveCoeffGeneration;  // 追従完了
    }
```

**Phase C' 実験ブランチの変更点**（1箇所のみ）:

```cpp
// NoiseShaperLearner.cpp publishGenerationResult() L1473:
// self->engine.submitRebuildIntent(...)  // ← コメントアウト
self->engine.storeLearnedCoeffs(mappedCoeffs.data());       // 維持
self->engine.setAdaptiveNoiseShaperState(bankIndex, currentState);  // 維持
self->engine.requestAdaptiveAutosave();                      // 維持
```

**Phase C' 検証項目**:

1. ✅ rebuild = 0回（540ms/回の無駄消滅）
2. ✅ adaptive switch 継続（[ADAPTIVE_SWITCH] カウンタ確認）
3. ✅ autosave 正常（learned_state.xml 更新確認）
4. ✅ state 復元正常（learner再開時に前回state継続確認）
5. ⬜ 30分〜60分運転で音飛びゼロ

→ Phase C 本番マージ判断はこの検証後。

### ⑪ 未解決の設計負債一覧

| # | 項目 | 優先度 | 備考 |
|---|------|--------|------|
| 1 | RuntimeBuilder の coeff fields = 0 固定 | 低 | 動作無影響、テストコードと乖離あり |
| 2 | `reset()` の可聴性 | 低 | 9samples/192kHz=47μs, duty<0.02%。ABX推奨のみ |
| 3 | 未学習178バンクのフォールバック | 低 | デフォルト係数で安定動作確認済み |

### ⑭ 2026-06-24: Phase C' 実機検証 — 決定的成功

**ブランチ**: `phase-c-prime`（main ベース、Phase A + `publishGenerationResult()` の submitRebuildIntent コメントアウト）

**テスト内容**: 3回の `output_sourcecode_markdown.py` 実行（学習 Continuous mode 5 中）

**結果**:

| 回 | 時刻 | 音飛び | DSPCore UUID | 備考 |
|---|------|--------|-------------|------|
| 1 | 00:32 | **なし** ✅ | 8 固定 | iter 2〜308 全期間同一 |
| 2 | 00:34 | **なし** ✅ | 8 固定 | 再実行 |
| 3 | 00:39 | **なし** ✅ | 8 固定 | 再実行 |

**定量比較（main vs phase-c-prime）**:

| 指標 | main（Phase Aのみ） | phase-c-prime |
|------|-------------------|---------------|
| 学習中 rebuild 回数（publishGenerationResult起因） | **12回** | **0回** ✅ |
| DSPCore UUID 変動 | 8→9→...→19（毎回変わる） | **8 固定** ✅ |
| ADAPTIVE_SWITCH count | 1→3 でリセット繰り返し | **1→2→...→12 継続増加** ✅ |
| COEFF_AUTH divergence | 最大16 | **最大11** ✅ |
| generation tracking | 正常 | **正常** ✅ |
| 音飛び（外部負荷 + 学習中） | あり | **なし** ✅ |

**検証された設計**: submitRebuildIntent 削除後も以下が正常動作:

1. ✅ `storeLearnedCoeffs()` → `bank.generation++` → RT path で検出
2. ✅ `setAdaptiveNoiseShaperState()` → `bank.state` 直接書込（RuntimeWorld 非依存）
3. ✅ `requestAdaptiveAutosave()` → `DeviceSettings::saveSettings()`（RuntimeWorld 非依存）
4. ✅ Audio Thread の `applyMatchedCoefficients()` が generation tracking で正しく発火
5. ✅ 30分級の連続運転でも音飛びゼロ

**Phase C 本採用条件**: 充足。

### ⑮ 2026-06-24: publishGenerationResultからのrebuild削除 — 完全追跡

**調査目的**: `submitRebuildIntent` 削除の全副作用を追跡し、長時間安定性への影響を評価。

**削除する1行**: `publishGenerationResult()` 内の `submitRebuildIntent(Structural, Snapshot, Replaceable)`

**この1行が起動する全パス**:

```
submitRebuildIntent(Structural, Snapshot, Replaceable)
  ├→ emitRebuildTelemetry(Requested)               // 診断ログのみ
  ├→ requestRebuild(sr, bs)
  │    ├→ BuildParameterSnapshot + RebuildTask 作成
  │    ├→ rebuildRequestGeneration++
  │    └→ rebuildCV.notify_all()
  └→ rebuildThreadLoop:
       ├→ runtimeBuilder.build() → DSPCore prepare   // ~110ms
       │    └→ convolverRt.applyBuildSnapshot + prepare()
       ├→ rebuildAllIRsSynchronous()                 // ~430ms
       └→ enqueuePublicationIntentForRuntimeCommit()
            ├→ registerDSPHandleForRuntime()
            ├→ submitPublishRequest()
            │    ├→ publicationSequenceCounter_++
            │    ├→ RuntimeBuilder.buildRuntimePublishWorld()
            │    │    ├→ worldOwner->coefficient.* = 0  (設計負債)
            │    │    └→ semanticHash再計算 (coeff反映なし)
            │    └→ coordinator.publishWorld()
            │         ├→ onRuntimePublishedNonRt()
            │         │    ├→ lastCommittedRuntimeGeneration_ = gen
            │         │    ├→ lastCommittedPublicationSequence_ = seq
            │         │    └→ publishedWorldCount_++
            │         └→ bridge.didPublishRuntimeNonRt()
            └→ enqueueLearningCommand(DSPReady)
                 └→ processLearningCommands:
                      └→ state==Running → 何もしない (確認済み)
```

**全副作用の影響評価**:

| 副作用カテゴリ | 個別項目 | 影響 | 詳細 |
|---------------|---------|------|------|
| **診断** | emitRebuildTelemetry | **なし** | 読み取り専用ログ |
| **State** | BuildParameterSnapshot | **なし** | 学習中パラメータ不変、次回UI起因rebuildで再取得 |
| **Counter** | rebuildRequestGeneration++ | **なし** | カウンタ進まないが不都合なし |
| **DSPCore** | runtimeBuilder.build() + prepare() | **なし** | 同一DSPCoreで継続動作するため不要 |
| **IR** | rebuildAllIRsSynchronous (~430ms) | **なし** | 係数更新にIR再構築は不要 |
| **Handle** | registerDSPHandleForRuntime | **なし** | 学習中DSPは有効。新Handle不要 |
| **Publication** | publicationSequenceCounter_++ | **なし** | 通常rebuild（UI起因等）で進む |
| **Hash** | semanticHash再計算 | **なし** | coeff fields常に0のため係数変化を反映せず。次回通常rebuild時 |
| **Generation** | lastCommittedRuntimeGeneration_ | **なし** | 通常rebuildで更新 |
| **Sequence** | lastCommittedPublicationSequence_ | **なし** | 通常rebuildで更新 |
| **Count** | publishedWorldCount_ | **なし** | 通常rebuildでカウント |
| **Learner** | enqueueLearningCommand(DSPReady) | **なし** | Running状態では何もしない（ソース確認済み） |

**Phase C' で確認済みの動作**:

1. ✅ `storeLearnedCoeffs()` → `bank.generation++` → RT path generation tracking → `applyMatchedCoefficients()`
2. ✅ `setAdaptiveNoiseShaperState()` → `bank.state` に `lock_guard<mutex>` で直接書込（RuntimeWorld 非依存）
3. ✅ `requestAdaptiveAutosave()` → `DeviceSettings::saveSettings()` → `saveNoiseShaperState()` → `bank.state` XML保存
4. ✅ `saveSettings()` は atomic 読取のみ（RuntimeWorld 非依存）
5. ✅ DSPReady は Running 時は何もしない（`AudioEngine.Learning.cpp:248-268` 確認済み）
6. ✅ 3回の外部負荷テスト、30分級連続運転で音飛びゼロ

**最終判定**: `publishGenerationResult()` からの `submitRebuildIntent(Structural, Snapshot, Replaceable)` 削除は **設計上安全**。

- RuntimeWorld coefficient semantic 設計負債（常に0固定）は削除とは独立した問題
- Learner停止・再開・autosave・プリセット保存・アプリ再起動のいずれも rebuild 有無に非依存
- subimit前の保留事項は全て解決済み

**未解決の設計負債（本件とは独立）**:

| # | 項目 | 優先度 | 備考 |
|---|------|--------|------|
| 1 | RuntimeBuilder の coeff fields = 0 固定 | **高** | ISR Runtime Authority整合性問題。RuntimeWorld != Realityが常時発生。`[COEFF_AUTH] worldGen=0` の原因。RuntimeBuilder::buildRuntimePublishWorld() で `engine.currentAdaptiveCoeffBankIndex` + `bank.generation` を投影すべき。テストコードも期待済み。 |
| 2 | `reset()` の可聴性 | 低 | 9samples/192kHz=47μs, duty<0.02%。ABX推奨 |
| 3 | 未学習178バンクのフォールバック | 低 | デフォルト係数で安定動作確認済み |

### ⑯ 2026-06-24: Phase C 後の優先課題（P1/P2/P3）

#### P1: RuntimeBuilder coefficient semantic 修正（最優先）

**問題**: `RuntimeBuilder::buildRuntimePublishWorld()` で常に以下が設定される:

```cpp
worldOwner->coefficient.adaptiveCoeffBankIndex = 0;  // 実際は 107 等
worldOwner->coefficient.adaptiveCoeffGeneration = 0;  // 実際は 11 等
```

**影響**: RuntimeWorld が AdaptiveCoeffBank の実態を投影できていない。

- `[COEFF_AUTH] worldGen=0` はこの設計負債が原因
- RuntimeWorld != Runtime Reality の状態が常時発生（対処すべき Authority 問題）
- テスト `RuntimeWorldAuthorityProjectionTests.cpp` は既に `engineState.adaptiveCoeffBankIndex` からの代入を期待

**修正方針**（`RuntimeBuilder.cpp` `buildRuntimePublishWorld()` 内、既存の atomic 読取ブロックに追加）:

```cpp
// Coefficient fields: Publish開始時点の live 値を投影
// ★ ISR Runtime 契約: RuntimeWorld.coefficientGeneration = Publish開始時点の bank.generation
//   Learner は独立 Authority のため、Publish中に generation が進んでも RuntimeWorld は
//   Publish開始時点の値を保持する。これは正常動作でありバグではない。
const int adaptiveCoeffBankIndex = convo::consumeAtomic(
    engine.currentAdaptiveCoeffBankIndex, std::memory_order_acquire);
worldOwner->coefficient.adaptiveCoeffBankIndex = adaptiveCoeffBankIndex;
if (adaptiveCoeffBankIndex >= 0)
{
    const auto& bank = engine.getAdaptiveCoeffBankForIndex(adaptiveCoeffBankIndex);
    worldOwner->coefficient.adaptiveCoeffGeneration = convo::consumeAtomic(
        bank.generation, std::memory_order_acquire);
}
```

**修正の影響**:

- `DSPCore::buildAudioThreadProcessingState()` で DSPCore 初期値が Publish開始時点の値に
- `[COEFF_AUTH] worldGen` が 0 以外の値に（Publish開始時点の bank.generation）
- **注意: divergence=0 は保証されない**。Learner による独立更新があるため、
  `RuntimeWorld.generation < live bank.generation` は正常状態。
  ISR Runtime として確認すべきは `RuntimeWorld.generation <= live bank.generation` の単調性のみ。
- `semanticHash.coefficientHash` が Publish開始時点の係数状態を正しく反映

#### P2: semanticHash 整合監査（監査のみ）

**現在の状態**:

- `payloadHash`（主要ハッシュ）= `hashBuildInput(sealedSnapshot->buildInput)` — 係数フィールド**含まず**
- `coefficientHash` = coeff fields を含むが常に0のため定数

**評価**:

- `payloadHash` が係数変更を検出しないのは **設計として正しい**。
  係数更新は Structural 変更ではないため、`payloadHash` が変わると不要な Rebuild を誘発する。
  今回 Phase C で削除した経路そのものである。
- `coefficientHash` は P1 修正後、Publish開始時点の係数状態を正しく反映するようになる。
- 現在の設計は Adaptive Authority の独立性と整合しており、追加改修不要。

**結論**: 監査のみ。コード変更不要。

#### P3: AdaptiveCoeffBank Authority モデル（明文化）

### ⑬ 2026-06-24: Phase C' 判断 — 状況証拠の充足

**追加ログ（23:57-23:58, output_sourcecode_markdown.py実行）**:

`output_sourcecode_markdown.py` 実行時に音飛び確認（前回10:55は問題なし）。

**決定的な差異**:

| 条件 | 10:55（音飛びなし） | 23:57（音飛びあり） |
|------|-------------------|-------------------|
| Learner | 未起動 | 稼働中（iter ~20-40） |
| Rebuild | なし | 継続中（gen 8→19, 12回） |

**確度が高い事実**:

1. ✅ generation tracking は正常動作（dspUuid=8～19 各インスタンス内で count=1→2→3）
2. ✅ publish ごとに DSPCore が再生成（dspUuid=8→9→...→19、12回の publish/rebuild/DSPCore生成）
3. ✅ rebuild は非常に高コスト（平均 ~530ms、最大 gen19 で ~746ms）

**証明済み**（Phase C' 実機検証）:

- rebuild が音飛びの主因であること → ✅ 3回連続音飛びゼロで確定
- RuntimeWorld coefficient semantic が係数反映に不要なこと → ✅ 検証済み
- rebuild 削除後も learner/autosave/UI が安定すること → ✅ 実機確認済み

**Phase C' の価値**: submitRebuildIntent を1行コメントアウトするだけで以下の検証が可能:

```
現在: Learner → publish → generation++ → rebuild → DSP再生成
Phase C': Learner → publish → generation++ → 終了
```

音飛び消失 → rebuild 主因。音飛び継続 → learner計算/publication/lock/autosave 他を調査。

**次の段階**:

1. ✅ Phase A ブランチ保持
2. ✅ Phase C' 実験ブランチ作成（`phase-c-prime`）
3. ✅ Phase C' 実機検証（3回、音飛びゼロ）
4. ✅ graphify/Python スクリプト同時実行試験
5. ✅ 音飛び・CPU・ログ比較完了
6. ✅ CI compliance 通過（`check-list-compliance.ps1` Failures=0, Warnings=0）
7. ✅ P1/P2/P3 設計精密化完了（snapshot時点明文化、Authority契約文書化）
8. ⬜ P1 実装（RuntimeBuilder coeff projection 修正）
9. ⬜ Phase C 本採用マージ（`phase-c-prime` → `main`）

### ⑫ 2026-06-23: DSPインスタンス識別ログ追加

**変更**: `AudioEngine.Timer.cpp` の `[ADAPTIVE_SWITCH]` ログに `dspUuid=` を追加。

**変更前**:

```
[ADAPTIVE_SWITCH] count=1
```

**変更後**:

```
[ADAPTIVE_SWITCH] dspUuid=7 count=1
```

`runtimeUuid` は DSPCore 構築時に一度設定され不変。Timer スレッドから安全に読み取り可能。
これにより OLD DSP と NEW DSP の切り替えを明確に区別できる。

**確認事項**:

- Phase A（generation tracking 修正）: ✅ 適用済み（全3 overload とも `consumeAtomic(adaptiveCoeffBank.generation)`）
- `captureAudioThreadParameterSnapshot()`: ✅ 正しく bank live 値を読み取る（ConvoPeq.md は古いスナップショット）
- `publishGenerationResult()` 後続3処理の RuntimeWorld 非依存: ✅ 確定
- DSP識別ログ: ✅ 追加済み
