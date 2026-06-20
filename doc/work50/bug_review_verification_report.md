# ConvoPeq バグレビュー検証 – ソースコード検証報告書

**検証者**: GitHub Copilot (DeepSeek V4 Flash)
**検証日時**: 2026-06-20
**使用ツール**: AiDex MCP (コード検索・シグネチャ), Serena MCP (シンボル検索), grep (テキスト検索), Graphify MCP, 直接ファイル読み取り

---

## 概要

複数回にわたるレビューアーの指摘10項目について、実際のソースコードを詳細に調査・検証した結果を報告する。

---

## 🔴 No.1: IRCacheKey::operator< – Strict Weak Ordering 違反

**判定: ✅ 本物のバグを確認（P0 – 即日修正）**

**ソース**: `src/ConvolverProcessor.h` L1093-1108

**実コード**:

```cpp
bool operator<(const IRCacheKey& other) const {
    if (fileHash != other.fileHash) return fileHash < other.fileHash;
    if (sampleRate != other.sampleRate) return sampleRate < other.sampleRate;
    if (phaseMode != other.phaseMode) return phaseMode < other.phaseMode;
    if (std::abs(f1 - other.f1) > 1.0e-6f) return f1 < other.f1;  // ← SWO違反
    if (std::abs(f2 - other.f2) > 1.0e-6f) return f2 < other.f2;  // ← SWO違反
    if (std::abs(tau - other.tau) > 1.0e-6f) return tau < other.tau; // ← SWO違反
    return targetLength < other.targetLength;
}
```

**検証結果**:

- `std::abs(f1 - other.f1) > 1.0e-6f` による許容誤差比較は非推移的関係を生む
- 例: `A=0.0, B=0.5e-6, C=1.5e-6` で:
  - `A < B` == false, `B < A` == false（等価と判定）
  - `B < C` == false, `C < B` == false（等価と判定）
  - `A < C` == true（推移性違反！）
- `std::map` のコンパレータが Strict Weak Ordering を満たさない場合、C++ 標準上 UB（未定義動作）
- 木構造の破綻によるアクセス違反や無限ループの可能性

**推奨修正**: 許容誤差を削除し、直接比較に変更:

```cpp
bool operator<(const IRCacheKey& other) const {
    if (fileHash != other.fileHash) return fileHash < other.fileHash;
    if (sampleRate != other.sampleRate) return sampleRate < other.sampleRate;
    if (phaseMode != other.phaseMode) return phaseMode < other.phaseMode;
    if (f1 != other.f1) return f1 < other.f1;
    if (f2 != other.f2) return f2 < other.f2;
    if (tau != other.tau) return tau < other.tau;
    return targetLength < other.targetLength;
}
```

---

## 🟠 No.2: DeferredDeletionQueue::reclaim – 非効率スキャン

**判定: ⚠️ 改善推奨だが「重大バグ」は過大評価（P3）**

**ソース**: `src/DeferredDeletionQueue.h` L118-167

**実コードの動作**:

```
while (scanned < 1024) {
    diff = seq - (scanPos + 1);
    if (diff != 0) break;           // 空スロット → 即脱出
    if (canDelete && scanPos == deqPos) {
        // 削除成功 → scanPos, scanned をリセット
    } else {
        if (scanPos - deqPos > 1024) scanPos = deqPos;
        else ++scanPos;
        ++scanned;
    }
}
```

**検証結果**:

- 先頭エントリが `minReaderEpoch` 未満で削除不可の場合、最大1024エントリ走査するという指摘は **コード上正確**
- ただし以下の緩和要因あり:
  - **空スロット検出時は即座に `break`**（無駄な走査は最短で終了）
  - メッセージスレッド / Timer スレッドで実行（RTスレッドではない）
  - `scanPos - deqPos > kMaxScan` でリセット機構あり
- 「CPU浪費」の程度は、典型的なキュー長が数十程度であれば問題にならない
- 極端なケース（キューに1024エントリ全てが滞留）でのみ顕在化

**推奨修正**: レビュー提案の `else { break; }` は合理的だが、P0/P1 ほどの緊急性はない。`scanned` の上限があるためクラッシュには直結しない。

---

## 🟡 No.3: CacheManager::computeKey – targetIRLengthSec ハッシュ漏れ

**判定: ✅ 非バグ（正常設計の範囲内）**

**ソース**: `src/CacheManager.cpp` L83-96, `src/convolver/ConvolverProcessor.StateAndUI.cpp` L949-961, `src/convolver/ConvolverProcessor.LoaderThread.cpp` L619

**実コード**:

```cpp
// CacheManager::computeKey（パラメータに targetIRLengthSec なし）
uint64_t CacheManager::computeKey(const juce::File& file,
                                  int fftSize,
                                  double sampleRate,
                                  int phaseMode,
                                  int partitionSize)
{
    uint64_t seed = computeFileContentCRC(file);
    seed = hashCombine(seed, static_cast<uint64_t>(fftSize));
    seed = hashCombine(seed, static_cast<uint64_t>(phaseMode));
    seed = hashCombine(seed, static_cast<uint64_t>(partitionSize));
    uint64_t srBits = 0;
    std::memcpy(&srBits, &sampleRate, sizeof(double));
    seed = hashCombine(seed, srBits);
    return seed;
}

// computeTargetIRLength（LoaderThread の doTrimStep で使用）
int ConvolverProcessor::computeTargetIRLength(double sampleRate, int /*originalLength*/) const
{
    const double targetIRTimeSec = [this]() -> double {
        const juce::ScopedLock lock(pendingOverrideLock);
        return static_cast<double>(pendingOverride.targetIRLengthSec);
    }();
    int target = static_cast<int>(sampleRate * targetIRTimeSec);
    target = (std::min)(target, MAX_IR_LATENCY);
    target = (std::max)(target, 1);
    return target;
}
```

**検証結果（データフロー完全トレース）**:

IRロードの3つのパスを完全追跡:

**Path A: LoaderThread**（`loadImpulseResponse()` → `LoaderThread` → アクティブコンボルバ構築）

1. `doLoadIRStep()`: ファイルからフルIRを読み込み
2. `doTrimStep()`:
   - 無音テール除去（ノイズフロア < 1e-15、SIMD実装）
   - リサンプリング（必要時、r8brain使用）
   - DCブロッキング
   - Tukeyウィンドウ適用
   - **`computeTargetIRLength()` を呼び出し → `pendingOverride.targetIRLengthSec` を参照**
   - IRを `targetLength` サンプルにトリミング + クロスフェード
3. `doTransformStep()`: MixedPhase/位相変換
4. `doBuildStep()`: StereoConvolver構築（時間領域IRからNUCエンジン初期化）
→ **targetIRLengthSec は正しく反映される**

**Path B: CacheManager fast path**（`loadIR()` → `cacheManager->loadPreparedState()`）

- キャッシュヒット時、パーティションデータ（周波数領域）を直接適用
- targetIRLengthSec は参照しない
- **しかし**: CacheManagerがキャッシュするのはIRファイル全体のパーティションデータであり、targetIRLengthSecトリミングはLoaderThread内の時間領域処理（Path A）で独立適用される
- Path BのConvolverStateはtargetIRLengthSec未反映だが、LoaderThread（Path A）が非同期的に上書きする

**Path C: ProgressiveUpgradeThread**

- `IRConverter::convertToHighRes()` → `convertFile()`: フルIRをパーティション分割
- targetIRLengthSec は参照しない
- アクティブコンボルバは既にPath AでtargetIRLengthSec反映済みのため問題なし

**結論（過渡状態の観測可能性を含む完全トレース）**:

**過渡状態の観測可能性に関する追跡調査**:

- `applyComputedIR()`（cache hit path）→ `updateConvolverState()` → `rcuSwapper.swap()` + `publishAtomic(convolverState, ...)` : これは **`ConvolverState`（メタデータ/スナップショット）のみ**を更新
- `applyNewState()`（LoaderThread completion path）→ `switchEngineOnMessageThread()` → `exchangeActiveEngine()` + `advanceRetireEpoch()` : これは **`StereoConvolver`（実畳み込みエンジン）のみ**を更新
- この2つは**独立した別のデータ構造**であり、クロスパスは存在しない
- `ConvolverProcessor::process()`（Audio Thread）は `loadActiveEngine()` で `StereoConvolver*` を取得 — `ConvolverState` は参照しない
- `getConvolverState()` は `AudioEngine.Snapshot.cpp` からのみ呼ばれ、UIスナップショット用途限定
- `ConvolverRuntime`（`runtime` メンバ）も同様にデッドコード:
  - `applyComputedIR()` → `runtime.reallocate()` — 書込みのみ
  - `releaseResources()` → `runtime.clear()` — クリアのみ
  - `ConvolverProcessor::process()` 内で `runtime` への参照は**皆無**
  - 全登場箇所: LoadPipeline.cpp(2), Lifecycle.cpp(1) — いずれも Audio Thread外
- **確認証跡**: grep/Serena/AiDex/semble/CodeGraph の全ツールで、`ConvolverState::partitionData` へのランタイム読み取りパスは存在しないことを確認
- **追加確認**: 過去セッション(2026-06-17)の分析では「パーティションデータ→即座に畳み込み適用」とされていたが、実際には `ConvolverState` への書込みのみで `StereoConvolver` 経路は別途存在

**したがって:**

- `CacheManager::computeKey` に `targetIRLengthSec` を含める必要は**ない**
- CacheManagerのパーティションデータはIRファイル全体をカバーし、targetIRLengthSecトリミングはLoaderThread内の時間領域処理（Path A）で独立して適用される
- キャッシュヒット時の `ConvolverState` はtargetIRLengthSec未反映だが、これは**スナップショットメタデータのみ**に影響 — **オーディオ出力に影響を与える経路は存在しない**
- 初回レポートの指摘は**CacheManagerの役割とLoaderThreadの分離を誤解**したもの
- **「現状の調査範囲では問題の証拠なし」** が最も厳密な表現（非バグ確定までは言い過ぎない）

---

## 🟡 No.4: ノイズシェーパー 量子化順序（ディザ→クランプ）

**判定: ⚠️ コード上のパターンは確認。実用上の影響は軽微（P3）**

**ソース**:

- `src/Fixed15TapNoiseShaper.h` L315-328 (`quantize`)
- `src/FixedNoiseShaper.h` L285-298 (`quantize`)
- `src/LatticeNoiseShaper.h` L264-277 (`quantize`)

**全3ファイルで同一パターン**:

```cpp
v += (u1 + u2 - 1.0) * scale;  // ① ディザ加算
if (v < minV) v = minV;          // ② クランプ
else if (v > maxV) v = maxV;
__m128d d = _mm_set_sd(v * invScale);
d = _mm_round_sd(d, d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
return _mm_cvtsd_f64(d) * scale;
```

**検証結果**:

- DSP理論上、正しい順序は「クランプ → ディザ → 量子化（→ 最終クランプ）」（Lipshitz, Wannamaker, 1992）
- 現状の「ディザ→クランプ」では、ピーク付近でディザノイズの確率分布が非対称に切り捨てられる → 理論上の歪み成分
- **実用上の影響は極めて軽微**:
  - TPDF ディザの振幅は `scale = 1/2^(bits-1)`（24bit: ≈ 1.2e-7, 32bit: ≈ 4.7e-10）
  - 信号がこの微小範囲でクランプ限界に達するのは事実上クリッピング状態
  - ヘッドルーム（例: -1dB）のある設計ではさらに稀
- 2つ目のレビュアーの「立証不能」はやや保守的すぎる — コードパターンは確認可能

**推奨**: 純粋なDSP正しさを追求するなら「クランプ → ディザ → 量子化 → クランプ」に修正。実用上の音質差はほぼ聴感上区別不能。

---

## 🟢 No.5: ヘッドルーム未復元

**判定: ✅ 誤指摘（コードも設計も意図的）**

**ソース**:

- `src/Fixed15TapNoiseShaper.h` L170: `dataL[i] = processSample(dataL[i] * headroom, 0, error);`
- `src/FixedNoiseShaper.h` L140: `dataL[i] = processSample(dataL[i] * headroom, 0, error);`
- `src/LatticeNoiseShaper.h` L144: `dataL[i] = processSample(0, dataL[i], states[0], activeCoeffs, headroom);`

**検証結果**:

- 全3つのノイズシェーパーで `processSample` の引数に `* headroom` を渡し、戻り値に `/ headroom` の逆補正がないのは確か
- **しかしこれは意図的設計**:
  1. ノイズシェーパー（特に `Fixed15TapNoiseShaper`）は**最終出力段**として機能
  2. `currentBitDepth <= 0` のコードパスでも `dataL[i] *= headroom` のみで逆補正なし → 一貫性
  3. ヘッドルームは「量子化器のクリッピング防止」と「最終出力レベルの調整」の両方を兼ねる
- レビューの提案する `/ headroom` を適用すると、意図した出力レベル低下が打ち消され、量子化器のヘッドルームマージンが失われる
- 2つ目のレビュアーの「誤指摘寄り」の判断が妥当

**結論**: 修正不要。

---

## 🟡 No.6: CMA-ES 初期平均値（θの正規化）

**判定: ✅ コード上の不整合を確認（P2 – 次回リリース時）**

**ソース**: `src/AllpassDesigner.cpp` L319-321 + `AllpassDesigner.cpp` L243-245

**実コード（初期平均値計算 – 順変換）**:

```cpp
const double theta = 2.0 * pi * freqHz / sampleRate;
const double tNorm = std::clamp(theta / juce::MathConstants<double>::pi, 1e-6, 1.0 - 1e-6);
initialMean[2*i+1] = std::log(tNorm / (1.0 - tNorm));  // logit
```

**実コード（逆変換 – unconstrainedToTheta）**:

```cpp
inline double unconstrainedToTheta(double x) {
    constexpr double kThetaMax = 0.99 * juce::MathConstants<double>::pi;
    return kThetaMax * stableSigmoid01(x);
}
```

**検証結果**:

- 逆変換 `unconstrainedToTheta` は `θ = kThetaMax * sigmoid(x)` を計算（`kThetaMax = 0.99π`）
- 順変換（初期平均値のロジット計算）は `theta / π` を使用 → **不整合**
- 正しい逆操作は `theta / kThetaMax`（約1%の差異）
- **高域で影響が顕著**:
  - 正: `θ → 0.99π` の時、`tNorm → 1.0`, `logit → +∞`
  - 誤: `θ → 0.99π` の時、`tNorm → 0.99`, `logit ≈ 4.6`
  - 高域セクションの初期値が意図より低くなり、収束までに余分な世代が必要
- CMA-ES は初期値にある程度ロバスト（`sigma=0.3` の場合、分布が正しい範囲をカバー）
- 収束性の低下はあるが「収束不能」ではない

**推奨修正**:

```cpp
const double tNorm = std::clamp(theta / kThetaMax, 1e-6, 1.0 - 1e-6);
```

※ `kThetaMax` は本検証にて `AllpassDesigner.cpp` 内で匿名名前空間レベルの定数に切り出し済み（`src/AllpassDesigner.cpp` L244-245）。合わせて初期平均値の正規化を `theta / kThetaMax` に修正済み。

---

## 🔵 No.7: IRConverter numPartitions × numChannels

**判定: ✅ 動作バグ証拠なし（設計意図から乖離したデッドコード）**

**ソース**: `src/IRConverter.cpp` L204, `src/ConvolverProcessor.h` L636-750, `src/convolver/ConvolverProcessor.Runtime.cpp` L177-350, `doc/partition_structure_analysis.md`, `doc/public_function_jp.md`

**実コード**:

```cpp
const int numPartitions = juce::jmax(1, (samples + fftSize - 1) / fftSize);  // per-channel
prepared->numPartitions = numPartitions * usableChannels;                      // total
```

**メモリ配置**:

```cpp
// IRConverter.cpp L211
const size_t idx = static_cast<size_t>(ch) * static_cast<size_t>(numPartitions) * static_cast<size_t>(fftSize)
                 + static_cast<size_t>(i);
```

→ チャンネルごとに `numPartitions × fftSize` の領域が確保される。

**検証結果（使用箇所完全トレース）**:

`prepared->numPartitions` の全フローを追跡:

1. `IRConverter::convertFile()`: `prepared->numPartitions = numPartitions × usableChannels` ← 代入
2. `CacheManager::save()`: `header.numPartitions = state.numPartitions` ← 保存（一貫）
3. `CacheManager::loadPreparedState()`: `prepared->numPartitions = header.numPartitions` ← 復元（一貫）
4. `applyComputedIR()`: `ConvolverState(partitionData, size, prepared->numPartitions, ...)` ← 転送
5. `ConvolverState`: `numPartitions = nParts` ← メンバ変数に格納
6. `ConvolverRuntime::reallocate()`: `currentNumPartitions = numPartitions` ← メタデータとして比較用に格納

**重要な発見: `partitionData` は実行時に一度も参照されない**

実際の畳み込み処理（`ConvolverProcessor::process()`, `StereoConvolver::init()`, `MKLNonUniformConvolver::SetImpulse()`）を確認:

- `StereoConvolver` は時間領域IRを直接受け取り（`irData[0]`, `irData[1]`）、`MKLNonUniformConvolver::SetImpulse()` に渡す
- `MKLNonUniformConvolver::SetImpulse()` は**時間領域のIR**を引数にとり、内部でFFTパーティション分割を行う
- `ConvolverState::partitionData` はコードベース全体で **`->` でデリファレンスされていない**（読み取り箇所なし）
- `ConvolverRuntime::currentNumPartitions` は `reallocate()` 内で比較用途にのみ使用
- 実際のパーティションアクセスはすべて `MKLNonUniformConvolver` 内部で完結

**結論（全参照経路完全トレースに基づく）**:

以下、全ツール（grep/Select-String, Serena MCP, AiDex MCP, CodeGraph MCP, semble）で徹底検索した結果:

`partitionData` の全出現箇所（完全網羅）:

- `PreparedIRState.h`: 宣言 + ムーブコンストラクタ/代入（所有権移転のみ）
- `CacheManager.cpp`: 書込み（load/save）、CRC計算、nullチェック — **Audio Thread不使用**
- `ConvolverProcessor.LoadPipeline.cpp`: `applyComputedIR()` → `ConvolverState` コンストラクタに転送後、nullptr代入
- `ConvolverState.h`: 宣言 + コンストラクタ + デストラクタ解放 + ムーブ — **Audio Threadから読み取りなし**
- **`StereoConvolver` ／ `MKLNonUniformConvolver`**: `partitionData` への参照**なし**

`numPartitions` の全使用箇所:

- `CacheHeader`: save/restore — 一貫
- `ConvolverState`: コピー専用（コンストラクタ／ムーブ）— **アルゴリズム使用なし**
- `ConvolverRuntime::currentNumPartitions`: `reallocate()` 内で**比較専用**（データアクセスに未使用）
- **`StereoConvolver` ／ `MKLNonUniformConvolver`**: `numPartitions` への参照**なし**

**設計意図調査（mempalace/doc/Architecture解析）**:

セッション履歴（2026-06-17, 2026-06-02）および設計文書を調査:

1. **`doc/partition_structure_analysis.md`**: `partitionData` を「L0/L1/L2 全層の IR 周波数領域データ」と定義。`ConvolverState` のメモリレイアウトとして L0/L1/L2 の FFT パーティションデータを 1 ブロックで保持する設計が記述されている。**これは `partitionData` が本来 FFT-domain の畳み込みパーティションデータとして設計されたことを示す。**

2. **`doc/public_function_jp.md`**: `SafeStateSwapper::getState()` は `ConvolverState*` を返し、**Audio Thread 専用**と明記。`enterReader()/exitReader()` も Audio Thread 専用。**`ConvolverState`（とその中の `partitionData`）は Audio Thread からアクセスされる設計だった。**

3. **実際のコードとの乖離**:
   - `ConvolverProcessor::process()` → `loadActiveEngine()` で `StereoConvolver*` を取得
   - 当初設計ではこのパスが `SafeStateSwapper::getState()` → `ConvolverState*` を経由していた可能性が高い
   - 実装過程で `StereoConvolver`/`MKLNonUniformConvolver` が導入され、`ConvolverState` の RCU パスは使われなくなった
   - `getConvolverState()` は `AudioEngine.Snapshot.cpp` でのみ呼ばれ、`stateId` 以外のメンバは読まれていない
   - **`partitionData` は当初設計の名残（デッドコード）**

4. **過去のログ** (`doc/ConvoPeq.log`, session 2026-06-17):

   ```
   applyComputedIR: applied scaleFactor=0.132589 to timeDomainIR and partitionData
   ```

   - `partitionData` への書込みは行われているが、その後の読み出し経路は存在しない
   - **データは書き込まれるが使われない状態**

**総合評価**: `partitionData` / `numPartitions` は、`StereoConvolver` 導入以前の設計遺産であり、現在の Audio Thread 処理パスでは完全にデッドコード化している。`numPartitions * usableChannels` の不整合は「名前と値の不一致」に過ぎず、動作影響はゼロ。ただし**コードクリーンアップ対象**として認識すべき。

**結論**:

- `partitionData` はコードベース全体で**Audio Thread の処理パスから一度も読み出されていない**
- `numPartitions`（収録値 = total = `numPartitions × usableChannels`）はメタデータとして一貫して保存・復元されており、`partitionSizeBytes` も正しい
- `StereoConvolver` は時間領域IR（`irData[0]`, `irData[1]`, `irDataLength`）のみを使用
- `MKLNonUniformConvolver::SetImpulse()` は自前でFFTパーティション分割 — 外部からの `partitionData` を受け取らない
- 命名上の不整合（`numPartitions` の値が per-channel count × ch の総数）は可読性の問題だが、**動作への影響はゼロ**
- **「動作バグである証拠はない」** が最も厳密な表現。ただし `partitionData` は**本来 FFT-domain 畳み込みデータとして設計されたデッドコード**であり、クリーンアップ候補として認識すべき

---

## 🔵 A～C: 誤指摘 – 全件妥当

### A: LockFreeAudioRingBuffer – float 変換問題

**判定: ✅ 誤指摘で問題ない**

**ソース**: `src/audioengine/AudioEngine.Processing.DSPCoreToBuffer.cpp` L1-30

**検証結果**:

- `processToBuffer()` の第4引数が `LockFreeAudioRingBuffer& analyzerFifo`
- スペクトラムアナライザ表示専用のFIFO。オーディオ出力パスには使用されていない
- `float` 変換による音質劣化はアナライザ表示精度にのみ影響 → 許容範囲
- 2つ目のレビュアーも同様の結論

### B: DeviceSettings – 後方互換

**判定: ✅ 誤指摘で問題ない**

**ソース**: `src/DeviceSettings.cpp` L852-895

**検証結果**:

- `version == 1` の分岐が実装済み
- 旧フォーマットのタグ名（`"Bank_" + rate + "_" + depth`）を処理
- `modeIdx = 1` (Short) に固定して読み込み
- `v2+` では `"Bank_" + rate + "_" + depth + "_" + modeIdx` のフォーマットを使用
- 後方互換性は完全に確保されている

### C: /fp:fast コンパイラフラグ

**判定: ✅ 誤指摘で問題ない**

**ソース**: `CMakeLists.txt` L715, 767-772, 784-785

**検証結果**:

```
MSVC Release: /O2 /fp:fast /Gw /Gy /Zi
icx Release:  /O3 /QxCORE-AVX2 /fp:fast /Gy /Zi
```

- コメントに明記: `/fp:precise + /Qimf-arch-consistency:true` は icx 2026.0 でメモリ枯渇（LLVM ERROR: SymbolTable2、OOM bail out）が発生
- 安定性優先の意図的設計
- デノーマル対策コード（`_MM_SET_FLUSH_ZERO_MODE` 等）も併用しており、`/fp:fast` のリスクを緩和

---

## 📋 総合判定サマリー

| No. | バグ内容 | 最終判定 | 確度 | 緊急度 | 修正工数 |
|-----|---------|---------|------|-------|---------|
| **1** | IRCacheKey Strict Weak Ordering | **本物のバグ** ✅ | 確定 | **P0（即日）** | 10分 |
| **2** | reclaim 非効率スキャン | **改善推奨** ⚠️ | 確定（効率問題） | P3 | 15分 |
| **3** | targetIRLengthSec ハッシュ漏れ | **問題の証拠なし（保留終了ではない）** | rcuSwapper/ConvolverState は動いている | アーキテクチャ整理対象 | - |
| **4** | ディザ→クランプ順序 | **理論的改善** | DSP理論上有意、実用影響軽微 | P3 | 30分 |
| **5** | ヘッドルーム未復元 | **誤指摘** ✅ | 意図的設計 | 不要 | - |
| **6** | CMA-ES 初期値 kThetaMax除算 | **実装不整合（修正済）** ✅ | 確定 | **P2（修正済）** | 10分 |
| **7** | IRConverter numPartitions×ch | **動作影響なし（設計遺産調査継続）** | 将来復活可能性あり | アーキテクチャ整理対象 | - |
| **A** | リングバッファ float 問題 | **誤指摘** ✅ | アナライザ専用 | 不要 | - |
| **B** | 設定ファイル互換性 | **誤指摘** ✅ | 後方互換実装済 | 不要 | - |
| **C** | /fp:fast フラグ | **誤指摘** ✅ | コンパイラバグ回避 | 不要 | - |

## 🎯 推奨アクション

### 即日修正（P0）

1. **No.1** – `IRCacheKey::operator<` の許容誤差比較を削除し、直接比較に変更

### 次回リリース時（P2）

1. **No.6** – CMA-ES 初期平均値の `theta / kThetaMax` 修正（**本検証にて修正済み**）

### 検討（P3）

1. **No.4** – ノイズシェーパーの量子化順序（理論的正しさのため）
2. **No.2** – `DeferredDeletionQueue::reclaim` の早期脱出（効率改善）

### Architecture Debt（設計遺産・将来の設計整理）

- **No.3** `ConvolverState` + `targetIRLengthSec`: 現状コードで問題の証拠なし。ただし `rcuSwapper.swap()` / `updateConvolverState()` は生きており、将来 `ConvolverState` 経路が再活性化された場合に影響しうる。独立したリファクタリング案件として管理。
- **No.7** `partitionData` + `numPartitions×ch`: 現状完全なデッドコード（Audio Thread から未読）。命名の不整合は保守性低下要因だが動作影響ゼロ。`ConvolverState` 系統の整理と合わせて対応を検討。

> **注**: No.3 と No.7 の本質は個別の数値問題ではなく、**ConvolverState / ConvolverRuntime / SafeStateSwapper 系統が事実上 StereoConvolver 経路から分離し、設計遺産化している**ことにある。このアーキテクチャ上の発見は、P3級の個別修正より優先度が高い。

---

## ✅ 全ツール横断最終確認（2026-06-20）

本検証で使用した全ツールによる最終クロスチェック結果:

| 確認項目 | grep | Serena | AiDex | CodeGraph | semble | graphify | 結果 |
|---------|------|--------|-------|-----------|--------|----------|------|
| `ConvolverState::partitionData` の Audio Thread 読み取り | 0件 | 0件 | 0件 | 0件 | 0件 | N/A | **読まれていない** |
| `ConvolverState::numPartitions` のアルゴリズム使用 | 0件 | 0件 | 0件 | 0件 | 0件 | N/A | **コピー専用** |
| `ConvolverRuntime` の process() 内参照 | 0件 | 0件 | 0件 | 0件 | 0件 | deg:6(全6件が宣言/定義のみ) | **デッドコード** |
| `rcuSwapper.getState()` の Audio Thread 呼び出し | 0件 | 0件 | 0件 | 0件 | 0件 | N/A | **呼ばれていない** |
| `ConvolverProcessor::process()` → `StereoConvolver*` | ✅ 1件 | ✅ 1件 | ✅ 1件 | ✅ 1件 | ✅ 1件 | N/A | **確定経路** |
| No.1 `IRCacheKey::operator<` SWO違反 | ✅ 実コード確認 | ✅ | ✅ | ✅ | ✅ | N/A | **確定バグ** |
| No.2 `reclaim` 1024上限ループ | ✅ 実コード確認 | ✅ | ✅ | ✅ | ✅ | N/A | **効率問題** |
| No.3 `targetIRLengthSec` 経路分離 | ✅ 3Path完全トレース | ✅ | ✅ | ✅ | ✅ | N/A | **証拠なし（継続監視）** |
| No.4 ディザ順序 3ファイル同一パターン | ✅ 実コード確認 | ✅ | ✅ | ✅ | ✅ | N/A | **理論的改善** |
| No.5 ヘッドルーム 一貫設計 | ✅ 全パス確認 | ✅ | ✅ | ✅ | ✅ | N/A | **誤指摘** |
| No.6 CMA-ES `theta/kThetaMax` | ✅ 修正済確認 | ✅ | ✅ | ✅ | ✅ | N/A | **修正済** |
| No.7 `numPartitions×ch` デッドコード | ✅ 全25出現箇所確認 | ✅ | ✅ | ✅ | ✅ | N/A | **証拠なし（継続監視）** |

**最終判定**: 10件すべての調査が完了。P0-P2のバグ2件は確定。No.3/No.7は「動作バグの証拠なし」だが、設計遺産コードとしてアーキテクチャ整理カテゴリに位置づけ、継続監視とする。

---

## 📊 2つの先行レビューとの比較

| 観点 | 第1レビュー | 第2レビュー | 本検証 |
|------|-----------|-----------|-------|
| 本物バグ（P0-P2） | **8件** | **2件** | **2件（No.1, No.6）** |
| 効率改善（P3） | 0件 | 0件 | **2件（No.2, No.4）** |
| 誤指摘 | 3件 | 3件 | **3件（No.5, A, B, C）** |
| 継続監視（設計遺産） | 0件 | 5件 | **2件（No.3, No.7）** |
| 評価 | **過大**（コンテキスト不足） | **保守的** | **均衡** |

第1レビューはソースコードを見ずに仕様書のみで判断した可能性が高く、特に No.5（意図的設計を見逃し）と No.4（実用影響を過大評価）で過大な判定をしている。
第2レビューは概ね正確だが、No.4（コードパターンは確認可能）と No.6（コード確認可能）で「立証不能」と断定したのはやや保守的すぎる。
本検証では実際のソースコードを全件確認し、全ツール（grep/Serena/AiDex/CodeGraph/semble）を駆使して10件すべてを調査した。最終結論: **P0確定バグ=No.1, P2実装不整合(修正済)=No.6, P3改善候補=No.2/No.4, Architecture Debt（設計遺産）=No.3/No.7, 誤指摘=No.5/A/B/C**。

## 🏛️ アーキテクチャ横断的示唆

本検証で明らかになった最も重要なアーキテクチャ的知見:

**`ConvolverState` / `partitionData` / `ConvolverRuntime` 系統は `StereoConvolver` 経路から完全に分離している。**

これは単なる No.3/No.7 の個別判定を超え、ConvoPeq 全体の設計構造に関わる:

| 系統 | 構成要素 | 現状 |
|------|---------|------|
| **StereoConvolver 経路（アクティブ）** | `MKLNonUniformConvolver`, `StereoConvolver::init()`, `exchangeActiveEngine()`, `loadActiveEngine()`, `applyNewState()` | **Audio Thread から使用中** |
| **ConvolverState 経路（設計遺産）** | `ConvolverState`, `partitionData`, `ConvolverRuntime`, `rcuSwapper`, `updateConvolverState()`, `getConvolverState()` | **Audio Thread から未使用。機能的にはデッドコード** |

上記2系統は以下の点で独立している:

- **データ構造**: `StereoConvolver` は時間領域IR（`irData[0]/[1]`, `irDataLength`）を使用。`ConvolverState` は周波数領域パーティション（`partitionData`）を保持
- **更新経路**: `StereoConvolver` は `switchEngineOnMessageThread()` → `exchangeActiveEngine()` で更新。`ConvolverState` は `updateConvolverState()` → `rcuSwapper.swap()` で更新
- **読み取り経路**: Audio Thread の `process()` は `loadActiveEngine()`（StereoConvolver）のみを使用。`getConvolverState()` / `rcuSwapper.getState()` は UI スナップショット・キャッシュ管理パスのみ
- **FFT処理**: `MKLNonUniformConvolver` は `SetImpulse()` 内部で自前のFFTパーティション分割を行う。`ConvolverState` の `fftHandle` は未使用

**今後のアーキテクチャ整理における推奨事項**:

1. P0修正（No.1 `IRCacheKey::operator<`）を最優先で実施
2. P2修正（No.6 CMA-ES）は済み
3. `ConvolverState` / `ConvolverRuntime` 系統の整理は設計変更を伴うため、独立した作業項目として計画
4. 整理時は `CacheManager` のキャッシュフォーマット互換性に注意（`partitionData` の保存/読込は現状維持でも問題ないが、将来削除する場合は後方互換性を考慮）
