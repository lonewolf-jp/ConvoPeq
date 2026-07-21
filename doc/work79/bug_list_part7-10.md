# ConvoPeq バグリスト（Part 7〜10）

**作成日**: 2026-07-21
**対象ファイル**: Part 7〜10 の調査結果を統合
**調査範囲**: `ISRRetire.cpp`, `EQProcessor.Coefficients.cpp`, `DSPCoreFloat.cpp`, `ConvolverProcessor.MixedPhase.cpp`

---

## バグ一覧（サマリ）

| Finding # | 重大度 | 状態 | ファイル | 概要 |
|-----------|--------|------|----------|------|
| #9 | **High（潜在的）** | 未修正 | `ISRRetire.cpp` | `emitRetireIntentRT()` の命名が誤解を招き、内部の mutex が RT-safe でない |
| #10 | **Medium** | 未修正 | `ConvolverProcessor.MixedPhase.cpp` | MKL 関連バッファに `std::vector`/`std::make_unique` を使用（規約違反の系統的再発） |

> Part 8（`EQProcessor.Coefficients.cpp`）と Part 9（`DSPCoreFloat.cpp`）では **新規バグなし**（良好な結果）。

### 検証結果（2026-07-21）

| Bug ID | 状態 | 検証結果 |
|--------|------|----------|
| Bug 1 | ✅ **修正済み** | `AudioBlock.cpp`/`BlockDouble.cpp` で DSP null 時にバッファクリア済み |
| Bug 2 | ⚠️ **未修正** | `processBypassWithLatencyCompensation` で delayBuffer null 時にバッファクリアなし |
| Bug 3 | ⚠️ **未修正** | `copyLatest` の TOCTOU リスク（読み取り順序が Writer と逆） |
| Bug 4 | ✅ **修正済み** | `performLoad` で全例外パターンをキャッチ |
| Bug 5 | ✅ **修正済み** | `makeEngineRuntimeState` に null 安全ガード追加 |
| Bug A | ⚠️ **未修正** | `killDenormal()` は Release で no-op（NaN/Inf 通過） |
| Bug B | ⚠️ **未修正** | `quantize()` で NaN が clamp 通過 |
| Bug C | ⚠️ **未修正** | `pushBlock` で境界チェック欠如 |
| Bug D | ⚠️ **未修正** | `Fixed15TapNoiseShaper` で fb NaN/Inf 未チェック |
| Bug E | ⚠️ **未修正** | Bug 3 と同じ TOCTOU 問題 |
| Bug F | ⚠️ **未修正** | `StereoConvolver::init` の空ブロック |
| Bug G | ⚠️ **未修正** | 冗長な負値チェック（C++20 では常に false） |
| Bug H | ⚠️ **未修正** | 例外安全性（bad_alloc のみキャッチ） |

---

## Finding #9 — 【High（潜在的）】emitRetireIntentRT() の誤解を招く命名と内部の mutex

### 概要

`emitRetireIntentRT()` は関数名から「Audio Thread（RT）から安全に呼べる版」を強く示唆するが、実装は `emitRetireIntent()` を素通しで呼ぶだけであり、**mutex ロック経路を含めて完全に同一のコードパスを通る**。関数名が実態と逆である。

### 該当コード

**ファイル**: `src/audioengine/ISRRetire.cpp`

```cpp
void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    const uint64_t ticket = convo::fetchAddAtomic(enqueueTicket_, 1, std::memory_order_acq_rel);
    const size_t idx = ticket % RETIRE_INTENT_QUEUE_SIZE;
    RetireIntent localIntent = intent;

    static constexpr int kMaxProducerSpin = 64;
    for (int spin = 0;; ++spin) {
        uint64_t slotSeq = convo::consumeAtomic(slots_[idx].sequence, std::memory_order_acquire);
        if (slotSeq == ticket) break;

        if (spin >= kMaxProducerSpin) {
            slots_[idx].payload = RetireIntent{};
            slots_[idx].payload.dspSlot = UINT32_MAX;
            convo::publishAtomic(slots_[idx].sequence, ticket + 1, std::memory_order_release);

            std::lock_guard<std::mutex> lock(fallbackMutex_);   // ← RT-safe でない
            // ...
            return;
        }
        _mm_pause();
    }
    // ...
}

void RetireRuntime::emitRetireIntentRT(const RetireIntent& intent) noexcept
{
    emitRetireIntent(intent);  // mutex 経路を含め完全に同一パス
}
```

### 呼び出し元調査（現状: 非RT のみ）

| 呼び出し元 | ファイル | スレッド判定 |
|-----------|----------|-------------|
| `emitRetireIntentRT()` | `AudioEngine.Commit.cpp` (`onRuntimeRetiredNonRt()` 内) | 非RT (`ASSERT_NON_RT_THREAD()` あり) |
| `emitRetireIntent()` ×2 | `AudioEngine.Processing.ReleaseResources.cpp` | 非RT (JUCE `releaseResources()`) |
| `emitRetireIntent()` ×1 | `AudioEngine.Timer.cpp` | 非RT (50ms timerCallback) |
| `emitRetireIntent()` ×3 | `ISRRuntimePublicationCoordinator.cpp` | 非RT (50ms timerCallback 経由) |

**現時点では Audio Thread から呼ばれている経路は見つからず、mutex ロックが Audio Thread 上で発生するリスクは現行コードにはない。**

### リスク評価

- 関数名 `emitRetireIntentRT` が「RTスレッドから呼んでよい」と解釈されやすく、将来 Audio Thread から誤呼び出しを追加するリスクが高い
- 通常時（spin 成功時）は正常動作するため、開発中のテストでは発覚しにくい
- 輻輳時にのみ mutex ロックが発生するため、再現困難な間欠的グリッチの原因になり得る

### 推奨対応

1. 関数名を実態に即したもの（例: `emitRetireIntentFromNonRTCaller`）に変更、または関数直上に「RT は RealTime thread safety を意味しない」旨のコメントを明記
2. 将来 Audio Thread から直接呼びたい場合は、mutex を使わない別実装（drop + カウンタ増加、またはロックフリーのオーバーフローリング）を用意
3. Debug ビルドで `emitRetireIntent()` 内の mutex 取得箇所に `ASSERT_NON_RT_THREAD()` 相当のガードを追加

### 出典

- Part 7: `ConvoPeq_bug_report_2026-07-18_part7.md` Finding #9

---

## Finding #10 — 【Medium】MKL 関連バッファへの std::vector / std::make_unique 使用の系統的再発

### 概要

`ConvolverProcessor.MixedPhase.cpp` の `convertToMixedPhaseAllpass` 関数内で、MKL DFTI API を直接呼ぶ関数に `std::vector<double>` / `std::make_unique` が 7 箇所使用されている。Finding #2（`IRAnalyzer.cpp`）と同型の規約違反。

### 該当コード

**ファイル**: `src/convolver/ConvolverProcessor.MixedPhase.cpp`（`convertToMixedPhaseAllpass` 内）

```cpp
if (DftiComputeForward(dfti.handle, linearSpec.get()) != DFTI_NO_ERROR) return {};
if (DftiComputeForward(dfti.handle, minimumSpec.get()) != DFTI_NO_ERROR) return {};

std::vector<double> phiMinUnwrapped(static_cast<size_t>(complexSize));   // ← 規約違反
phiMinUnwrapped[0] = std::atan2(minimumSpec.get()[0].imag, minimumSpec.get()[0].real);
```

同一関数内で `convo::makeAlignedArray` によるアライン済み確保と、`std::vector` による規約違反の確保が混在。

### 違反箇所一覧

| 変数 | 種別 | 用途（推定） |
|------|------|-------------|
| `cachedRho`, `cachedTheta` | `std::vector<double>` | ディスクキャッシュ読み込み先 |
| `entry.ir` | `std::make_unique<juce::AudioBuffer<double>>` | キャッシュエントリ |
| `phiMinUnwrapped` | `std::vector<double>` | 最小位相スペクトルの位相アンラップ |
| `targetGroupDelayStd`, `optim_freq_hz`, `optim_target_gd` | `std::vector<double>` | Allpass 設計の最適化ターゲット |
| `freq_hz` | `std::vector<double>` | 周波数軸配列 |
| `entry.ir`（2 箇所目） | `std::make_unique<juce::AudioBuffer<double>>` | 変換結果のキャッシュ登録 |
| `rho`, `theta` | `std::vector<double>` | Allpass 係数 |

### 重大度・実害の評価

- 本関数は非 RT スレッド（IR ロード/変換パイプライン）専用であり、Audio Thread 規約そのものへの抵触ではない
- 実害は「規約からの逸脱」自体であり、一貫性の欠如が将来のメモリ管理判断を誤らせるリスク
- `noexcept` 修飾子の有無は要確認（`juce::AudioBuffer<double>` を返り値とするシグネチャのため、投げ越しの可能性あり）

### 推奨対応

プロジェクト全体を対象にした機械的監査を推奨:

1. `#include <mkl.h>` または `#include <mkl_dfti.h>` を含む全ファイルを列挙
2. それらのファイル内で `std::vector`、`std::make_unique`、裸の `new` の出現箇所を洗い出す
3. 各出現箇所が MKL API 呼び出しと同一関数スコープ内にあるかを確認
4. Audio Thread から到達しないことを確認した上で、`convo::makeAlignedArray<T>` 等の既存ラッパーへ一括置換

### 出典

- Part 10: `ConvoPeq_bug_report_2026-07-18_part10.md` Finding #10

---

## バグなし確認ファイル（Part 8〜9）

### Part 8: `EQProcessor.Coefficients.cpp`（686行）

Biquad 係数（5 種類）と SVF 係数（5 種類）を、参照文献（Audio EQ Cookbook / Cytomic TPT SVF）と数式レベルで突き合わせ。**全 10 種類のフィルタタイプで完全に一致**。バグなし。

### Part 9: `AudioEngine.Processing.DSPCoreFloat.cpp`（471行）

Float 版と Double 版の `process()` / `processDouble()` を入力〜 bypass ブレンドまで突き合わせ。コードスタイルの差異はあるが、**実行順序は完全に同一**。バグなし。

---

## 累積 Finding 番号の振り分け（Part 1〜10）

| Finding # | 重大度 | ファイル | Part |
|-----------|--------|----------|------|
| #1〜#8 | — | Part 1〜6 で報告済み | Part 1〜6 |
| #9 | High（潜在的） | `ISRRetire.cpp` | Part 7 |
| #10 | Medium | `ConvolverProcessor.MixedPhase.cpp` | Part 10 |

---

## 調査済みファイル一覧（Part 7〜10）

| ファイル | 行数 | 結果 |
|---------|------|------|
| `ISRRetire.cpp` | 273 | Finding #9 発見 |
| `EQProcessor.Coefficients.cpp` | 686 | バグなし |
| `AudioEngine.Processing.DSPCoreFloat.cpp` | 471 | バグなし（出力段は未突き合わせ） |
| `ConvolverProcessor.MixedPhase.cpp` | 869 | Finding #10 発見（残り ~300行は未読） |

---

## 今後の推奨ステップ

1. `ConvolverProcessor.MixedPhase.cpp` の残り（`convertToMixedPhaseFallback` 以降）
2. `ConvolverProcessor.Rebuild.cpp` / `ResampleAndFallback.cpp` / `StateAndUI.cpp`
3. **MKL ファイル横断監査**（`#include <mkl.h>` ファイル × `std::vector`/`make_unique` 出現箇所）
4. `DSPCoreFloat.cpp` / `DSPCoreDouble.cpp` の出力段突き合わせ継続
5. ISR 系残り約 12 ファイル
6. `EQProcessor.h`（1225行）

---

## 検証結果サマリ（2026-07-21 実施）

### 検証方法
- seemble v0.5.2 によるコード検索
- ソースコード直接読み込みによる確認
- 呼び出し元・呼び出し先の横断調査

### 検証結果一覧

| Bug ID | 状態 | 検証結果 |
|--------|------|----------|
| Finding #9 | **未修正** | ✅ 確認。`emitRetireIntentRT()` は `emitRetireIntent()` を素通し。呼び出し元は全て非RT（`ASSERT_NON_RT_THREAD()` 確認済み）。命名リスクは継続。 |
| Finding #10 | **未修正** | ✅ 確認。`ConvolverProcessor.MixedPhase.cpp` に `std::vector<double>` 7箇所。`ConvolverProcessor.ResampleAndFallback.cpp` にも `std::vector<int>` あり。横断監査推奨。 |
| Bug 1 | **修正済み** | ✅ 確認。`AudioBlock.cpp` と `BlockDouble.cpp` ともに `dsp == nullptr` 時に `buffer.clear()` / `bufferToFill.clearActiveBufferRegion()` を正しく実行。 |
| Bug 2 | **未修正** | ✅ 確認。`processBypassWithLatencyCompensation` で `delayBuf` null 時に `return` のみでバッファクリアなし。 |
| Bug 3 | **未修正** | ✅ 確認。`copyLatest` で `totalSamples` → `writePosition` の順で読み取り（Writer と逆順）。TOCTOU リスク継続。 |
| Bug 4 | **修正済み** | ✅ 確認。`performLoad` は `std::bad_alloc`、`std::exception`、`...` の全パターンをキャッチ。呼び出し元 `run()` でも `!result.success` 時にクリーンアップ。 |
| Bug 5 | **修正済み** | ✅ 確認。`makeEngineRuntimeState` に `runtimeWorld == nullptr` の安全ガード（fallback EngineRuntime 生成）が追加済み。 |
| Bug A | **未修正** | ✅ 確認。`killDenormal()` は Release ビルドで no-op。NaN/Inf は通過。`DspNumericPolicy.h` で確認。 |
| Bug B | **未修正** | ✅ 確認。`quantize()` で NaN は比較演算が false のため clamp 通過。`FixedNoiseShaper.h` で確認。 |
| Bug C | **未修正** | ✅ 確認。`pushBlock` で `numSamples > kCapacity` のチェックなし。バッファオーバーフローリスク継続。 |
| Bug D | **未修正** | ✅ 確認。`Fixed15TapNoiseShaper.h` の `processSample` で `fb` は `killDenormal()` のみ。Release では NaN/Inf 通過。 |
| Bug E | **未修正** | ✅ 確認。Bug 3 と同じ `copyLatest` の TOCTOU 問題。 |
| Bug F | **未修正** | ✅ 確認。`StereoConvolver::init` で `ownerProcessor != nullptr` 時のブロックが空。 |
| Bug G | **未修正** | ✅ 確認。`processBypassWithLatencyCompensation` で `if (readPos < 0)` は C++20 では冗長。 |
| Bug H | **未修正** | ✅ 確認。`StereoConvolver::init` で `std::bad_alloc` のみキャッチ。他の例外時は `irData` リーク。 |

### 重要な発見

1. **Bug 1 と Bug 5 は修正済み**: バグ報告時に既に修正が適用されていた
2. **Bug 4 も修正済み**: 例外ハンドリングが改善されている
3. **追加問題**: `ConvolverProcessor.ResampleAndFallback.cpp` にも `std::vector<int>` があり、Finding #10 と同型の規約違反の可能性
4. **Bug A/B/D は連鎖**: NaN/Inf が `killDenormal` → `quantize` → 出力と伝播する経路が確認された

---

## 追加バグ報告（bug1 / bug2）

以下は `doc/work79/bug2/bug1.md` および `doc/work79/bug2/bug2.md` から統合した追加バグです。

### 追加バグ一覧

| Bug ID | 重大度 | カテゴリ | ファイル | 概要 |
|--------|--------|----------|----------|------|
| Bug 1 | **重大** | Audio | `AudioEngine.Processing.AudioBlock.cpp`, `AudioEngine.Processing.BlockDouble.cpp` | DSP null 時にバッファ未クリア（stale data 残留） |
| Bug 2 | **中** | Audio | `ConvolverProcessor.h` | `processBypassWithLatencyCompensation` delayBuffer null 時未クリア |
| Bug 3 | **低〜中** | スレッド安全性 | `AudioSegmentBuffer.h` | `copyLatest` 読み取り順序の TOCTOU リスク |
| Bug 4 | **低** | メモリ安全性 | `ConvolverProcessor.h` | `LoaderThread::performLoad` 例外時 newConv リーク |
| Bug 5 | **低** | 安全性 | `AudioEngine.h` | `makeEngineRuntimeState` runtimeWorld null チェック欠如 |
| Bug A | **重大** | 数値安定性 | `FixedNoiseShaper.h`, `Fixed15TapNoiseShaper.h`, `LatticeNoiseShaper.h` | ノイズシェイパー NaN/Inf 伝播（Release で killDenormal が no-op） |
| Bug B | **重大** | 数値安定性 | `FixedNoiseShaper.h`, `Fixed15TapNoiseShaper.h`, `LatticeNoiseShaper.h` | `quantize()` NaN 通過（比較演算が false のため clamp 通過） |
| Bug C | **重大** | メモリ安全性 | `AudioSegmentBuffer.h` | `pushBlock` 境界チェック欠如（バッファオーバーフロー） |
| Bug D | **中** | 数値安定性 | `Fixed15TapNoiseShaper.h` | `processSample` fb NaN/Inf 未チェック |
| Bug E | **中** | スレッド安全性 | `AudioSegmentBuffer.h` | `copyLatest` TOCTOU（totalSamples と writePosition の不整合） |
| Bug F | **低** | コード品質 | `ConvolverProcessor.h` | `StereoConvolver::init` 空ブロック（未実装処理の可能性） |
| Bug G | **低** | コード品質 | `ConvolverProcessor.h` | `processBypassWithLatencyCompensation` 冗長な負値チェック（C++20 では常に false） |
| Bug H | **低** | メモリ安全性 | `ConvolverProcessor.h` | `StereoConvolver::init` 例外安全性（bad_alloc のみキャッチ） |

---

### Bug 1 — 【重大】processBlock / processBlockDouble — DSP null 時に出力バッファ未クリア

**対象ファイル**: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`, `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`

**問題箇所**:
```cpp
// AudioBlock.cpp
DSPCore* dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
if (!dsp)
    return;  // ← BUG: バッファをクリアせずに return

// BlockDouble.cpp — 同一パターン
if (!dsp)
    return;  // ← BUG: buffer.clear() がない
```

**発生条件**: `runtimeWorld` は存在するが `engine.current`（DSPCore*）が null の場合。初期化直後（World 公開後、初回 DSP publish 前）またはシャットダウン中（DSP 退役後、World クリア前）に発生し得る。

**影響**: 出力バッファに前回の処理結果（stale data）が残留し、**音声グリッチ・ノイズ**として出力される。

**修正**:
```cpp
if (!dsp)
{
    bufferToFill.clearActiveBufferRegion();  // AudioBlock.cpp
    // buffer.clear();                       // BlockDouble.cpp
    return;
}
```

**根拠**: JUCE の `getNextAudioBlock` はバッファの事前クリアを保証しないため、null DSP 時は明示的にゼロクリアが必要。

---

### Bug 2 — 【中】processBypassWithLatencyCompensation — delayBuffer null 時に出力未クリア

**対象ファイル**: `src/ConvolverProcessor.h`（`processBypassWithLatencyCompensation`）

**問題箇所**:
```cpp
if (delayBuf[0] == nullptr || delayBuf[1] == nullptr || activeDelayCapacity < DELAY_BUFFER_SIZE)
    return;  // ← BUG: block をクリアせずに return
```

**発生条件**: `prepareToPlay()` 未呼び出し、または `releaseResources()` 後にバイパス遷移中で `process()` が呼ばれた場合。

**修正**:
```cpp
if (delayBuf[0] == nullptr || delayBuf[1] == nullptr || activeDelayCapacity < DELAY_BUFFER_SIZE)
{
    block.clear();
    return;
}
```

---

### Bug 3 — 【低〜中】AudioSegmentBuffer::copyLatest — 読み取り順序の TOCTOU リスク

**対象ファイル**: `src/AudioSegmentBuffer.h`

**問題箇所**:
```cpp
// Writer側: writePosition → totalSamples の順で更新
// Reader側:
const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);   // ①先に読む
const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire); // ②後に読む
```

**問題**: Writer は `writePosition` → `totalSamples` の順で release するが、Reader は逆順で acquire する。release-acquire の連鎖により理論上は安全だが、読み取り順序が Writer の書き込み順序と逆であり、将来の修正で順序が崩れた場合に即座にデータ不整合が発生する脆弱な設計。

**推奨修正**: Writer の書き込み順序と Reader の読み取り順序を一致させる。

---

### Bug 4 — 【低】LoaderThread::performLoad — 例外発生時の newConv リーク可能性

**対象ファイル**: `src/ConvolverProcessor.h`（`LoaderThread::performLoad`）

**問題**: `stepResult.newConv` が設定された後に例外が発生した場合、catch ブロック内でクリーンアップがない。

**緩和要因**: 呼び出し元 `run()` が `!result.success` 時に `retireStereoConvolver` でクリーンアップするため実害は限定的。

---

### Bug 5 — 【低】makeEngineRuntimeState — runtimeWorld null チェック欠如

**対象ファイル**: `src/audioengine/AudioEngine.h`

**問題**: Release ビルドでは `jassert` は no-op のため、`runtimeWorld` が null の場合にクラッシュする。

---

### Bug A — 【重大】ノイズシェイパー NaN/Inf 伝播

**対象**: `FixedNoiseShaper.h`, `Fixed15TapNoiseShaper.h`, `LatticeNoiseShaper.h`

**問題**: `killDenormal()` は Release ビルドで **no-op** であり、NaN/Inf を一切処理しない。フィードバックループが不安定化した場合、NaN/Inf が出力に伝播し、**音声出力が完全に破綻**する。

**修正提案**:
```cpp
inline double killDenormal(double x) noexcept
{
    const auto bits = std::bit_cast<uint64_t>(x);
    const bool isSubnormal = ((bits >> 52) & 0x7FFu) == 0u && (bits & 0x000FFFFFFFFFFFFFULL) != 0u;
    const bool isNanOrInf = ((bits >> 52) & 0x7FFu) == 0x7FFu;
    return (isSubnormal || isNanOrInf) ? 0.0 : x;
}
```

---

### Bug B — 【重大】quantize() の NaN 通過

**対象**: `FixedNoiseShaper.h`, `Fixed15TapNoiseShaper.h`, `LatticeNoiseShaper.h` の `quantize()`

**問題**: NaN は比較演算が常に false を返すため、clamp を通過して出力に伝播する。

**修正提案**:
```cpp
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    if (!std::isfinite(v)) v = 0.0;
    // ... 既存の clamp 処理
}
```

---

### Bug C — 【重大】AudioSegmentBuffer::pushBlock の境界チェック欠如

**対象**: `AudioSegmentBuffer.h`

**問題**: `numSamples > kCapacity` の場合、リングバッファのラップアラウンド計算が破綻し、**バッファオーバーフロー**が発生する。

**修正提案**:
```cpp
void pushBlock(const double* left, const double* right, int numSamples) noexcept
{
    if (left == nullptr || right == nullptr || numSamples <= 0)
        return;
    numSamples = std::min(numSamples, kCapacity);
    // ...
}
```

---

### Bug D — 【中】Fixed15TapNoiseShaper::processSample の fb NaN/Inf 未チェック

**対象**: `Fixed15TapNoiseShaper.h`

**問題**: フィードバック値 `fb` は `killDenormal()` のみで処理され、NaN/Inf がチェックされない。Bug A と同一の伝播経路。15 次フィルタは 4 次より発散リスクが高い。

---

### Bug E — 【中】AudioSegmentBuffer::copyLatest の TOCTOU

**対象**: `AudioSegmentBuffer.h`

**問題**: `totalSamples` と `writePosition` を別々に読み取るため、読み取り間に書き込みが発生すると不整合なデータを読み取る可能性がある。`currentTotal >= kCapacity` の場合に書き込み中のデータを読み取る可能性あり。

---

### Bug F — 【低】StereoConvolver::init の空ブロック

**対象**: `ConvolverProcessor.h`

**問題**: `ownerProcessor` が非 null の場合の処理ブロックが空。意図された処理（例: レイテンシ通知）が欠落している可能性。

---

### Bug G — 【低】processBypassWithLatencyCompensation の冗長な負値チェック

**対象**: `ConvolverProcessor.h`

**問題**: C++20 では二の補数が義務付けられているため、`if (readPos < 0)` チェックは常に false。C++20 以前のコンパイラでコンパイルした場合に問題になる可能性。

---

### Bug H — 【低】StereoConvolver::init の例外安全性

**対象**: `ConvolverProcessor.h`

**問題**: `try-catch` が `std::bad_alloc` のみキャッチ。`SetImpulse` が他の例外を投げた場合、`irData` がリークする。
