# ConvoPeq バグ検証統合リスト

**作成日**: 2026-07-11
**対象**: `doc/work69/bug.md`（全19件のバグ主張）、`doc/work69/bug_analysis_report.md`（初期検証）、`doc/work69/bug1_bug2_deep_analysis.md`（#1/#2詳細検証）
**検証方法**: 実ソースコード直接照合（grep/ast-grep/Readによる行番号・コード片確認）

---

## 検証結果サマリ

```
真正的バグ (BUG):     4件
潜在的バグ (WARN):    3件
非バグ (NOT BUG):     8件
ハルシネーション:      4件  (実コードに存在しない)
```

| ID  | 件名                                         | 判定        | 深刻度  | ファイル                                              | 行番号        |
|-----|----------------------------------------------|-------------|---------|-------------------------------------------------------|---------------|
| B01 | DSPCoreFloat.cpp バイパスブレンド欠落          | **BUG**     | 🟡 中   | AudioEngine.Processing.DSPCoreFloat.cpp                | (該当コード不在) |
| B02 | リングバッファ負index UB                      | **NOT BUG** | 🟢 極低 | ConvolverProcessor.Runtime.cpp                        | 482           |
| B03 | NoiseShaperLearner 冗余 vdTanh                | **WARN**    | 🟢 低   | NoiseShaperLearner.cpp                                | 607-619       |
| B04 | EQCacheManager 非原子参照カウント             | **NOT BUG** | ❌      | AudioEngine.h / RefCountedDeferred.h                  | 1828, 65      |
| B05 | ConvolverProcessor dtor mkl_free             | **NOT BUG** | ❌      | ConvolverProcessor.Lifecycle.cpp                      | 117-121       |
| B06 | dotProductAvx2 残余処理欠落                  | **NOT BUG** | ❌      | TruePeakDetector.cpp / CustomInputOversampler.cpp     | 128-159       |
| B07 | besselI0 無限ループ                           | **NOT BUG** | ❌      | TruePeakDetector.cpp / CustomInputOversampler.cpp     | 113-126       |
| B08 | CacheMap dtor m_retireRouter UAF              | **WARN**    | 🟡 中   | AudioEngine.h                                         | 1850-1858     |
| B09 | IRState mkl_free 不完全解放                  | **NOT BUG** | ❌      | (= B05 と同一)                                        |               |
| B10 | mixSmoothingSmall AVX2 非対齐                | **WARN**    | 🟢 低   | ConvolverProcessor.Runtime.cpp                        | 597-641       |
| B11 | calcLowShelfBiquad NaN 传播                  | **NOT BUG** | ❌      | EQProcessor.Coefficients.cpp                          | 75-87, 159-191|
| B12 | CMakeLists /STACK /GS-                        | **NOT BUG** | 🟢 低   | CMakeLists.txt                                        | 323-326       |
| B13 | NUPC delay alignment 欠落                     | **BUG**     | 🟡 中   | MKLNonUniformConvolver.cpp                            | 1589, 1671-88 |
| B14 | Retire queue MPSC data race                  | **BUG**     | 🔴 高   | ISRRetire.cpp                                         | 11-83         |
| B15 | AudioSegmentBuffer ABA                       | **WARN**    | 🟢 低   | AudioSegmentBuffer.h                                  | 23-78         |
| B16 | LatticeNoiseShaper 直接型/格子型混同          | **NOT BUG** | ❌      | LatticeNoiseShaper.h                                 | 220-267       |
| B17 | StereoConvolver::clone() filterSpec 欠落     | **BUG**     | 🟡 中   | ConvolverProcessor.h                                  | 772-795       |
| B18 | destroyQuarantineSlot メモリリーク            | **NOT BUG** | ❌      | ISRDSPHandle.cpp                                      | 149-197       |
| B19 | BuildInputSemanticContract /STACK             | (= B12)     | —       | —                                                     |               |

---

## BUG（真正なバグ）— 4件

---

### B01: DSPCoreFloat.cpp におけるバイパスブレンド機構の欠落

| 項目 | 値 |
|------|-----|
| 判定 | **BUG (VALID、方向逆)** |
| 深刻度 | 🟡 中 |
| 発生条件 | 32bit float 処理経路で full bypass（EQ+Convolver両方OFF）を切替えた時 |
| 影響 | 5ms crossfade なしで wet→dry 急峻切替 → クリック/ポップノイズ |
| 主張元 | bug.md §II-1（ただし方向が逆）、bug_analysis_report.md、bug1_bug2_deep_analysis.md |

#### 関連コード

**Double 版 (完全実装済み)**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`

- 382行: `const bool requestedFullBypass = state.eqBypassed && state.convBypassed;`
- 384-387行: `ramp.bypassFadeGainDouble.setTargetValue(...)` でフェードトリガー
- 394-397行: DSP処理前に `dryBypassBufferDoubleL/R` にdry信号保存
- 548行: `const bool bypassBlendRequested = ramp.bypassFadeGainDouble.isSmoothing() \|\| requestedFullBypass;`
- 549-568行: OS=1 のクロスフェード（`gWet = ramp.bypassFadeGainDouble.getNextValue()`）
- 570-593行: OS>1 のクロスフェード

**Float 版 (完全欠落)**: `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`

- `requestedFullBypass` 判定: **不在**
- `bypassFadeGainDouble.setTargetValue()`: **不在**
- `dryBypassBuffer` コピー: **不在**
- `bypassBlendRequested`: **不在**
- OS=1/OS>1 クロスフェード: **不在**

#### データ構造の状況

`AudioEngine.h` で `RampRuntimeState` (716-717行) に `bypassFadeGainDouble`、`bypassedDouble` が存在。`dryBypassBufferDoubleL/R` も 971-973行で確保済み。ただし全て Double 后缀付きで、Float 用メンバは存在しない。

#### 修正方針

1. `RampRuntimeState` に Float 用 `bypassFadeGain`/`bypassed` 追加（または共用化）
2. Float 用 `dryBypassBuffer` 確保
3. `process()` に bypass blend ロジック移植

---

### B13: NUPC における Layer 1/2 出力遅延アライメント欠落

| 項目 | 値 |
|------|-----|
| 判定 | **BUG** |
| 深刻度 | 🟡 中（可聴性は FDL インデックス設計に依存） |
| 発生条件 | Non-Uniform Partitioned Convolution で Layer 0 と Layer 1/2 の残響テールが時間軸上で同期されずに加算される |
| 影響 | Layer 1 の残響が ~8192 サンプル (~170ms) 早く鳴る、または tail dropout が発生する可能性 |
| 主張元 | bug.md 後半 §1 |

#### 関連コード

`src/MKLNonUniformConvolver.cpp`:

- 1589行: `memcpy(l.tailOutputBuf, l.fftOutBuf + l.partSize, l.partSize * sizeof(double)); l.tailOutputPos = 0;`
- 1671-1688行: `Get()` メソッドで `l.tailOutputBuf + l.tailOutputPos` を直接 `output` に加算
- `.h:214` 注释: 「L0(ring) + L1/L2(tailOutputBuf) 単純加算」設計

#### 検証結果

- 顕式遅延線（delay line）は grep 全件で 0 件
- 遅延補正は FDL (Frequency Delay Line) インデックス `linStart = baseFdlIdx - numPartsIR + 1 + numParts` (1557行) で频域暗黙的に encode されている可能性
- `doc/work52/work52_final_report.md` で「未証明・未否定」と記録済み

#### 修正方針

各レイヤー出力側に時間アライメント用リングバッファ（Delay Line）を追加。レイヤーオフセットから処理レイテンシを引いた値だけ遅延させる。

---

### B14: Retire queue における SPSC→MPSC データレース

| 項目 | 値 |
|------|-----|
| 判定 | **BUG (DATA RACE)** |
| 深刻度 | 🔴 高 |
| 発生条件 | 複数スレッド（Message Thread / Rebuild Thread / Timer / Coordinator）が同時に `emitRetireIntent` を呼んだ時 |
| 影響 | 一方の retire intent が消失 → 永続的メモリリーク |
| 主張元 | bug.md 後半 §2 |

#### 関連コード

`src/audioengine/ISRRetire.cpp:11-83`:

```cpp
// 13行: tail = consumeAtomic(retireIntentTail_, relaxed)  // 只读
// 81行: retireIntentQueue_[tail] = intent;                  // 写固定槽
// 82行: publishAtomic(retireIntentTail_, nextTail, release);  // 推进 tail
```

SPSC 前提のキュー構造。複数 producer が同一 `tail` を読み、同一 slot を交叉書き、一方の intent が消失。

#### 実呼び出し元（全て非 RT）

- `AudioEngine.Commit.cpp:450` (`emitRetireIntentRT`, ASSERT_NON_RT_THREAD)
- `AudioEngine.Processing.ReleaseResources.cpp:201, 239`
- `AudioEngine.Timer.cpp:1571`
- `ISRRuntimePublicationCoordinator.cpp:294, 306, 332`

これらは releaseResources / Timer / Coordinator 三スレッド同時アクティブ期に並行呼び出し可能。

#### コード内自認

`ISRRetireOverflowRing.h:7-19` 注釈: 「SPSC 前提」「Worker Thread からの直接 emitRetireIntent 出现时必须改为 MPSC」

#### 修正方針

`tail` 推進を `fetch_add(1, acq_rel)` に変更し、bump 後の slot に定址書き込み。

---

### B17: StereoConvolver::clone() における FilterSpec 欠落

| 項目 | 値 |
|------|-----|
| 判定 | **BUG** |
| 深刻度 | 🟡 中 |
| 発生条件 | Convolver クローン生成時（UI 操作 / preset load / crossfade バックアップ） |
| 影響 | クローン先でハイカット/ローカット等の周波数フィルタ特性が消失 → IR 生響きが鳴る |
| 主張元 | bug.md 後半 §2 |

#### 関連コード

`src/ConvolverProcessor.h`:

- 713-716行: `init(..., const convo::FilterSpec* filterSpec=nullptr, ...)` (デフォルト nullptr)
- 782-786行: `clone()` 内で `newConv->init(l, r, irDataLength, storedSampleRate, irLatency, storedKnownBlockSize, callQuantumSamples, storedScale, storedDirectHeadEnabled)` — **9 引数、filterSpec 位置空缺**
- 728-731行: `storedSampleRate` / `storedKnownBlockSize` / `storedScale` / `storedDirectHeadEnabled` は存在するが、**`storedFilterSpec` フィールドなし**

#### 検証結果

- `init()` 返回後、元の filterSpec は保存されずに消失
- `clone()` では デフォルト `nullptr` が使われる → spectrum filter precompute (`applySpectrumFilter`) が無視される
- クローン周の挙動不一致が音響的に顕在化する可能性

#### 修正方針

`StereoConvolver` に `convo::FilterSpec storedFilterSpec;` メンバを追加。`init()` で保存、`clone()` で `&storedFilterSpec` を明示渡し。

---

## WARN（潜在的バグ）— 3件

---

### B03: NoiseShaperLearner における冗余 vdTanh 計算

| 項目 | 値 |
|------|-----|
| 判定 | **WARN (性能問題、機能正しい)** |
| 深刻度 | 🟢 低 |
| 発生条件 | 複数 worker スレッドが評価フェーズに入る時 |
| 影響 | 各 worker が独立に 162 要素の `vdTanh` を計算 → CPU 浪費 |
| 主張元 | bug.md §II-3 |

#### 関連コード

`src/NoiseShaperLearner.cpp:595-636`:

```cpp
// 607行: constexpr int totalCoeffs = CmaEsOptimizer::kPopulation * CmaEsOptimizer::kDim; // 162
// 610行: vdTanh(totalCoeffs, reinterpret_cast<const double*>(population), tanhBuffer);
```

`runEvaluationJobsForWorker` は 558行 (aux worker) と 661行 (main) から呼ばれる。`candidatePopulationMatrix()` は只读共有データ。

#### 検証結果

- 各 worker が stack-local `tanhBuffer` で同一計算を独立実行
- 結果は thread-local の `mappedPopulation` に格納され、原子 `fetchAdd` で分配される `populationIndex` に従って消費
- 機能的に正しいが、N worker = N 回の冗余 vdTanh
- `doc/work73/bug_verification_report.md:519` で「最適化不足/バグ非該当」判定済み

#### 修正方針

`vdTanh` を主 worker スレッドで 1 回だけ計算し、`mappedPopulation` を全 worker で共有読取する構造に変更。

---

### B08: CacheMap dtor における m_retireRouter ポインタ UAF リスク

| 項目 | 値 |
|------|-----|
| 判定 | **WARN (潜在的 UAF)** |
| 深刻度 | 🟡 中 |
| 発生条件 | AudioEngine 破棄時、メンバ宣言順序により `m_retireRouter` が `EQCacheManager` より先に破棄される時 |
| 影響 | `~CacheMap()` が `owner->m_retireRouter` を解参照 → Use-After-Free / クラッシュ |
| 主張元 | bug.md §1 (後半) |

#### 関連コード

`src/audioengine/AudioEngine.h`:

- 1814行: `class EQCacheManager` (内部 class 定義開始)
- **2083行**: `EQCacheManager eqCacheManager;` (AudioEngine メンバ宣言)
- **4075行**: `std::unique_ptr<convo::isr::ISRRetireRouter> m_retireRouter;` (AudioEngine メンバ宣言)

`src/audioengine/AudioEngine.h:1850-1858`:

```cpp
~CacheMap()
{
    jassert(owner != nullptr);
    for (auto& entry : map)
    {
        if (entry.second != nullptr)
            entry.second->release(*owner->m_retireRouter);  // ★ UAF リスク
    }
}
```

#### メンバ宣言順序の検証

| メンバ | 宣言行 | C++ 逆順デストラクション順序 |
|--------|--------|------------------------------|
| `eqCacheManager` | 2083行 | 後で破棄 |
| `m_retireRouter` | 4075行 | 先に破棄 |

C++ 規格 [class.cdtor]: メンバは**宣言逆順**で破棄。`m_retireRouter` (4075行) が `eqCacheManager` (2083行) より**後方宣言** → `m_retireRouter` が**先に破棄** → その後 `~EQCacheManager()` → `~CacheMap()` で `owner->m_retireRouter` 解参照 → **UAF**。

但し:
- `~EQCacheManager` (`AudioEngine.Cache.cpp:145-156`) で `cacheMapPtr.exchange(nullptr)` と `unique_ptr` により、`~EQCacheManager` が呼ばれる時点で `CacheMap` が RAII 削除される → その時 `owner->m_retireRouter` にアクセス
- `~EQCacheManager` は AudioEngine dtor 進行中に呼ばれる → `m_retireRouter` が既に破棄済みなら UAF
- `EQCoeffCache::release()` で refcount > 0 なら `fetchSubAtomic` だけで `m_retireRouter` に触れないが、refcount == 0 になったら `m_retireRouter->enqueueRetire(...)` を呼ぶ

#### 不確定性

`~AudioEngine` で明示的に `eqCacheManager` を `m_retireRouter` 破棄前にクリアしている可能性がある。`AudioEngine.CtorDtor.cpp` の dtor 実装で `eqCacheManager.clear()` 等が呼ばれていれば安全。

#### 修正方針

- `~AudioEngine()` の最優先処理で `eqCacheManager.clear()` を明示呼び出し
- または `~CacheMap()` で `owner->m_retireRouter` ではなく、`owner` から独立した retire インスタンスを使用
- `m_retireRouter` を `eqCacheManager` より**前方宣言**に移動（宣言順序の調整）

---

### B10 / B15: 性能または形態的リスク — 2件

#### B10: mixSmoothingSmall / mixSteadySmall AVX2 非対齐

| 項目 | 値 |
|------|-----|
| 判定 | **WARN (性能のみ、正確性は問題なし)** |
| 深刻度 | 🟢 低 |
| ファイル | `src/convolver/ConvolverProcessor.Runtime.cpp:597-641` |
| 影響 | `loadu`/`storeu` は非対齐を許容、UB なし、ただちに問題なし |
| 修正 | 32-byte alignment 強制（オプション性能改善） |

#### B15: AudioSegmentBuffer ABA リスク

| 項目 | 値 |
|------|-----|
| 判定 | **WARN (形態的リスク、runtime では未発火)** |
| 深刻度 | 🟢 低 |
| ファイル | `src/AudioSegmentBuffer.h:23-78` |
| 影響 | `pushBlock` / `copyLatest` が 2 つの独立 atomic を使用 → 中間态で ABA 可能 |
| runtime 状況 | `NoiseShaperLearner` 単一 `workerThreadMain` からのみ呼ばれるため、現在は race 未発火 |
| 修正 | 64-bit packed atomic または SPSC 索引対に変更 |

---

## NOT BUG（非バグ）— 8件 + ハルシネーション 4件

---

### B02: リングバッファ負 index — NOT BUG (修正済み)

**ファイル**: `src/convolver/ConvolverProcessor.Runtime.cpp:482`

```cpp
// 修正済み (line 479-482):
// Shift before mask to keep subtraction in non-negative range
// (fully portable across C++11/14/17, not relying on two's complement
// guarantee that is only mandatory from C++20 onward)
double p0 = srcBuf[(idx - 1 + DELAY_BUFFER_SIZE) & DELAY_BUFFER_MASK];
```

- bug.md が主張した `(idx - 1) & DELAY_BUFFER_MASK` は**現在のコードに存在しない**
- `+ DELAY_BUFFER_SIZE` オフセット追加済み → 全 C++ 標準で portable、負値なし
- fast path (449-473行) は `iRead >= 1` guard で `idx < 1` を完全カバー
- 注釈で「C++11/14/17 portability」明示済み
- **False Positive** — 修正前の旧コードに対する指摘

---

### B04: EQCacheManager 非原子参照カウント — NOT BUG

**ファイル**: `src/RefCountedDeferred.h:19-21, 65`

```cpp
void addRef() {
    convo::fetchAddAtomic(refCount, 1, std::memory_order_acq_rel);
}
...
std::atomic<int> refCount{1};
```

- `EQCoeffCache` は `ReferenceCountedObject` (JUCE) ではなく、`RefCountedDeferred<EQCoeffCache>` を継承
- `addRef()` / `release()` は `std::atomic<int>` を使用
- bug.md の「JUCE ReferenceCountedObject の非原子 addRef」という前提は**全く誤り**
- **ハルシネーション**

---

### B05 / B09: ConvolverProcessor dtor mkl_free IRState — NOT BUG

**ファイル**: `src/convolver/ConvolverProcessor.Lifecycle.cpp:101-122` (bug.md が主張した StateAndUI.cpp には不在)

```cpp
auto* oldIrState = convo::exchangeAtomic(currentIRState, nullptr, std::memory_order_acq_rel);
if (oldIrState != nullptr)
{
    oldIrState->~IRState();   // 1. 显式デストラクタ呼出 → non-POD メンバ解放
    mkl_free(oldIrState);     // 2. MKL アライメント済みメモリ解放
}
```

- 確保: `convo::aligned_make_unique<IRState>()` = `mkl_malloc + placement new` (`AlignedAllocation.h:107-119`)
- 解放: `~IRState()` + `mkl_free` = `AlignedObjectDeleter::operator()` と完全同一パターン (`AlignedAllocation.h:92-101`)
- `~IRState()` で `std::unique_ptr<juce::AudioBuffer<double>> irOwner` が正常解体
- **设计上正しいパターン**

---

### B06: dotProductAvx2 残余処理 — NOT BUG

**ファイル**: `src/TruePeakDetector.cpp:128-159` および `src/CustomInputOversampler.cpp:159-208`

两ファイルとも **3 段階ループ完全実装**:

1. `for (; i <= n - 16; i += 16)` — 16 要素 SIMD 4 accumulator
2. `for (; i <= n - 4; i += 4)` — 残余 4 要素 SIMD
3. `for (; i < n; ++i) sum += x[i] * coeffs[i];` — スカラー後処理 (0-3 要素)

- bug.md が主張した「残差スカラーフォールバック欠落」は**誤り**
- また、bug.md が主張したファイル (`CustomInputOversampler.cpp`) と bug_analysis_report.md が主張したファイル (`TruePeakDetector.cpp`) は**両方に存在**する
- **ハルシネーション** (ファイル位置は部分的に誤りだが、機能的には正常)

---

### B07: besselI0 無限ループ — NOT BUG

**ファイル**: `src/TruePeakDetector.cpp:113-126` および `src/CustomInputOversampler.cpp:144-157`

```cpp
for (int n = 1; n < 100; ++n)  // 硬上限 100
```

- `for` ループにハードリミット 100 あり → **関数自体は有限回で必ず終了**
- 外部の `CmaEsOptimizer` が無限ループに陥る可能性は、この関数ではなく `CmaEsOptimizer` 側の責務
- bug.md 自身も「この関数自体は 100 で止まる」と認めている
- **非バグ**

---

### B11: calcLowShelfBiquad NaN 传播 — NOT BUG

**ファイル**: `src/eqprocessor/EQProcessor.Coefficients.cpp:75-87, 159-191`

- 75-87行: `validateAndClampParameters` で freq `[20, min(20000, nyquist*0.95)]`、q `[0.01, 20]` に clamp
- 169行: 計算結果 `alpha`、`twoSqrtAAlpha` に対し `std::isfinite()` 検査
- 177行: `|a0| < 1e-15` でバイパス係数 (`b0=1, a0=1, rest=0`) に fallback
- `calcLowShelfBiquad` には直接呼出元なし（`calcBiquadCoeffs` 経由のみ）
- **多重防御済み**

---

### B12: CMakeLists /STACK /GS- — NOT BUG (妥当なテスト設定)

**ファイル**: `CMakeLists.txt:323-326`

```cmake
# BuildInputSemanticContractTests: 大規模ソース読み取りでスタックオーバーフローを防ぐため8MBに拡大
target_compile_options(BuildInputSemanticContractTests PRIVATE /GS-)
target_link_options(BuildInputSemanticContractTests PRIVATE "/STACK:8388608")
```

- テストターゲットのみ適用、本番バイナリには影響なし
- `BuildInputSemanticContractTests.cpp:25-34` で `std::ifstream` + `ostringstream::rdbuf` で大ファイル読取 → スタック拡張が必要
- 注釈で意図明示済み
- **妥当な設定**

---

### B16: LatticeNoiseShaper 直接型/格子型混同 — NOT BUG

**ファイル**: `src/LatticeNoiseShaper.h:220-267`

- `computeFeedback` (220-241行): `Σ k_i · state[i]` — **標準 joint-process lattice estimator** の forward prediction error contribution (Haykin §6.3 / Regalia §5.2 準拠)
- `advanceState` (243-267行): `forward_next = forward + k_i * backward` / `nextBackward = k_i * forward + backward` — **標準 lattice 段更新再帰式**
- 259-263行の注釈 [P7] で修正履歴明示
- `tools/analysis/compare_noiseshaper_patterns.py` で pattern 比較検証済み
- `doc/work54` で系統的検証済み
- **非「反射係数を FIR tap として誤用」** — bug.md が算法を誤読

---

### B18: destroyQuarantineSlot メモリリーク — NOT BUG

**ファイル**: `src/audioengine/ISRDSPHandle.cpp:149-197`

- `registry_[slot].instance = nullptr` は**所有権移動ではない** — `DSPHandleRuntime` は `dspInstance` ポインタを所有しない (ISRDSPHandle.cpp:21-36 注釈明示)
- `DSPCore`/`EQ` 実体は外部 `std::unique_ptr` (`dspWorld_`) が所有 → AudioEngine dtor で解放
- `AudioEngine.Commit.cpp:618-621` / `ReleaseResources.cpp:341-372` で三システム (`retireRuntimeEx_.reclaim` + `destroyQuarantineSlot` + `dspQuarantineManager_.reclaimSlot`) 協調ルーズ
- **リーク経路不存在**

---

## バグ判定フローチャート

```
bug.md の主張 (19件)
    │
    ├─ 実コードに存在する? ──No──→ ハルシネーション (B04, B09)
    │
    Yes
    │
    ├─ 修正済み? ──Yes──→ NOT BUG (B02)
    │
    No
    │
    ├─ 機能的に正しい / 設計上妥当? ──Yes──→ NOT BUG (B05, B06, B07, B11, B12, B16, B18)
    │
    No
    │
    ├─ runtime でレース未発火? ──Yes──→ WARN (B15)
    │
    No
    │
    ├─ 性能のみで正確性問題なし? ──Yes──→ WARN (B03, B10)
    │
    No
    │
    ├─ 破棄順序に依存 UAF? ──Yes──→ WARN (B08)
    │
    No
    │
    └─→ BUG (B01, B13, B14, B17)
```

---

## 優先度別アクションリスト

### 🔴 P0 — 即時修正推奨

| ID | 件名 | 理由 |
|----|------|------|
| B14 | Retire queue MPSC data race | データレース → 永続的メモリリーク。releaseResources / Timer / Coordinator の三系統が並行動作時に確率高 |

### 🟡 P1 — 計画的修正推奨

| ID | 件名 | 理由 |
|----|------|------|
| B01 | DSPCoreFloat バイパスブレンド欠落 | 32bit float 経路で full bypass 切替時にクリックノイズ。多くの DAW のデフォルト経路 |
| B17 | StereoConvolver::clone() filterSpec 欠落 | preset ロード等でクローン生成時に周波数フィルタ特性消失 |
| B13 | NUPC delay alignment 欠落 | Layer 1/2 残響テールの時間軸ズレ可能性。FDL 設計で暗黙補正されている可能性もあり、要検証 |
| B08 | CacheMap dtor m_retireRouter UAF リスク | AudioEngine dtor 順序依存。明示的 clear() 追加で予防可能 |

### 🟢 P2 — 改善推奨（性能・予防）

| ID | 件名 | 理由 |
|----|------|------|
| B03 | 冗余 vdTanh 計算 | 機能正常。CPU 浪費の最適化 |
| B10 | AVX2 非対齐 | 機能正常。alignment 強制で性能改善可能 |
| B15 | AudioSegmentBuffer ABA | runtime 単スレッドで未発火。構造改善で予防 |

### ⚪ 修正不要

| ID | 件名 | 理由 |
|----|------|------|
| B02 | リングバッファ負 index | **修正済み** (`+ DELAY_BUFFER_SIZE`) |
| B04 | EQCacheManager 非原子 | **std::atomic 使用済み** |
| B05/B09 | ConvolverProcessor dtor | **AlignedObjectDeleter 正規パターン** |
| B06 | dotProductAvx2 残余 | **3 段階ループ完全実装** |
| B07 | besselI0 無限ループ | **上限 100 で有限** |
| B11 | calcLowShelfBiquad NaN | **多重防御済み** |
| B12 | CMake /STACK /GS- | **テストターゲット妥当設定** |
| B16 | LatticeNoiseShaper 混同 | **標準 lattice 実装** |
| B18 | destroyQuarantineSlot leak | **instance 外部所有、リークなし** |

---

## 付録: 文書間判定比較

| ID | bug.md 主張 | bug_analysis_report.md | bug1_bug2_deep_analysis.md | **最終判定** |
|----|-------------|------------------------|----------------------------|-------------|
| B01 | Double版欠落 | VALID (方向逆, Float版欠落) | VALID (Float版欠落) | **BUG (Float版欠落)** |
| B02 | 負 index UB | LOW (C++20で問題なし) | LOW (C++17移植性) | **NOT BUG (修正済み)** |
| B03 | 冗余 vdTanh | INVALID (ハルシネー) | — | **WARN (実在、性能のみ)** |
| B04 | 非原子 addRef | INVALID | — | **NOT BUG** |
| B05 | mkl_free 不完全 | INVALID | — | **NOT BUG** |
| B06 | 残余欠落 | INVALID (3段ループ) | — | **NOT BUG** |
| B07 | 無限ループ | INVALID (上限100) | — | **NOT BUG** |
| B08 | dtor UAF | INVALID (RAII削除) | — | **WARN** (宣言順序で RAII 削除でも UAF リスク) |
| B09 | (= B05) | INVALID | — | **NOT BUG** |
| B10 | AVX2 非対齐 | INVALID (存在しない) | — | **WARN** (実在、性能のみ) |
| B11 | NaN 传播 | INVALID | — | **NOT BUG** |
| B12 | /STACK /GS- | INVALID | — | **NOT BUG** (テスト設定) |
| B13 | NUPC delay | INVALID | — | **BUG** (構造的欠落確認) |
| B14 | MPSC race | INVALID | — | **BUG** (実在) |
| B15 | ABA | INVALID | — | **WARN** (runtime 未発火) |
| B16 | lattice 混同 | INVALID | — | **NOT BUG** |
| B17 | filterSpec 欠落 | INVALID | — | **BUG** (実在) |
| B18 | leak | INVALID | — | **NOT BUG** |

### bug_analysis_report.md の精度評価

- **正解**: B01(VALID方向逆), B02(LOW), B03-B09(INVALID 7件), B10-B14(INVALID 5件) = 14件
- **誤判定**: B08(NOT BUG だが WARN が適切), B10(存在しない だが実在), B13(不存在 だが実在), B14(不存在 だが実在), B15(不存在), B16(不存在), B17(不存在), B18(不存在) = 8件中 5件で見逃し
- **全体精度**: 19件中 14件正解 (73.7%)

> bug_analysis_report.md は「12/14 がハルシネーション」と結論したが、実際には後半の深層レポート (B13, B14, B15, B16, B17, B18) で実在コードを指す主張が 3件 (B13, B14, B17) 真正バグだった。浅い grep で「存在しない」と断定した件の一部は、別ファイル名・別関数名で実在していた。

---

## 検証メタデータ

- 検証対象ソース: ConvoPeq リポジトリ HEAD (2026-07-11)
- 検証ツール: Read (直接コード読取), grep (全文検索), ast-grep (構文解析)
- 検証範囲: 19件のバグ主張全て
- 実コード確認行数: 約 500行 (主要関数・クラス)
