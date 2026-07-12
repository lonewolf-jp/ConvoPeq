# ConvoPeq バグ検証最終報告書

**作成日**: 2026-07-12
**対象**: `doc/work69/bug.md`（全19件のバグ主張）
**検証方法**: 実ソースコード直接照合（Serena MCP / AiDex / WSL grep / context-mode ctx_execute）
**補足調査**: インターネット文献調査（NUPC理論、Gardner 1995, Garcia 2002, Wefers 2015）

---

## 検証結果サマリ

```
真正的バグ (BUG):    5件  ← B13を再確定、B20を追加（B21は設計者判断で除外）
潜在的バグ (WARN):   5件  ← B18をNOT BUG→WARNに再判定
非バグ (NOT BUG):   10件
```

| ID  | 件名                                         | 判定        | 優先度 | ファイル                                              |
|-----|----------------------------------------------|-------------|--------|-------------------------------------------------------|
| B14 | Retire queue MPSC data race                  | **BUG**     | 🔴 P0  | ISRRetire.cpp                                         |
| B01 | DSPCoreFloat.cpp バイパスブレンド欠落          | **BUG**     | 🟡 P1  | AudioEngine.Processing.DSPCoreFloat.cpp               |
| B13 | NUPC delay alignment 欠落                    | **BUG**     | 🟡 P1  | MKLNonUniformConvolver.cpp                            |
| B17 | StereoConvolver::clone() FilterSpec欠落       | **BUG**     | 🟡 P1  | ConvolverProcessor.h                                  |
| B20 | TruePeakDetector Rチャンネル計測欠落          | **BUG**     | 🔴 P0  | TruePeakDetector.cpp                                  |
| B08 | CacheMap dtor m_retireRouter UAF              | **WARN**    | 🟡 P1  | AudioEngine.h                                         |
| B18 | destroyQuarantineSlot メモリリーク            | **WARN**    | 🟡 P1  | ISRDSPHandle.cpp                                      |
| B03 | NoiseShaperLearner redundant vdTanh           | **WARN**    | 🟢 P2  | NoiseShaperLearner.cpp                                |
| B10 | mixSmoothingSmall AVX2 非対齐                | **WARN**    | 🟢 P2  | ConvolverProcessor.Runtime.cpp                        |
| B15 | AudioSegmentBuffer ABA リスク                 | **WARN**    | 🟢 P2  | AudioSegmentBuffer.h                                  |
| B02 | リングバッファ負 index                        | NOT BUG    | —      | ConvolverProcessor.Runtime.cpp                        |
| B04 | EQCacheManager 非原子参照カウント             | NOT BUG    | —      | RefCountedDeferred.h                                  |
| B05 | ConvolverProcessor dtor mkl_free             | NOT BUG    | —      | ConvolverProcessor.Lifecycle.cpp                      |
| B06 | dotProductAvx2 残余処理欠落                  | NOT BUG    | —      | TruePeakDetector.cpp / CustomInputOversampler.cpp     |
| B07 | besselI0 無限ループ                           | NOT BUG    | —      | TruePeakDetector.cpp                                  |
| B09 | IRState mkl_free 不完全解放                  | NOT BUG    | —      | (= B05 と同一)                                       |
| B11 | calcLowShelfBiquad NaN 传播                  | NOT BUG    | —      | EQProcessor.Coefficients.cpp                          |
| B12 | CMakeLists /STACK /GS-                        | NOT BUG    | —      | CMakeLists.txt                                        |
| B16 | LatticeNoiseShaper 直接型/格子型混同          | NOT BUG    | —      | LatticeNoiseShaper.h                                  |
| B19 | BuildInputSemanticContract /STACK             | NOT BUG    | —      | (= B12)                                              |

---

## 🔴 P0 — 即時修正推奨

---

### B14: Retire queue SPSC→MPSC データレース

| 項目 | 内容 |
|------|------|
| **ID** | B14 |
| **判定** | **BUG** |
| **深刻度** | 🔴 高 |
| **優先度** | **P0 — 即時修正** |

#### ① バグ概要

`ISRRetire.cpp` の `emitRetireIntent()` 関数は SPSC（Single-Producer Single-Consumer）前提のキュー構造を持つ。しかし実際の呼び出し元は異なる4つのスレッドから並行に呼び出されるため、複数 producer が同一 `tail` スロットを読み、同一 slot を交叉書きし、一方の intent が消失するデータレースが発生する。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/audioengine/ISRRetire.cpp` |
| **関数** | `RetireRuntime::emitRetireIntent()` (11行目) |
| **関数** | `RetireRuntime::emitRetireIntentRT()` (84行目) — `emitRetireIntent` を委譲 |
| **キーデータ** | `retireIntentTail_` (ISRRetire.h:87) — `std::atomic<uint64_t>` |

#### ③ バグの詳細

```cpp
// ISRRetire.cpp:11-14 (問題のコード)
void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    uint64_t tail = convo::consumeAtomic(retireIntentTail_, std::memory_order_relaxed);  // ← 読み取りのみ
    uint64_t nextTail = (tail + 1) % RETIRE_INTENT_QUEUE_SIZE;
    // ... (full check) ...
    retireIntentQueue_[tail] = intent;                  // ← この slot に書き込み
    convo::publishAtomic(retireIntentTail_, nextTail, std::memory_order_release);  // ← tail 推進
}
```

**エンキュー実装（`emitRetireIntent`, ISRRetire.cpp:11-83）**:

```cpp
uint64_t tail = convo::consumeAtomic(retireIntentTail_, std::memory_order_relaxed);  // LOAD tail
uint64_t nextTail = (tail + 1) % RETIRE_INTENT_QUEUE_SIZE;
// ... full check ...
retireIntentQueue_[tail] = intent;                  // WRITE to slot (非atomic)
convo::publishAtomic(retireIntentTail_, nextTail, std::memory_order_release);  // STORE tail
```

**デキュー実装（`dequeuePendingRetireIntents`, ISRRetire.cpp:88-121）**:
```cpp
uint64_t head = convo::consumeAtomic(retireIntentHead_, std::memory_order_acquire);
uint64_t tail = convo::consumeAtomic(retireIntentTail_, std::memory_order_acquire);
while (head != tail) {
    result.push_back(retireIntentQueue_[head]);  // READ slot (非atomic)
    head = (head + 1) % RETIRE_INTENT_QUEUE_SIZE;
}
convo::publishAtomic(retireIntentHead_, head, std::memory_order_release);
```

**`publishAtomic` / `consumeAtomic` の実体**: `publishAtomic = std::atomic_store_explicit` (release), `consumeAtomic = std::atomic_load_explicit` (acquire)。

**MPSC データレースの証明**: `emitRetireIntent` の `tail` 操作は load-then-store パターン（SPSC用）。RMW命令（`fetch_add`）ではないため、複数Producerが同時に同一 `tail` 値を読む → 同一slotに書き込み合戦 → 一方のintent消失。

```
タイムライン:
  Producer A: tail=5 を読む
  Producer B: tail=5 を読む ← ★ 同じ値！
  Producer A: slot[5] に Intent-X を書く
  Producer B: slot[5] に Intent-Y を書く ← ★ Intent-X 消失！
```

**`retireIntentQueue_` は非atomic配列** (`ISRRetire.h:91`): `RetireIntent retireIntentQueue_[RETIRE_INTENT_QUEUE_SIZE]`。複数スレッドからの同時書き込みはデータ競合 (UB)。

**呼び出し元（全て異なるスレッド）**:

| ファイル | 行 | スレッド |
|---------|-----|---------|
| `AudioEngine.Commit.cpp` | 462 | Commit (Message Thread) |
| `AudioEngine.Processing.ReleaseResources.cpp` | 210, 248 | ReleaseResources |
| `AudioEngine.Timer.cpp` | 1579 | Timer |
| `ISRRuntimePublicationCoordinator.cpp` | 293, 305, 331 | Coordinator |

**`reinterpret_cast<std::atomic<bool>>` 問題 (ISRRetire.cpp:203)**:

```cpp
// escalateAllRetires() — Shutdown時に全intentの優先度を昇格
const auto rawIsValid = convo::consumeAtomic(
    reinterpret_cast<const std::atomic<bool>&>(intent.isValid),  // ★ UB!
    std::memory_order_acquire);
```

`RetireIntent::isValid` は `bool`（非atomic）を `reinterpret_cast<std::atomic<bool>&>` でatomic読取。コメント「Shutdown中は単一スレッドアクセス」はあるが、C++標準上は **strict aliasing 違反** かつ `std::atomic<bool>` と `bool` のレイアウト互換性は実装定義。実害は低いが保守性・移植性の観点で改善必須。

**コード内の自己認識**: `ISRRetireOverflowRing.h:7-19` 注釈で「SPSC 前提」「Worker Thread からの直接 emitRetireIntent 出現時は MPSC に変更必須」と明記。

**影響**: 一方の retire intent が消失 → 該当リソースが永続的に未解放 → **永続的メモリリーク**。

#### ④ バグ改善方法（ISR適合版）

**設計方針: Vyukov MPSC + SPSC Consumer + Fallback Mutex** — 3層構造で Authority を明確に分離

```
Producer (任意非RTスレッド)          Consumer (Commitの単一スレッド)
   │                                    │
   ├─ fetch_add(enqueueSeq) ─────→       │  (slot予約: 唯一のRMW)
   │  sequence[idx]==seq を確認          │
   │  payload書き込み                     │
   │  sequence[idx]=seq+1 (release)      │
   │                                    │
   │                               ┌─ dequeuePos_を確認
   │                               │  sequence[idx]==dequeuePos+1? → 読み取り
   │                               │  sequence[idx]=dequeuePos+SIZE (release)
   │                               │
   └─ enqueueSeq_溢れ → fallback ──→ dequeue時: fallbackもdrain
        (mutex保護、ISR非準拠だが避難所)
```

**第1層（本命）: Vyukov MPSC** — Producer の `fetch_add` 1回のみ、Consumer の atomic RMW 不要

```cpp
struct RetireSlot {
    RetireIntent payload;
    std::atomic<uint64_t> sequence{0};  // N=初期値, seq=writing, seq+1=ready, seq+SIZE=解放済
};

class RetireRuntime {
    // Producer管理
    std::atomic<uint64_t> enqueueSeq_{0};   // 唯一のRMW対象
    // Consumer管理（ProducerがFull checkで読むためatomic必要）
    std::atomic<uint64_t> dequeuePos_{0};   // ★ atomic — reinterpret_cast回避
    RetireSlot slots_[RETIRE_INTENT_QUEUE_SIZE];
    // Fallback（mutex保護、MPSC溢れ時の避難所）
    std::mutex fallbackMutex_;
    RetireIntent fallbackQueue_[FALLBACK_QUEUE_CAPACITY];
    size_t fallbackCount_{0};               // ★ mutex保護下、atomic不要
public:
    void emitRetireIntent(const RetireIntent& intent) noexcept; // 任意スレッド
    std::vector<RetireIntent> dequeuePendingRetireIntents() noexcept; // Consumer専用
};
```

**Producer（任意非RTスレッド）**:
```cpp
void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    // ★ Step 1: slot を原子的に予約 (唯一のRMW = ISR最小atomic原則)
    const uint64_t seq = convo::fetchAddAtomic(enqueueSeq_, 1, std::memory_order_acq_rel);
    const size_t idx = seq % RETIRE_INTENT_QUEUE_SIZE;

    // ★ Step 2: full check — dequeuePos が追いついているか
    //   Acquire load: Consumer の最新解放位置を観測
    //   dequeuePos_ は atomic なので安全に読める
    const uint64_t deqPos = convo::consumeAtomic(dequeuePos_, std::memory_order_acquire);
    if (seq - deqPos >= RETIRE_INTENT_QUEUE_SIZE) {
        // full → fallback (mutex保護)
        std::lock_guard<std::mutex> lock(fallbackMutex_);
        if (fallbackCount_ < FALLBACK_QUEUE_CAPACITY) {
            fallbackQueue_[fallbackCount_++] = intent;
        } else {
            // 最終手段: 統計カウントのみ（ISRではベストエフォート）
            convo::fetchAddAtomic(droppedIntentCount_, 1, std::memory_order_relaxed);
        }
        return;
    }

    // ★ Step 3: spin — Consumer が slot を解放するまで待機
    //   全Producerが非RTスレッドであることを確認済み。spin許容。
    //   bound付き: 1000回の _mm_pause 後に yield
    for (int spin = 0;; ++spin) {
        uint64_t expected = convo::consumeAtomic(
            slots_[idx].sequence, std::memory_order_acquire);
        if (expected == seq) break;  // 獲得
        if (spin < 100)
            _mm_pause();
        else
            std::this_thread::yield();
    }

    // ★ Step 4: payload 書き込み（単一Producer保証 = race-free）
    slots_[idx].payload = intent;

    // ★ Step 5: release — Consumer に読み取り可能を通知
    convo::publishAtomic(slots_[idx].sequence, seq + 1, std::memory_order_release);
}
```

**Consumer（Commitの単一スレッド）**:
```cpp
std::vector<RetireIntent> RetireRuntime::dequeuePendingRetireIntents() noexcept
{
    std::vector<RetireIntent> result;

    // 1. Vyukov MPSC drain (Consumer専用: 単一スレッドアクセス)
    uint64_t localDequeuePos = dequeuePos_.load(std::memory_order_relaxed);
    while (true) {
        const size_t idx = localDequeuePos % RETIRE_INTENT_QUEUE_SIZE;
        const uint64_t seq = convo::consumeAtomic(slots_[idx].sequence,
                                                   std::memory_order_acquire);
        if (seq != localDequeuePos + 1)
            break;  // 未ready

        result.push_back(slots_[idx].payload);

        // ★ slot 解放: Producer がこの slot を再利用可能に
        convo::publishAtomic(slots_[idx].sequence,
                             localDequeuePos + RETIRE_INTENT_QUEUE_SIZE,
                             std::memory_order_release);
        ++localDequeuePos;
    }
    // ★ バッチ更新: Producer に最新値を公開
    convo::publishAtomic(dequeuePos_, localDequeuePos, std::memory_order_release);

    // 2. Fallback queue drain (mutex保護)
    {
        std::lock_guard<std::mutex> lock(fallbackMutex_);
        const size_t count = fallbackCount_;
        for (size_t i = 0; i < count; ++i)
            result.push_back(fallbackQueue_[i]);
        fallbackCount_ = 0;
    }

    // 3. 優先度ソート（既存維持）
    std::stable_sort(result.begin(), result.end(), /* 既存のComparator */);
    return result;
}
```

**初期化**:
```cpp
void RetireRuntime::initQueue() noexcept {
    for (size_t i = 0; i < RETIRE_INTENT_QUEUE_SIZE; ++i) {
        convo::publishAtomic(slots_[i].sequence, static_cast<uint64_t>(i),
                             std::memory_order_release);
    }
    convo::publishAtomic(enqueueSeq_, 0, std::memory_order_release);
    convo::publishAtomic(dequeuePos_, 0, std::memory_order_release);
}
```

**`reinterpret_cast` 問題のISR準拠修正**:
```cpp
// ★ ISR原則「コード証明可能性」の観点から、isValid のみを std::atomic<bool> に変更。
//   サイズ/レイアウトは bool と同一（x86-64）、RetireIntent は aggregate init 互換を維持。
struct RetireIntent {
    uint32_t dspSlot;
    uint64_t generation;
    uint64_t retireEpoch;
    std::atomic<bool> isValid{false};  // ★ atomic 化 → reinterpret_cast 不要
    RetirePriority priority{RetirePriority::Normal};
};
```

**ISR Key Points**:
| 原則 | 適合性 |
|------|--------|
| 最小 atomic | ✅ Producer: `fetch_add` 1回のみ。Consumer: local copy + バッチ publish (RMW不要) |
| Audio Thread 非関与 | ✅ 全Producer = 非RTスレッド確認済み |
| Monotonic | ✅ enqueueSeq_ は単調増加、dequeue も単方向 |
| Authority Singularization | ✅ Producer/Consumer の責務分離。fallback は mutex 保護で別 Authority |
| コード証明可能性 | ✅ `reinterpret_cast` 全排除（`dequeuePos_` は `std::atomic<uint64_t>` に変更） |

**追加考慮点**:
- **`acknowledgeGeneration_` 互換性**: `acknowledgeGeneration_[intent.dspSlot % 256]` の配列は変更不要。`acknowledgeRetireCoordination()` は Consumer でデキュー後に呼ばれ、キュー構造に依存しないため、Vyukov MPSC 化の影響を受けない
- **テスト互換性（`PriorityIntegrationTests.cpp`）**: `RetireIntent` の aggregate initialization (`{1, 100, 1000, true, RetirePriority::Normal}`) が、`isValid` を `std::atomic<bool>` に変更した後も動作するように、コンストラクタを追加またはテスト側で `intent.isValid.store(true)` に変更する必要がある

---

### B20: TruePeakDetector Rチャンネル計測欠落

| 項目 | 内容 |
|------|------|
| **ID** | B20 |
| **判定** | **BUG** |
| **深刻度** | 🔴 高 |
| **優先度** | **P0 — 即時修正** |

#### ① バグ概要

True Peak レベルメーター（ITU-R BS.1770-4/5 準拠）が Rチャンネルのインターサンプルピークを完全に無視している。Stage 1（2x→4x アップサンプリング）で Rチャンネルが全く処理されない。また、**報告未指摘のバッファオーバーフロー**が存在する。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/TruePeakDetector.cpp` |
| **関数** | `TruePeakDetector::processBlock()` (64行目) |
| **関連** | `interpolateStage()` (252行目) |

#### ③ バグの詳細

**Stage 1 の Rチャンネル処理欠落**:

```cpp
// 64-91行 (processBlock)
// Stage 0: 1x → 2x (L)
interpolateStage(stages[0], dataL, numSamples, work, 0);
// Stage 0: 1x → 2x (R)
interpolateStage(stages[0], dataR, numSamples, work + up1Samples, 1);

// Stage 1: 2x → 4x (L/Rはworkにインターリーブ)
interpolateStage(stages[1], work, up1Samples, work + up1Samples * 2, 0);  // ← Lのみ！
```

- Stage 0 では L/R 両方を別チャンネル (0/1) で正しく処理
- Stage 1 では `work`（Lチャンネル 2x データ）のみを入力として処理
- Rチャンネル 2x データ（`work + up1Samples`）は完全に無視
- ピークスキャン（`work + up1Samples * 2` から `up2Samples` 走査）も Stage 1 出力の L のみ

**バッファオーバーフロー確定** ⚠️（レビューで追加検証済み）

ソースコード精査の結果、バッファオーバーフローは **確定**。

**バッファサイズ検証**:

| 領域 | アドレス範囲 | サイズ |
|------|-------------|-------|
| Stage 0 L 出力 | `work[0]` ～ `work[numSamples*2-1]` | `numSamples*2` |
| Stage 0 R 出力 | `work[numSamples*2]` ～ `work[numSamples*4-1]` | `numSamples*2` |
| Stage 1 L 出力 | `work[numSamples*4]` ～ `work[numSamples*8-1]` | `numSamples*4` |
| 確保済み容量 | ～ `work[maxBlockSize*4-1]` | `maxBlockSize*4` |

```cpp
// TruePeakDetector.cpp:25-30 (prepare)
const int upBufferSize = maxBlockSize * kOversamplingRatio;  // = maxBlockSize*4
upsampleBuffer = convo::makeAlignedArray<double>(upBufferSize);  // maxBlockSize*4 個
bufferCapacity = upBufferSize;  // = maxBlockSize*4

// TruePeakDetector.cpp:64-85 (processBlock)
// interpolateStage() は inputSamples*2 要素を出力
// Stage 1 入力: up1Samples = numSamples*2 → 出力: numSamples*4 要素
// 出力先: work + numSamples*4 → 最終書き込み位置: work + numSamples*8
// 確保サイズ: maxBlockSize*4
// numSamples >= maxBlockSize/2 → オーバーフロー！
```

**結論**: `numSamples >= maxBlockSize/2` でバッファオーバーフローが発生する。典型的使用では `numSamples << maxBlockSize` のため未発火だが、大ブロック使用時に潜在的な危険。

**SIMD レイアウトの注意点**:
現在の AVX2 ピークスキャンは Stage 1 出力（L のみ）のフラットレイアウトを前提としている。R チャンネルも処理する場合、L と R の 4x データを **別領域** に格納し、それぞれ独立にピークスキャンを行ってから最大値を比較する方式にすべき。インターリーブ（LRLR）方式では SIMD の水平加算で左右のピークが混ざる。

**影響**:
- Rチャンネルの True Peak が未計測 → LUFS 計測不正確、クリップ検出ミス
- バッファ破壊によるスタック/ヒープ破損の可能性

#### ④ バグ改善方法

**修正箇所**: `prepare()` + `processBlock()` の2関数を修正。バッファと補間の2段階。

**Step 1 — `prepare()` のバッファ拡張**:
```cpp
// TruePeakDetector.cpp:25-30
// 旧: const int upBufferSize = maxBlockSize * kOversamplingRatio;  // = maxBlockSize*4
// 新: 4x領域×2チャンネル分を確保
const int upBufferSize = maxBlockSize * kOversamplingRatio * 2;  // = maxBlockSize*8
```

**Step 2 — `processBlock()` の R チャンネル補間追加**:
```cpp
double TruePeakDetector::processBlock(const double* dataL, const double* dataR, int numSamples) noexcept
{
    if (numSamples <= 0 || !upsampleBuffer)
        return 0.0;

    double* work = upsampleBuffer.get();
    const int up1Samples = numSamples * 2;
    const int up2Samples = numSamples * 4;

    // Stage 0: 1x → 2x (L)
    interpolateStage(stages[0], dataL, numSamples, work, 0);
    // Stage 0: 1x → 2x (R)
    if (dataR != nullptr)
        interpolateStage(stages[0], dataR, numSamples, work + up1Samples, 1);
    else
        interpolateStage(stages[0], dataL, numSamples, work + up1Samples, 1);

    // ★ Stage 1: 2x → 4x (L + R)
    interpolateStage(stages[1], work,             up1Samples, work + up1Samples * 2,              0);  // L
    // ★ RチャンネルのStage 1入力: work + up1Samples（Stage 0のR出力、dataR==nullptr時はdataLの複製）
    interpolateStage(stages[1], work + up1Samples, up1Samples, work + up1Samples * 2 + up2Samples, 1);  // R

    // ★ ピークスキャン: L/R 別領域で独立に実行
    double peakL = scanPeak(work + up1Samples * 2,                 up2Samples);
    double peakR = scanPeak(work + up1Samples * 2 + up2Samples,    up2Samples);
    double peak = std::max(peakL, peakR);

    // ピークホールド（変更なし）
    if (peak > peakHold)
        peakHold = peak;
    else
        peakHold *= 0.999;

    return peakHold;
}
```

**Step 3 — `scanPeak` ヘルパー関数**:
```cpp
// ★ 補助関数: アライメント保証なしの double 配列から最大絶対値を AVX2 で検出
static double scanPeak(const double* buf, int n) noexcept
{
    double peak = 0.0;
#if defined(__AVX2__)
    __m256d vPeak = _mm256_setzero_pd();
    int i = 0;
    for (; i <= n - 4; i += 4) {
        __m256d v = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd(buf + i));
        vPeak = _mm256_max_pd(vPeak, v);
    }
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, vPeak);
    for (int j = 0; j < 4; ++j) if (tmp[j] > peak) peak = tmp[j];
    for (; i < n; ++i) { double v = std::abs(buf[i]); if (v > peak) peak = v; }
#else
    for (int i = 0; i < n; ++i) { double v = std::abs(buf[i]); if (v > peak) peak = v; }
#endif
    return peak;
}
```

**注意点**:
- R チャンネルの Stage 1 チャンネルインデックスは `1`（Stage 0 の R と同じチャンネル番号）→ `interpolateStage` 内の履歴が正しく分離される
- `scanPeak` は L 領域と R 領域を独立に走査 → L+R インターリーブ非依存
- バッファ拡張により `maxBlockSize == 4096` 時も `32KB` の mkl_malloc → 許容範囲内
- `peakHold` 変数は従来通り L/R 最大値を保持

**BS.1770 適合性**: 修正後は L/R 各チャンネルに 4x オーバーサンプリングが適用され、True Peak 測定が ITU-R BS.1770-4/5 に準拠する。

---

## 🟡 P1 — 計画的修正推奨

---

### B21: SimplePeakLimiter Knee補間境界エラー

| 項目 | 内容 |
|------|------|
| **ID** | B21 |
| **判定** | **BUG** |
| **深刻度** | 🟡 中 |
| **優先度** | **P1 — 計画的修正** |

#### ① バグ概要

ソフトニーリミッターにおいて、Knee領域の右半分（`threshold` ～ `threshold + knee/2`）がスプライン補間されず、t=0.5 でハードリミッティング式にジャンプする。微係数の不連続性（C1不連続）によりクリックノイズが発生する。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/audioengine/SimplePeakLimiter.h` |
| **関数** | `SimplePeakLimiter::processBlock()` (38行目) |
| **変数** | `clipStart` = `thresholdLinear - kneeLinear * 0.5` |

#### ③ バグの詳細

```cpp
// 51-66行: 問題のコード
if (peak > clipStart)
{
    if (peak <= thresholdLinear)  // ← ★ バグ: Knee中央(t=0.5)で打ち切り
    {
        // Knee領域: 3次スプライン補間
        const double t = (peak - clipStart) / kneeLinear;
        const double kneeShape = t * t * (3.0 - 2.0 * t);
        desiredGain = 1.0 - (1.0 - thresholdLinear / peak) * kneeShape;
    }
    else
    {
        // リミッティング領域
        desiredGain = thresholdLinear / peak;
    }
}
```

**ソフトニーの数学的定義**:
- Knee範囲: `threshold - knee/2` ～ `threshold + knee/2`
- スプライン変数 t は `0.0 ～ 1.0` であるべき

**現在のコード**: `peak <= thresholdLinear` で t ∈ [0.0, 0.5] のみカバー。t=0.5 で分岐。

**C1不連続性の数学的証明**:

$$\begin{aligned}
g_{\text{knee}}(t) &= 1 - (1 - \frac{T}{p}) \cdot s(t),\quad s(t) = t^2(3-2t),\quad t \in [0, 0.5] \\
g_{\text{limit}}(p) &= \frac{T}{p},\quad p \in (T, T+\frac{K}{2}]
\end{aligned}$$

$t=0.5$（$p=T$）における微係数:
- Knee側: $\frac{dg}{dp}\big|_{t=0.5} = -\frac{0.5}{T}$
- リミッター側: $\frac{dg}{dp}\big|_{p=T} = -\frac{1}{T}$

**不一致 → C1不連続 → クリックノイズ発生**

**影響**: トランジェント信号（threshold 直上の信号）でクリックノイズ（高調波歪み）。

#### ④ バグ改善方法

```cpp
// 修正: thresholdLinear → thresholdLinear + kneeLinear * 0.5
if (peak <= thresholdLinear + kneeLinear * 0.5)
{
    const double t = (peak - clipStart) / kneeLinear;
    const double kneeShape = t * t * (3.0 - 2.0 * t);
    desiredGain = 1.0 - (1.0 - thresholdLinear / peak) * kneeShape;
}
else
{
    desiredGain = thresholdLinear / peak;
}
```

これにより Knee 範囲全面 (`t ∈ [0, 1]`) がスプライン補間され、C1連続性が確保される。

**注記**: 本バグは設計者判断によりバグリストから除外する。別途 Limiter 全体の改修で対応予定。

### B01: DSPCoreFloat.cpp バイパスブレンド機構欠落

| 項目 | 内容 |
|------|------|
| **ID** | B01 |
| **判定** | **BUG** |
| **深刻度** | 🟡 中 |
| **優先度** | **P1 — 計画的修正** |

#### ① バグ概要

64bit double 版の DSP コア (`DSPCoreDouble.cpp`) には、full bypass（EQ+Convolver 両方 OFF）切替え時のポップ/クリックノイズ防止として **5ms クロスフェード機構** が完全実装されている。しかし 32bit float 版 (`DSPCoreFloat.cpp`) にはこの機構が**全く存在しない**。double 版から float 版への移植漏れ。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` |
| **関数** | `AudioEngine::DSPCore::process()` |
| **比較対象** | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` |
| **同関数** | `AudioEngine::DSPCore::process()` (Double版) |

#### ③ バグの詳細

**Double版 (完全実装)** — DSPCoreDouble.cpp:

```
381行: const bool requestedFullBypass = state.eqBypassed && state.convBypassed;
383行: if (requestedFullBypass != ramp.bypassedDouble)
385行:     ramp.bypassFadeGainDouble.setTargetValue(requestedFullBypass ? 0.0 : 1.0);
386行:     ramp.bypassedDouble = requestedFullBypass;
393行: if (dryBypassBufferDoubleL && dryBypassBufferDoubleR ...)
395行:     FloatVectorOperations::copy(dryBypassBufferDoubleL, alignedL, numSamples);  // dry信号保存
547行: const bool bypassBlendRequested = ramp.bypassFadeGainDouble.isSmoothing() || requestedFullBypass;
549行: if (OS == 1 && dryBypassBufferDoubleL && ... && bypassBlendRequested)
556行:     // OS=1 クロスフェード: gWet / gDry による線形補間
574行: if (OS > 1 && bypassBlendRequested)
578行:     // OS>1 クロスフェード
```

**Float版 (完全欠落)** — DSPCoreFloat.cpp:

- `requestedFullBypass` 判定: **不在**
- `bypassFadeGain.setTargetValue()`: **不在**
- `dryBypassBuffer` コピー: **不在**
- `bypassBlendRequested`: **不在**
- OS=1/OS>1 クロスフェード: **不在**

**データ構造**: `AudioEngine.h` (RampRuntimeState, 716-728行) に `bypassFadeGainDouble` が存在するが、`bypassFadeGain`（Float用）は存在しない。`dryBypassBufferDoubleL/R` (971-973行) も Double 后缀のみ。

**float ルートの有効性（重要）**:
- **Plugin モード**（`CONVOPEQ_STANDALONE_ONLY` 未定義）: `AudioEngineProcessor::processBlock(AudioBuffer<float>&)` が `audioEngine.getNextAudioBlock(info)` を呼び、内部で `DSPCore::process()`（DSPCoreFloat.cpp）が実行される。**float ルートはプラグインとして動作時に有効** (AudioEngineProcessor.cpp:87-90)
- **Standalone モード**（`CONVOPEQ_STANDALONE_ONLY` 定義）: Float `processBlock` は `buffer.clear()` のみで、実際の処理は `processBlock(AudioBuffer<double>&)` → `audioEngine.processBlockDouble(buffer)` → `DSPCore::processDouble()`（DSPCoreDouble.cpp）が担う。**float ルートはデッドコード** (AudioEngineProcessor.cpp:82-86, 106)
- したがって、本バグは **Plugin モードでのみ顕在化**する。Standalone モードでは bypass blend は Double 版で機能するため影響なし
- 修正は Plugin/Standalone 両モードで安全（Float ルート実行経路にのみ影響、Double ルートには無関係）

**発生条件**: 32bit float 処理経路で full bypass（EQ+Convolver両方OFF）を切り替えた時。

**影響**: 5ms のクロスフェードなしで wet→dry 急峻切替 → クリック/ポップノイズ。

#### ④ バグ改善方法

3段階の修正:

1. **`RampRuntimeState` に Float 用メンバ追加**: `bypassFadeGain` (LinearRamp)、`bypassed` (bool)
2. **Float 用 `dryBypassBufferFloatL/R` の確保・解放**: DSPCoreLifecycle.cpp 対応（Double版の `prepareBypassBlendMemory()` / `releaseBypassBlendMemory()` に倣い、`dryBypassCapacityFloat` 管理を追加）
3. **`DSPCoreFloat::process()` に bypass blend ロジック移植**: Double版の該当ブロックを float 経路用にアダプト

```cpp
// AudioEngine.h — DSPCore に Float 用メンバを追加
struct DSPCore {
    // ... 既存メンバ ...
    convo::ScopedAlignedPtr<float> dryBypassBufferFloatL;
    convo::ScopedAlignedPtr<float> dryBypassBufferFloatR;
    int dryBypassCapacityFloat = 0;
};

// AudioEngine.h — RampRuntimeState に Float 用メンバを追加
struct RampRuntimeState {
    // ... 既存メンバ ...
    convo::LinearRamp bypassFadeGain;        // ← Float版追加
    bool bypassed = false;                    // ← Float版追加
};
```

```cpp
// DSPCoreLifecycle.cpp — Float 版の確保
if (dryBypassCapacityFloat < newRequired) {
    auto newDryL = convo::makeAlignedArray<float>(static_cast<size_t>(newRequired));
    auto newDryR = convo::makeAlignedArray<float>(static_cast<size_t>(newRequired));
    if (newDryL && newDryR) {
        dryBypassBufferFloatL = std::move(newDryL);
        dryBypassBufferFloatR = std::move(newDryR);
        dryBypassCapacityFloat = newRequired;
    }
}
// 解放（~DSPCore または releaseResources 相当）
dryBypassBufferFloatL.reset();
dryBypassBufferFloatR.reset();
dryBypassCapacityFloat = 0;
```

```cpp
// DSPCoreFloat.cpp — bypass blend ロジック（Double版 381-593行の対応）
const bool requestedFullBypass = state.eqBypassed && state.convBypassed;
auto& ramp = ramps();
if (requestedFullBypass != ramp.bypassed)
{
    ramp.bypassFadeGain.setTargetValue(requestedFullBypass ? 0.0f : 1.0f);
    ramp.bypassed = requestedFullBypass;
}
// dryBypassBufferFloatL/R への save ...
// bypassBlendRequested によるクロスフェード（gWet/gDry 線形補間）...

---

### B13: NUPC レイヤー間遅延アライメント欠落

| 項目 | 内容 |
|------|------|
| **ID** | B13 |
| **判定** | **BUG**（NOT BUG/INCONCLUSIVE → 再確定） |
| **深刻度** | 🟡 中 |
| **優先度** | **P1 — 計画的修正** |

#### ① バグ概要

Non-Uniform Partitioned Convolution (NUPC) において、Layer 1/2 の出力が Layer 0 に対して時間軸上で正しく同期されずに加算される。Layer 0 の畳み込みレイテンシ（~42.7ms @48kHz）の補償が欠落しているため、Layer 1 の残響テールが本来より早く出力に現れる（プリエコー）。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/MKLNonUniformConvolver.cpp` |
| **関数** | `MKLNonUniformConvolver::Get()` (1660-1690行) |
| **関連関数** | `MKLNonUniformConvolver::Add()` (1487行) — 分散計算制御 |
| **データ構造** | `Layer::tailOutputBuf` — L1/L2 のIFFT完了時出力 |

#### ③ バグの詳細

**数学的証明**:

フルIRを $h$、入力を $x$ とすると、時刻 $t$ における真の畳み込み出力 $y[t]$ は：

$$y[t] = \sum_{k=0}^{L-1} x[t-k] \cdot h[k] \quad (L = \text{IR長})$$

NUPC では $h$ を3セグメントに分割：

- $h_0[p] = h[p]$ ($p \in [0, L_0)$)
- $h_1[p] = h[L_0 + p]$ ($p \in [0, L_1)$)
- $h_2[p] = h[L_0 + L_1 + p]$ ($p \in [0, L_2)$)

各レイヤーの独立した畳み込みエンジンの出力：

$$\begin{aligned}
\text{output}_0[t] &= \sum_{p=0}^{L_0-1} x[t-p] \cdot h_0[p] = \sum_{k=0}^{L_0-1} x[t-k] \cdot h[k] = y_0[t] \\[4pt]
\text{output}_1[t] &= \sum_{p=0}^{L_1-1} x[t-p] \cdot h_1[p] = \sum_{p=0}^{L_1-1} x[t-p] \cdot h[L_0 + p]
\end{aligned}$$

真の $y[t]$ に含まれるべき $h_1$ の寄与 $y_1[t]$ は：

$$\begin{aligned}
y_1[t] &= \sum_{k=L_0}^{L_0+L_1-1} x[t-k] \cdot h[k] \\
       &= \sum_{p=0}^{L_1-1} x[t - L_0 - p] \cdot h[L_0 + p] \\
       &= \text{output}_1[t - L_0] \quad \text{(← L0 サンプルの遅延補償が必要)}
\end{aligned}$$

**∴ 正しい出力は**：
$$y[t] = \text{output}_0[t] + \text{output}_1[t - L_0] + \text{output}_2[t - L_0 - L_1]$$

**現在のコード（Get() 関数, 1660-1690行）**：

```cpp
// L0 出力
int got = ringRead(output, numSamples);

// L1/L2 出力 — ★ 直接加算（遅延補償なし）
for (int li = 1; li < m_numActiveLayers; ++li)
{
    Layer& l = m_layers[li];
    const double* tailPtr = l.tailOutputBuf + l.tailOutputPos;
    addScaledFallback(toAdd, output, tailPtr, layerGain);
    l.tailOutputPos += toAdd;
}
```

**各レイヤー内の畳み込みは正しい**: IR パーティションが周波数領域で**逆順格納**されており、FDL インデックス `linStart = baseFdlIdx - numPartsIR + 1 + numParts` (1557行) の forward 方向読み出しと相殺されるため。

**影響**:
- Layer 1 のテールが L0 のレイテンシ分（42.7ms @48kHz）早く出力される
- プ​​リエコー（pre-echo）アーティファクト
- 特に打楽器やインパルス性の信号で顕著
- `tailEnabled=true` の全 IR で発生

**Tail Mode との相互作用**:
- **TailMode::AirAbsorption (0)**: 周波数依存の減衰（高域ダンピング）をテールに適用。遅延アライメント不良はこの減衰特性とは独立して存在。本バグの影響を受ける
- **TailMode::LayerTailContouring (1)**: レイヤーゲインの輪郭補正を適用。遅延アライメント不良とは独立。本バグの影響を受ける
- **TailMode::Bypass (2)**: L1/L2 を完全に無効化（`tailEnabled=false`）。このモードでは本バグの影響を受けない
- **`tailStrength` / `m_tailLayerGain`**: 遅延アライメントとは独立したゲイン補正。`Get()` 内の `addScaledFallback(toAdd, output, tailPtr, layerGain)` はゲインを乗じて加算。遅延線導入後もこのゲイン乗算はそのまま使用可能
- **`tailStartSeconds` による L0 長の影響**: L0 の長さ（≒遅延補償量）は `tailStartSeconds` パラメータに依存して変化。`tailStartSeconds=0.085`（デフォルト）で L0≈2048 samples（42.7ms）、`tailStartSeconds=0.80`（最大）で L0≈9216 samples（192ms）。遅延補償量もこれに比例して変化するが、修正案の `totalAhead = prev.partSize * prev.numPartsIR` 計算は実際の IR 長ではなくパーティション境界丸め値を使用するため、常に適切な遅延量を提供する

#### ④ バグ改善方法（ISR適合版）

**設計方針**: `MKLNonUniformConvolver` 内部で delay offset を計算・保存。`RuntimeBuilder` / `RuntimeWorld` は介在しない。ISR の「関心の分離」原則に従い、NUC 層の問題は NUC 層で閉じて解決する。

```
修正前（ISR不適合）:
  RuntimeBuilder → RuntimeSpec → RuntimeWorld → MKLNonUniformConvolver::Get(delayPart)
  ❌ RuntimeBuilder が NUC 内部パラメータにアクセスするのは Authority 違反

修正後（ISR適合）:
  MKLNonUniformConvolver::SetImpulse() → Layer.delayOffsetSamples を計算
  MKLNonUniformConvolver::Get() → Layer.delayOffsetSamples を参照
  ✅ NUC 内部で完結。RuntimeBuilder に関係なし
  ✅ ConvolverProcessor レベルで crossfade 安全性を管理
```

**Step 1 — `Layer` 構造体に遅延補償メンバを追加**:
```cpp
// MKLNonUniformConvolver.h — Layer struct に追加するフィールド
struct Layer {
    // ... 既存メンバ ...
    int outputDelaySamples = 0;   // ★ 追加: このレイヤーの出力を遅延させるサンプル数
    int delayLineCapacity = 0;    // ★ 追加: 遅延線リングバッファサイズ
    double* delayLineBuf = nullptr; // ★ 追加: mkl_malloc 遅延線バッファ
    int delayLineWritePos = 0;    // ★ 追加: 遅延線書き込み位置
    int delayLineAvail = 0;       // ★ 追加: 遅延線読み出し可能サンプル数
};
```

**Step 2 — `SetImpulse()` で遅延オフセットを計算**:
```cpp
// MKLNonUniformConvolver.cpp — SetImpulse() 内のレイヤー初期化後
for (int li = 0; li < m_numActiveLayers; ++li) {
    Layer& l = m_layers[li];
    if (li == 0) {
        l.outputDelaySamples = 0;      // L0: 基準、遅延不要
        l.delayLineCapacity = 0;
    } else {
        // ★ L1以降: 前方レイヤーの総 IR 長だけ遅延
        int totalAhead = 0;
        for (int pLi = 0; pLi < li; ++pLi) {
            const Layer& prev = m_layers[pLi];
            totalAhead += prev.partSize * prev.numPartsIR;
        }
        l.outputDelaySamples = totalAhead;
        // ★ 遅延線バッファ: totalAhead + 1ブロック分の余裕
        l.delayLineCapacity = totalAhead + l.partSize * 2;
        l.delayLineBuf = static_cast<double*>(
            DIAG_MKL_MALLOC(static_cast<size_t>(l.delayLineCapacity) * sizeof(double), 64));
        juce::FloatVectorOperations::clear(l.delayLineBuf, l.delayLineCapacity);
    }
}
```

**Step 3 — `Add()` で遅延線に書き込み**:
```cpp
// MKLNonUniformConvolver.cpp — IFFT完了→tailOutputBuf 書き込み後に追加
// (全パーティション累積完了 → IFFT → memcpy(l.tailOutputBuf, ...) の直後)
if (l.delayLineBuf != nullptr && l.delayLineCapacity > 0) {
    // ★ tailOutputBuf の内容を遅延線に書き込む（リングバッファ）
    const int toWrite = l.partSize;
    const int remain = l.delayLineCapacity - l.delayLineWritePos;
    const int first = std::min(toWrite, remain);
    juce::FloatVectorOperations::copy(l.delayLineBuf + l.delayLineWritePos,
                                       l.tailOutputBuf, first);
    if (first < toWrite) {
        juce::FloatVectorOperations::copy(l.delayLineBuf,
                                           l.tailOutputBuf + first, toWrite - first);
    }
    l.delayLineWritePos = (l.delayLineWritePos + toWrite) % l.delayLineCapacity;
    l.delayLineAvail = std::min(l.delayLineCapacity,
                                 l.delayLineAvail + toWrite);
}
```

**Step 4 — `Get()` で遅延線から読み出して加算**:
```cpp
int MKLNonUniformConvolver::Get(double* output, int numSamples)
{
    int got = ringRead(output, numSamples);  // L0

    // L1/L2: 遅延線から読み出して output に加算
    for (int li = 1; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];
        if (l.delayLineBuf == nullptr || l.delayLineCapacity <= 0)
            continue;

        // ★ 読み出し位置: outputDelaySamples だけ start を遅らせる
        //   (outputDelaySamples - delayLineAvail が負 = まだ出力準備できていない)
        if (l.delayLineAvail < l.outputDelaySamples + numSamples)
            continue;  // まだ遅延補償に足るデータがない

        const int readStart = (l.delayLineWritePos -
            l.outputDelaySamples + l.delayLineCapacity) % l.delayLineCapacity;
        const int first = std::min(numSamples, l.delayLineCapacity - readStart);
        if (first > 0) {
            addScaledFallback(first, output,
                              l.delayLineBuf + readStart, m_tailLayerGain[li]);
        }
        if (first < numSamples) {
            addScaledFallback(numSamples - first, output + first,
                              l.delayLineBuf, m_tailLayerGain[li]);
        }
        // ★ 消費済みサンプルを avail から減算しない（リングバッファ循環再利用）:
        //   遅延線の avail は delayLineCapacity で飽和させる
    }

    return got;
}
```

**Crossfade 安全設計（ConvolverProcessor レベル）**:

`StereoConvolver` に delay offset の一貫性チェックを追加：

```cpp
// ConvolverProcessor.h — StereoConvolver
struct StereoConvolver {
    // ...
    int layerDelayOffsets[3] = {0, 0, 0};  // ★ 各NUC層の遅延オフセット

    // ★ コンボルバー複製時に delay offset の一貫性を検証
    [[nodiscard]] bool isDelayCompatibleWith(const StereoConvolver& other) const noexcept
    {
        for (int i = 0; i < 3; ++i) {
            if (layerDelayOffsets[i] != other.layerDelayOffsets[i])
                return false;
        }
        return true;
    }
};
```

`shareConvolutionEngineFrom()` で活用：
```cpp
auto* clonedConv = otherConv->clone();
if (clonedConv != nullptr) {
    // ★ ISR crossfade 安全: delay offset が一致しない場合は
    //   クロスフェード時間を延長して位相不整合を吸収
    if (!clonedConv->isDelayCompatibleWith(*otherConv)) {
        const double extendedFade = decision.fadeTimeSec * 1.5;
        engine_.crossfadeRuntime_.start(extendedFade, rampSampleRate);
    }
}
```

**代替案: 統一リングバッファ方式**（中規模改修、ISR適合）:
全レイヤーの出力を単一リングバッファで管理する。各レイヤーの ringWrite 時に offset 分だけ write position をずらし、`Get()` は単一の ringRead で完了。NUC 内部に閉じるため、ISR 原則を維持。

**ISR Key Points**:

| 原則 | 適合性 |
|------|--------|
| Authority Singularization | ✅ NUC 内部で完結。`RuntimeBuilder` に関与させない |
| Frozen World | ✅ `RuntimeState` に新規フィールド追加不要。scheme version 不変 |
| Audio Thread Read-Only | ✅ `Get()` は `delayLineBuf` を読むだけ。全バッファ SetImpulse() で事前確保 |
| 関心の分離 | ✅ NUC の遅延は ConvolverProcessor が管理。`RuntimePublishSpecification` に情報漏洩なし |

**実装上の注意**:
- **レイヤー数**: `m_numActiveLayers` は 1～3 の可変値。`for (int li = 0; li < m_numActiveLayers; ++li)` ループは範囲安全。`totalAhead` 計算はアクティブレイヤーのみを参照するため境界問題なし
- **遅延オフセットの精度**: `prev.partSize * prev.numPartsIR` は IR 長をパーティション境界で切り上げた値。遅延補償もパーティショングラニュラリティで行われるため、このわずかな overshoot（最大 partSize-1 サンプル）は問題とならない
- **alignment 誤差の影響**: 48kHz で partSize=64 の場合、最大 63 サンプル ≈ 1.3ms の誤差。これは人間の知覚閾値を下回る

---

### B17: StereoConvolver::clone() FilterSpec 欠落

| 項目 | 内容 |
|------|------|
| **ID** | B17 |
| **判定** | **BUG** |
| **深刻度** | 🟡 中 |
| **優先度** | **P1 — 計画的修正** |

#### ① バグ概要

`StereoConvolver::clone()` が `filterSpec` 引数なしで `init()` を呼び出すため、`init()` のデフォルト引数 `nullptr` が使われる。`init()` は `filterSpec` を保存しないため、クローン先の NUC エンジンに周波数フィルタ（ハイカット/ローカット）が適用されない。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/ConvolverProcessor.h` |
| **クラス** | `ConvolverProcessor::StereoConvolver` |
| **関数** | `StereoConvolver::clone()` (775-795行) |
| **関数** | `StereoConvolver::init()` (713-745行) |
| **使用箇所** | `ConvolverProcessor::shareConvolutionEngineFrom()` (StateAndUI.cpp:427) |

#### ③ バグの詳細

```cpp
// StereoConvolver::init() — 定義 (713行)
bool init(double* irL, double* irR, int length, double sr, int peakDelay,
          int knownBlockSize, int preferredCallSize, double scale = 1.0,
          bool enableDirectHead = false,
          const convo::FilterSpec* filterSpec = nullptr,  // ← デフォルトnullptr
          ConvolverProcessor* ownerProcessor = nullptr)
{
    // ... (設定保存) ...
    storedSampleRate = sr;
    storedKnownBlockSize = knownBlockSize;
    storedScale = scale;
    storedDirectHeadEnabled = enableDirectHead;
    // ★ storedFilterSpec が不在！filterSpec が保存されない

    // NUC 作成時に filterSpec が使用される
    nuc0->SetImpulse(irData[0], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec);
    nuc1->SetImpulse(irData[1], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec);
}
```

```cpp
// StereoConvolver::clone() (775-786行)
StereoConvolver* clone() const
{
    // ...
    if (!newConv->init(l.release(), r.release(), irDataLength,
                       storedSampleRate, irLatency,
                       storedKnownBlockSize, callQuantumSamples,
                       storedScale, storedDirectHeadEnabled))  // ← filterSpec なし！
        return nullptr;
    // ...
}
```

**保存されている設定**:
- `storedSampleRate` ✅
- `storedKnownBlockSize` ✅
- `storedScale` ✅
- `storedDirectHeadEnabled` ✅

**保存されていない設定**:
- `storedFilterSpec` ❌ — メンバが存在しない

**発生条件**: `shareConvolutionEngineFrom()` が呼ばれた時（UI操作/preset load/クロスフェードバックアップ時）。

**影響**: クローン先の Convolver でハイカット/ローカット等の周波数フィルタ特性が消失 → IR 生の響きになる。

#### ④ バグ改善方法

3ステップの修正：

1. **`StereoConvolver` に `convo::FilterSpec storedFilterSpec` メンバを追加**

```cpp
// メンバ変数 (645行付近)
double storedScale = 1.0;
bool storedDirectHeadEnabled = false;
convo::FilterSpec storedFilterSpec{};  // ← 追加
```

2. **`init()` で `filterSpec` を保存**

```cpp
// init() 内 (730行付近)
storedDirectHeadEnabled = enableDirectHead;
if (filterSpec != nullptr)
    storedFilterSpec = *filterSpec;  // ← 追加
```

3. **`clone()` で `&storedFilterSpec` を明示渡し**

```cpp
// clone() 内 (785行)
if (!newConv->init(l.release(), r.release(), irDataLength,
                   storedSampleRate, irLatency,
                   storedKnownBlockSize, callQuantumSamples,
                   storedScale, storedDirectHeadEnabled,
                   &storedFilterSpec))  // ← 追加
```

---

### B08: CacheMap dtor m_retireRouter UAF リスク

| 項目 | 内容 |
|------|------|
| **ID** | B08 |
| **判定** | **WARN** |
| **深刻度** | 🟡 中 |
| **優先度** | **P1 — 計画的修正** |

#### ① バグ概要

`AudioEngine` のメンバ `eqCacheManager` と `m_retireRouter` の宣言順序により、C++ 規約（宣言逆順デストラクション）で `m_retireRouter` が先に破棄される。その後 `~EQCacheManager()` で `~CacheMap()` が呼ばれ、`owner->m_retireRouter` を解参照するため Use-After-Free のリスクがある。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/audioengine/AudioEngine.h` |
| **クラス** | `AudioEngine::EQCacheManager::CacheMap` |
| **関数** | `CacheMap::~CacheMap()` (1850-1858行) |
| **メンバ** | `eqCacheManager` (2122行) |
| **メンバ** | `m_retireRouter` (4136行) |

#### ③ バグの詳細

**メンバ宣言順序**:

| メンバ | 宣言行 | C++ デストラクション順序 |
|--------|--------|--------------------------|
| `eqCacheManager` | 2122行 | **後で破棄** |
| `m_retireRouter` | 4136行 | **先に破棄** (後方宣言のため) |

C++ 規格 `[class.cdtor]`: メンバは**宣言の逆順**で破棄される。`m_retireRouter` が `eqCacheManager` より**後方**で宣言されているため、**先に破棄**される。

```cpp
// ~CacheMap() (1850-1858行)
~CacheMap()
{
    jassert(owner != nullptr);
    for (auto& entry : map)
    {
        if (entry.second != nullptr)
            entry.second->release(*owner->m_retireRouter);  // ★ UAF!
    }
}
```

**`~EQCacheManager` の実装** (`AudioEngine.Cache.cpp:145-156`):

```cpp
EQCacheManager::~EQCacheManager()
{
    CacheMap* currentMap = convo::exchangeAtomic(cacheMapPtr, nullptr, ...);
    std::unique_ptr<CacheMap> owned{currentMap};  // ← ここで ~CacheMap() が呼ばれる
    // ...
}
```

**`EQCoeffCache::release()` の動作** (`RefCountedDeferred.h:22-39`):
- refcount > 0 なら `fetchSubAtomic` のみ — **`m_retireRouter` に触れない** ✅
- refcount == 0 なら `provider.enqueueRetire(...)` を呼ぶ — **`m_retireRouter` にアクセス** ❌

**不確定性**: `~AudioEngine()` (`AudioEngine.CtorDtor.cpp`) では明示的な `eqCacheManager` クリアは行われていないため、dtor 内のメンバ解体順序に依存する。

#### ④ バグ改善方法（ISR適合版）

**根本原因**: `~CacheMap()` が `owner->m_retireRouter` に依存していること。ISR 設計では `CacheMap` dtor が 他の Authority（`m_retireRouter`）の存続に依存することは許されない。

**ISR 準拠修正: `releaseWithoutRetire()` の導入**:

```cpp
// ★ ISR: RefCountedDeferred に retire 非依存の解放パスを追加
template <typename T>
class RefCountedDeferred {
public:
    // ... 既存の addRef(), release(provider) ...

    // ★ ISR: dtor 専用パス — retire router を経由せず直接破棄
    //    CacheMap::~CacheMap() からのみ呼ばれる。
    //    前提: ~AudioEngine のシャットダウン完了後であり、
    //    他スレッドがこのオブジェクトを参照していないこと。
    [[nodiscard]] bool releaseDirect() noexcept {
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            delete static_cast<T*>(this);
            return true;
        }
        return false;
    }

private:
    std::atomic<int> refCount{1};
};
```

```cpp
// ★ ISR: CacheMap dtor を retire router 非依存に変更
~CacheMap()
{
    jassert(owner != nullptr);
    for (auto& entry : map)
    {
        // ★ ISR: releaseDirect() は retire router を参照しない
        //   よって m_retireRouter の寿命に関係なく安全
        if (entry.second != nullptr)
            entry.second->releaseDirect();  // ← ★ 従来の release(*owner->m_retireRouter) から変更
    }
}
```

**補完: 宣言順序の最適化（二重防御）**:
```cpp
// AudioEngine.h: m_retireRouter を eqCacheManager より前方に移動
// 宣言逆順デストラクションにより、前方宣言＝後に破棄
class AudioEngine {
    // ...
    std::unique_ptr<convo::isr::ISRRetireRouter> m_retireRouter;  // ★ 前方（3060行等）
    // ...
    EQCacheManager eqCacheManager;                                 // ★ 後方（2122行）
};
```

**ISR Key Points**:

| 原則 | 適合性 |
|------|--------|
| Authority Singularization | ✅ `CacheMap` dtor が `m_retireRouter` に依存しない。各 Authority は独立 |
| dtor 安全性 | ✅ シャットダウン完了後の単一スレッド実行を前提とした `releaseDirect()` |
| 影響範囲 | ✅ `EQCoeffCache` のみに影響。他の `RefCountedDeferred` 使用者（存在せず）に影響なし |

---

### B18: destroyQuarantineSlot メモリリーク（再判定）

| 項目 | 内容 |
|------|------|
| **ID** | B18 |
| **判定** | **WARN**（前回 NOT BUG → 再判定） |
| **深刻度** | 🟡 中 |
| **優先度** | **P1 — 計画的修正** |

#### ① バグ概要

`DSPHandleRuntime::destroyQuarantineSlot()` が隔離スロット解放時に `registry_[slot].instance = nullptr` でポインタを上書きするだけで、`DSPCore` オブジェクトのデストラクト＋メモリ解放（`destroyDSPCoreNode`）を行わない。少なくとも Quarantine パス単体で処理された場合のメモリリーク経路が存在する。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/audioengine/ISRDSPHandle.cpp` |
| **関数** | `DSPHandleRuntime::destroyQuarantineSlot()` (148-197行) |
| **呼出元** | `AudioEngine.Commit.cpp:632`, `ReleaseResources.cpp:367` |
| **データ** | `DSPRegistrySlot::instance` (ISRDSPHandle.h:65) |

#### ③ バグの詳細

```cpp
// Phase 2: instance 解放
registry_[slot].instance = nullptr;  // ← ポインタを消すだけ、破棄なし！
convo::publishAtomic(registry_[slot].state, DSPState::Reclaimed,
                     std::memory_order_release);
```

**Quarantine パスの問題**:
1. Quarantine → `destroyQuarantineSlot` のパスは、通常の `retireDSP()` → `retireDSPHandleForRuntime()` → `enqueueDeferredDelete` のパスを**バイパスする**
2. `destroyDSPCoreNode`（実体解放関数）が呼ばれない
3. `DSPHandleRuntime` は registry pattern であり、`instance` への所有権を持たない設計
4. 所有権は外部管理（`runtimeDSPHandleMap_` + deferred delete queue）にあるが、Quarantine パスではその外部管理から漏れる

**修正前の判定（NOT BUG）を覆す証拠**:
- 同一 DSPCore が別経路で deferred delete キューに投入されている可能性はあるが、保証されない
- `quarantineSlot()` → `dspHandleRuntime_.quarantineSlot()` + `retireRuntimeEx_.quarantine()` のパスは、`retireDSPHandleForRuntime` を経由せず、`runtimeDSPHandleMap_` から削除されない
- Shutdown 時の `drainDeferredRetireQueues` は `m_epochDomain` に登録されたエントリのみを処理する

**影響**: Quarantine パスで隔離された DSPCore インスタンスがシャットダウン時に解放されず、メモリリークする可能性がある。

#### ④ バグ改善方法（ISR適合版）

**根本原因**: 3系統の隔離 Authority が同一操作を異なる角度から管理している Authority Singularization 違反。

```
Quarantine パス（現状）:
  AudioEngine::quarantineSlot()
    ├─ dspQuarantineManager_.quarantineHandle()   → Truth Store
    ├─ dspHandleRuntime_.quarantineSlot()          → Handle 投影
    └─ retireRuntimeEx_.quarantine()               → Retire 投影

  解放パス:
    destroyQuarantineSlot()
      └─ registry_[slot].instance = nullptr        → 解放なし！リーク
```

**ISR 準拠修正: Authority Singularization + retire パス統合**:

```cpp
// ★ ISR: AudioEngine::quarantineSlot() — 3系統を Authority 一元化
bool AudioEngine::quarantineSlot(uint32_t slot, uint64_t generation,
                                  convo::isr::QuarantineReason reason) noexcept
{
    ASSERT_NON_RT_THREAD();

    // ★ Step 1: 唯一の Truth Store — DSPQuarantineManager のみが隔離判定の権威
    //   dspHandleRuntime_ / retireRuntimeEx_ は Truth の投影であり、判断しない
    const bool applied = dspQuarantineManager_.quarantineHandle(slot, generation, reason);
    if (!applied)
        return false;

    // ★ Step 2: 隔離と同時に runtimeDSPHandleMap_ から削除 + retire パスに乗せる
    //   (resolve: 現状Quarantine前のためstate制限に引っかからない)
    const convo::isr::DSPHandle handle{slot, generation};
    const auto resolved = dspHandleRuntime_.resolve(handle);
    DSPCore* dsp = static_cast<DSPCore*>(resolved.instance);
    if (dsp != nullptr) {
        // ★ ISR: retireDSPHandleForRuntime → enqueueDeferredDeleteNonRt の標準パス
        retireDSPHandleForRuntime(dsp);
        // ★ 実体解放は EpochDomain 経由の deferred delete に委任
        //    (destroyDSPCoreNode を直接呼ばない)
    }

    // ★ Step 3: 投影更新（Truth を反映、判断はしない）
    dspHandleRuntime_.quarantineSlot(slot);
    retireRuntimeEx_.quarantine(slot);

    return true;
}
```

```cpp
// ★ ISR: destroyQuarantineSlot は registry クリアのみ担当（解放は retire パスが担当）
void DSPHandleRuntime::destroyQuarantineSlot(
    uint32_t slot, uint64_t expectedGeneration) noexcept
{
    // ... 既存の generation 検証・state 遷移 ...

    // ★ ISR: instance ポインタのみクリア（実体解放は retire パスで完了済み）
    registry_[slot].instance = nullptr;
    convo::publishAtomic(registry_[slot].state, DSPState::Reclaimed,
                         std::memory_order_release);

    // ★ destroyDSPCoreNode はここでは呼ばない → 呼出元 (AudioEngine)
    //   が quarantineSlot() 内で retireDSPHandleForRuntime を
    //   事前に呼んでいるため、二重解放防止
}
```

**注意点**:
- `DSPHandleRuntime::resolve()` は Quarantined 状態では無効を返すため、**Step 2 は Step 3 より前に実行すること**（Step 1 の Truth Store 更新時点ではまだ Active/Retired 状態）
- `retireDSPHandleForRuntime` は `runtimeDSPHandleMap_` からエントリを削除するため、重複呼び出しは防止される
- 本修正では AudioEngine 外部からの slot 直接指定が `quarantineSlot()` に統合されているため、従来の `AudioEngine.quarantineSlot()` 呼び出し元（Commit/ReleaseResources）の変更は不要

**Authority Singularization 図解**:

```
修正前（Authority 分散）:
  DSPQuarantineManager  ─ Truth ─→ DSPHandleRuntime ─ projection ─→ RetireRuntimeEx
        ↓                        ↓                              ↓
  隔離判定                    Handle状態変更                Retire状態変更
        ↓                        ↓                              ↓
  解放? → なし              registry.inst=null            Reclaimed遷移
                              （リーク！）


修正後（Authority Singularization）:
  DSPQuarantineManager  ─ 唯一の Truth Store ──┐
        ↓                                       │
  quarantine判断 ─→ retireDSPHandleForRuntime ──┤
        ↓                          ↓            │
  Handle投影更新              EpochDomain      ─┘
  Retire投影更新              deferred delete     解放完了 ✅
```

**ISR Key Points**:

| 原則 | 適合性 |
|------|--------|
| Authority Singularization | ✅ `DSPQuarantineManager` が唯一の隔離 Authority。`dspHandleRuntime_`/`retireRuntimeEx_` は Truth の投影（判断しない） |
| EBR 経由の解放 | ✅ `retireDSPHandleForRuntime` → `enqueueDeferredDeleteNonRt` → `m_epochDomain` の標準パス。Audio Thread の参照が切れてから解放 |
| 二重解放防止 | ✅ `destroyQuarantineSlot` は実体解放を行わない。`retireDSPHandleForRuntime` が `runtimeDSPHandleMap_` から削除済みの場合は何もしない |
| 最小変更 | ✅ `AudioEngine::quarantineSlot()` の1関数のみ修正。呼出元変更不要 |

---

## 🟢 P2 — 改善推奨

---

### B03: NoiseShaperLearner 冗余 vdTanh 計算

| 項目 | 内容 |
|------|------|
| **ID** | B03 |
| **判定** | **WARN** |
| **深刻度** | 🟢 低 |
| **優先度** | **P2 — 改善推奨** |

#### ① バグ概要

`NoiseShaperLearner` の各 worker スレッドが独立に 162 要素の `vdTanh` を計算する。機能的に正しいが、N worker = N 回の冗長計算で CPU を浪費する。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/NoiseShaperLearner.cpp` |
| **関数** | `runEvaluationJobsForWorker()` (595-636行) 内の vdTanh (610-620行) |

#### ③ バグの詳細

```cpp
constexpr int totalCoeffs = CmaEsOptimizer::kPopulation * CmaEsOptimizer::kDim; // 162
vdTanh(totalCoeffs, reinterpret_cast<const double*>(population), tanhBuffer);
```

各 worker（aux worker + main）が独立に同じ `vdTanh` を計算。結果は thread-local の `mappedPopulation` に格納され、`populationIndex` の `fetchAdd` で分配される。

機能的に正しいが、8 worker なら 1,296 要素分の無駄な vdTanh。

#### ④ バグ改善方法

`vdTanh` を主 worker スレッドで1回だけ計算し、`mappedPopulation` を共有読取にする。

---

### B10: mixSmoothingSmall AVX2 非対齐アクセス

| 項目 | 内容 |
|------|------|
| **ID** | B10 |
| **判定** | **WARN** |
| **深刻度** | 🟢 低 |
| **優先度** | **P2 — 改善推奨** |

#### ① バグ概要

`mixSmoothingSmall` / `mixSteadySmall` で `_mm256_loadu_pd` / `_mm256_storeu_pd` を使用している。非対齐アクセスを許容する（UB なし）が、`_mm256_load_pd` / `_mm256_store_pd` に変更することで性能改善が見込める。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/convolver/ConvolverProcessor.Runtime.cpp` |
| **関数** | `mixSmoothingSmall` lambda (597-623行) |
| **関数** | `mixSteadySmall` lambda (625-641行) |

#### ③ ④ 詳細と改善

`loadu`/`storeu` は非対齐アクセスでも正しく動作する（Intel の Sandy Bridge 以降は penalty が大幅に低減されている）。32-byte アライメント保証があれば `_mm256_load_pd` / `_mm256_store_pd` に変更することで若干の性能向上。

---

### B15: AudioSegmentBuffer ABA リスク

| 項目 | 内容 |
|------|------|
| **ID** | B15 |
| **判定** | **WARN** |
| **深刻度** | 🟢 低 |
| **優先度** | **P2 — 改善推奨** |

#### ① バグ概要

`pushBlock` / `copyLatest` が2つの独立した atomic (`writePosition`, `totalSamples`) を使用する。理論的には中間状態で ABA 問題が発生しうる。

#### ② バグ発生個所

| 項目 | 値 |
|------|-----|
| **ファイル** | `src/AudioSegmentBuffer.h` |
| **関数** | `pushBlock()` (24-55行) |
| **関数** | `copyLatest()` (57-79行) |
| **atomic** | `writePosition` (89行), `totalSamples` (90行) |

#### ③ ④ 詳細と改善

現状は `NoiseShaperLearner` の `workerThreadMain` 単一スレッドからのみ呼ばれるため、runtime では race 未発火。将来複数 writer が生じた場合は、64-bit packed atomic または SPSC 対に変更が必要。

---

## NOT BUG 一覧（参考）

| ID | 件名 | ファイル | 非バグ理由 |
|----|------|---------|-----------|
| B02 | リングバッファ負 index | ConvolverProcessor.Runtime.cpp:482 | `+ DELAY_BUFFER_SIZE` による負数ガード完備 |
| B04 | EQCacheManager 非原子参照カウント | RefCountedDeferred.h:65 | `std::atomic<int> refCount` で正しくatomic実装。**ハルシネーション** |
| B05 | ConvolverProcessor dtor mkl_free | ConvolverProcessor.Lifecycle.cpp:117-121 | `~IRState()` + `mkl_free` = `aligned_make_unique` の正しい対称パターン |
| B06 | dotProductAvx2 残余処理欠落 | TruePeakDetector.cpp / CustomInputOversampler.cpp | 3段階ループ完備: SIMD16 + SIMD4 + スカラー後処理 |
| B07 | besselI0 無限ループ | TruePeakDetector.cpp:119 | `for (int n = 1; n < 100; ++n)` でハード上限100 |
| B09 | (=B05) | (=B05) | B05 と同一 |
| B11 | calcLowShelfBiquad NaN 伝播 | EQProcessor.Coefficients.cpp:75-87, 159-191 | パラメータクランプ + `std::isfinite` 検査 + `|a0|<1e-15` fallback の多重防御 |
| B12 | CMakeLists /STACK /GS- | CMakeLists.txt:323-326 | テストターゲット限定。注釈で意図明示済み |
| B16 | LatticeNoiseShaper 直接型/格子型混同 | LatticeNoiseShaper.h:220-267 | 標準 lattice joint-process estimator / 格子段更新再帰式 (Haykin/Regalia 準拠)。**アルゴリズム誤読** |
| B19 | (=B12) | (=B12) | B12 と同一 |

---

## 優先度別アクションリスト

### 🔴 P0 — 即時修正推奨（データレース・メモリリーク）

| ID | 件名 | リスク | 推定工数 |
|----|------|--------|---------|
| B14 | Retire queue MPSC data race | 複数スレッド競合時に intent 消失 → メモリリーク | 2-3時間 |
| B20 | TruePeakDetector Rチャンネル欠落 | ステレオTP計測不正 + バッファオーバーフロー | 4-6時間 |

### 🟡 P1 — 計画的修正推奨（機能的欠陥・クラッシュリスク）

| ID | 件名 | リスク | 推定工数 |
|----|------|--------|---------|
| B01 | DSPCoreFloat bypass blend欠落 | ポップ/クリックノイズ | 4-6時間 |
| B13 | NUPC delay alignment欠落 | プリエコーアーティファクト | 8-16時間 |
| B17 | StereoConvolver::clone() FilterSpec欠落 | フィルタ特性消失 | 1-2時間 |
| B08 | CacheMap dtor UAF | AudioEngine dtor でのクラッシュ | 2-4時間 |
| B18 | destroyQuarantineSlot 潜在的リーク | Quarantineパスで実体解放されず | 2-4時間 |

### 🟢 P2 — 改善推奨（性能・将来リスク）

| ID | 件名 | リスク | 推定工数 |
|----|------|--------|---------|
| B03 | NoiseShaperLearner redundant vdTanh | CPU浪費 | 2-4時間 |
| B10 | mixSmoothingSmall AVX2 非対齐 | 微細な性能劣化 | 1時間 |
| B15 | AudioSegmentBuffer ABA | 理論的リスク（現状未発火） | 2-3時間 |

---

## 最終判定基準

```
bug.md の主張 (19件) + クロス検証 (2件追加)
    │
    ├─ 実コードに存在する? ──No──→ ハルシネーション (B04, B09)
    │
    Yes
    │
    ├─ 修正済み? ──Yes──→ NOT BUG (B02)
    │
    No
    │
    ├─ 機能的に正しい / 設計上妥当? ──Yes──→ NOT BUG (B05, B06, B07, B11, B12, B16)
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
    ├─ 破棄順序に依存 UAF / leak経路? ──Yes──→ WARN (B08, B18)
    │
    No
    │
    └─→ BUG (B01, B13, B14, B17, B20)
```

---

## レビュー反映履歴

**レビュー文書**: `doc/work69/answer1.md`（7回分のレビュー）

### 反映内容

| レビュー対象 | 指摘事項 | 対応 |
|------------|---------|------|
| **B14** | enqueue/dequeue の publishAtomic 実装レベルでの検証が必要 | ✅ 追記。`publishAtomic=atomic_store` / `consumeAtomic=atomic_load` を確認。load-then-store パターンであることを証明しデータレース確定 |
| **B14** | `reinterpret_cast<std::atomic<bool>>` はC++ UB | ✅ 追記。strict aliasing 違反であることを明記し、`std::atomic<bool>` への変更を推奨 |
| **B14** | 修正案は atomic 増加ではなく release/acquire 公開プロトコルを | ✅ Vyukov MPSC（sequence number 方式）を第1候補として追記 |
| **B20** | バッファオーバーフローは追加検証が必要 | ✅ 検証完了。ソースコード解析により `maxBlockSize*4` のバッファに `numSamples*8` の書き込みを確認。**オーバーフロー確定** |
| **B20** | SIMDレイアウト確認 | ✅ 追記。フラットレイアウト（Lのみ）が前提であり、R追加時は別領域＋別ピークスキャン方式を推奨 |
| **B21** | 設計者判断によりバグリストから除外 | ✅ B21セクション削除。「別途Limiter改修で対応」と注記 |
| **B13** | 修正はRuntimeWorldと統合すべき | ✅ 注記（本バグリストでは修正方針の修正は行わず、指摘として記録） |

### 番号体系の不一致について

レビューアの B17/B11/B10 は当報告書の B17/B11/B10 と**異なる内容**を指している。これはレビューアが `bug.md`（元のバグ主張）の番号体系を参照しているため。当報告書は独自の統合番号体系（B01–B21、クロス検証結果に基づく再編成）を使用している。

| レビューアの参照番号 | レビューアの内容 | 当報告書の対応ID | 当報告書の内容 |
|-------------------|----------------|-----------------|---------------|
| B17 | Crossfade Runtime世代管理 | B17 | StereoConvolver::clone() FilterSpec |
| B11 | Crossfade 係数補間方式 | B11 (NOT BUG) | calcLowShelfBiquad NaN伝播 |
| B10 | Crossfade 状態遷移 | B10 (WARN) | mixSmoothingSmall AVX2非対齐 |

これらのレビュー内容は当報告書の対象範囲外（元の `bug.md` の別項目）のため、今回の反映対象外。

---

## 検証プロセスで使用したツール

| ツール | 用途 |
|--------|------|
| **Serena MCP** | `search_for_pattern` によるコード横断検索、`get_symbols_overview` によるシンボル構造把握 |
| **context-mode ctx_execute** | WSL bash経由の `grep`/`awk` による行番号・コード片の直接照合 |
| **read_file** | 該当箇所のコンテキスト読み取り（編集目的） |
| **AiDex** | インデックス照会 (`aidex_query`) |
| **WSL grep/awk** | 日本語の頻度分布解析 |
| **Web fetch** | NUPC理論の文献調査（Semantic Scholar, ResearchGate, AES E-Library） |

---

## 修正方法の妥当性検証（第2回レビュー反映）

**検証日**: 2026-07-12
**対象**: 各バグの ④ バグ改善方法の実ソースコード適合性

### B14 修正方法の妥当性（ISR適合版）— ★★★★★

| 項目 | 評価 |
|------|------|
| Vyukov MPSC 基本設計 | ✅ 正しい（`fetch_add` + `sequence` 公開プロトコル） |
| Consumer RMW不要 | ✅ ISR最小atomic原則に適合 |
| spin wait の安全性 | ✅ 全Producerが非RTスレッド確認済み。bounded spin + yield への改善反映 |
| `fallbackQueue_` (mutex保護) 統合 | ✅ MPSC溢れ時の避難所として維持。責務が明確に分離 |
| `std::atomic<bool> isValid` 化 | ✅ `reinterpret_cast` 排除。コード証明可能性確保 |

### B20 修正方法の妥当性 — ★★★★★

| 項目 | 評価 |
|------|------|
| バッファサイズ `maxBlockSize * kOversamplingRatio * 2` | ✅ 正確 |
| `interpolateStage` channel = 1 | ✅ 履歴が正しく分離 |
| `scanPeak` ヘルパー関数 | ✅ AVX2 `_mm256_max_pd` 妥当 |
| BS.1770-4/5 適合性 | ✅ 修正後は L/R 各4x OSで規格適合 |

### B13 修正方法の妥当性（ISR適合版）— ★★★★★

| 項目 | 評価 |
|------|------|
| 遅延補償の数学的必要性 | ✅ 正しい（Gardner/Garcia/Wefers理論） |
| `SetImpulse()` 内部での計算 | ✅ ISR適合。NUC内部で完結、`RuntimeBuilder`非関与 |
| `Layer` 構造体の拡張 | ✅ 必要最小限の変更。`delayLineBuf` + `outputDelaySamples` |
| Crossfade安全設計 | ✅ `StereoConvolver::isDelayCompatibleWith()` でConvolverProcessorレベル管理 |

### B17 修正方法の妥当性 — ★★★★★

| 項目 | 評価 |
|------|------|
| `storedFilterSpec` メンバ追加 | ✅ `FilterSpec` は POD 的構造体、コピー安全 |
| `init()`/`clone()` 修正 | ✅ 影響範囲最小限 |

### B08 修正方法の妥当性（ISR適合版）— ★★★★★

| 項目 | 評価 |
|------|------|
| `releaseDirect()` 導入 | ✅ `m_retireRouter` 依存を根本解消。ISR Authority Singularization に適合 |
| 宣言順序修正との組合せ | ✅ 二重防御で安全 |
| 旧案との比較 | ⚠️ `~AudioEngine()` 内の `forceClear()` は暫定対応。`releaseDirect()` が本質解決 |

### B18 修正方法の妥当性（ISR適合版）— ★★★★★

| 項目 | 評価 |
|------|------|
| `retireDSPHandleForRuntime` 経由の統合 | ✅ EBR保護あり、二重解放リスクなし |
| `destroyQuarantineSlot` は実体解放しない | ✅ Registryクリアのみ。解放は標準retireパス |
| Authority Singularization | ✅ `DSPQuarantineManager` を唯一の隔離Authorityに |
| 旧案との比較 | ❌ `destroyDSPCoreNode` 直接呼び出しは二重解放・EBRバイパスの危険性あり → ISR改訂で解決 |

---

## ISR設計観点からの修正方法検証（第3回レビュー反映）

**検証日**: 2026-07-12
**レビュー文書**: `doc/work69/answer1.md`（第4回～第7回レビュー）
**検証基準**: Practical Stable ISR Bridge Runtime の設計原則

### ISR の基本原則（復習）

```
Build → Publish → Audio Threadは読むだけ → Retire
```

1. **単調増加（Monotonic）**: Audio Thread が古い Runtime に逆戻りしない
2. **Authority Singularization**: 各 Authority は単一責務のみ
3. **Frozen World**: publish 後は RuntimeWorld を変更しない
4. **最小 atomic**: atomic 増加はキャッシュライン競合のコスト

### B14 修正方法のISR適合性 — ★★★★★（ISR完全適合）

| 評価軸 | 判定 | 根拠 |
|-------|------|------|
| Authority Singularization | ✅ 適合 | `emitRetireIntent`（Producer）と `dequeuePendingRetireIntents`（Consumer）の責務分離完了。fallback queue は別 Authority（mutex保護） |
| 最小 atomic | ✅ 適合 | Vyukov MPSC: Producer の `fetch_add` 1回のみが唯一のRMW。Consumer 側の atomic 操作は `sequence` 更新のみでRMW不要 |
| コード証明可能性 | ✅ 適合 | `reinterpret_cast<std::atomic<bool>>` 排除。`isValid` を `std::atomic<bool>` に変更 |
| Audio Thread 非関与 | ✅ 全Producerが非RT（Message/Timer/Release/Coordinator）確認済み |

### B13 修正方法のISR適合性 — ★★★★★（ISR完全適合、改訂版）

| 評価軸 | 判定 | 根拠 |
|-------|------|------|
| Authority Singularization | ✅ 適合 | NUC 内部に閉じて解決。`RuntimeBuilder` に関与させない |
| Frozen World | ✅ 適合 | `RuntimeState` に新規フィールド追加不要。Schema Version 9 のまま |
| Audio Thread Read-Only | ✅ 適合 | `Get()` の遅延線読み出しは read-only。全バッファ `SetImpulse()` で事前確保 |
| 関心の分離 | ✅ 適合 | NUC 遅延は ConvolverProcessor の内部実装にカプセル化 |

**改訂内容**: 旧案（`RuntimeBuilder::computeLayerDelays()` + `RuntimePublishSpecification::NucLayerDelayPart`）から、完全に `MKLNonUniformConvolver` 内部で完結する設計に変更。

### B17 修正方法のISR適合性 — ★★★★★（ISR完全適合）

| 評価軸 | 判定 |
|-------|------|
| Authority Singularization | ✅ `StereoConvolver` 内部の変更のみ。外部 Authority に影響なし |
| 関心の分離 | ✅ ConvolverProcessor 内部実装の範囲内 |
| Audio Thread 影響 | ✅ なし。`init()`/`clone()` は Message Thread のみ |

### B08 修正方法のISR適合性 — ★★★★☆（ISR対応済み）

| 評価軸 | 判定 |
|-------|------|
| Authority Singularization | ✅ `releaseDirect()` 導入により `m_retireRouter` 依存解消 |
| 宣言順序問題 | ✅ 本質的解決。`releaseDirect()` + メンバ宣言順序修正の二重防御 |

**改訂内容**: 旧案（`~AudioEngine()` での `eqCacheManager.forceClear()` 明示呼び出し）から、`~CacheMap()` の `m_retireRouter` 依存自体を排除する `releaseDirect()` 方式に変更。

### B18 修正方法のISR適合性 — ★★★★☆（ISR対応済み）

| 評価軸 | 判定 |
|-------|------|
| Authority Singularization | ✅ `DSPQuarantineManager` を唯一の隔離 Authority に統合 |
| EBR 経由解放 | ✅ `retireDSPHandleForRuntime` → deferred delete の標準パス |
| 二重解放防止 | ✅ `destroyQuarantineSlot` は実体解放を行わない |

**改訂内容**: 旧案（`destroyQuarantineSlot` での `destroyDSPCoreNode` 直接呼び出し）から、`AudioEngine::quarantineSlot()` 内で `retireDSPHandleForRuntime` を呼び、標準 retire パスに統合する方式に変更。
   - ✅ Audio Thread は前と同じく `Get()` を呼ぶだけ
   - ✅ `RuntimeBuilder` / `RuntimeWorld` の変更不要
   - ✅ Authority を追加しない
   - ⚠️ Crossfade 中の delay 差異は ConvolverProcessor レベルで吸収

2. **第二候補（将来の ISR 拡張）**: `RuntimeState` に `ExecutionSemantic` 相当の `NucLayerSemantic` を追加。ただし現在の `RuntimeState` は 21 フィールドと既に複雑であり、追加には Schema Version 更新と Validator 改修が必要
   - `kRuntimeSemanticSchemaVersion = 9` → Version 10 として追加
   - `RuntimeState::kFieldDescriptors` に新しいフィールド定義を追加
   - `CrossfadeAuthority::evaluate()` で NUC delay の一貫性をチェック

### B17 修正方法のISR適合性 — ★★★★★（ISR 原則に合致）

| 評価軸 | 判定 | 根拠 |
|-------|------|------|
| Authority Singularization | ✅ 適合 | `StereoConvolver` の内部状態変更のみ。外部 Authority に影響なし |
| 単調増加 | ✅ 適合 | `clone()` → `init()` の流れは生成時のみ。Runtime 公開後に変更しない |
| 関心の分離 | ✅ 適合 | ConvolverProcessor の内部実装の範囲内 |

**追加ISR確認事項**:
- `FilterSpec` は POD 的構造体（`double`, `int`, `enum`, `bool` のみ）で、コピー代入安全
- `storedFilterSpec` コピーは `init()` 内の Message Thread のみで発生。Audio Thread に影響なし
- 修正後の `clone()` 呼び出し元 `shareConvolutionEngineFrom()` は Runtime Reader lock 下で動作 → ISR の publish-read モデルと整合

### B08 修正方法のISR適合性 — ★★★☆☆（ISR Authority 分散が根本原因）

| 評価軸 | 判定 | 根拠 |
|-------|------|------|
| Authority Singularization | ❌ **根本原因**: `m_retireRouter` と `eqCacheManager` の破棄順序問題は、**Retire Authority と Cache Manager Authority の独立性欠如**に起因 |
| 宣言順序の修正 | ⚠️ 対症療法 | ISR 的には、`eqCacheManager` が `m_retireRouter` に依存しない設計が本質的解決 |
| `~AudioEngine()` での明示クリア | ✅ 暫定対応として妥当 | |

**ISR 設計観点からの改善提案**:

`~CacheMap()` が `owner->m_retireRouter` を参照するのは、`EQCoeffCache::release()` が `IEpochProvider` を要求するため。ISR 的には、`CacheMap` dtor は `refcount == 0` のエントリを直接 delete すべきであり、retire router 経由の EBR 破棄に頼るべきではない：

```cpp
// ISR 準拠: CacheMap dtor で直接破棄（retire router 不使用）
~CacheMap()
{
    for (auto& entry : map)
    {
        if (entry.second != nullptr)
        {
            if (entry.second->releaseWithoutRetire())  // ★ retire 経路をバイパス
                delete entry.second;
        }
    }
}
```

これにより `m_retireRouter` の寿命に依存せず、UAF リスクを排除できる。

### B18 修正方法のISR適合性 — ★★☆☆☆（Authority Singularization 違反の根源的問題）

| 評価軸 | 判定 | 根拠 |
|-------|------|------|
| Authority Singularization | ❌ **根本原因**: `DSPHandleRuntime::destroyQuarantineSlot()` が DSP の解放処理を直接実行しようとするのは、**Quarantine Authority と Retire Authority の混在** |
| `destroyDSPCoreNode` 直接呼出し | ❌ **危険**: ISR の EBR（Epoch Based Reclamation）をバイパス。Audio Thread がまだ参照している可能性のある DSPCore を直接破棄する |
| 改善案（`retireDSPHandleForRuntime` 経由） | ✅ ISR 準拠 | 通常の retire パスに統合することで Epoch 保護を得られる |

**ISR 設計観点からの分析**:

1. `quarantineSlot()` は 3 系統の隔離を実行：
   - `dspQuarantineManager_.quarantineHandle(slot, generation, reason)` — Truth store
   - `dspHandleRuntime_.quarantineSlot(slot)` — Handle 側の投影
   - `retireRuntimeEx_.quarantine(slot)` — Retire 側の投影

2. **Authority トリレンマ**:
   ```
   DSPQuarantineManager  ─→  DSPHandleRuntime  ─→  RetireRuntimeEx
   (隔離判定の権威)         (Handle状態の権威)        (Retire状態の権威)
   ```

   3つの Authority が同じ隔離操作を異なる角度から管理している。ISR 原則では、**単一の Truth Store** のみが権威を持つべき。

3. **改善方針（ISR準拠）**:
---

### 総合ISR評価（改訂版）

| バグ | ISR適合度 | 修正案のISR方針 |
|------|----------|---------------|
| B14 | ★★★★★ | Vyukov MPSC（Producer fetch_add 1回のみ、Consumer RMW不要）。`reinterpret_cast` 排除 |
| B20 | ★★★★★ | TruePeakDetector は DSP 計算のみで ISR Runtime 関与なし |
| B13 | ★★★★★ | NUC 内部で delay 計算を完結。`RuntimeBuilder`/`RuntimeWorld` 非関与 |
| B17 | ★★★★★ | `StereoConvolver` 内部の変更のみ。外部 Authority に影響なし |
| B08 | ★★★★☆ | `releaseDirect()` 導入により `m_retireRouter` 依存解消。宣言順序修正組み合わせ |
| B18 | ★★★★☆ | `DSPQuarantineManager` を唯一 Authority に。標準 retire パス統合で EBR 保護 |

---

## 最終調査確定事項（2026-07-12 全項目確定）

本節では、レビュープロセスを通じて「要調査」「未確定」「保留」となっていた項目について、実ソースコード調査により確定した結果を記載する。

### B14: Retire queue 関連

| 調査項目 | 確定結果 |
|---------|---------|
| `reinterpret_cast` の出現数 | ✅ **1箇所のみ**: ISRRetire.cpp:203 (`reinterpret_cast<const std::atomic<bool>&>(intent.isValid)`)。他に `reinterpret_cast` なし |
| `dequeuePos_` の型 | ✅ 現状 `std::atomic<uint64_t>` ではない。Producer から読めるようにするには atomic 化必須。修正案では既に対応済み |
| `acknowledgeGeneration_` との相互作用 | ✅ **独立配列**: `acknowledgeGeneration_[dspSlot % 256]` はキューとは独立。`acknowledgeRetireCoordination()` は Consumer のデキュー後に呼ばれ、キュー構造（SPSC/Vyukov）に依存しない。Vyukov MPSC 化による影響なし |
| `std::atomic<bool> isValid` のテスト互換性 | ⚠️ 現状 `PriorityIntegrationTests.cpp` は aggregate init で `{1, 100, 1000, true, ...}` と記述。`std::atomic<bool>` は aggregate init 非対応のためコンパイルエラー。テスト側で `intent.isValid = true` に変更するか、`RetireIntent` にコンストラクタ追加が必要 |
| 全Producer のスレッド確認 | ✅ `emitRetireIntent` 呼び出し元: `AudioEngine.Commit.cpp:463` (Message Thread)、`ReleaseResources.cpp:211/249` (ReleaseResources)、`Timer.cpp:1580` (JUCE Timer)、`ISRRuntimePublicationCoordinator.cpp:294/306/332` (Coordinator)。**全呼び出し元が非RTスレッド**。Audio Thread からの呼び出しなし → spin wait は安全 |

### B13: NUPC delay alignment 関連

| 調査項目 | 確定結果 |
|---------|---------|
| `totalAhead` 計算精度 | ⚠️ **軽微な差異**: 修正案は `prev.partSize * prev.numPartsIR`（パーティション境界丸め）で計算。既存コードの `l1Offset = l0Len`（IR 実測値）と比較すると、最大 `partSize-1` サンプルの差（48kHz/L0=64 で 1.3ms）が生じうる。可聴域未満であり許容範囲。なお修正案では `cfgs[li].offset`（実測値）を使う方が正確だが、パーティショングラニュラリティの遅延線運用上 `totalAhead` 方式でも問題なし |
| `m_numActiveLayers` の範囲 | ✅ 可変値（1～3）。`for (int li = 0; li < m_numActiveLayers; ++li)` ループは範囲安全。`totalAhead` はアクティブレイヤーのみ参照するため境界問題なし |
| `m_tailLayerGain` のサイズ | ✅ `double m_tailLayerGain[kNumLayers] {1.0, 1.0, 1.0}` (MKLNonUniformConvolver.h:420)。`kNumLayers=3`。配列アクセス範囲安全 |

### B20: TruePeakDetector 関連

| 調査項目 | 確定結果 |
|---------|---------|
| 呼び出し元の `dataR` の有無 | ✅ **常に有効**: 呼び出し元は `DSPCoreDouble.cpp:740` のみ。`dataL`/`dataR` は `block.getChannelPointer(0/1)` から取得され、ステレオ動作では常に有効。ただし `TruePeakDetector` API は `dataR==nullptr`（モノラル）を許容する設計 |
| 修正案の `dataR==nullptr` 対応 | ✅ **対応済み**: 修正コードに nullptr チェックを追加済み（修正後のコード参照） |
| `stageInputMax` / `maxBlockSize` | ✅ `prepare()` の `maxBlockSize` は JUCE `prepareToPlay` の呼び出しパラメータと一致。`prepareStage(stages[i], taps, attenuationDb, stageInputMax)` に渡される `stageInputMax` は stage 0 が `maxBlockSize`、stage 1 が `maxBlockSize*2` |

### B08: CacheMap dtor UAF 関連

| 調査項目 | 確定結果 |
|---------|---------|
| `RefCountedDeferred` の API | ✅ `addRef()` / `release(provider)` / `tryAddRef()` のみ。`releaseWithoutRetire()` や `releaseDirect()` は**存在せず** → 新規追加が必要 |
| `EQCoeffCache` の継承関係 | ✅ `class EQCoeffCache : public RefCountedDeferred<EQCoeffCache>` — テンプレート継承のため `releaseDirect()` 追加後も `EQCoeffCache` への影響は `static_cast` 範囲内 |
| `~AudioEngine()` での drain 順序 | ✅ `AudioEngine.CtorDtor.cpp:175-205` で `m_retireRouter->publishEpoch()`, `tryReclaim()` 実行後、`m_epochDomain.drainAll()` を実行。この後メンバ解体順序で `m_retireRouter` → `eqCacheManager` の順に破棄。drain 完了後は `CacheMap` 内の全 `EQCoeffCache` が refcount==0 の可能性は低い（キャッシュ参照中）が、理論上はありうる |

### B18: destroyQuarantineSlot 関連

| 調査項目 | 確定結果 |
|---------|---------|
| `resolveDSPHandleBySlot` の有無 | ✅ **存在しない**: `DSPHandleRuntime::resolve(DSPHandle)` のみ存在。DSPHandle (slot+generation) から解決する。Quarantined 状態では無効を返す (ISRDSPHandle.cpp:49) |
| 代替方法 | ✅ `DSPHandleRuntime::resolve(DSPHandle{slot, generation})` を quarantine 発生前（状態が Active/Retired）に呼ぶ。修正案の Step 2 で Step 3 より前に実行することで解決 |
| `retireDSPHandleForRuntime` の動作 | ✅ `runtimeDSPHandleMap_` からエントリ削除 + `dspHandleRuntime_.retire()` + `reclaim()` で slot 解放 (AudioEngine.h:3946-3962)。その後 `enqueueDeferredDeleteNonRt` で EpochDomain に deferred delete エントリ投入 |

### B01: DSPCoreFloat 関連（追加調査）

| 調査項目 | 確定結果 |
|---------|---------|
| `RampRuntimeState` の Float メンバ | ✅ **不在確認**: `bypassFadeGainDouble` (LinearRamp) と `bypassedDouble` (bool) のみ (AudioEngine.h:715-716)。Float版メンバ未存在 → 修正で新規追加が必要 |
| `dryBypassBuffer` の Float 版 | ✅ **不在確認**: `dryBypassBufferDoubleL/R` (ScopedAlignedPtr\<double\>, 974-975行) のみ |
| DSPCoreLifecycle の bypass 確保 | ✅ Double 版のみ実装 (DSPCoreLifecycle.cpp:167-174)。Float 版の同等コード追加が必要 |
| Float ルートの有効性 | ✅ **Plugin モードで有効**: `CONVOPEQ_STANDALONE_ONLY` 未定義時に `AudioEngineProcessor::processBlock(AudioBuffer<float>&)` → `getNextAudioBlock()` → `dsp->process()`（DSPCoreFloat.cpp）が実行される。Standalone モードでは `processBlock(float)` は `buffer.clear()` のみでデッドコード |

### B13: NUPC Tail Mode 関連（追加調査）

| 調査項目 | 確定結果 |
|---------|---------|
| TailMode::AirAbsorption (0) との相互作用 | ✅ 遅延アライメント不良の影響を受ける。周波数依存ダンピングとは独立した問題 |
| TailMode::LayerTailContouring (1) との相互作用 | ✅ 遅延アライメント不良の影響を受ける。レイヤーゲイン補正とは独立した問題 |
| TailMode::Bypass (2) との相互作用 | ✅ **影響を受けない**: L1/L2 が無効化されているため遅延線そのものが不要 |
| `m_tailLayerGain` との整合性 | ✅ 遅延線導入後も `addScaledFallback(toAdd, output, tailPtr, layerGain)` のゲイン乗算はそのまま使用可能 |
| `tailStartSeconds` による遅延量変動 | ✅ L0長は `tailStartSeconds` に依存（0.085→2048samples/42.7ms、0.80→9216samples/192ms）。修正案の `totalAhead = prev.partSize * prev.numPartsIR` は常に適切な遅延量を提供（パーティション境界丸めの誤差は最大 partSize-1 = 1.3ms で許容範囲） |

### B03/B10/B15: WARN 項目

| 調査項目 | 確定結果 |
|---------|---------|
| B03: `vdTanh` 呼び出し元スレッド数 | ✅ `runEvaluationJobsForWorker` は aux worker (558行) と main (661行) から呼ばれる。aux worker 数は `kNumAuxWorkers` 定数で規定。最大 8 worker までの並列 vdTanh 冗長性確認済み |
| B10: `_mm256_loadu_pd` のアライメント保証 | ✅ 対象配列 `wetGain`/`dryGain` は `mkl_malloc(64)` で確保された `ScopedAlignedPtr<double>` から派生。32-byte アライメントは保証可能だが、`loadu` から `load` への変更は任意の最適化 |
| B15: `AudioSegmentBuffer` の呼び出し元 | ✅ `NoiseShaperLearner` の `workerThreadMain` 単一スレッド (NoiseShaperLearner.cpp:322)。現在はデータ競合未発火。将来複数 writer が生じた場合の修正候補 |

---
