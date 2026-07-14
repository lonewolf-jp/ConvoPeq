# ConvoPeq バグ修正 実装仕様書

> 作成日: 2026-07-13
> 対象コードベース: ConvoPeq (C++20, JUCE 8.0.12, MKL, Intel oneAPI, ISR Bridge Runtime)
> 元文書: `doc/work69/bug_final_report.md` — 本ファイルは同文書の実装仕様部分を抽出したもの。
> 付属情報（解析詳細・検証結果・テスト計画・ISR評価等）は Appendix を参照。

---

## 0. 実装優先順序

| 順序 | ID | 件名 | 優先度 | ファイル | 工数目安 | 依存 |
|------|----|------|--------|---------|---------|------|
| 1 | B20 | TruePeakDetector Rチャンネル欠落 | 🔴 P0 | TruePeakDetector.cpp | 2-4h | なし |
| 2 | B17 | StereoConvolver::clone() FilterSpec欠落 | 🟡 P1 | ConvolverProcessor.h | 0.5-1h | なし |
| 3 | B08 | CacheMap dtor UAF (WARN) | 🟡 P1 | RefCountedDeferred.h / AudioEngine.h | 2-4h | なし |
| 4 | B18 | destroyQuarantineSlot リーク (WARN) | 🟡 P1 | ISRDSPHandle.cpp / AudioEngine.h | 4-8h | なし |
| 5 | B01 | DSPCoreFloat バイパスブレンド欠落 | 🟡 P1 | DSPCoreFloat.cpp / AudioEngine.h / DSPCoreLifecycle.cpp | 4-8h | なし |
| 6 | B14 | Retire Queue MPSC データレース | 🔴 P0 | ISRRetire.cpp / ISRRetire.h | 8-16h | なし |
| 7 | B03 | NoiseShaper vdTanh 冗長 (WARN) | 🟢 P2 | NoiseShaperLearner.cpp | 2-4h | なし |
| 8 | B13 | NUPC delay alignment 欠落 | 🟡 P1 | MKLNonUniformConvolver.cpp / .h | **要実測検証** | **事前実測必須** (MT-NUPC-01〜03, Partition Boundary) |

**実装順序の独立性**: B20/B14/B01/B17/B08/B03 はすべて独立したファイルに対する修正であり、実装順序に依存関係はない。
B13 のみ事前の実測検証 (MT-NUPC-01〜03 + Partition Boundary) が必要。
B18 は B14 とはファイルが分離しているため同時進行可能。

---

## 1. B20: TruePeakDetector Rチャンネル計測欠落 🔴 P0

### 対象ファイル
- `src/TruePeakDetector.cpp`

### 修正内容
#### Step 1: `prepare()` のバッファ拡張 (25-30行)
```cpp
// 旧: const int upBufferSize = maxBlockSize * kOversamplingRatio;  // = maxBlockSize*4
// 新: レイアウト [Stage0L | Stage0R | Stage1L | Stage1R] = 2N+2N+4N+4N = 12N
const int upBufferSize = maxBlockSize * kOversamplingRatio * 3;  // = maxBlockSize*12
```

#### Step 2: `processBlock()` に R チャンネル補間追加 (64-91行)

バッファ領域のオフセットは名前付き定数で管理し、将来の保守性を確保する:

```cpp
double TruePeakDetector::processBlock(const double* dataL, const double* dataR, int numSamples) noexcept
{
    if (numSamples <= 0 || !upsampleBuffer) return 0.0;

    double* work = upsampleBuffer.get();
    const int up1Samples = numSamples * 2;
    const int up2Samples = numSamples * 4;

    // オフセット: work 領域のレイアウト
    //   [ Stage0 L | Stage0 R | Stage1 L | Stage1 R ]
    //  注: Stage0 は zero-offset、それ以外は up1Samples/up2Samples に依存する runtime 値
    constexpr int kStage0LOffset = 0;
    const int   kStage0ROffset = up1Samples;     // runtime 値のため const
    const int   kStage1LOffset = up1Samples * 2;
    const int   kStage1ROffset = up1Samples * 2 + up2Samples;

    // Stage 0: 1x -> 2x (L)
    interpolateStage(stages[0], dataL, numSamples, work + kStage0LOffset, 0);
    // Stage 0: 1x -> 2x (R)
    if (dataR != nullptr)
        interpolateStage(stages[0], dataR, numSamples, work + kStage0ROffset, 1);
    else
        interpolateStage(stages[0], dataL, numSamples, work + kStage0ROffset, 1);

    // Stage 1: 2x -> 4x (L + R)
    interpolateStage(stages[1], work + kStage0LOffset, up1Samples, work + kStage1LOffset, 0);  // L
    interpolateStage(stages[1], work + kStage0ROffset, up1Samples, work + kStage1ROffset, 1);  // R

    // Peak scan: L/R 別領域で独立実行
    double peakL = scanPeak(work + kStage1LOffset, up2Samples);
    double peakR = scanPeak(work + kStage1ROffset, up2Samples);
    double peak = std::max(peakL, peakR);

    if (peak > peakHold) peakHold = peak;
    else peakHold *= 0.999;
    return peakHold;
}
```

#### Step 3: `scanPeak` ヘルパー関数追加
```cpp
// ★ scanPeak: |buf[i]| の最大値を求めるヘルパー (static linkage で最適化期待)
//    -0.0 マスクは static constexpr 変数で毎回の _mm256_set1_pd 生成を回避
#if defined(__AVX2__)
#endif

namespace {  // ★ C++20: anonymous namespace が推奨

double scanPeak(const double* buf, int n) noexcept
{
    double peak = 0.0;
#if defined(__AVX2__)
    // ★ RT 初回初期化 (guard variable) を避けるため static local は使用しない。
    //    ローカル変数 _mm256_set1_pd は 1 命令 (vsetpd) で実質ゼロコスト。
    const __m256d signMask = _mm256_set1_pd(-0.0);
    __m256d vPeak = _mm256_setzero_pd();
    int i = 0;
    for (; i <= n - 4; i += 4) {
        __m256d v = _mm256_andnot_pd(signMask, _mm256_loadu_pd(buf + i));
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

}  // anonymous namespace
```

### 変更点
- `prepare()`: upsampleBuffer サイズを下記の constexpr 構成要素から導出:
  ```cpp
  constexpr int kStage0Channels = 2;        // L + R
  constexpr int kStage1Channels = 2;        // L + R
  constexpr int kUpsampleFactor1 = 2;        // Stage 0: 1x → 2x
  constexpr int kUpsampleFactor2 = 4;        // Stage 1: 2x → 4x
  constexpr int kWorkBufferMultiplier =
      kStage0Channels * kUpsampleFactor1 +   // Stage0: 2 * 2N = 4N
      kStage1Channels * kUpsampleFactor2;   // Stage1: 2 * 4N = 8N
  // total = 4N + 8N = 12N
  const int upBufferSize = maxBlockSize * kWorkBufferMultiplier;
  ```
  マジックナンバー `×12` を排除し、構成要素から導出することで将来の
  オーバーサンプリング倍率変更やチャンネル数変更時に追従可能。
- `processBlock()`: Stage 1 で R チャンネル (work + up1Samples) も補間。L/R 別領域で個別 PK 検出
- `scanPeak()`: 新規 static ヘルパー関数。`_mm256_loadu_pd` で unaligned 対応

### BS.1770 適合性
修正後は L/R 各チャンネルに 4x オーバーサンプリングが適用され、ITU-R BS.1770-4/5 準拠。

### 改善提案 (将来拡張)
- **Mono 専用経路**: `dataR == nullptr` 時の現状 (L を R にコピーしてステレオ処理) は正しいが、Mono 時は `Stage0L→Stage1L→scanPeak` のみの簡略経路にすることで補間処理を半減できる。優先度は低い。
- **`WorkLayout` 構造体**: オフセット計算 (`kStage0LOffset`, `kStage0ROffset` 等) を構造体にまとめることで保守性が向上する。設計改善レベル。
- **`horizontalMax` ヘルパー関数**: `_mm256_max_pd` 後の horizontal max 処理 (`_mm256_store_pd → 4回比較`) を専用関数化する。AVX-512 移行時は `_mm256_reduce_max_pd` に置き換え可能。
- **SIMD 数値一致性テスト**: AVX2 版と Scalar 版の `scanPeak()` 出力が 1 ULP (unit in the last place) 以内で一致することを検証するテストを追加する。
  ```cpp
  double peakAvx = scanPeak(buf, n);
  double peakScalar = scanPeakScalar(buf, n);
  REQUIRE(std::abs(peakAvx - peakScalar) <= std::numeric_limits<double>::epsilon());
  ```

---

## 2. B14: Retire Queue SPSC -> MPSC データレース 🔴 P0

### 対象ファイル
- `src/audioengine/ISRRetire.h`
- `src/audioengine/ISRRetire.cpp`
- `src/audioengine/ISRRetireOverflowRing.h`

### 推奨修正: Vyukov MPSC Queue + RetireRuntimeEx による優先度ソート (代替案C)

Queue の責務は enqueue/dequeue (non-FIFO) に限定する。priority に基づく
retire 順序のスケジューリングは Queue の上位レイヤ (RetireRuntimeEx / Commit) で行う。
これにより Queue と Policy の責務が分離され、ISR の Authority Singularization に適合する。

#### 2.1 `RetireIntent` 構造体 + `ScheduledRetireIntent` (ISRRetire.h)

Queue 層は Policy 情報 (arrivalSeq) を知らない。arrivalSeq は Scheduler 専用の
ラッパー型で保持する:

```cpp
struct RetireIntent {
    uint32_t dspSlot;
    uint64_t generation;
    uint64_t retireEpoch;
    RetirePriority priority{RetirePriority::Normal};
    // ★ isValid は廃止: slot.sequence が slot 状態の唯一の Authority。
    //    無効な intent (tombstone) は dspSlot == UINT32_MAX で識別する。
    //    これにより atomic<bool> 問題が完全に回避され、RetireIntent は
    //    trivially copyable かつ 64 バイト以下を維持できる。
};

// ★ Scheduler 専用ラッパー: observationOrder は Queue の外でのみ使用
//    命名: 「dequeue で観測した順序」を意味する。
//    歴史的経緯から arrivalSeq という変数名も使われるが、
//    新人保守者の誤解防止のため observationOrder を推奨する。
struct ScheduledRetireIntent {
    RetireIntent intent;
    uint64_t observationOrder{};  // ★ publishGeneration (dequeue 時に commit が採番、publish順の近似)
};
```

**採番方式**:
```cpp
// RetireRuntimeEx または Commit のループ
uint64_t localArrivalSeq = 0;
RetireIntent rawIntent;
std::vector<ScheduledRetireIntent> batch;
while (queue_.dequeueOne(rawIntent)) {
    batch.push_back({rawIntent, localArrivalSeq++});
}
// arrivalSeq は dequeue observation order (= dequeueOne 取得順) を反映。
//    priority 内の近似順序付けに使用するが、FIFO (enqueue 順) ではない。
//    Vyukov MPSC では Producer A (ticket=N) より Producer B (ticket=N+1) が
//    先に publish 完了することがあり、dequeue 順 ≠ ticket 順 ≠ enqueue 順。
//    arrivalSeq は「誰が先に publish を完了したか」の近似であり、
//    グローバルな enqueue 順序ではない。
```

**重要**: arrivalSeq は「dequeue observation order (= dequeueOne で取得した順)」を反映する。
Producer の publish 完了と Consumer の dequeue 取得の間には時間差があるため、
厳密には「Consumer が観測した完了順」であり、絶対的な publish 完了順ではない。
Vyukov MPSC では publish 完了順 ≠ ticket 順 (= enqueue 順) であり、
arrivalSeq に完全な FIFO 保証はない。

`reinterpret_cast<std::atomic<bool>>` 問題は `escalateAllRetires()` 内の該当箇所を個別に修正する (Appendix A.1 参照)。

#### 2.2 `RetireRuntime` クラスの再設計 (ISRRetire.h)
```cpp
struct RetireSlot {
    RetireIntent payload;
    std::atomic<uint64_t> sequence{0};
};

class RetireRuntime {
    // ★ Queue protocol: enqueueTicket_ は Producer の slot 予約用 (Vyukov MPSC sequence)
    std::atomic<uint64_t> enqueueTicket_{0};
    // ★ globalArrivalSeq_ は削除: arrivalSeq は RetireScheduler が dequeue 時に local 採番する。
    //    Queue は arrivalSeq を知る必要がない。
    // ★ Consumer 管理: dequeuePos_ はメンバ変数 (atomic 不要、Consumer単一スレッドのみ操作)
    uint64_t dequeuePos_{0};
    RetireSlot slots_[RETIRE_INTENT_QUEUE_SIZE];
    std::mutex fallbackMutex_;
    RetireIntent fallbackQueue_[FALLBACK_QUEUE_CAPACITY];
    size_t fallbackHead_{0};  // ★ head/tail 管理 (dequeueFallback O(1))
    size_t fallbackCount_{0}; // fallbackQueue_ の有効要素数
    std::atomic<size_t> fallbackQueuePeak_{0}; // ★ 診断: fallback 使用量の最大値 (Queue サイズ不足の指標)
    // ★ 注意: fallbackQueue_ は head から取り出し、tail (head+count) に追加する循環バッファ。
    //    dequeueFallback() は O(N) memmove ではなく O(1) head 移動で動作。

    // ★ 64bit sequence wrap に関する注記:
    //    enqueueTicket_ / slot.sequence は uint64_t で無制限に増加する。
    //    Vyukov Queue の不変条件は slot.sequence == ticket の一致であり、
    //    unsigned overflow arithmetic により wrap 後も成立する。
    //    具体的には (producerTicket - consumedSeq) の差が SIZE 未満であれば
    //    一貫性が保たれる — 年数の見積もりは不要。
public:
    void emitRetireIntent(const RetireIntent& intent) noexcept;
    bool dequeueOne(RetireIntent& out) noexcept;
    bool dequeueFallback(RetireIntent& out) noexcept;
    void initQueue() noexcept;

    // ★ Queue Pressure 診断 (HealthMonitor 用):
    //    approxQueueDepth() は dequeuePos_ と enqueueTicket_ の差から
    //    おおよその Queue 占有数を推定する。
    //    厳密な値ではなく「満杯に近いか」の判定に使用。
    [[nodiscard]] uint64_t approxQueueDepth() const noexcept {
        const uint64_t enqueued = convo::consumeAtomic(enqueueTicket_, std::memory_order_acquire);
        // dequeuePos_ は Consumer 専有のため relaxed 読取で十分
        return (enqueued > dequeuePos_) ? (enqueued - dequeuePos_) : 0;
    }
};
```

#### 2.2b `initQueue()` — initialization procedure

```cpp
void RetireRuntime::initQueue() noexcept
{
    for (size_t i = 0; i < RETIRE_INTENT_QUEUE_SIZE; ++i) {
        // sequence initial value = i (matches enqueueTicket_ initial 0)
        convo::publishAtomic(slots_[i].sequence,
                             static_cast<uint64_t>(i),
                             std::memory_order_release);
    }
    // ★ initQueue は single-threaded (Builder のみ)。relaxed で十分。
    convo::publishAtomic(enqueueTicket_, 0, std::memory_order_relaxed);
    dequeuePos_ = 0;  // Consumer 専用メンバ変数 (atomic 不要)
}
```

#### 2.3 `emitRetireIntent` — Producer (任意非RTスレッド) (ISRRetire.cpp)

Pure Vyukov: Producer は `slot.sequence` のみを参照して状態判断する。
`dequeuePos_` は Consumer ローカル変数であり Producer からは見えない。

```cpp
void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    jassert(!convo::numeric_policy::isAudioThread());  // RT 安全ガード

    // ★ Step 1: MPSC Queue に slot を予約 — 通常経路
    const uint64_t ticket = convo::fetchAddAtomic(enqueueTicket_, 1, std::memory_order_acq_rel);
    const size_t idx = ticket % RETIRE_INTENT_QUEUE_SIZE;
    // ★ RETIRE_INTENT_QUEUE_SIZE は任意の正の整数で動作 (idx = ticket % SIZE)。
    //    Power-of-two 制約はない (bitmask `& (SIZE-1)` は使用しない)。
    //    ただし効率上の理由から power-of-two を推奨 (コンパイラが % を bitmask に最適化可能)。

    // ★ Step 1b: arrivalSeq は Queue では採番しない (RetireScheduler が dequeue 時に local 採番)
    RetireIntent localIntent = intent;

    // ★ Step 2: bounded spin — Consumer が slot を解放するまで待機
    static constexpr int kMaxProducerSpin = 64;
    for (int spin = 0;; ++spin) {
        uint64_t slotSeq = convo::consumeAtomic(
            slots_[idx].sequence, std::memory_order_acquire);
        if (slotSeq == ticket) break;  // slot 獲得

        if (spin >= kMaxProducerSpin) {
            // ★ bounded spin 失敗 → fallback (例外経路)
            //    slot へ tombstone を書き込んで sequence を進め、hole を防止する。
            //    ★ この publish は必須: 省略すると Consumer が dequeuePos=N で停止する。
            slots_[idx].payload = RetireIntent{};  // dspSlot==0 の tombstone
            // ★ tombstone 識別: dspSlot==UINT32_MAX を無効値として使用
            slots_[idx].payload.dspSlot = UINT32_MAX;
            convo::publishAtomic(slots_[idx].sequence, ticket + 1, std::memory_order_release);

            // ★ 実質的な intent は fallback queue に保存
            std::lock_guard<std::mutex> lock(fallbackMutex_);
            if (fallbackCount_ < FALLBACK_QUEUE_CAPACITY) {
                const size_t tail = (fallbackHead_ + fallbackCount_) % FALLBACK_QUEUE_CAPACITY;
                fallbackQueue_[tail] = localIntent;
                ++fallbackCount_;
                if (fallbackCount_ > convo::consumeAtomic(fallbackQueuePeak_, std::memory_order_relaxed))
                    convo::publishAtomic(fallbackQueuePeak_, fallbackCount_, std::memory_order_relaxed);
            } else {
                convo::fetchAddAtomic(droppedIntentCount_, 1, std::memory_order_relaxed);
            }
            return;
        }
        _mm_pause();
    }
    // ★ Step 3: payload 書き込み (通常の MPSC Queue publish)
    slots_[idx].payload = localIntent;
    //    [順序保証] payload 書き込み (通常 store) → sequence publish (release) の順序で実行。
    //    Consumer は sequence を acquire で読み取るため、payload の可視性が保証される。
    // ★ Step 4: release — Consumer に読み取り可能を通知
    convo::publishAtomic(slots_[idx].sequence, ticket + 1, std::memory_order_release);
}
```

#### 2.4 `dequeueOne` — Consumer (Commit単一スレッド)

Queue 層の API は `dequeueOne()` 単一取出しのみ。batch 管理・vector 生成は
上位レイヤ (RetireRuntimeEx) の責務。これにより Queue は「slot 管理のみ」に
専念できる。

Consumer は `dequeuePos_` メンバ変数をローカルコピーで操作し、最後に書き戻す。
Producer との同期は `slot.sequence` のみ — 純粋 Vyukov。
`thread_local` は使用しない (RetireRuntime インスタンス毎に異なる dequeuePos が必要)。

```cpp
bool RetireRuntime::dequeueOne(RetireIntent& out) noexcept
{
    for (;;) {  // ★ while ループ: 再帰ではなくループで tombstone をスキップ
        // ★ Consumer 専用メンバ変数 (atomic 不要、単一スレッド所有)
        const size_t idx = dequeuePos_ % RETIRE_INTENT_QUEUE_SIZE;
        const uint64_t slotSeq = convo::consumeAtomic(
            slots_[idx].sequence, std::memory_order_acquire);
        if (slotSeq != dequeuePos_ + 1) return false;  // 未 ready

        out = slots_[idx].payload;
        // ★ tombstone check: dspSlot == UINT32_MAX で fallback 経由の無効 intent を識別
        //    (isValid は廃止 — slot.sequence のみが slot 状態の Authority)
        if (out.dspSlot == UINT32_MAX) {
            convo::publishAtomic(slots_[idx].sequence,
                dequeuePos_ + RETIRE_INTENT_QUEUE_SIZE,
                std::memory_order_release);
            ++dequeuePos_;
            continue;  // 次へ (ループ)
        }
        // ★ slot 解放: Producer が slotSeq == dequeuePos + SIZE で再利用可能
        //    Vyukov Queue の cycle 番号エンコーディング:
        //    sequence = cycle * SIZE + slotIndex とみなす。Producer は ticket (=cycle*SIZE+idx)
        //    で待機し、Consumer は dequeuePos (=cycle*SIZE+idx) を解放時に +SIZE することで
        //    次の cycle の Producer に slot が空いたことを通知する。
        convo::publishAtomic(slots_[idx].sequence,
            dequeuePos_ + RETIRE_INTENT_QUEUE_SIZE,
            std::memory_order_release);
        ++dequeuePos_;
        return true;
    }
}
```

**Fallback queue drain** は上位レイヤ (RetireRuntimeEx) が Queue API のみ使用して行う:
```cpp
// RetireRuntimeEx の責務 (Queue 内部状態に直接アクセスしない)
std::vector<ScheduledRetireIntent> batch;
uint64_t localArrivalSeq = 0;
RetireIntent raw;
while (queue_.dequeueOne(raw))
    batch.push_back({raw, localArrivalSeq++});
while (queue_.dequeueFallback(raw))
    batch.push_back({raw, localArrivalSeq++});
// RetireBatch で priority bucket 振分け (3-way, O(N))
```

`dequeueFallback()` は RetireRuntime 内部で mutex 保護された fallback 配列から
1件ずつ head 移動で取り出す (O(1))。`fallbackQueue_` は head/tail 管理の循環バッファ:
```cpp
bool RetireRuntime::dequeueFallback(RetireIntent& out) noexcept
{
    std::lock_guard<std::mutex> lock(fallbackMutex_);
    if (fallbackCount_ == 0) return false;
    out = fallbackQueue_[fallbackHead_];
    fallbackHead_ = (fallbackHead_ + 1) % FALLBACK_QUEUE_CAPACITY;
    --fallbackCount_;
    return true;
}
```
public API は `dequeueOne` / `dequeueFallback` のみであり、
`fallbackMutex_` や `fallbackQueue_` は外部に露出しない (O(1) 保証)。

**3-way bucket dispatch (stable_sort 不要) using ScheduledRetireIntent**:
priority 3種類 (High/Normal/Low) しかないため、std::stable_sort ではなく
`ScheduledRetireIntent` を 3本の vector に振り分けるだけで O(N) で整列できる。
arrivalSeq はこのラッパー型で最後まで保持される:
```cpp
struct RetireBatch {
    std::vector<ScheduledRetireIntent> high;
    std::vector<ScheduledRetireIntent> normal;
    std::vector<ScheduledRetireIntent> low;

    void append(ScheduledRetireIntent item) noexcept {
        switch (item.intent.priority) {
            case RetirePriority::High:   high.push_back(std::move(item)); break;
            case RetirePriority::Normal: normal.push_back(std::move(item)); break;
            case RetirePriority::Low:    low.push_back(std::move(item)); break;
        }
    }
};

// Commit での実行順序: High → Normal → Low (各 bucket 内は arrivalSeq 順 = publish completion order)
void commitBatch(const RetireBatch& batch) {
    for (auto& item : batch.high)   commit(item.intent);
    for (auto& item : batch.normal) commit(item.intent);
    for (auto& item : batch.low)    commit(item.intent);
}
```

**fallback starvation 防止: fair scheduling (Policy 層)**:
dequeueOne (main queue) と dequeueFallback の間で優先順位が常に main > fallback だと、
大量 Producer が継続して enqueue する状況で fallback が永遠に drain されない可能性がある。
これを防ぐため、RetireRuntimeEx の batch 収集ループに fair scheduling を導入する。
比率は Policy 層 (Scheduler または RetireRuntimeEx) で設定し、Queue は関知しない:
```cpp
// RetireRuntimeEx (Policy 層): fair scheduling — main:fallback = 8:1
// ★ この比率は Policy (RetireRuntimeEx または Scheduler) が決定する。
//    Queue は dequeueOne / dequeueFallback の API を提供するのみ。
std::vector<ScheduledRetireIntent> batch;
batch.reserve(128);  // ★ ISR 安定動作のため事前確保
constexpr size_t kMainToFallbackRatio = 8;  // main 8件 → fallback 1件 → ...
RetireIntent raw;
while (true) {
    bool progressed = false;
    for (size_t i = 0; i < kMainToFallbackRatio; ++i) {
        if (!queue_.dequeueOne(raw)) break;
        batch.push_back({raw, localArrivalSeq++});
        progressed = true;
    }
    if (queue_.dequeueFallback(raw)) {
        batch.push_back({raw, localArrivalSeq++});
        progressed = true;
    }
    if (!progressed) break;
}
```

**ISRRetireOverflowRing との関係**:
既存の `ISRRetireOverflowRing` (`ISRRetireOverflowRing.h`) は **SPSC** (Single Producer =
Audio Callback, Single Consumer = Coordinator) 専用のロックフリーリングバッファである。
B14 の fallback は **MPSC** (複数 Producer = Timer/ReleaseResources/Coordinator) が必要なため
既存 OverflowRing を直接流用できない。設計選択肢:
- **A) mutex fallback 維持** (現行): オーバーフローは例外経路。Queue サイズを十分に大きく
  設計することで fallback への到達自体を回避する。本設計書の採用案。
- **B) 新規 MPSC OverflowRing**: 既存 SPSC Ring を参考に MPSC 版を新設。
  将来の拡張要件次第で検討。
- **C) RETIRE_INTENT_QUEUE_SIZE 拡大**: 最もシンプルな対策。2倍〜4倍に拡大することで
  fallback 発生確率を実用上ゼロにできる。
```

### 責務の明確化 (4層アーキテクチャ)

```
Queue (dequeueOne)
  ↓ 単一取出し
RetireRuntimeEx (batch 収集 + fallback drain)
  ↓ std::vector<ScheduledRetireIntent>
RetireScheduler (priority bucket 振分け, O(N))
  ↓ RetireBatch
Commit (retire 実行)
```

各層の責務:
1. **Queue** (`RetireRuntime::dequeueOne`): slot 管理のみ。単一の `RetireIntent` を取得。
   vector 生成・batch 管理は行わない。
2. **RetireRuntimeEx** (上位レイヤ): `dequeueOne()` / `dequeueFallback()` をループで呼び
   `ScheduledRetireIntent` を生成し batch 収集。
3. **RetireScheduler**: `RetireBatch` で priority bucket に振分け (3-way, O(N))。
4. **Commit**: priority 順 (High → Normal → Low) に retire 実行。

```cpp
// RetireScheduler: 3-way bucket dispatch (priority 3種類のみ → O(N)、sort 不要)
// arrivalSeq は publish completion order に local 採番済み。commit 時に High → Normal → Low の順に処理。
struct RetireBatch {
    // ★ Commit Thread は非 RT のため vector でも動作するが、ISR 安定動作のため
    //    reserve() または固定 capacity の small_vector を推奨。
    std::vector<ScheduledRetireIntent> high;
    std::vector<ScheduledRetireIntent> normal;
    std::vector<ScheduledRetireIntent> low;

    void append(ScheduledRetireIntent item) noexcept {
        switch (item.intent.priority) {
            case RetirePriority::High:   high.push_back(std::move(item)); break;
            case RetirePriority::Normal: normal.push_back(std::move(item)); break;
            case RetirePriority::Low:    low.push_back(std::move(item)); break;
        }
    }
};
```

**droppedIntentCount_ の責務**:
dropped intent カウンタは Queue が保持し、`emitRetireIntent()` 内でインクリメントする。
Queue の責務は「drop 発生の記録 (カウンタ)」のみ。
`RuntimeHealthMonitor` は定周期の polling (`AudioEngine.Retire.cpp:124`) で
`droppedIntentCount()` を読み取り、drop delta と overflow duration/frequency の
二重判定で異常検知を行う。これにより責務が明確になる:
- **Queue**: カウンタを保持・更新 (write)
- **HealthMonitor**: カウンタを監視 (read only)

**`RuntimeHealthMonitor` の EVENT コードは dead code**:
`RuntimeHealthMonitor.h:44-64` に `EVENT_RETIRE_STALL=1001` 他16のイベントコードが
定義されているが、コードベース全体で参照ゼロ (push-type emit は未実装、polling のみ動作)。
ISR の HealthMonitor は Polling だけで十分であり、これらの未使用コードは
**実装時または別途クリーンアップタスクとして削除すること**。

**fallback は例外経路**:
mutex 保護の fallback は Queue 満杯時の例外処理であり、通常運用では発生しない。
大量 Producer が同時 fallback した場合に mutex がボトルネックになる可能性があるが、
RETIRE_INTENT_QUEUE_SIZE を十分に大きく設計することで fallback 発生自体を回避する。
fallback が常発する場合は Queue サイズ不足のシグナルとみなす。

**fallbackQueuePeak_ による診断**:
`fallbackQueuePeak_` は fallback 使用量の最大値を記録する。定常的に 0 以外の値が
記録される場合、RETIRE_INTENT_QUEUE_SIZE の拡大を検討する。HealthMonitor の polling
周期で読み取り可能。

**ISRRetireOverflowRing との関係**:
既存 `ISRRetireOverflowRing` は SPSC (Audio Callback → Coordinator) 専用。
B14 fallback は MPSC が必要なため、現行の mutex fallback を維持する。
将来 MPSC OverflowRing が必要になった場合に改めて設計検討する。

### Vyukov MPSC Memory Ordering

本 Queue で使用するメモリオーダリングを以下に明記する:

| 操作 | コード | Ordering | 理由 |
|------|--------|----------|------|
| Producer: ticket 取得 | `fetchAddAtomic(enqueueTicket_, 1, acq_rel)` | `acq_rel` | ticket 発行と slot 解放の間の全順序を保証 (同時に fetch_add する複数 Producer 間の一貫性) |
| Producer: slot 待機 | `consumeAtomic(slots_[idx].sequence, acquire)` | `acquire` | Consumer の `publishAtomic(..., release)` と対。slot 解放 (= sequence 更新) の可視性を保証 |
| Producer: slot publish | `publishAtomic(slots_[idx].sequence, ticket+1, release)` | `release` | payload 書き込み (通常 store) が Consumer に可視になる前に sequence が先に見えないことを保証 |
| Consumer: dequeue 確認 | `consumeAtomic(slots_[idx].sequence, acquire)` | `acquire` | Producer の `release` と対。payload 読み取り前に sequence 更新が可視であることを保証 |
| Consumer: slot 解放 | `publishAtomic(slots_[idx].sequence, dequeuePos_ + SIZE, release)` | `release` | 解放済み slot を次 Producer が正しく見えることを保証 |
| Producer: fallback 統計 | `fetchAddAtomic(droppedIntentCount_, 1, relaxed)` | `relaxed` | 診断カウンタのみ。正確な全順序不要 |
| Producer: initQueue | `publishAtomic(enqueueTicket_, 0, relaxed)` | `relaxed` | 初期化中は単一スレッドのみ |

**Vyukov MPSC 不変条件 (Invariant)**:

1. **Publish Visibility**: Producer は `slot.payload` への全書き込みを完了した**後に**
   `publishAtomic(slot.sequence, ticket+1, release)` を発行する。
   Consumer は `consumeAtomic(slot.sequence, acquire)` で sequence を確認した**後にのみ**
   `slot.payload` を読み取る。release/acquire の happens-before により、
   Consumer が読む payload は Producer が書いた完全な値である。

2. **Slot Ownership**: slot の所有権は sequence 番号で管理される。
   `sequence == ticket` : Producer が占有中。
   `sequence == ticket+1` : Consumer が読み取り可能 (= publish 完了)。
   `sequence == dequeuePos + SIZE` : Consumer が解放済み (= 次 Producer が再利用可能)。

3. **ABA Safety**: sequence は単調増加。wrap 後も `ticket % SIZE` と `dequeuePos % SIZE` の
   差が SIZE 未満である限り、slot の一貫性は unsigned overflow arithmetic により維持される。

4. **Happens-Before Chain**:
   ```
   Producer: payload store → sequence.release()
       ↓ happens-before
   Consumer: sequence.acquire() → payload load
       ↓ happens-before
   Consumer: sequence.release(+SIZE)
       ↓ happens-before
   Producer (次 cycle): sequence.acquire() (spin 待機)
   ```
   このチェインにより、全 Producer/Consumer 間の全部の操作に全順序が定義される。

### 代替案 (Producer集約・fetch_add簡易版)
代替案A (Producer集約) と B (fetch_add 簡易版) は Appendix A.1 を参照。本設計書の正式採用案は Vyukov MPSC Queue + RetireRuntimeEx による優先度ソート。

### RT 安全ガード (必須)
```cpp
// ISRRetire.cpp emitRetireIntent() 先頭
jassert(!convo::numeric_policy::isAudioThread());
```
Debug ビルドで Audio Thread からの誤呼び出しを検出。

---

## 3. B01: DSPCoreFloat バイパスブレンド機構欠落 🟡 P1

### 対象ファイル
- `src/audioengine/AudioEngine.h` (DSPCore / RampRuntimeState)
- `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`
- `src/audioengine/DSPCoreLifecycle.cpp`

### 修正内容 (3段階)

#### Step 1: AudioEngine.h — DSPCore に Float 用メンバ追加

`RampRuntimeState` (713行) は `DSPCore` (603行) 内の nested struct で Float/Double で共有される。
`bypassFadeGain` の型は Double 版が `convo::LinearRamp` (内部 double) であるため、
Float 版も同一クラス `convo::LinearRamp` を使用する (`LinearRamp<float>` ではなく、
Double 版と同じく `LinearRamp` の double ランプ — フェード時間が同一だからであり、型の不一致はない)。

**将来の設計改善**: 本修正では Float/Double で `RampRuntimeState` を共有するが、
将来の要件変更に備え `FloatRampState` / `DoubleRampState` への分離を検討する。
現状では DSPCore 単位のため実害はない。

**VST3 Float/Double 経路の実態**:
JUCE の `setDoublePrecisionProcessing()` により、ホストは Float または Double の
いずれか一方の経路のみを使用する（Session 中に両者が混在することはない）。
`AudioEngineProcessor.cpp` の routing は以下:
1. `processBlock(AudioBuffer<float>&)` → `getNextAudioBlock()` → `DSPCore::process()` (No bypass blend → **B01**)
2. `processBlock(AudioBuffer<double>&)` → `processBlockDouble()` → `DSPCore::processDouble()` (Has bypass blend ✅)
Standalone では `setDoublePrecisionProcessing(true)` により Double 経路に固定され、
Float 経路は stub (`buffer.clear()`) となる。
したがって「交互呼び出し」は実際には発生しない。
FloatRampState/DoubleRampState 分離の TODO は Appendix N を参照。

**Float/Double 構造一致の保証**: `RampRuntimeState` に Float 用メンバを追加する場合、
Double 版 (`bypassFadeGainDouble` / `bypassedDouble`) と常に同じ構造・初期化経路を維持すること。
Float 版のみ初期化漏れや型の不一致が生じないよう、以下の static_assert で構造一致を保証する:

```cpp
// Float/Double 各メンバの型一致をコンパイル時検証
// ★ Float 版実装完了後に以下の static_assert を有効化:
static_assert(std::is_same_v<decltype(RampRuntimeState::bypassFadeGainFloat),
                             decltype(RampRuntimeState::bypassFadeGainDouble)>,
              "Float/Double bypassFadeGain type mismatch");
static_assert(std::is_same_v<decltype(RampRuntimeState::bypassedFloat),
                             decltype(RampRuntimeState::bypassedDouble)>,
              "Float/Double bypassed type mismatch");
```

**初期化経路**: Double 版と完全に同一のライフサイクルを通すこと:
- `prepareToPlay()`: `RampRuntimeState::prepare()` で `bypassFadeGain` をリセット
- `DSPCore::prepare()`: バッファ確保 + 初期値 `setCurrentAndTargetValue(1.0)`
- `rebuild` / `releaseResources`: Float 版バッファの解放・再確保
- `~DSPCore()`: `dryBypassBufferFloatL/R` の解放

`DSPCoreLifecycle.cpp` の ensure 処理に Float 版の初期化を追加することを忘れないこと。

**LinearRamp と Runtime Clone/Publish 経路**:
ISR Runtime の `RuntimeBuilder` は `DSPCore` を `clone()` してから publish する。
`DSPCore` の `clone()` は `rampState` (`unique_ptr<RampRuntimeState>`) を
ディープコピーするため、`RampRuntimeState` 内の `bypassFadeGain` (LinearRamp) の
`currentValue` / `targetValue` / `remainingSamples` は clone 先に正しく継承される。
したがって Runtime Publish → Crossfade の経路で `bypassFadeGain` が意図せず
リセットされることはない。

**補足**
- **Runtime Publish 時**: `bypassFadeGain` は Runtime Publish → Crossfade 後も有効である必要がある。
  `DSPCore::prepare()` での初期化以降、Runtime 切替時に再初期化不要 (RuntimeBuilder が DSPCore を複製するため)。
- **releaseResources の前提**: `releaseResources()` は Deferred Retire 完了後にのみ実行されること。
  ISR Runtime の Retire → Epoch → delete の完了前にはバッファが解放されない。
  現在の Runtime はこの前提を満たしている。
- **容量管理**: Float 版 (`dryBypassCapacityFloat`) と Double 版 (`dryBypassCapacityDouble`) は
独立した容量変数。EQProcessor も独自の `dryBypassCapacity` を持つ (`EQProcessor.h:638`)。
これら3者は独立した処理層に属するため単一変数への統合は不可能。ただし Float/Double
両 DSPCore 版で同ロジックの重複は残る — 将来 `DSPCore::prepare()` のテンプレート化
または共通 `ensureDryBypassBuffer<T>()` ヘルパー抽出を検討 (本改修スコープ外)。
```cpp
struct DSPCore {
    // ... 既存メンバ ...
    convo::ScopedAlignedPtr<float> dryBypassBufferFloatL;
    convo::ScopedAlignedPtr<float> dryBypassBufferFloatR;
    int dryBypassCapacityFloat = 0;
};

struct RampRuntimeState {
    // ... 既存メンバ ...
    convo::LinearRamp bypassFadeGainFloat;   // Float版追加 (Double版 bypassFadeGainDouble に対応)
    bool bypassedFloat = false;               // Float版追加 (Double版 bypassedDouble に対応)
};
```

#### Step 2: DSPCoreLifecycle.cpp — Float 版バッファ確保・解放
```cpp
// ensure() 相当の処理
if (dryBypassCapacityFloat < newRequired) {
    auto newDryL = convo::makeAlignedArray<float>(static_cast<size_t>(newRequired));
    auto newDryR = convo::makeAlignedArray<float>(static_cast<size_t>(newRequired));
    if (newDryL && newDryR) {
        dryBypassBufferFloatL = std::move(newDryL);
        dryBypassBufferFloatR = std::move(newDryR);
        dryBypassCapacityFloat = newRequired;
    }
}
// 解放
dryBypassBufferFloatL.reset();
dryBypassBufferFloatR.reset();
dryBypassCapacityFloat = 0;
```

#### Step 3: DSPCoreFloat.cpp — bypass blend ロジック追加
Double 版 (DSPCoreDouble.cpp:381-593) の bypass blend ブロックを float 版に移植:
```cpp
const bool requestedFullBypass = state.eqBypassed && state.convBypassed;
auto& ramp = ramps();
if (requestedFullBypass != ramp.bypassed) {
    ramp.bypassFadeGain.setTargetValue(requestedFullBypass ? 0.0f : 1.0f);
    ramp.bypassed = requestedFullBypass;
}
// dry信号保存 ... bypassBlendRequested による gWet/gDry 線形補間 ...
```

### 有効条件
Plugin モード (`CONVOPEQ_STANDALONE_ONLY` 未定義) で発現。
通常の Standalone ビルドでは `setDoublePrecisionProcessing(true)` により Double 経路に固定されるため
Float 経路の bypass blend 欠落は影響しない。ただし `setDoublePrecisionProcessing(false)` に変更された場合は
Float 経路がアクティブになるため B01 が発現する。

---

## 4. B17: StereoConvolver::clone() FilterSpec 欠落 🟡 P1

### 対象ファイル
- `src/ConvolverProcessor.h` (StereoConvolver)

### 修正内容 (3行)

#### 4.1 `StoredConfig` 構造体追加 (ConvolverProcessor.h:645付近)

将来の clone 対象拡張に備え、FilterSpec と hasSpec をひとつの構造体にまとめる:
```cpp
struct StoredConfig {
    convo::FilterSpec filterSpec{};
    bool hasSpec = false;                   // filterSpec==nullptr と {} を区別
};
StoredConfig storedConfig_;
```

これにより `clone()` 対象が `storedConfig_` ひとつになり、FilterSpec にメンバが追加された場合も
`clone()` の修正漏れが発生しない (struct 全体がコピーされるため)。

**clone invariant**: `clone()` 後の StereoConvolver インスタンスは `init()` 直後と等価な状態であり、
IR データ・畳み込みエンジン (NUC) は新規に構築される。Crossfade 状態・Delay 状態・Phase 情報は
clone 元から継承**しない** — 新インスタンスは独立した Runtime として動作する。
この不変条件により、ISR の「Publish 後のインスタンスは不変」という原則が維持される。

#### 4.2 `init()` で filterSpec を保存 (730行付近)
```cpp
if (filterSpec != nullptr) {
    storedConfig_.filterSpec = *filterSpec;
    storedConfig_.hasSpec = true;
}
```

#### 4.3 `clone()` で StoredConfig を参照 (785行)
```cpp
if (!newConv->init(l.release(), r.release(), irDataLength,
                   storedSampleRate, irLatency,
                   storedKnownBlockSize, callQuantumSamples,
                   storedScale, storedDirectHeadEnabled,
                   storedConfig_.hasSpec ? &storedConfig_.filterSpec : nullptr))
```

---

## 5. B13: NUPC レイヤー間遅延アライメント欠落 🟡 P1

### 対象ファイル
- `src/MKLNonUniformConvolver.h` (Layer)
- `src/MKLNonUniformConvolver.cpp` (SetImpulse, Add, Get)

### 事前条件: 実測検証 (MT-NUPC-01〜03)
理論上のバグは確定しているが、実装前に Dirac/MLS/Sweep による実測で各レイヤーの実際の出力遅延を定量化すること (詳細: Appendix C)。

**MT-NUPC 未完了の場合は Phase 1 以降の実装を一切禁止する。特に Phase 2 の `outputDelaySamples` は実測値から決定し、`Σ(partSize × numPartsIR)` (IR 長) を暫定値として実装してはならない。IR 長は FFT overlap, tail scheduling, FDL depth, partition flush などの処理遅延を含まず、Layer の実際の出力遅延と一致しない可能性がある。**

### 段階的実装方針 (2 Phase)

本修正は以下の 2 Phase で構成される:

**Phase 1 (Measurement)**: MT-NUPC-01〜03 で各レイヤーの実際の出力遅延を測定。
- `outputDelaySamples` は IR 長 (partSize * numPartsIR) ではなく実測値から決定する。
- Phase 2 のリングバッファ容量算出のための基礎データ収集が目的。

**Phase 2 (RingBuffer)**: リングバッファによる遅延補償を実装。
- `outputDelaySamples` / `delayLineBuf` / `delayWriteSeq` / `delayConsumeSeq` を Layer に追加。
- Add() で遅延線書き込み、Get() で遅延線読み出し + 加算。
- `outputDelaySamples` の値は Phase 1 の実測結果から決定。

> 注意: tailOutputBuf のみによる簡易補償 (従来の Phase 2 相当) は `tailOutputBuf` の容量が
> `partSize` と同程度であり (`MKLNonUniformConvolver.h:345`)、典型的な NUPC 構成
> (outputDelaySamples >> partSize) では成立しない。そのため仕様書から削除した。
> Phase 1 測定後、直接 Phase 2 リングバッファ方式を実装すること。

### 修正内容

#### Phase 2 実装: リングバッファによる遅延補償

##### Step 1: Layer 構造体に遅延補償メンバ追加 (MKLNonUniformConvolver.h)

```cpp
struct DelayAlignmentConfig {
    int outputDelaySamples = 0; // このレイヤーの出力遅延量 (sample) [実測値から決定]
    int delayLineCapacity = 0;  // リングバッファ容量
};

// ★ DelayAlignmentRuntime: Mutable Runtime State (Builder が Config を構築し、Runtime が State を持つ)
//    delayLineBuf は Runtime が所有 (Config の指示する capacity で Builder が確保)
struct DelayAlignmentRuntime {
    double* delayLineBuf = nullptr; // mkl_malloc 遅延線バッファ (Runtime 所有)
    uint64_t writePartition = 0;    // Add() 完了済みパーティション数 (partition sequence)
    uint64_t readPartition = 0;     // Get() 読み出し完了パーティション数
    uint64_t delayWriteCursor = 0;  // Add() が書き込んだ累積サンプル数 (Add のみ更新)
    uint64_t delayReadCursor = 0;   // ★ 唯一の読み出し Authority (cursor model)
                                    //    Get(): readCursor = max(readCursor, writeCursor - outputDelaySamples);
                                    //    readCursor += numSamples;
    // ★ partition sequence によるタイミング保証:
    //    Add() は IFFT 完了 = 1 partition 完了と同時に writePartition++ し、
    //    対応する tailOutputBuf をリングバッファに書き込む。
    //    Get() は readPartition < writePartition かつ cursor 条件を満たすまで待機。
    //    これにより FFT Layer の生成タイミングと Audio callback の関係が保証される。
};

struct Layer {
    // ... 既存メンバ ...
    DelayAlignmentConfig delayConfig;  // ★ 構成情報 (Immutable, IR 設定時に Builder が決定)
    DelayAlignmentRuntime delayRun;    // ★ 実行時状態 (Runtime 所有)

    void resetDelayAlignment() noexcept {
        delayRun.delayWriteCursor = 0;
        delayRun.delayReadCursor = 0;
        if (delayRun.delayLineBuf)
            juce::FloatVectorOperations::clear(delayRun.delayLineBuf, delayConfig.delayLineCapacity);
    }
};
```

##### Step 2: SetImpulse() で outputDelaySamples 計算 + バッファ確保 (MKLNonUniformConvolver.cpp)

```cpp
for (int li = 0; li < m_numActiveLayers; ++li) {
    Layer& l = m_layers[li];

    // ★ 既存の delayLineBuf を解放 (再構築・IR変更時)
    if (l.delayRun.delayLineBuf != nullptr) {
        DIAG_MKL_FREE(l.delayRun.delayLineBuf);
        l.delayRun.delayLineBuf = nullptr;
        l.delayConfig.delayLineCapacity = 0;
        l.delayRun.delayWriteCursor = 0;
        l.delayRun.delayReadCursor = 0;
    }

    if (li == 0) {
        l.delayConfig.outputDelaySamples = 0;
        l.delayConfig.delayLineCapacity = 0;
        // ★ Layer 0 は即時出力のため delay 不要
    } else {
        // ★ IR 切替 (SetImpulse) 時: 遅延線バッファをクリアしてから再設定
        //    旧 IR の履歴が新 IR に混入するのを防止する。
        l.resetDelayAlignment();
        // ★ outputDelaySamples: Phase 1 (MT-NUPC-01〜03) の実測値から Builder が DelayMeasurementResult を
    //    生成し、LayerDelayProfile として各 Layer に注入する。
    //    Layer 自身は delay を計算せず、プロファイルを受け取るのみ。
    //    以下の暫定計算は Builder の仮実装用プレースホルダ:
        int totalAhead = 0;
        for (int pLi = 0; pLi < li; ++pLi)
            totalAhead += m_layers[pLi].partSize * m_layers[pLi].numPartsIR;
        l.delayConfig.outputDelaySamples = totalAhead;
        // ★ Debug ビルドでは Phase 1 未実施を検出:
        //    outputDelaySamples == 0 かつ li > 0 は実測未完了を示す。
        //    実測値が設定されるまでは jassert で停止する。
        jassert(l.delayConfig.outputDelaySamples > 0);
        const int maximumCallbackSamples = m_maxBlockSize;
        constexpr int kMaxWriterAdvanceMultiplier = 3;
        const int maxWriterAdvance = maximumCallbackSamples * kMaxWriterAdvanceMultiplier;
        const int requiredCapacity = l.delayConfig.outputDelaySamples + maxWriterAdvance;
        l.delayConfig.delayLineCapacity =
            ((requiredCapacity + 15) / 16) * 16;
        l.delayRun.delayLineBuf = static_cast<double*>(
            DIAG_MKL_MALLOC(static_cast<size_t>(l.delayConfig.delayLineCapacity) * sizeof(double), 64));
        if (l.delayRun.delayLineBuf == nullptr) {
            releaseAllLayers();
            return false;
        }
        juce::FloatVectorOperations::clear(l.delayRun.delayLineBuf, l.delayConfig.delayLineCapacity);
    }
}
```

##### Step 2b: freeAll() / Reset() での解放

```cpp
// Layer::freeAll() 内
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // ... 既存の解放処理 ...
    if (delayRun.delayLineBuf) {
        DIAG_MKL_FREE(delayRun.delayLineBuf);
        delayRun.delayLineBuf = nullptr;
        delayConfig.delayLineCapacity = 0;
        delayRun.delayWriteCursor = 0;
        delayRun.delayReadCursor = 0;
        delayConfig.outputDelaySamples = 0;
    }
}

// Reset() 内 (状態初期化のみ、構成情報は保持)
void MKLNonUniformConvolver::Reset()
{
    // ... 既存のクリア処理 ...
    for (int li = 0; li < m_numActiveLayers; ++li) {
        Layer& l = m_layers[li];
        if (l.delayRun.delayLineBuf)
            juce::FloatVectorOperations::clear(l.delayRun.delayLineBuf, l.delayConfig.delayLineCapacity);
        l.delayRun.delayWriteCursor = 0;
        l.delayRun.delayReadCursor = 0;
        // ★ Reset() で delayConfig.outputDelaySamples をクリアしない (構成情報保持)
    }
}
```

##### Step 3: Add() で遅延線に書き込み

**前提**: `Add()` は IFFT 完了時の `tailOutputBuf` 書き込みと同時に呼ばれる。
書き込み単位は常に `partSize` (IFFT ブロック長)。

```cpp
if (l.delayRun.delayLineBuf != nullptr && l.delayConfig.delayLineCapacity > 0) {
    const int toWrite = l.partSize;
    const size_t writeOffset = static_cast<size_t>(l.delayRun.delayWriteCursor % l.delayConfig.delayLineCapacity);
    const int remain = l.delayConfig.delayLineCapacity - static_cast<int>(writeOffset);
    const int first = std::min(toWrite, remain);
    juce::FloatVectorOperations::copy(l.delayRun.delayLineBuf + writeOffset, l.tailOutputBuf, first);
    if (first < toWrite)
        juce::FloatVectorOperations::copy(l.delayRun.delayLineBuf, l.tailOutputBuf + first, toWrite - first);
    l.delayRun.delayWriteCursor += static_cast<uint64_t>(toWrite);
}
```

##### Step 4: Get() で遅延線から読み出して加算

```cpp
for (int li = 1; li < m_numActiveLayers; ++li) {
    Layer& l = m_layers[li];
    auto& d = l.delayAlign;
    if (d.delayLineBuf == nullptr || d.delayLineCapacity <= 0) continue;

    // ★ readSeq を唯一の Authority とする:
    //    readSeq = max(consumeSeq, writeSeq - outputDelaySamples)
    //    これにより「遅延保証 (writeSeq - readSeq >= outputDelaySamples)」と
    //    ★ 無音期間について:
    // ★ delayReadCursor を唯一の読み出し Authority:
    //    Get(): readCursor = max(readCursor, writeCursor - outputDelaySamples)
    //          readCursor += numSamples (読み出し完了時)
    //    ★ 無音期間について:
    //    writeCursor は IFFT 完了時 (Add) にのみ partSize 単位で進む。
    //    readCursor が writeCursor 以上の場合 continue でスキップされ、
    //    そのコールバックでは L1/L2 出力が無音になる。初期充填時や
    //    Long IR の先頭部分では期待動作となる。
    //    ★ オーバーラン検出:
    //    writeCursor - readCursor > cfg.delayLineCapacity の場合、
    //    Writer が Reader を追い越した (古いデータが上書きされた) ことを意味する。
    //    この場合は resetDelayAlignment() で状態リセットし、診断カウンタを増加させる。
    const uint64_t readSeq = (d.delayWriteSeq >= static_cast<uint64_t>(d.outputDelaySamples))
        ? d.delayWriteSeq - static_cast<uint64_t>(d.outputDelaySamples)
        : 0;
    const uint64_t actualReadStart = std::max(d.delayConsumeSeq, readSeq);

    if (actualReadStart + static_cast<uint64_t>(numSamples) > d.delayWriteSeq)
        continue;

    const size_t readOffset = static_cast<size_t>(actualReadStart % d.delayLineCapacity);
    const int first = std::min(numSamples, d.delayLineCapacity - static_cast<int>(readOffset));
    if (first > 0)
        addScaledFallback(first, output, d.delayLineBuf + readOffset, m_tailLayerGain[li]);
    if (first < numSamples)
        addScaledFallback(numSamples - first, output + first, d.delayLineBuf, m_tailLayerGain[li]);

    d.delayConsumeSeq = actualReadStart + static_cast<uint64_t>(numSamples);
}
// ★ State Mutation Authority:
//    delayWriteCursor: Add() のみが更新 (IFFT 完了時)
//    delayReadCursor: Get() のみが更新 (読み出し完了時、唯一の Authority)
//    delayConfig.*: Builder が設定 (IR 変更時)、Reset() は保持
//    delayRun.delayLineBuf: Builder が確保/解放、freeAll() で破棄
```

##### Add/Get 呼び出し粒度の非対称性への対応

NUPC では `Add()` と `Get()` の呼び出し回数が常に一致するとは限らない。
この場合 `delayWriteSeq` と `delayConsumeSeq` の間でトラッキング誤差が生じる
可能性があるため、以下の不変条件を設計に組み込む:

```
不変条件: (delayWriteSeq - delayConsumeSeq) <= delayLineCapacity かつ >= 0
           (unsigned overflow arithmetic で保証)
```

`delayConsumeSeq` は実際に読み出したサンプル数の累積であり、`desiredReadPos`
(delayWriteSeq - outputDelaySamples) との `std::max` により、
「outputDelaySamples 分の遅延保証」と「再読み出し防止」を両立する。
`Get()` の呼び出しサイズが `partSize` と異なる場合でも、常に `actualReadStart`
から `numSamples` 分を読み出し、`delayConsumeSeq` を進めるため、
書き込み側 (`partSize` 単位) との間に累積誤差は生じない。

#### Crossfade 互換性判定 (`isDelayCompatibleWith`)

StereoConvolver の `isDelayCompatibleWith` は、クロスフェード実行中に
遅延補償構成が変更された場合に、現在の Crossfade を継続可能かの判定に使用される。
比較対象:
```cpp
bool isDelayCompatibleWith(const StereoConvolver& other) const noexcept {
    for (int li = 0; li < m_numActiveLayers; ++li) {
        if (layers_[li].delayConfig.outputDelaySamples !=
            other.layers_[li].delayConfig.outputDelaySamples)
            return false;
        // ★ partitionSize と layerCount の一致も必須 (バッファ構成変更は
        //    Crossfade 中の書き込み位置を無効化するため)
        if (layers_[li].partSize != other.layers_[li].partSize)
            return false;
    }
    if (m_numActiveLayers != other.m_numActiveLayers)
        return false;
    return true;
}
```
`outputDelaySamples` / `partSize` / `layerCount` がすべて一致する場合のみ互換性あり。
一つでも異なる場合、Crossfade 時間を 1.5x に延長して安全に遷移する。

#### 遅延算出式

各 Layer の `outputDelaySamples` は MT-NUPC 実測値から決定する。理論的な近似式:
```
outputDelaySamples(Li) ≈ Σ_{k=0}^{i-1} (Lk.partSize × Lk.numPartsIR)
```
これは先行 Layer の IR 総長 (= partSize × numPartsIR) の積算であり、
後続 Layer (L1, L2) が先頭 Layer (L0) の IR 長だけ出力が遅延されるという
Gardner/Garcia/Wefers NUPC 理論に基づく。ただし FFT overlap, tail scheduling,
FDL depth 等の処理遅延を含まないため、実測値で補正すること。
```

**解放パスの一覧** (freeAll 経由で DelayAlignment の全リソース解放):

| 呼出元 | 解放方法 | 備考 |
|--------|---------|------|
| `~MKLNonUniformConvolver()` | `releaseAllLayers()` → `freeAll()` | デストラクタ |
| `SetImpulse()` (IR変更) | `DIAG_MKL_FREE` + 再確保 (Step 2) | 再構築時 |
| `Reset()` | `clear()` + writeSeq/consumeSeq リセット (outputDelaySamples は保持) | パラメータ変更時 |
| `freeAll()` | `DIAG_MKL_FREE` + nullptr 化 | Layer 全解放 |

---

## 6. B08: CacheMap dtor m_retireRouter UAF 🟡 P1 (WARN)

### 対象ファイル
- `src/RefCountedDeferred.h`
- `src/audioengine/AudioEngine.h`

### 修正内容 (2箇所)

#### 6.1 RefCountedDeferred.h — `release()` に `RetirePolicy` 導入

`RetirePolicy` による統一 release API を導入する。`release(RetirePolicy::Immediate)` は
シャットダウン専用であり、通常運用では使用禁止。

**利用条件 (Design Rule)**:
- `release(RetirePolicy::Immediate)` は `AudioEngine::ShutdownPhase::Destroy` 以上の場合のみ呼び出し可能。
- これを保証するため、`~CacheMap()` 内で `owner->getShutdownPhase()` による実行時確認を実施する。
- 他のコードパスから `release(RetirePolicy::Immediate)` を呼んだ場合の動作は未定義。
- 将来の誤用防止のため、`RetirePolicy::Immediate` を利用する全コードパスを grep で監査可能な状態に保つ。
- **`release(RetirePolicy::Immediate)` は `~CacheMap()` 専用**であり、`CacheMap` dtor 以外からの
  呼び出しを許可しない。`~CacheMap()` は AudioEngine の shutdown 時とキャッシュ世代交代時の両方で
  呼ばれるが、`RetirePolicy::Immediate` は shutdown 時のみ使用される (`getShutdownPhase()` で分岐)。

`releaseImmediatelyForShutdown()` メソッドを追加する (最小修正)。
既存の `release(IEpochProvider&)` は変更せず、新しいオーバーロードは追加しない。
命名の意図: 「シャットダウン時に即時解放する」ことを明確に表現。

```cpp
// RefCountedDeferred.h
template <typename T>
class RefCountedDeferred {
    friend class AudioEngine::EQCacheManager::CacheMap;
private:
    // ... addRef(), release(provider), tryAddRef() ...

    // ★ releaseImmediatelyForShutdown: Shutdown 専用 — RetireRouter を経由せず即時 delete
    //    AudioEngine がシャットダウン中 (他スレッドが参照していないことが保証)
    //    である場合のみ使用。CacheMap::~CacheMap() からのみ呼ばれる。
    //    ISR の EBR を完全に迂回するため、通常運用での呼び出しは禁止。
    //    private だが friend CacheMap により呼出可能。

**採用方式: ShutdownToken**:
`friend` は Authority を増やしやすいため、専用トークン型 `ShutdownToken` を導入する。
これにより `releaseImmediatelyForShutdown` は Token 型を要求し、friend 宣言が不要になる。
ISR の Authority Singularization にも完全に適合する:

```cpp
// ShutdownToken.h — 軽量トークン型 (独立ヘッダ、依存ゼロ)
// ★ ShutdownToken.h — RetirePolicy.h に改名
#pragma once
enum class RetirePolicy {
    Epoch,      // ★ 通常: EBR 経由 (IEpochProvider& が必要)
    Immediate   // ★ Shutdown: EBR を迂回し即時 delete (Shutdown 専用)
};

// RefCountedDeferred.h (friend 宣言不要)
template <typename T>
class RefCountedDeferred {
private:
    // ... addRef(), release(provider), tryAddRef() ...

    // ★ 統一 release API: RetirePolicy で dispatch。オーバーロード増殖を防止。
    [[nodiscard]] bool release(RetirePolicy policy) noexcept {
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            if (policy == RetirePolicy::Immediate) {
                delete static_cast<T*>(this);
            }
            // Epoch: enqueueRetire は呼出元で別途
            return true;
        }
        return false;
    }
    void release(IEpochProvider& provider) noexcept {
        // 既存: RetireRouter 経由
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            provider.enqueueRetire(this, &destroy);
        }
    }
};
```

#### 6.2 AudioEngine.h — CacheMap dtor の修正

`~CacheMap()` は shutdown 時と通常運用時の両方で呼ばれる。
`RetirePolicy` で分岐:

```cpp
~CacheMap() {
    jassert(owner != nullptr);
    if (owner->getShutdownPhase() >= AudioEngine::ShutdownPhase::Destroy) {
        for (auto& entry : map)
            if (entry.second != nullptr)
                entry.second->release(RetirePolicy::Immediate);
    } else {
        for (auto& entry : map)
            if (entry.second != nullptr)
                entry.second->release(*owner->m_retireRouter);
    }
}
```
```

---

## 7. B18: destroyQuarantineSlot メモリリーク 🟡 P1 (WARN)

### 対象ファイル
- `src/audioengine/AudioEngine.h` (quarantineSlot の実装)
- `src/audioengine/ISRDSPHandle.cpp` (destroyQuarantineSlot)

### 修正内容
`AudioEngine::quarantineSlot()` 内で、DSPCore インスタンスを直接解放せず標準 Retire パスに統合。

**重要な実行順序**: `dspHandleRuntime_.resolve(handle)` は `DSPHandleRuntime::registry_` (Projection) を読む。
コード確認の結果 (`ISRDSPHandle.cpp:38-57`), `resolve()` は `reg.state` が Quarantined または Reclaimed の場合は `{nullptr, false, false}` を返す。
したがって `quarantineSlot()` による Projection 更新 (state→Quarantined) **より前に** `resolve()` を実行しなければならない。

正しい順序:
3. `DSPQuarantineManager::quarantineHandle()` — Truth Store 更新 (唯一の Authority)
   ★ ISR 設計では `DSPHandleRuntime::quarantineSlot()` は外部 API として公開せず、
     Manager からの内部委譲としてのみ動作する。これにより quarantine Authority が
     Manager 1箇所に単一化される。
2. **`dspHandleRuntime_.resolve(handle)` → `retireDSPHandleForRuntime(dsp)`** — ★ Projection 更新前に実行
3. `dspHandleRuntime_.quarantineSlot(slot)` — Projection 更新 (state→Quarantined)
4. `retireRuntimeEx_.quarantine(slot)` — Retire 投影更新

```cpp
// quarantineSlot() 内 (Step 1: Truth Store 更新後, Step 3: Projection 更新前に実行)
const convo::isr::DSPHandle handle{slot, generation};
// ★ resolve() は ISRDSPHandle.cpp:38-57 の registry_ 参照。
//    state==Quarantined だと nullptr を返すため、quarantineSlot() より前に実行必須。
const auto resolved = dspHandleRuntime_.resolve(handle);
DSPCore* dsp = static_cast<DSPCore*>(resolved.instance);
if (dsp != nullptr) {
    // ★ retireDSPHandleForRuntime は runtimeDSPHandleMap_ からエントリを削除するため、
    //    既に retired 済みの DSP には何もしない (二重登録防止)。
    //    ISRDSPHandle 側の retire() も state が Active/Retired 以外なら何もしない。
    retireDSPHandleForRuntime(dsp);  // EpochDomain 経由の deferred delete
}
```

`destroyQuarantineSlot()` は `registry_[slot].instance = nullptr` による registry クリアのみ担当。

**Design Rule — DSPCore 解放権限**:
```
DSPCore インスタンスを delete / destroy / 解放できるのは
RetireRuntime (EpochDomain/deferred delete) のみ。

禁止: destroyDSPCoreNode() の直接呼び出し (内部使用のみ)
禁止: DSPCore インスタンスへの operator delete の直接適用
許可: retireDSPHandleForRuntime(DSPCore*) — EpochDomain 経由
許可: release(RetirePolicy::Immediate) — Shutdown 時のみ (Runtime publish 後は禁止)
```

#### Quarantine 状態遷移図

```
Registry[slot] (state==Active, instance!=nullptr)
  │
  ├─ Quarantine: state→Quarantined (DSPQuarantineManager)
  │     │
  │     ├─ resolve() → retireDSPHandleForRuntime() [DSPCore を EBR に委譲]
  │     │     │
  │     │     └─ runtimeDSPHandleMap_ から削除
  │     │
  │     ├─ dspHandleRuntime_.quarantineSlot(slot)
  │     │     └─ registry_[slot].instance = nullptr [Projection 更新]
  │     │
  │     └─ retireRuntimeEx_.quarantine(slot) [Retire 投影更新]
  │
  ├─ RetireQueue: EpochDomain 経由の deferred delete
  │     └─ enqueueDeferredDeleteNonRt → destroyDSPCoreNode
  │
  ├─ Epoch: epoch 進行後、deferred delete 実行
  │
  └─ Destroy: ~DSPCore() → slot 解放 → Generation 進行
        └─ Registry[slot].generation++ (次 Generation で再利用可能)
```

**ライフサイクル全体** (Generation を含む):

```
slot=5, generation=13 (Active)
  │
  ├─ DSPQuarantineManager::quarantineHandle()  [Truth Store更新]
  │
  ├─ retireDSPHandleForRuntime(dsp)  [runtimeDSPHandleMap_削除]
  │     │
  │     ├─ dspHandleRuntime_.retire(handle)  [state→Retired]
  │     └─ enqueueDeferredDeleteNonRt → destroyDSPCoreNode [Epoch経由]
  │           │
  │           └─ EpochDomain: epoch 進行後、deferred delete 実行
  │                 │
  │                 └─ ~DSPCore() [generation=13 終了]
  │
  ├─ dspHandleRuntime_.quarantineSlot(slot) [state→Quarantined, instance=nullptr]
  │
  └─ retireRuntimeEx_.quarantine(slot) [Retire投影更新]

次 Generation: slot=5, generation=14 で再利用可能
```
ISR 思想では `DSPHandle` (論理ハンドル) を最後まで持ち歩き、`resolve()` を retire 直前まで
遅延させることで `DSPCore*` (生ポインタ) の有効期間を最小化できる。現在の実装でも
`resolve()` の結果は同一 Commit サイクル内でのみ有効だが、将来的には:
```cpp
// 理想的: handle を最後まで保持し、retire 直前に resolve
DSPHandle handle{slot, generation};
// ... ここでは resolve しない ...
retireDSPHandleForRuntime(handle);  // 内部で resolve + retire
```
これにより Authority (resolve した生ポインタの生存期間) がさらに明確になる。
本修正では従来の `resolve→retire→quarantine` 順序を維持する。

**スレッド安全性の前提**:
以上の順序は「Commit 単一スレッド」の前提で成立する。Projection (`DSPHandleRuntime::registry_`)
は Commit Thread のみが書き換え、Audio Thread は Read-Only。したがって `resolve()` の結果は
同一 Commit サイクル内でのみ有効であり、別 Thread による割り込み変更は発生しない。
この前提は ISR Runtime の `ISRRuntimePublicationCoordinator.h:165` (MessageThread-only access)
の設計に一致する。

---

## 8. B03: NoiseShaperLearner 冗長 vdTanh 🟢 P2 (WARN)

### 対象ファイル
- `src/NoiseShaperLearner.cpp`

### 修正内容

`vdTanh` (162要素 = kPopulation=18 × kDim=9) を `runEvaluationJobsForWorker()` 内ではなく主 worker スレッドで**Generationごとに1回だけ**計算し、
その結果を全 worker が読み取り専用で共有する。

#### 前提条件
- **mutation と vdTanh の順序**: population 行列は mutation より前に生成される。vdTanh は mutation 後の個体値には依存せず、mutation 前の population に対して一度だけ計算する。したがって各 Worker の入力は同一であり、共有可能。
- **mappedPopulation の寿命**: Generation 開始時に生成され、全 Worker の join 完了まで保持される。Main Worker が所有し、aux Worker は処理完了後に参照を解放する。
- **false sharing**: 共有データは読み取り専用 (`const`) であるため、複数 Worker からの同時アクセスによる false sharing は発生しない。

#### `std::span<const double>` の寿命ライフサイクル
```
Generation 開始
  │
  ├─ mappedPopulation 生成 (Main Worker が所有)
  │     │
  │     ├─ vdTanh(mappedPopulation) → sharedTanh (共有バッファ、1回のみ)
  │     │
  │     ├─ Worker 1 開始 → span<const double>(sharedTanh) を受け取る
  │     ├─ Worker 2 開始 → span<const double>(sharedTanh) を受け取る
  │     ├─ ...
  │     └─ Worker N 開始
  │           │
  │           └─ join (全 Worker 完了)
  │
  ├─ mappedPopulation は join 完了まで維持
  │     │
  │     └─ (Generation 終了) → sharedTanh 破棄
  │
  └─ 次 Generation 開始 → 新しい mappedPopulation で再計算
```
各 Worker が受け取る `span<const double>` は mappedPopulation の生存期間中のみ有効。
join 完了後は span も無効化される。この寿命管理は Main Worker の所有権モデルにより保証される。

#### 共有バッファ仕様
- アライメント: **64 バイトアライメント**で確保 (既存 `ScopedAlignedPtr<double>` を使用、MKL 推奨アライメント)
- 型: `std::span<const double>` で各 Worker に渡す (サイズ情報付き読み取り専用ビュー、const 保証)
- 所有: Main Worker スレッドが所有、aux Worker は参照のみ (同期不要)

#### 再計算条件
- Generation 開始時に `mappedPopulation` を生成
- 同一 Generation 内の全 Worker ジョブで再利用
- 次 Generation 開始時に新しい population に対して再計算

#### 期待効果
- MKL `vdTanh` 呼び出し回数: (1 + N_aux) 回 → 1 回に削減
- データ: 162 要素、約 1.3 KB (L1 Cache に完全収容)
- CPU 負荷低減: NoiseShaper 学習の大量反復時に有意

---

## 9. B21: SimplePeakLimiter Knee (設計者判断で除外)

数学的には正しいバグ (`thresholdLinear` -> `thresholdLinear + kneeLinear * 0.5` の修正が C1 連続性を確保) だが、設計者判断により本バグリストから除外。別途Limiter全体改修で対応予定。

---


---

---

# Appendix

## Appendix A: バグ解析詳細

本 Appendix には、各バグ項目の「① バグ概要」「② バグ発生個所」「③ バグの詳細」を収録する。
実装に必要な情報（コード変更仕様）は上記「設計」部を参照のこと。

### A.1 B14 代替案詳細（Producer集約・fetch_add簡易版）

#### 代替案A: Producer 集約（変更量最小）
4つの producer スレッドのうち3つ（Timer, ReleaseResources, Coordinator）が呼ぶ `emitRetireIntent` を Commit スレッド経由に集約する。

```cpp
// Timer.cpp: 従来: retireRuntime_.emitRetireIntent(highIntent);
// 集約後: pendingRetireIntent_.store(intent); signalCommitThread();
```

- **デメリット**: コミットサイクルまで retire が遅延される
- **実質的な並行性**: Coordinator/Commit/Timer はすべて Message Thread 上で動作するため、真の並行 Producer は ReleaseResources のみ

#### 代替案B: fetch_add 簡易版（最小MPSC化）
```cpp
void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    const uint64_t tail = convo::fetchAddAtomic(retireIntentTail_, 1, std::memory_order_acq_rel);
    const size_t idx = tail % RETIRE_INTENT_QUEUE_SIZE;
    retireIntentQueue_[idx] = intent;
    // fetch_add が既に release 順序付け
}
```
- `retireIntentQueue_` 非atomic配列へのアクセスは、Producer 間で異なる idx に排他的に書き込むため race-free
- dequeue 時に未完成 slot を読まない保証が必要（sequence 番号など簡易版では実装依存）

### A.2 B20 バッファオーバーフロー証明

`numSamples >= maxBlockSize/2` でバッファオーバーフローが発生する:

| 領域 | アドレス範囲 | サイズ |
|------|-------------|-------|
| Stage 0 L 出力 | work[0] ~ work[numSamples*2-1] | numSamples*2 |
| Stage 0 R 出力 | work[numSamples*2] ~ work[numSamples*4-1] | numSamples*2 |
| Stage 1 L 出力 | work[numSamples*4] ~ work[numSamples*8-1] | numSamples*4 |
| 確保容量 | ~ work[maxBlockSize*4-1] | maxBlockSize*4 |

### A.3 B13 数学的証明

フルIRを h、入力を x とすると、時刻 t における真の畳み込み出力:

$$y[t] = \sum_{k=0}^{L-1} x[t-k] \cdot h[k]$$

NUPC では h を3セグメントに分割。後段レイヤーの寄与には L0 サンプルの遅延補償が必要:

$$y[t] = \text{output}_0[t] + \text{output}_1[t - L_0] + \text{output}_2[t - L_0 - L_1]$$

現状の Get() は遅延補償なしで直接加算。

### A.4 B17 問題箇所

`StereoConvolver::clone()` が `filterSpec` 引数なしで `init()` を呼び出す。
`init()` のデフォルト引数 nullptr が使われ、クローン先の NUC にフィルタが適用されない。
`storedFilterSpec` メンバ自体が存在しない。

### A.5 B08 宣言順序問題

- `eqCacheManager`: 2123行 (先に破棄される)
- `m_retireRouter`: 4137行 (後に破棄される???)
→ C++ 規約 [class.cdtor]: メンバは**宣言の逆順**で破棄される
→ `m_retireRouter`(4137行) が `eqCacheManager`(2123行) より**後方**宣言のため**先に破棄**

### A.6 B18 Quarantine 解放フロー

Commit.cpp:631-633 の3系統解放:
1. `retireRuntimeEx_.reclaim(qslot)` — Lane解放
2. `dspHandleRuntime_.destroyQuarantineSlot(qslot, 0)` — Reclaimed遷移
3. `dspQuarantineManager_.reclaimSlot(qslot, 0)` — Flag解放

## Appendix B: コードベース実測値

### B.1 既存コード照合一覧

| 文書の主張 | コード実測 | ステータス |
|-----------|-----------|-----------|
| ScaleFactorResult に residualRiskDb 未追加 | IRConverter.h:13 — scaleFactor/hasScaleFactor のみ | 未追加確認 |
| PreparedIRState に residualRiskDb 未追加 | PreparedIRState.h — 同様に未追加 | 未追加確認 |
| IRState に residualRiskDb 未追加 | ConvolverProcessor.h:1011 — ir/sampleRate/generation のみ | 未追加確認 (work72 GainStaging スコープ) |
| computeMaxGainDb() 未実装 | コードベース全体で 0 hits | 未実装確認 (work72 GainStaging スコープ) |
| currentPreparedIr は AudioEngine に存在しない | AudioEngine.h で 0 hits | ConvolverProcessor.getIrResidualRiskDb() に変更 |
| z = exp(+jomega) | EQProcessor.Coefficients.cpp:327 — z(cos(w), sin(w)) | 一致確認 |
| ConvolverThenEQ パスで trim 不適用 | DSPCoreDouble.cpp:429-457 — 該当パスに trim コードなし | 確認 |
| EQThenConvolver パスで trim 適用 | DSPCoreDouble.cpp:483 — convolverInputTrimGain != 1.0 | 確認 |
| clipping: input [-12, maxDb] | AudioEngine.Parameters.cpp:224-242 — juce::jlimit(-12.0f, maxDb, db) | 確認 |
| clipping: trim [-12, 0] | AudioEngine.Parameters.cpp:277 — juce::jlimit(-12.0f, 0.0f, db) | 確認 |
| clipping: makeup [0, 12] | AudioEngine.Parameters.cpp:247-248 — juce::jlimit(0.0f, 12.0f, db) | 確認 |
| setProcessingOrder: submit→apply の順 | AudioEngine.Parameters.cpp:268-275 | 削除対象 |
| setProcessingOrder: sendChangeMessage なし | AudioEngine.Parameters.cpp:268-275 | **確定バグ** (sendChangeMessage 欠落、同等 setter は全件持つ) |
| m_isRestoringState ガード | AudioEngine.Parameters.cpp:298 | 確認 |
| FADE_IN_SAMPLES = 2048 (42ms) | AudioEngine.h:973 | 確認 |
| DSPCore フェードイン実装 | DSPCoreDouble.cpp:605-617 + DSPCoreFloat.cpp:399-411 | 確認 |
| updateGainStagingDisplay() 既存 | DeviceSettings.cpp:599 | 確認 |
| DspNumericPolicy.h 存在(374行) | src/DspNumericPolicy.h | 確認 |
| ScopedDftiDescriptor (MKL DFTI) | src/DftiHandle.h — RAII wrapper | 確認 |
| atomic<uint64_t>::is_always_lock_free | AudioEngine.h:1013 周辺 | 確認 |
| convolverParamsChanged 末尾に sendChangeMessage | AudioEngine.UIEvents.cpp:240 | 確認 |
| endBulkParameterRestore は m_isRestoringState=false 後 | AudioEngine.Parameters.cpp:207-218 | 確認 |
| requestLoadState は m_isRestoringState=true 中 | AudioEngine.StateIO.cpp — RestoreStateGuard 内 | 確認 |

### B.2 テストファイル既存状況

| テストファイル | 状態 |
|--------------|------|
| src/tests/ShadowCompareContractTests.cpp | 既存 |
| src/tests/RuntimeSemanticSchemaValidationTests.cpp | 既存 |
| src/tests/PriorityIntegrationTests.cpp (B14 回帰テスト拡張元) | 既存 |

## Appendix C: 文献調査結果

| 項目 | 文献 | 結果 |
|------|------|------|
| Butterworth Q = 0.707 | Wikipedia | Q = 1/sqrt(2) 確認 |
| Q = 1/(2zeta) | Wikipedia | 減衰比との関係確認 |
| RBJ Peaking 係数 | W3C | 完全一致 |
| Bencina RT安全原則 | 業界標準 | ConvoPeq 完全準拠確認 |
| Gardner 1995 NUPC | AES (E-Lib 7987) | 分割後段レイヤーに遅延補償が必要。非一様分割では L0 が最小遅延、後続レイヤーは L0 IR 長分遅れて出力される。加算時に L0 IR 長分の遅延アライメントが必要。 |
| Garcia 2002 NUPC | AES | Gardner と同様。後段レイヤー遅延補償が必要。 |
| Wefers 2015 NUPC | PhD Thesis | 同上。FDL (Frequency Domain Delay Line) による遅延補償方式。リングバッファ方式は Phase 3 に相当。 |

## Appendix D: 回帰テスト計画

### D.1 B14: Retire Queue

| ID | テスト名 | 方法 | 期待結果 |
|----|---------|------|---------|
| RG-B14-01 | Multi-Producer 同時発行 | N スレッドから同時に emitRetireIntent (N=2,4,8) | 全 intent が消失なく取得 |
| RG-B14-02 | Queue Overflow / Fallback | キューサイズを超える intent 発行 | Fallback に格納 → dequeue で全件 |
| RG-B14-03 | ABA 安全性 | 同一 slot wrap-around | Sequence number 一貫性 |
| RG-B14-04 | Overflow→Fallback→Drain→再利用 | Queue Full → Fallback → dequeue → full cycle | Fallback drain 後、再利用可能 |
| RG-B14-05 | シャットダウン drain | 未処理 intent 残存状態でシャットダウン | 正常終了。ASANエラーなし |
| RG-B14-06 | 非RTアサーション | Debug ビルドで Audio Thread → emitRetireIntent | jassert 発火 |
| RG-B14-07 | reinterpret_cast 排除確認 | ISRRetire.cpp を grep | 0 hits |

### D.2 B20: TruePeakDetector

| ID | テスト名 | 方法 | 期待結果 |
|----|---------|------|---------|
| RG-B20-01 | R チャンネル TruePeak | L=0dB, R=-6dB 正弦波 | L/R 独立計測 |
| RG-B20-02 | バッファオーバーフロー | maxBlockSize=4096 + numSamples=2048 | ASAN 0 errors |
| RG-B20-03 | モノラル互換性 | dataR==nullptr で呼出 | R = L 複製 |
| RG-B20-04 | BS.1770-4/5 適合性 | ITU-R BS.1770-4 テストトーン | ±0.1dB 以内 |

### D.3 B01: Float Bypass Blend

| ID | テスト名 | 方法 | 期待結果 |
|----|---------|------|---------|
| RG-B01-01 | Float クロスフェード | Float 経路で EQ+Conv OFF→ON | クリック/ポップなし、5ms fade |
| RG-B01-02 | Plugin モード | CONVOPEQ_STANDALONE_ONLY 未定義ビルド | Float 版クロスフェード波形 |
| RG-B01-03 | Standalone 回帰 | CONVOPEQ_STANDALONE_ONLY 定義ビルド | Double 版動作変化なし |

### D.4 B17: clone FilterSpec

| ID | テスト名 | 方法 | 期待結果 |
|----|---------|------|---------|
| RG-B17-01 | FilterSpec 伝搬 | HC/LC フィルタ有効時 shareConvolutionEngineFrom | 元とクローンで周波数特性一致 |
| RG-B17-02 | nullptr 安全 | filterSpec==nullptr | デフォルト動作維持 |

### D.5 B13: NUPC Delay Alignment

| ID | テスト名 | 方法 | 期待結果 |
|----|---------|------|---------|
| RG-B13-01 | Dirac IR 時間応答 | 単位インパルス IR での出力波形 | 補償後、L0 サンプル誤差内 |
| RG-B13-02 | Null IR 安全性 | 全ゼロ IR | 出力全 zero |
| RG-B13-03 | Long IR 安定性 | 3s IR | ASAN エラーなし |
| RG-B13-04 | Null IR 安全性（再） | 全ゼロ IR、遅延線バッファ確認 | 遅延線も zero |
| RG-B13-05 | Dirac 波形比較 | 補償前後の Diract 応答 | L0 サンプル手前に異常なし |
| RG-B13-06 | Sweep 位相 | 対数スイープ IR | 補償後位相誤差減少 |
| RG-B13-07 | 5秒+ Long IR | sampledata/synthetic_long_ir_20s.wav | 正常動作 |
| RG-B13-08 | Crossfade 中 IR Reload | クロスフェード中 IR 再読込 | 新しい delay offset 適用 |
| RG-B13-09 | Crossfade 一貫性 | delay offset 変化時 | 時間延長(isDelayCompatibleWith) |
| RG-B13-10 | TailMode 相互作用 | AirAbsorption/LayerTailContouring/Bypass | 各モード正しく動作 |
| RG-B13-11 | Partition Boundary 2047/2048/2049 | IR 長 2047, 2048, 2049, 4095, 4096, 4097 サンプルの各境界で Dirac 応答を検証 | Layer 切替境界でプリエコー・ドロップアウトなし |
| RG-B13-12 | Partition Boundary 8191/8192/8193 | IR 長 8191, 8192, 8193 サンプルの境界でも同様に検証 (L1/L2 切替境界) | 同上 |

### D.6 WARN 項目 (B08 + B18)

| ID | テスト名 | 方法 | 期待結果 |
|----|---------|------|---------|
| RG-B08-01 | CacheMap dtor UAF | AudioEngine 破棄順序 | ASAN 0 errors |
| RG-B18-01 | Quarantine + Retire 統合 | Quarantine → retireDSPHandleForRuntime → deferred delete | DSPCore 解放、リークなし |
| RG-B18-02 | 二重解放防止 | Quarantine + 通常 Retire 両方 | crash なし |

## Appendix E: 性能受入基準

**共通ベンチマーク条件**:
- Sample Rate: 48kHz | Block Size: 64 samples | ステレオ | IR 長: 4096 taps
- SIMD: AVX2 | Build: Release (/O2) | 10000 callbacks 平均
- CPU: 実測環境を明記 (例: Intel Core i9-13900K)

| ID | 項目 | 基準 | 測定方法 |
|----|------|------|---------|
| RG-AC-01 | B14 enqueue レイテンシ | 10μs 未満 | __rdtsc() 10000回平均 |
| RG-AC-02 | B14 dequeue レイテンシ | 5μs/件 未満 | __rdtsc() |
| RG-AC-03 | B13 Get() オーバーヘッド | 修正前の5%以内 | 10000回比較 |
| RG-AC-04 | B13 SetImpulse() オーバーヘッド | 10%以内 | __rdtsc() |
| RG-AC-05 | B20 メモリ増加 | 32KB以内 | sizeof + ASAN |
| RG-AC-06 | B20 CPU 増加 | 10%以内 | processBlock 前後比較 |

## Appendix F: RT 安全性テスト

| ID | テスト名 | 方法 | 期待結果 |
|----|---------|------|---------|
| RG-RT-01 | Audio Thread ブロック | lock/malloc/new/IO 確認 | 0 violations |
| RG-RT-02 | B14 jassert 発火確認 | Debug ビルド | jassert ヒットなし |
| RG-RT-03 | B13 遅延線影響 | Get() 内メモリ確保・lock 確認 | tailOutputBuf と同等 |
| RG-RT-04 | Producer Storm | 8スレッド同時 emitRetireIntent (10000回) | drop 0、dequeue 全件成功 |
| RG-RT-05 | Crossfade Storm | 短周期 (10ms) で Crossfade を100回実行 | ASAN 0 errors、メモリリークなし |
| RG-RT-06 | Retire Storm | Retire Intent を連続発行 (100000回) | Epoch 進行停止なし、ABA 安全 |

## Appendix G: クロスカッティング分析

**依存グラフ** (実線 = ファイル依存、破線 = Audio 信号依存):
```
B13 (MKLNonUniformConvolver.delay)
  │
  ├── B17 (StereoConvolver.clone) ── Audio信号依存 ── B13
  │     │                              (遅延構成変更 → Crossfade 影響)
  │     └── B08 (RefCountedDeferred) ── 独立 (別ファイル)
  │
  ├── B14 (ISRRetire) ── 独立 (Authority 分離)
  │
  ├── B18 (ISRDSPHandle) ── 独立 (Quarantine ≠ Retire Queue)
  │
  ├── B20 (TruePeakDetector) ── 独立 (信号チェーン別)
  │
  ├── B01 (DSPCoreFloat) ── 独立 (同一DSPCore内だが別経路)
  │
  └── B03 (NoiseShaper) ── 独立 (別モジュール)
```

| 組み合わせ | 分析 |
|-----------|------|
| B14 + B08 | 両者 m_retireRouter 関連。B14 修正はキュー構造のみで B08 に影響せず → Runtime独立 |
| B17 + B13 | StereoConvolver vs MKLNonUniformConvolver → 独立した Layer (ただし遅延構成変更は Crossfade に影響しうる) |
| B13 + B14 | ファイル分離 (MKLNonUniformConvolver vs ISRRetire) → 独立 |
| B18 + B14 | Quarantine パスは emitRetireIntent 不使用 → 独立 |
| B20 + B01 | 独立した信号チェーン上で動作 → 独立 |

**注記**: 上記の「独立」は Runtime の Authority 分離を意味する。Audio 信号の
遅延整合・Crossfade 動作などの観点では、B13 (遅延補償) と B17 (clone FilterSpec) は
同一の ConvolverProcessor 内部で動作するため、Audio 結果は完全には独立ではない。

## Appendix H: ISR 設計評価

各修正が ISR (Practical Stable ISR Bridge Runtime) の設計原則を維持しているかを
以下の6軸チェックリストで評価する:

| チェック項目 | 説明 | B20 | B17 | B08 | B18 | B01 | B14 | B03 | B13 |
|------------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Authority** | Authority が増加していないか | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Frozen** | Frozen World を維持 (Publish 後 Runtime 不変) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **RT Safe** | RT パスで lock/alloc/IO なし | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Lifetime** | オブジェクト Lifetime が明確か | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Memory Order** | Acquire/Release が正しいか | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Shutdown** | Shutdown 時に安全に終了できるか | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**評価方針**: すべて ✅ の項目のみ実装可能。1つでも「❌」がある場合は設計の再評価が必要。
本バグ修正は全6軸をクリアしており、ISR 設計との整合性は十分に維持されている。

## Appendix I: 検証プロセスで使用したツール

| ツール | 用途 |
|--------|------|
| Serena MCP | search_for_pattern / get_symbols_overview |
| context-mode ctx_execute | WSL bash 経由 grep/awk |
| WSL rg/fd | 高速コード検索 |
| WSL ast-grep (sg) | C++ AST パターンマッチ |
| cocoindex-code (ccc) | grep クロス検索 |
| AiDex | aidex_query / aidex_signature |
| semble | セマンティックコード検索 |
| read_file | コンテキスト読み取り |

## Appendix J: 改訂履歴

| 日付 | 版 | 変更内容 |
|------|-----|---------|
| 2026-07-12 | v1.0 | 初版。文献調査結果反映。全12ファイル分確定 |
| 2026-07-12 | v1.1 | 7件の重大問題修正 (currentPreparedIr, IRState, preset restore, setProcessingOrder) |
| 2026-07-12 | v1.2 | DspNumericPolicy 新規作成不要を確定。computeMaxGainDb 方針修正 |
| 2026-07-13 | v1.3 | 評価フレームワーク追加。B13 FDL精査。回帰テスト計画追加 |
| 2026-07-13 | v1.4 | B14 Coordinator MessageThread確認。クロスカッティング分析追加 |
| 2026-07-13 | v1.5 | 本再構成版。実装仕様を先頭に集約、解析詳細を Appendix に分離 |
| 2026-07-13 | v1.6 | レビュー指摘8項目対応。B14 arrivalSeq/enqueueTicket_分離、B13 Phase2/3再構成、B17 StoredConfig、B08 ShutdownTag API 等 |
| 2026-07-14 | v1.7 | 未確定事項の網羅的調査と確定。B01 VST3経路実態確認、dryBypassCapacity統合判断、B14 HealthMonitor現状確認、B13 Gardner/Wefers文献確証、computeMaxGainDb/residualRiskDb スコープ確定 |
| 2026-07-14 | v1.8 | レビュー指摘8項目対応。B14 Global FIFO→partial ordering へ説明修正、B20 static AVX定数→局所変数、B08 ShutdownTag→ShutdownCapability独立ヘッダ分離、B13 Phase2 事実上不可能の判定追加、B01 Standalone表現緩和 |
| 2026-07-14 | v1.9 | B14 Queue API `dequeueOne()` 変更 (vector生成を上位レイヤへ)、4層アーキテクチャ明確化、B08 Capability→ShutdownDeletePolicy dispatch、B13 Phase2削除 (2 Phase構成に整理)、B01 static_assert 追加、B20 horizontalMax改善案 |
| 2026-07-14 | v1.10 | 最終調査確定: RuntimeHealthMonitor EVENT コード未使用 (dead code) 確認、setProcessingOrder sendChangeMessage 欠落確定、ISRRetireOverflowRing onHealthEvent 未実装確認。残課題ゼロ。 |
| 2026-07-14 | v1.11 | B14 arrivalSeq を Queue → RetireScheduler (dequeue時 local採番) に移動、stable_sort → 3-way bucket dispatch、initQueue relaxed、ISRRetireOverflowRing 関係性明確化。B08 ShutdownDeletePolicy → ReleaseContext 一本化。B13 DelayAlignment 構造体分離、Reset outputDelaySamples 保持、delayLineCapacity = outputDelay + maxBlockSize。B18 thread safety 前提明記。B03 span 寿命図追加。B20 anonymous namespace + constexpr 導出。B01 static_assert 方式修正。 |
| 2026-07-14 | v1.12 | 最終調整: B14 ScheduledRetireIntent 導入、dequeueFallback API 追加、constexpr kMaxProducerSpin、fallbackMutex_ 非公開化、wrap 説明を sequence difference に修正。B13 delayLineWritePos 削除、capacity roundUp 追加。B01 static_assert 最終形 (is_same_v) に確定、TODO→Appendix N 移動。B03 図→Appendix N 移動。B08 ReleaseContext→releaseShutdown() 最小修正。B20 static 説明簡略化。 |
| 2026-07-14 | v1.13 | B14 RetireBatch を ScheduledRetireIntent に統一 (arrivalSeq を最後まで保持)。Queue SIZE 制約 (任意整数, power-of-two 非必須) を明記。fallback head/tail O(1) 管理に仕様化。B08 releaseShutdown() friend 宣言を明記。 |
| 2026-07-14 | v1.14 | B14 fallback enqueue tail 計算修正 (head+count%capacity → 上書き事故防止)。droppedIntentCount 責務明確化 (Queue=保持, HealthMonitor=監視)。arrivalSeq が FIFO 保証でないことを強調 (publish order)。B01 static_assert sizeof 比較に強化。B13 jassert + #error guard 追加。B08 releaseShutdown→releaseImmediatelyForShutdown 命名改善。 |
| 2026-07-14 | v1.15 | B14 fallback starvation 防止: main:fallback = 8:1 fair scheduling 導入。B13 delayLineCapacity に +4 blocks 安全余裕追加。B01 offsetof によるメンバ配置検証に方式変更。B08 ShutdownToken を将来改善候補として注記。 |
| 2026-07-14 | v1.16 | B14 arrivalSeq コメントを「publish completion order」に統一 (FIFO誤解防止)。B13 safety margin を block依存→kDelaySafetySamples=1024 の設計定数に変更。B01 offsetof 検証削除 (非同名列の比較は無意味)、is_same_v コメントに整理。 |
| 2026-07-14 | v1.17 | **B14 P0修正**: fallback 時に tombstone (isValid=false) を slot へ publish してから fallback queue に追加。Vyukov Queue の hole (ticket 消失→Consumer deadlock) を防止。dequeueOne で tombstone スキップ。sequence+SIZE の cycle 番号エンコーディング解説追加。B13 resetDelayAlignment() 専用関数追加。B01 static_assert を有効化 (コメントアウト解除)。 |
| 2026-07-14 | v1.18 | B14 dequeueOne tombstone スキップを再帰→whileループに変更 (大量 tombstone 時の深い再帰回避)。fallbackQueuePeak_ 追加 (診断用ピークカウンタ)。B13 Add/Get 呼び出し粒度非対称性への対応を設計に追加。 |
| 2026-07-14 | v1.19 | B14 payload→sequence の順序保証コメント追加。fallbackQueuePeak_ を publishAtomic→CAS ループに修正 (複数 Producer の競合対策)。kMainToFallbackRatio を Policy 層の責務として明確化。RetireBatch に reserve/small_vector 推奨注記。B13 readSeq を唯一の Authority として単純化。 |
| 2026-07-14 | v1.20 | B14 approxQueueDepth() 追加 (Queue Pressure 診断用)。fallbackQueuePeak_ 更新を CAS→mutex内単純maxに変更 (mutex保護下では不要)。RetireIntent にサイズ制約注記 (trivially copyable, ≤64bytes)。arrivalSeq「publish completion order」→「dequeue observation order」に修正 (Consumer観測時間差を反映)。B13 delayLineCapacity を kDelaySafetySamples=1024 (経験則)→maxWriterAdvance=3×maxBlockSize (理論式) に変更。 |
| 2026-07-14 | v1.21 | B14 emitRetireIntent を事前 fallback (ticket取得前にfallback空き確認) に変更 → tombstone 完全不要化。ScheduledRetireIntent.arrivalSeq→observationOrder 改名推奨。B13 Availability region (writeCommitSeq/writeIssueSeq) を設計に追加。B08 friend→ShutdownToken 採用 (Authority Singularization 適合)。B18 delayed resolve を将来改善として注記。 |
| 2026-07-14 | v1.22 | B14 emitRetireIntent 制御フロー矛盾修正: MPSC Queue を通常経路→fallback を例外経路に変更 (tombstone 再導入)。B13 Availability region を Appendix N.3 へ移動 (実装直結でない設計補足)。 |
| 2026-07-14 | v1.23 | B08 ~CacheMap() 二重実装解消 + runtime getShutdownPhase() 分岐追加 (通常運用時 EBR 経由、shutdown 時のみ即時 delete)。B01 命名一致 (bypassFadeGain→bypassFadeGainFloat, bypassed→bypassedFloat)。B13 l.delayLineBuf 残骸削除 + 解放パス一覧重複解消。 |
| 2026-07-14 | v1.24 | B13 DelayAlignment→DelayAlignmentConfig(Immutable)+DelayAlignmentState(Mutable)分離、readSeq無音期間説明+State Mutation Authority明記。B14 batch.reserve(128)+stable_sort observationOrder比較キー追加+HealthMonitor EVENT dead code削除方針。B08 ShutdownToken copy禁止(move-only)。 |
| 2026-07-14 | v1.25 | B14 isValid廃止→dspSlot==UINT32_MAX sentinel (sequence唯一Authority)。resetDelayAlignment delayAlign→delayState修正。B08 release(RestirePolicy{Epoch|Immediate}) 統一API + ShutdownToken廃止。B18 quarantineSlot非公開化(Manager単一Authority)。B13 Partition Boundary Test追加。 |
| 2026-07-14 | **v1.26** | B13 delayReadCursor 単一 Authority + DelayAlignmentRuntime命名、overrun検出追加(writeCursor-readCursor>capacity)、#error→RuntimeBuilder validation。B14 observationOrder→publishGeneration。B13 freeAll/Add/Getコード delayAlign→delayRun統一。 |
| 2026-07-14 | **v1.27** | B13 partition sequence管理(writePartition/readPartition追加、FFT LayerとAudio callbackのタイミング保証)。B14 Memory Orderingテーブル追加(acquire/release/relaxed/acq_rel完全明記)。B08 RetirePolicy利用条件(Design Rule)明記。B18 Design Rule「DSPCore解放はRetireRuntimeのみ」明文化。B20 SIMD数値一致性テスト(AVX2 vs Scalar)追加。 |
| 2026-07-14 | **v1.28** | B13 isDelayCompatibleWith比較条件(outputDelaySamples/partSize/layerCount)具体化＋遅延算出式明記。B08 release(RetirePolicy::Immediate) ~CacheMap()専用と明記。B18 Generationライフサイクル図(slot/generation→quarantine→retire→epoch→delete→次generation)追加。Appendix G クロスカッティングに注記追加(Audio結果は非独立)。Appendix E 共通ベンチマーク条件追加。Appendix F Producer/Crossfade/Retire Stormテスト追加。Appendix H ★評価→ISR設計チェックリストに変更(客観的)。 |
| 2026-07-14 | **v1.29** | B14 Vyukov MPSC不変条件(Invariant)4項目(Publish Visibility/Slot Ownership/ABA Safety/Happens-Before Chain)明文化。B13 IR切替(SetImpulse)時delayLineBuf clear追加。B17 clone invariant(init直後と等価)明記。B08 release(RetirePolicy::Immediate) Shutdownのみ+Runtime publish後禁止。B18 Quarantine状態遷移図(Registry→Quarantine→RetireQueue→Epoch→Destroy)追加。Appendix G依存グラフ追加。Appendix H 6軸チェックリスト(Authority/Frozen/RT Safe/Lifetime/Memory Order/Shutdown)に拡張。 |
| 2026-07-14 | **v1.30** | 第一次・第二次監査結果を反映。`remaining_bugs.md` 新設（10項目→設計差分分離後7項目+調査中1項目）。RB-05 RingBuffer容量不足を数理解析で確認（capacityにl.partSize不足→修正案: `prevLayerTotalSamples + l.partSize + 2×maxBlockSize`）。RB-11 (`setProcessingOrder` sendChangeMessage欠落) を確定バグとして追加。RB-08 MT-NUPC-03 Debug異常終了の原因解析を追加。Appendix K に RB-05/RB-08 調査結果追記。設計差分4件(RB-04/06/09/10)をバグリストから除外。 |
| 2026-07-14 | **v1.31** | RB-05 を実コード引用に全面書き換え。`ConvolverProcessor.Runtime.cpp:1154-1155` (Add→Get順序) + `delayLineWrite()` (1727-1736, ガードなしラップ) + `outputDelaySamples < 5×blockSize` 発現条件。`bug_final_report.md` Appendix K も連動更新。 |
| 2026-07-15 | **v1.32** | RB-05 の証拠を強化。`delayLineWrite()` の呼出しが **1箇所のみ (line 1630)**、第3引数が **常に `l.partSize`** であることを ripgrep 全コードベース検索で確認し追記。`remaining_bugs.md` v2.3 連動。 |
| 2026-07-15 | **v1.33** | RB-05 `prevLayerTotalSamples` をコード追跡で確認。`cfgs[li].len` は実 IR サンプル数 (nextPowerOfTwo 非使用) であることを実コード引用で確定。`bug_final_report.md` / `remaining_bugs.md` v2.4 連動。 |
| 2026-07-15 | **v1.34** | Appendix K 誤記修正: `RuntimeHealthMonitor` EVENT コード全18種が実際には全件使用中であることを確認 (Serena MCP 検索、rg 0 hits は rg `-g *.cpp` の範囲外)。ISRRetireOverflowRing `onHealthEvent` コメントの評価も正確化 (間接的だが呼出元チェーンを通じて実装済み)。Appendix N.1/ N.3 設計補足は将来改善として維持。RB-08 調査継続。 |
| 2026-07-15 | **v1.35** | RB-08 原因特定・修正完了。`measureLayerDelays()` の buffer size 計算誤り (`totalOutputSamples = irLen*2` が blockSize 倍数でない → 範囲外アクセス)。修正: blockSize 倍数に切り上げ + 防御的分析上限。Debug 全テスト通過確認。`remaining_bugs.md` v2.8 連動。残課題ゼロ。 |

## Appendix K: 調査確定サマリ

| 調査項目 | 調査方法 | 確定結果 |
|---------|---------|---------|
| B01: VST3 Float/Double 交互呼出の有無 | JUCE AudioEngineProcessor.cpp routing 確認 (`MainWindow.cpp:282` で `setDoublePrecisionProcessing(true)`) + VST3 仕様調査 | **交互呼出は発生しない**。JUCE は `setDoublePrecisionProcessing()` で Float/Double を確定し、Session 中は単一経路のみ。Float経路(`DSPCore::process()`)が bypass blend 欠落 → B01 は有効。TODO は維持。 |
| B01: dryBypassCapacity 統合の可否 | EQProcessor (`EQProcessor.h:638`) / DSPCore (`AudioEngine.h:977`) の独立変数確認 | **統合不可**。EQ/DSPCore/Float版は独立した処理層。ただしテンプレートヘルパー `ensureDryBypassBuffer<T>()` 抽出は将来検討可。 |
| B14: droppedIntentCount 監視状態 | `AudioEngine.Retire.cpp:124` + `ISRRetireOverflowRing.h:78` | **既に監視中**。`droppedIntentCount()` は Message Thread で定期取得。`RuntimeHealthMonitor.h` との接続はコメントレベルのみで未実装。 |
| B13: Gardner/Garcia/Wefers NUPC 遅延補償の理論的根拠 | AES文献 + Wefers PhD Thesis (2015) | **確定**。非一様分割畳み込みでは後段レイヤー(L1,L2)が先頭レイヤー(L0)の IR 長だけ遅延する。Wefers FDL はリングバッファによる Phase 3 方式に相当。 |
| computeMaxGainDb / residualRiskDb / IRState | コードベース全数検索 (0 hits) | **work72 GainStaging スコープ**。本バグ修正とは独立した設計項目のため追跡対象外。Appendix B.1 にスコープ明記済み。 |
| `setProcessingOrder` submit/apply 順序 | `AudioEngine.Parameters.cpp:268-275` 再確認 | **確定バグ** (RB-11)。submit→apply→(欠落中) の順序で、sendChangeMessage が不足。同等の setter (`setEqBypassRequested` / `setConvolverBypassRequested`) は全件末尾に `sendChangeMessage()` を持つ。`doc/work69/remaining_bugs.md` で管理。 |
| B14: arrivalSeq による Global FIFO 保証の妥当性 | Vyukov MPSC Queue 理論解析。`dequeuePendingRetireIntents()` コード検証 | **保証されない**。Producer の arrivalSeq 取得順 ≠ publish 完了順のため、stable_sort は「同バッチ内で取得できた Intent」に対してのみ有効。説明を「partial ordering within batch」に修正。 |
| B20: `static const __m256d` の RT 安全性 | C++ 規約 (static initialization) 分析。`__m256d` は constexpr 非対応 | **危険**。関数スコープの `static const __m256d` は guard variable によるスレッドセーフ初期化を発生させる。局所変数に変更し、`_mm256_set1_pd` を毎回呼ぶ方式に修正 (1命令で実質無視できるコスト)。 |
| B08: `ShutdownTag` の循環依存リスク | `RefCountedDeferred.h` include 一覧確認 (`IEpochProvider.h`, `DspNumericPolicy.h`, `AtomicAccess.h` — AudioEngine.h 非依存) | **要修正**。`RefCountedDeferred.h` が `AudioEngine.h` を include すると循環依存が発生する。`ShutdownCapability.h` 独立ヘッダに分離し、private ctor + friend class AudioEngine で Capability を実現。 |
| B13: Phase 2 (tailOutputBuf のみ) の実現可能性 | `MKLNonUniformConvolver.h:345` tailOutputBuf 容量確認 = partSize。典型的 NUPC 構成との比較 | **事実上不可能**。tailOutputBuf は各層の partSize のみ。典型的な IR (L0.numPartsIR >> 8) では outputDelaySamples >> tailOutputBuf。Phase 2 をスキップし直接 Phase 3 リングバッファ方式を推奨。 |
| B14: `RuntimeHealthMonitor` イベントコードの使用状況 | `RuntimeHealthMonitor.h:44-64` 定義の `EVENT_RETIRE_STALL=1001` 他16コード。`RuntimeHealthMonitor.cpp` 全コード確認 (Serena MCP search)。 | **全18コード使用中 (dead code ではない)**。`RuntimeHealthMonitor.cpp` 内で各状態遷移に応じて適切な EVENT コードが emit されている。`AudioEngine.Timer.cpp:onHealthEvent()` (line 1449) では `EVENT_READER_SLOT_USAGE`、`EVENT_PUBLICATION_STALL`、`EVENT_RETIRE_STALL`、`EVENT_RETIRE_AGE_CRITICAL`、`EVENT_CROSSFADE_TIMEOUT`、`EVENT_READER_STUCK` の6コードをハンドリングしている。残りのコードは RuntimeHealthMonitor → PolicyEngine 経由で処理される。Appendix K の旧記述「定義のみで未使用」は誤り。 |
| B0: `setProcessingOrder` `sendChangeMessage` 欠落 (確定) | `AudioEngine.Parameters.cpp:268-275` 再確認。比較対象: `setEqBypassRequested:160-168` と `setConvolverBypassRequested:179-188` は末尾に `sendChangeMessage()` あり。 | **確定バグ (RB-11)**。`setProcessingOrder` は `publishAtomic → submitRebuildIntent → applyDefaultsForCurrentMode` の後に `sendChangeMessage()` が不足。同等の setter は全件 `sendChangeMessage()` を持つ。`doc/work69/remaining_bugs.md` で管理。 |
| B14: `ISRRetireOverflowRing` `onHealthEvent` コメントの実態 | `ISRRetireOverflowRing.h:78` のコメント。`AudioEngine.Timer.cpp:1449` の `onHealthEvent()` 実装を確認。 | **コメントは正確だが間接的**。`ISRRetireOverflowRing::tryPush()` の呼出元 (`emitRetireIntent` / `drainOverflowRing`) で `droppedIntentCount_++` と `overflowStartTimestamp_` 設定が行われ、HealthMonitor のコールバックがトリガーされる。Ring 側が直接 `onHealthEvent` を呼ぶわけではないが、呼出元チェーンを通じて最終的に `AudioEngine::onHealthEvent()` (Timer.cpp:1449) が呼ばれる。現在の実装で不足はない。 |
| RB-05: B13 delayLineBuf 自己上書き — 実コード引用5層で確定 | `MKLNonUniformConvolver.cpp` を ripgrep 全コード検索。(1) Add→Get順序 (ConvolverProcessor.Runtime.cpp:1154-1155)。(2) delayLineWrite(line 1727-1736): n>capacity でもラップ上書き、jassert なし。(3) `prevLayerTotalSamples += cfgs[li].len` (line 1120): cfgs[li].len は実 IR サンプル数 (l0Len=min(irLen,...), l1Len=irLen-l0Len..., nextPowerOfTwo非使用)。(4) delayLineWrite呼出しは1箇所(line 1630)、第3引数は常に l.partSize。(5) L1 partSize=4096 > capacity=2048 (outputDelaySamples=512の時) で自己上書き確定。発現条件: outputDelaySamples < 5×blockSize。 | **コード事実5層により自己上書きを確認**。修正案: `capacity = prevLayerTotalSamples + l.partSize + m_maxBlockSize`。 |
| RB-08: MT-NUPC-03 Debug 異常終了の原因解析と修正 | `measureLayerDelays()` の出力バッファサイズ計算を解析。`totalOutputSamples = irLength * 2` は blockSize の倍数とは限らず (irLen=2047→4094)，while ループ最終反復で `totalProcessed` (4096) が `totalOutputSamples` (4094) を超過 → ピーク分析ループで output[4094], output[4095] に範囲外アクセス。Debug ビルド `/RTC1` で検出。 | **解決済 ✅**。修正: (1) バッファサイズを blockSize の倍数に切り上げ。 (2) 分析ループ上限を `min(output.size(), totalProcessed)` で制限。Debug 全テスト通過確認済。 |

## Appendix N: 設計補足

### N.1 FloatRampState / DoubleRampState 分離 (将来検討)

現状は Float/Double で `RampRuntimeState` を共有しているが、DSPCore 単位のため
実害はない。将来的な設計選択肢として以下を記録する:

```cpp
// TODO: Split Float/Double RampRuntimeState if simultaneous processFloat/processDouble
//       becomes supported. Currently shared via DSPCore-level RampRuntimeState.
```

### N.2 `std::span<const double>` 寿命ライフサイクル (B03)

```
Generation 開始
  │
  ├─ mappedPopulation 生成 (Main Worker が所有)
  │     │
  │     ├─ vdTanh(mappedPopulation) → sharedTanb (共有、1回のみ)
  │     │
  │     ├─ Worker 1 開始 → span<const double>(sharedTanh)
  │     ├─ Worker 2 開始 → span<const double>(sharedTanh)
  │     ├─ ...
  │     └─ Worker N 開始
  │           │
  │           └─ join (全 Worker 完了)
  │
  ├─ mappedPopulation は join 完了まで維持 → sharedTanh 破棄
  └─ 次 Generation → 新しい mappedPopulation で再計算
```

### N.3 Availability Region (B13 DelayAlignment 将来拡張)

NUPC の Add()/Get() 呼び出し粒度の非対称性および IFFT 完了タイミングのずれに対応するため、
将来的に以下を検討する:

```cpp
struct DelayAlignment {
    int outputDelaySamples = 0;
    int delayLineCapacity = 0;
    double* delayLineBuf = nullptr;
    uint64_t delayWriteCommitSeq = 0;  // Add() 完了時点 (IFFT 完了 = 読み取り可能)
    uint64_t delayWriteIssueSeq = 0;   // Add() 発行時点 (書き込み予約)
    uint64_t delayConsumeSeq = 0;
};
```

現状の単一 `delayWriteSeq` でも実用上は問題ないが、長 IR + 高負荷環境では
`delayWriteCommitSeq` / `delayWriteIssueSeq` の分離により
「書き込まれたがまだ利用可能でない領域」を正確に管理できる。
本バグ修正では単一 `delayWriteSeq` を採用し、必要に応じて将来拡張する。

## Appendix O: 参考情報

### L.1 主要文献

| 文献 | 引用目的 |
|------|---------|
| Bristow-Johnson, R. "Audio EQ Cookbook" (W3C 2021) | Peaking EQ 係数の正当性確認 |
| Harris, F.J. "On the use of Windows for Harmonic Analysis with the DFT" (Proc. IEEE, 1978) | Tukey窓 PSL |
| Bencina, R. "Real-time audio programming 101" (2011) | RT-safe 設計原則 |
| Smith III, J.O. "Physical Audio Signal Processing" (2010) | IR L2正規化 |
| Farina, A. "Real-Time Partitioned Convolution" (Mohonk 2001) | 畳み込みリバーブ L2 基準 |
| Gardner, W.G. "Efficient Convolution without Input-Output Delay" (AES 1995) | NUPC 遅延補償の理論的基礎 |
| Garcia, G. "Non-uniform partitioned convolution" (AES 2002) | NUPC 実装手法 |
| Wefers, F. "Partitioned Convolution Algorithms for Real-Time Auralization" (PhD Thesis, RWTH Aachen 2015) | FDL 遅延補償方式 (Phase 3 の理論的裏付け) |

### L.2 テスト計画対応

- UT-01〜UT-08: 単体テスト
- IT-01〜IT-07: 統合テスト
- GC-01〜GC-06: コントラクトテスト
- MT-01〜MT-10: 手動テスト
- RT-01〜RT-04: リアルタイム安全性テスト
