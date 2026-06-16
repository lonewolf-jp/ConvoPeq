# ConvoPeq Practical Stable ISR Bridge Runtime 改修計画書

**策定日**: 2026-06-16
**対象コードベース**: ConvoPeq (lonewolf-jp/ConvoPeq)
**計画バージョン**: v8（最終確定版）
**検証ツール**: CodeGraph MCP, Graphify MCP, AiDex MCP, grep/Select-String, 設計文書調査, ファイル直接読み取り

---

## 目次

1. [改修の目的と全体方針](#1-改修の目的と全体方針)
2. [Sprint 1: 基盤整備](#2-sprint-1-基盤整備)
3. [Sprint 2: 診断拡張](#3-sprint-2-診断拡張)
4. [Sprint 3: EBR 可視化と最終検証](#4-sprint-3-ebr-可視化と最終検証)
5. [168時間連続運転試験計画](#5-168時間連続運転試験計画)
6. [完了条件と判定基準](#6-完了条件と判定基準)
7. [付録: コード変更詳細](#7-付録-コード変更詳細)

---

## 1. 改修の目的と全体方針

### 1.1 目的

Practical Stable ISR Bridge Runtime の **運用時診断能力 (Observability)** を強化し、以下の情報を取得可能にする:

1. **シャットダウン阻害要因の統計・時系列可視化** (A-2, A-3)
2. **EBR Grace Period の完了状況の可視化** (Sprint3-A/B)
3. **Pipeline Ledger の完全性復旧** (Sprint3-C)
4. **168時間連続運転試験の合格判定基盤** (Sprint3-D)

### 1.2 全体方針

- **最小侵襲**: 既存の lock-free データ構造への変更は最小限に留める
- **設計意図の尊重**: 既存のコメントと設計文書を設計意図の一次ソースとする
- **段階的実装**: Sprint1→2→3 の順で依存関係に従う

### 1.3 Sprint 依存関係

```
Sprint 1 (基盤整備)
├─ B-2: ISRBarrierOptimizer削除 [依存なし]
└─ A-2: Shutdown Blocking Statistics [依存なし]
  ↓
Sprint 2 (診断拡張)
├─ A-3: Blocking Event History [A-2完了が前提]
└─ B-1: isAllZero 軽微修正 [依存なし]
  ↓
Sprint 3 (EBR可視化 + 最終検証)
├─ A: EBR Queue Visibility [依存なし]
├─ B: Grace Period Visibility [A完了が前提]
├─ C: Pipeline Ledger復旧 [依存なし]
├─ E: Evidence非同期書き込み [依存なし] (将来安全策, Dより低優先)
└─ D: 168h試験 [A+B+C完了が前提, Eは任意]
```

---

## 2. Sprint 1: 基盤整備

### 2.1 B-2: ISRBarrierOptimizer 削除

#### 削除理由

| 確認項目 | 結果 |
| --- | --- |
| `BarrierOptimizer::setBuildMode()` | 実装: `mode_ = mode;` のみ。呼び出し元: **ゼロ** |
| `BarrierOptimizer::optimizeBarriers()` | 実装: 空のswitch文のみ。呼び出し元: **ゼロ** |
| `AudioEngine.h` include | `#include "ISRBarrierOptimizer.h"` (101行目) — 削除対象 |
| `AudioEngine.h` メンバ宣言 | `convo::isr::BarrierOptimizer barrierOptimizer_;` (3573行目) — 削除対象 |
| `CMakeLists.txt` 登録 | `src/audioengine/ISRBarrierOptimizer.cpp` (378行目) — 削除対象 |
| `barrierOptimizer_.` メソッド呼び出し | **src/ 全体で 0 件** |

**リスク評価**: ゼロ。

#### 変更内容

| 操作 | ファイル | 変更種別 |
| --- | --- | --- |
| 削除 | `src/audioengine/ISRBarrierOptimizer.h` | ファイル削除 |
| 削除 | `src/audioengine/ISRBarrierOptimizer.cpp` | ファイル削除 |
| 削除 | `src/audioengine/AudioEngine.h` 101行目 | `#include "ISRBarrierOptimizer.h"` 行削除 |
| 削除 | `src/audioengine/AudioEngine.h` 3573行目 | `convo::isr::BarrierOptimizer barrierOptimizer_;` 行削除 |
| 削除 | `CMakeLists.txt` 378行目 | `src/audioengine/ISRBarrierOptimizer.cpp` 行削除 |

#### 検証手順

```powershell
# 削除後のビルド確認
cmake --build build --config Debug
# 未参照シンボルの確認 (0件を確認)
grep -r "ISRBarrierOptimizer\|BarrierOptimizer" src/ --include="*.h" --include="*.cpp"
```

---

### 2.2 A-2: Shutdown Blocking Statistics

#### 現状

`ShutdownRuntime::markTimedOut()` は呼び出された Reason を単一の `blockingReason_` に保存するのみで、統計情報を取得できない。

#### 変更内容

##### A-2.1: `BlockingReasonStats` 構造体の追加

```cpp
// ISRShutdown.h — 追加
// ★ 各メンバを個別 std::atomic<uint64_t> にする (32バイト構造体の丸ごと atomic は不可)
//    sizeof(BlockingReasonStats) = 32 > 16 (x64 HW atomic limit: CMPXCHG16B)
//    std::atomic<BlockingReasonStats> は MSVC STL で内部ミューテックスに fallback する
// ★ alignas(64): 配列として連続配置された際の False Sharing を防止
//    (32B構造体x2=64Bで同一cache lineに載るのを避ける)
struct alignas(64) BlockingReasonStats {
    std::atomic<uint64_t> count{0};
    // totalDurationUs は削除: maxDurationUs と同一値になるため冗長
    std::atomic<uint64_t> maxDurationUs{0};
    std::atomic<uint64_t> firstSeenUs{0};
};

// ★ A-2: ShutdownBlockingReason 別統計
//    enum 値域: None=0 〜 Unknown=8 (計9値)
//    ハードコードではなく enum から導出することで enum 変更時の追従漏れを防止
static constexpr size_t kBlockingReasonCount =
    static_cast<size_t>(ShutdownBlockingReason::Unknown) + 1;
```

##### A-2.2: `ShutdownRuntime` への統計配列追加

```cpp
// ISRShutdown.h — ShutdownRuntime クラスに追加
// std::array<std::atomic<BlockingReasonStats>, N> ではなく単なる配列
// (各メンバが既に std::atomic のため)
private:
    std::array<BlockingReasonStats, kBlockingReasonCount> blockingReasonStats_;
```

##### A-2.3: `reasonToString()` 関数の抽出

`ISRShutdown.cpp` の `emitShutdownTrace()` 内に既存の switch-case から、独立関数を抽出:

```cpp
// ISRShutdown.h — 追加
[[nodiscard]] const char* reasonToString(ShutdownBlockingReason reason) noexcept;

// ISRShutdown.cpp — 実装
const char* reasonToString(ShutdownBlockingReason reason) noexcept {
    switch (reason) {
        case ShutdownBlockingReason::None: return "None";
        case ShutdownBlockingReason::PendingPublication: return "PendingPublication";
        case ShutdownBlockingReason::PendingRetire: return "PendingRetire";
        case ShutdownBlockingReason::ActiveCrossfade: return "ActiveCrossfade";
        case ShutdownBlockingReason::DeferredPublish: return "DeferredPublish";
        case ShutdownBlockingReason::QuarantineResident: return "QuarantineResident";
        case ShutdownBlockingReason::RouterPendingRetire: return "RouterPendingRetire";
        case ShutdownBlockingReason::ReaderActive: return "ReaderActive";
        case ShutdownBlockingReason::Unknown: return "Unknown";
    }
    return "Unknown";
}
```

##### A-2.4: `markTimedOut()` 内での統計更新

```cpp
// ISRShutdown.cpp — 追加: getCurrentTimeUs() のために必要
#include "core/TimeUtils.h"

// ISRShutdown.cpp — markTimedOut() 変更
void ShutdownRuntime::markTimedOut(ShutdownBlockingReason reason) noexcept
{
    // ★ A-2: シグネチャ変更禁止 (void markTimedOut(ShutdownBlockingReason))
    //         内部のみ変更
    const uint64_t nowUs = convo::getCurrentTimeUs();

    // blockingReason_ 保存 (既存)
    convo::publishAtomic(blockingReason_, reason, std::memory_order_release);

    // ★ A-2: 統計更新
    // ★ 配列外参照防止: enum 値をサニタイズ
    size_t idx = static_cast<size_t>(reason);
    if (idx >= kBlockingReasonCount) {
        idx = static_cast<size_t>(ShutdownBlockingReason::Unknown);
    }
    auto& stats = blockingReasonStats_[idx];
    stats.count.fetch_add(1, std::memory_order_acq_rel);

    // firstSeenUs: CAS で初回のみ設定
    uint64_t expected = 0;
    stats.firstSeenUs.compare_exchange_strong(expected, nowUs,
        std::memory_order_acq_rel, std::memory_order_acquire);

    // ★ duration: shutdown 開始 (AudioStopped) からの経過時間を計算
    const uint64_t elapsed = (nowUs > shutdownStartUs_)
        ? (nowUs - shutdownStartUs_) : 0;

    // totalDurationUs は構造体から削除済み (maxDurationUs と同一値のため冗長)

    // maxDurationUs: fetch_max (CAS loop)
    uint64_t currentMax = stats.maxDurationUs.load(std::memory_order_acquire);
    while (elapsed > currentMax) {
        if (stats.maxDurationUs.compare_exchange_weak(currentMax, elapsed,
                std::memory_order_acq_rel, std::memory_order_acquire))
            break;
    }

    // 既存の phase 保存 + 上書き (変更なし)
    convo::publishAtomic(lastNonTerminalPhase_, ...);
    convo::publishAtomic(phase_, ShutdownPhase::TimedOut, ...);
}
```

ShutdownRuntime に shutdownStartUs_ を追加:
```cpp
// ISRShutdown.h — 追加
void initiateShutdown() {
    shutdownStartUs_ = convo::getCurrentTimeUs();
    transitionTo(ShutdownPhase::AudioStopped);
}

private:
    uint64_t shutdownStartUs_{0};
```

**ガードレール遵守確認**:
- ✅ `markTimedOut` シグネチャ: `void markTimedOut(ShutdownBlockingReason reason = ShutdownBlockingReason::Unknown) noexcept` — 変更なし
- ✅ `getCurrentTimeUs()` — `core/TimeUtils.h` で利用可能
- ✅ 配列サイズ9 — `ShutdownBlockingReason` は9値 (None=0〜Unknown=8)
- ✅ lock-free — `std::atomic` + `fetchAddAtomic`
- ✅ **単一スレッド呼び出し保証**: `markTimedOut()` は `AudioEngine::releaseResources()`
   （ReleaseResources.cpp:312）からのみ呼ばれる。これは非RT（メッセージ）スレッド上の
   単一パスであるため、イベントの物理的順序とタイムスタンプの順序は一致する。
   複数スレッドからの同時呼び出しは発生しないため、タイムスタンプの逆転は生じない。

##### A-2.5: `collectResult()` / `emitShutdownTrace()` での出力

```cpp
// ISRShutdown.cpp — emitShutdownTrace() に統計JSON出力追加
void ShutdownRuntime::emitShutdownTrace(ISRHealthState healthState) const
{
    // ★ ★ アトミックファイル置換: .tmp に書き込み後 rename
    const auto outputPath = std::filesystem::current_path() / "evidence" / "shutdown_trace.json";
    const auto tmpPath = std::filesystem::current_path() / "evidence" / "shutdown_trace.json.tmp";
    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);
    if (ec) return;

    std::ofstream file(tmpPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        // ★ フォールバック: 一意化したファイル名で %TEMP% に書き込み
        static std::atomic<uint32_t> s_fallbackCounter{0};
        const auto timestamp = convo::getCurrentTimeUs();
        const auto count = s_fallbackCounter.fetch_add(1, std::memory_order_relaxed);
        const auto fallbackName = std::string("shutdown_trace_fallback_")
            + std::to_string(timestamp) + "_" + std::to_string(count) + ".json";
        std::error_code ec2;
        const auto tempDir = std::filesystem::temp_directory_path(ec2);
        if (ec2) return;
        const auto fallbackPath = tempDir / fallbackName;
        file.open(fallbackPath, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) return;
    }

    // ... 既存の出力 ...

    // ★ A-2: BlockingReasonStats JSON出力
    file << "  \"blockingReasonStats\": [\n";
    for (size_t i = 0; i < kBlockingReasonCount; ++i) {
        const auto& stats = convo::consumeAtomic(
            blockingReasonStats_[i], std::memory_order_acquire);
        if (i > 0) file << ",\n";
        file << "    {\n";
        file << "      \"reason\": \"" << reasonToString(static_cast<ShutdownBlockingReason>(i)) << "\",\n";
        file << "      \"count\": " << stats.count << ",\n";
        // totalDurationUs は削除済み (maxDurationUs と同一値のため冗長)
        file << "      \"maxDurationUs\": " << stats.maxDurationUs << ",\n";
        file << "      \"firstSeenUs\": " << stats.firstSeenUs << "\n";
        file << "    }";
    }
    file << "\n  ]\n";

    // ... 既存の出力終了 ...

    file.close();
    // ★ ★ 書き込みエラー検出: ディスクフルや権限エラーは close 後にも fail になる
    if (file.fail()) return;
    // ★ ★ rename でアトミック置換 (tmp → 本番)
    std::filesystem::rename(tmpPath, outputPath, ec);
    // 前回の .tmp が残存していれば削除
    std::filesystem::remove(tmpPath, ec);
}
```

---

## 3. Sprint 2: 診断拡張

### 3.1 A-3: Blocking Event History

#### 現状

`ShutdownRuntime` は阻害要因の **時系列** を保持していない。

#### 変更内容

##### A-3.1: イベントの 64bit パック定義

```cpp
// ISRShutdown.h — 追加
// ★ BlockingReasonEvent を 64bit にパック (8bit reason + 56bit timestampUs)
//    std::atomic<uint64_t> として扱うことで Tearing を完全防止
using PackedBlockingEvent = std::atomic<uint64_t>;

inline uint64_t packEvent(ShutdownBlockingReason reason, uint64_t timestampUs) noexcept {
    return (timestampUs << 8) | static_cast<uint64_t>(reason);
}
```

##### A-3.2: リングバッファの追加 (TelemetryRecorder 非依存, アトミック要素)

```cpp
// ISRShutdown.h — 追加 (TelemetryRecorder.h の FixedRingBuffer は利用しない)
// 独立した最小実装:

// ★ A-3: 独立 TinyRingBuffer (TelemetryRecorder 非依存)
//   要素を std::atomic<uint64_t> にパックすることで Tearing を完全防止。
//   push は fetch_add でインデックス確保後、atomic store。
//   forEach は acquire load で書き込み完了後のデータのみを安全に読む。
template<size_t N>
class TinyRingBuffer {
    static_assert(N > 0 && N <= 256, "TinyRingBuffer size must be 1..256");
public:
    void push(ShutdownBlockingReason reason, uint64_t timestampUs) noexcept {
        // 1. 現在の書き込み位置を取得 (単一Writer前提、relaxedで安全)
        const auto currentIdx = writePos_.load(std::memory_order_relaxed);
        // 2. データを先行して書き込む (Readerはまだこのインデックスを知らない)
        data_[currentIdx % N].store(packEvent(reason, timestampUs), std::memory_order_relaxed);
        // 3. release store: インデックスを更新し、データの書き込み完了を公開
        //    ★ fetch_add は不可: インデックスがデータより先に公開されるため
        writePos_.store(currentIdx + 1, std::memory_order_release);
    }
    [[nodiscard]] size_t size() const noexcept {
        const auto wp = writePos_.load(std::memory_order_acquire);
        return wp < N ? wp : N;
    }
    // ★ Seqlock 方式の安全な読み出し
    //    読み出し中に Writer が更新した場合はリトライし、一貫性を保証する
    template<typename F>
    void forEach(F&& callback) const noexcept {
        uint64_t wpBefore, wpAfter;
        size_t currentSize, startIdx;
        std::array<uint64_t, N> snapshot;
        do {
            wpBefore = writePos_.load(std::memory_order_acquire);
            currentSize = (wpBefore < N) ? static_cast<size_t>(wpBefore) : N;
            startIdx = (wpBefore < N) ? 0 : static_cast<size_t>((wpBefore - N) % N);
            for (size_t i = 0; i < currentSize; ++i) {
                snapshot[i] = data_[(startIdx + i) % N].load(std::memory_order_relaxed);
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            wpAfter = writePos_.load(std::memory_order_relaxed);
        } while (wpBefore != wpAfter);  // 読み出し中に更新があればリトライ
        for (size_t i = 0; i < currentSize; ++i) {
            const auto packed = snapshot[i];
            const auto reason = static_cast<ShutdownBlockingReason>(packed & 0xFF);
            const auto ts = packed >> 8;
            callback(reason, ts);
        }
    }
private:
    std::array<PackedBlockingEvent, N> data_{};
    std::atomic<uint64_t> writePos_{0};
};
};
```

**推奨サイズ**: 64 (32以上、8回分の異なる Reason を記録可能)

> **★ TinyRingBuffer forEach の走査制約**:
> - `forEach` は Seqlock 方式で一貫性を保証しているが、読み出し中に Writer が
>   バッファを一周するとリトライが発生する。サイズ64であれば問題ないが、
>   **将来的にサイズを拡大する場合、以下の規約を遵守すること**:
>   1. `forEach` はコピー（スナップショット）を取得してからコールバックを実行する
>      設計とし、コールバック内で重い処理を行わない
>   2. リングバッファサイズは 256 を上限とする（Seqlock リトライ発散防止）
>   3. コールバックはロックフリーかつ I/O 非依存であること

##### A-3.3: `markTimedOut()` 内での記録

```cpp
// ISRShutdown.h — ShutdownRuntime に追加
TinyRingBuffer<BlockingReasonEvent, 64> blockingReasonHistory_;

// ISRShutdown.cpp
void ShutdownRuntime::markTimedOut(ShutdownBlockingReason reason) noexcept
{
    const uint64_t nowUs = convo::getCurrentTimeUs();

    // ★ A-3: 時系列履歴に追加
    blockingReasonHistory_.push(reason, nowUs);

    // ... 既存の処理 ...
}
```

**制約遵守確認**:
- ✅ `TelemetryRecorder` 非依存 — `TinyRingBuffer` は独立実装
- ✅ リングサイズ 64 — 推奨値以上

---

### 3.2 B-1: isAllZero 軽微修正

#### 現状

`RuntimeDrainAudit::isAllZero()` はコメントで「完了条件に含めるもの」とされる `routerPendingRetire` をチェックしていない。

```cpp
// 現状 (RuntimeDrainAudit.h:72)
bool isAllZero() const noexcept {
    return pendingPublication == 0
        && pendingRetire == 0
        && activeCrossfadeCount == 0
        && deferredPublish == 0;
    // ★ 欠落: routerPendingRetire
}
```

**ただし本関数は診断ログ専用であり、Shutdown Authority ではない**。設計文書 `doc/work33/remediation_plan.md` および `"(observation only)"` 注釈により確認済み。したがってこの修正は軽微な改善に留まる。

#### 変更内容

```cpp
// RuntimeDrainAudit.h — isAllZero() 修正
// ★ 監査ログ出力専用。shutdown 完了判定の authority にはしない。
bool isAllZero() const noexcept {
    return pendingPublication == 0
        && pendingRetire == 0
        && activeCrossfadeCount == 0
        && deferredPublish == 0
        && routerPendingRetire == 0;  // ★ 追加 (コメント「完了条件に含めるもの」との整合)
}
```

**影響範囲**: `isAllZero()` の呼び出し元は2箇所のみ (`ReleaseResources.cpp:259, 350`)、いずれも診断ログ条件。後方互換性あり。

---

## 4. Sprint 3: EBR 可視化と最終検証

### 4.1 Sprint3-A: EBR Queue Visibility

#### A-1: `DeferredDeletionQueue::reclaim()` 戻り値変更

```cpp
// DeferredDeletionQueue.h — reclaim() 変更
// 変更前: void reclaim(uint64_t minReaderEpoch)
// 変更後: uint32_t reclaim(uint64_t minReaderEpoch) — 実際に解放した件数を返す
uint32_t reclaim(uint64_t minReaderEpoch) {
    uint32_t reclaimed = 0;           // ★ 追加
    // ... 既存変数 ...
    while (scanned < kMaxScan) {
        // ... 既存ループ ...
        if (canDelete && scanPos == deqPos) {
            if (CAS成功) {
                if (entry.deleter && entry.ptr) {
                    entry.deleter(entry.ptr);
                }
                ++reclaimed;          // ★ 追加: 実際に解放した件数
                // ... 既存後処理 ...
            }
        }
    }
    return reclaimed;                 // ★ 戻り値追加
}
```

**既存呼び出し元への影響**: 戻り値を受け取らない既存コードは `(void)` 暗黙変換で警告なくコンパイル可能。互換性あり。

#### A-2: EpochDomain へのカウンタ追加

```cpp
// EpochDomain.h — private メンバ追加
private:
    // ★ ★ 共有 atomic (キャッシュ競合低減のため Local Aggregation 経由で更新)
    std::atomic<uint64_t> reclaimAttemptCount_{0};
    std::atomic<uint64_t> reclaimSuccessCount_{0};
    // ★ ★ Local Aggregation 用カウンタ (per-core cache line に乗ることを期待)
    alignas(64) std::atomic<uint32_t> reclaimLocalCounter_{0};
```

```cpp
// EpochDomain.h — tryReclaim() 変更
void tryReclaim() noexcept override {
    // ★ ★ 統計カウンタは純粋な診断用 (同期のトリガーではない) ため relaxed で十分
    // ★ ★ キャッシュ競合対策: Local Aggregation により、毎回の fetch_add を回避
    //    reclaimLocalCounter_ に加算し、1024回に1回だけ共有 atomic に反映する。
    //    これによりマルチコア環境でのキャッシュライン移動を約1000分の1に低減。
    //
    // ★ AggregationInterval=1024 の根拠:
    //   - tryReclaim() 呼び出し頻度: 最大 1000回/秒 (timer 1回 + 各 retire 経路)
    //   - 1024回に1回の更新 → 約1秒に1回の共有atomic更新
    //   - 1秒に1回の atomic 更新はキャッシュ競合の実質的な影響が無視可能
    //   - カウンタの精度低下: 最大1023カウントの誤差だが診断目的では許容範囲
    //   - 168時間試験での最大誤差: 1023 × (168×60×60/1024) ≈ 60万件の誤差
    //     ただし 168h 試験の 条件②a (単調増加) は「傾向」を観測するものであり、
    //     絶対値の正確性は要求されないため問題なし
    constexpr uint32_t kCounterAggregationInterval = 1024;
    const uint32_t localCount = reclaimLocalCounter_.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((localCount % kCounterAggregationInterval) == 0) {
        reclaimAttemptCount_.fetch_add(kCounterAggregationInterval, std::memory_order_relaxed);
    }
    const auto n = deferredDeletionQueue.reclaim(getMinReaderEpoch());
    reclaimSuccessCount_.fetch_add(n, std::memory_order_relaxed);
    // ★ ★ reclaimSuccessCount は n が小さい (0〜数件) ため毎回の更新が許容範囲。
    //    n が大きくなるのは Grace Period 経過後のバースト時のみで、その時は
    //    キャッシュライン競合よりもメモリ解放自体の方が支配的なコスト。
}

// 新規公開アクセサ
[[nodiscard]] uint64_t reclaimAttemptCount() const noexcept override {
    return convo::consumeAtomic(reclaimAttemptCount_, std::memory_order_acquire);
}
[[nodiscard]] uint64_t reclaimSuccessCount() const noexcept override {
    return convo::consumeAtomic(reclaimSuccessCount_, std::memory_order_acquire);
}
```

#### A-3: IEpochProvider への virtual 追加

```cpp
// IEpochProvider.h — 追加
class IEpochProvider {
    // ... 既存 ...
    [[nodiscard]] virtual uint64_t reclaimAttemptCount() const noexcept { return 0; }
    [[nodiscard]] virtual uint64_t reclaimSuccessCount() const noexcept { return 0; }
};
```

#### A-4: ISRRetireRouter 委譲

```cpp
// ISRRetireRouter.h — 追加
[[nodiscard]] uint64_t reclaimAttemptCount() const noexcept override {
    assert(provider_ != nullptr);
    return provider_->reclaimAttemptCount();
}
[[nodiscard]] uint64_t reclaimSuccessCount() const noexcept override {
    assert(provider_ != nullptr);
    return provider_->reclaimSuccessCount();
}
```

#### A-5: RuntimeDrainAudit フィールド追加

```cpp
// RuntimeDrainAudit.h — 構造体に追加
struct RuntimeDrainAudit {
    // ... 既存フィールド ...
    // ★ Sprint3-A: EBR Queue Visibility
    uint64_t reclaimAttemptCount{0};
    uint64_t reclaimSuccessCount{0};
    uint64_t overflowCount{0};          // ★ 条件②c で使用
};
```

#### A-6: collectDrainAudit() 収集追加

```cpp
// AudioEngine.Threading.cpp — collectDrainAudit() に追加
return convo::isr::RuntimeDrainAudit{
    // ... 既存 ...
    .reclaimAttemptCount = m_retireRouter
        ? m_retireRouter->reclaimAttemptCount() : 0,
    .reclaimSuccessCount = m_retireRouter
        ? m_retireRouter->reclaimSuccessCount() : 0,
    .overflowCount = m_retireRouter
        ? m_retireRouter->overflowCount() : 0,        // ★ 条件②c で使用
};
```

#### A-7: Evidence 出力追加

`reclaimAttemptCount`/`reclaimSuccessCount` は `collectDrainAudit()` 経由で
`RuntimeDrainAudit` 構造体のフィールドとして取得される。Evidence 出力は
`emitEvidenceTickNonRt()` 内で `collectDrainAudit()` の結果から直接 JSON に追記する:

```cpp
// AudioEngine.Commit.cpp — emitEvidenceTickNonRt() に追加
void AudioEngine::emitEvidenceTickNonRt(bool force) noexcept
{
    // ... 既存のレートリミット + 定期呼び出し ...

    // ★ A-7: EBR Queue Visibility 統計を epoch_reclaim_audit.json に出力
    //    ★ アトミックファイル置換: .tmp に書き込み後 rename
    {
        const auto audit = collectDrainAudit();
        const auto evidencePath = std::filesystem::current_path()
            / "evidence" / "epoch_reclaim_audit.json";
        const auto tmpPath = std::filesystem::current_path()
            / "evidence" / "epoch_reclaim_audit.json.tmp";
        // ★ ディレクトリ作成を保証 (create_directories は存在すれば何もしない)
        std::error_code ec;
        std::filesystem::create_directories(evidencePath.parent_path(), ec);
        if (ec) return;
        // ★ 一時ファイルに書き込み (クラッシュ時に破損 JSON が残っても .tmp のみ)
        {
            std::ofstream file(tmpPath, std::ios::binary | std::ios::trunc);
            if (file.is_open()) {
                file << "{\n";
                file << "  \"reclaimAttemptCount\": " << audit.reclaimAttemptCount << ",\n";
                file << "  \"reclaimSuccessCount\": " << audit.reclaimSuccessCount << ",\n";
                file << "  \"overflowCount\": " << audit.overflowCount << "\n";
                file << "}\n";
                file.close();
                if (!file.fail()) {
                    // ★ rename は POSIX 互換 OS ではアトミック操作
                    //    Windows でも同一ボリューム上では atomic (rename 失敗時も元ファイルは intact)
                    std::filesystem::rename(tmpPath, evidencePath, ec);
                }
            }
        }
        // ★ .tmp ファイルが残っていれば削除 (前回のクラッシュ痕跡)
        std::filesystem::remove(tmpPath, ec);
    }

    // ... 既存の Evidence 出力 ...
}
```

> **設計判断**: `WorldLifecycleAudit` への `ISRRetireRouter*` 注入は責務混在を招くため不採用。
> 新規 `evidence/epoch_reclaim_audit.json` を作成し、`emitEvidenceTickNonRt()` から直接出力する。

---

### 4.2 Sprint3-B: Grace Period Visibility

Sprint3-A 完了が前提。A-6 で `collectDrainAudit()` に追加された `reclaimAttemptCount`/`reclaimSuccessCount` を Evidence 出力に自動反映するのみ。

**新規実装は不要**。Sprint3-A の A-7 で対応。

**168h試験条件②で使用する指標**:
- `reclaimSuccessCount` の単調増加性
- `isChronic == false` (30秒超の Reader 慢性滞留なし)
- `overflowCount == 0` (キュー溢れなし)

---

### 4.3 Sprint3-C: Pipeline Ledger 復旧

#### C-1: `RuntimePublicationOrchestrator::notifyWorldRetired()` 新設

```cpp
// RuntimePublicationOrchestrator.h — 追加
/// ★ Sprint3-C: World 退役を Pipeline Ledger に通知
void notifyWorldRetired(uint64_t worldId) noexcept {
    stateOwner_.onRetired(worldId);
}
```

#### C-2: `onRuntimeRetiredNonRt()` からの呼び出し

```cpp
// AudioEngine.Commit.cpp — onRuntimeRetiredNonRt() に追加
void AudioEngine::onRuntimeRetiredNonRt(const RuntimePublishWorld* world) noexcept
{
    // ... 既存処理 ...
    worldLifecycleAudit_.onWorldRetired(world->worldId, world->publication.epoch);

    // ★ Sprint3-C: Pipeline Ledger 復旧
    if (runtimeOrchestrator_) {
        runtimeOrchestrator_->notifyWorldRetired(world->worldId);
    }

    // ... 残りの既存処理 ...
}
```

#### C-3: `publishHealthSnapshot()` シグネチャ変更

```cpp
// RuntimePublicationOrchestrator.h — シグネチャ変更
// 変更前: void publishHealthSnapshot() noexcept;
// 変更後: void publishHealthSnapshot(uint64_t externalReclaimedCount) noexcept;
void publishHealthSnapshot(uint64_t externalReclaimedCount) noexcept;

// RuntimePublicationOrchestrator.cpp — 実装
void RuntimePublicationOrchestrator::publishHealthSnapshot(
    uint64_t externalReclaimedCount) noexcept
{
    const auto& state = stateOwner_.state();
    const auto nowUs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    OrchestratorHealthSnapshot snapshot;
    snapshot.submittedCount = state.progress.submittedCount;
    snapshot.publishedCount = state.progress.publishedCount;
    snapshot.retiredCount = state.progress.retiredCount;
    snapshot.reclaimedCount = externalReclaimedCount;  // ★ EpochDomain から受け取る
    snapshot.executorQueueDepth = state.progress.executorQueueDepth;
    snapshot.lastProgressTimestampUs = state.progress.lastProgressTimestampUs;
    snapshot.stuckStage = state.progress.detectStuckStage();
    snapshot.timestampUs = nowUs;

    telemetryRecorder_.recordHealth(snapshot);
}
```

#### C-4: `emitEvidenceTickNonRt()` 定期呼び出し

```cpp
// AudioEngine.Commit.cpp — emitEvidenceTickNonRt() に追加
void AudioEngine::emitEvidenceTickNonRt(bool force) noexcept
{
    // ... 既存のレートリミット処理 ...

    // ★ Sprint3-C: Orchestrator 健全性スナップショット (1秒周期)
    if (runtimeOrchestrator_) {
        const uint64_t reclaimed = m_retireRouter
            ? m_retireRouter->reclaimSuccessCount() : 0;
        runtimeOrchestrator_->publishHealthSnapshot(reclaimed);
    }

    // ... 既存の Evidence 出力 ...
    retireRuntimeEx_.emitRetireTimeline(evidenceRoot / "retire_timeline.json");
    evidenceExporter_.exportEvidence();
    worldLifecycleAudit_.tryDumpPeriodic();
}
```

**安全性確認**: `publishHealthSnapshot()` 内の `telemetryRecorder_.recordHealth()` は `convo::publishAtomic` のみ (no locks, no I/O)。1秒タイマからの定期呼び出しは完全に安全。

### 4.4 Sprint3-E: Evidence 非同期書き込み (Async EvidenceWriter)

#### 背景: 同期ファイルI/Oによるメッセージスレッドブロックリスク

現状の `emitEvidenceTickNonRt()` は **JUCE Message Thread（メッセージスレッド）** 上の
`timerCallback()` から呼ばれており、以下の同期ファイルI/Oを実行している:

| 呼び出し | I/O内容 | ブロック時間（想定） |
| --- | --- | --- |
| `retireRuntimeEx_.emitRetireTimeline()` | ファイル書き込み+close | 0.1〜10ms |
| `evidenceExporter_.exportEvidence()` | 11ファイル書き込み | 0.5〜50ms |
| `worldLifecycleAudit_.tryDumpPeriodic()` | ファイル書き込み+close | 0.1〜10ms |
| **合計** | **〜13ファイル/秒** | **0.7〜70ms/秒** |

このメッセージスレッドは `timerCallback()` 内で他にも以下の重要処理を実行している:
- RCU Reader 操作
- EQ/Runtime 状態クリーンアップ
- Dither RNG 補充
- HealthMonitor 評価
- Publication Orchestrator 制御
- ディスクI/Oブロック中は **これら全てが遅延** する

ただしアンチウイルスソフトによるファイルスキャンやHDDのヘッド待機、
ディスクフル等でI/Oが10ms超ブロックした場合でも、
**メッセージスレッド自体はRTスレッドではないため、オーディオ処理が直接停止することはない**。
ブロックの影響は「タイマーの応答遅延」と「UI描画の一瞬のカクつき」に限定される。

> **リスク評価**: 現実的なリスクは低い（SSD環境では1ファイル0.1ms未満）。
> 本 Sprint3-E は将来の安全策として設計する。優先度は Sprint3-D より低い。

#### 設計: ロックフリーSPSCキュー + 専用Writerスレッド

既存の `core/CommandBuffer.h` の `SPSCRingBuffer` パターンを流用し、
Evidenceデータをメモリ上のRingBufferにプッシュするのみとし、
専用の低優先度スレッドが非同期でディスクにFlushする:

```cpp
// ★ ISREvidenceWriter.h — 新規ファイル (Sprint3-E)
#include "core/CommandBuffer.h"  // SPSCRingBuffer を流用

// Evidence書き込みリクエスト (固定長、コピー軽量)
struct EvidenceWriteRequest {
    enum class Type : uint8_t {
        RetireTimeline,
        WorldLifecycleAudit,
        EpochReclaimAudit,
        ShutdownTrace,
        FlushAll    // シャットダウン時や強制モードで使用
    };
    Type type;
    // ★ 実際のシリアライズ済みデータは固定長バッファに格納
    //    (動的確保を避けるため、リングバッファの要素は小さく保つ)
    uint8_t payload[256];  // 最大256バイトのペイロード
};

// ★ 非同期Evidence Writer
class AsyncEvidenceWriter {
public:
    // ★ BufferSize=256 の根拠:
    //   - Evidence書き込みは最大1秒に1回、1回あたり最大5種類のファイル
    //   - ピーク時でも秒間5エントリ、256エントリで51.2秒分のバッファ
    //   - ディスクI/Oが一時的にブロック（アンチウイルススキャン等）しても
    //     51秒以内に回復すればデータ消失なし
    //   - 256エントリ × sizeof(EvidenceWriteRequest) ≈ 256 × 264B ≈ 66KB
    //     メモリフットプリントは無視可能
    using Buffer = SPSCRingBuffer<EvidenceWriteRequest, 256>;

    explicit AsyncEvidenceWriter(Buffer& buffer)
        : buffer_(buffer) {}

    void start();  // 専用スレッド起動
    void stop();   // スレッド停止 + 残り全Flush

private:
    void run();  // SPSCRingBuffer から取り出して書き込み
    Buffer& buffer_;
    std::thread writerThread_;
};
```

**既存 `emitEvidenceTickNonRt()` の変更**:
```cpp
// AudioEngine.Commit.cpp — emitEvidenceTickNonRt() 変更
void AudioEngine::emitEvidenceTickNonRt(bool force) noexcept
{
    // ... 既存のレートリミット処理 ... (軽量演算のみ, I/Oなし)

    // ★ Sprint3-C: Orchestrator 健全性スナップショット (軽量, I/Oなし)
    if (runtimeOrchestrator_) {
        const uint64_t reclaimed = m_retireRouter
            ? m_retireRouter->reclaimSuccessCount() : 0;
        runtimeOrchestrator_->publishHealthSnapshot(reclaimed);
    }

    // ★ Sprint3-E: Evidence 書き込みリクエストをキューに投入
    //    実際のファイルI/Oは専用スレッドが実行
    if (force) {
        evidenceWriterBuffer_.push(EvidenceWriteRequest{EvidenceWriteRequest::Type::FlushAll});
    }
    evidenceWriterBuffer_.push(EvidenceWriteRequest{EvidenceWriteRequest::Type::RetireTimeline});
    evidenceWriterBuffer_.push(EvidenceWriteRequest{EvidenceWriteRequest::Type::WorldLifecycleAudit});

    // ★ Sprint3-A: EBR Queue Visibility (A-7)
    if (m_retireRouter) {
        evidenceWriterBuffer_.push(EvidenceWriteRequest{EvidenceWriteRequest::Type::EpochReclaimAudit});
    }

    // ★ 旧来の同期書き込み呼び出しは削除:
    //    retireRuntimeEx_.emitRetireTimeline(...);       // → 削除
    //    evidenceExporter_.exportEvidence();              // → 削除
    //    worldLifecycleAudit_.tryDumpPeriodic();          // → 削除
}
```

**シャットダウン時の特別処理**:
```cpp
// ★ シャットダウン手順内 (AudioEngine.Processing.ReleaseResources.cpp)
void AudioEngine::releaseResources()
{
    // ... 既存の処理 ...

    // ★ emitShutdownTrace はシャットダウン時に同期的に実行 (1回のみ)
    //    シャットダウントレースは最も重要な証跡であり、非同期にすると
    //    プロセス終了前に書き込みが完了しないリスクがあるため同期待機する
    shutdownRuntime_.emitShutdownTrace(healthState);
    // ★ Writerスレッドに全Flushを指示し完了を待機
    evidenceWriter_.stop();  // 内部で SPSCRingBuffer の残りを全て書き込んでから join

    // ... 残りの処理 ...
}
```

**ファイル変更一覧（追加）**:

| Sprint | # | ファイル | 操作 |
| --- | --- | --- | --- |
| 3 | E-1 | `src/audioengine/ISREvidenceWriter.h` | 新規 |
| 3 | E-2 | `src/audioengine/ISREvidenceWriter.cpp` | 新規 |
| 3 | E-3 | `src/audioengine/AudioEngine.Commit.cpp` | emitEvidenceTickNonRt() 変更 |
| 3 | E-4 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | シャットダウン処理変更 |
| 3 | E-5 | `CMakeLists.txt` | 新規cpp登録 |

---

## 5. 168時間連続運転試験計画

### 5.1 試験環境

| 項目 | 要件 |
| --- | --- |
| CPU | Intel Core i7/i9 (AVX2対応) |
| RAM | 32GB以上 |
| OS | Windows 11 24H2 |
| サンプリングレート | 48000Hz / 96000Hz |
| バッファサイズ | 256 samples / 512 samples |
| IR数 | 1〜4基（実運用想定範囲） |

> **OS ページフォールト対策**: ダミーメモリ確保による Pre-faulting は無効。
> `std::vector` で確保した領域と `new`/`malloc` が返す領域は別物であり、
> ページフォールト防止効果はないため実施しない。
>
> ConvoPeq は既に **Audio Thread 内での動的メモリ確保を禁止** する設計に準拠しており、
> 全必要なメモリは `prepareToPlay()` 等の初期化フェーズで事前確保される。
> 試験開始前の **ウォームアップ処理** により、全オブジェクトの全バッファに実際にアクセスされ、
> 物理メモリが割り当てられるため、以降の Audio Thread でのページフォールトは発生しない。
> 具体的には試験開始直後に以下を実行:
> ```cpp
> // 試験開始直後に3秒間の無音処理を実行
> // これにより AudioEngine, DSPCore, EpochDomain, キュー等の全バッファが
> // 物理メモリにフォールトされ、以降の RT 処理でページフォールトが発生しなくなる
> engine.startProcessing(48000, 256);  // 通常の初期化
> engine.processSilence(3.0);          // 3秒間の無音処理で全バッファをウォームアップ
> ```
>
> #### ★ ウォームアップの拡張: IR コンボリューションの最大負荷投入
>
> 3秒間の無音処理だけでは、IRコンボリューションで使用する大きなメモリブロック
> （IR係数バッファ、コンボリューション中間バッファ）が物理メモリにフォールトされない
> 可能性がある。168時間の長期間試験では、ヒープ断片化によるメモリ割り当て失敗を
> 防止するため、**「使用する最大のIRコンボリューションをロード/アンロードするループ」**
> を事前に5〜10回実行し、メモリ使用量の最大値（Steady State）に到達させること:
> ```cpp
> // ウォームアップ: 使用する最大IR構成をロード/アンロードしてメモリ使用量を安定させる
> // このループにより、ヒープが最悪の断片化状態に収束し、
> // 168時間の試験中に新たなメモリ割り当てが発生するリスクを排除する
> constexpr int kWarmupIterations = 8;
> for (int i = 0; i < kWarmupIterations; ++i) {
>     // ★ 実運用で想定される最大のIR構成をロード
>     engine.loadIRConvolution(maxIRConfig);
>     engine.processSilence(0.5);  // 0.5秒間の処理で全バッファにアクセス
>     engine.unloadIRConvolution();
>     engine.processSilence(0.2);  // アンロード後も短時間処理して安定化
> }
> // 最終ウォームアップ: 実運用の通常状態と同じIR構成で開始
> engine.loadIRConvolution(normalIRConfig);
> engine.processSilence(3.0);  // 最終ウォームアップ
> ```
> このウォームアップにより、以下の効果が得られる:
> - 全IRコンボリューションバッファが物理メモリにフォールトされ、RT中のページフォールト回避
> - ヒープが最悪の断片化状態に収束し、168時間の試験中に新たなメモリ割り当て失敗が発生しない
> - キャッシュライン競合が実運用と同じ状態に収束

### 5.2 自動収集 Evidence

| ファイル | 内容 | 収集間隔 | ローテーション |
| --- | --- | --- | --- |
| `evidence/world_lifecycle_audit.json` | World発行/退役、reclaim統計 | 60秒 | 上書き (常に最新1ファイル) |
| `evidence/publication_progress_log.json` | Orchestrator進捗 | 1秒 (Sprint3-C後) | 上書き (常に最新1ファイル) |
| `evidence/shutdown_trace.json` | シャットダウン時統計 | シャットダウン時のみ | 上書き (試験終了後のみ参照) |
| `evidence/retire_timeline.json` | 退役タイムライン | 1秒 | 上書き (常に最新1ファイル) |

> **補足**: Evidence ファイルはすべて「アトミックファイル置換 (tmp + rename)」方式で書き込まれる。
> 1. `.json.tmp` 一時ファイルに完全なJSONを書き込む
> 2. 書き込み完了後 `file.fail()` でエラーをチェック（ディスクフル/権限エラー検出）
> 3. `std::filesystem::rename` で `.json` に原子的に置換
> 4. 書き込み中にクラッシュが発生しても、既存の `.json` ファイルは無傷で保持される
>
> この方式により、168時間の試験を通じてファイル数は増加せず（.tmp はクラッシュ時のみ残存）、
> 各 artifact につき常に1ファイルが保持される。総ディスク使用量は数十KB〜数百KBに留まる。
> ローテーションは不要。`.tmp` ファイルが前回のクラッシュ痕跡として残存した場合、
> 次回書き込み時に `std::filesystem::remove` で自動清除される。
>
> **適用範囲**:
> - `evidence/epoch_reclaim_audit.json` (Sprint3-A 新規, A-7 にtmp+rename実装済)
> - `evidence/shutdown_trace.json` (Sprint2 変更, A-2.5 にtmp+rename実装済)
> - `evidence/retire_timeline.json` (既存 `ISRRetireRuntimeEx::emitRetireTimeline` — **Sprint3-E で修正**
>   AsyncEvidenceWriter の専用スレッドに移行する際に tmp+rename + file.fail() へ自動対応)
> - `evidence/world_lifecycle_audit.json` (既存 `WorldLifecycleAudit::emitSnapshot` — **Sprint3-E で修正**、同上)

### 5.3 合格条件 (5条件)

#### 条件①: World 整合性

```text
publishedCount ≈ retiredCount + activeWorldCount
```

- **定常偏差: 動的閾値を採用**（サンプリング間隔(60s) × World最大処理スループット）
  - 定常偏差 = `kMaxConcurrentWorlds × 2` を上限とする絶対値ベース
  - サンプリング間隔(60s)に最大何件の publish/retire が発生するかを
    `maxThroughputPerSec × 60` で見積もり、これを偏差上限とする
  - 例: 最大スループットが 5 Worlds/sec の場合 → 偏差上限 = 5 × 60 = 300
  - これにより、低スループット時は厳格に、高スループット時もリークを見逃さない
- **終端偏差: 0% (全オーディオストリーム停止後の最終状態で完全一致)**
- 出典: `WorldLifecycleAudit` (`evidence/world_lifecycle_audit.json`)
- 理論的根拠: World の publish と retire は1:1対応。
  動的偏差とすることで、サンプリングタイミングのズレをハードコードではなく
  環境適応的に許容する。終端の完全一致条件により絶対的なリーク検出を保証。
  activeWorldCount は publish で+1、retire で-1

#### 条件②: Grace Period 健全性

```text
条件②a: reclaimSuccessCount が単調増加していること
条件②b: isChronic == false (30秒超の慢性 Reader 滞留がないこと)
条件②c: overflowCount == 0 (キュー溢れがないこと)
```

- 出典: `ISRRetireRouter` (`collectDrainAudit()` 経由)
- `isChronic` は `detectStuckReaders()` の既存フィールド (residency > 30秒)
- `overflowCount` は `ISRRetireRouter` の既存カウンタ

> **`isChronic` 閾値の2フェーズ調整**: 30秒という閾値は設計者の経験値に基づいており、
> 実運用環境における Reader 滞留分布は未知である。そこで試験を2フェーズに分割する:
> 1. **第1週 (計測フェーズ)**: `isChronic` を不合格条件とせず、情報提供として記録する。
>    `stuckReaderCount`, `maxReaderResidencyUs`, `isChronic` の実測分布を取得する。
> 2. **第2週 (判定フェーズ)**: 第1週で得られた分布の99パーセンタイル値を基準に、
>    `isChronic` の閾値を調整した上で合格判定に使用する。
>
> **閾値の動的調整（再コンパイル不要）**:
> `kChronicResidencyUs` をコンパイル時定数 (`constexpr`) から `EpochDomain` の
> メンバ変数に変更し、コンストラクタまたは setter で外部から注入可能にする。
> デフォルト値は 30 秒とし、試験フェーズ移行時にプロセス再起動のみで閾値を変更できる:
> ```cpp
> // EpochDomain.h — 定数→メンバ変数へ変更
> class EpochDomain {
> public:
>     static constexpr uint64_t kDefaultChronicResidencyUs = 30'000'000;
>
>     explicit EpochDomain(uint64_t chronicThresholdUs = kDefaultChronicResidencyUs) noexcept
>         : chronicResidencyUs_(chronicThresholdUs) {}
>
>     void setChronicResidencyUs(uint64_t us) noexcept {
>         chronicResidencyUs_.store(us, std::memory_order_relaxed);
>         // ★ ★ 閾値変更時: 内部状態リセット + クールダウン期間
>         //    閾値を「下げた」場合、これまで正常とされていた滞留が
>         //    突然 Chronic と判定され、アラートが急増する。
>         //    以下の対策で誤検知を防止する:
>         chronicResidencyChangedUs_ = convo::getCurrentTimeUs();
>     }
>     // ★ 変更時点を記録 (変更直後は判定を無効化するため)
>     [[nodiscard]] uint64_t chronicResidencyChangedTimeUs() const noexcept {
>         return chronicResidencyChangedUs_.load(std::memory_order_relaxed);
>     }
>
> private:
>     std::atomic<uint64_t> chronicResidencyUs_{kDefaultChronicResidencyUs};
>     std::atomic<uint64_t> chronicResidencyChangedUs_{0};  // ★ 追加: 最終変更時刻
> };
> ```
> 読み出し側 `detectStuckReaders()` も `chronicResidencyUs_.load(std::memory_order_relaxed)` で
> アクセスする (relaxed で十分: マルチWriter不可、診断目的の閾値のため)。
>
> **変更直後のクールダウン期間**: `detectStuckReaders()` 内で、閾値変更から
> 5分間（300秒）は `isChronic` の判定を無効化する。これにより、変更直後に
> 新閾値を超える滞留が急増しても、それは変更前の正常な滞留が新閾値に
> 引っかかっただけであり、誤検知として扱う:
>
> **★ ヒステリシス（履歴依存性）の明示的定義**:
> 閾値変更時、変更前にすでに滞留中の全Readerに対して以下のルールを適用する:
> - 「変更前の古い閾値を超えているが、変更前から継続して滞留しているReader」は、
>   新閾値がそれより低くても `isChronic` と判定しない
> - これは変更後の `kChronicGracePeriodUs` の期間（5分間）の判定無効化により
>   自動的に達成される
> - クールダウン期間経過後も滞留が継続しているReaderは、**新たに滞留を開始した
>   ものとして扱われ**、新閾値での判定が適用される
> - この「一度無効化してから再判定」の設計により、履歴依存の不整合を排除する
> ```cpp
> // EpochDomain.h — detectStuckReaders() 内判定
> // ★ 閾値変更後 kChronicGracePeriodUs 以内は慢性滞留判定を無効化
> const uint64_t changedUs = chronicResidencyChangedUs_.load(std::memory_order_relaxed);
> constexpr uint64_t kChronicGracePeriodUs = 5 * 60 * 1'000'000;  // 5分
> const bool inGracePeriod = (changedUs != 0 && (nowUs - changedUs) < kChronicGracePeriodUs);
>
> if (depth > 0 && residencyUs > chronicResidencyUs_.load(std::memory_order_relaxed)
>     && info.pendingRetireCount > 0 && !inGracePeriod) {
>     info.isChronic = true;
>     // ...
> }
> ```
> これにより、168時間試験を連続して実行したまま第1週→第2週の移行が可能になり、
> 再コンパイル＋再試験(計336時間)の工数リスクを排除できる。また閾値変更時の
> スパイクアラートを誤検知として扱うことで、運用現場のアラート疲れを防止する。

#### 条件③: Backlog 非増加

```text
pendingRetireCount の最大値が試験開始時の2倍を超えない
```

- 出典: `ISRRetireRouter::pendingRetireCount()`
- Evidence: `evidence/retire_timeline.json`

#### 条件④: Shutdown 収束性

```text
routerPendingRetire がシャットダウン開始後60秒以内に0に収束
```

- 出典: `collectDrainAudit().routerPendingRetire`
- Evidence: `evidence/shutdown_trace.json`

> **注意**: `emitShutdownTrace()` は `std::ofstream` によるファイル書き込みを行う。
> 万が一ディスクフル等で書き込みに失敗した場合、書き込み完了後に `file.fail()` を
> チェックし、エラー時は `rename` を行わないことで破損ファイルの作成を防止する。
> また、tmp+rename 方式により書き込み中のクラッシュからも既存ファイルを保護する。
> 詳細なコードは **A-2.5** に記載済み。以下のポイントを実装で遵守すること:
> 1. **tmp+rename**: `.json.tmp` に書き込み → `std::filesystem::rename` でアトミック置換
> 2. **`file.fail()` チェック**: `close()` 後に `file.fail()` でディスクフル/権限エラーを検出
> 3. **フォールバック**: tmp 書き込み失敗時は `%TEMP%` へ一意化ファイル名で退避
> 4. **`s_fallbackCounter` は `static std::atomic<uint32_t>` でプロセス生存中維持
>
> #### ★ Windows ファイルロック対策: rename リトライ + 別名フォールバック
>
> `std::filesystem::rename` は Windows において、対象ファイルが別プロセスや
> エクスプローラで開かれている（読み取り専用ロック等）と `error_code` が設定される。
> 以下のリトライ戦略を実装に追加する:
> ```cpp
> // rename リトライ: 最大3回、100ms 間隔
> constexpr int kMaxRenameRetries = 3;
> constexpr auto kRenameRetryInterval = std::chrono::milliseconds(100);
> for (int retry = 0; retry < kMaxRenameRetries; ++retry) {
>     std::filesystem::rename(tmpPath, outputPath, ec);
>     if (!ec) break;  // 成功
>     if (retry < kMaxRenameRetries - 1) {
>         std::this_thread::sleep_for(kRenameRetryInterval);
>     }
> }
> // ★ 全リトライ失敗時は別名で書き込む (上書きではなく別ファイルとして保存)
> if (ec) {
>     static std::atomic<uint32_t> s_renameFallbackCounter{0};
>     const auto altPath = std::filesystem::current_path() / "evidence"
>         / ("shutdown_trace_" + std::to_string(
>             s_renameFallbackCounter.fetch_add(1, std::memory_order_relaxed)) + ".json");
>     std::filesystem::rename(tmpPath, altPath, ec);
>     // altPath への rename も失敗 → そのまま諦める (次回上書きで再試行)
> }
> ```
> このリトライにより、エクスプローラでのファイル閲覧やウイルススキャンによる
> 一時的なロックから回復できる。全リトライ失敗時も元の `.json` は無傷で保持される。

#### 条件⑤: Recovery 健全性

```text
強制タイムアウト後、publishedCount / retiredCount / reclaimSuccessCount が
60秒以内に正常増加を再開する
```

- Sprint3-A の `reclaimSuccessCount` により観測可能

> **注意**: EBR の仕様上、1 World の退役が複数の DeletionEntry を生成する可能性があるため、
> `retiredCount` と `reclaimSuccessCount` の絶対値は一致しなくてよい。
> 傾向としての相関（ともに増加し、ともに停滞しないこと）を評価の主軸とする。

### 5.4 合否判定基準の定量的詳細

| 確認項目 | 観測ファイル | 合格基準 |
| --- | --- | --- |
| `publishedCount` vs `retiredCount` 差分 | `evidence/world_lifecycle_audit.json` | 定常運転中の差分が `kMaxConcurrentWorlds × 2` 以下で安定 |
| `DeferredDeletionQueue` バックログ | `evidence/retire_timeline.json` | 168時間で単調増加しない (メモリリーク判定) |
| `blockingReasonStats.count` | `evidence/shutdown_trace.json` | Start/Stop 1000回あたりの `ReaderActive` / `PendingRetire` 発生率が規定閾値以下 |
| `blockingReasonHistory` 遷移パターン | `evidence/shutdown_trace.json` | `ReaderActive → PendingRetire → ReaderActive` の循環が検出されないこと |
| ★ BlockingReason 条件付き遷移確率 | `evidence/shutdown_trace.json` | 特定の Reason A→B の遷移確率が時間経過とともに増加しないこと |
| `routerPendingRetire` 最大 epoch 数 | `collectDrainAudit()` | Shutdown 開始から収束までの所要 epoch 数が規定値以下 |

### 5.5 条件付き確率テスト (異常遷移の定量評価)

`BlockingReasonHistory` の時系列イベントから、異常パターンの出現確率を
定量的に評価する。単なる合格/不合格の二元論ではなく、
**エンジニアリング的安定性**を数値で証明する。

#### 評価方法

```text
1. 全シャットダウンイベントから BlockingReasonHistory を収集
2. 隣接する Reason の遷移ペア (A→B) を全て抽出
3. 各遷移ペアの出現確率を計算:
   P(A→B) = count(A→B) / count(A)
4. 時間経過による変化を監視:
   - 試験前半 (0〜84h) と後半 (84〜168h) で P(A→B) を比較
   - 増加傾向を示す遷移がないことを確認
5. 特に以下の「循環パターン」の確率が増加しないことを検証:
   - ReaderActive → PendingRetire → ReaderActive
   - PendingPublication → DeferredPublish → PendingPublication
```

#### 合格基準

| 評価項目 | 基準 | 根拠 |
| --- | --- | --- |
| P(ReaderActive→PendingRetire→ReaderActive) | 0.01未満で安定 | 循環が発生しても1%未満なら偶発的 |
| 全 Reason の単独発生確率 | 試験前後半で±20%以内 | シャットダウン特性が時間経過で変化しないこと |
| 新規遷移パターンの出現 | 後半で新たな遷移が出現しない | システムの経年劣化による異常誘発がないこと |

### 5.6 強制 Recovery 試験の合格基準

意図的に `waitForDrain()` タイムアウトを発生させた後、次の Start 時に
`publishedCount / retiredCount / reclaimSuccessCount` が正常に動作することを確認する。
前 Session のカウンタ残留 (リセット漏れ) がないことが合格条件。

### 5.6 試験手順

#### 停止条件（Early Termination Criteria）

168時間を待たずに試験を中断すべき異常条件を定義する:

| レベル | 条件 | 対応 |
| --- | --- | --- |
| **Critical** | 条件②c違反: `overflowCount > 0` が発生 | 即時中断。キュー溢れは修正必須の設計欠陥 |
| **Critical** | 条件②b: `isChronic == true` が30分以上継続 | 即時中断。Readerスレッドのデッドロックまたはリークの可能性 |
| **Warning** | 条件①: `publishedCount - retiredCount > kMaxConcurrentWorlds × 4` が10分以上継続 | 中断検討。World整合性の恒常的崩れ |
| **Warning** | 条件③: `pendingRetireCount` が試験開始時の4倍超過 | 中断検討。Backlogの異常増加 |
| **Info** | 条件①: 定常偏差超過が30秒未満で回復 | 継続。サンプリングタイミングの一時的ズレ |

#### 通常手順

```text
1. 試験環境のセットアップ
   - Debug ビルドで全 Evidence が出力されることを確認
   - Release ビルドで 168h 試験を開始

2. 常時監視 (自動)
   - 1分間隔で Evidence を収集
   - 条件①〜③を自動判定
   - 異常(条件違反)を検出した場合は上記停止条件テーブルに従い対応

3. 168h 経過後
   - 全 Evidence を分析
   - 条件④: シャットダウン時に自動判定
   - 条件⑤: 強制タイムアウト後に自動判定
   - 条件付き確率テスト: BlockingReasonHistory の遷移確率を算出

4. 最終判定
   全条件充足 → Production-Proven Practical Stable ISR Bridge Runtime 🏆
   不合格 → 該当条件の原因分析 + 修正後再試験
```

---

## 6. 完了条件と判定基準

### 6.1 Sprint 完了条件

| Sprint | 完了条件 | 検証方法 |
| --- | --- | --- |
| Sprint1 | B-2 + A-2 実装完了 | Debugビルド成功 / ユニットテスト通過 |
| Sprint2 | A-3 + B-1 実装完了 | Debugビルド成功 / blockingHistory動作確認 |
| Sprint3 | A+B+C+D 実装完了 | Debugビルド成功 / 全Evidence出力確認 |

### 6.2 最終完了条件

```text
□ Sprint1 完了
□ Sprint2 完了
□ Sprint3-A 完了 (EBR Queue Visibility)
□ Sprint3-B 完了 (Grace Period Visibility, A完了が前提)
□ Sprint3-E 完了 (Async EvidenceWriter, 任意)
□ Sprint3-D 完了 (168h試験 全5条件 + 条件付き確率テストdger復旧)
□ Sprint3-D 完了 (168h試験 全5条件合格)
```

全条件充足時 → **Production-Proven Practical Stable ISR Bridge Runtime** 到達 🏆

### 6.3 マイルストーン定義

```text
現在地
└─ Practical Stable ISR Bridge Runtime
   ├─ ソースコード解析完成度: 93〜96%
   └─ 実運用検証: 未評価

Sprint 1〜3 完了後
└─ Practical Stable ISR Bridge Runtime (Observability Complete)
   ├─ 全 Observability レイヤー実装済み
   └─ 実運用検証: 未実施

168h 連続運転 + 全試験通過後
└─ Production Proven Practical Stable ISR Bridge Runtime (完成宣言)
   ├─ ソースコード解析完成度: 93〜96%
   └─ 実運用検証: 通過済み
   → 完成宣言の唯一の条件: 168h 試験 全5条件合格
```

---

## 7. 付録: コード変更詳細

### 7.1 コードベース検証結果（最終確認: 2026-06-16）

| 確認項目 | 確認ツール | 結果 | 計画への影響 |
| --- | --- | --- | --- |
| `ISRBarrierOptimizer` 呼び出し元 | grep / CodeGraph | **0件** | ✅ B-2 削除リスクゼロ確定 |
| `publishHealthSnapshot()` 呼び出し元 | CodeGraph `find_callers` | **0件** | ✅ C-3/C-4 接続必要確定 |
| `onRetired()` 呼び出し元 | CodeGraph `find_callers` | **0件** | ✅ C-1/C-2 復旧必要確定 |
| `onReclaimed()` 呼び出し元 | CodeGraph `find_callers` | **0件** | ✅ C-3 で external 注入方式に決定 |
| `isChronic` 実装 | ファイル読み取り | ✅ 3条件: depth>0 AND residency>30s AND pendingRetire>0 | ✅ 条件②b で使用可能 |
| `overflowCount()` 実在 | grep | ✅ `ISRRetireRouter.h:91` | ✅ 条件②c で使用可能 |
| Evidence 書込みモード | ファイル読み取り | ✅ `std::ios::trunc` (上書き) | ✅ ローテーション不要 |
| `m_retireRouter` アクセスパス | grep | ✅ `AudioEngine.h:3456` unique_ptr メンバ | ✅ `emitEvidenceTickNonRt()` からアクセス可能 |
| `runtimeOrchestrator_` アクセスパス | grep | ✅ `AudioEngine.h:2801` unique_ptr メンバ | ✅ C-4 で `publishHealthSnapshot()` 呼出可能 |
| `detectStuckReaders()` `kChronicResidencyUs` | ファイル読み取り | ✅ 30秒 (コード確認済) | ✅ 条件②b 閾値確定 |
| `collectDrainAudit()` 既存フィールド | ファイル読み取り | ✅ pending~, stuckReader, routerPendingRetire 等 | ✅ A-5/A-6 拡張箇所確定 |

### 7.2 変更ファイル一覧

| Sprint | # | ファイル | 操作 |
| --- | --- | --- | --- |
| 1 | B-2 | `src/audioengine/ISRBarrierOptimizer.h` | 削除 |
| 1 | B-2 | `src/audioengine/ISRBarrierOptimizer.cpp` | 削除 |
| 1 | B-2 | `src/audioengine/AudioEngine.h` | include行削除、メンバ宣言削除 |
| 1 | B-2 | `CMakeLists.txt` | cpp登録削除 |
| 1 | A-2 | `src/audioengine/ISRShutdown.h` | BlockingReasonStats追加、reasonToString宣言 |
| 1 | A-2 | `src/audioengine/ISRShutdown.cpp` | markTimedOut統計更新、emitShutdownTrace拡張 |
| 2 | A-3 | `src/audioengine/ISRShutdown.h` | TinyRingBuffer追加、BlockingReasonEvent定義 |
| 2 | A-3 | `src/audioengine/ISRShutdown.cpp` | blockingReasonHistory記録追加 |
| 2 | B-1 | `src/audioengine/RuntimeDrainAudit.h` | isAllZero()にrouterPendingRetire追加 |
| 3 | A-1 | `src/DeferredDeletionQueue.h` | reclaim()戻り値void→uint32_t |
| 3 | A-2 | `src/core/EpochDomain.h` | reclaimAttemptCount_/reclaimSuccessCount_追加 |
| 3 | A-3 | `src/core/IEpochProvider.h` | reclaimAttemptCount()/reclaimSuccessCount() virtual追加 |
| 3 | A-4 | `src/audioengine/ISRRetireRouter.h` | 委譲override追加 |
| 3 | A-5 | `src/audioengine/RuntimeDrainAudit.h` | reclaimAttemptCount/reclaimSuccessCountフィールド追加 |
| 3 | A-6 | `src/audioengine/AudioEngine.Threading.cpp` | collectDrainAudit()収集追加 |
| 3 | A-7 | `src/audioengine/AudioEngine.Commit.cpp` | Evidence出力追加 (epoch_reclaim_audit.json) |
| 3 | C-1 | `src/audioengine/RuntimePublicationOrchestrator.h` | notifyWorldRetired()追加 |
| 3 | C-2 | `src/audioengine/AudioEngine.Commit.cpp` | onRuntimeRetiredNonRt()から呼出 |
| 3 | C-3 | `src/audioengine/RuntimePublicationOrchestrator.h/.cpp` | publishHealthSnapshot()シグネチャ変更 |
| 3 | E-1 | `src/audioengine/ISREvidenceWriter.h` | 新規 (AsyncEvidenceWriter) |
| 3 | E-2 | `src/audioengine/ISREvidenceWriter.cpp` | 新規 (AsyncEvidenceWriter) |
| 3 | E-3 | `src/audioengine/AudioEngine.Commit.cpp` | emitEvidenceTickNonRt()非同期化 |
| 3 | E-4 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | シャットダウン時同期Flush |
| 3 | E-5 | `CMakeLists.txt` | 新規cpp登録 |

**総変更ファイル数**: 24ファイル（新規追加2、削除2、修正20を含む場合: +5ファイル（新規2 + 修正3ngine.Commit.cpp` | emitEvidenceTickNonRt()定期呼出追加 |

**総変更ファイル数**: 19ファイル（新規追加0、削除2、修正17）

### 7.2 設計上の判断根拠

| 判断 | 根拠 |
| --- | --- |
| `ShutdownCompletionState` 分離不要 | 設計文書 `remediation_plan.md` が `collectDrainAudit()` を Authority と明記。`isAllZero()` は `"(observation only)"` 診断ログ |
| `reclaimEfficiency` 不採用 | `tryReclaim()` が10箇所以上から呼ばれ、grace period 中は効率が低くても正常。根拠不足 |
| `isChronic` 採用 (stuckReaderCount==0 不採用) | 既存フィールド。30秒閾値。一時的な Reader 滞留を許容 |
| `onReclaimed()` 呼び出し不要 | デリータは stateless lambda で this をキャプチャ不可。代わりに `publishHealthSnapshot(externalReclaimedCount)` で EpochDomain 値を受け渡し |
| World Reclaim Visibility 廃止 | World 数と DeletionEntry 数の単位不一致。EBR 設計上、World 単位の reclaim 追跡は不可能 |
| `retired - active` 推定式不採用 | `activeWorldCount` は retire 時点でデクリメント。reclaim 時には既に減っている。理論的に不成立 |

### 7.3 将来課題

| 課題 | 理由 | 対応時期 |
| --- | --- | --- |
| World 単位の reclaim 追跡 | `DeletionEntry` に `publicationSequenceId` は既にあるが、`worldId` なし。`ISRRetireRouter::enqueueRetire()` 経路では `publicationSequenceId`/`generation` パラメータが渡されていない。対応には `DeferredDeletionQueue::enqueue` の完全パラメータ経路の整備と `DeletionEntry` への `worldId` フィールド追加(8バイト拡張)が必要。`DeletionEntry` は `is_trivially_copyable` 要件のため、追加可能なフィールドサイズには制限あり（現在72B、キャッシュライン境界をまたがないよう128B未満を維持）。 | 将来 |
| `WorldLifecycleAudit` への統合 | EBR Queue 統計と World 統計の分離が現状正しい。`reclaimSuccessCount` と `publishedCount` の相関分析は後処理スクリプトで対応可能。 | 要件発生時 |
| `DeletionEntryType` の拡張 | 現在 `Generic=0` のみ。World 追跡や種別別統計のために `WorldRuntime`, `DSPHandle`, `CrossfadeBuffer` 等の種別追加が可能。`DeletionEntry` の `type` フィールドは既存(1バイト)。新しい種別を追加してもトリビアルコピー可能性は維持される。 | 将来 |
| Evidence 非同期化の恒久化 | Sprint3-E の `AsyncEvidenceWriter` は将来安全策。恒久実装とするかは168h試験の結果で判断。ディスクI/Oによるメッセージスレッドブロックが実運用で観測された場合のみ必須。 | Sprint3-E完了後判断 |
