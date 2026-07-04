# CPU_MIG および String 構築コスト分析レポート
## Numeric-Only DiagEvent 設計

**作成日**: 2026-06-30
**分析者**: AI Assistant
**対象**: ConvoPeq Audio Engine 診断ログ RT 違反

---

## 1. 要約

### 問題の核心
現在の診断コードが、計測対象（RT スレッド）自体を乱している。

### 主要な発見
1. **CPU_MIG 実測頻度**: 153.3回/秒（ログ解析195.6秒間の実測。理論上限187.5回/秒を超えない）
2. **String 構築コスト**: CPU_MIG イベントあたり 400-750ns（推定値、MSVC/JUCE/SSOで変動）
3. **RT バジェットへの影響**: CPU_MIG 最大 115μs/秒 = RT バジェットの 2.16%（推定値）
4. **既存の解決策**: LockFreeRingBuffer<XRunEvent> のパターンが動作実証済み

### 推奨解決策
**Numeric-Only DiagEvent 構造体**の導入：
- RT 側: 数値のみ収集（push() のみ、ユーザーコードで memcpy を書く必要がない）
- Timer 側: String フォーマット（非 RT 安全）

---

## 2. CPU_MIG 分析

### 2.1 発生箇所
CPU_MIG は 2 箇所でのみ検出：

1. **AudioEngine.Processing.AudioBlock.cpp:173**
`cpp
juce::Logger::writeToLog(
    "[CPU_MIG] callback=" + juce::String(cbIdx)
    + " seq=" + juce::String(pubSeq)
    + " gen=" + juce::String(gen)
    + " cpu=" + juce::String(cpu)
    + " prev=" + juce::String(prev));
`

2. **AudioEngine.Processing.BlockDouble.cpp:155**
`cpp
juce::Logger::writeToLog(
    "[CPU_MIG] callback=" + juce::String(cbIdx)
    + " seq=" + juce::String(pubSeq)
    + " gen=" + juce::String(gen)
    + " cpu=" + juce::String(cpu)
    + " prev=" + juce::String(prev));
`

### 2.2 頻度の物理的制約と実測値

**理論上の上限:**
| パラメータ | 値 |
|-----------|-----|
| サンプリングレート | 192kHz |
| バッファサイズ | 1024 サンプル |
| コールバック頻度（理論値） | 192kHz / 1024 = **187.5 回/秒** |

CPU_MIG は callback 毎に最大1回検出可能であり、**理論上の最大頻度 = コールバック頻度 = 187.5 回/秒**。

**実測値（ログ解析 195.6秒間）:**
| 指標 | 値 |
|-----|-----|
| 総CPU_MIGイベント | **29,980 件** |
| 総コールバック数 | **34,208 件**（CALLBACK_STAGE max seq） |
| 実コールバック頻度 | **174.9 回/秒**（XRUN損失により理論値を下回る） |
| CPU_MIG イベント頻度 | **153.3 回/秒** |
| コールバック中CPU Migration発生率 | **87.6%** |

**29,980件と187.5回/秒の整合性:**
```
CPU_MIG 頻度 153.3/s < コールバック頻度 174.9/s < 理論上限 187.5/s
```
→ すべて整合している。29,980件/195.6秒 = 153.3回/秒となり、理論上限を超えない。
CPU_MIG イベントは左右チャンネルや Float/Double の合算ではなく、
全コールバックの87.6%でCPU Migrationが発生していることを示す単一カウントである。
※ 検証: CPU_MIG の callback= 値がすべてユニーク（重複なし、29,980個の異なるコールバック）。

### 2.3 タイミング測定
CPU_MIG 検出箇所には getCurrentTimeUs() がないため、直接的なコスト測定は不可能。
ただし、String 構築コストは推定可能。

---

## 3. String 構築コスト分析

### 3.1 CPU_MIG の String 構築
`cpp
"[CPU_MIG] callback=" + juce::String(cbIdx)      // 1 回
+ " seq=" + juce::String(pubSeq)                 // 2 回
+ " gen=" + juce::String(gen)                    // 3 回
+ " cpu=" + juce::String(cpu)                    // 4 回
+ " prev=" + juce::String(prev);                 // 5 回
`

**コストの内訳（推定値）:**
- juce::String(int64_t) × 5 回: 各 50-100ns（推定値）
- String 連結 × 5 回: 各 30-50ns（推定値）
- **合計: 400-750ns / イベント（推定値）**

### 3.2 logEqTime の String 構築（効率的なパターン）
`cpp
juce::String eqtLog("[EQ_TIME]");   // 1 回の割り当て
eqtLog += " seq=" + juce::String(callbackSeq);  // += 連結
eqtLog += " cpu=" + juce::String(cpu);
eqtLog += " test=" + juce::String(testIdx);
eqtLog += " bands=" + juce::String(bands);
`

**コストの内訳（推定値）:**
- juce::String 割り当て × 1 回: 50-100ns（推定値）
- += 連結 × 4 回: 各 2-5ns（推定値、割り当て済みバッファ）
- **合計: 60-120ns / イベント（推定値）**

**CPU_MIG との比較（推定値）:**
- CPU_MIG: 400-750ns（5 回の割り当て）
- logEqTime: 60-120ns（1 回の割り当て）
- **CPU_MIG は 6.7-12.5 倍遅い**

---

## 4. RT バジェットへの影響

### 4.1 計算基礎
| パラメータ | 値 |
|-----------|-----|
| RT バジェット | 5.33ms (192kHz/1024) |
| CPU_MIG 頻度（実測） | 1,110回/秒（全コールバックの87.6%で発生） |
| CPU_MIG コスト（推定値） | 400-750ns/イベント（String構築5回、値は推定） |

### 4.2 CPU_MIG の影響（推定値。実測ではない）
```
CPU_MIG コスト/秒 = 153.3 回/秒 × 750ns = 115.0μs/秒（推定値）
CPU_MIG 影響率 = 115.0μs / 5,333μs = 2.16%（推定値）
```

### 4.3 logEqTime の影響（推定値。実測ではない）
最大コスト/秒 = 187.5 回/秒 × 120ns = 22.5μs/秒（推定値）
影響率 = 22.5μs / 5,333μs = 0.42%（推定値）
`

### 4.4 Input タイミングへの影響（参考値。測定条件の明確化が必要）
現在の診断コードによる Input タイミングの劣化:

| メトリクス | 診断あり（実測） | 診断なし（参考値） | 劣化 |
|-----------|----------------|-----------------|------|
| P50 | 2.8ms | 0.3ms（異なる測定構成での参考値） | +933% |
| P90 | 3.1ms | 0.4ms（同上） | +675% |
| P95 | 3.2ms | 0.5ms（同上） | +540% |
| P99 | 3.6ms | 0.8ms（同上） | +350% |

注: 「診断なし」の値は String/Logger を完全に除去した別構成での測定結果であり、本設計の直接的な測定値ではない。実際の改善量は実装後に計測して確認する。

**診断コード自体が計測対象を大幅に乱していることは確かである（診断あり列は実測値）。**

---

## 5. Numeric-Only DiagEvent 設計

### 5.1 基本概念
`
RT スレッド（オーディオコールバック）
  ├─ 数値のみ収集（push() メソッドのみ）
  └─ LockFreeRingBuffer にプッシュ

Timer スレッド（非 RT）
  ├─ RingBuffer から読み出し
  └─ String フォーマット＆ファイル書き込み
`

### 5.2 DiagCategory 列挙型
`cpp
enum class DiagCategory : uint8_t {
    None = 0,
    CpuMig = 1,           // CPU_MIG イベント
    CallbackSequence = 2, // CB_SEQ イベント
    DspTiming = 3,        // DSP_TIMING イベント
    CallbackStage = 4,    // CALLBACK_STAGE イベント
    EqTime = 5,           // EQ_TIME イベント
    ConvTime = 6,         // CONV_TIME イベント
    StereoConvTime = 7,   // STCONV_TIME イベント
};
`

**注**: `category` フィールドは `uint8_t` 型とする。bitfield は使用しない（ABI/compiler/memcpy との相性問題回避）。

### 5.3 DiagEvent 構造体

**カテゴリ固有のデータ構造（調査結果に基づく正確な定義）:**

`cpp
// CPU_MIG 用データ（4 fields, 24 bytes）
// eventIndex は DiagEvent.Header から取得。generation も Header から取得可能だが、
// 世代ごとの集計に必要なため struct 内にも保持する。
struct CpuMigData {
    uint64_t pubSeq;      // publication sequence (Header.eventIndex ≠ pubSeq)
    uint64_t generation;  // runtimeWorld->generation
    uint32_t cpu;         // current CPU ID
    uint32_t prevCpu;     // previous CPU ID
};

// CB_SEQ 用データ（3 fields, 24 bytes）
// eventIndex は DiagEvent.Header から取得。
struct CallbackSequenceData {
    uint64_t generation;  // runtimeWorld->generation
    uint64_t seq;         // current publication sequence
    uint64_t prevSeq;     // previous publication sequence
};

// DSP_TIMING 用データ（一部条件付き）
// 注: サイズはコンパイラ・アーキテクチャ・alignas 設定で変動する（compiler dependent）
//   おおよその目安: 8(uint64_t)×7 + 1(uint8_t) + padding ≈ 57-64B
enum class PublicationDirection : uint8_t {
    None = 0,
    Forward = 1,    // publicationSequence が増加（正常前進）
    Rollback = 2,   // publicationSequence が減少（後退）
    Replay = 3      // 上記以外（現在の制御フローでは到達不能）
};

struct DspTimingData {
    uint64_t dspSeq;               // publication sequence（Header.eventIndexとは別の値）
    uint64_t generation;            // runtimeWorld->generation
    uint64_t worldId;               // runtimeWorld->worldId
    PublicationDirection  direction;   // enum class : uint8_t（型安全）
    // callbackIndex(=thisCallbackIndex)は Header.eventIndex から取得
    uint64_t observeLatencyUs;      // (条件付き) observeUs - matchedPublishEndUs
    uint64_t pubToObserveUs;        // (条件付き) 同上
    uint64_t callbacksUntilObserve; // (条件付き) thisCallbackIndex - matchedPublishCallbackIdx
    uint64_t publishCallbackIdx;    // (条件付き) matchedPublishCallbackIdx
};

// CALLBACK_STAGE 用データ
// 注: サイズはコンパイラ依存。uint64_t×7 + uint32_t + int64_t + uint16_t + padding ≈ 62-80B（compiler dependent）
// seq(thisCallbackIndex) は DiagEvent.Header.eventIndex から取得。
struct CallbackStageData {
    uint32_t cpu;              // current CPU ID
    uint64_t generation;        // runtimeWorld->generation
    uint64_t expectedUs;        // expected callback interval
    int64_t  driftUs;            // callback drift
    uint64_t inputUs;           // INPUT stage time
    uint64_t dspUs;             // DSP stage time
    uint64_t outputUs;          // OUTPUT stage time
    uint64_t totalUs;           // total callback time
    uint16_t budgetPermille;    // budget percentage in permille (e.g., 123 for 12.3%)
};

// EQ_TIME 用データ（5 fields, 20 bytes）
// seq(callbackSeq) は DiagEvent.Header.callbackSeq から取得。
struct EqTimeData {
    uint32_t cpu;         // current CPU ID
    uint64_t us;          // eqElapsedUs - EQ elapsed time
    uint8_t  activeBands; // number of active bands
    uint8_t  order;       // 0=Conv->EQ, 1=EQ->Conv
    uint32_t budgetPercent; // budget percentage (e.g., 125 for 12.5%)
};

// CONV_TIME 用データ（7 fields, 32 bytes、一部条件付き）
// seq(cbIdx/eventIndex) は DiagEvent.Header.eventIndex から取得。
struct ConvTimeData {
    uint32_t cpu;         // current CPU ID
    uint64_t us;          // convElapsedUs
    uint32_t blockSamples; // block.getNumSamples()
    uint16_t budgetPercent; // (elapsedUs / expectedUs) * 100
    uint32_t expectedUs;  // (blockSamples / srForConv) * 1e6
    uint32_t callQuantumSamples; // (条件付き) conv->callQuantumSamples
    uint32_t latency;     // (条件付き) conv->latency
};

// STCONV_TIME 用データ（5 fields, 20 bytes）
// seq(cbIdx/eventIndex) は DiagEvent.Header.eventIndex から取得。
struct StereoConvTimeData {
    uint32_t cpu;         // current CPU ID
    uint64_t us;          // scElapsedUs - stereo convolution time
    uint32_t chunkSamples; // chunk size
    uint8_t  channels;    // ch - channel count
    uint16_t budgetPercent; // (scElapsedUs / expectedUs) * 100
};
`

**メイン構造体:**

```cpp
enum class DiagCategory : uint8_t {
    None = 0,
    CpuMig = 1,
    CallbackSequence = 2,
    DspTiming = 3,
    CallbackStage = 4,
    EqTime = 5,
    ConvTime = 6,
    StereoConvTime = 7,
};

struct DiagEvent {
    DiagCategory category;  // enum class : uint8_t（生の uint8_t ではなく型安全な enum で保持）
                            // キャスト不要、switch が安全、他値代入防止。ABI は uint8_t と同一。
    uint64_t eventIndex;    // グローバルコールバック通し番号（audioCallbackEpochCounter）。
                            // 全イベントに共通する時系列順序付け用。各イベント固有の seq とは別。
                            // (例) CPU_MIG では pubSeq が別途必要、CALLBACK_STAGE では eventIndex が seq として機能
                            // ★「callbackSeq」から「eventIndex」に名称変更（2026-07-02）:
                            //   イベント固有の seq/pubSeq/dspSeq との混同を防止。注意: 「epoch」という用語は
                            //   RCU/ISR Runtime では世代番号を意味するため、ここでは「callback index」の意。

    union {
        CpuMigData cpuMig;                  // 24B (pubSeq, gen, cpu, prevCpu)
        CallbackSequenceData callbackSequence; // 24B (gen, seq, prevSeq)。Header の eventIndex とは別、publicationSequence
        DspTimingData dspTiming;             // ~64B (dspSeq, gen, worldId, reason, obsLat, pubObs, cbsUntilObs, pubCbIdx) 最大メンバー
        CallbackStageData callbackStage;     // 62B (cpu, gen, expected, drift, input, dsp, output, total, budget)
        EqTimeData eqTime;                  // 18B (cpu, us, bands, order, budget)
        ConvTimeData convTime;               // 30B (cpu, us, block, budget, expected, callQuantum, latency)
        StereoConvTimeData stereoConvTime;   // 19B (cpu, us, chunk, ch, budget)
    } data;
};

static_assert(std::is_trivially_copyable_v<DiagEvent>,
    "DiagEvent must be trivially copyable for LockFreeRingBuffer");
static_assert(std::is_standard_layout_v<DiagEvent>,
    "DiagEvent must be standard layout for memcpy safety");
static_assert(std::is_trivial_v<DiagEvent>,
    "DiagEvent must be trivial (trivially_copyable + default_constructible)");
static_assert(std::is_trivially_destructible_v<DiagEvent>,
    "DiagEvent must be trivially destructible; check data members");
static_assert(alignof(DiagEvent) == alignof(uint64_t),
    "DiagEvent alignment mismatch; check struct members");
static_assert(offsetof(DiagEvent, data) % alignof(uint64_t) == 0,
    "DiagEvent.data must be uint64_t-aligned for efficient union access");
// TODO 実装時: 実際に sizeof(DiagEvent) を出力確認した後、
//   下の <= を == に置き換え、値を実測値に固定すること。例:
//   static_assert(sizeof(DiagEvent) == 88, "DiagEvent layout changed; review alignment");
// 期待値: Header(1+7pad+8=16B) + max union(~64B) = ~80-88B（alignas/compiler で変動）
// 注意: <= のまま放置すると padding 変化を見逃すリスクがある。
//       実装後は sizeof(DiagEvent) を出力確認し、== に変更すること。
static constexpr size_t kExpectedDiagEventSize = 96;
static_assert(sizeof(DiagEvent) <= kExpectedDiagEventSize,
    "DiagEvent size exceeded; replace <= with == after confirming actual sizeof");
```

**generation は Header ではなく各構造体に保持:**

ソースコード検証の結果、以下の事実が判明した:

| イベントタイプ | generation有無 | 件数/秒 | 根拠 |
|---------------|---------------|---------|------|
| CPU_MIG | ✅ 必要 (gen=3/4/5) | 153.3/s | 実行世代の特定に必須 |
| CALLBACK_STAGE | ✅ 必要 | 10.9/s | 世代ごとの統計集計に使用 |
| DSP_TIMING | ✅ 必要 | <0.1/s | 世代切替のタイミング分析 |
| CB_SEQ | ✅ 必要 | <0.1/s | 世代変化の追跡が本質 |
| CONV_TIME | ❌ 不要 | 9.4/s | cbIdx/eventIndexのみで十分 |
| STCONV_TIME | ❌ 不要 | 18.7/s | cbIdx/eventIndexのみで十分 |
| EQ_TIME | ❌ 不要 | 間引き | cbIdx/eventIndexのみで十分 |

判定: **generation を Header に強制しない。** CONV_TIME (1,833件/195.6s) や STCONV_TIME (3,666件) は generation を出力しておらず、Header に追加すると 8B × 5,499イベント = 44KB の無駄になる。

**カテゴリ別リングバッファ（検討事項）:**

単一 `DiagEvent` + `union` とカテゴリ別リングバッファの比較:

| 方式 | 総メモリ | 管理対象数 | switch文 | 拡張性 |
|------|---------|-----------|---------|-------|
| 単一 union (512slot) | ~44KB | 1 buffer | 必要 | 容易 |
| カテゴリ別 (7 buffers) | ~39KB | 7 buffers | 不要 | 複雑 |

判定: メモリ差はわずか 5KB (11%) であり、保守性を考慮して **単一 union 方式を採用する。** カテゴリ別バッファは拡張性と管理コストが釣り合わない。

**ドロップカウンタ（新設）:**

```cpp
// ★ 実行時定数（DiagRuntimeLimits 構造体にまとめて管理）
//    Timer 側の kMaxDiagEventsPerTick もここ参照。
//    ★ 各 atomic 統計カウンタは個別の DiagPerTickCounter（alignas(64)）インスタンス。
//      RT(write) と Timer(read&reset) が競合しても隣接オブジェクトへの影響を防止。
struct DiagRuntimeLimits {
    static constexpr size_t BufferCapacity = 512;   // 2^9、~45KB
    static constexpr size_t MaxDrainPerTick = 64;   // Timer 1回あたり最大処理件数
};

// DiagEvent バッファ（LockFreeRingBuffer）
static constexpr size_t DIAG_EVENT_BUFFER_CAPACITY = DiagRuntimeLimits::BufferCapacity;
LockFreeRingBuffer<DiagEvent, DIAG_EVENT_BUFFER_CAPACITY> diagBuffer;

// ★ 新設: リングバッファ運用統計カウンタ
//   - pushed/popped/dropped: per-tick（exchangeAtomic でリセット）
//   - totalPushed/totalPopped: モノトニック（一度もリセットしない）
//     現在の占有数近似値 = diagBuffer.size()（ベストエフォート）、累積トレンド = totalPushed - totalPopped
// ★ DiagPerTickCounter: per-tick統計カウンタ。各インスタンスが alignas(64) で自身のキャッシュラインを占有。
//   3カウンタはそれぞれ独立した DiagPerTickCounter インスタンスとして宣言する。
//   ★ DiagPerTickCounter 自体が alignas(64) なので各インスタンスは64B境界に配置される。
//     ただし RTAuxMutable 構造体内での配置順次第では隣接カウンタが同一キャッシュラインに
//     乗る可能性がゼロではない（コンパイラの struct レイアウト依存）。
//     この用途（数十件/sec）では実用上問題にならず、過剰な最適化は避ける。
//   ★ diagTotalPushed/Popped は alignas(64) 付きの個別 atomic<uint64_t> 変数。
//     ただし atomic は8Bなので64Bライン上に間隔が空くとは限らない。他メンバとの兼ね合いで
//     同一ライン競合は起こり得るが、統計カウンタとしては実用上十分。
//   実体は RTAuxMutable 構造体のメンバ変数として宣言する（static グローバルではなく）。
//   ★ 以下のコードは説明用。実際の宣言位置は AudioEngine.h の RTAuxMutable 内。
struct alignas(64) DiagPerTickCounter {
    std::atomic<uint64_t> value { 0 };
};

// RTAuxMutable 内のメンバ変数（xRunDropCount と同じ定義パターン）:
// ★ per-tick 統計: 毎 Timer tick で exchangeAtomic リセット
//   DiagPerTickCounter diagTickPushed;   // push成功数
//   DiagPerTickCounter diagTickPopped;   // pop成功数（Timer側で増加）
//   DiagPerTickCounter diagTickDropped;  // drop数
//
// ★ モノトニックカウンタ（一度もリセットしない）。バッファ利用率の累積トレンド把握用。
//   alignas(64) std::atomic<uint64_t> diagTotalPushed { 0 };
//   alignas(64) std::atomic<uint64_t> diagTotalPopped { 0 };
```

**サイズ計算（推定。実際は `sizeof(DiagEvent)` で確認すべき）:**
- 最大メンバー: DspTimingData (dspSeq, gen, worldId, reason, observeLatencyUs, pubToObserveUs, callbacksUntilObserve, publishCallbackIdx)
  callbackIndex は Header で統一したため削減。ObserveExtra 4フィールドは分離候補。
  → 約64B（アラインメントで変動）
- DiagEvent のレイアウト:
  - `category` (1 byte, uint8_t)
  - `padding` (7 bytes) ← uint8_t の後に uint64_t が来るときのアラインメント
  - `callbackSeq` (8 bytes, uint64_t)
  - `data` union (最大 ~64B, DspTimingData)
  - **合計 (推定): 16 + ~64 = ~80B（alignas 調整で 88B 程度になる可能性あり）**
- キャッシュライン: 2 つ (128 bytes)
- **注**: 実際のサイズは `sizeof(DiagEvent)` と `alignof(DiagEvent)` で確認すること。
  実装時は `static_assert(sizeof(DiagEvent) == 実測値)` でレイアウト変更を検出できるようにする。

**重要な設計上の決定:**
1. `category` は `uint8_t` 型（bitfield ではない）
2. `seq` は callbackSeq または callbackIndex（timestampUs は不要）
3. `union` を使用（std::variant ではない - RT 安全）
4. 全メンバーが POD 型であるため trivially_copyable
5. **DspTimingData の ObserveExtra 分離（検討事項）**: `observeLatencyUs`, `pubToObserveUs`, `callbacksUntilObserve`, `publishCallbackIdx` は Observe イベント時のみ意味を持つ。これらを `struct ObserveExtra` として切り出し、union の別メンバーにすれば DspTimingData のサイズを削減できる（約64B→約32B、compiler dependent）。ただし現状のサイズでも実用上問題ないため、必要に応じて段階的に実装する。
6. **64件制限の constexpr 定数化（検討事項）**: `kMaxDiagEventsPerTick = 64` は `DiagRuntimeLimits` のような構造体にまとめると将来変更しやすい。

### 5.4 利点
1. **RT 安全**: push() メソッドのみ（String 構築なし、ログ I/O なし）
2. **高速**: String 構築（400-750ns推定）と比較して十分小さい。なお push() 内部では atomic acquire×2 + 条件判定 + コピー代入 + atomic release が発生するため、「<50ns」などの数値は実測による確認が必要。
3. **非 RT でフォーマット**: Timer スレッドで遅延許容
4. **一貫性**: 既存の xRunBuffer パターン（LockFreeRingBuffer）に従う
5. **メモリ効率**: Union によりカテゴリ固有データのみ使用
6. **タイプセーフ**: カテゴリごとの専用構造体
7. **ISR Runtime 高互換性**: trivially_copyable、ロックフリー、動的メモリ割り当てなし
8. **Timer バッチ制限 64 件/tick**: GUI Timer のスパイク防止。残りは次回 tick で処理
9. **DiagStatistics 運用統計**: per-tick統計（diagTickPushed/Popped/Dropped、exchangeAtomic でリセット）+ モノトニック（diagTotalPushed/Popped、リセットなし、`load(relaxed)` 読取）。`approxOcc ≈ diagBuffer.size()`（O(1)ロックフリー、SPSC acquire×2、厳密な瞬間値ではない）。`backlog = totalPushed - totalPopped`（累積処理待ち件数、lifetime backlog）。各カウンタは `alignas(64)` 配置（コンパイラ依存あり・統計用途では問題なし）。バッファ容量は実測データで決定。

---

## 6. 実装計画

### 6.1 AudioEngine.h への追加
`cpp
// DiagCategory 列挙型と DiagEvent 構造体を追加（上記参照）

// DiagEvent バッファ（LockFreeRingBuffer）
// 512 = 2^9。平均~10イベント/tick、Timer遅延に備えて512を採用。
// メモリ: 512 events × 88 bytes ≈ 45 KB
static constexpr size_t DIAG_EVENT_BUFFER_CAPACITY = 512;  // 512 = 2^9
LockFreeRingBuffer<DiagEvent, DIAG_EVENT_BUFFER_CAPACITY> diagBuffer;

// 容量計算: 512 events × 88 bytes ≈ 45 KB
`

### 6.2 修正が必要な 9 箇所（調査結果に基づく更新）

| # | ファイル | 行 | カテゴリ | 現在のコード |
|---|---------|-----|---------|-------------|
| 1 | AudioBlock.cpp | 173 | CpuMig | juce::Logger::writeToLog(...5x String...) |
| 2 | BlockDouble.cpp | 155 | CpuMig | juce::Logger::writeToLog(...5x String...) |
| 3 | AudioBlock.cpp | 182-200 | CallbackSequence | juce::Logger::writeToLog(...4x String...) |
| 4 | AudioBlock.cpp | 549-561+ | DspTiming | juce::Logger::writeToLog(...13x String...) |
| 5 | AudioBlock.cpp | 574-608 | CallbackStage | juce::Logger::writeToLog(...12x String...) |
| 6 | DSPCoreFloat.cpp | 33-65 | EqTime | logEqTime 内の 5x String 構築 |
| 7 | DSPCoreDouble.cpp | 516-538 | EqTime | logEqTime 内の 5x String 構築 |
| 8 | ConvolverProcessor.Runtime.cpp | 726-732 | ConvTime | juce::Logger::writeToLog(...6x String...) |
| 9 | ConvolverProcessor.Runtime.cpp | 658-664 | StereoConvTime | juce::Logger::writeToLog(...5x String...) |

**注:**
- CpuMig, CallbackSequence, DspTiming は無サンプリング（イベント発生時のみ出力）
- CallbackStage, EqTime, ConvTime, StereoConvTime は 1/16 サンプリング
- String 構築数は調査結果に基づく

### 6.3 修正パターン例（CPU_MIG）

**現在のコード:**

`cpp
if (CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS && cpu != prev) {
    juce::Logger::writeToLog(
        "[CPU_MIG] callback=" + juce::String(static_cast<int64_t>(cbIdx))
        + " seq=" + juce::String(static_cast<int64_t>(pubSeq))
        + " gen=" + juce::String(static_cast<int64_t>(gen))
        + " cpu=" + juce::String(static_cast<int64_t>(cpu))
        + " prev=" + juce::String(static_cast<int64_t>(prev)));
}
`

**修正後のコード:**

`cpp
if (CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS && cpu != prev) {
    // Numeric-Only RT サイド（push() のみ、ユーザーコードで memcpy を書く必要がない）
    // ★ {} でゼロ初期化: union 未使用領域の未初期化データを排除（将来の memcpy/memdump 対策）
    // ★ category は enum class DiagCategory 型。static_cast 不要。
    DiagEvent event{};
    event.category = DiagCategory::CpuMig;
    event.eventIndex = cbIdx;  // ユニバーサルコールバック番号
    event.data.cpuMig.pubSeq = pubSeq;  // 出版シーケンス（callbackSeq とは別）
    event.data.cpuMig.generation = gen;
    event.data.cpuMig.cpu = cpu;
    event.data.cpuMig.prevCpu = prev;

    // LockFreeRingBuffer にプッシュ。成功/失敗を問わず統計を記録。
    // ★ memory_order_relaxed: 統計カウンタであり、同期用途ではないため。
    //   ISR Runtime: 診断情報は「全部残す」より「RTを守る」が優先。drop は許容、drop数は把握。
    if (diagBuffer.push(event))
    {
        diagTickPushed.value.fetch_add(1, std::memory_order_relaxed);
        diagTotalPushed.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        diagTickDropped.value.fetch_add(1, std::memory_order_relaxed);
    }
}
`

**重要な点:**

1. `diagBuffer.push(event)` + `diagTickPushed.value`(成功) / `diagTickDropped.value`(失敗) の `fetch_add(1, relaxed)` で完結（ISR Runtime 思想: drop は許容、統計は把握）
2. `DiagEvent event{}` でゼロ初期化（union 未使用領域のゴミを防止）
3. `category` は `uint8_t` 型（bitfield ではない）
4. Header の `callbackSeq` は全イベント共通の時系列順序付け、union 側の `pubSeq` 等はイベント固有の意味
5. `union` を使用（std::variant ではない）
6. xRunBuffer と同じパターンで実装（動作実証済み）

### 6.4 Timer スレッドでの処理

`cpp
void AudioEngine::timerCallback()
{
    // ... 既存の処理 ...

    // DiagEvent バッファを処理（LockFreeRingBuffer.pop() を使用）
    // ★ xRunBuffer と同じ while(pop()) パターン。
    //    ただし GUI Timer のスパイク防止のため 1回あたり最大 64 件に制限。
    //    残りは次回の Timer  tick で処理される（drop は発生しない）。
    static constexpr size_t kMaxDiagEventsPerTick = DiagRuntimeLimits::MaxDrainPerTick;
    size_t diagProcessedThisTick = 0;
    DiagEvent event;
    // ★ 条件順序: 上限チェックを先に。逆順だと64件超えたpop()を捨てることになる。
    while (diagProcessedThisTick < kMaxDiagEventsPerTick
        && diagBuffer.pop(event))
    {
        ++diagProcessedThisTick;
        diagTickPopped.value.fetch_add(1, std::memory_order_relaxed);
        diagTotalPopped.fetch_add(1, std::memory_order_relaxed);  // ★ モノトニック累積カウンタ（リセットなし）
        // String フォーマット（非 RT 安全）
        // ★ Header の callbackSeq を各イベントの seq として使用（全イベントに共通）
        const uint64_t cbIdx = event.eventIndex;
        juce::String log;
        switch (event.category) {
                case DiagCategory::CpuMig:
                    log = "[CPU_MIG] callback=" + juce::String(static_cast<int64_t>(cbIdx))
                        + " seq=" + juce::String(static_cast<int64_t>(event.data.cpuMig.pubSeq))
                        + " gen=" + juce::String(static_cast<int64_t>(event.data.cpuMig.generation))
                        + " cpu=" + juce::String(static_cast<int>(event.data.cpuMig.cpu))
                        + " prev=" + juce::String(static_cast<int>(event.data.cpuMig.prevCpu));
                    break;

                case DiagCategory::CallbackSequence:
                    log = "[CB_SEQ] callback=" + juce::String(static_cast<int64_t>(cbIdx))
                        + " seq=" + juce::String(static_cast<int64_t>(event.data.callbackSequence.seq))
                        + " gen=" + juce::String(static_cast<int64_t>(event.data.callbackSequence.generation))
                        + " prevSeq=" + juce::String(static_cast<int64_t>(event.data.callbackSequence.prevSeq));
                    break;



                case DiagCategory::CallbackStage:
                    log = "[CALLBACK_STAGE] seq=" + juce::String(static_cast<int64_t>(cbIdx))
                        + " cpu=" + juce::String(static_cast<int>(event.data.callbackStage.cpu))
                        + " gen=" + juce::String(static_cast<int64_t>(event.data.callbackStage.generation))
                        + " expected=" + juce::String(static_cast<int64_t>(event.data.callbackStage.expectedUs))
                        + " drift=" + juce::String(static_cast<int64_t>(event.data.callbackStage.driftUs))
                        + " input=" + juce::String(static_cast<int64_t>(event.data.callbackStage.inputUs))
                        + " dsp=" + juce::String(static_cast<int64_t>(event.data.callbackStage.dspUs))
                        + " output=" + juce::String(static_cast<int64_t>(event.data.callbackStage.outputUs))
                        + " total=" + juce::String(static_cast<int64_t>(event.data.callbackStage.totalUs))
                        + " budget=" + juce::String(event.data.callbackStage.budgetPermille / 10) + "." + juce::String(event.data.callbackStage.budgetPermille % 10) + "%";
                    break;

                case DiagCategory::EqTime:
                    log = "[EQ_TIME] seq=" + juce::String(static_cast<int64_t>(cbIdx))
                        + " cpu=" + juce::String(static_cast<int>(event.data.eqTime.cpu))
                        + " us=" + juce::String(static_cast<int64_t>(event.data.eqTime.us))
                        + " bands=" + juce::String(static_cast<int>(event.data.eqTime.activeBands))
                        + " order=" + juce::String(static_cast<int>(event.data.eqTime.order))
                        + " budget=" + juce::String(event.data.eqTime.budgetPercent / 10) + "." + juce::String(event.data.eqTime.budgetPercent % 10) + "%";
                    break;

                case DiagCategory::ConvTime:
                    log = "[CONV_TIME] seq=" + juce::String(static_cast<int64_t>(cbIdx))
                        + " cpu=" + juce::String(static_cast<int>(event.data.convTime.cpu))
                        + " us=" + juce::String(static_cast<int64_t>(event.data.convTime.us))
                        + " block=" + juce::String(static_cast<int>(event.data.convTime.blockSamples))
                        + " budget=" + juce::String(event.data.convTime.budgetPercent / 10) + "." + juce::String(event.data.convTime.budgetPercent % 10) + "%"
                        + " expected=" + juce::String(static_cast<int>(event.data.convTime.expectedUs))
                        + " callQ=" + juce::String(static_cast<int>(event.data.convTime.callQuantumSamples))
                        + " lat=" + juce::String(static_cast<int>(event.data.convTime.latency));
                    break;

                case DiagCategory::StereoConvTime:
                    log = "[STCONV_TIME] seq=" + juce::String(static_cast<int64_t>(cbIdx))
                        + " cpu=" + juce::String(static_cast<int>(event.data.stereoConvTime.cpu))
                        + " us=" + juce::String(static_cast<int64_t>(event.data.stereoConvTime.us))
                        + " chunk=" + juce::String(static_cast<int>(event.data.stereoConvTime.chunkSamples))
                        + " ch=" + juce::String(static_cast<int>(event.data.stereoConvTime.channels))
                        + " budget=" + juce::String(event.data.stereoConvTime.budgetPercent / 10) + "." + juce::String(event.data.stereoConvTime.budgetPercent % 10) + "%";
                    break;

                case DiagCategory::DspTiming:
                {
                    // ★ PublicationDirection をログ解析しやすい文字列へ変換
                    //   RT側に影響なし、Timer側の文字列化で完結。
                    static const char* directionToString(PublicationDirection d) noexcept {
                        switch (d) {
                            case PublicationDirection::None: return "None";
                            case PublicationDirection::Forward: return "Forward";
                            case PublicationDirection::Rollback: return "Rollback";
                            case PublicationDirection::Replay: return "Replay";
                            default: return "Unknown";  // 将来の拡張に備える
                        }
                    }
                    log = "[DSP_TIMING] seq=" + juce::String(static_cast<int64_t>(event.data.dspTiming.dspSeq))
                        + " gen=" + juce::String(static_cast<int64_t>(event.data.dspTiming.generation))
                        + " worldId=" + juce::String(static_cast<int64_t>(event.data.dspTiming.worldId))
                        + " direction=" + juce::String(directionToString(event.data.dspTiming.direction))
                        + " callbackIndex=" + juce::String(static_cast<int64_t>(cbIdx))
                        + " observeLatencyUs=" + juce::String(static_cast<int64_t>(event.data.dspTiming.observeLatencyUs))
                        + " pubToObserveUs=" + juce::String(static_cast<int64_t>(event.data.dspTiming.pubToObserveUs))
                        + " callbacksUntilObserve=" + juce::String(static_cast<int64_t>(event.data.dspTiming.callbacksUntilObserve))
                        + " publishCallbackIdx=" + juce::String(static_cast<int64_t>(event.data.dspTiming.publishCallbackIdx));
                    break;

                default:
                    // ★ 未知のカテゴリ。開発時は jassertfalse で検出、Release では UNKNOWN ログを出力
                    jassertfalse;
                    log = "[UNKNOWN] category=" + juce::String(static_cast<int>(event.category));
                    break;
            }

            diagLog(log);  // ★ Timer.cpp の既存パターン diagLog() を使用（DBG + Logger::writeToLog をラップ）
        }
    }

    // ★ PublicationDirection の文字列変換は DSP_TIMING case 内の directionToString() で完結している。
    //   将来 direction 値が増えた場合も、directionToString() の switch と enum の拡張だけで対応可能。

    // ★ DiagStatistics: リングバッファ運用統計
    //   per-tick統計（diagTickPushed/Popped/Dropped）は exchangeAtomic で読み取り＆リセット（xRunDropCount と同一パターン）。
    //   モノトニックカウンタ（diagTotalPushed/Popped）はリセットしない。
    //   ★ 注意: diagBuffer.size() は SPSC リングバッファの writeIndex-readIndex を atomic acquire で読み取るベストエフォート値。
    //     厳密な瞬間値ではない（近似値として扱うこと）。
    //   ISR Runtime ではバッファ容量は推測ではなく運用データで決める。
    {
        const uint64_t gen = (runtimeWorld != nullptr)
            ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
        const uint64_t pushed  = convo::exchangeAtomic(
            diagTickPushed.value, 0, std::memory_order_acq_rel);
        const uint64_t popped  = convo::exchangeAtomic(
            diagTickPopped.value, 0, std::memory_order_acq_rel);
        const uint64_t dropped = convo::exchangeAtomic(
            diagTickDropped.value, 0, std::memory_order_acq_rel);
        const uint64_t totalPushed = diagTotalPushed.load(
            std::memory_order_relaxed);
        const uint64_t totalPopped = diagTotalPopped.load(
            std::memory_order_relaxed);

        if (dropped > 0 || pushed > 0)
        {
            const auto tickTotal = pushed + dropped;
            if (tickTotal > 0)
            {
                const double dropRate = 100.0 * static_cast<double>(dropped) / static_cast<double>(tickTotal);
                // ★ backlog = totalPushed - totalPopped: 累積処理待ち件数（リング占有数の長期トレンド）
                //   approxOcc ≈ diagBuffer.size() の近似値（SPSC の atomic load なので瞬間的に不正確）
                const uint64_t approxOccupancy = diagBuffer.size();
                const uint64_t backlog = totalPushed - totalPopped;
                diagLog(diagPrefix(gen)
                    + " [DIAG_STAT] pushed=" + juce::String(static_cast<juce::int64>(pushed))
                    + " popped=" + juce::String(static_cast<juce::int64>(popped))
                    + " dropped=" + juce::String(static_cast<juce::int64>(dropped))
                    + " approxOcc=" + juce::String(static_cast<juce::int64>(approxOccupancy))
                    + " backlog=" + juce::String(static_cast<juce::int64>(backlog))
                    + " (dropRate=" + juce::String(dropRate, 2)
                    + "%)");
            }
        }
    }
}
`

**重要な点:**

1. `diagBuffer.pop(event)` を使用。上限 `kMaxDiagEventsPerTick=64` で GUI Timer の長時間占有を防止
2. `event.category` は `DiagCategory` 型（enum class）。`static_cast` 不要で `switch (event.category)` が直接使える
3. 各カテゴリのデータは `event.data.xxx.xxx` でアクセス（union 経由）
4. 全カテゴリのケースを処理（switch() 文は approved）
5. `diagTickPushed/Popped/Dropped` の per-tick統計カウンタは `exchangeAtomic` で読み取り＆リセット（xRunDropCount と同一パターン）。`diagTotalPushed/Popped` はモノトニック（リセットなし）で `load(relaxed)` 読み取り（統計用途のため relaxed で十分）。`approxOcc ≈ diagBuffer.size()`（SPSCのatomic load acquire×2, O(1)ロックフリー、厳密な瞬間値ではない）。`backlog = totalPushed - totalPopped` で累積処理待ちトレンドを監視。各 atomic は `alignas(64)` で配置（コンパイラの struct レイアウト次第では同一ライン競合の可能性あり。数十件/sec では実用上問題にならない）。

---

## 7. 期待される改善

### 7.1 String 構築コストの削減（推定値）

| 項目 | 現在（推定値） | 修正後（推定値） | 改善効果（推定） |
|------|------|--------|------|
| CPU_MIG コスト | 400-750ns（String構築, 推定） | String構築なし（push()はatomic+代入程度、実測要） | String構築時間分縮小（推定） |
| logEqTime コスト | 60-120ns（String構築, 推定） | String構築なし（同上） | String構築時間分縮小（推定） |
| 全体的な改善 | - | - | **顕著（String排除によるRT安全性向上）** |

### 7.2 RT バジェットへの影響削減（推定値）

```
現在の CPU_MIG 影響（推定値）: 115.0μs/秒 = 2.16%（推定値。String構築5回 + Logger呼び出しベース）
修正後の CPU_MIG 影響: push()のみとなるため大幅に低減されると期待されるが、具体的な数値は実装後に実測で検証する。
改善効果: String構築 + Logger呼び出し + File I/O を排除できるため、RT負荷は大幅に低減すると期待される。
```

### 7.3 Input タイミングへの期待値

**現状の診断コードが Input フェーズに与えている影響（実測値）:**

| メトリック | 診断あり（実測） | 診断なし（期待値、実測による確認が必要） |
|-----------|-----------------|--------------------------------------|
| P50 | 2.8ms | String/Logger排除により大幅短縮（実測要） |
| P90 | 3.1ms | 同上 |
| P95 | 3.2ms | 同上 |
| P99 | 3.6ms | 同上 |

**診断コードの影響を大幅に低減可能と期待されるが、具体的な改善量は実装後に計測して確認する。**
特に、CPU_MIG(153回/s)の String 構築＋Logger 呼び出しは RT スレッド上で唯一の重い処理であり、
これを排除することで、Input フェーズのレイテンシとそのばらつきが改善されると見込まれる。

---

## 8. 調査結果

このレポート作成にあたり、以下の詳細調査を実施しました。

### 8.1 DiagCategory 列挙型の存在確認
- **結果**: 既存の DiagCategory 列挙型は見つからず
- **結論**: 新規追加が必要

### 8.2 timestampUs 使用パターンの調査
- **結果**: timestampUs は診断ログでは使用されていない
- **確認された代替**: callbackSeq, callbackIndex, generation が利用可能
- **結論**: timestampUs 不要、callbackSeq または callbackIndex を使用

### 8.3 callbackSeq / generation の可用性確認
- **結果**: すべての診断カテゴリで利用可能
- **CPU_MIG**: pubSeq + gen (generation) ✅
- **CB_SEQ**: cbIdx/eventIndex + gen ✅
- **DSP_TIMING**: seq + generation ✅
- **CALLBACK_STAGE**: seq + generation ✅
- **EQ_TIME**: seq ✅
- **CONV_TIME**: seq ✅
- **STCONV_TIME**: seq ✅

### 8.4 7 つの診断カテゴリのデータ構造調査

#### CPU_MIG (5 fields, 40 bytes)
- callback, generation, seq, cpu, prevCpu

#### CB_SEQ (4 fields, 32 bytes)
- callback, generation, seq, prevSeq

#### DSP_TIMING (9 fields + 3 conditional, size compiler dependent)
- seq, generation, worldId, reason, callbackIndex, observeLatencyUs, pubToObserveUs, callbacksUntilObserve, publishCallbackIdx
- conditional: observeLatencyUs (when reason != NoError), pubToObserveUs (when observeLatencyUs valid), callbacksUntilObserve (when pubToObserveUs valid)

#### CALLBACK_STAGE (10 fields, 80 bytes)
- seq, cpu, generation, expectedUs, driftUs, inputUs, dspUs, outputUs, totalUs, budgetPermille

#### EQ_TIME (6 fields, 32 bytes)
- seq, cpu, us, activeBands, order, budgetPercent

#### CONV_TIME (8 fields, 56 bytes)
- seq, cpu, us, blockSamples, budgetPercent, expectedUs, callQuantumSamples, latency

#### STCONV_TIME (6 fields, 40 bytes)
- seq, cpu, us, chunkSamples, channels, budgetPercent

### 8.5 xRunBuffer 実装パターンの調査
- **確認箇所**: `src/audioengine/AudioEngine.h:2053`
- **パターン**: `LockFreeRingBuffer<XRunEvent, kXRunBufferCapacity> xRunBuffer`
- **使用方法**: `xRunBuffer.push(ev)` - memcpy 不要
- **結論**: 診断バッファも同じパターンで実装可能

### 8.6 サンプリング頻度の確認
- **確認箇所**: `src/DiagnosticsConfig.h:20-21`
- **定義**: `#define CONVOPEQ_DIAG_SAMPLE_MASK 0xF` (1/16 サンプリング)
- **結論**: 既存のサンプリング設定を維持

---

## 9. 次のステップ

### 9.1 ファイル保存完了
✅ このレポートを doc/work60/ に保存

### 9.2 実装開始（ユーザーの許可待ち）
1. AudioEngine.h に DiagCategory と DiagEvent を追加
2. 8 箇所の診断コードを Numeric-Only に修正
3. Timer スレッドで String フォーマットを追加
4. テスト＆検証

### 9.3 注意事項
- **絶対にレポート保存後、ユーザーの許可なく実装を開始しないこと**
- ユーザーの指示: "まずは、レポートを doc\work60\ に保存してください。許可があるまで絶対に実装を開始しないでください"

---

## 10. 付録

### 10.1 既存の成功事例: xRunBuffer
`cpp
// LockFreeRingBuffer の使用例（動作実証済み）
juce::AbstractFifo xRunBuffer{xRunBufferSize};
std::array<XRunEvent, xRunBufferSize> xRunBufferData;

// RT 側（write）
const auto scope = xRunBuffer.write(1);
if (scope.blockSize1 == 1) {
    std::memcpy(&xRunBufferData[scope.startIndex1], &xrun, sizeof(XRunEvent));
}

// Timer 側（read）
const auto scope = xRunBuffer.read(1);
if (scope.blockSize1 == 1) {
    const XRunEvent& event = xRunBufferData[scope.startIndex1];
    // String フォーマット＆ファイル書き込み
}
`

### 10.2 JUCE FileLogger の RT 違反
`cpp
void FileLogger::logMessage(const String& message)
{
    const ScopedLock sl(lock);           // ❌ SpinLock（RT 違反）
    if (outputStream == nullptr)
        openFile();                       // ❌ ファイルオープン（RT 違反）

    outputStream->writeString(message);   // ❌ 同期書き込み（RT 違反）
    outputStream->writeString("\n");
    outputStream->flush();                // ❌ Flush（RT 違反）
}
`

### 10.3 CPU 追跡機構
`cpp
// AudioEngine.h:1283
std::atomic<uint32_t> lastCallbackProcessor{0};

// AudioBlock.cpp:158-163
auto cpu = GetCurrentProcessorNumber();
auto prev = lastCallbackProcessor.exchange(cpu);
if (CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS && cpu != prev) {
    // CPU_MIG 検出
}
`

---

**レポート終了**

このレポートは、診断コードが RT スレッドに与える影響を詳細に分析し、Numeric-Only DiagEvent 構造体による解決策を提案しています。実装開始前に、必ずユーザーの許可を得ること。
