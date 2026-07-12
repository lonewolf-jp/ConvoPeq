# メモリ占有調査のためのインストルメンテーション改修案 v3 — 3階層 allocatedBytes + RetireQueue 計測

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v2 からの変更**: 全確保・解放を正しく追跡する 3階層カウンタ + RetireQueue 指標追加。

---

## 0. 設計思想（v2 からの修正点）

### 修正①: allocatedBytes を解放時にも減算する

v2 では `mkl_free` 時に減算できず、累積値になっていた。
v3 では **各 Layer に `size_t layerAllocated` を保持** し、`freeAll()` で `globalAllocated.fetch_sub(layerAllocated)` する。

### 修正②: nullptr ガード

全 `DIAG_ALLOC` 呼び出しで `if (ptr)` チェックを行い、`mkl_malloc` 失敗時はカウントしない。

### 修正③: 3階層カウンタ構造

```text
Global (static, 全インスタンス合計)
  └─ Instance (コンストラクタで 0 初期化、各インスタンス固有)
       └─ Layer (freeAll() で instance から減算)
```

### 修正④: RetireQueue + EpochDomain 指標を追加

`pendingRetireCount()`, `quarantineCount`, `reclaimAttemptCount` を PUBLISH ログに含める。

### 修正⑤: DSPCore++/-- イベントログ廃止

代わりに Publish 時スナップショットに集約する。

---

## 1. Patch A: MKLNonUniformConvolver 3階層カウンタ

### A-1. Layer 構造体に `layerAllocated` 追加

**ファイル**: `src/MKLNonUniformConvolver.h`
**位置**: `struct Layer` 内、`tailOutputBuf` の隣（AoS 領域の終端）

```cpp
        double* tailOutputBuf = nullptr;  // mkl_malloc(partSize * sizeof(double), 64)
        // [DIAG] メモリ追跡: この Layer が確保した total bytes (freeAll で減算に使用)
        size_t layerAllocated = 0;
```

### A-2. 静的カウンタ宣言

```cpp
class MKLNonUniformConvolver {
public:
    // [DIAG] メモリ追跡用: 全インスタンスの現在使用量・ピーク・個数
    static std::atomic<uint64_t> globalAllocated;
    static std::atomic<uint64_t> peakAllocated;
    static std::atomic<int> liveCount;

    // [DIAG] このインスタンスの現在使用量
    uint64_t thisAllocated = 0;
```

### A-3. 全 `mkl_malloc` 呼び出しでの DIAG_ALLOC 挿入

**ファイル**: `src/MKLNonUniformConvolver.cpp` — 全28箇所の `mkl_malloc` 直後。

```cpp
// ★ ヘルパーマクロ: ptr が非 null の場合のみカウント
#define DIAG_ALLOC(ptr, bytes) do { \
    if (ptr) { \
        const auto _d = static_cast<uint64_t>(bytes); \
        l.layerAllocated += _d; \
        thisAllocated += _d; \
        globalAllocated += _d; \
        uint64_t cur = globalAllocated.load(); \
        uint64_t peak = peakAllocated.load(); \
        while (cur > peak && !peakAllocated.compare_exchange_weak(peak, cur)) {} \
    } \
} while(0)

// 使用例（全28箇所に同パターン）:
l.irFreqDomain = static_cast<double*>(mkl_malloc(irBufSize * sizeof(double), 64));
DIAG_ALLOC(l.irFreqDomain, irBufSize * sizeof(double));

l.irFreqReal = static_cast<double*>(mkl_malloc(irSoaSize * sizeof(double), 64));
DIAG_ALLOC(l.irFreqReal, irSoaSize * sizeof(double));

// ... 以下全メンバの mkl_malloc 直後に同様のパターン ...
```

### A-4. `freeAll()` での減算

```cpp
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // [DIAG] 解放前に Layer 使用量をグローバルから減算
    if (layerAllocated > 0) {
        globalAllocated -= layerAllocated;
        // thisAllocated の減算は Layer が所属するインスタンスが別管理のため、ここでは減算しない
        // （デストラクタで一括減算）
    }
    // ... 既存の mkl_free 群 ...
    fftSize = partSize = numParts = numPartsIR = 0;
    fdlMask = complexSize = partStride = 0;
    // ...
    layerAllocated = 0;  // [DIAG]
}
```

`thisAllocated` の減算はデストラクタで行う:

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    liveCount--;
    releaseAllLayers();
    // [DIAG] このインスタンスの使用量をグローバルからも減算
    // （Layer の freeAll() で既に Layer 分は減算済み。重複に注意）
}
```

> **設計判断**: `layerAllocated` は freeAll() で `globalAllocated` から減算する。`thisAllocated` は追跡したいが、マルチインスタンスでの管理が複雑になるため、今回の調査では `globalAllocated` + `liveCount` から平均を推定する。インスタンス別の正確な値が必要なら別途対応。

### A-5. NUC 合計ログ（Layer 別 + 合計）

**ファイル**: `src/MKLNonUniformConvolver.cpp` — SetImpulse() 成功直後、`m_ready = true` の前。

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    diagLog(juce::String::formatted(
        "[NUC_MEM] NUC#%p L0:%zuMB L1:%zuMB L2:%zuMB total=%.0fMB | global=%lluMB peak=%lluMB live=%d",
        (void*)this,
        m_numActiveLayers >= 1 ? m_layers[0].layerAllocated / (1024*1024) : 0,
        m_numActiveLayers >= 2 ? m_layers[1].layerAllocated / (1024*1024) : 0,
        m_numActiveLayers >= 3 ? m_layers[2].layerAllocated / (1024*1024) : 0,
        thisAllocated / (1024.0 * 1024.0),
        (unsigned long long)globalAllocated.load() / (1024*1024),
        (unsigned long long)peakAllocated.load() / (1024*1024),
        (int)liveCount.load()));
}
#endif
```

### A-6. static 変数の実体

```cpp
std::atomic<uint64_t> MKLNonUniformConvolver::globalAllocated { 0 };
std::atomic<uint64_t> MKLNonUniformConvolver::peakAllocated { 0 };
std::atomic<int> MKLNonUniformConvolver::liveCount { 0 };
```

---

## 2. Patch B: StereoConvolver liveCount

v2 と同じ。`ctor/`dtor` で `liveCount++/--`。

```cpp
StereoConvolver() { liveCount++; }
~StereoConvolver() { liveCount--; ... }
```

---

## 3. Patch C: DSPCore liveCount（簡略版）+ DSPCore::prepare 確保量ログ

### C-1. DSPCore liveCount

```cpp
// AudioEngine.h — struct DSPCore
static std::atomic<int> liveCount;
DSPCore() { liveCount++; }
~DSPCore() { liveCount--; convolver.forceCleanup(); }
```

### C-2. DSPCore::prepare 確保量ログ

```cpp
// DSPCoreLifecycle.cpp — alignedL/R 確保後
diagLog(juce::String::formatted(
    "[MEM_BUF] alignedL/R: %d doubles = %.1f MB",
    newRequired, (newRequired * sizeof(double) * 2) / (1024.0 * 1024.0)));
```

---

## 4. Patch D: イベント駆動ログ（Publish スナップショット + 5秒簡略）

### D-1. Publish 時フルスナップショット

**ファイル**: `src/audioengine/AudioEngine.Timer.cpp` — `publishWorld()` 成功直後。

```cpp
// [DIAG] Publish 時メモリスナップショット
const uint64_t nucAlloc = convo::MKLNonUniformConvolver::globalAllocated.load();
const uint64_t nucPeak  = convo::MKLNonUniformConvolver::peakAllocated.load();
const int nucLive  = (int)convo::MKLNonUniformConvolver::liveCount.load();
const int stereoLive = (int)ConvolverProcessor::StereoConvolver::liveCount.load();
const int dspLive    = (int)AudioEngine::DSPCore::liveCount.load();
const uint32_t pending = m_retireRouter ? m_retireRouter->pendingRetireCount() : 0;
const uint64_t reclaim = m_retireRouter ? m_retireRouter->reclaimAttemptCount() : 0;

juce::Logger::writeToLog(juce::String::formatted(
    "[MEM_SNAP] PUBLISH gen=%d | "
    "NUC: live=%d alloc=%.0fMB peak=%.0fMB | "
    "Stereo=%d DSPCore=%d | "
    "Retire: pending=%u reclaim=%llu",
    gen,
    nucLive, nucAlloc / (1024.0 * 1024.0), nucPeak / (1024.0 * 1024.0),
    stereoLive, dspLive,
    pending, (unsigned long long)reclaim));
```

### D-2. 5秒タイマー簡略ログ（変化検出時のみ）

```cpp
static juce::uint32 lastMemLog = 0;
const juce::uint32 nowMs = juce::Time::getApproximateMillisecondTimer();
if (nowMs - lastMemLog > 5000) {
    lastMemLog = nowMs;
    static uint64_t lastAlloc = 0;
    const uint64_t curAlloc = convo::MKLNonUniformConvolver::globalAllocated.load();
    if (curAlloc == lastAlloc) return;  // 変化なし → スキップ
    lastAlloc = curAlloc;
    juce::Logger::writeToLog(juce::String::formatted(
        "[MEM] NUC=%d alloc=%.0fMB peak=%.0fMB | Retire pending=%u",
        (int)convo::MKLNonUniformConvolver::liveCount.load(),
        curAlloc / (1024.0 * 1024.0),
        convo::MKLNonUniformConvolver::peakAllocated.load() / (1024.0 * 1024.0),
        m_retireRouter ? m_retireRouter->pendingRetireCount() : 0));
}
```

---

## 5. Patch E: RetireQueue / EpochDomain 指標の確認

**ファイル**: `src/audioengine/ISRRetireRouter.h`

`ISRRetireRouter` には以下のメソッドが既に存在することを確認済み:

| メソッド | 戻り値 | 意味 |
| :--- | :--- | :--- |
| `pendingRetireCount()` | `uint32_t` | 退役キュー滞留数 |
| `reclaimAttemptCount()` | `uint64_t` | 累積 reclaim 試行回数 |
| `reclaimSuccessCount()` | `uint64_t` | 累積 reclaim 成功回数 |
| `activeReaderCount()` | `uint32_t` | アクティブリーダー数 (AudioEngine 側) |

これらは既に DIAG ログに含めるだけで計測可能であり、**コード変更は不要**（Patch D のログ書式に組み込むのみ）。

---

## 6. 出力例

### 期待されるログ出力（正常時）

```text
[NUC_MEM] NUC#000002 L0:30MB L1:0MB L2:0MB total=30MB | global=62MB peak=142MB live=2
...
[MEM_SNAP] PUBLISH gen=8 seq=5 |
  NUC: live=2 alloc=62MB peak=142MB |
  Stereo=1 DSPCore=1 |
  Retire: pending=0 reclaim=47
...
[MEM] NUC=2 alloc=62MB peak=142MB | Retire pending=0
```

### 異常検出例

```text
[MEM_SNAP] PUBLISH gen=8 |
  NUC: live=8 alloc=496MB peak=620MB |    ← NUC が 8 個も存在 → リーク！
  Stereo=4 DSPCore=4 |
  Retire: pending=232 reclaim=12          ← Retire キュー滞留！
```

---

## 7. 診断フロー

| globalAllocated | liveCount | pendingRetire | 診断 |
| :--- | :--- | :--- | :--- |
| ~60MB | 2 | 0 | **正常。2.33GB は NUC 外部** → ProcessingBuffer/EQ 調査 |
| ~600MB | 8 | 0 | **NUC リーク** → StereoConvolver の退休不備 |
| ~60MB | 2 | ≥100 | **RetireQueue 滞留** → ISR EpochDomain の reclaim 不備 |
| ~60MB | 2 | 0, かつ 2.33GB | **DSPCore 内部バッファが主因** → EQ/ProcessingBuffer 調査 |
| peak≫current | — | — | **一瞬のピークが原因** → ProgressiveUpgrade の中間生成調査 |

---

## 8. 実装コスト見積もり

| Patch | 変更ファイル数 | 追加行数 | 備考 |
| :--- | :--- | :--- | :--- |
| A: MKLNonUniformConvolver 3階層カウンタ | 2 | ~50行 | DIAG_ALLOC × 28箇所 + freeAll 減算 |
| B: StereoConvolver liveCount | 1 | ~5行 | v2 と同じ |
| C: DSPCore liveCount + 確保量ログ | 2 | ~12行 | |
| D: イベント駆動ログ | 1 | ~25行 | RetireQueue 指標含む |
| **合計** | **4〜5ファイル** | **〜92行** | |

---

## 9. v2 からの改善点一覧

| 項目 | v2（問題） | v3（修正） |
| :--- | :--- | :--- |
| 解放時の減算 | なし（累積値のみ） | **Layer::layerAllocated → freeAll() で globalAllocated.fetch_sub** |
| nullptr ガード | なし | **DIAG_ALLOC マクロ内で `if (ptr)` チェック** |
| カウンタ階層 | 1層（global のみ） | **3層（global → instance → layer）** |
| Layer 別集計 | ログのみ計算式 | **Layer::layerAllocated で直接計測** |
| RetireQueue | なし | **pendingRetireCount / reclaimAttemptCount 追加** |
| DSPCore ctor ログ | 毎回 Logger 出力 | **Publish 時スナップショットに集約** |
| ProcessingBuffer | ログのみ | **alignedL/R 確保量ログ維持（優先度下げ）** |
