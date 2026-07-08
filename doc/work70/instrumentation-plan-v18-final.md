# メモリ占有調査のためのインストルメンテーション改修案 v18（最終版）— IR_RELEASE + L1Part/L2Part

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v17 からの変更**: 3 点の修正。

---

## 0. v17 の問題点と v18 での修正方針

| # | v17 の問題 | v18 の修正 |
|:--|:----------|:----------|
| 1 | `m_isBuilding` は契約で十分（MessageThread 単一なら競合しない） | **`m_isBuilding` 削除、契約ベースに回帰** |
| 2 | `liveCount` が compare_exchange → Debug で原因特定困難 | **`fetch_sub` + `jassert(old>0)` に回帰（v15 方式）** |
| 3 | `IR_LOAD` に L0Part のみ → 全 partition 構成を記録できない | **`L1Part` / `L2Part` 追加** |
| 4 | `releaseAllLayers()` 前後のメモリ変化が不明 | **`[IR_RELEASE] before/after/delta` 追加** |

---

## 1. Patch A: DiagnosticsConfig.h（v18 確定版、変更なし）

v17 の内容をそのまま維持。

---

## 2. Patch B: MKLNonUniformConvolver — m_isBuilding 削除 + fetch_sub 回帰 + IR_RELEASE

### B-1. getDiagnostics() — m_isBuilding 削除、契約ベースに（★ v18）

```cpp
/// ★ v18: 診断用スナップショット。
///
/// 呼び出し契約:
///   1. Message Thread からのみ呼び出すこと。
///   2. SetImpulse() / releaseAllLayers() の実行中には呼ばないこと。
///      推奨呼び出し箇所: Publish 直後、SetImpulse 完了後、タイマー（非ビルド時）。
///
/// データ競合安全性:
///   Layer のメンバは Message Thread からのみ書き込まれる。
///   本メソッドも Message Thread からのみ読むため、C++ memory model 上安全。
///   ただし SetImpulse 実行途中で呼ぶと中間状態を読む可能性がある。
[[nodiscard]] NucDiagnosticsSnapshot getDiagnostics() const noexcept
{
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    NucDiagnosticsSnapshot snap{};
    snap.numActiveLayers = m_numActiveLayers;
    snap.isReady = convo::consumeAtomic(m_ready, std::memory_order_acquire);

    // ... 既存の集計ロジック ...
}
```

### B-2. デストラクタ — fetch_sub + jassert 回帰（★ v18）

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    // ★ v18: fetch_sub + jassert。Debug でアンダーフロー原因を特定しやすい。
    //   Release では UINT_MAX になるが、診断ビルドのみで使用するため許容。
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    jassert(oldLive > 0);
    releaseAllLayers();
}
```

### B-3. releaseAllLayers() — IR_RELEASE ログ追加（★ v18 最重要）

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ v18: 解放前のメモリ使用量
    const uint64_t beforeBytes = convo::diag::allocatedBytes();
#endif

    // ... 既存の guard チェック ...

    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;

    // ... NUC レベルバッファ解放 ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ v18: 解放後のメモリ使用量
    const uint64_t afterBytes = convo::diag::allocatedBytes();
    const int64_t delta = static_cast<int64_t>(afterBytes) - static_cast<int64_t>(beforeBytes);
    diagLog(juce::String::formatted(
        "[IR_RELEASE] NUC#%p before=%lluMB after=%lluMB delta=%lldMB",
        (void*)this,
        (unsigned long long)(beforeBytes / (1024*1024)),
        (unsigned long long)(afterBytes / (1024*1024)),
        (long long)(delta / (1024*1024))));
#endif
}
```

### B-4. SetImpulse — L1Part/L2Part 追加（★ v18）

```cpp
    // IR_LOAD: 構成情報（★ v18: L1Part/L2Part 追加）
    const int l0Part = m_numActiveLayers >= 1 ? m_layers[0].partSize : 0;
    const int l1Part = m_numActiveLayers >= 2 ? m_layers[1].partSize : 0;
    const int l2Part = m_numActiveLayers >= 3 ? m_layers[2].partSize : 0;
    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p irLen=%d blockSize=%d "
        "L0Part=%d L1Part=%d L2Part=%d "
        "directTaps=%d ringSize=%d "
        "before=%lluMB after=%lluMB delta=%lldMB live=%u",
        (void*)this, irLen, blockSize,
        l0Part, l1Part, l2Part,
        m_directTapCount, m_ringSize,
        (unsigned long long)(beforeBytes / (1024*1024)),
        (unsigned long long)(afterBytes / (1024*1024)),
        (long long)(delta / (1024*1024)),
        (unsigned)liveCount.load(std::memory_order_relaxed)));
```

---

## 3. 全ログの v18 統一フォーマット

### IR_RELEASE（★ v18 新規）

```text
[IR_RELEASE] NUC#%p before=%lluMB after=%lluMB delta=%lldMB
```

### IR_LOAD

```text
[IR_LOAD] NUC#%p irLen=%d blockSize=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d before=%lluMB after=%lluMB delta=%lldMB live=%u
```

### IR_LAYOUT

```text
[IR_LAYOUT] NUC#%p IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB
```

### NUC_MEM / MEM_SNAP（v17 から変更なし）

---

## 4. 出力例（v18）— リークあり

```text
[IR_RELEASE] NUC#001 before=820MB after=110MB delta=-710MB
[IR_LOAD]    NUC#001 irLen=327680 blockSize=4096 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 before=110MB after=812MB delta=+702MB live=8
[IR_LAYOUT]  NUC#001 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB
[MEM_SNAP]   PUBLISH gen=8 | NUC: live=8 alloc=832MB ... | OtherPrivate=1498MB
```

### 出力例（v18）— 正常

```text
[IR_RELEASE] NUC#001 before=820MB after=110MB delta=-710MB
[IR_LOAD]    NUC#001 irLen=327680 blockSize=4096 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 before=110MB after=812MB delta=+702MB live=8
[IR_LAYOUT]  NUC#001 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB
[MEM_SNAP]   PUBLISH gen=8 seq=5 | NUC: live=8 alloc=812MB Peak(reset)=1200MB totalA=28.0GB totalF=27.42GB lostFree=0 | OS: Private=890MB WorkingSet=950MB | OtherPrivate=78MB
[IR_RELEASE] NUC#002 before=812MB after=110MB delta=-702MB
[IR_LOAD]    NUC#002 irLen=65536 blockSize=4096 L0Part=4096 L1Part=0 L2Part=0 directTaps=32 ringSize=4096 before=110MB after=120MB delta=+10MB live=8
[IR_LAYOUT]  NUC#001 IRFreq=40MB FDL=60MB Accum=12MB Tail=0MB Direct=24MB Ring=4MB
```

**IR_RELEASE で解放漏れを検出**:

```text
[IR_RELEASE] NUC#005 before=812MB after=812MB delta=+0MB   ← 解放されていない！
```

---

## 5. 診断フロー（v18）

時系列で 4 つのログを追跡するだけでメモリ増減の因果関係が分かる：

```text
IR_RELEASE  →  解放量を確認（解放漏れ検出）
IR_LOAD     →  確保量を確認（delta で増分把握）
IR_LAYOUT   →  どのバッファが支配的か特定
MEM_SNAP    →  OS Private Usage と突き合わせ
```

---

## 6. v17 からの改善点一覧

| # | v17（問題） | v18（修正） |
|:--|:----------|:-----------|
| 1 | `m_isBuilding` が契約ベースで十分なものを atomic フラグに | **削除、契約ベースに回帰** |
| 2 | `liveCount` compare_exchange で Debug 時に原因特定困難 | **`fetch_sub` + `jassert(old>0)` に回帰** |
| 3 | `IR_LOAD` に L0Part のみ（partition 構成不完全） | **`L1Part` / `L2Part` 追加** |
| 4 | `releaseAllLayers()` 前後のメモリ変化が不明 | **`[IR_RELEASE] before/after/delta` 追加（最重要改善）** |
