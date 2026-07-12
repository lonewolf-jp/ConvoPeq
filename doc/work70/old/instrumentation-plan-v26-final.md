# メモリ占有調査のためのインストルメンテーション改修案 v26（最終版）— API 非変更版

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v25 からの変更**: 2 点の修正。

---

## 0. v25 の問題点と v26 での修正方針

| # | v25 の問題 | v26 の修正 |
|:--|:----------|:----------|
| 1 | `enqueueRetire()` に `objectBytes` パラメータ追加 → API 変更 | **API 変更なし。`#if` で `DeletionEntry` に `objectBytes` を追加するのみ。呼び出し元の既存コードは変更不要** |
| 2 | `objBytes` / `pendingRetireBytes` が「追跡対象のみ」と誤解されうる | **`trackedBytes` に名称変更 + コメントで「診断用概算、実ヒープ使用量ではない」と明記** |

---

## 1. Patch C: 設計の核心 — API を変えずに objectBytes を追跡

### 方針

- `DeletionEntry` に `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で `objectBytes` を追加
- `enqueueRetire()` の**シグネチャは変更しない**（新しいオーバーロードも追加しない）
- 呼び出し元（`DSPLifetimeManager`、`AudioEngine` 等）のコードは**一切変更不要**
- `ISRRetireRouter` は、`DeletionEntry` がキューから取り出された時点で `objectBytes` を読み取り、`m_pendingRetireBytes_` を更新する

ただし、最初の実装では `objectBytes` は常に 0（呼び出し元が設定していないため）。
`trackedRatio` と併せて「全エントリがサイズ不明」であることを正直に報告する。

**将来、特定の呼び出し元でサイズを記録したい場合のみ、その呼び出し元のコードを変更する。**

### 1-1. DeletionEntry

```cpp
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
    uint64_t publicationSequenceId{0};
    uint64_t generation{0};
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    size_t objectBytes{0};  // 呼び出し元が設定したオブジェクトサイズ（0=不明）
#endif
};
```

### 1-2. ISRRetireRouter

```cpp
class ISRRetireRouter : public convo::IEpochProvider {
    std::atomic<uint64_t> m_pendingRetireBytes_ { 0 };  // ★ v26: trackedBytes

    // ★ v26: API 変更なし。既存の enqueueRetire シグネチャのまま。
    //   DeletionEntry の objectBytes は enqueue 時に 0。
    //   将来、呼び出し元でサイズを設定した場合のみ値が入る。

public:
    /// Diagnostic estimate only. Returns the sum of objectBytes from
    /// tracked retire entries. Does NOT represent the actual heap usage
    /// of the retire queue — entries with size 0 are not counted.
    /// See also trackedRatio() for coverage.
    [[nodiscard]] virtual uint64_t pendingRetireBytes() const noexcept override
    {
        return convo::consumeAtomic(m_pendingRetireBytes_, std::memory_order_acquire);
    }
};
```

### 1-3. MEM_SNAP の名称変更

```
// v25: objBytes=12.8MB(sizeof)
// v26: trackedBytes=12.8MB(diag only)

// tracked=8/232 (3%) → 232 エントリ中 8 エントリのみサイズ追跡中
```

---

## 2. 全ログフォーマット（v26 確定版）

### IR_RELEASE
```
[IR_RELEASE] NUC#%p MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD
```
[IR_LOAD] NUC#%p irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u
```

### IR_LAYOUT
```
[IR_LAYOUT] NUC#%p IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP
```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u | Stereo=%d DSPCore=%d | Retire: pending=%u trackedBytes=%.1fMB(diag only) tracked=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB
```

---

## 3. 出力例（v26 最終版）

```text
[IR_RELEASE] NUC#001 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) | OS: beforePrivate=1620MB afterPrivate=2330MB live=8
[IR_LAYOUT]  NUC#001 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | Stereo=4 DSPCore=4 | Retire: pending=232 trackedBytes=12.8MB(diag only) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB
```

---

## 4. v25 からの改善点一覧

| # | v25（問題） | v26（修正） |
|:--|:----------|:-----------|
| 1 | `enqueueRetire()` に `objectBytes` パラメータ追加 → API 変更 | **API 変更なし。`DeletionEntry.objectBytes` を `#if` で追加するのみ** |
| 2 | `objBytes` / `pendingRetireBytes` が実ヒープ使用量と誤解されうる | **`trackedBytes(diag only)` に名称変更 + コメントで「診断用概算」と明記** |
