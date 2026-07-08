# メモリ占有調査のためのインストルメンテーション改修案 v27（最終版）— 全レビュー反映

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v26 からの変更**: レビュー指摘を反映（v26 で既に解決済みの事項は明確化）。

---

## 0. v26 に対するレビュー指摘と v27 での対応

| # | 指摘 | v26 の状態 | v27 の対応 |
|:--|:-----|:----------|:----------|
| 1 | IR_LAYOUT が Layer を二重走査している | **v24 で既に修正済み**。`NucDiagnosticsSnapshot` に `irFreqBytes/fdlBytes/accumBytes/tailBytes` があり、`getDiagnostics()` 一回で全情報を取得 | 変更不要。設計書に明確に記載 |
| 2 | IR_LAYOUT に gen がないと MEM_SNAP と対応付けにくい | gen 未追加 | **`seq` カウンタを NUC に追加（SetImpulse のたびに increment）**。API 変更不要 |
| 3 | IR_LOAD/IR_RELEASE の OS Private 取得は MEM_SNAP で十分 | OS Private を Before/After で取得 | **IR_RELEASE は維持（解放前後の OS Private 変化が診断上最も価値が高い）。IR_LOAD の OS Private は削除し簡素化** |
| 4 | liveCount は Crossfade で倍になることを文書化 | 未記載 | **v27 設計書に注記追加** |

---

## 1. 確認: IR_LAYOUT の二重走査は v26 で既に解消済み

v26 の `NucDiagnosticsSnapshot` は以下のフィールドを持ちます：

```cpp
struct NucDiagnosticsSnapshot {
    uint64_t layerBufs[3] = { 0, 0, 0 };
    uint64_t irFreqBytes  = 0;  // irFreqDomain + irFreqReal + irFreqImag
    uint64_t fdlBytes     = 0;  // fdlBuf + fdlReal + fdlImag
    uint64_t accumBytes   = 0;  // fftTimeBuf + fftOutBuf + prevInputBuf + accumBuf + accumReal + accumImag + inputAccBuf
    uint64_t tailBytes    = 0;  // tailOutputBuf
    uint64_t directBytes  = 0;
    uint64_t ringBytes    = 0;
    int      numActiveLayers = 0;
    bool     isReady         = false;
    [[nodiscard]] uint64_t totalBytes() const noexcept { ... }
};
```

`getDiagnostics()` 内の**一回の Layer 走査**で全フィールドが計算される。
`SetImpulse()` 内で `getDiagnostics()` を 1 回呼び出し、その結果を `IR_LAYOUT` と `NUC_MEM`（layerBufs）の両方に使用する。
**二重走査は発生しない。**

---

## 2. Patch B: NUC seq カウンタ追加（IR_LAYOUT 対応用）

`SetImpulse()` の API を変更せずに、NUC 内部の単調増加カウンタで IR イベントを識別する。

### B-1. MKLNonUniformConvolver.h

```cpp
class MKLNonUniformConvolver {
    // ... 既存メンバ ...
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    uint32_t m_diagSeq_ { 0 };  // ★ v27: 診断用単調増加カウンタ
#endif
};
```

### B-2. SetImpulse() 内

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint32_t diagSeq = ++m_diagSeq_;  // ★ v27
#endif
```

### B-3. IR_LOAD / IR_LAYOUT に seq 追加

```text
[IR_LOAD]   ... seq=%d ...
[IR_LAYOUT] ... seq=%d ...
```

---

## 3. IR_LOAD の OS Private 削除（簡素化）

| ログ | OS Private Before/After | 根拠 |
|:----|:----------------------|:-----|
| `[IR_RELEASE]` | ✅ **維持**（解放前後の OS Private 変化が最も価値が高い。「MKL が解放されたのに OS が減らない」の検出に必須） |
| `[IR_LOAD]` | ❌ **削除**（MEM_SNAP で十分。IR_RELEASE + MEM_SNAP で OS 全体の増減は追跡可能） |

### IR_LOAD の v27 フォーマット

```
[IR_LOAD] NUC#%p seq=%d irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

---

## 4. liveCount Crossfade 注記

```cpp
/// ★ v27: NUC インスタンスの生存数。
///   Crossfade 期間中は新旧両方の Convolver が一時的に共存するため、
///   liveCount は通常の約 2 倍になる。これは正常な動作である。
///   例: 通常時 live=4 → Crossfade 時 live=8 → 完了後 live=4
```

---

## 5. 全ログフォーマット（v27 確定版）

### IR_RELEASE
```
[IR_RELEASE] NUC#%p seq=%d MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD
```
[IR_LOAD] NUC#%p seq=%d irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

### IR_LAYOUT
```
[IR_LAYOUT] NUC#%p seq=%d IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP
```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u | Stereo=%d DSPCore=%d | Retire: pending=%u trackedBytes=%.1fMB(diag only) tracked=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB
```

---

## 6. 出力例（v27 最終版）

```text
[IR_RELEASE] NUC#001 seq=5 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=6 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=6 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | Stereo=4 DSPCore=4 | Retire: pending=232 trackedBytes=12.8MB(diag only) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB
```

**seq=5 → seq=6** で IR_RELEASE と IR_LOAD/IR_LAYOUT の対応が明確。

---

## 7. v26 からの改善点一覧

| # | 指摘 | v26 | v27 |
|:--|:-----|:----|:----|
| 1 | IR_LAYOUT 二重走査 | 既に修正済み（設計書に未記載） | **明確に記載** |
| 2 | IR_LAYOUT に識別子がない | gen なし | **`seq` カウンタ追加（API 変更不要）** |
| 3 | IR_LOAD の OS Private が重複 | OS Private あり | **IR_LOAD から OS Private 削除（簡素化）** |
| 4 | liveCount Crossfade 時の挙動 | 未記載 | **注記追加** |
