# Snapshot Authority Inventory

作成日: 2026-06-06
ベース: 現行ソース (RuntimeBuildTypes.h / RuntimeBuilder.cpp / CrossfadeAuthority.cpp)

---

## 1. 現状の RuntimeBuildSnapshot

ソース: `src/audioengine/RuntimeBuildTypes.h:34-41`

```cpp
struct RuntimeBuildSnapshot
{
    int generation = 0;
    BuildInput buildInput {};
    std::uint64_t convolverFingerprint = 0;
    RuntimeBuildFingerprint rebuildFingerprint {};
    bool sealed = false;
};
```

## 2. 追加が必要な投影フィールド (候補)

### 2.1 CrossfadeAuthority が参照するフィールド (PR-2 で必須)

| フィールド | 現在の供給元 (DSPCore直読) | 追加理由 |
|---|---|---|
| `irLoaded` | `current->convolverRt().isIRLoaded()` | CrossfadeAuthority が computeDecision/evaluateFromWorlds で参照 |
| `structuralHash` | `current->convolverRt().getStructuralHash()` | CrossfadeAuthority が computeDecision/evaluateFromWorlds で参照 |
| `oversamplingFactor` | `static_cast<int>(current->oversamplingFactor)` | CrossfadeAuthority が computeDecision/evaluateFromWorlds で参照 |

### 2.2 dspProjection に存在するが CrossfadeAuthority は未参照のフィールド

| フィールド | 現在の供給元 | 追加の要否 |
|---|---|---|
| `irFinalized` | `current->convolverRt().isIRFinalized()` | dspProjectionに存在。追加は候補。ただしCrossfadeAuthority未参照のためPR-2での追加は確定的ではない |
| `sampleRate` | `current->sampleRate` | dspProjectionに存在。他用途(LatencyService)用の可能性あり |
| `baseLatencySamples` | `engine.estimateRuntimeLatencyBaseRateSamples()` | dspProjectionに存在。CrossfadeAuthority未参照。PR-2での追加は確定的ではない |

## 3. 推奨追加方針

### PR-2 で実施 (必須)

CrossfadeAuthority が参照する3フィールドを `RuntimeBuildSnapshot` に追加する:

- `bool irLoaded = false;`
- `std::uint64_t structuralHash = 0;`
- `int oversamplingFactor = 1;`

### PR-2 で実施 (推奨)

dspProjection 全体を Snapshot 化するため、残り3フィールドも追加する:

- `bool irFinalized = false;`
- `double sampleRate = 48000.0;`
- `double baseLatencySamples = 0.0;`

**注意**: `baseLatencySamples` は CrossfadeAuthority が未参照であるため、PR-0 の棚卸しで必要性を確認してから判断する。

## 4. Builder Projection Coverage (0-15 用)

PR-2 完了時、以下のマッピングが 100% 存在することを確認する:

| dspProjection フィールド | Snapshot 供給元 | Build 使用箇所 |
|---|---|---|
| `irLoaded` | `snapshot.irLoaded` | `RuntimeBuilder.cpp` dspProjection 構築ブロック |
| `irFinalized` | `snapshot.irFinalized` | 同上 |
| `structuralHash` | `snapshot.structuralHash` | 同上 |
| `oversamplingFactor` | `snapshot.oversamplingFactor` | 同上 |
| `sampleRate` | `snapshot.sampleRate` | 同上 |
| `baseLatencySamples` | `snapshot.baseLatencySamples` | 同上 |
