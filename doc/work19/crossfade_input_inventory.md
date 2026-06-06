# Crossfade Decision Input Inventory

作成日: 2026-06-06
ベース: 現行ソース (CrossfadeAuthority.cpp / RuntimeBuilder.cpp)

---

## 1. computeDecision() が参照する DSPCore フィールド

ソース: `src/audioengine/CrossfadeAuthority.cpp:32-93`

| DSPCore フィールド | アクセス方法 | dspProjection 対応 | 備考 |
|---|---|---|---|
| `oldDSP->convolverRt().isIRLoaded()` | `convolverRt()` 経由のメソッド呼び出し | `dspProjection.irLoaded` ✅ | `evaluateFromWorlds()` で代替検証済み |
| `newDSP->convolverRt().isIRLoaded()` | 同上 | `dspProjection.irLoaded` ✅ | |
| `oldDSP->convolverRt().getStructuralHash()` | `convolverRt()` 経由のメソッド呼び出し | `dspProjection.structuralHash` ✅ | |
| `newDSP->convolverRt().getStructuralHash()` | 同上 | `dspProjection.structuralHash` ✅ | |
| `oldDSP->oversamplingFactor` | 直接フィールドアクセス | `dspProjection.oversamplingFactor` ✅ | |
| `newDSP->oversamplingFactor` | 直接フィールドアクセス | `dspProjection.oversamplingFactor` ✅ | |

## 2. computeDecision() が参照する Engine Atomic フィールド (DSPCore 非依存)

| Engine Atomic フィールド | 用途 | 備考 |
|---|---|---|
| `engine.m_osFadeTimeSec` | Oversampling変更時のフェード時間 | Atomic, 非ブロッキング |
| `engine.m_irFadeTimeSec` | IR構造変更時の基本フェード時間 | Atomic, 非ブロッキング |
| `engine.m_irLengthFadeTimeSec` | IR長変更時のフェード時間 | Atomic, 非ブロッキング |
| `engine.m_phaseFadeTimeSec` | 位相変更時のフェード時間 | Atomic, 非ブロッキング |
| `engine.m_directHeadFadeTimeSec` | DirectHead変更時のフェード時間 | Atomic, 非ブロッキング |
| `engine.m_nucFilterFadeTimeSec` | NucFilter変更時のフェード時間 | Atomic, 非ブロッキング |
| `engine.m_tailFadeTimeSec` | Tail変更時のフェード時間 | Atomic, 非ブロッキング |

**重要**: これらは全て `convo::consumeAtomic()` 経由で読み取られる atomic 値であり、DSPCore に依存しない。PR-1 移行後も変更不要。

## 3. dspProjection の供給フィールド一覧

ソース: `src/audioengine/RuntimeBuilder.cpp:190-198`

| dspProjection フィールド | 現在の供給元 (DSPCore直読) | 将来の供給元 (Snapshot) | CrossfadeAuthority が参照 |
|---|---|---|---|
| `irLoaded` | `current->convolverRt().isIRLoaded()` | Snapshot `irLoaded` | ✅ 参照 |
| `irFinalized` | `current->convolverRt().isIRFinalized()` | Snapshot `irFinalized` | ❌ 未参照 |
| `structuralHash` | `current->convolverRt().getStructuralHash()` | Snapshot `structuralHash` | ✅ 参照 |
| `oversamplingFactor` | `static_cast<int>(current->oversamplingFactor)` | Snapshot `oversamplingFactor` | ✅ 参照 |
| `sampleRate` | `current->sampleRate` | Snapshot `sampleRate` | ❌ 未参照 |
| `baseLatencySamples` | `engine.estimateRuntimeLatencyBaseRateSamples(current, false)` | Snapshot `baseLatencySamples` | ❌ 未参照 |

## 4. 結論

- CrossfadeAuthority が Decision 入力として実際に参照している DSPCore フィールドは **3項目のみ** (`irLoaded` / `structuralHash` / `oversamplingFactor`)
- これら3項目は全て `dspProjection` に存在し、`evaluateFromWorlds()` で代替済み
- Engine Atomic 7項目は DSPCore 非依存で移行対象外
- 残り3項目 (`irFinalized` / `sampleRate` / `baseLatencySamples`) は dspProjection に存在するが CrossfadeAuthority は未参照（他用途用）
- **PR-1 の API を `evaluate(const DSPProjection&, const DSPProjection&)` とする判断を支持**
