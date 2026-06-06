# CrossfadeAuthority Dual-Path Audit (0-14)

作成日: 2026-06-06
対象: CrossfadeAuthority evaluate() / evaluateFromWorlds() / computeDecision() 比較

---

## 1. 経路比較

| 評価項目 | 旧 computeDecision (削除済み) | 新 evaluate (現在の実装) | 等価性 |
|---|---|---|---|
| IR Loaded | `oldDSP->convolverRt().isIRLoaded()` | `oldWorld.dspProjection.irLoaded` | ✅ PR-2 で Snapshot 経由に変更 |
| IR Loaded (new) | `newDSP->convolverRt().isIRLoaded()` | `newWorld.dspProjection.irLoaded` | ✅ 同上 |
| Oversampling | `newDSP->oversamplingFactor != oldDSP->oversamplingFactor` | `newWorld.dspProjection.oversamplingFactor != oldWorld.dspProjection.oversamplingFactor` | ✅ 同上 |
| Structural Hash | `oldDSP->convolverRt().getStructuralHash()` | `oldWorld.dspProjection.structuralHash` | ✅ 同上 |
| Structural Hash (new) | `newDSP->convolverRt().getStructuralHash()` | `newWorld.dspProjection.structuralHash` | ✅ 同上 |
| Engine Atomic (fade times) | `convo::consumeAtomic(engine.m_*FadeTimeSec)` | `convo::consumeAtomic(engine.m_*FadeTimeSec)` | ✅ 同一 (DSPCore非依存) |

## 2. 判定ロジックの同一性

`evaluate()` のロジックフローは削除前の `evaluateFromWorlds()` と完全に同一（関数改名のみ）。
`evaluateFromWorlds()` のロジックは `computeDecision()` と以下の点で同一:

- IR presence check → hasAudibleConvolverTransition / irPresenceChanged
- Oversampling change → fadeTimeSec = max(..., m_osFadeTimeSec)
- IR structural change → 3パターン分岐 (presenceChanged / structuralChanged + presenceSame)

**結論: 二経路間の判定ロジックに差異は存在しない。投影値の供給元が DSPCore 直読から Snapshot 経由に変わったのみ。**

## 3. 4ケース比較 (理論検証)

PR-2 により dspProjection の値供給元が Snapshot になった後、以下の4ケースで動作が等価であることを確認する:

| # | ケース | oldWorld.dspProjection | newWorld.dspProjection | evaluate結果 |
|---|---|---|---|---|
| a | IR未ロード→ロード | irLoaded=false | irLoaded=true | needsCrossfade: irPresenceChanged により baseIrFade clamp |
| b | IR構造変更 (hash変化) | hash=A | hash=B (≠A) | needsCrossfade: true, fadeTimeSec: 全atomicのmax |
| c | Oversampling変更 | osFactor=1 | osFactor=2 | needsCrossfade: true, fadeTimeSec: m_osFadeTimeSec |
| d | 両方 null | DSPCore=nullptr | DSPCore=nullptr | needsCrossfade: false (early return) |

**全てのケースで、旧 computeDecision() と同じ結果が得られる。**

## 4. 評価: ✅ 合格

Dual-Path Audit: **通過**。`evaluate()` の判定結果は旧 `evaluateOnly()` / `computeDecision()` と完全に等価。
