# BUG-029: DSPTransition Emergency Override パスが exchangeFadingRuntimeDSP を呼ばない

## 発見日
2026-07-26

## ファイル
`src/audioengine/DSPTransition.h:54-74`

## 問題
通常のクロスフェードパス（DSPTransition.h:91-92）は以下を実行する:
```cpp
auto* prevRaw = engine_.exchangeFadingRuntimeDSP(oldDSP);
```

これにより `oldDSP` が `fadingRuntimeDSPSlot` にアトミックに格納され、
前回の occupant が取得される（必要なら retire）。

Emergency Override パス（line 54-74）はこの呼び出しをスキップする:
```cpp
{
    auto ref = engine_.getHealthStateRef();
    if (ref) {
        auto health = convo::consumeAtomic(*ref, std::memory_order_acquire);
        if (health == convo::ISRHealthState::Critical) {
            lifetime.activate(newDSP);             // activeRuntimeDSPSlot = newDSP
            if (oldDSP != nullptr) {
                engine_.crossfadeRuntime_.complete();
                lifetime.retire(oldDSP);           // oldDSP を直接 retire
                // ★★★ exchangeFadingRuntimeDSP を呼ばない！★★★
                ...
            }
            return;  // 通常パスをスキップ
        }
    }
}
```

### 漏れシナリオ
1. 遷移 #1: fadingRuntimeDSPSlot ← DSP-A（通常のクロスフェード）
2. 遷移 #1 完了: fadingRuntimeDSPSlot = nullptr（Timer::880 でクリア）
3. 遷移 #2: fadingRuntimeDSPSlot ← DSP-B（通常のクロスフェード）
4. 遷移 #2 の途中で HealthState::Critical が発生
5. 遷移 #3（Emergency Override）:
   - exchangeFadingRuntimeDSP を呼ばない
   - DSP-B は fadingRuntimeDSPSlot に残ったまま
6. Timer::1000 で `!m_coordinator.isFading()` → exchangeFadingRuntimeDSP(nullptr)
   → DSP-B を取得 → retire → **まだ feding 中の DSP-B が先に retire される！**

### 影響
- Audio Thread が DSP-B への参照を RuntimeWorld 経由で保持している可能性
- DSP-B が EBR 経由で破棄された後、Audio Thread が使用 → Use-After-Free
- 現実的には HealthState::Critical は稀なため発生頻度は低い

### リスク評価
- **重大度**: HIGH — Use-After-Free の可能性
- **発生頻度**: 非常に低い（HealthState::Critical + 進行中のクロスフェードの重なり）
- **影響範囲**: オーディオクラッシュ、メモリ破壊

### 修正方針
Emergency Override パスでも `exchangeFadingRuntimeDSP` を呼ぶ:
```cpp
if (health == convo::ISRHealthState::Critical) {
    lifetime.activate(newDSP);
    if (oldDSP != nullptr) {
        auto* prevRaw = engine_.exchangeFadingRuntimeDSP(oldDSP);
        engine_.crossfadeRuntime_.complete();
        lifetime.retire(oldDSP);
        // prevRaw があれば retire
        if (auto* prev = recoverDSPPtr(prevRaw))
            lifetime.retire(prev);
        ...
    }
    return;
}
```
