# Authority Inventory（Phase0成果物）

作成日: 2026-05-31
対象スコープ（Production Runtime Tree）: `src/audioengine/**`, `src/convolver/**`, `src/eqprocessor/**`, `src/core/**`

---

## 1. 目的

Practical Stable ISR Bridge Runtime 完全移行に向け、実行時 Authority の所在と競合経路を列挙し、Phase1 以降の改修対象を固定する。

---

## 2. 収集方法

- キーワード監査: `commit|publish|retire|build|generation|transition|snapshot`
- 呼び出し点監査: `publishState()`, `prepareCommit()`, `executeCommit()`, `commitNewDSP()`
- 補助経路監査: `runtimePublicationBridge_`, `deferredDeleteFallbackQueue`, `SafeStateSwapper`

---

## 3. 最新ベースライン（2026-05-31）

| 指標 | 実測値 | 主なファイル |
| --- | --- | --- |
| `publishState()` callsite（audioengine 本番 `.cpp`） | 9 | `AudioEngine.Commit.cpp`, `AudioEngine.Processing.PrepareToPlay.cpp`, `AudioEngine.Processing.ReleaseResources.cpp`, `AudioEngine.Timer.cpp` |
| commit 旧経路エントリ | 3 | `AudioEngine::prepareCommit`, `AudioEngine::executeCommit`, `AudioEngine::commitNewDSP` in `AudioEngine.Commit.cpp` |
| `transition.active` 実行分岐 | 5 | `AudioEngine.Processing.AudioBlock.cpp`, `BlockDouble.cpp`, `Snapshot.cpp`, `AudioEngine.Timer.cpp` |
| RuntimeBuilder の `ConvolverProcessor::BuildSnapshot` 依存 | 2 | `RuntimeBuilder.h` / `RuntimeBuilder.cpp` |
| EQ 側 fallback queue 利用 | 複数（10 hit） | `EQProcessor.h`, `EQProcessor.Core.cpp` |

---

## 4. Authority 分類（現状）

### 4.1 Publication / Commit

| シンボル | 所在 | 現在の役割 | Authority判定 | 目標 |
| --- | --- | --- | --- | --- |
| `convo::RuntimePublicationCoordinator::publishState` | `src/core/RuntimePublicationCoordinator.h` | RuntimeStore への publish/swap 実行 | Authoritative（実行） | 単一路の実行主体として維持 |
| `AudioEngine::prepareCommit` | `src/audioengine/AudioEngine.Commit.cpp` | commit 旧経路 | Authoritative（旧） | Phase1 で到達不能化 |
| `AudioEngine::executeCommit` | 同上 | commit 旧経路 | Authoritative（旧） | Phase1 で到達不能化 |
| `AudioEngine::commitNewDSP` | 同上 | publish/retire 呼出し元を内包 | Authoritative（旧） | Phase1 で到達不能化 |
| `AudioEngine::commitRuntimePublication` | `src/audioengine/AudioEngine.h` | ISR bridge へ publication metadata 通知 | Governance/Telemetry | Topology Decision 後に扱い固定 |
| `AudioEngine::retireRuntimePublication` | `src/audioengine/AudioEngine.h` | ISR bridge へ retire metadata 通知 | Governance/Telemetry | Topology Decision 後に扱い固定 |

### 4.2 Transition / DSP Selection

| シンボル | 所在 | 現在の役割 | Authority判定 | 目標 |
| --- | --- | --- | --- | --- |
| `runtimePublishView.transition.active` | `src/audioengine/*.cpp` | DSP 選択分岐（fading 有無） | Authoritative（実行分岐） | Phase5で責務移設後に診断化 |
| `TransitionState::policy` | `src/audioengine/RuntimeTransition.h` + `AudioEngine.h` | 値投影（実行分岐なし） | Projection寄り | 診断/投影へ限定 |
| `TransitionState::latencyDeltaSamples` | 同上 | 値投影（実行分岐なし） | Projection寄り | 診断/投影へ限定 |

### 4.3 Build / Convolver Coupling

| シンボル | 所在 | 現在の役割 | Authority判定 | 目標 |
| --- | --- | --- | --- | --- |
| `RuntimeBuilder::build(...BuildSnapshot...)` | `src/audioengine/RuntimeBuilder.h/.cpp` | RuntimeBuilder が Convolver 型を直参照 | Coupling Source | Phase8-A で依存消滅 |
| `PendingParams` | `src/ConvolverProcessor.h` | Convolver 内 SoT | Authoritative（局所） | Phase8-B で撤去 |
| `PreparedIRState` | `src/convolver/**` | IR 準備結果の保持 | Authoritative（局所） | Phase8-B で撤去/統合 |
| `SafeStateSwapper` | `src/SafeStateSwapper.h`, `ConvolverProcessor.h` | 旧RCU保護基盤 | Alternate Runtime Path 候補 | Phase8-B で撤去 |

### 4.4 Retire / Deletion

| シンボル | 所在 | 現在の役割 | Authority判定 | 目標 |
| --- | --- | --- | --- | --- |
| `retireDSP` | `src/audioengine/AudioEngine.h` | DSP retire 入口 | Authoritative（現行） | Phase7 で RetireManager へ移行 |
| `audioThreadRetireOverflowPtr` | `src/audioengine/AudioEngine.h` | RT overflow 退避 | Retire補助 | Phase7 で統合 |
| `deferredDeleteFallbackQueue`（AudioEngine） | 同上 | 非RT fallback queue | Retire補助 | Phase7 で統合 |
| `deferredDeleteFallbackQueue`（EQProcessor） | `src/eqprocessor/EQProcessor.h/.Core.cpp` | EQ専用 fallback queue | Retire補助（独立経路） | Phase7 で統合（C15） |
| `RetireRuntimeEx` | `src/audioengine/ISRRetireRuntimeEx.*` | rollback/epoch policy | Governance | Coordinator/Retire責務境界を固定 |

---

## 5. 優先度付き課題（Phase対応）

### P0（最優先）

1. Publication authority 多点経路（`publishState` callsite=9）
2. commit 旧経路（`prepareCommit`/`executeCommit`/`commitNewDSP`）
3. Coordinator 二層の責務未固定（Phase0.5 未完）

### P1（高）

1. RuntimeBuilder の BuildSnapshot 逆依存
2. `transition.active` が DSP選択を支配（Phase5設計ゲート未通過）

### P2（中）

1. Convolver 局所 SoT（`PendingParams`, `PreparedIRState`, `SafeStateSwapper`）
2. Retire 経路の分散（AudioEngine + EQ）

---

## 6. Phase1/8/5/7 へのトレーサビリティ

- Phase1 Exit-A: `publishState()` callsite = 1
- Phase1 Exit-B: `prepareCommit` / `executeCommit` / `commitNewDSP` 到達不能
- Phase8-A Exit-1/2: RuntimeBuilder の `ConvolverProcessor` / `BuildSnapshot` 依存消滅
- Phase5-Gate: `transition.active` 代替 state machine 設計承認
- Phase7: RetireManager 新設 + AudioEngine/EQ fallback 統合

---

## 7. 機械検証用チェック（初期版）

- `publishState\(` in `src/audioengine/**/*.cpp`
- `void AudioEngine::(prepareCommit|executeCommit|commitNewDSP)\(` in `src/audioengine/*.cpp`
- `runtimePublishView\.transition\.active` in `src/audioengine/*.cpp`
- `ConvolverProcessor::BuildSnapshot` in `src/audioengine/RuntimeBuilder.*`
- `deferredDeleteFallbackQueue` in `src/eqprocessor/**`

---

## 8. まとめ

Authority Collapse の主戦場は `AudioEngine` の publish/commit/retire 経路にあり、Convolver/Transition/Retire はそれに連鎖する第二段階課題である。
したがって、Phase0.5 を起点に **Publication → BuildSnapshot依存崩し → Convolver統合 → RuntimeState閉包化** の順で進める。
