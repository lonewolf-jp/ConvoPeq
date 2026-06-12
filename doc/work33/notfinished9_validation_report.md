# notfinished9.md 妥当性検証報告書

**日付**: 2026-06-12
**検証者**: GitHub Copilot (DeepSeek V4 Flash)
**検証対象**: `doc/work33/notfinished9.md` — Practical Stable ISR Bridge Runtime との乖離レビュー

---

## 検証方法

以下のツールを駆使してソースコードの網羅的調査を実施:

| ツール | 用途 | 結果 |
|--------|------|------|
| **grep/Select-String** | 識別子・パターン検索 | ✓ 全項目網羅 |
| **Serena MCP** | プロジェクト構造把握・メモリ読み取り | ✓ 16件のメモリ参照 |
| **CodeGraph MCP** | 依存関係・呼び出し関係解析 | ✓ クエリ実行 |
| **graphify MCP** | ナレッジグラフ: 15,779ノード, 34,204エッジ, 297コミュニティ | ✓ 近傍探索 |
| **AiDex MCP** | コードインデックス: 275ファイル, 3,952メソッド, 369型 | ✓ セッション開始・ステータス確認 |
| **semble CLI** | セマンティックコード検索 | ✓ 20件結果取得 |
| **cocoindex-code** | CLI解析 | ✗ バイナリ未インストール |

---

## 第1章: 各乖離項目の妥当性検証

### 項目1: HealthMonitor が Shutdown 完了条件に統合されていない

**レビューの主張**: HealthMonitorは異常検出のみで、Shutdown Runtime / Publication Admission / RuntimeDrainAudit との閉ループ制御が不十分。

**検証結果**: ✅ **概ね妥当。ただし「検出のみ」表現は古い**

HealthState は以下の広範囲へ伝播しており、notfinished9 が想定したより閉ループは進んでいる：

- `RuntimeHealthMonitor` → `ISRHealthState` (atomic) への5系統統合: **実装済み** ✓
- `PublicationAdmission` が HealthState Critical/Degraded で publish 拒否: **実装済み** ✓
- `shouldRejectRebuildAdmissionForPressure()` が Critical チェック: **実装済み** ✓
- `DSPTransition::onPublishCompleted()` が Critical 時 crossfade スキップ: **実装済み** ✓
- `CrossfadeAuthority::decide()` が Critical 時 none 返却: **実装済み** ✓
- `RuntimeBuilder::build()` が Critical 時 early reject: **実装済み** ✓

**乖離の残存**:

- `RuntimeDrainAudit::isAllZero()` は HealthState を考慮しない。**review 指摘の通り未実装**
- `RuntimeDrainAudit` に `healthState` フィールドがない。
- HealthMonitor のコールバックは timer 経由で回復処理を実行するが、Shutdown FSM の VerifyDrained フェーズと直接接続されていない。

→ **notfinished9 の指摘は「閉ループが無い」ではなく「Shutdown Authority に統合されていない」に修正すべき**

**結論**: 閉ループは「検出→Admission拒否→一部回復」までは構築されているが、Shutdown 完了条件に HealthState が統合されていない点は review の指摘通り。

---

### 項目2: WorldLifecycleAudit が Diagnostic のみ

**レビューの主張**: WorldLifecycleAudit は Diagnostic 限定で、Shutdown Authority にしていない。publishedCount == retiredCount + activeWorldCount の保証がない。

**検証結果**: ✅ **完全に妥当**

**実装状況**:

- `WorldLifecycleAudit.h` L17: `// Diagnostic 限定 Shutdown Authority にはしない` と明記
- `activeWorldCount_`, `publishedCount_`, `retiredCount_` のみ管理
- リングバッファ (`FixedRingBuffer<WorldLifecycleRecord, 4096>`) による履歴保存
- `emitSnapshot()` / `tryDumpPeriodic()` による evidence JSON 出力のみ
- `verifyConsistency()` メソッド: **未実装** ✗

**乖離の残存**:

- `verifyConsistency()`: **実装されていない** (review 指摘通り)
- リングバッファは追記専用のため `onWorldRetired()` で該当レコードの更新が不可能 (separate `lastRetiredWorldId_`/`lastRetireEpoch_` で代替)

**結論**: review の指摘は完全に正しく、`verifyConsistency()` は実装されていない。

---

### 項目3: Retire Overflow が回復制御へ接続されていない

**レビューの主張**: `m_overflowCount_` は検出されているが、Overflow → Publication制限 → Rebuild抑止の自動制御がない。

**検証結果**: ⚠️ **「回復制御へ接続されていない」は現在では厳密には誤り。ただし明示的なFreeze Stateはない**

**実装状況**:

- `ISRRetireRouter` に `m_overflowCount_` (atomic): **実装済み** ✓
- `RuntimeHealthMonitor::checkOverflowRate()` → Rate 計算 + ヒステリシス付き MonitorState 遷移: **実装済み** ✓
- Overflow Rate は `updateHealthState()` で統合 → Critical/Degraded: **実装済み** ✓
- HealthState Critical → `PublicationAdmission::RejectedPressure`: **実装済み** ✓
- HealthState Critical → `shouldRejectRebuildAdmissionForPressure()`: **実装済み** ✓
- `RuntimeBuilder::build()` が Critical 時 Build 拒否: **実装済み** ✓

**乖離の残存**:

- 「Publication Freeze」という明示的な状態遷移はない。HealthState Critical 経由での間接的な制御のみ。
- review 指摘の `submitPublishRequest()` が Deferred に落ちる動作は、実際には `RejectedPressure` で完全拒否される（Deferred にはならない）。
- 直接的な「Freeze」状態を持たず、Admission の `Decision::RejectedPressure` に委譲している。

→ **notfinished9 の「接続されていない」は過小評価。「閉ループは存在するが明示的な Freeze State は無い」が正確**

---

### 項目4: Reader Leak 自動隔離がない

**レビューの主張**: `detectStuckReaders()` で検出のみで、Zombie Reader として隔離しない。

**検証結果**: ✅ **完全に妥当 — 未実装を確認**

**実装状況**:

- `EpochDomain::detectStuckReaders()`: **実装済み** (epoch gap + depth + residency 複合判定) ✓
- `EpochDomain::ReaderSlot` の構造:

```cpp
struct ReaderSlot {
    std::atomic<uint64_t> epoch { kInactiveEpoch };
    std::atomic<uint32_t> depth { 0 };
    std::atomic<uint64_t> enterCount { 0 };
    std::atomic<uint64_t> residencyStartTimestampUs { 0 };
    std::atomic<uint64_t> ownerThreadId { 0 };
    char ownerTag[32] {};
};
```

- Active/Suspect/Zombie 状態: **未実装** ✗
- 30秒以上停止 Reader の quarantine: **未実装** ✗
- Reader slot の強制解放: **未実装** ✗

**乖離の残存**:

- ReaderSlot に `state: Active/Suspect/Zombie` がない (review 指摘通り)
- 強制隔離の仕組みが一切ない (review 指摘通り)
- DSP 側には `DSPState::Quarantined` があるが、Reader 側には相当機構がない

**結論**: review の指摘は完全に正しい。Reader 隔離機構はまったく実装されていない。

---

### 項目5: RuntimeDrainAudit が Reader 状態を持たない

**レビューの主張**: RuntimeDrainAudit に `activeReaderCount` と `stuckReaderCount` がない。

**検証結果**: ✅ **完全に妥当**

**実装確認** (`RuntimeDrainAudit.h`):

```cpp
struct RuntimeDrainAudit {
    uint64_t pendingPublication;
    uint64_t pendingRetire;
    uint64_t activeCrossfadeCount;
    uint64_t routerPendingRetire;
    uint64_t maxDeferredAgeMs;
    uint64_t deferredPublish;
    uint64_t quarantineResident;    // 監査のみ
    uint64_t oldestPendingAgeMs;    // 監査のみ
    uint64_t maxQuarantineAgeSec;   // 監査のみ
    uint64_t activeWorldCount{0};   // ★ C-1: WorldLifecycleAudit 連携
    uint64_t publishedCount{0};     // ★ C-1
    uint64_t retiredCount{0};       // ★ C-1
};
```

- `activeReaderCount`: **未実装** ✗
- `stuckReaderCount`: **未実装** ✗
- ShutdownBlockingReason に `ReaderActive` があるが、これは `markTimedOut`/`markFailed` 以外でセットされない

**結論**: review の指摘は完全に正しい。

---

### 項目6: Crossfade 完了保証が弱い

**レビューの主張**: Crossfade Timeout は監視のみで、30秒超過時の強制完了がない。

**検証結果**: ⚠️ **部分的に実装済みだが、完全な強制完了ではない**

**実装状況**:

- `RuntimeHealthMonitor::checkCrossfadeTimeout()`: timeout 検出 + `emitOnTransition` → timer コールバック: **実装済み** ✓
- `AudioEngine::onHealthEvent()` の `EVENT_CROSSFADE_TIMEOUT` 処理:
  1. `exchangeFadingRuntimeDSP(nullptr)` で fading DSP 取得 + retire ✓
  2. `crossfadeAuthorityRuntime_.unregisterCrossfade(activeId)` ✓
  3. `crossfadeRuntime_.complete()` ✓

**乖離の残存**:

- **Timeout 回復時に `onTransitionComplete()` が呼ばれない** → idling world が publish されない ✗
- `crossfadeRuntime_.complete()` は pending=false にするが、DSP の retire 後も新たな RuntimePublishWorld が作られない
- review 指摘の「`DSPTransition::onTransitionComplete()` を起動」は、実際の timer ハンドラでは未実装

**結論**: 検出と一部回復（fading DSP retire, crossfade unregister）は実装されているが、`onTransitionComplete()` を呼んでいないため idling world publish が欠落している。この点は review よりさらに悪い状態。

---

### 項目7: RuntimeBuilder の Warmup Validation が非常に弱い

**レビューの主張**: `isIRLoaded() && !isIRFinalized()` のみで、DSP Runtime の健全性を確認していない。

**検証結果**: ✅ **完全に妥当**

**実装確認** (`RuntimeBuilder.cpp`):

```cpp
BuildError RuntimeBuilder::validateWarmup(const AudioEngine::DSPCore& runtime) const noexcept
{
    juce::ignoreUnused(engine);

    if (runtime.convolverRt().isIRLoaded() && !runtime.convolverRt().isIRFinalized())
        return BuildError::WarmupFailed;

    return BuildError::None;
}
```

- `validateRuntimeIntegrity()`: **未実装** ✗
- sampleRate/blockSize/convolver state/EQ coefficient/DSPHandle state の検査: **未実装** ✗
- review の指摘通り、実質的に何も検査していない

**結論**: review の指摘は完全に正しい。

---

### 項目8: Shutdown が「待つ」だけで「収束」しない

**レビューの主張**: `while(...) { publishEpoch(); tryReclaim(); }` ループで収束しない原因を除去していない。

**検証結果**: ⚠️ **部分的に実装済みだが、EmergencyDrain Phase は未実装**

**実装状況**:

- `waitForDrain()` は `publishEpoch()` + `tryReclaim()` ループ: **実装済み** ✓
- ただし timeout 後の fallback として `drainDeferredRetireQueues(true)` + `m_epochDomain.tryReclaim()`: **実装済み** ✓
- Shutdown FSM phase 遷移: `AudioStopped → ObserverDrained → RetireClosed → EpochSettled → ReclaimComplete → VerifyDrained → ShutdownComplete`: **実装済み** ✓
- VerifyDrained で collectDrainAudit の isAllZero() 確認: **実装済み** ✓

**乖離の残存**:

- `EmergencyDrain` ShutdownPhase: **未実装** ✗ (plan 文書でのみ言及)
- Shutdown 中の HealthState Critical → Admission 拒否は働くが、Explicit Emergency Phase はない
- review 指摘の「Deferred Publish 中止 / Crossfade 強制終了 / Reader 隔離 / Publication Admission 停止」は部分的にしか実装されていない
  - Deferred Publish は `clearDeferredForShutdown()` が timer の HealthEvent ハンドラで呼ばれる
  - Crossfade 強制終了は crossfade timeout recovery でのみ
  - Reader 隔離は未実装
  - Publication Admission 停止は HealthState Critical 経由

**結論**: review の主張は概ね妥当だが、実際には timeout 後の回復処理はある程度実装されている。ただし「EmergencyDrain」という明示的な Phase はなく、Reader 隔離もない。

---

## 第2章: 拾い漏れ — notfinished9.md で指摘されていない乖離

### 漏れ1: Crossfade Timeout 回復が `onTransitionComplete()` を呼ばない — 状態遷移経路の分岐

**重大度**: 🔴 **高**（ただし Runtime 即破綻はしない）

**問題**: `AudioEngine::onHealthEvent()` の `EVENT_CROSSFADE_TIMEOUT` 処理で、`crossfadeRuntime_.complete()` により pending=false には戻るが、`DSPTransition::onTransitionComplete()` を呼んでいない。

通常完了経路では `buildRuntimePublishWorld(...)` → `publishWorld(...)` により Idle World が再公開されるが、Timeout Recovery 経路ではそれが行われない。そのため **RuntimePublishWorld の Semantic Projection（投影状態）と実 Runtime 状態が乖离する**。

ただし `crossfadeRuntime_.complete()` により pending=false に戻り、Audio 側の `activeDSP` / `fadingDSP` で実動作は継続できるため、**即座に Runtime が破綻するわけではない**。

**正しい評価**:

- ❌ ~~RuntimePublishWorld が壊れる／stale state が残る~~（言い過ぎ）
- ⭕ **RuntimePublishWorld の Semantic Projection が古くなる**（設計不整合）

**さらに本質的な問題: 状態遷移経路の分岐**

現在 `onTransitionComplete()` にぶら下がっている副作用:

- `exchangeFadingRuntimeDSP(nullptr)` → retire old fading DSP
- `refreshCrossfadePreparedSnapshotFromAtomics()`
- `buildRuntimePublishWorld(...)` + `publishWorld(...)` （Idle World 再公開）

Timeout Recovery 経路はこれらをすべてスキップする。もし今後 `onTransitionComplete()` に以下が追加された場合:

- audit更新
- telemetry更新
- publication更新
- authority更新

Timeout 経路だけ別挙動（分岐）になり、保守性・検証性が低下する。

→ **本質は「Projectionが古くなる」だけでなく「状態遷移経路が分岐している」こと。Practical Stable観点では経路統合が重要。**

**該当ファイル**: `src/audioengine/AudioEngine.Timer.cpp` (EVENT_CROSSFADE_TIMEOUT handler)

```cpp
// 現状: onTransitionComplete() が呼ばれていない
crossfadeRuntime_.complete();
// 欠落: onTransitionComplete() による idling world publish + 全副作用
```

---

### 漏れ2: `WorldLifecycleAudit::verifyConsistency()` が未実装

**重大度**: 🟡 **中**

**問題**: notfinished9.md の項目2で指摘されているが「改修方法」として記載されているのみ。実際の実装有無の確認が不足。現状 **完全に未実装**。

---

### 漏れ3: Shutdown 中の HealthState Critical で Deferred Queue が明示クリアされない

**重大度**: 🟡 **中**

**問題**: `submitPublishRequest()` は HealthState Critical で `RejectedPressure` を返すが、すでに Deferred Queue に滞留している publish request は `clearDeferredForShutdown()` が timer 経由で呼ばれない限り残り続ける。Shutdown FSM の VerifyDrained フェーズで遅延の原因になる。

**該当ファイル**: `src/audioengine/RuntimePublicationOrchestrator.cpp`

---

### 漏れ4: `RuntimeDrainAudit` が HealthState を参照しない

**重大度**: 🟡 **中**

**問題**: `isAllZero()` は pendingPublication/pendingRetire/activeCrossfadeCount/deferredPublish のみ判定し、HealthState を考慮しない。HealthState Critical でも isAllZero()==true になり得る。

**該当ファイル**: `src/audioengine/RuntimeDrainAudit.h`

---

### 漏れ5: `activeReaderCount` が `collectDrainAudit()` に含まれていない

**重大度**: 🟡 **中**

**問題**: `RuntimeDrainAudit` に `activeReaderCount`/`stuckReaderCount` がないため、Shutdown 最終監査 (VerifyDrained) で Reader 状態が考慮されない。Reader が Stuck したままでも Drain 完了と判定される。

**該当ファイル**: `src/audioengine/AudioEngine.Threading.cpp` (collectDrainAudit)

---

### 漏れ6: ShutdownCompletion が HealthState 遷移をトリガーしない

**重大度**: � **中**（Priority B〜C）

**問題**: `ShutdownRuntime::transitionTo(ShutdownComplete)` の前後で HealthMonitor の状態をクリアしない。次回 `prepareToPlay()` → `releaseResources()` → `prepareToPlay()` のサイクルを繰り返す DAW 環境では、HealthState が Critical のまま再初期化される可能性がある。

現時点では証拠不足だが、実運用での影響は無視できない。

---

### 漏れ7: `DSPRegistrySlot` の quarantine が Reader の隔離と独立している

**重大度**: 🟢 **低**

**問題**: DSP 側には `DSPState::Quarantined` があり、Reader Stuck による DSP アクセス不能を検出できる。しかし Reader 側に隔離機構がないため、DSP を quarantine しても Reader が同じスロットを再使用し続ける可能性がある。

---

### 漏れ8: `ISRShutdown.h` の `ShutdownBlockingReason::ReaderActive` が誰もセットしない

**重大度**: 🟢 **低**

**問題**: `ShutdownBlockingReason` に `ReaderActive` が定義されているが、コード内でこの値がセットされるパスがない。`markTimedOut()`/`markFailed()` に渡す引数として定義されているのみ。

**該当ファイル**: `src/audioengine/ISRShutdown.h` (enum class ShutdownBlockingReason)

---

### 漏れ9: `collectDrainAudit()` が World カウンタを収集するが Shutdown 判定に使わない

**重大度**: � **中**（Priority B）

**理由**: Audit欠陥・Shutdown監査欠陥ではあるが、Reader Zombie（reclaim停止・retire停止・memory回収停止を引き起こす）より実害は低い。

**問題**: `collectDrainAudit()` は `WorldLifecycleAudit` から `activeWorldCount`/`publishedCount`/`retiredCount` を収集している。しかし `isAllZero()` や `ShutdownRuntime::VerifyDrained` では**これらの値を一切判定に使用していない**。

つまり「収集しているのに判定に使っていない」状態であり、World 発行数と退役数の不整合（例: 100回 publish したが 99回しか retire していない）を Shutdown 時に検出できない。

これは WorldLifecycleAudit 単体の問題ではなく、**Shutdown Authority 側の欠陥**である。

```cpp
// RuntimeDrainAudit::isAllZero() — World カウンタを完全に無視
bool isAllZero() const noexcept {
    return pendingPublication == 0
        && pendingRetire == 0
        && activeCrossfadeCount == 0
        && deferredPublish == 0;
    // activeWorldCount/publishedCount/retiredCount を考慮しない
}
```

**該当ファイル**:

- `src/audioengine/RuntimeDrainAudit.h` (isAllZero)
- `src/audioengine/AudioEngine.Threading.cpp` (collectDrainAudit)

---

### 漏れ10: `detectStuckReaders()` と `ShutdownBlockingReason::ReaderActive` が断絶している

**重大度**: 🔴 **高**

**問題**: `EpochDomain::detectStuckReaders()` は Reader 異常を検出できるが、その結果が `ShutdownBlockingReason::ReaderActive` に接続されていない。

つまり:

- `detectStuckReaders()` は診断のみで Shutdown FSM に影響を及ぼさない
- `ShutdownBlockingReason::ReaderActive` は enum 定義のみで、誰もこの値をセットしない（漏れ8）
- `collectDrainAudit()` に `activeReaderCount`/`stuckReaderCount` がなく、VerifyDrained で Reader 状態が考慮されない（漏れ5）

**結果**: Reader 異常を検出しても Shutdown FSM が「Reader により停止中」と認識できない。Shutdown タイムアウトしても blocking reason は `Unknown` または誤った理由になる。

これは以下の個別問題の中間にある本質的な欠陥:

- 項目4（Reader Leak隔離なし）
- 項目5（DrainAuditにReader状態なし）
- 漏れ8（ReaderActive未使用）

**該当ファイル**:

- `src/core/EpochDomain.h` (detectStuckReaders)
- `src/audioengine/ISRShutdown.h` (ShutdownBlockingReason)
- `src/audioengine/RuntimeDrainAudit.h`

---

## 第3章: 各乖離の重要度マップ

```text
重要度: 🔴高 = 実運用で破綻し得る
        🟡中 = 長期間運用で劣化
        🟢低 = 監査・可観測性の不足

項目                                  重要度  実装有無  乖離の深さ
─────────────────────────────────────────────────────────────
1. HealthMonitor閉ループ               🟡中   広範囲伝播済 Shutdown Authority未統合
2. WorldLifecycleAudit                 🟡中   診断のみ     verifyConsistency()未実装
3. Retire Overflow制御                 🟡中   間接済       明示的Freezeなし
4. Reader Leak隔離                    🔴高   未実装       Zombie Reader機構なし
5. DrainAudit Reader状態               🟡中   未実装       activeReaderCount欠落
6. Crossfade完了保証                   🔴高   部分済       Semantic Projection乖離
7. Warmup Validation                   🟡中   極小         実質2条件のみ
8. Shutdown収束                        🟡中   部分済       EmergencyDrain Phaseなし

【拾い漏れ】
漏1: Timeout→onTransitionComplete欠落  🔴高（設計不整合、即破綻はしない）
漏2: verifyConsistency未実装             🟡中
漏3: Shutdown中Deferred Queue残留        🟡中
漏4: DrainAuditがHealthState不参照       🟡中
漏5: activeReaderCount欠落               🟡中
漏6: ShutdownCompleteでHealthState未初期化 🟢低
漏7: Reader-DSP quarantine非連動          🟢低
漏8: ReaderActiveが誰もセットしない       🟢低
漏9: Worldカウンタ収集→判定未使用        �中   Shutdown Authority欠陥
漏10: detectStuckReaders→Shutdown断絶    🔴高   検出とFSMの結合欠如
```

---

## 第4章: 結論

### notfinished9.md の評価精度

| 観点 | 評価 |
|------|------|
| **全体の正確性** | 約85〜90%が妥当。ソース確認精度は高い |
| **過小評価** | 項目1(「検出のみ」)は古い。HealthStateはBuilder/Rebuild/Crossfade/Admissionへ広範囲伝播済み |
| **過小評価** | 項目3(「接続されていない」)も古い。間接的閉ループはHealthState経由で成立している |
| **過大評価** | 項目6(Crossfade完了)の「stale state」は言い過ぎ。実際はSemantic Projectionの乖離 |
| **重大な見落とし** | 漏れ9: `collectDrainAudit()` がWorldカウンタを収集するがShutdown判定に使わない — Shutdown Authority欠陥 |

### Practical Stable ISR Bridge Runtime 観点での優先順位

#### Priority A（最優先 — Runtime 健全性に直結）

1. **🔴 Reader Zombie / Quarantine 機構**
   - `ReaderSlot` に段階的状態機械を追加
   - 状態遷移: `Active → Suspect → ZombieCandidate → Manual/Emergency Override`
   - 単純な「30秒→Zombie→slot解放」は **危険**（breakpoint/debugger/OS suspend との区別不能）
   - EpochDomain Reader は所有権を持つため、強制解放は Manual Override または Emergency 経由のみ
2. **🔴 DrainAudit に Reader 状態統合 + detectStuckReaders→ShutdownBlocking 結合**
   - `activeReaderCount` / `stuckReaderCount` を `RuntimeDrainAudit` に追加
   - `detectStuckReaders()` の結果を `ShutdownBlockingReason::ReaderActive` に反映
   - 漏れ10の「Reader異常検出とShutdown FSMの断絶」を解消
3. **🔴 Crossfade Timeout Recovery と Normal Completion の状態経路統一**
   - `onTransitionComplete()` を timeout recovery でも呼び、状態遷移経路の分岐を解消
   - Semantic Projection 乖離 + 今後の副作用追加時の分岐リスクを防止

#### Priority B（次優先 — 監査・Authority 強化）

1. **🟡 World Consistency Verification**
   - `collectDrainAudit()` が収集した World カウンタを `VerifyDrained` で判定に使用
   - `published == retired + active` の保証
2. **🟡 HealthState を Shutdown Authority に統合**
   - `RuntimeDrainAudit` に `healthState` フィールド追加、`isAllZero()` でも考慮
3. **🟡 Warmup Validation 強化**
   - `validateRuntimeIntegrity()` で sampleRate/blockSize/convolver state/EQ coeff を検査

#### Priority C（計画的対応）

1. **🟢 Overflow Freeze State**
   - 明示的な Publication Freeze 状態を追加（現状は HealthState 経由の間接制御のみ）
2. **🟢 EmergencyDrain Phase**
   - 既存の HealthEvent Recovery / Forced Reclaim / Deferred Clear / Timeout Recovery で実用はカバー済み
3. **🟢 ReaderActive BlockingReason の活用**
   - `ReaderActive` が現在誰もセットしない問題の解消
4. **🟢 HealthState Reset 整理**
    - `ShutdownComplete` 前後での HealthState 初期化（DAW 環境での持ち越し対策）

### 現状の総評

2026-06-12版 ConvoPeq は、**ISR Retire、Publication、Health Monitoring、Lifecycle Audit、Shutdown Audit の基本構造は確立されている**。HealthState は Builder/Rebuild/Crossfade/Admission へ広範囲に伝播しており、notfinished9.md が想定したより閉ループは進んでいる。

しかし「Practical Stable ISR Bridge Runtime（実運用で長期間破綻しない）」という観点での最優先課題は、**Reader 異常を Shutdown・Drain・Reclaim へどう統合するか**である。Reader Zombie は reclaim 停止・retire 停止・memory 回収停止を引き起こすため、World Consistency Verification より優先度が高い。

その次に **Crossfade Timeout 経路を通常遷移へ収束させること**（状態遷移経路の分岐解消）、その後に **World Consistency Verification**（Shutdown Authority 欠陥の修復）を置くのが実運用寄りの評価となる。
