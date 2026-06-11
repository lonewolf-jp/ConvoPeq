# ISR Bridge Runtime 第三者レビュー 妥当性検証レポート

**作成日**: 2026-06-11
**調査者**: GitHub Copilot (DeepSeek V4 Flash)
**対象**: ConvoPeq ISR Bridge Runtime (Practical Stable)
**使用ツール**: grep/Select-String, Serena MCP, CodeGraph MCP, ccc (cocoindex-code), Graphify MCP (DeepSeek), semble

---

## 調査方法

全10指摘について、以下のプロセスで検証:

1. **静的コード解析**: grep/Select-String でパターン検索
2. **セマンティックコード検索**: Serena MCP (LSPベース), semble (ベクトル検索), ccc (ASTベース)
3. **依存関係解析**: CodeGraph MCP (KuzuDB), Graphify MCP (知識グラフ + DeepSeek)
4. **コード実読**: 該当箇所の全文読取

---

## 総評

| 観点 | 評価 |
|------|------|
| レビューの正確性 | 10指摘中 **8件は完全に妥当**、2件は部分的妥当 |
| 現状完成度 | 85〜90%（レビュー評価と一致） |
| 最も深刻な乖離 | ISRRetireRouter の関数ポインタ未初期化 (動的キャスト逃避) |
| 最も修正効果が大きい項目 | HealthMonitor → Admission 閉ループ完成 |
| レビューが過小評価した点 | Retire Queue圧力制御の実装進捗、Shutdown FSMの完成度 |

---

## 個別検証結果

### Point 1: EpochDomain 完全隠蔽が未達成 — ✅ **妥当（高優先）**

**エビデンス**: `ISRRetireRouter.cpp` L110-120

```cpp
// pendingRetireCount()
auto* ed = dynamic_cast<EpochDomain*>(provider_);
return ed ? ed->pendingRetireCount() : 0;

// drainAll()
if (auto* ed = dynamic_cast<EpochDomain*>(provider_))
    ed->drainAll();
```

**より深刻な問題**: `ISRRetireRouter.h` に関数ポインタ宣言あり:

```cpp
uint32_t (*pendingRetireFn_)(void*) = nullptr;
void     (*drainAllFn_)(void*)      = nullptr;
void*    epochContext_              = nullptr;
```

→ **コンストラクタで一度も初期化されていない。** 設計図（ヘッダ）と実装（cpp）が完全に乖離。

**確認ツール**: grep, Serena (find_file/find_symbol), CodeGraph (get_code_snippet), semble (semantic search)

---

### Point 2: Deprecated Epoch API がまだ生存 — ⚠️ **部分的に妥当（中優先）**

**エビデンス**:

`EpochDomain.h` に deprecated 付きで生存:

| API | 呼び出し実績 |
|-----|-------------|
| `advanceEpoch()` | CI gate で `RuntimePublicationOrchestrator` のみ許可（1箇所） |
| `reclaimRetired()` | ソースコード内での実呼び出しなし（docファイルのみ） |
| `enqueueRetire()` (4-param deprecated) | **EQProcessor.Core.cpp L60 で呼び出しあり**（fallback経路） |
| `enterReader()`/`exitReader()` deprecated | RCUReader から呼び出し → Interface経由で警告抑制済み |

**問題**: deprecated 関数が存在することで C4996 警告が全翻訳単位で大量発生。移行はかなり進行しているが、EQProcessor の fallback 経路が deprecated を呼び続けている。

**確認ツール**: grep (advanceEpoch/reclaimRetired), Serena (find_symbol)

---

### Point 3: Direct EpochDomain Fallback が残存 — ✅ **妥当（高優先）**

**エビデンス**: `EQProcessor.Core.cpp` L56-66

```cpp
// Fallback: direct EpochDomain path (backward compat before coordinator is set)
// [Phase-B] coordinator 常時設定確認後、この経路は削除.
#pragma warning(push)
#pragma warning(disable : 4996)
    const bool ok = m_epochDomain.enqueueRetire(ptr, deleter, ...);
#pragma warning(pop)
```

**問題**:

1. コメントに「削除」とあるが削除されていない
2. `deprecated` 警告を握り潰している
3. coordinator 未設定時の挙動が未定義

**確認ツール**: grep ("Fallback.*direct EpochDomain"), ccc, Serena

---

### Point 4: Reader Slot Exhaustion の監視不足 — ✅ **妥当（中優先）**

**エビデンス**:

`EpochDomain::registerReaderThread()`:

```cpp
return -1;  // ← 監視・通知なし
```

`RCUReader::enter()`:

```cpp
if (tid >= 0)
    epochProvider->enterReader(tid);
else
{
    // ← slot取得失敗: 静かに失敗するのみ
}
```

`readerRegistrationFailureCount` のようなカウンタは**存在しない**。
`RuntimeHealthMonitor` に reader exhaustion 検出機構も**ない**。

**確認ツール**: grep (readerRegistrationFailure), Serena (find_symbol), ccc

---

### Point 5: Retire Queue 圧力制御が不十分 — ⚠️ **限定的に妥当（低〜中優先）**

**エビデンス**: レビューの認識より**相当程度実装が進んでいる**。

`AudioEngine.Retire.cpp` に実装済み:

- `retirePressureLevel_` (0-3 段階制御)
- `retirePressurePublicationThrottleActive_` (level>=2)
- `retirePressureAdmissionStrict_` (level>=3)
- 適応的 HWM/LWM 調整（saturation 検出 + recovery）
- overflow 頻度ベースの chronic 検出

**不足**: `RuntimeHealthMonitor::checkRetireStall()` と上記圧力制御が独立動作。HealthMonitor の診断結果が圧力制御にフィードバックされる経路は**ない**。

**確認ツール**: grep (retirePressure), Serena (find_symbol), CodeGraph

---

### Point 6: Runtime Health Monitor が Runtime 制御に未接続 — ✅ **妥当（中優先）**

**エビデンス**:

`RuntimeHealthMonitor` の出力先:

- ✅ `HealthEventCallback` → ログ/診断は可能
- ❌ Admission へのフィードバック → **なし**
- ❌ `PublicationAdmission::PressureLevel` への接続 → **なし**
- ❌ `retirePressurePublicationThrottleActive_` への配線 → **なし**

HealthMonitor は純粋な pull型監視エンジンであり、診断結果を制御に反映する閉ループが存在しない。

```cpp
// RuntimeHealthMonitor.h: "Pull型監視エンジン" ← 診断色が強い
class RuntimeHealthMonitor {
    void tick() noexcept;        // checkRetireStall + checkPublicationStall + diagnoseRetireStall
    // ↑ 診断のみ。制御なし。
};
```

**確認ツール**: grep (HealthMonitor.*Admission), Serena, CodeGraph

---

### Point 7: Shutdown 完了条件が弱い — ⚠️ **限定的（低優先）**

**エビデンス**: レビューの認識より**はるかに進んでいる**。

`ShutdownRuntime` の現状:

- 10フェーズ FSM: Running → ... → VerifyDrained → ShutdownComplete/TimedOut/Failed
- Bounded teardown counters: SH-1〜SH-6
- `RuntimeDrainAudit` 構造体: pendingPublication / pendingRetire / activeCrossfadeCount / deferredPublish / routerPendingRetire
- `VerifyDrained` phase での最終監査

**不足**: `RuntimeDrainAudit::isAllZero()` は「監査ログ出力専用。shutdown 完了判定には使用しない」と明記。完了条件として使用されていない。

**確認ツール**: Serena (find_symbol), grep, CodeGraph

---

### Point 8: Runtime World と Retire Epoch の整合監査不足 — ⚠️ **限定的（低優先）**

**エビデンス**:

- `RuntimeWorldIdGenerator` 存在、worldId は診断目的で割り当て
- `RuntimePublicationCoordinator` が sequenceId/epoch/generation の単調増加を検証
- しかし「全 RuntimeWorld が worldId/generation/publishEpoch/retireEpoch を持つ」形の構造的追跡は未実装

**確認ツール**: grep (WorldLifecycleAudit/worldId), Serena

---

### Point 9: Crossfade 完了保証が Timer 依存 — ✅ **妥当（高優先）**

**エビデンス**:

`DSPTransition.h` L95-97:

```cpp
// notifyTransitionComplete: クロスフェード完了時の処理
// Timer から呼ばれる
```

`AudioEngine.Timer.cpp` L379-410:

```cpp
const bool fadeCompleted = m_coordinator.tryCompleteFade();
if (fadeCompleted) {
    // DSP retire, crossfadeRuntime_.complete(), publish idling world...
    // ← 全て Timer (MessageThread) 依存
}
```

`advanceFade()`（AudioThreadで実行）と `tryCompleteFade()`（Timer=MessageThreadで実行）の非対称設計。

**問題**: Timer 遅延・停止・MessageThread飽和時に crossfade 完了検出が遅延/停止する。

**確認ツール**: grep (Timer.*crossfade), ccc, CodeGraph

---

### Point 10: Runtime Admission 自己防衛が弱い — ⚠️ **限定的（低〜中優先）**

**エビデンス**:

`PublicationAdmission::evaluate()` の現状:

| チェック項目 | 状態 |
|---|---|
| Shutdown 中 | ✅ `RejectedShutdown` |
| Generation 不一致 | ✅ `RejectedStaleGeneration` |
| IR未Finalize | ✅ `RejectedNotFinalized` |
| Pressure 有効 | ✅ `RejectedPressure`（`retirePressurePublicationThrottleActive_` 経由） |
| Fading 中 | ✅ `DeferredFadingActive` |

**不足**:

- HealthMonitor 診断（Reader stuck / Publication stall / Retire backlog）→ フィードバックなし
- Quarantine 増加 → チェックなし
- Publication 失敗継続 → チェックなし
- `RuntimeHealthSnapshot` による動的 PressureLevel 調整 → なし

**確認ツール**: Serena, CodeGraph, grep

---

## 使用ツール一覧と成果

| ツール | 用途 | 成果 |
|--------|------|------|
| grep/Select-String | パターン検索 | dynamic_cast, deprecated API, Fallback, Timer依存 等の全該当箇所特定 |
| Serena MCP | シンボル検索・宣言検索 | IEpochProvider階層, RCUReader, 全関連シンボルの特定 |
| CodeGraph MCP | 依存関係・コード断片取得 | ISRRetireRouter依存関係, 関数呼び出し関係の可視化 |
| ccc (cocoindex-code) | ASTベースコード検索 | EQProcessor Fallback, CrossfadeAuthority, HealthMonitor の正確な該当行特定 |
| Graphify MCP (DeepSeek) | 知識グラフ解析 | ISRRetireRouter→EpochDomain のグラフ構造、72ノードの関係マップ |
| semble | セマンティックコード検索 | レビュー文書と実コードの対応付け、既存の検討経緯の把握 |
