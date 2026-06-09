# 未確定事項 最終確定レポート v6

**作成日**: 2026-06-09
**調査手段**: CodeGraph MCP (12,739 entities), Serena MCP, grep, 直接読取

---

## 調査結果

| # | 項目 | 確定内容 |
| --- | --- | --- |
| 1 | runtimeStore 読取 | AudioEngine.h内9箇所 + Orchestrator.cpp 1箇所 (observe)。Storeはpublic (line 2716) |
| 2 | m_epochDomain 直接呼び出し | **AudioEngine: 2箇所** (drainAllのみ) + RCUReader初期化2箇所 + コンストラクタ注入。EQProcessor: 9箇所 (別ドメイン) |
| 3 | FailureRecord 既存 | **なし**。`PublicationFailureTaxonomyVerifier` はスキーマ検証器のみ |
| 4 | Monotonicity検出 | **既存**: `observeMonotonicViolationCount_` と `observeMonotonicRollbackRequested_` が `makeRuntimeReadHandle()` 内で実装済み。ただし診断専用でFaulted遷移なし |
| 5 | ReaderContext 直接構築 | **2箇所**: Orchestrator.cpp + ReleaseResources.cpp。ともに正しい組合せ |

---

## 詳細

### 1. runtimeStore 全アクセス

**AudioEngine.h 内の読取 (9箇所)**:

- `getRuntimeGraph()` (line 905-906): acquireReadToken + consumeWorldHandle
- `makeRuntimeReadHandle()` (line 2333-2334): 同上
- `computeRuntimePublishComputation()` (line 2599-2600): 同上
- `logRuntimeTransitionEvent()` (line 2767-2768): 同上
- `makeRuntimePublicationCoordinator()` (line 2727): Coordinator生成時に渡す

**Orchestrator からの直接読取 (1箇所)**:

- `RuntimePublicationOrchestrator.cpp:67`: `engine_.runtimeStore.observe()`

**書換**: `Coordinator::publishWorld()` → `WriteAccess::publishAndSwap()` のみ。

### 2. m_epochDomain 直接呼び出し — 2箇所 (AudioEngine)

| # | ファイル | 呼び出し | 対処 |
|---|---------|---------|------|
| 1 | `AudioEngine.CtorDtor.cpp:131` | `m_epochDomain.drainAll()` | Router委譲 |
| 2 | `AudioEngine.Processing.ReleaseResources.cpp:208` | `m_epochDomain.drainAll()` | Router委譲 |

**RCUReader初期化 (変更不可)**:

- `AudioEngine.h:3370`: `audioThreadRcuReader { m_epochDomain }`
- `AudioEngine.h:3372`: `messageThreadRcuReader { m_epochDomain }`

**コンストラクタ注入**:

- `AudioEngine.CtorDtor.cpp:21`: `m_coordinator(m_epochDomain)` → `m_coordinator(*m_retireRouter)` に変更可能
- `AudioEngine.CtorDtor.cpp:26`: `make_unique<ISRRetireRouter>(m_epochDomain)` — やむを得ない

### 3. Monotonicity 検出 — 既存 (`makeRuntimeReadHandle`)

```cpp
// AudioEngine.h:2340-2352 (makeRuntimeReadHandle 内)
const bool generationBackward = (previousGeneration != 0 && currentGeneration < previousGeneration);
const bool sequenceBackward = (previousSequence != 0 && currentSequence < previousSequence);

if (generationBackward || sequenceBackward) {
    fetchAddAtomic(observeMonotonicViolationCount_, 1, acq_rel);
    publishAtomic(observeMonotonicRollbackRequested_, true, release);
}
```

**既存だが不足**:

- `observeMonotonicViolationCount_` は診断カウンタ — 誰も参照して自動対応しない
- `observeMonotonicRollbackRequested_` はフラグ — 誰も消費しない
- Evidence 出力なし
- Faulted 遷移なし

### 4. ReaderContext 直接構築 — 2箇所のみ

| ファイル | コード | 正誤 |
|---------|--------|:----:|
| `RuntimePublicationOrchestrator.cpp:24` | `RuntimeReaderContext{ publicationReader, ObserveChannel::Publication }` | ✅ |
| `AudioEngine.Processing.ReleaseResources.cpp:92` | `RuntimeReaderContext{ messageThreadRcuReader, ObserveChannel::Message }` | ✅ |

ヘルパー関数使用:

- `NoiseShaperLearner.cpp:1034`: `makeWorkerReaderContext(rcuReader, 0)` ✅
- `SpectrumAnalyzerComponent.cpp:278`: `makeMessageReaderContext(rcuReader)` ✅

---

## 改修計画への影響

| 発見 | 影響 |
|------|------|
| Store読取は全getter経由に変更可能 | P0-4 の移行範囲確定 (getter追加 + 10箇所変更) |
| AudioEngineのm_epochDomain直接呼び出しは2箇所のみ | P1-4 の工数小 (drainAll委譲のみ) |
| Monotonicity検出は既存 | P1-9 は検出部ではなく対応部 (Faulted + Evidence) のみ実装 |
| ReaderContext直接構築2箇所 (ともに正しい) | P3-2 の優先度低で確定 |
