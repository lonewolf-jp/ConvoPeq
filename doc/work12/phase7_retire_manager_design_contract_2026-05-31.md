# Phase7: RetireManager 設計契約

作成日: 2026-05-31
対象: `src/audioengine/**`, `src/eqprocessor/**`, `src/core/**`

## 1. 目的

Retire 経路（AudioEngine/EQProcessor の fallback queue、RT overflow ptr、DeferredFreeThread/ISRRetireRuntimeEx）を
単一の `RetireManager` に統合し、retire authority を単一路化する。

## 2. 統合対象

- `audioThreadRetireOverflowPtr`
- `deferredDeleteFallbackQueue` (AudioEngine)
- `deferredDeleteFallbackQueue` (EQProcessor)
- `DeferredFreeThread` 連携
- `ISRRetireRuntimeEx` 連携

## 3. RetireManager 責務

1. retire intent の受理・順序管理（generation monotonic）。
2. grace 判定（reader/executor/transition）の一元評価。
3. backlog 圧力制御（throttle/escalation）
4. fallback queue 収束（EQ/AE を単一コンテナへ集約）
5. shutdown drain 経路の deterministic 完了保証。

## 4. API 契約（To-Be）

- `enqueueRetire(ptr, deleter, generation, domain)`
- `tickNonRt(now, pressurePolicy)`
- `drainForShutdown(timeout)`
- `snapshotTelemetry()`

## 5. Authority 境界

| 操作 | Owner | 備考 |
| --- | --- | --- |
| retire admission | RetireManager | 非RTのみ |
| grace completion | RetireManager + ISRRetireRuntimeEx | 実行観測入力を使用 |
| reclaim execute | RetireManager | deleter実行責務 |
| telemetry export | RetireManager | C15判定源 |

## 6. 品質ゲート

- C15: `EQProcessor::deferredDeleteFallbackQueue = 0`（PRT）
- Retire authority owner = 1
- shutdown 時 pending retire zero
- build/test green

## 7. 承認記録

- 承認フェーズ: `Phase7 design pre-approval`
- 判定: **Approved (Design Ready)**
- 実装着手条件:
  - Phase6 完了
  - Phase5 の state machine 運用安定を確認
