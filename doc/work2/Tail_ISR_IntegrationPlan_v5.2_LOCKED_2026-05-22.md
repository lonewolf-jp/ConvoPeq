# Tail ISR 統合計画（v5.2 LOCKED）

- Project: ConvoPeq
- Date: 2026-05-22
- Status: **LOCKED**（この版を設計基準として固定）
- Scope: 既存 ISR runtime を壊さず Tail 機能を安全復活

---

## 1. 目的（固定）

本計画は以下を同時に満たすことを目的とする。

1. correctness-first（まず正しく動く）
2. observability-first（先に可観測化）
3. shutdown safety（late publish抑止）
4. rebuild suppression（storm抑止）
5. optimization postponement（最適化は後段へ延期）

---

## 2. 非目的（固定）

本フェーズでは実施しない。

- 新runtime frameworkの新設
- 高度な incremental rebuild（Tier B最適化）
- 多層 transition graph の導入
- retire queue の高度な backpressure 制御

---

## 3. コア契約（v5.2）

### 3.1 世代管理（最小構成）

- 管理軸は `requestedGeneration` と `runtimeVersion` のみを必須とする。
- publish 成功時のみ `runtimeVersion` を増加させる。

### 3.2 可視化順序（必須）

`requestedGeneration` の更新は、**pending state が fully finalized かつ可視化済み**の後に行う。

運用規約（実装指針）:

1. pending fields update
2. release 可視化（必要に応じて fence）
3. `requestedGeneration` increment

### 3.3 obsolete discard（必須）

- build 開始時に取得した generation と、publish 直前の最新 generation が不一致なら publish しない。
- 不一致 build は obsolete discard 扱いで破棄する。

### 3.4 debounce 適用位置（必須）

**Debounce applies to rebuild scheduling only.**

- UI state 更新は即時
- rebuild request のみ coalesce/debounce
- publish 層には debounce を入れない

### 3.5 debounce 値ポリシー（必須）

固定値ではなく、**configurable minimum debounce floor** を採用する。

- `not lower than X ms` の下限契約を採る
- 実測で調整可能にする

### 3.6 lifecycle guard（必須）

- guard owner は `ConvolverProcessor` または runtime manager として明示固定する。
- publish 直前に guard 有効性を検証し、無効なら publish をスキップする。

### 3.7 diagnostics スレッド規約（必須）

- 文字列ログは message/build thread only
- RT thread は必要時に RT-safe ring buffer tracing のみ

### 3.8 diagnostics 軽量性（必須）

**Diagnostics must remain lightweight in rebuild hot paths.**

- 高頻度ログは sampled/rate-limited
- hot path で過剰な動的確保を避ける

### 3.9 legacy semantic（必須）

旧プリセット（tail未指定）は、**Tail processing pipeline skipped entirely** を基本契約とする。

- 可能な限り上流で bypass する
- 下流関数内だけの部分bypassに依存しない

---

## 4. rebuild 方針（固定）

Tail 復活フェーズでは correctness 優先のため、Tail 関連変更は **full rebuild 固定** とする。

- Tier B 最適化は Phase 2 へ延期
- まず退行リスク最小化を優先

---

## 5. 実装順序（固定）

1. observability（最小診断基盤）
2. suppression（scheduling debounce + coalescing）
3. generation safety（可視化順序 + discard）
4. lifecycle safety（guard owner固定 + late publish抑止）
5. Tail restore（full rebuild固定、legacy bypass）

---

## 6. 最小診断イベント（Debug）

以下のみを必須イベントとする（高頻度は rate-limit）。

- rebuild queued
- rebuild started
- obsolete discard
- publish success
- publish skipped（guard invalid 含む）
- retire drain

---

## 7. 受け入れ基準（固定）

1. 連続 slider 操作時に UI 即時性を維持しつつ rebuild storm が抑制される
2. obsolete discard 判定が世代不整合で破綻しない
3. shutdown 中の late publish が guard で抑止される
4. 高頻度操作でもログ洪水が発生しない（rate-limit 有効）
5. 旧プリセットで Tail pipeline bypass が成立する
6. RT 制約（lock/alloc/blocking/非RT-safe logging）に違反しない

---

## 8. ロールアウト方針

- staged rollout を採用する
- instrumentation validation 後に Tail 復活を有効化する
- 問題時は Tail path を機能フラグで即時切り戻し可能にする

---

## 9. ロック宣言

本書 `Tail_ISR_IntegrationPlan_v5.2_LOCKED_2026-05-22.md` を、Tail ISR 統合の設計基準としてロックする。

- 以後の変更は差分追記（v5.2.x）で管理する
- 本体方針（correctness-first / observability-first / optimization-postponed）は不変とする
