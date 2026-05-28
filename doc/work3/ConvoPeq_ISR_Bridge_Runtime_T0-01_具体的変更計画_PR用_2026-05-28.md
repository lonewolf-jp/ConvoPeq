# ConvoPeq ISR Bridge Runtime T0-01 具体的変更計画（PR用）

- Task ID: PR-T0-01
- Tier: Tier0
- Date: 2026-05-28
- Scope: Admission Gate 原子契約と TOCTOU 封じ
- Source:
  - `doc/work3/ConvoPeq_ISR_Bridge_Runtime_v6.x_詳細設計_2026-05-28.md`
  - `src/audioengine/AudioEngine.Commit.cpp`
  - `src/audioengine/AudioEngine.RebuildDispatch.cpp`
  - `src/audioengine/AudioEngine.h`

---

## 1. 目的

以下を最小差分で実装する。

1. Admission 判定の atomic 契約を固定（acquire/release 意味論の明文化）
2. 判定後〜enqueue/publish 直前の TOCTOU 窓を追加ガードで封止
3. shutdown 遷移後の intent 流入を reject 側へ確実に収束

---

## 2. 変更対象関数（確定）

## 2.1 `src/audioengine/AudioEngine.Commit.cpp`

- `[[nodiscard]] bool AudioEngine::acceptsRuntimePublication() const noexcept`
- `void AudioEngine::appendPublicationIntentForCommitSlot(DSPCore* newDSP, int generation, CommitReaderSlot readerSlot) noexcept`
- `void AudioEngine::prepareCommit(DSPCore* newDSP, int generation)`
- `void AudioEngine::executeCommit()`

## 2.2 `src/audioengine/AudioEngine.RebuildDispatch.cpp`

- `void AudioEngine::submitRebuildIntent(...) noexcept`
- `void AudioEngine::requestRebuild(convo::RebuildKind kind) noexcept`
- `void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock, bool forceMustExecute)`
- `void AudioEngine::handleAsyncUpdate()`（bridge 経路の最終 gate 観点）

## 2.3 `src/audioengine/AudioEngine.h`

- `acceptsRuntimePublication()` 宣言周辺コメント
- 必要なら T0-01 専用 helper 宣言（例: post-check helper）

---

## 3. 現状の観測ポイント

- 入口 gate は既に多くの関数で実施済み（例: `prepareCommit`, `executeCommit`, `submitRebuildIntent`）
- ただし、`acceptsRuntimePublication()` 通過後に queue/CAS 進行する経路があり、shutdown 競合時の窓が残る
- `appendPublicationIntentForCommitSlot()` は `tail` 取得後に CAS ループへ入るため、直前の再判定位置が重要

---

## 4. 差分方針（最小変更）

## 4.1 方針A: Admission 契約の明示（振る舞い不変）

### 変更内容

- `acceptsRuntimePublication()` のコメントを「判定の有効期間は call-scope 内のみ」に更新
- `lifecycleState` / `shutdownPhase` / `shutdownRuntime_` の参照順と memory order 意味論をコメントで固定

### 方針Bの目的

- 実装者が「1回判定すれば後段で安全」と誤解しないようにする

---

## 4.2 方針B: enqueue/publish 直前 double-check の追加

### 変更内容（候補）

- `appendPublicationIntentForCommitSlot()` の CAS ループ突入前に `acceptsRuntimePublication()` を再評価
- `prepareCommit()` / `executeCommit()` は既に入口 gate があるため、必要なら軽量再判定を追加
- `submitRebuildIntent()` / `requestRebuild(...)` は queue 化直前（`pendingTask` 反映直前）で再判定

### 失敗時処理

- `publicationRejectCount_` or rebuild telemetry を increment
- 生成済み `intent/newDSP` は `retireDSP()` へ戻してリーク回避

### 目的

- TOCTOU（判定→queue反映の間）で shutdown へ遷移したケースを握りつぶさず reject 側に倒す

---

## 4.3 方針C: shutdown 後流入の明示 reject 一貫化

### 方針Cの変更内容

- shutdown 系 reject 理由を telemetry reason として統一（既存 `ShutdownInProgress` を優先利用）
- `handleAsyncUpdate()` 経由の non-MT bridge でも同一 reject 規則を適用

### 方針Cの目的

- 経路ごとの判定差異を減らし、「どの入口でも同じ拒否規則」を成立させる

---

## 5. 非対象（このPRでやらない）

- `waitForDrain(timeout)` の導入/調整（T0-03）
- drain SSOT 集約の実装拡張（T0-02）
- saturation hysteresis 実装（T0-04）
- crossfade/latency 系の純化（Tier2）

---

## 6. 実装手順（推奨順）

1. `acceptsRuntimePublication()` コメント契約を更新（動作変更なし）
2. `appendPublicationIntentForCommitSlot()` 直前再判定を追加
3. `submitRebuildIntent()` の queue化直前再判定を追加
4. `requestRebuild(sr,bs)` の `pendingTask` 書込み直前再判定を追加
5. reject telemetry / counter の整合を確認
6. 最後に `prepareCommit` / `executeCommit` の冗長判定有無を整理（最小差分優先）

---

## 7. 受入条件（T0-01 固有）

1. shutdown 遷移後、publication intent の新規連結が発生しない
2. shutdown 遷移後、rebuild pending task が新規投入されない
3. `queue.push -> rejectLater` 型の後段 reject パターンを増やさない
4. reject 時に `newDSP` / `intent` がリークしない
5. 既存 commit/rebuild 成功パスの動作を崩さない

---

## 8. 検証手順

## 8.1 静的検証

- `acceptsRuntimePublication()` 呼び出し位置を再検索し、queue反映直前の未ガード箇所を確認
- 追加した reject path の解放処理（`retireDSP` / deferred delete）を確認

## 8.2 ビルド検証

1. `Debug Build (cmd env retry)`
2. `Release Build (cmd env retry)`

両方成功を必須とする。

## 8.3 ルール検証

- `Strict Atomic Dot-Call Scan`
- RT path に lock/alloc/blocking を導入していないことを確認

## 8.4 挙動検証（手動シナリオ）

- Scenario A: rebuild 連打中に shutdown 開始
  - 期待: 新規 queue 化が reject 側へ倒れる
- Scenario B: async bridge (`handleAsyncUpdate`) 実行中に shutdown 開始
  - 期待: `ShutdownInProgress` 理由で suppress/return
- Scenario C: 通常運転時の rebuild
  - 期待: 既存成功パス維持

---

## 9. ロールバック計画

- 段階 rollback を可能にするため、差分を以下3コミットに分割
  1. コメント/契約明文化（無害）
  2. commit 経路 double-check
  3. rebuild 経路 double-check
- 問題時は 2/3 を個別 revert 可能
- 1（契約明文化）は原則残す

---

## 10. PR本文テンプレート（T0-01用）

```text
## Purpose
Admission Gate の atomic 契約を明示し、判定後の TOCTOU 窓を封止する。

## Scope (Files)
- src/audioengine/AudioEngine.Commit.cpp
- src/audioengine/AudioEngine.RebuildDispatch.cpp
- src/audioengine/AudioEngine.h

## Acceptance Criteria
- shutdown 遷移後に publication/rebuild の新規 queue 化が発生しない
- reject path でリークが発生しない
- Debug/Release build green

## Verification
- Strict Atomic Dot-Call Scan
- Debug Build (cmd env retry)
- Release Build (cmd env retry)

## Rollback Plan
- commit 経路 / rebuild 経路を独立 revert 可能
```

---

## 11. Go/No-Go

- **Go**: 受入条件 1〜5 全充足 + ビルド/スキャン green
- **Conditional Go**: telemetry 文言差のみ残る（機能影響なし）
- **No-Go**: shutdown 後の queue 化が1件でも再現、またはリーク疑い
