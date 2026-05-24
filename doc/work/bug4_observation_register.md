# bug4 観察項目レジスタ（P3）

対象: `doc/work/bug4_action_plan.md` の P3-7

このファイルは、現時点で「不成立 / 保留」と判断した項目を、
将来の変更で再評価できるようにするための運用台帳です。

---

## 運用ルール

- 対象周辺を変更した PR では、該当項目の「再確認結果」を 1 行追記する
- 再現した場合は、`状態` を `Open` に変更し、再現条件と影響を更新する
- 3 回連続で問題なしの場合は `Dormant` へ移行して監視頻度を下げる

---

## 観察項目

| ID | 項目 | 監視トリガー | 状態 | 最終確認 | メモ |
| --- | --- | --- | --- | --- | --- |
| P3-1 | `SafeStateSwapper::tryReclaim` 複数スレッド化懸念 | reclaim 呼び出し元追加、`DeferredFreeThread` 変更 | Monitoring | 2026-05-24 | 再評価: 呼び出しは `DeferredFreeThread::run` と shutdown後 drain 経路。同時実行は未確認、debug thread-affinity assert 継続監視 |
| P3-2 | `AllpassDesigner::applyAllpassToIR` の引数/契約齟齬 | `applyAllpassToIR` 呼び出し追加、MixedPhase 設計変更 | Monitoring | 2026-05-24 | 再評価: `src/**` 参照検索で呼び出し0件（宣言/定義のみ） |
| P3-3 | `NoiseShaperLearner` の `candidatePopulation` 同期 | 学習並列度変更、評価ワーカー構造変更 | Monitoring | 2026-05-24 | 再評価: `sample→並列評価(read-only)→全ワーカー完了待ち→update` を確認。現時点で顕在競合なし |
| P3-4 | `LinearRamp::skip` | ランプ実装改修、クロスフェード進行方式変更 | Monitoring | 2026-05-24 | 再評価: `numSamples>=remaining` で `target` 収束、使用側も `start→skip→end` パターンで整合 |
| P3-5 | `ConvolverProcessor::prepareToPlay` 例外安全性 | prepare/rebuild 系メモリ確保経路変更 | Monitoring | 2026-05-24 | 再評価: NUC 再初期化は `std::bad_alloc` 捕捉 + 失敗時「既存 engine 維持」ログ経路を確認 |
| P3-6 | `dspCrossfadeArmed_RT` 初期化順序 | 初期化順序変更、`initialize` 呼び出し順変更 | Monitoring | 2026-05-24 | 再評価: 宣言時 `false` + `initialize/prepareToPlay` で明示 `false` 再設定、pending無し時は `armCrossfadeIfPending` で解除 |

---

## 再確認ログ

- 2026-05-24: P0/P1/P2 実施後に再確認、P3 項目は新規再現なし。
- 2026-05-24: P3-1 を再評価。`tryReclaim` 呼び出しは `DeferredFreeThread::run` / `drainAllRetired` / `ConvolverProcessor::releaseResources` 末尾の drain に存在。`shutdownAndDrain` は join 後に drain するため同時並行の reclaim は現時点で未再現。`SafeStateSwapper` の debug thread-affinity assert（`reclaimThreadIdDebug`）で誤用検知を継続。
- 2026-05-24: P3-2 を再評価。`applyAllpassToIR(` は `AllpassDesigner.h/.cpp` の宣言/定義のみで、実呼び出しは未検出。`ConvolverProcessor.MixedPhase.cpp` は `design*` + `computeResponse` 経路を使用しており、現時点で齟齬は顕在化せず。
- 2026-05-24: P3-3 を再評価。`NoiseShaperLearner::workerThreadMain/evaluatePopulation/runEvaluationJobsForWorker` で、`optimizer.sample(candidatePopulation)` 後に評価ワーカーが read-only 参照し、`completedAuxEvaluationWorkers` 到達後に `optimizer.update(...)` が実行される順序を確認。現時点では同期不備は未再現。
- 2026-05-24: P3-5 を再評価。`ConvolverProcessor::prepareToPlay`（`src/convolver/ConvolverProcessor.Lifecycle.cpp`）で、再初期化失敗時に `std::bad_alloc` を捕捉し、旧エンジンを交換せず維持する分岐（"Keeping existing engine."）を確認。現時点で例外起因の即時破綻経路は未再現。
- 2026-05-24: P3-6 を再評価。`AudioEngine.h` で `dspCrossfadeArmed_RT` は宣言時 `false` 初期化、`AudioEngine::initialize` と `AudioEngine::prepareToPlay` で関連RT値を再初期化し、`armCrossfadeIfPending` でも pending 無しなら `dspCrossfadeArmed_RT=false` に戻す経路を確認。現時点で初期化順序起因の再現なし。
- 2026-05-24: P3-4 を再評価。`DspNumericPolicy.h` の `LinearRamp::skip` は `numSamples<=0 || remaining<=0` の早期return、`numSamples>=remaining` で `current=target` へ収束、以外は `current += step*numSamples` で進行する実装。`EQProcessor.Processing.cpp` では `startGain -> skip(numSamples) -> endGain` でランプ区間を算出しており、現時点で不整合/発散は未再現。

---

## 新規変更時 監視運用ログ

### 2026-05-24 / サイクル #1

- 入力差分: `git status --short` / `git diff --name-only` で変更ファイルを収集（コード変更多数）
- トリガー判定:
  - 発火: P3-1（`src/SafeStateSwapper.h`, `src/convolver/ConvolverProcessor.Lifecycle.cpp`）
  - 発火: P3-4（`src/eqprocessor/**`, `src/DspNumericPolicy.h` 参照経路）
  - 発火: P3-5（`src/convolver/ConvolverProcessor.Lifecycle.cpp`, `src/ConvolverProcessor.h`）
  - 発火: P3-6（`src/audioengine/AudioEngine.h`, `AudioEngine.Init.cpp`, `AudioEngine.Processing.AudioBlock.cpp`）
  - 非発火: P3-2（`applyAllpassToIR` 呼び出し追加なし）、P3-3（`NoiseShaperLearner` 変更なし）
- 実施結果:
  - 発火項目はすべて再評価済み（本台帳の再確認ログ参照）
  - 新規 `Open` 化なし（全項目 `Monitoring` 維持）
- 次回運用:
  - `NoiseShaperLearner.*` 変更時は P3-3 を優先再評価
  - `AllpassDesigner` / `MixedPhase` の呼び出し経路変更時は P3-2 を優先再評価
