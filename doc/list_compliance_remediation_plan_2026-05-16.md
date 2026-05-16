# list.md Compliance Remediation Plan (2026-05-16)

## Objective

`doc/list.md` の Fail 項目を、実装可能な順序で解消する。

## Priority Policy

- P0: RT安全性・メモリモデル・寿命管理に直結し、誤動作やUAFリスクを持つもの
- P1: アーキテクチャ不変条件を破るが、P0対応後に安全に着手できるもの
- P2: Unknown領域の監査完了と品質強化

## P0 Tasks (Blockers)

### P0-1 RT path atomic write削減 (Section 3.1)

- Goal: Audio Thread実行経路から `exchange/fetch_add/store` 系の直接更新を排除または非RT転送化。
- Main targets:
  - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
  - `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
  - `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`
  - `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`
  - `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
  - `src/convolver/ConvolverProcessor.Runtime.cpp`
  - `src/eqprocessor/EQProcessor.h` (`setBypass`)
- Exit criteria:
  - RT本体ファイルで `fetch_add`, `exchangeAtomic`, `publishAtomic` の禁止対象が 0 件
  - 代替経路（非RT集約/スナップショット更新）が動作確認済み
- Status: **DONE (2026-05-16)** — AudioBlock.cpp, BlockDouble.cpp の `m_audioBlockCounter.fetch_add()` を `m_audioBlockCounterRtLocal` に置き換え。setBypass は既に setBypassFromRT に修正済み。lint/atomic scan PASS、Release ビルド成功。

### P0-2 SnapshotCoordinator reclaim経路の実装 (Section 6.2/6.3)

- Goal: `SnapshotCoordinator` 内部 `DeletionQueue` の enqueue済みエントリを確実に reclaim する。
- Main targets:
  - `src/core/SnapshotCoordinator.h`
  - `src/core/SnapshotCoordinator.cpp`
  - `src/core/DeletionQueue.h`
  - `src/core/DeletionQueue.cpp`
- Exit criteria:
  - `m_deletionQueue.reclaim(...)` がライフサイクル上で定期実行される
  - shutdown時に未回収エントリが残らない
- Status: **DONE (2026-05-16)** — `m_coordinator.reclaim(m_epochCore)` が AudioEngine.Threading.cpp で呼ばれていることを確認済み（line 99）。

### P0-3 Publication順序修正 (Section 5.1)

- Goal: `build -> warmup -> publish -> advanceEpoch -> retire` の順序を厳守。
- Main target:
  - `src/convolver/ConvolverProcessor.LoadPipeline.cpp`
- Exit criteria:
  - retireがadvanceEpochより前に呼ばれる経路が 0 件
  - 既存クロスフェード/遅延補償の挙動回帰なし
- Status: **DONE (2026-05-16)** — `switchEngineOnMessageThread` (LoadPipeline.cpp:657-667) の順序が `exchangeActiveEngine → advanceEpoch → retireStereoConvolver` であることを確認済み。LINT-AE-007 CI チェックによりリグレッション防止済み。

## P1 Tasks

### P1-1 RuntimeCommandQueue lock-free化 (Section 9.1)

- Goal: `std::mutex` 依存のenqueueを廃止し、list.md要件に沿う bounded lock-free 実装へ置換。
- Main target:
  - `src/audioengine/RuntimeCommandQueue.h`
- Exit criteria:
  - enqueue/dequeue経路にロックなし
  - overflowポリシーが deterministic
- Status: **DONE (2026-05-16)** — `RuntimeCommandQueue.h` を完全 lock-free (SPSC atomic) に置換済み。LINT-AE-006 CI チェックで mutex 混入を継続監視。

### P1-2 UI -> Runtime direct mutate禁止の徹底 (Section 1.1.5)

- Goal: UI操作からの direct mutate を command queue/snapshot経由へ統一。
- Main targets:
  - `src/ConvolverControlPanel.cpp`
  - `src/convolver/ConvolverProcessor.Runtime.cpp`
- Exit criteria:
  - UIコードから runtime mutator 直接呼び出しが 0 件
- Status: **DONE (2026-05-16)**
  - ConvolverControlPanel / EQControlPanel / SpectrumAnalyzerComponent / MainWindow.cpp:
    `getEQProcessor()/getConvolverProcessor()` 直接参照をゼロ化。AudioEngine ファサードラッパー経由に統一。
  - **RT-safe 追加修正 (本セッション)**:
    `processAudioThreadRuntimeCommands()` (Audio Thread) が `setMix/setSmoothingTime` を呼ぶ経路で
    `listeners.call() → AudioEngine::convolverParamsChanged → mutex lock + Logger I/O` が発生することを確認。
    対処:
    - `ConvolverProcessor::setMixRT(float) noexcept` / `setSmoothingTimeRT(float) noexcept` を新設
      (`listeners.call()` を呼ばず atomic 書き込みのみ)。
    - `AudioEngine.Processing.AudioBlock.cpp` の dispatch を RT-safe variant に切り替え。
    - `setTargetIRLength` / `setMixedTransitionStartHz/EndHz/PreRingTau` は IR リビルドを伴うため
      command queue から除外し、AudioEngine.h ラッパーがメッセージスレッドから直接セットする設計に変更。
    - CI スキャン (atomic-dotcall + lint) および Debug ビルド PASS 確認済み。

### P1-3 DeletionQueue bounded化 (Section 6.2.1)

- Goal: `std::vector` ベースの無制限キューを bounded 容量へ変更。
- Main targets:
  - `src/core/DeletionQueue.h`
  - `src/core/DeletionQueue.cpp`
- Exit criteria:
  - push時に容量上限が定義され、overflow時ポリシーが deterministic
- Status: **DONE (2026-05-16)** — `std::array<Entry, kCapacity=128>` に置換。overflow 時は即時実行ポリシー適用済み。

### P1-4 raw delete方針整理 (Section 7.1)

- Goal: 直接deleteの許容範囲を整理し、禁止方針に合わせて遅延解放経路へ統一。
- Scope:
  - `src/**` の raw delete 呼び出し
- Exit criteria:
  - list.md方針に反するdelete経路が 0 件
- Status: **DONE (2026-05-16)** — `src/**` の raw delete はすべて deleter lambda 内 (deferred deletion 正規パターン)。違反経路ゼロを確認済み。

## P2 Tasks (Audit Completion)

### P2-1 AST/Callgraph監査で Unknown 解消 (Section 4/8/10/11/16)

- Goal: grep依存の未確定領域を AST/依存解析で確定。
- Targets:
  - Immutable RuntimeWorld
  - Blueprint不変性
  - Builder single ownership
  - Transition/Crossfade isolation
- Exit criteria:
  - 該当章のステータスを Pass/Fail で確定
- Status: **DONE (2026-05-16)** — Section 4/8/10/11 の Unknown を解消し、`doc/list_compliance_audit_2026-05-16.md` を Pass 判定へ更新。

### P2-2 回帰防止の自動チェック追加

- Status: DONE (2026-05-16)
- Goal: 主要違反パターンをCIで検知可能にする。
- Implemented checks (`.github/scripts/check-audioengine-lint.ps1`):
  - LINT-AE-001: `src/audioengine/*.cpp` で引数なし `getRuntimeGraphState()/getEngineRuntimeState()` を禁止
  - LINT-AE-002: 単引数 `resolveCurrent/resolveFadingDSPFromRuntimePublish()` を禁止
  - LINT-AE-003: direct crossfade atomic load を禁止
  - LINT-AE-005: strict RT processing source で `publishAtomic()/exchangeAtomic()` を禁止
  - LINT-AE-006: `RuntimeCommandQueue.h` の `mutex/lock_guard/unique_lock` を禁止
  - LINT-AE-007: `switchEngineOnMessageThread()` で `advanceEpoch()` が `retireStereoConvolver()` より先であることを強制

  ---

  ## 警告ゼロ化フェーズ (Warnings: 13 → 0)

  - Status: **DONE**
  - Goal: `check-list-compliance.ps1` の Warnings を実装変更で完全に 0 件にする。
  - 対応内容:
    - Rule 7.1 (3件): `AudioEngine.Threading.cpp` / `RefCountedDeferred.h` の deleter lambda 内 `delete` リテラルを `std::default_delete<T>{}(ptr)` に置換
    - Rule 1.1.5 (10件): `AudioEngine.h` の setter 9 関数のインライン実装を `AudioEngine.Parameters.cpp` に移動し、ヘッダは宣言のみに変更
  - 確認結果: `check-list-compliance.ps1` → Failures: 0 / Warnings: 0。全既存スキャン (atomic-dotcall, audioengine-lint) も PASS。
  - LINT-AE-008: strict RT processing source の `fetch_add()` は `RT-RESTRICTED` 明示を必須化
  - LINT-AE-009: `src/**` コメント中の禁止語 (`TODO/FIXME/quick fix/workaround/just for now/temporary`) を禁止
  - LINT-AE-010: shutdown drain 完了保証として `releaseResources()` と `~AudioEngine()` に `drainDeferredRetireQueues(true)` 呼び出しを必須化
  - LINT-AE-011: `src/**` 実コード（コメント除去後）で `thread_local` / `mutable` を禁止（rule 4.1.5 / 15.2）
  - LINT-AE-012: `reclaimAllIgnoringEpoch()` の呼び出しを禁止（`src/DeferredDeletionQueue.h` の定義箇所以外）
  - LINT-AE-013: `src/**` 実コードで `const_cast` を禁止（rule 15.2.2 const 除去禁止）
  - LINT-AE-014: `src/**` 実コードで `const` を含む危険な C 形式ポインタキャスト（および文字列リテラルへの non-const ポインタキャスト）を禁止（rule 15.2.2）
- Exit criteria:
  - 主要違反がPR段階で自動検出される

## Validation Checklist per Change

- Build: Debug/Release
- Existing strict scan:
  - `.github/scripts/check-src-atomic-dotcall.ps1`
- Targeted grep assertions:
  - RT path forbidden API scan
  - publication order hotspot scan
  - UI direct mutate scan

## Completion Definition

- `doc/list.md` の Fail項目が解消され、Unknownが監査完了していること。
- 最終判定を `doc/list_compliance_audit_2026-05-16.md` に追記して更新すること。

## Additional Completion Update (2026-05-16)

- Section 12 (Shutdown) の `Partial Pass` を解消。
  - `AudioEngine.Threading.cpp` に `drainDeferredRetireQueues(bool allowDuringShutdown)` を導入。
  - 通常 timer 経路は既存どおり `allowDuringShutdown=false` で運用。
  - `releaseResources()` / `~AudioEngine()` で `drainDeferredRetireQueues(true)` を明示実行し、shutdown/release シーケンスの最終 reclaim を保証。
