# ConvoPeq ISR Bridge Runtime v5.5.1 実装タスク表（1コミット=1責務）

- Date: 2026-05-28
- Source of truth:
  - `doc/work2/ConvoPeq_ISR_Bridge_Runtime_v5.5_詳細設計_FINAL_2026-05-28.md`
  - `doc/work2/ISR_Bridge_Runtime_v5.5_FINAL_FREEZE_2026-05-28.md`
  - `doc/work2/ConvoPeq_ISR_Bridge_Runtime_v5.5_AI_実装統制規約_FINAL_2026-05-28_LINT.md`
  - `doc/work2/ConvoPeq_ISR_Bridge_Runtime_v5.5.1_文書追補パッチ_2026-05-28.md`
- Rule: **1コミット = 1責務（混在禁止）**
- Target: DAW-safe hardening（deterministic shutdown / rebuild determinism / RT boundedness）

---

## 0. 運用ルール（全タスク共通）

- 各コミットは以下を満たすこと。
  1. 変更責務が単一
  2. 旧経路削除まで同コミットで閉じる（移設のみ禁止）
  3. 失敗時ロールバック点が明示されている
  4. Debug build または Release build のどちらかが green
  5. `Strict Atomic Dot-Call Scan` を通す

- 禁止事項（再掲）
  - RT path で lock/alloc/wait/I/O/logger formatting
  - publish 後 mutable topology mutation
  - runtime pointer 由来の worker 判断
  - 大規模 rewrite

---

## 1. フェーズ別実装タスク（固定順）

## Phase 1: Unified Admission Gate

### P1-C01: acceptsRuntimePublication API導入

- 責務: Admission判定APIを単一路線に固定
- 主対象: `src/audioengine/AudioEngine.h`（必要なら `.cpp`）
- 実装:
  - `bool acceptsRuntimePublication() const noexcept;` 追加
  - `lifecycleState/shutdownPhase/shutdownRuntime_` に基づく判定実装
- 完了条件:
  - Running 以外 false
- ロールバック点:
  - API追加のみなのでヘッダ差分を丸ごとrevert

### P1-C02: Gate適用（requestRebuild / appendPublicationIntent）

- 責務: publication入口の早期reject化
- 主対象: `AudioEngine.RebuildDispatch.cpp` / `AudioEngine.Commit.cpp`
- 実装:
  - `requestRebuild` 先頭に gate
  - `appendPublicationIntent` 先頭に gate
- 完了条件:
  - shutdown中に intent が追加されない
- ロールバック点:
  - 各関数先頭の guard ブロック単位でrevert

### P1-C03: Gate適用（prepareCommit / executeCommit）

- 責務: commit段階の封止
- 主対象: `AudioEngine.Commit.cpp` / `RuntimePublicationCoordinator` 実装
- 実装:
  - `prepareCommit` 先頭に gate
  - `executeCommit` 先頭に gate
- 完了条件:
  - gate false 時は副作用ゼロreturn
- ロールバック点:
  - commit関数内 guard 追加差分のみrevert

### P1-C04: 命名統一（appendPublicationIntent 正規名）

- 責務: `enqueuePublication` alias運用の整理
- 主対象: `AudioEngine.*` / coordinator関連
- 実装:
  - 呼称を `appendPublicationIntent` に統一
  - 旧称はコメント上 legacy alias としてのみ保持
- 完了条件:
  - 実コード上の主要呼称が統一
- ロールバック点:
  - renameコミット単独でrevert可

---

## Phase 2: RuntimeBuildSnapshot 完全移行

### P2-C01: RuntimeBuildFingerprint 構造固定

- 責務: fingerprint要素の固定
- 主対象: snapshot定義ヘッダ（`src/core/` もしくは `audioengine`）
- 実装:
  - `fingerprintVersion/irIdentityHash/convolutionConfigHash/dspParameterHash/sampleRate/blockSize` 固定
- 完了条件:
  - 構造体が設計と一致
- ロールバック点:
  - struct定義差分のみrevert

### P2-C02: capture→finalize→seal→handoff パイプ化

- 責務: snapshot lifecycle の順序固定
- 主対象: rebuild dispatch + worker handoff 実装
- 実装:
  - capture, finalize, seal, handoff を分離関数化（最小）
- 完了条件:
  - worker入力が sealed snapshot のみ
- ロールバック点:
  - 新規関数分離コミット単位でrevert

### P2-C03: finalize determinism 禁止要素の遮断

- 責務: finalize純粋化
- 主対象: finalize実装箇所
- 実装:
  - wall clock / thread-order / allocation order / pointer identity 依存除去
- 完了条件:
  - finalizeがsemantic inputのみ参照
- ロールバック点:
  - finalize内部変更のみrevert

### P2-C04: version mismatch reuse禁止導入

- 責務: fingerprintVersion 不一致時の再利用封止
- 主対象: reuse判定分岐
- 実装:
  - mismatch時 `reuse=false`（再構築へ）
- 完了条件:
  - 部分一致再利用が発生しない
- ロールバック点:
  - 判定分岐の差分のみrevert

### P2-C05: worker runtime pointer参照の物理ゼロ化

- 責務: Rule-4 完全準拠
- 主対象: rebuild worker 実装
- 実装:
  - active/fading runtime 直参照削除
- 完了条件:
  - worker は snapshot-only
- ロールバック点:
  - worker関連差分のみrevert

---

## Phase 3: Retire Backpressure Hardening

### P3-C01: HWM/LWM 定数固定（3072/1024）

- 責務: saturation閾値固定
- 主対象: retire/backpressure管理コード
- 実装:
  - HWM=3072, LWM=1024
- 完了条件:
  - しきい値が一意
- ロールバック点:
  - 定数変更差分のみrevert

### P3-C02: scale clamp 全適用

- 責務: 4scale の clamp 統一
- 主対象: scale算出関数
- 実装:
  - $0.75 \le scale \le 1.50$ を全scaleに適用
- 完了条件:
  - clamp漏れゼロ
- ロールバック点:
  - clamp適用差分のみrevert

### P3-C03: memoryPressureScale 入力源制限

- 責務: runtime-local metrics only
- 主対象: pressure計算
- 実装:
  - 許可7項目のみに限定
  - OS/global/external 依存を除去
- 完了条件:
  - 禁止ソース参照ゼロ
- ロールバック点:
  - pressure関数の差分revert

### P3-C04: saturation 単調安定化ルール適用

- 責務: 緩和禁止・強化方向限定
- 主対象: saturation state 遷移
- 実装:
  - saturation中に HWM/LWM 下降禁止
- 完了条件:
  - stabilization direction only
- ロールバック点:
  - 遷移分岐差分のみrevert

### P3-C05: stepwise recovery 実装（128刻み）

- 責務: conservative recovery
- 主対象: recovery関数
- 実装:
  - `queueDepth < LWM` で解除判定
  - HWM/LWM を段階的に緩和
- 完了条件:
  - oscillationを誘発しない
- ロールバック点:
  - recoveryロジック差分revert

### P3-C06: telemetry first 実装

- 責務: 必須観測項目の固定
- 主対象: 非RT telemetry collector
- 実装:
  - `retireQueueDepth/fallbackQueueDepth/quarantineResident/publicationBacklog/rebuildBacklog/saturationEnterCount/saturationExitCount/publicationRejectCount/rebuildCollapseCount/reclaimLatency`
- 完了条件:
  - 必須10項目が収集可能
- ロールバック点:
  - telemetry差分単独revert

---

## Phase 4: DSP Execution State 分離

### P4-C01: execution-local mutable state 境界の固定

- 責務: Rule-3 / Rule-19 境界のコード化
- 主対象: DSP state 保持構造
- 実装:
  - execution-local のみ mutable 許可
  - publication/state authority から分離
- 完了条件:
  - shared mutable authority が混入しない
- ロールバック点:
  - state保持構造変更のみrevert

### P4-C02: runtime visibility object と execution state の分離

- 責務: RuntimeGraph責務の純化
- 主対象: RuntimeGraph/RuntimeState 関連
- 実装:
  - visibility object と execution authority を分離
- 完了条件:
  - RuntimeGraphに実行可変状態を持たせない
- ロールバック点:
  - 分離差分のみrevert

### P4-C03: CI fail-stop ルール接続

- 責務: Rule-2/4/11/23 を自動検査化
- 主対象: `.github/scripts` / CI wiring
- 実装:
  - RT禁則 / worker runtime参照 / resurrection を fail-stop
- 完了条件:
  - CIで違反をブロック
- ロールバック点:
  - CI設定コミット単位でrevert

---

## Phase 5: Crossfade Authority Isolation

### P5-C01: CrossfadePreparedState の準備責務を Message Thread に固定

- 責務: prepare-activate 分離
- 主対象: `AudioEngine.Commit.cpp` / `AudioEngine.h`
- 実装:
  - `preparedGeneration/fadeSec/startDelayBlocks/old-new delay/useDryAsOld` を prepare段で固定
- 完了条件:
  - Audio Thread が初期化を行わない
- ロールバック点:
  - crossfade準備導入差分のみrevert

### P5-C02: Audio Thread activate-only 化

- 責務: RT path 初期化禁止
- 主対象: `AudioEngine.Processing.*`
- 実装:
  - activate + progression のみ
  - reset/init API 呼び出し排除
- 完了条件:
  - RT path で crossfade 初期化ゼロ
- ロールバック点:
  - RT helper差分revert

### P5-C03: cross-runtime mutable progression 排除

- 責務: shared mutable authority ゼロ化
- 主対象: crossfade補助状態
- 実装:
  - runtime間共有可変進行を削除
- 完了条件:
  - progression は execution-local のみ
- ロールバック点:
  - progression管理差分のみrevert

---

## 2. 横断タスク（フェーズ横断・独立コミット）

### X-C01: Rebuild Collapse deterministic 実装

- 責務: latest-generation-wins + safe-to-collapse 条件固定
- 完了条件:
  - must-execute rebuild が collapse されない
  - cross-class collapse 不可
- ロールバック点:
  - collapse判定差分revert

### X-C02: Drained strict + resurrection禁止 実装

- 責務: shutdown完了定義と復活禁止
- 完了条件:
  - drained 5条件全成立でのみ完了
  - drained後 enqueue/retry/relaunch 不可
- ロールバック点:
  - drained判定差分revert

### X-C03: DoD 自動チェック雛形追加

- 責務: DoD 10項目の検証導線を固定
- 完了条件:
  - 静的/挙動検証項目がチェックリスト化
- ロールバック点:
  - テスト雛形コミットrevert

---

## 3. 各コミットの記録テンプレート（必須）

- `YYYY-MM-DD | Phase Px-Cyy | files: <changed files> | removed-legacy: <old path> | verify: scan=<OK/NG> build=<OK/NG> tests=<OK/NG> | rollback: <revert target> | risk: <open risk> | next: <next commit>`

記入例:

- `2026-05-28 | Phase P1-C02 | files: AudioEngine.RebuildDispatch.cpp,AudioEngine.Commit.cpp | removed-legacy: ungated append/request path | verify: scan=OK build=OK tests=OK | rollback: guard block revert | risk: none | next: P1-C03`

---

## 4. Go / No-Go 判定

### Go 条件

- フェーズ順を守って全コミットを実施
- 各コミットでロールバック可能点が維持
- DoD 10項目を全満足

### No-Go 条件

- 1コミット内で複数責務が混在
- RT禁則違反
- drained/resurrection 規約違反
- gate/collapse/backpressure のいずれかが未固定
