# Practical Stable ISR Bridge Runtime 実装優先タスク（具体タスク化版）

- 作成日: 2026-05-30
- ベース資料: `doc/work7/practical_stable_isr_bridge_runtime_migration_review_2026-05-30.md`
- 目的: Bridge Runtime を「実運用で破綻しにくい」水準まで段階的に到達させるための、実行可能なタスク計画を定義する。

---

## 1. 実行方針

- 最優先は **semantic divergence（意味論ドリフト）低減**。
- 優先順位は「入口の数」ではなく **Invariant の一貫性** と **観測面の単一化** に置く。
- Audio Thread 制約（非ブロッキング・非割当・非ロック）を崩さない変更のみ許容する。
- 1タスクは「対象ファイル」「変更点」「完了条件（DoD）」「検証方法」を必須項目とする。

### Authority View Definition（実装前提）

- **定義**: Authority View は、Audio callback が 1 回だけ取得する観測スコープ。
- **許容**: 内部で複数フィールドを保持してよい。
- **禁止**: callback 実行中の追加 authority 再取得（例: graph/snapshot/crossfade を別ルートで再フェッチ）。
- **注意**: 単なる集約構造体（例: `graph/snapshot/fade` を詰めるだけ）を作っても、再取得が残るなら Observe Path Collapse ではない。

---

## 2. マイルストーン（M1〜M6）

- **M1（最優先）**: Observe Path Collapse
- **M2**: Snapshot Semantic Unification（※ T2-1 は M1 の前提として先行）
- **M3**: Publication Invariant Centralization
- **M4**: Retire Governance Validation
- **M5**: RuntimeGraph → RuntimeWorld 依存整理（次段）
- **M6**: Contract/Verifier 可視化（監査性向上）

---

## 3. 具体タスク一覧

## M1: Observe Path Collapse（Priority 1）

### T1-1 Audio callback 観測ソース一本化（float path）

- 対象: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- 現状課題:
  - `runtimeGraph->activeNode`
  - `getRuntimeSnapshot(runtimeReadView)`
  - `consumeCrossfadePreparedSnapshot()`
  が同一 callback 内で併存。
- 実施内容:
  - callback 冒頭で単一の「authority view」を構築し、以降の参照を統一。
  - 補助関数への引き渡しも同 view に限定。
- DoD:
  - callback 本体で authority 取得経路が実質1つに収束。
  - 旧経路への直接再参照が消滅。
  - Audio callback 配下で、authority 取得API以外から `RuntimeGraph` / `RuntimeSnapshot` / `CrossfadePreparedSnapshot` へ直接アクセスするコードが存在しない。
- 検証:
  - 影響ファイルの静的確認（直接参照残存チェック）。
  - Debug ビルド成功。
  - 構造制約チェック（`getNextAudioBlock` 配下での再取得禁止パターンが 0 件）。

### T1-2 Audio callback 観測ソース一本化（double path）

- 対象: `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
- 実施内容/DoD/検証:
  - T1-1 と同様に double path へ適用（再取得禁止の構造制約を含む）。

### T1-3 `processWithSnapshot` の観測面統一

- 対象: `src/audioengine/AudioEngine.Processing.Snapshot.cpp`
- 現状課題:
  - `RuntimeGraph` と `GlobalSnapshot` の二面参照。
- 実施内容:
  - `processWithSnapshot` 入力契約を見直し、関数内で追加 authority を解決しない方針へ寄せる。
- DoD:
  - 関数内の authority 逆引き（`readAudioRuntimeView()` 依存）を削減または排除。
- 検証:
  - 呼び出し元/受け側で契約が整合していることを確認。

---

## M2: Snapshot Semantic Unification（Priority 2）

### T2-1 Bridge 用 semantic source の定義文書化

- 対象: `doc/work7/`（本計画の補助文書として追加）
- 実施内容:
  - 「現在時点でどの値をどの authority から読むか」を表で固定。
  - `active/fading/crossfade` の取得元を明記。
- DoD:
  - 実装と1対1対応する source-of-truth 表を作成。
- 検証:
  - コードレビューで差分解釈が一致すること。

### T2-2 crossfade prepared snapshot の責務再定義

- 対象: `AudioEngine` の crossfade 参照/更新系
- 実施内容:
  - prepared snapshot を「派生キャッシュ」にするか「正規authority」にするかを決定し、参照側を統一。
- DoD:
  - 読み手が authority を迷わない構造になる。
- 検証:
  - callback 内の crossfade 参照が単一路線であることを確認。

---

## M3: Publication Invariant Centralization（Priority 3）

### PublicationInvariant v1（先行固定）

- I-1: world frozen（publish 実行前に world が凍結済みであること）
- I-2: world sealed（公開対象 state が封印済みであること）
- I-3: closure valid（参照クロージャが有効で dangling でないこと）
- I-4: publication state legal（state machine 上で遷移が合法であること）
- I-5: fade transition legal（fade/policy 組合せが合法であること）
- I-6: retire registration complete（retire 登録が完了していること）

> 注: `T3-1` の共通チェック関数は上記 I-1〜I-6 を最小セットとして必ず評価する。

### T3-1 publish 前後 invariant チェックの共通化

- 対象:
  - `src/audioengine/AudioEngine.Commit.cpp`
  - `src/audioengine/AudioEngine.Timer.cpp`
  - `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`
  - `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
- 実施内容:
  - `makeRuntimePublicationCoordinator().publishState(...)` 呼び出し前後で共通チェック関数を必ず通す。
  - チェック内容は `PublicationInvariant v1`（I-1〜I-6）を準拠セットとして固定する。
- DoD:
  - 4経路すべてが共通 invariant 経由で publish される。
- 検証:
  - 経路漏れがないことを静的検索で確認。

### T3-2 publish エラーパス方針の統一

- 対象: 上記 publish 呼び出し群
- 実施内容:
  - invariant 不一致時の挙動（安全側フォールバック）を明文化して統一。
- DoD:
  - 経路ごとに異なる失敗動作が残らない。
- 検証:
  - 失敗時ログ/診断タグを統一し、追跡可能性を確保。

---

## M4: Retire Governance Validation（Priority 4）

### T4-1 閾値運用基準の明文化

- 対象: `src/audioengine/AudioEngine.Threading.cpp` と運用文書
- 実施内容:
  - `retireHighWatermark_ / retireLowWatermark_ / retireSaturationActive_` の設計意図・期待挙動を記述。
- DoD:
  - 「通常域/警戒域/飽和域」の運用基準が文書化される。
- 検証:
  - 監査時に閾値変更理由を説明できる状態。

### T4-2 メトリクス監視観点の固定

- 対象: retire/fallback depth 周辺
- 実施内容:
  - `fallbackQueueDepth_`, `retireQueueDepth_` の観測ポイントと閾値超過時アクションを決める。
- DoD:
  - 「飽和検知→回復確認」の最低限フローが定義される。
- 検証:
  - 長時間運用ログで回復性を確認できる指標が揃う。

### T4-3 Retire Telemetry Baseline の固定

- 対象: retire/fallback telemetry と運用監視設定
- 実施内容:
  - 次の基準メトリクスを固定し、変更管理対象にする。
    - `retireQueueDepth`
    - `fallbackQueueDepth`
    - `retireHighWatermark`
    - `retireLowWatermark`
  - 観測周期（サンプリング間隔）と集計窓を定義する。
- DoD:
  - メトリクス名・単位・観測周期・閾値が文書化され、実装/運用で同一名称で追跡可能。
- 検証:
  - 長時間ログで同一フォーマットの時系列比較ができる。

---

## M5: RuntimeGraph → RuntimeWorld 依存整理（Priority 5）

### T5-1 依存分離の設計スパイク

- 対象: `src/audioengine` の runtime graph / dsp resolve 経路
- 実施内容:
  - 現状の `RuntimeGraph -> DSPCore*` 直結を棚卸しし、`RuntimeWorld -> Execution Topology -> DSPCore` への移行ステップを設計。
- DoD:
  - 非破壊で進められる分割手順（段階移行案）が作成される。
- 検証:
  - 既存Bridge安定性を崩さない順序になっていること。

---

## M6: Contract/Verifier 可視化（Priority 6）

### T6-1 Contract/Verifier 相当レイヤーの命名導入

- 対象: `src/audioengine`（非RT側）
- 実施内容:
  - 既存 validator 群を束ねる命名を導入（registry 相当の監査窓口）。
- DoD:
  - 「どの invariant をどこで検証するか」を1か所から辿れる。
- 検証:
  - 監査レビュー時の追跡コストが低下。

---

## 4. 実行順（推奨）

- **Wave A0（先行定義）**: T2-1
- **Wave A（即時）**: T1-1, T1-2, T1-3
- **Wave B**: T2-2, T3-1, T3-2, T5-1（設計スパイクのみ前倒し）
- **Wave C**: T4-1, T4-2, T4-3
- **Wave D（次段）**: T6-1

---

## 5. 受け入れゲート（Bridge Runtime 完了判定）

以下を満たした時点で「Practical Stable ISR Bridge Runtime」を完了扱いとする。

- Audio callback 経路が単一 authority view で運用される。
- `processWithSnapshot` 含む補助経路でも semantic source が統一される。
- publish 経路が共通 invariant を強制し、経路別ルール分岐がない。
- retire governance の閾値と観測手順が文書化され、運用ログで回復性が確認可能。

### 運用試験ゲート（必須）

- **Long Run IR Switch Test**
  - 30分以上の連続 IR 切替で破綻しないこと。

- **Publish Burst Test**
  - 100回以上の連続 publish 実行で invariant 逸脱・停止がないこと。
  - 100回完了後、`retireQueueDepth` / `fallbackQueueDepth` がベースライン近傍（`T4-3` で定義した許容レンジ内）へ回復すること。

- **Retire Queue Saturation Test**
  - 飽和条件を意図的に発生させ、回復経路が機能すること。
  - 飽和解除後、`retireQueueDepth` / `fallbackQueueDepth` がベースライン近傍へ収束すること。

- **Crossfade Stress Test**
  - 連続 crossfade 実行下で click/pop・状態不整合・意味論ドリフトが顕在化しないこと。

---

## 6. リスクと回避策

- リスク: authority 統一中に既存挙動差分が発生
  - 回避: Wave A を小分けし、1経路ずつ収束。

- リスク: invariant 強制で既存経路が想定外に fail
  - 回避: fail 時フォールバック方針を先に定義。

- リスク: retire 閾値調整が過剰防御/過小防御になる
  - 回避: 監視指標（queue depth/saturation）ベースで段階調整。

---

## 7. 成果物チェックリスト

- [ ] Observe Path Collapse 実装差分
- [ ] Snapshot Semantic Unification 実装差分
- [ ] Publication Invariant Centralization 実装差分
- [ ] Retire Governance 運用基準文書
- [ ] Retire Telemetry Baseline 文書（メトリクス/周期/閾値）
- [ ] 依存整理スパイク文書（Wave B で設計のみ）
- [ ] 監査性向上（Contract/Verifier 可視化）
- [ ] Long Run / Publish Burst / Retire Saturation / Crossfade Stress 試験記録

---

## 8. 現時点の成熟度（再掲）

- RT安全化フェーズ: **ほぼ完了**
- Practical Stable ISR Bridge Runtime: **80～85%**
- Final ISR Runtime: **未到達**
