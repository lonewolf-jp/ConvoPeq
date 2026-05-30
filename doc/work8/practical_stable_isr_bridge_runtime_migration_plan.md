# Practical Stable ISR Bridge Runtime 移行計画

## 目的

実運用で破綻しにくい **Practical Stable ISR Bridge Runtime** を段階的に実現する。
今回の監査結果を前提に、既存の動作を壊しにくい順序で、Runtime の真実・可視性・退役・回復力を強化する。

## 移行方針

### 基本原則

1. **観測経路を先に整える**
   - 先に可視化・診断を強化し、その後に制御経路を絞る。
2. **既存挙動を急に壊さない**
   - `activeRuntimeDSPSlot` のような legacy 経路は、即削除ではなく段階的に封じ込める。
3. **単調性を防御に昇格する**
   - 診断だけでなく、後退検知時に fail-close / quarantine / rollback に繋げる。
4. **Rollback は最小実装から始める**
   - まずは SoftRollback のみ。複雑な多段復旧は後回し。
5. **Schema は機械可読化を優先する**
   - コメントではなく、データ駆動で分類できる形を目指す。

## 現状の整理

### 既に達成寄りのもの

- Publication Sequence の追跡
- Shadow Compare の診断
- Runtime publication precheck
- Retire のバックプレッシャー計測

### まだ弱いもの

- RuntimeWorld の単一 Authority 化
- Legacy Runtime Semantic の完全除去
- Visibility Monotonicity の防御化
- SoftRollback の実運用配線
- Runtime Semantic Schema の機械可読化

## RuntimeWorld の定義

移行計画における RuntimeWorld は、単なる `RuntimeGraph` の入れ物ではなく、実運用で authoritative とみなす runtime semantic authority を保持する構造体とする。

### 必須メンバー

- `generation`
  - 生成順序と後退検知の基準。
- `graph`
  - 現在観測される runtime graph。
  - DSP topology および DSP 実行に必要な不変構造のみ保持する。
  - publication / retire / visibility / rollback / telemetry を保持してはならない。
- `publication.sequenceId`
  - publication の単調追跡と事故時の因果追跡に使う。
- `retire`
  - 退役・回収・fail-safe 状態を保持する。
- `visibility`
  - observe 境界での後退検知・可視性防御に使う。
  - 保持対象は publication visibility metadata / visibility policy state / visibility violation status とする。
  - `lastSeenGeneration` / `lastSeenSequenceId` は含めない。

### 設計上の扱い

- authoritative な runtime semantic は RuntimeWorld 以外に置かない。
- derived 値は RuntimeWorld に入れてもよいが、`schema` 上で derived と明示する。
- legacy slot は移行期間の補助に限定し、authoritative source とは見なさない。
- RuntimeWorld は Authoritative Runtime Semantic のみ保持する。
- 診断状態 / 監査状態 / Evidence 状態 / Telemetry 状態 / Debug 状態は RuntimeWorld に格納してはならない。
- それらは derived または diagnostic として扱い、Schema 上でも明示的に区別する。

## 段階的移行計画

### フェーズ 1: 監視と境界の明確化

#### 1.1 目的

現状の二重系を壊さずに、どの経路が authoritative でどの経路が legacy かを明示する。

#### 1.2 作業項目

- `RuntimeState` / `RuntimePublishWorld` を phase 1 の暫定 authoritative source として明文化する
- Phase 2 完了後は `RuntimeWorld` のみが authoritative source となる
- `activeRuntimeDSPSlot` / `fadingRuntimeDSPSlot` を「移行用」役割に限定する
- `observeCurrentRuntime()` を runtime read の主経路として明文化する
- `getActiveRuntimeDSP()` の利用箇所を棚卸しし、用途を以下に分類する
  - publish 直前の作業用
  - 退役処理用
  - 診断用
  - 互換維持用

#### 1.3 完了条件

- legacy 経路の利用目的がコード上で区別できる
- runtime read path の一覧表を作成できる
- 診断ログで current world と legacy slot の差分を追える

#### 1.4 検証

- Serena で `getActiveRuntimeDSP()` 参照一覧を再確認
- `AudioEngine` の read path / publish path を再点検

---

### フェーズ 2: RuntimeWorld 単一 Authority 化

#### 2.1 目的

Runtime の真実を world に収束させる。

#### 2.2 作業項目

- `RuntimeWorld` の struct / class 定義を固定し、必須メンバーを明記する
- authoritative な runtime semantic を RuntimeWorld に集約する
- read path の既定値を `RuntimePublicationCoordinator::observePublishedWorld()` へ寄せる
- `RuntimeGraph` / `RuntimeState` から読める値を優先し、legacy slot は補助へ回す
- `logRuntimeTransitionEvent()` や `validateDistinctRuntimeSlots()` の役割を「監査専用」に整理する
- `prepareToPlay()` / `releaseResources()` / `commitNewDSP()` の中で、slot 参照を最終的に world 参照へ置換できる箇所を切り出す

#### 2.3 完了条件

- `RuntimeWorld` 以外は Runtime Semantic Authority を生成・変更してはならない
- `RuntimeWorld` の authoritative field は RuntimeWorld 管理コンポーネント以外から直接変更できない
- `RuntimeWorld` 以外に authoritative runtime semantic を保持する構造体が存在しない
- `RuntimeWorld` が `generation` / `graph` / `publication.sequenceId` / `retire` / `visibility` を保持する
- `visibility` は publication visibility metadata / visibility policy state / visibility violation status のみを保持する
- legacy slot は「構造の保持」以外に使われなくなる
- 進行中でも world と slot の不一致は診断に閉じる

#### 2.4 検証

- `AudioEngine::makeRuntimeReadView()` の依存関係を再確認
- `RuntimePublicationCoordinator::observePublishedWorld()` の参照箇所を確認

---

### フェーズ 3: Runtime Semantic Schema の機械可読化

#### 3.1 目的

新規フィールド追加時の逸脱を、レビューではなく機械で検出する。

#### 3.2 作業項目

- `RuntimeFieldDescriptor` 相当の定義を追加する
- 各 field に以下を付与する
  - `AuthorityClass`
  - `Ownership`
  - `Mutability`
  - `Visibility`
  - `Lifetime`
- `RuntimeWorld` / `RuntimeGraph` / `PublicationSemantic` のフィールド群を schema で検証する
- CI またはローカルチェックで schema drift を検出する

#### 3.3 完了条件

- `RuntimeWorld` の各フィールドが authoritative / derived として分類される
- comment 依存だった分類を data-driven に移行できる
- developer が追加した field の責務が曖昧になりにくい

#### 3.4 検証

- Schema 逸脱のテストを 1 本以上用意する
- `kRuntimeSemanticSchemaVersion` 更新条件を明文化する

---

### フェーズ 4: PublicationSequenceId 導入

#### 4.1 目的

全 publish に因果追跡可能な sequence を付与する。

#### 4.2 作業項目

- `publication.sequenceId` を RuntimeWorld の一部として常時保持する
- `publication.sequenceId` は process lifetime で単調増加する 64bit 値とする
- `PublicationSequenceId` 生成器はシステム内で一意とする
- publish 直前に sequence を単調増分する
- `RuntimePublicationCoordinator` / retire / evidence 出力で同一 sequence を参照できるようにする
- sequence 欠損時は publish を fail-close できるようにする

#### 4.3 完了条件

- 全 publish に `PublicationSequenceId` が付与される
- 後追いで「なぜ Runtime が飛んだか」を追跡できる
- sequence 欠損 publish を受理しない

#### 4.4 検証

- evidence と runtime log で同一 sequence を照合する
- sequence の欠損 / 重複シナリオを人工的に作って拒否を確認する

---

### フェーズ 5: Visibility Monotonicity の防御化

#### 5.1 目的

後退観測を「診断」から「異常制御」に昇格する。

#### 5.2 作業項目

- `lastSeenGeneration` / `lastSeenSequenceId` 相当の state を read side に保持する
- それらは Observe 側ローカル状態とし、Publication Authority に影響してはならない
- `newGeneration < lastSeenGeneration` を検出したら異常とみなす
- `publication.sequenceId` の後退は異常、飛びは Telemetry 対象、欠損は観測不整合として扱う
- 異常時の動作を段階化する
  - まずは telemetry escalation
  - 次に quarantine / fail-close
  - 必要なら SoftRollback
- `DebugRuntime::recordShadowCompareObservation()` の結果を、監査だけでなく制御判断に接続できる形へ整える

#### 5.3 完了条件

- generation / sequence の後退が全 observe 経路で検知される
- 異常時に「静かに誤表示する」状態を減らせる
- 監査ログに加えて制御フラグが立つ

#### 5.4 検証

- ダミーの後退シナリオで monotonic violation が検出される
- 同一反復内の不整合が telemetry に記録される

---

### フェーズ 6: Legacy Runtime Semantic の除去

#### 6.1 目的

「active / current / latest / visible」などの曖昧な runtime semantic を減らし、観測経路を一本化する。

#### 6.2 作業項目

- `active runtime` / `current runtime` / `visible runtime` / `latest runtime` の用語を整理する
- `observeCurrentRuntime()` 以外の「現在値取得」の意味をコードコメントと呼び出し点で明確化する
- legacy slot 参照を用途別に分離し、単純な read path と混ぜない
- 二重定義している概念の名称を統一する

#### 6.3 完了条件

- runtime の「現在」を表す表現が 1 種類に収束する
- legacy runtime semantic を提供する API は deprecated / migration-only / diagnostic-only のいずれかに分類され、新規コードから利用してはならない
- legacy semantic が read path に入り込まない
- 参照箇所の説明だけ読めば責務境界が分かる

#### 6.4 検証

- Serena で `observeCurrentRuntime()` / `getActiveRuntimeDSP()` / legacy alias の参照差分を確認
- `AudioEngine.h` と `SnapshotCoordinator.h` の責務文言を再点検

---

### フェーズ 7: Shadow Compare を fail-safe へ接続

#### 7.1 目的

比較結果を単なるレポートで終わらせず、異常制御へ繋げる。

#### 7.2 作業項目

- `RuntimeSemanticHash` / `TopologyHash` / `PublicationHash` / `Generation` の比較結果を整理する
- 旧 observe 経路と新 observe 経路の結果を比較する
- mismatch や monotonic violation の閾値を定義する
- 一定以上の不整合では Quarantine 推奨または SoftRollback 候補までに留める
- Phase 7 では直接 publish 抑制しない
- 比較結果を evidence と runtime policy に接続する

#### 7.3 完了条件

- mismatch が出ても放置しない
- 旧経路と新経路の semantic hash を比較できる
- 監査ログと制御が同じ事象を指す
- review / ops で追跡しやすい

#### 7.4 検証

- shadow compare mismatch シナリオを再現し、制御フラグが変化することを確認
- `ASSERT(oldHash == newHash)` 相当の監査が働くことを確認

---

### フェーズ 8: Retire Governance の強化

#### 8.1 目的

退役バックログが長時間運転で肥大化しても、回収可能な範囲に抑える。

#### 8.2 作業項目

- retire pressure の主要指標を整理する
  - retire queue depth
  - fallback queue depth
  - quarantine resident
  - reclaim latency
- 水位制御を明文化する
  - high watermark
  - low watermark
  - saturation enter/exit
- `RetireRuntimeEx` の lane / lifecycle / rollback state の関係を整理する
- `maxRetireDeferralEpochs` に相当する制御が必要かを判断する
- `maxObservedRetireDeferralEpochs` を必ず記録する

#### 8.3 完了条件

- backlog の増大と回復が観測可能
- retire queue が有限時間で減衰する
- saturation 時の挙動が安定している
- 退役が無限滞留しにくい

#### 8.4 検証

- 長時間運転相当の負荷で retire queue が回復することを確認
- `isFullyDrained()` が妥当な条件で true になることを確認

---

### フェーズ 9: SoftRollback の最小実装

#### 9.1 目的

異常 publish 時に最低限の復旧手段を持つ。

#### 9.2 作業項目

- 異常検知条件を 1～2 個に絞る
  - monotonic violation
  - publication mismatch
- 最後に正常 Publish された RuntimeWorld を SoftRollback 対象とする
- 直前 world の再 publish を最小機能として実装する
- rollback が失敗した場合の安全側挙動を決める
- `RetireRuntimeEx::requestRollback()` を実運用経路へ接続する

#### 9.3 完了条件

- rollback API が実際に呼ばれる
- 失敗時の安全側動作が明確
- 「戻せる」ことが運用上確認できる

#### 9.4 検証

- 人工的な異常 publish シナリオで soft rollback を確認
- 失敗時に再悪化しないことを確認

---

## 実行順序の推奨

1. フェーズ 1
2. フェーズ 2
3. フェーズ 3
4. フェーズ 4
5. フェーズ 5
6. フェーズ 6
7. フェーズ 7
8. フェーズ 8
9. フェーズ 9

## 優先度

### 最優先

- フェーズ 2
- フェーズ 3
- フェーズ 4
- フェーズ 5
- フェーズ 9

### 次点

- フェーズ 6
- フェーズ 7
- フェーズ 8

### 補助

- フェーズ 1

## リスクと注意点

### 1. Legacy slot を急に消さない

- `activeRuntimeDSPSlot` は多くのコードから参照されている。
- いきなり削除すると publish / teardown の両方で壊れやすい。

### 2. Monotonicity の強制は段階的に

- 最初から hard fail にすると運用停止のリスクがある。
- まずは diagnostic + quarantine、次に fail-close。

### 3. Schema と sequence を RuntimeWorld 導入時に固定する

- RuntimeWorld の定義が曖昧なままだと、後工程で意味論が肥大化する。
- `schema` と `publication.sequenceId` は導入初期に固定する。

### 4. Rollback は最小から

- いきなり多段 rollback を作らない。
- まずは直前 world の再 publish に限定する。

### 5. Schema 化は既存コードとの整合を崩しやすい

- `kRuntimeSemanticSchemaVersion` の更新条件を明文化する。
- 既存 artifact / evidence との互換を確認する。

## 完了の定義

Practical Stable ISR Bridge Runtime の中核到達条件は以下。

1. RuntimeWorld 単一 Authority 化
2. Runtime Semantic Schema の固定
3. Publication SequenceId の全 publish 付与
4. Visibility Monotonicity の全 observe 保証
5. Legacy Runtime Semantic の除去
6. Shadow Compare の fail-safe 接続

Observe 経路 / Publication 経路 / Retire 経路のすべてで `generation` と `publication.sequenceId` を追跡可能であることを、最終完了条件に含める。
Runtime Semantic Authority の生成経路、更新経路、観測経路がそれぞれ一意に特定できることを、最終完了条件に含める。

この 6 つがそろったら、実運用の観点で「真実が 1 つ」「意味論が固定されている」「因果が追跡可能」「可視状態が後退しない」が成立する。

---

作成日: 2026-05-30
根拠: `doc/work8/practical_stable_isr_bridge_runtime_audit.md`
