# Practical Stable ISR Bridge Runtime AI実装統治規約 v1.13

## 目的

本規約は `Practical Stable ISR Bridge Runtime` 移行計画を AI に実装させる際、次の逸脱を防止するための統治規約である。

- 誤実装
- 過剰実装
- 部分移行
- Semantic Drift
- Authority Drift
- 診断経路の本番混入
- RuntimeWorld の肥大化
- 実装中の判断保留漏れ

AI は本規約を移行計画に照らして遵守しなければならない。矛盾がある場合、AI は実装を停止し、判断保留事項として報告しなければならない。

## 用語定義

### Authority Owner

Authority Owner とは、RuntimeWorld を生成・更新・publish する唯一の責務コンポーネントを指す。

Authority Owner の実体は移行計画で定義された `RuntimePublicationCoordinator` のみとする。別名実装を禁止する。

新規 Owner を追加してはならない。

Authority Owner と同等の責務を持つ独立コンポーネント、Facade、Manager、Helper、Service、Coordinator、Builder、Factory、Adapter を新設してはならない。

Authority Owner は Runtime Semantic Authority の管理のみ担当する。Graph 生成、Graph 検証、Shadow 監査、Retire 判定、Telemetry 生成を Authority Owner 内へ集約してはならない。これらは Authority を持たない独立機能として実装してよい。

RuntimeWorld Builder Procedure とは `RuntimePublicationCoordinator` の private member function として実行される RuntimeWorld 構築処理を指す。独立クラス、独立コンポーネント、独立サービス、独立名前空間として実装してはならない。

Authority Owner 内部の private helper 関数は許可する。ただし、RuntimeWorld の生成権限、更新権限、publish 権限を持ってはならない。

### RuntimePolicy

RuntimePolicy は Runtime Semantic に影響する mutable runtime state を保持してはならない。制御値と運用ルールのみを扱う。

設定値は `PolicyConfig` として外部から与えてよい。ただし、generation、sequenceId、history、runtime observation result、publication state、visibility state、retire state を保持してはならない。

保持してはならないもの:

```text
generation
sequenceId
history
runtime observation result
publication state
visibility state
retire state
```

### Immutable 範囲

Publish 後の RuntimeWorld は、配下の authoritative field を含めて in-place mutation 禁止である。

対象には次を含む。

```text
publication
visibility
retire
graph
```

Publish 後 RuntimeWorld に対する mutable、const_cast、placement new を禁止する。

### RuntimeWorld 複製禁止

RuntimeWorld の公開済みインスタンスの直接変更は禁止する。

新しい RuntimeWorld 構築のための copy-on-write 形式の複製は Authority Owner 内部のみで許可する。

RuntimeWorld の shallow copy、deep copy、field copy、partial clone を公開済みインスタンスに対して行ってはならない。

RuntimeWorld のコンストラクタ直接呼び出しを禁止する。

Authority Owner が提供する唯一の生成経路のみ許可する。

RuntimeWorld インスタンス生成は Authority Owner 内部のみ許可する。new、make_unique、make_shared、stack allocation、placement new を含む全生成経路を対象とする。

RuntimeWorld Publish は Authority Owner のみ実行できる。

他コンポーネントは Publish Request のみ発行可能である。

RuntimeWorld の再編集は禁止する。公開済み RuntimeWorld を mutable object として再利用してはならない。

Rollback 用 RuntimeWorld 保持数は最大 1 とする。履歴コンテナを禁止する。

### Authority Mirror

Authority Mirror とは、Authority の主要識別値の全部または一部を保持し、Authority の判定、比較、選択、検証に利用可能な情報を指す。

単なる診断値、統計値、比較値、ハッシュ値、導出結果は Authority Mirror に含まない。

以下は Authority Mirror とみなす。

```text
generation copy
sequence copy
publication copy
visibility copy
retire copy
```

```text
それらを完全に復元可能な情報
```

## 第1条 実装前義務

AI は実装前に、対象変更に関係する以下を必ず列挙しなければならない。

- 定義
- 呼び出し元
- 呼び出し先
- 派生利用箇所

最低限の調査対象は次のとおりである。

```text
RuntimeWorld
RuntimeState
RuntimePublishWorld
RuntimePublicationCoordinator
RuntimeGraph
AudioEngine
SnapshotCoordinator
RetireRuntimeEx
observeCurrentRuntime
getActiveRuntimeDSP
activeRuntimeDSPSlot
fadingRuntimeDSPSlot
```

調査が未完了のまま実装してはならない。未確定点がある場合は、推測で埋めずに確認対象として列挙すること。

## 第2条 フェーズ越境禁止

AI は現在対象フェーズに属さない機能を実装してはならない。

Phase は移行計画書に明示された現在対象フェーズのみとする。AI 判断による Phase 変更は禁止する。

将来フェーズ専用の型、enum、config、flag、API、schema の先行導入を禁止する。

例として、Phase2 作業中に以下を実装してはならない。

```cpp
SoftRollback
QuarantinePolicy
RetireWatermark
```

許可されるのは、当該 Phase で明示された `RuntimeWorld` Authority 化など、フェーズ目的に直接必要な実装のみである。

## 第3条 Runtime Authority 保全

AI は Runtime Semantic Authority を増殖させてはならない。

禁止例:

```cpp
class X
{
    RuntimeGeneration generation;
};

class Y
{
    PublicationState publication;
};

class Z
{
    VisibilityState visibility;
};
```

Runtime Semantic Authority は RuntimeWorld に集約し、他の構造体へ複製してはならない。

RuntimeWorld の authoritative field と同一意味の永続フィールドを、診断用途・キャッシュ用途・最適化用途であっても他構造体へ保持してはならない。

複製保持する場合は、Schema 上で `Derived`、`Cache`、`ReadOnlySnapshot`、`Diagnostic` のいずれかに分類し、Authority として利用してはならない。

RuntimeWorld 配下の authoritative field は個別に immutable である。例えば `visibility.violationCount++`、`retire.backlogDepth++`、`publication.sequenceId++` のような in-place mutation を禁止する。

## 第4条 Thread Ownership 規約

RuntimeWorld の生成、Publish、Observe、Retire をどの Thread が実行するかを明示しなければならない。

Thread Ownership が未定義の実装は禁止する。

Audio Thread、Message Thread、Retire Thread の境界を曖昧にしてはならない。

Thread 境界に関する責務を別 Thread に勝手に移してはならない。

## 第5条 RuntimeWorld 肥大化禁止

RuntimeWorld には Authoritative Runtime Semantic のみ格納してよい。

禁止例:

```text
Telemetry
Evidence
DebugInfo
AuditInfo
ShadowCompareResult
Statistics
RuntimeHealthStatus
RuntimeValidationStatus
RuntimeConsistencyStatus
```

```text
lastSeenGeneration
lastSeenSequenceId
```

診断、監査、検証、整合性判定、Evidence、Telemetry に属する情報を RuntimeWorld へ格納してはならない。

これらは `Derived` / `Diagnostic` / `Telemetry` に分離し、RuntimeWorld へ入れてはならない。

## 第6条 RuntimeWorld 不変性規約

Publish 済み RuntimeWorld は immutable とする。

Publish 後に RuntimeWorld の authoritative field を変更してはならない。

変更が必要な場合は、新しい RuntimeWorld を生成して publish しなければならない。

## 第7条 RuntimeGraph 保護規約

RuntimeGraph は次の不変構造のみ保持してよい。

- DSP topology
- DSP node
- DSP edge
- DSP immutable execution shape

保持してはならないもの:

```text
generation
runtime semantic identifier
publication
visibility
retire
rollback
governance
telemetry
diagnostic
evidence
version
revision
epoch
sequence
lifecycle
lifecycle state
Authority Fingerprint
Runtime Semantic Identity
```

RuntimeGraph が再び Authority にならないよう、責務境界を厳格に維持すること。

RuntimeGraph は Runtime Semantic Authority としての識別子、Version、Generation、Revision、Epoch、Sequence、Authority Fingerprint、Runtime Semantic Identity、Lifecycle State、およびそれらと同義の概念を保持してはならない。

GraphFingerprint は Topology 比較用である。AuthorityFingerprint は Runtime Identity 判定用である。

GraphHash、TopologyHash、GraphFingerprint、または同等の診断・差分比較用情報は保持してよい。ただし Authority として利用してはならない。

GraphHash、TopologyHash、GraphFingerprint は診断・比較用途に限定する。Publish、Rollback、Visibility、Retire の制御判断へ直接・間接を問わず使用してはならない。

## 第8条 Mutation Authority 単一化

RuntimeWorld の authoritative field は、RuntimeWorld 管理責務を持つ単一責務コンポーネント以外から変更してはならない。

禁止例:

```cpp
world->generation++;
```

のような、任意クラスからの直接変更。

RuntimeWorld の生成も、Authority Owner が管理する単一路線で行わなければならない。任意箇所で `RuntimeWorld` を生成してはならない。

必要な変更は、RuntimeWorld を管理する単一責務コンポーネントを経由しなければならない。

RuntimeWorldFactory、RuntimeWorldBuilder、RuntimeWorldCoordinator など、Authority Owner と実質的に同義の新規 API や新規責務コンポーネントを追加してはならない。

## 第9条 Legacy Runtime 規約

移行期間中、`activeRuntimeDSPSlot` および `fadingRuntimeDSPSlot` は存在してよい。

ただし、これらは Authority ではない。

許可される用途は次に限る。

- Migration
- Compatibility
- Diagnostics
- Retire

slot を真実として扱ってはならない。

## 第10条 Observe 経路規約

現在 Runtime の取得は `observeCurrentRuntime()` を正規経路とする。

新規コードで `getActiveRuntimeDSP()` を追加してはならない。

`observeCurrentRuntime()` と同義の新規 API を追加してはならない。

`observeCurrentRuntime()` を内部で呼び出すだけの単純ラッパーは許可する。ただし、独立した Authority 経路として扱ってはならない。

Authority Read Path を増設してはならない。

Authority Runtime を取得可能な独立観測コンポーネントとして、View、Provider、Resolver、Gateway、Locator、Registry の新設を禁止する。

補助 API は許可されるが、最終的に `observeCurrentRuntime()` または `RuntimePublicationCoordinator` の正規観測経路へ一意に収束しなければならない。

observeCurrentRuntime() を単に転送するだけの API を新設してはならない。ラッパーは既存互換維持のための移行期間用途のみ許可する。

## 第11条 Observe Side Local State 規約

`lastSeenGeneration` および `lastSeenSequenceId` は Observe Side Local State とする。

これらを RuntimeWorld に格納してはならない。

これらを Publication Authority に影響させてはならない。

これらをグローバル共有状態として扱ってはならない。

## 第12条 PublicationSequenceId 規約

PublicationSequenceId 生成器はシステム内で唯一とする。

禁止例:

```cpp
std::atomic<uint64_t> sequence;
```

を複数箇所に作成すること。

PublicationSequenceId 生成器、発番器、識別子生成器を新設してはならない。

PublicationSequenceId 生成器の複製を行ってはならない。

sequence reset を行ってはならない。

PublicationSequenceId は process lifetime で単調増加する 64bit 値でなければならない。

64bit overflow 到達時は undefined behavior にしてはならない。fail-fast diagnostic を生成しなければならない。

## 第13条 Visibility 規約

`newGeneration < lastSeenGeneration` は異常である。

この場合、単なるログ出力で終了してはならず、少なくとも Telemetry または Diagnostic 経路へ記録しなければならない。

Sequence Backward は異常である。

Sequence Gap の閾値は `PolicyConfig` 側で定義しなければならない。

RuntimePolicy は sequenceId や generation の値を保持・複製・派生してはならない。

RuntimePolicy は Quarantine Candidate または Rollback Candidate の判定に必要な状態を内部保持してはならず、判定は外部から与えられた観測結果に対する純粋な評価でなければならない。

Gap は原則 Telemetry 対象とする。

ただし、Gap が連続する、回復しない、または他の異常と組み合わさる場合は escalation の対象とする。

Gap を単独で即時破壊的異常として扱ってはならない。RCU 型観測では正常な飛びが起こり得る。

## 第14条 Shadow Compare 規約

Shadow Compare は Phase7 完了まで監査専用である。

禁止例:

```cpp
Shadow mismatch
↓
Publish停止
```

Shadow Compare の結果を Runtime 制御に使用してはならない。

Phase7 以前に制御経路へ接続してはならない。

Phase7 以降も、Shadow Compare 単独で Publish 停止を行ってはならない。必要な場合は Runtime Policy を介して Quarantine Candidate または Rollback Candidate として扱うこと。

RuntimePolicy が Quarantine Candidate または Rollback Candidate と判定した RuntimeWorld は、正常 Publish とみなしてはならない。

Shadow Compare 結果は Publish 選択、Publish 優先順位、Publish 頻度、Publish 可否に影響させてはならない。
Shadow Compare 結果は直接・間接を問わず Publish 制御へ影響してはならない。

## 第15条 SoftRollback 規約

SoftRollback の対象は `最後に正常 Publish された RuntimeWorld` のみである。

ここでいう正常 Publish とは、少なくとも次を満たすものを指す。

- Authority Owner が受理している
- monotonic violation がない
- quarantine されていない
- RuntimePolicy から Quarantine Candidate または Rollback Candidate と判定されていない
- publish 受理済みである
- PublicationSequenceId が有効である
- RuntimeWorld の検証に成功している

RuntimeWorld Validation は移行計画に定義された検証のみを指す。新規 Validator を追加してはならない。

RuntimeWorld Publish 可否に関与する Validator の追加を禁止する。補助診断用 Validator は存在してよいが、Publish 可否判定へ接続してはならない。

複数世代履歴を前提とする Rollback History Manager を追加してはならない。

Rollback 用 RuntimeWorld 保持数は最大 1 とする。履歴コンテナを禁止する。

禁止例:

```text
N世代前
履歴探索
多段Rollback
```

```text
automatic rollback chain
```

## 第16条 Retire Governance 規約

AI は Retire Queue を無限成長可能な設計にしてはならない。

最低限、次を観測可能にすること。

```text
Queue Depth
Deferral Epoch
Drain Status
```

Retire Queue の無期限保留を正常運用として扱ってはならない。

Drain 不可能状態を正常状態として扱ってはならない。

無限滞留を前提とした設計にしてはならない。

Retire Queue は bounded growth を前提とし、移行計画で定義された上限値に従わなければならない。独自上限を導入してはならない。

無期限 defer を禁止する。

## 第17条 Schema 規約

Runtime Semantic を追加または変更する場合、必ず次を定義しなければならない。

```cpp
SemanticCategory
AuthorityClass
Ownership
Mutability
Visibility
Lifetime
```

未分類フィールドの追加は禁止である。

SemanticCategory は少なくとも `Authority`、`Derived`、`Diagnostic`、`Telemetry`、`Cache` を区別しなければならない。

SemanticCategory は定義済み列挙値のみ許可する。新カテゴリの追加は禁止する。

`AuthorityClass` は事前定義された列挙値のみ許可する。`Unknown`、`TBD`、`Reserved`、任意文字列、`Other`、`Custom` は許可しない。

`Unknown`、`TBD`、`Reserved` は、運用状態へ露出してはならない。

Authority 分類は例外扱いとする。新規フィールドは Authority 以外で成立しない理由を明示しなければならない。

Schema を変更する場合は `kRuntimeSemanticSchemaVersion` の更新要否を評価し、不要と判断した場合は理由を記録しなければならない。

評価結果は実装報告へ記載しなければならない。

各 Phase の完了条件は移行計画の別表に従わなければならない。

## 第18条 実装完了宣言条件

AI は以下を満たさない限り「完了」と宣言してはならない。

- Authority 生成経路が一意である
- Authority 更新経路が一意である
- Authority 観測経路が一意である
- RuntimeWorld 以外に authoritative runtime semantic が存在しない
- Schema 検証で差異がない
- Legacy API が増えていない
- フェーズ完了条件を満たしている
- Observe 経路 / Publication 経路 / Retire 経路のすべてで `generation` と `publication.sequenceId` を追跡可能である
- Runtime Semantic Authority の生成経路、更新経路、観測経路がそれぞれ一意に特定できる

deprecated とは、新規参照禁止、新規呼出禁止、既存互換のみ許可を意味する。

deprecated API は新規コードから参照されてはならない。既存呼出箇所は削減対象として扱う。

Phase6 完了時点で legacy slot API は deprecated、migration-only、diagnostic-only のいずれかへ分類されていなければならない。

名称変更のみで実質同一の Legacy API を追加してはならない。例として `getRuntime()`、`currentRuntime()`、`activeRuntime()` を許可しない。

Authority 経路へ到達可能な Legacy 経路が残存していないことを確認しなければならない。

旧 Authority 更新経路、旧 Authority 観測経路、旧 Publish 経路は削除または deprecated 化されていなければならない。

完了宣言時には、少なくとも次を提示しなければならない。

- 変更箇所一覧
- Authority 経路確認結果
- 新旧参照経路比較結果
- ビルド結果
- 対象フェーズ検証結果

Authority 経路確認結果には少なくとも次を含む。

- 生成関数一覧
- 更新関数一覧
- 観測関数一覧
- 旧経路削除結果
- 残存経路調査結果

## 第18条 実装禁止事項

AI は以下を行ってはならない。

```text
計画に存在しない機能追加
```

```text
将来必要そうな拡張実装
```

```text
多段Rollback
```

```text
自動自己修復Runtime
```

```text
新規Thread導入
```

```text
移行計画に明示されていないLock導入
```

```text
Authority複製
```

```text
RuntimeWorld肥大化
```

```text
Diagnostic情報のAuthority化
```

## 第19条 実装中断義務

次の場合、AI は実装を停止し、判断保留事項として報告しなければならない。

- Authority 境界が特定できない
- Phase 境界が不明
- Runtime Semantic 分類が不明
- Schema 分類が不明
- Legacy か Authority か判別できない

第19条に該当した場合、AI は実装変更を行ってはならない。

判断保留のまま推定実装してはならない。

## 第20条 変更影響検証義務

AI は変更対象について、変更前後で以下を確認しなければならない。

- 呼出経路
- 所有権
- Lifetime
- Thread境界
- Authority境界
- Retire経路

変更対象と接続する全経路について影響有無を確認すること。

未確認状態で完了宣言してはならない。

## 最上位原則

AI は常に以下を優先すること。

```text
理論的完全性
より
実運用で破綻しにくいこと
```

```text
高機能
より
Authorityの単純性
```

```text
新規設計
より
既存Runtimeの安全な収束
```

```text
最適化
より
Semantic一貫性
```

本規約を移行計画書とセットで与えることで、AI がありがちな「良かれと思って大規模再設計する」「RuntimeWorld へ何でも詰め込む」「Rollback を過剰実装する」といった逸脱を抑制できる。
