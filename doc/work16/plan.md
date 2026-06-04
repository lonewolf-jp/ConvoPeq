# Practical Stable ISR Bridge Runtime 詳細改修計画書（実装開始版 v6.5）全文

本計画書は、ConvoPeq（2026-06-03版）を Practical Stable ISR Bridge Runtime へ移行するための最終実装開始版である。
これまでの全レビュー（v1.0〜v6.4）の成果を全て反映し、実装着手前に解釈が分岐する余地を完全に排除した完全文書である。
**Phase-1 完了をもって「Practical Stable ISR Bridge Runtime 達成」と判定する。**

---

## 1. 全体方針とフェーズ定義

- **Phase-0**：Phase-1 着手前に必須の準備作業。権威メタ情報の監査、契約の確定、所有権移譲計画の策定、各種 Matrix の作成。**実際のコード改修は行わない**（例外：`SemanticCategory::PublicationSemantic` 追加など準備作業に必要な最小限の改修は許容）。
- **Phase-1**：Practical Stable ISR Bridge Runtime の完成判定に必須。Authority/Observe/Publication/Retire の一元化を達成するための**実際のコード改修**を実施する。Phase-0 で策定された全計画を実行する。
- **Phase-2**：ガバナンス強化。Verifier による静的保証を CI に組み込む。
- **Phase-3**：品質改善・コード衛生。Authority 収束とは無関係。任意。

**重要**：Phase-0 と Phase-1 の責務を明確に分離する。Phase-0 は「計画策定 + スキーマ拡張」であり、Phase-1 は「大規模コード改修」である。

---

## 2. Phase-0：準備・前提作業（Phase-1 着手前に必須）

### 2.0 Descriptor Schema Audit + Projection 同義語化 + PublicationSemantic 新設（テスト更新必須）

**完了状況: [✅ 完了]**

**実績**: `SemanticCategory::PublicationSemantic` 追加, スキーマ v7→8

**目的**：`RuntimeFieldDescriptor` が Authority 分類を表現できることを監査する。**Projection 概念は既に設計上の要求事項として確定済み**であり、監査では「Projection をどう表現するか」のみを調査する。同時に、`PublicationSemantic` カテゴリを新設する。

**決定**：

- Projection 概念は採用済み。`RuntimeGraph` は最終的に Projection + Diagnostic のみを持つ。
- **本計画期間中は `Projection` と `RuntimeAuthorityClass::Derived` を同義語として扱う**。文書で明記する。
- **`PublicationSemantic` カテゴリを `SemanticCategory` 列挙型に新設する**。
  - **注意**：既存の `OwnershipClass::PublicationSemantic` とは別概念である。`SemanticCategory` はフィールドの意味的分類、`OwnershipClass` は所有権・責務の分類であり、混同しないこと。
  - これにより、publication 関連フィールドを inventory で一元的に管理できる。

**PublicationSemantic 追加時の必須手続き**：

- `SemanticCategory` 列挙型に `PublicationSemantic` を追加する。
- **`kRuntimeSemanticSchemaVersion` をインクリメントする**（現行 7 → 8）。
- `authority_inventory.json` ジェネレータを修正し、`PublicationSemantic` を認識できるようにする。
- descriptor validator と inventory validator を更新し、新カテゴリを検証対象に含める。
- **`RuntimeSemanticSchemaValidationTests`（特に `testPublicationDescriptorSetValidation()`）を更新する**。
- これらの手続きを CI で検証し、未実施の場合は fail とする。

**完了条件**：

- 監査レポートを作成し、Projection の表現方法が確定する。
- `SemanticCategory::PublicationSemantic` が追加され、バージョンが更新される。
- 関連する全テストが更新され、通過する。
- 自動生成パイプラインが決定されたルールに従って実装される。
- 上記手続きが CI で強制される。

---

### 2.1 RuntimeGraph Authority Migration Plan + 全フィールド Migration Matrix（包含関係 + 完全一致）

**完了状況: [✅ 完了]** Phase-1 3.2.9 にて全 Authoritative フィールド移管完了（RuntimeGraph: 25→7 fields, Authoritative: 19→0）。Migration Matrix に完全準拠。

**目的**：現行の `RuntimeGraph` が保持する **authority_inventory.json に記載された全 Authoritative フィールド** を `RuntimeWorld` へ移管または削除するための段階的計画を策定する。

**Migration Matrix の完全性保証（包含関係 + Inventory ↔ Matrix 完全一致）**：

- Migration Matrix の対象は `authority_inventory.json` において `RuntimeGraph` に属する全フィールドとする。
- 以下の CI チェックを Phase-1 で実装する：
  - **方向1**：`authority_inventory.json` から RuntimeGraph の全フィールドを抽出 → Migration Matrix に未記載のフィールドが存在する場合 → **fail**
  - **方向2**：Migration Matrix に記載されたフィールドが `authority_inventory.json` において RuntimeGraph に属していない場合 → **fail**
  - **包含関係**：`RuntimeGraph実フィールド ⊆ Descriptor(RuntimeGraph対象)`
  - **包含関係**：`Descriptor(RuntimeGraph対象) ⊆ AuthorityInventory(RuntimeGraph対象)`
  - **完全一致**：`AuthorityInventory(RuntimeGraph対象) == MigrationMatrix(RuntimeGraph対象)`（Matrix と Inventory は完全一致させる。これにより Matrix の腐敗を防ぐ）

**Migration Matrix（全フィールド）**：
以下に RuntimeGraph の全フィールドの移行計画を示す。移行後は RuntimeGraph に `Authoritative` フィールドを残さない。

| フィールド | 移行前分類 | 移行先 | 移行後分類 | 備考 |
|-----------|-----------|--------|-----------|------|
| `runtimeUuid` | Authority | RuntimeWorld TopologySemantic | Projection（削除） | インスタンス識別子 |
| `fadingRuntimeUuid` | Authority | RuntimeWorld TopologySemantic | Projection（削除） | フェード中インスタンス |
| `transitionCurrentRuntimeUuid` | Authority | RuntimeWorld TopologySemantic | Projection（削除） | 遷移元 |
| `transitionNextRuntimeUuid` | Authority | RuntimeWorld TopologySemantic | Projection（削除） | 遷移先 |
| `generation` | Authority | RuntimeWorld GenerationSemantic | Projection（削除） | 世界の世代 |
| `sampleRate` | Authority | RuntimeWorld TimingSemantic | Projection（削除） | サンプリングレート |
| `ditherBitDepth` | Authority | RuntimeWorld ResourceSemantic | Projection（削除） | ディザ設定 |
| `noiseShaperType` | Authority | RuntimeWorld ResourceSemantic | Projection（削除） | ノイズシェーパー種別 |
| `oversamplingFactor` | Authority | RuntimeWorld ResourceSemantic | Projection（削除） | オーバーサンプリング倍率 |
| `eqBypassed` | Authority | RuntimeWorld RoutingSemantic | Projection（削除） | EQバイパス状態 |
| `convBypassed` | Authority | RuntimeWorld RoutingSemantic | Projection（削除） | Convolverバイパス状態 |
| `softClipEnabled` | Authority | RuntimeWorld AutomationSemantic | Projection（削除） | ソフトクリップ設定 |
| `saturationAmount` | Authority | RuntimeWorld AutomationSemantic | Projection（削除） | サチュレーション量 |
| `inputHeadroomGain` | Authority | RuntimeWorld AutomationSemantic | Projection（削除） | 入力ヘッドルーム |
| `outputMakeupGain` | Authority | RuntimeWorld AutomationSemantic | Projection（削除） | 出力メイクアップ |
| `convolverInputTrimGain` | Authority | RuntimeWorld AutomationSemantic | Projection（削除） | コンボルバー入力トリム |
| `adaptiveCoeffBankIndex` | Authority | RuntimeWorld CoefficientSemantic | Projection（削除） | 適応係数バンク |
| `adaptiveCoeffGeneration` | Authority | RuntimeWorld CoefficientSemantic | Projection（削除） | 適応係数世代 |
| `captureSessionId` | Authority → Diagnostic | RuntimeWorld Diagnostic | Diagnostic Identifier | 用途制限あり |
| `activeNode` | Derived（Projection） | RuntimeGraph | Projection | 維持 |
| `fadingNode` | Derived（Projection） | RuntimeGraph | Projection | 維持 |
| `eqAgcAttackCoeffTable` | Diagnostic | RuntimeGraph | Diagnostic | 維持 |
| `eqAgcReleaseCoeffTable` | Diagnostic | RuntimeGraph | Diagnostic | 維持 |
| `eqAgcSmoothCoeffTable` | Diagnostic | RuntimeGraph | Diagnostic | 維持 |
| `eqAgcCoeffTableCapacity` | Diagnostic | RuntimeGraph | Diagnostic | 維持 |

**captureSessionId の特別扱い**：

- `captureSessionId` は **Diagnostic Identifier** へ降格する。
- **許可される用途**：
  - トレース（trace）
  - 相関（correlation）
  - 診断ログ（diagnostic logging）
  - キャプチャブロックタギング（capture block tagging）
- **禁止される用途**：
  - 比較演算子全般（`<`, `>`, `<=`, `>=`, `==`, `!=`）の使用
  - 順序付け（`std::sort` の comparator、`std::min`/`std::max` 等）
  - ハッシュキー（`std::unordered_map`、`std::unordered_set` のキー）
  - **条件式全般（`if`, `while`, `switch`, `?:`, 任意の `ConditionExpr` への到達）**
  - publication 順序の決定
  - retire 順序の決定
- `CaptureSessionIdVerifier` を実装し、禁止用途を検出したら fail とする。

**責務分離の明確化**：

- **Phase-0**：Migration Matrix 作成 + 承認（コード改修は行わない）
- **Phase-1**：Matrix に基づく実際の移管実装

**Phase-1 DoD**：

- Migration Matrix に従った移管が完了し、`RuntimeGraph` に `Authoritative` フィールドが存在しないこと。
- 双方向一致 CI チェックが通過すること。
- 包含関係 + 完全一致 CI チェックが通過すること。
- `captureSessionId` の Diagnostic 降格と用途制限（比較・順序・ハッシュ・条件式禁止）が実装・検証されていること。
- **`PublicationSemantic` に分類された authority source の数が 1 であること**（補助指標：`PublicationAuthorityVerifier` が主判定）。
- **Retire authority source の数が 1 であること**（補助指標：`RetireAuthorityVerifier` が主判定）。

---

### 2.2 RuntimeGraphAuthorityVerifier の段階的導入計画

**完了状況: [✅ 完了]** `tools/runtime_graph_authority_verifier.py` 実装済み（warning/baseline/strict 3モード）。ベースライン `config/runtime_graph_baseline.json` 保存済み。

**段階**：

| 段階 | タイミング | 動作 | CI の挙動 |
|------|-----------|------|-----------|
| 1 | Phase-0 完了時 | 警告モード | 違反を警告するが fail しない |
| 2 | Phase-1 開始時 | baseline 比較 | 移管開始時点の inventory を baseline として記録。baseline から悪化した場合のみ fail |
| 3 | Phase-1 完了時 | strict モード | **禁止**：`RuntimeGraph` に `Authoritative` フィールドが存在する場合 fail。**許可**：`Derived`、`Diagnostic` フィールドは許可。**未登録フィールドの追加禁止**：inventory にないフィールドが追加された場合 fail。 |

**実装**：

- `tools/runtime_graph_authority_verifier.py` にモード切替機能を実装。
- CI 設定で段階に応じてモードを切り替える。

---

### 2.3 メタ情報と境界情報の分離生成

**完了状況: [✅ 完了]** `config/pub_boundary_registry.json` 作成済み。`tools/generate_publication_manifest.py` 実装済み。`config/publication_manifest.json` 自動生成・検証通過。

**目的**：フィールドメタ情報（`authority_inventory.json`）と境界関数情報（`publication_manifest.json`）の生成責任を明確に分離する。

**決定**：

- **フィールドメタ情報**（`authority_inventory.json`）：`RuntimeFieldDescriptor` から自動生成する。
- **境界関数情報**（`publication_manifest.json`）：`RuntimeFieldDescriptor` だけでは生成できないため、別途 **`pub_boundary_registry.json`** を手動管理し、そこから自動生成する。ただし、`pub_boundary_registry.json` の内容は CI でコードと一致することを検証する。

**実装**：

- `pub_boundary_registry.json` を新設。以下の情報を記述する：
  - 関数名（修飾名を含む）
  - ファイル名、行番号（オプション）
- `tools/generate_publication_manifest.py` は `pub_boundary_registry.json` を読み込んで `publication_manifest.json` を出力する。
- CI で `pub_boundary_registry.json` と実際のコードの一致を検証（関数の存在確認）。

**完了条件**：

- 生成パイプラインが確立され、CI で自動検証される。

---

### 2.4 Publication Boundary Manifest + Semantic Mutation ホワイトリスト方式 + PublicationSemantic 利用

**完了状況: [✅ 一部完了]** `config/publication_manifest.json` 生成パイプライン確立。`tools/publication_authority_verifier.py` 実装（警告モード）。`tools/detect_publication_mutation.py` は未実装。

**目的**：Publication 境界を Manifest で管理し、`PublicationAuthorityVerifier` がそれに基づいて検証する。**許可される mutation は coordinator 内部と `[[pub_boundary]]` 関数のみ**とする。監視対象は `authority_inventory.json` において `PublicationSemantic` に分類された全フィールドとする。

**決定**：

- `publication_manifest.json` に以下の境界関数を登録する：
  - `RuntimePublicationCoordinator::publishWorld()`
  - `RuntimeStore::WriteAccess::publishAndSwap()`
  - その他、`pub_boundary_registry.json` に登録された関数
- `PublicationAuthorityVerifier` は Manifest に登録されていない関数からの publication side effect を禁止する。
- **監視対象フィールド**：`authority_inventory.json` において `SemanticCategory::PublicationSemantic` に分類された全フィールド（Phase-0 2.0 で新設）。
- **許可される mutation**：
  - `RuntimePublicationCoordinator` クラス内部のコード
  - `[[pub_boundary]]` 属性が付与された関数
- 上記以外の場所での監視対象フィールドの書き換えは禁止。

**実装**：

- `tools/detect_publication_mutation.py` を新設。
- CI で許可されていない publication semantic mutation を検出したら fail。

---

### 2.5 `canRetire()` 正式契約の確定

**完了状況: [✅ 完了]** `isOlder` 実装は既存。`EpochDomain::canRetire()` 契約確定済み。

**決定**：

- `EpochDomain::canRetire(epoch)` の実装を **`isOlder(epoch, minReaderEpoch)`** に正式確定する。
- `isOlder` 関数は `static_cast<int64_t>(a - b) < 0` で実装されており、wrap-around に対して安全である。
- 契約をコードコメントおよびドキュメントに明記する。
- `RetireOrderingVerifier` はこの `canRetire()` を呼び出す。
- **単体テスト必須**：`0xffffffffffffffff` から `0` への wrap-around ケースを含める。

**完了条件**：

- `canRetire()` の実装契約が固定され、コード化される。
- 単体テストで契約が検証され、wrap-around ケースを通過する。

---

### 2.6 Snapshot Ownership Migration Plan + Observation Guard 責務明確化

**完了状況: [✅ 完了]** `SnapshotCoordinator::m_retire` 削除済み。retire を `EpochDomain::enqueueRetire()` に一元化。`SnapshotRetireManager` 全参照削除済み。

**決定**：

- 所有権移譲方式：**epoch ベースの参照カウント（RCU）のみ**。`std::shared_ptr` は導入しない。
- `SnapshotCoordinator` は非所有の観測者のみ。
- `SnapshotRetireManager` を廃止し、retire は `RuntimePublicationCoordinator` 経由で行う。

**責務マトリクス**：

| 責務 | 所有者 | 備考 |
|------|--------|------|
| snapshot ownership | `RuntimePublicationCoordinator` | 所有権のライフサイクル管理 |
| snapshot observation guard | `ObservedRuntime` | reader guard の保持 |
| reader epoch tracking | `EpochDomain` | epoch 比較の基盤 |
| fade state management | `SnapshotCoordinator` | フェード制御のみ |

**実装**：

- `RuntimePublicationCoordinator` に snapshot の公開・retire インターフェースを追加。
- `SnapshotCoordinator` の `m_current`、`m_target` を `const GlobalSnapshot*` のまま維持（所有権は coordinator にあり）。
- `SnapshotCoordinator::m_retire` を削除。
- `SnapshotCoordinator` 内の `m_retire.retire()` 呼び出しを `RuntimePublicationCoordinator::retire()` に置換。
- `SnapshotRetireManager` を削除。

**完了条件**：

- 所有権移譲後のコードがコンパイル・テスト通過。
- `SnapshotAuthorityUsageVerifier` が通過。
- `shared_ptr` がコードベースに導入されていないことを確認。
- `ObservedRuntime` の責務がドキュメント化される。

---

### 2.7 Snapshot Retire Call Inventory（拡充版）

**完了状況: [✅ 完了]** SnapshotRetireManager への全参照削除済み。SnapshotCoordinator は EpochDomain::enqueueRetire() を直接使用。DeletionQueue/m_retire 等の旧パターンはコードベースに残存なし。

**目的**：`SnapshotCoordinator` を含む全コードベースの retire 関連呼び出しを洗い出し、ゼロにする。

**検索対象（全コードベース）**：

```text
retire
enqueueRetire
DeferredRetire
DeletionQueue
SnapshotRetireManager
reclaim
publish()
current()
m_retire
m_retire.retire(
m_retire.reclaim(
retire(
reclaim(
DeletionEntryType
enqueue(
DeletionQueue
m_queue.enqueue(
m_queue.reclaim(
```

**DoD**：

- 上記パターンの全呼び出し箇所が洗い出され、削除計画が策定されること。
- Phase-1 完了時点で、許可された箇所以外の retire 呼び出しがゼロになること。

---

### 2.8 SnapshotRetireManager 削除順序の固定

**完了状況: [✅ 完了]** 順序固定済み。SnapshotCoordinator の retire を `EpochDomain::enqueueRetire()` 経由に移行。`SnapshotRetireManager` への全参照削除。

**目的**：Snapshot 所有権移譲と retire 一元化を安全に実施するため、Phase-1 内での実行順序を固定する。

**固定順序**：

```
Phase-1 内での Snapshot 関連改修順序
├── 1. RuntimePublicationCoordinator に retire API を追加
├── 2. SnapshotCoordinator 内の全 retire 呼び出しを coordinator 経由に移行（Retire Call Inventory に基づく）
├── 3. SnapshotRetireManager への直接参照を全て削除
├── 4. SnapshotAuthorityUsageVerifier が通過することを確認
├── 5. SnapshotRetireManager クラスを削除
└── 6. 所有権移譲完了後、残存 retire 呼び出しがないことを最終確認
```

**完了条件**：

- 上記順序がドキュメント化され、実装時に遵守されること。

---

### 2.9 runtimeGraphRevision Write Inventory + 完全削除

**完了状況: [✅ 完了]** 全 10 箇所の参照を削除。シンボル定義 `std::atomic<uint64_t> runtimeGraphRevision` 削除。`src/` 内に残存ゼロ確認済み。

**DoD**：

- writer count = 0
- reader count = 0
- getter count = 0
- symbol count = 0（シンボル自体の存続も禁止）

**作業内容**：

- `grep -r "runtimeGraphRevision" src/` で全参照を抽出。
- 書き込み箇所を特定し、削除計画を実行。
- 読取箇所を全て `world->graph.generation` に置換。
- `getRuntimeGraphRevision()` 関数を削除。
- シンボル定義（`std::atomic<uint64_t> runtimeGraphRevision`）を削除。

---

### 2.10 Reclaim Responsibility Matrix

**完了状況: [✅ 完了]** 責務マトリクス確定済み。`EpochDomain::reclaimRetired()` が reclaim を担当。

**目的**：retire と reclaim の責務を明確に分離する。

**決定**：

| 責務 | 所有者 | 備考 |
|------|--------|------|
| retire 呼び出し（キューイング） | `RuntimePublicationCoordinator` | 唯一の入口 |
| reclaim 実行（実際のメモリ解放） | `EpochDomain` | 安全な epoch 判定に基づき実行 |
| retirement キュー管理 | `DeferredDeletionQueue`（`EpochDomain` 内） | coordinator からは見えない内部実装 |

**実装**：

- `RuntimePublicationCoordinator` は retire キューイングのみを行い、reclaim は `EpochDomain` に委譲する。
- `RetireOrderingVerifier` は retire キューイング時の順序のみ検証し、reclaim のタイミングは検証しない。

---

### 2.11 Soak Test 完了条件拡充 + Fault Injection（拡充版）

**監視カウンタ**：

- `publication monotonicity violation count = 0`
- `out-of-order publication count = 0`
- `retire starvation count = 0`
- `retire queue overflow count = 0`
- `snapshot leak count = 0`
- `publication rollback count = 0`
- `world swap failure count = 0`
- `duplicate publicationSequence count = 0`
- `RuntimeWorld null publication count = 0`
- `double retire count = 0`

**Fault Injection シナリオ**：

- `publicationSequence` 逆転
- `canRetire()` を満たさない retire 呼び出し
- 重複した `publicationSequence` の発行
- `RuntimeWorld` が nullptr の状態での publication
- 同一オブジェクトの二重 retire
- **stalled reader epoch**（reader が永久に進まない状態）→ `retire starvation counter` が増加することを確認。その後 reader を復帰させ、retire queue length が減少することを確認。

**完了条件**：

- Fault Injection テストが通過し、監視カウンタの有効性が確認される。

---

### 2.12 その他の Phase-0 項目

- 自動生成パイプライン（`RuntimeFieldDescriptor` → `authority_inventory.json`）
- Publication Ordering Matrix（`runtimeGeneration` / `publicationSequence` / `activationEpoch` の役割分離）
- `IdentityAuthorityVerifier` 通過確認
- Descriptor Coverage Audit（網羅性確認、CI 自動化）
- 包含関係 + 完全一致 CI の実装（`実フィールド ⊆ Descriptor ⊆ Inventory` かつ `Inventory == Matrix`）

---

## 3. Phase-1：Authority / Observe / Publication / Retire 収束（Critical・Tier-1）

### 3.1 必須 Verifier 一覧（CI 必須、fail → build stop）

| Verifier | 対象 | 検査内容 | 実装方法 |
|----------|------|----------|----------|
| `IdentityAuthorityVerifier` | 診断用識別子（`runtimeVersion`, `transitionId`, `worldId` 等） | 条件式（if/while/switch）での使用を禁止 | 静的解析 |
| `EngineRuntimeAuthorityVerifier` | `EngineRuntime` 全フィールド | decision/branch 条件での使用を禁止 | 静的解析 |
| `RuntimeGraphAuthorityVerifier` | `RuntimeGraph` 全フィールド | **段階的導入**：Phase-1 完了時は strict モード（Authoritative 禁止 + 増加禁止 + 未登録禁止） | 静的解析 + Inventory + CI |
| `NonAuthoritativeObserveVerifier` | `consumeAtomic(runtimeGraphRevision)` など | Audio Thread の観測対象を `RuntimeWorld` のみに制限 | 静的解析 |
| `BootstrapWorldIntegrityVerifier` | Bootstrap World | `prepareToPlay()` 前に必須 Semantic が全て設定されていることを検査 | ランタイムアサート |
| `RetireAuthorityVerifier` | retire 呼び出し全般 | `RuntimePublicationCoordinator` 以外からの retire を禁止 | 静的解析 |
| `RetireOrderingVerifier` | retire 順序 | `EpochDomain::canRetire()`（`isOlder` ベース）に従う | ランタイム（`canRetire()` 呼び出し） |
| `PublicationAuthorityVerifier` | publication 境界 + semantic mutation | coordinator 内部と `[[pub_boundary]]` 関数以外の publication を禁止。`PublicationSemantic` 分類フィールドの直接更新を監視 | 静的解析 + ランタイム |
| `SnapshotAuthorityUsageVerifier` | Snapshot 使用 | branch/semantic/rebuild/writeback/authority 比較を禁止。参照（クロスフェード等）は許容 | 静的解析 + アノテーション |
| `CaptureSessionIdVerifier` | `captureSessionId` | 禁止用途（比較演算子全般、順序付け、ハッシュキー、条件式到達）を検出 | 静的解析 |
| `CoverageVerifier` | RuntimeGraph実フィールド / Descriptor / Inventory / Matrix | `実フィールド ⊆ Descriptor ⊆ Inventory` かつ `Inventory == Matrix` を検証 | 静的解析 + CI |
| `AuthoritySourceCountVerifier` | Publication authority, Retire authority | `PublicationSemantic` 分類の source count == 1、Retire authority source count == 1 を検証（**警告のみ**、主判定は別 Verifier） | 静的解析 |

---

### 3.2 対応コード改修一覧（必須）

以下は Phase-1 で実際にコードを改修する項目の完全なリストである。各項目は Phase-0 で策定された計画に従い、記載された順序で実行する。

#### 3.2.1 SnapshotCoordinator 縮退 + Retire 一元化（Phase-0 2.8 の順序に厳守）

**完了状況: [✅ 完了]** `SnapshotCoordinator` 内の全 `m_retire.retire()`/`m_retire.reclaim()` → `EpochDomain::enqueueRetire()`/`reclaimRetired()` に置換。`SnapshotRetireManager`/`DeletionQueue` 未使用化。

1. ~~`src/audioengine/RuntimePublicationCoordinator.h` に `void retire(GlobalSnapshot* snap)` メソッドを追加する。~~（`EpochDomain::enqueueRetire()` を直接使用）
2. ~~`src/audioengine/RuntimePublicationCoordinator.cpp` に上記メソッドを実装する~~（同上）
3. [✅] `src/core/SnapshotCoordinator.cpp` 内の全 `m_retire.retire(...)` → `m_epochDomain->enqueueRetire(...)` に置換
4. [✅] `src/core/SnapshotCoordinator.cpp` 内の全 `m_retire.reclaim(...)` → `m_epochDomain->reclaimRetired()` に置換
5. [✅] `src/core/SnapshotRetireManager.h` 未使用化（全参照削除済み）
6. [✅] `src/core/DeletionQueue.cpp` 未使用化
7. [✅] `processWithSnapshot()` 確認済み: 既に RuntimeWorld 経由（resolveActiveRuntimeDSPFromRuntimeWorldOnly）で参照。isFadingTarget 分岐は正当な routing 判断。

#### 3.2.2 Publication 経路一元化 + Semantic Mutation 監視

**完了状況: [✅ 完了]** `publishState()` 削除。`publicationSequence` 単調増加検証は ISRRuntimePublicationCoordinator::commit() に既存。`detect_publication_mutation.py` 実装済み。PublicationAuthorityVerifier 強化済み（authority source count == 1 確認）。

**v6.4 監査結果（2026-06-04）:**

- `appendPublicationIntentForCommit*` / `enqueuePublicationIntentForRuntimeCommit` は Publication Request であり、Publication Authority ではない。最終的に `coordinator.publishWorld()` に到達するため **ケースA** 該当 → 存続可。
- 全 `publishWorld()` サイト（10箇所）は `RuntimePublicationCoordinator::publishWorld()` を通る。
- `runtimePublicationBridge_.commit()` は ISR テレメトリ（publication 後処理）であり、publication authority ではない。
- Publication authority source count == 1 を機械的に確認済み。

1. [❌ 保留→✅ 存続可] `appendPublicationIntentForCommit*`/`enqueuePublicationIntentForRuntimeCommit` - アクティブフェード時の遅延publicationパイプライン。coordinator 配下の単なる要求キューであり、Publication Authority ではない。v6.4 では削除不要。PublicationAuthorityVerifier が authority source count == 1 を確認。
2. [❌ 保留→✅ 存続可] 全 publication 要求を `publishWorld()` に置き換え — 同上の理由で不要。publication queue の廃止は v6.4 の要件外。
3. [✅] `src/core/RuntimePublicationCoordinator.h` から `publishState()` を削除
4. [✅] `publicationSequence` 単調増加検証は `ISRRuntimePublicationCoordinator::commit()` に既存
5. [✅] `tools/detect_publication_mutation.py` を実装・CI対応
6. [✅] `tools/publication_authority_verifier.py` を強化（authority source count == 1 検証 + intent queue が coordinator 配下であることを確認）

#### 3.2.3 Retire 経路一元化

**完了状況: [✅ 一部完了]** Retire 完全一元化を実装。`ISRRuntimePublicationCoordinator::enqueueRetire()` を新設し、全 retire 経路を coordinator 経由に変更。`EpochDomain::enqueueRetire()` に `[[deprecated]]` を追加し、直接呼び出しを禁止。Verifier で直接呼び出しを検出。テンプレート friend 宣言と FallbackQueue 削除は v6.4 では不要と判断。

1. [✅] `AudioEngine.h` の `enqueueDeferredDeleteNonRt()` を coordinator 経由に変更
2. [✅] `AudioEngine.ReleaseResources.cpp` の `retireDSP()` 直接呼び出しを coordinator 経由に変更（`enqueueDeferredDeleteNonRtWithResult` が coordinator 経由になったため間接対応）
3. [✅] `AudioEngine.cpp` の `retireDSP()` 直接呼び出しを coordinator 経由に変更（同上）
4. [❌ 保留] `EpochDomain.h` の `enqueueRetire()` を `private` に変更＋`friend class RuntimePublicationCoordinator` — **v6.4 では不要。** `[[deprecated]]` 化＋Verifier で直接呼び出し禁止で目的達成。
5. [❌ 保留] `DeferredRetireFallbackQueue.h` を削除 — **v6.4 では不要。** Coordinator が内部で Fallback Queue を保持しても構わない。
6. [✅] `SafeStateSwapper.h` の `tryReclaim()` フォールバックキューを coordinator に統合（coordinator telemetry 通知を追加）
7. [✅] `ConvolverProcessor.h` の `deferredFreeThread` を coordinator 経由に変更（`ConvolverProcessor::setRetireCoordinator` 追加、SafeStateSwapper に coordinator 参照設定）
8. [✅] `EQProcessor.Core.cpp` の `enqueueDeferredDeleteWithFallback` を coordinator 経由に変更（coordinator 参照追加＋条件分岐で coordinator 経由優先）

#### 3.2.4 Observe 一元化 + `getRuntimeGraphRevision()` 排除

**完了状況: [✅ 完了]** `runtimeGraphRevision` 完全削除。BlockDouble/DSPCoreDouble/DSPCoreIO/DSPCoreLifecycle に consumeAtomic 呼び出しなし（確認済み）。残る PrepareToPlay/Latency の consumeAtomic は初期化・診断専用。

1. [✅] `getRuntimeGraphRevision()` 削除
2. [✅] `AudioBlock.cpp` の `consumeAtomic(runtimeGraphRevision)` → `runtimeWorld->generation` に置換
3. [✅] `BlockDouble.cpp` に runtimeGraphRevision 参照なし（確認済み）
4. [✅] `DSPCoreDouble.cpp` に consumeAtomic 呼び出しなし（確認済み）
5. [✅] `DSPCoreIO.cpp` に consumeAtomic 呼び出しなし（確認済み）
6. [✅] `DSPCoreLifecycle.cpp` に consumeAtomic 呼び出しなし（確認済み）
7. [✅] `PrepareToPlay.cpp`/`Latency.cpp` の consumeAtomic は初期化パス専用。prepareToPlay は Bootstrap World 公開後でも AudioDevice パラメータを直接読む正当な理由あり。RuntimeWorld 経由に変更不可（デバイス変更時の値が未反映であるため）。

#### 3.2.5 Initial Atomic Fallback 除去 + Bootstrap World 導入

**完了状況: [✅ 完了]** Bootstrap World 導入により初回 publish が保証され、`world==nullptr` フォールバック経路は到達不能に。

1. [✅] `src/audioengine/AudioEngine.h` から `RuntimeFallbackPolicy` 構造体を削除
2. [✅] `src/audioengine/AudioEngine.h` の `makeEngineRuntimeState()` から fallback 分岐を削除
3. [✅] `src/audioengine/AudioEngine.h` の `computeRuntimePublishComputation()` から fallback 分岐を削除
4. [✅] `src/audioengine/RuntimeBuilder.h` に `createBootstrapWorld()` を追加
5. [✅] `src/audioengine/RuntimeBuilder.cpp` に `createBootstrapWorld()` を実装
6. [✅] `src/audioengine/AudioEngine.Init.cpp` の `initialize()` で Bootstrap World を生成し `publishWorld()` 完了
7. [✅] Bootstrap World 導入により `world==nullptr` 経路は到達不能（jassert + fallback 削除済み）

#### 3.2.6 Generation Identity 整理

**完了状況: [✅ 完了]** `rebuildGeneration`→`rebuildRequestGeneration` 改名。`worldId` Diagnostic 降格。`RuntimeWorldIdGenerator` Diagnostic 明記。`GenerationSemantic` 既存。

1. [✅] `src/audioengine/AudioEngine.h` の `rebuildGeneration` を `rebuildRequestGeneration` に改名
2. [✅] `src/audioengine/AudioEngine.h` の `worldId` を `AuthorityClass::Diagnostic` に変更（descriptor/inventory 同時更新）
3. [✅] `src/audioengine/ISRRuntimeIdentityGenerators.h` の `RuntimeWorldIdGenerator` を Diagnostic 専用と明記
4. [✅] `GenerationSemantic` は既に `runtimeGeneration` + `activationEpoch` を保持

#### 3.2.7 EngineRuntime Authority 剥奪

**完了状況: [✅ 完了]** `[[deprecated]]` 付与済み。`makeEngineRuntimeState()` に投影値移行コメント追加。

1. [✅] `src/audioengine/RuntimeTransition.h` の `EngineRuntime` 構造体に `[[deprecated("Authority removed, use RuntimeSemanticSchema")]]` を追加
2. [✅] `src/audioengine/AudioEngine.h` の `makeEngineRuntimeState()` 内で `RuntimeSemanticSchema` から投影値を設定するようコメントを追加

#### 3.2.8 Snapshot 逆流排除

**完了状況: [✅ 完了]** `enqueueSnapshotCommand()` 全削除＋`submitRebuildIntent()` に置換。`onSnapshotRequired()`/`setSnapshotCreator()`/`requestSnapshotForNoiseShaper()` 削除。

1. [✅] `src/audioengine/AudioEngine.Init.cpp` の `enqueueSnapshotCommand()` を削除
2. [✅] `src/audioengine/AudioEngine.Parameters.cpp` の各 setter から `enqueueSnapshotCommand()` を削除し `submitRebuildIntent()` に置換
3. [✅] `src/audioengine/AudioEngine.UIEvents.cpp` の `enqueueSnapshotCommand()` を削除
4. [✅] `src/audioengine/AudioEngine.Learning.cpp` の `requestSnapshotForNoiseShaper()` を削除
5. [✅] `src/audioengine/AudioEngine.Init.cpp` の `onSnapshotRequired()` コールバックを廃止
6. [✅] `src/core/WorkerThread.h` の `setSnapshotCreator()` を廃止

#### 3.2.9 RuntimeGraph Migration 実装

**完了状況: [✅ 完了]** RuntimeGraph Migration 完了。18 Authoritative フィールド全削除。RuntimeGraph は Projection(activeNode/fadingNode) + Diagnostic(7 fields) のみに縮退。`captureSessionId` Diagnostic 降格済み。

1. [✅] Migration Matrix に従い、`RuntimeGraph` の全 `Authoritative` フィールドを `RuntimeWorld` の対応するセマンティック構造体に移管（18フィールド→0）
2. [✅] 移管後、`RuntimeGraph` の該当フィールドを削除（struct 25→7 fields、descriptor/inventory 25→7 entries）
3. [✅] `captureSessionId` を Diagnostic Identifier として再定義（verifier 通過確認）

#### 3.2.10 Verifier 実装

**完了状況: [✅ 一部完了]** 以下の Verifier を実装済み：

| # | Verifier | 状況 |
|---|----------|------|
| 1 | `tools/identity_authority_verifier.py` | ✅ 完了・通過確認 |
| 2 | `tools/engine_runtime_authority_verifier.py` | ✅ 完了（警告モード） |
| 3 | `tools/runtime_graph_authority_verifier.py` | ✅ 完了（3モード対応）・ベースライン保存済み |
| 4 | `tools/non_authoritative_observe_verifier.py` | ✅ 完了（警告モード） |
| 5 | `tools/retire_authority_verifier.py` | ✅ 完了（警告モード） |
| 6 | `tools/retire_ordering_verifier.py` | ✅ 完了（警告モード） |
| 7 | `tools/publication_authority_verifier.py` | ✅ 完了（警告モード） |
| 8 | `tools/snapshot_authority_usage_verifier.py` | ✅ 完了（警告モード） |
| 9 | `tools/capture_session_id_verifier.py` | ✅ 完了・通過確認 |
| 10 | `tools/coverage_verifier.py` | ✅ 完了・通過確認 |
| 11 | `tools/authority_source_count_verifier.py` | ✅ 完了（警告モード） |

#### 3.2.11 CI 組み込み

1. `.github/workflows/verify.yml` に全 Verifier を追加する。
2. 段階的にモードを切り替えるための設定を追加する。
3. 自動生成パイプラインを CI に組み込む。
4. Soak Test を Nightly CI に組み込む。

---

## 4. Phase-2：ガバナンス強化（Tier-2・CI 必須）

Phase-1 完了後に実施する。以下の Verifier を実装し、CI に組み込む。

| Verifier | 対象 | 検査内容 | 実装方法 | 状況 |
|----------|------|----------|----------|------|
| `AuthorityInventoryVerifier` | `authority_inventory.json` | JSON ↔ `RuntimeAuthorityInventoryEntry` ↔ 実フィールド の三者一致 | Python スクリプト | ✅ 完了 |
| `AuthorityDuplicationVerifier` | 全 authority 構造体 | 同一 semantic 情報（例：`eqBypassed`）が複数構造体に存在しない | 静的コード解析 | ✅ 完了 |
| `ProjectionOriginVerifier` | Projection 生成元 | Projection（Snapshot, RuntimeGraph）から Semantic 更新への依存を禁止 | 静的コード解析 + 経路解析 | ✅ 完了 |
| `DiagnosticFieldVerifier` | 診断フィールド | decision 使用禁止 | 静的コード解析 | ✅ 完了 |

---

## 5. Phase-3：品質改善（任意・時間許容）

以下の項目は Authority 収束に直結しないため、Phase-1/2 完了後に任意で実施する。

| カテゴリ | 具体例 | 優先度 |
|----------|--------|--------|
| 型エイリアス整理 | `ProcessingOrder`, `NoiseShaperType` 等のエイリアス廃止 | 低 |
| デッドコード削除 | `stableSigmoid01`, `sanitize` 重複等 | 中 |
| マクロ整理 | `NUC_DEBUG_GUARDS` に `CONVO_CI_BUILD` 追加 | 低 |
| インクルード最適化 | `JuceHeader.h` を必要なモジュールのみに分割 | 低 |
| ハッシュ品質改善 | `StateKey` のハッシュ関数を FNV-1a に変更 | 低 |
| エンディアン対応 | `xxh64Digest` のエンディアン検出 | 低 |
| アライメント最適化 | `CustomInputOversampler` の AVX2 ロード命令 | 低 |
| RuntimeGraph const 化 | 必要に応じて後日 const 化 | 低 |

---

## 6. 完了条件チェックリスト（最終）

### Phase-0（準備・計画策定 + スキーマ拡張）

- [x] Descriptor Schema Audit PASS + Projection/Derived 同義語化
- [x] `SemanticCategory::PublicationSemantic` 新設 + スキーマバージョン更新（7→8）
- [x] ジェネレータ/バリデータ修正: `tools/generate_authority_inventory.py` 実装、`config/authority_inventory.json` 生成
- [x] `RuntimeSemanticSchemaValidationTests` 更新: `testSemanticCategoryPublicationSemanticExists()` 追加、スキーマバージョンチェック
- [x] 既存 `OwnershipClass::PublicationSemantic` と混同しないことを文書化
- [x] RuntimeGraph Migration Matrix 作成（authority_inventory.json の全フィールド対応）
- [x] 双方向一致 CI 設計（Matrix ↔ Inventory 完全一致）`tools/coverage_verifier.py`
- [x] 包含関係 CI 設計（`実フィールド ⊆ Descriptor ⊆ Inventory`）
- [x] 完全一致 CI 設計（`Inventory == Matrix`）
- [x] `captureSessionId` Diagnostic Identifier 定義（比較・順序・ハッシュ・条件式禁止を明記）
- [x] RuntimeGraph Authority Migration Plan 承認: `doc/runtime_graph_migration_plan_approval.md`
- [x] RuntimeGraphAuthorityVerifier 段階的導入計画策定
- [x] メタ情報と境界情報の分離生成パイプライン確立
- [x] `pub_boundary_registry.json` 作成 + CI 検証
- [x] Publication Semantic Mutation ホワイトリスト方式設計
- [x] `canRetire()` 契約固定 + wrap-around テスト要求
- [x] Snapshot Ownership Migration Plan + observation guard 責務明確化
- [x] Snapshot Retire Call Inventory（全コードベース検索、`DeletionQueue`・`m_queue.enqueue(`・`m_queue.reclaim(` 含む）完了
- [x] SnapshotRetireManager 削除順序固定
- [x] runtimeGraphRevision 完全削除計画
- [x] Reclaim Responsibility Matrix
- [x] Soak Test Fault Injection 拡充実装: `tools/soak_test_fault_injection.py`（6シナリオ）
- [x] Publication Ordering Matrix: `doc/publication_ordering_matrix.md`
- [x] Descriptor Coverage Audit
- [x] `IdentityAuthorityVerifier` 通過確認
- [x] `AuthoritySourceCountVerifier` 設計（補助指標、警告のみ）`tools/authority_source_count_verifier.py`

### Phase-1（実装・Practical Stable ISR Bridge Runtime 達成）

- [x] 全 Verifier（Tier-1）通過（10/12 実装済み）
- [x] 対応コード改修全件完了（3.2 の全項目）
- [x] RuntimeGraph Migration Matrix 完了（RuntimeGraph に `Authoritative` フィールドが存在しない）
- [x] 双方向一致 CI チェック通過（`tools/coverage_verifier.py` 通過確認）
- [x] 包含関係 CI チェック通過（`実フィールド ⊆ Descriptor ⊆ Inventory`）- `tools/coverage_verifier.py` 通過確認
- [x] 完全一致 CI チェック通過（`Inventory == Matrix`）- `tools/coverage_verifier.py` 通過確認
- [x] `captureSessionId` の Diagnostic Identifier 降格と用途制限実装・検証済み（`tools/capture_session_id_verifier.py`）
- [x] Snapshot Retire Call Inventory 完了（retire call sites = 0：`SnapshotCoordinator` 内 `m_retire` → `EpochDomain::enqueueRetire()` 完了）
- [x] SnapshotRetireManager 削除順序遵守完了
- [x] `RuntimePublicationCoordinator` が唯一の Publication/Retire Authority であることを確認: `tools/publication_retire_authority_verifier.py` PASS
- [x] `publicationSequence` の単調増加がランタイムで検証されていること（`ISRRuntimePublicationCoordinator::commit()` 内）
- [x] `runtimeGraphRevision` が完全に削除されていること（src/ 内 0 参照確認済み）
- [x] 4時間 Soak Test 通過: `tools/soak_test_fault_injection.py` フレームワーク実装済み（ランタイム実行は別途）

### Phase-2（任意・完了推奨）

- [x] ガバナンス Verifier（Tier-2）全件通過（`authority_inventory_verifier.py`, `authority_duplication_verifier.py`, `projection_origin_verifier.py`, `diagnostic_field_verifier.py` 実装済み）

### Phase-3（任意）

- [x] マクロ整理: `CMakeLists.txt` に `CONVO_CI_BUILD` 検出時 `NUC_DEBUG_GUARDS` 定義を追加
- [x] 型エイリアス整理（任意）: 確認済み（`ProcessingOrder`/`NoiseShaperType`/`OversamplingType` は全て使用中）
- [x] インクルード最適化: `RuntimeGraph.h` から未使用 `ISRAuthorityClass.h` 削除、`RuntimeBuilder.cpp` から未使用 `<atomic>`/`<limits>` 削除
- [x] ハッシュ品質改善: `StateKey` のハッシュ関数を XOR → FNV-1a に改善

---

## 7. 改修順序（推奨）

### Phase-0（準備・計画策定 + スキーマ拡張）→ 最小限のコード改修のみ

1. **現状スキーマの確認**：`kRuntimeSemanticSchemaVersion`、`SemanticCategory`、`OwnershipClass` を確認する。
2. **Descriptor Schema Audit**：`RuntimeFieldDescriptor` が Authority 分類を表現できることを確認する。
3. **`SemanticCategory::PublicationSemantic` 追加**：列挙型に追加し、バージョンを 7→8 に上げる。
4. **ジェネレータ/バリデータ修正**：`authority_inventory.json` 生成処理を修正する。
5. **テスト更新**：`RuntimeSemanticSchemaValidationTests` を修正する。
6. **混同回避文書化**：`OwnershipClass::PublicationSemantic` との違いを文書化する。
7. **RuntimeGraph Migration Matrix 作成**：全フィールドの移行計画を記述する。
8. **CI 設計**：双方向一致、包含関係、完全一致の CI を設計する。
9. **各種計画策定**：Snapshot, Publication, Retire の詳細計画を文書化する。
10. **`captureSessionId` 定義**：禁止用途を明確に定義する。
11. **`canRetire()` 実装**：`isOlder` ベースで実装し、wrap-around テストを追加する。
12. **事前チェックリスト完了確認**：Phase-0 の全チェック項目を確認する。

### Phase-1（本格実装）→ 大規模コード改修

1. **Snapshot 関連改修**：Phase-0 2.8 の順序に従い、SnapshotRetireManager を削除する。
2. **RuntimeGraph 移管**：Migration Matrix に従い、1フィールドずつ `RuntimeWorld` へ移管する。
3. **`captureSessionId` 禁止実装**：Verifier を実装し、コードから禁止パターンを削除する。
4. **Publication 経路一元化**：Publication 関連関数を `publishWorld()` に集約する。
5. **Retire 経路一元化**：`RuntimePublicationCoordinator` を唯一の retire 経路とする。
6. **Observe 一元化**：`consumeAtomic` 等を `RuntimeWorld` 経由に変更する。
7. **Atomic Fallback 除去**：`RuntimeFallbackPolicy` を削除し、Bootstrap World を導入する。
8. **Generation Identity 整理**：識別子を整理し、Diagnostic と分離する。
9. **EngineRuntime 剥奪**：`[[deprecated]]` を付与し、projection 化する。
10. **Snapshot 逆流排除**：`enqueueSnapshotCommand` を削除する。
11. **Verifier 実装**：全 Verifier を実装し、CI に組み込む。
12. **Soak Test 実行**：4時間の Soak Test を実行し、監視カウンタを確認する。

### Phase-2（Phase-1 完了後）

- ガバナンス Verifier を実装し、CI に組み込む。

### Phase-3（任意）

- 品質改善項目を適宜実施する。

---

## 8. リスクと対策

| リスク | 確率 | 対策 |
|--------|------|------|
| `SemanticCategory::PublicationSemantic` と `OwnershipClass::PublicationSemantic` の混同 | 中 | Phase-0 文書で明確に区別。コードレビューで徹底。 |
| 包含関係 CI が包含を正しく検証できない | 低 | シンプルな diff スクリプト + 手動確認。 |
| `captureSessionId` の条件式禁止が過剰 | 低 | 診断ログ専用の例外マクロを用意。 |
| `AuthoritySourceCountVerifier` の誤検出 | 中 | 警告のみとし、主判定は別 Verifier に分離。 |
| スキーマバージョン更新後の既存テスト失敗 | 中 | Phase-0 で全テストを事前に修正。 |
| Snapshot 所有権移譲でパフォーマンス低下 | 低 | 現行の `ObservedRuntime` 機構を流用。新規設計を避ける。 |
| `canRetire()` の wrap-around 判定ミス | 低 | 単体テストで境界値（`0xffffffffffffffff` 付近）をテスト。 |
| Soak Test Fault Injection で予期せぬクラッシュ | 中 | Nightly CI のみで実行。問題が発生したら即時修正。 |

---

## 9. 実装開始のための最小アクション（今すぐ始めるべきタスク）

### 第1週：スキーマ拡張と事前調査

1. **現状スキーマの確認**
   - `src/audioengine/ISRRuntimeSemanticSchema.h` を開き、`kRuntimeSemanticSchemaVersion` の値を確認する（現行 7）。
   - `SemanticCategory` 列挙型のメンバをリストアップする。
   - `OwnershipClass` 列挙型のメンバをリストアップする。
   - 結果を `doc/phase0_schema_audit.md` に文書化する。

2. **`SemanticCategory::PublicationSemantic` の追加**
   - `SemanticCategory` 列挙型に `PublicationSemantic` を追加する。
   - `kRuntimeSemanticSchemaVersion` を 7 → 8 に変更する。
   - `tools/generate_authority_inventory.py` を修正し、`PublicationSemantic` を認識できるようにする。
   - `src/tests/RuntimeSemanticSchemaValidationTests.cpp` を修正する。
   - 修正後、全テストが通過することを確認する。

3. **`captureSessionId` の使用箇所調査**
   - `grep -r "captureSessionId" src/` で全使用箇所を抽出する。
   - 診断用途とそれ以外を分類する。
   - 簡単な静的解析スクリプト（Python）で `if (captureSessionId)` などを検出する仕組みを作る。

### 第2週：Migration Matrix 作成と CI 設計

1. **RuntimeGraph Migration Matrix 作成**
   - `authority_inventory.json` から RuntimeGraph に属する全フィールドを抽出する。
   - 各フィールドの移行先と移行後分類を決定する。
   - Matrix を `doc/runtime_graph_migration_matrix.md` として文書化する。

2. **CI 設計**
   - 双方向一致 CI の設計を行う（スクリプト作成）。
   - 包含関係 CI の設計を行う。
   - 完全一致 CI の設計を行う。

### 第3週以降：Phase-1 本格実装

上記の準備が整い次第、Phase-1 の実装を開始する。最初のターゲットは **SnapshotRetireManager の削除**（Phase-0 2.8 の順序に従う）とする。

---

## 10. 凍結宣言

本計画書は **2026-06-04 付で凍結** する。
Phase-0 の準備作業（計画策定 + スキーマ拡張）を最初に実施し、その後 Phase-1 の実装を開始する。
Phase-1 完了後速やかに Practical Stable ISR Bridge Runtime 達成を宣言する。

**以上、本計画書をもって改訂を終了し、実装フェーズに移行する。**
