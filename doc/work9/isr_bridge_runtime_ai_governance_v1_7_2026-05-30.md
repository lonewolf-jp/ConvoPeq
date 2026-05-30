# Practical Stable ISR Bridge Runtime AI実装統治規約 v1.7（rev13準拠・1ファイル完結）

作成日: 2026-05-30

適用対象:

- `doc/work9/isr_bridge_runtime_implementation_tasks_phase_file_2026-05-30_rev13.md`

---

## 1. 本規約の位置付け

本規約は、rev13 を前提に AI が ISR Bridge Runtime 改修を行う際の最上位統治規約である。

AI は以下を行ってはならない。

- 推測実装
- 独自解釈による仕様追加
- アーキテクチャ変更（rev13 非記載）
- 未確認推定による修正

rev13 に明記がない事項は、現行実装維持を原則とする。

### 1.1 未確認推定修正の禁止（最重要）

以下を確認せずに修正してはならない。

- caller 未確認修正
- callee 未確認修正
- 到達経路未確認修正
- ownership 未確認修正
- lifecycle 未確認修正

不明点がある場合は、推定実装ではなく調査を優先すること。

---

## 2. 規範語（拘束レベル）

- MUST: 必須。違反時は完了宣言不可。
- SHOULD: 強く推奨。未実施時は理由記載必須。
- MAY: 任意。

本規約の禁止条項はすべて MUST とする。

---

## 3. 実装前義務

### 3.1 全コード読了義務（実行可能版）

AI は「対象領域に到達可能な関連コード」を読了してから実装開始すること。

最低対象:

- Runtime
- Snapshot
- Publish
- Retire
- Transition
- RuntimeStore
- RuntimeGraph
- RuntimeExecutionDescriptor
- RuntimeInstance
- DSP Runtime

注記:

- リポジトリ全ファイル読了は要求しない。
- 影響範囲に到達可能な関連コード読了を MUST とする。

### 3.2 依存関係調査義務（一覧提出必須）

修正対象クラス/関数について、実装前に以下を列挙すること。

- direct callers
- indirect callers
- direct callees
- indirect callees
- 保持関係
- ownership 関係
- ライフサイクル

「調査済み」の宣言のみは不可。実一覧を提出すること。

### 3.3 indirect caller/callee 定義固定

indirect caller/callee の調査は曖昧運用を禁止し、以下を最小要件とする。

- `RuntimeWorldAuthority` から修正対象シンボルまでの到達経路を全列挙
- 修正対象シンボルから `RuntimeWorldAuthority` / `RuntimeStore` / `RuntimeRetireCoordinator` へ戻る主要経路を全列挙
- virtual dispatch / interface 実装経由 / template インスタンス経由を含める

### 3.4 旧経路調査義務

変更対象に対して、以下の全検索を行うこと。

- class 名
- struct 名
- typedef/using 名
- public API 名
- private API 名

調査結果未確認で実装開始してはならない。

### 3.5 MCP静的解析義務（必須・条件明文化）

AI は grep 単独で影響判断してはならない。

- Serena と CodeGraph が利用可能な環境では、両方とも MUST
- 利用不可の場合のみ、利用不能理由を明記し、grep/ripgrep 結果を代替提出すること

最低対象シンボル:

- `RuntimeWorld`
- `RuntimeWorldAuthority`
- `RuntimeStore`
- `RuntimePublicationCoordinator`
- `RuntimeRetireCoordinator`
- `RuntimeGenerationGenerator`
- `RuntimeWorldIdGenerator`
- `RuntimeSnapshot`
- `WorldBuilder`

### 3.5.1 到達型追跡義務

MCP解析で発見された関連型は、最低対象シンボル一覧に存在しなくても調査対象へ追加しなければならない。

対象:

- メンバ型
- 戻り値型
- 引数型
- 継承型
- Interface 型
- Container 保持型

### 3.6 MCP結果提出義務

MCP解析を実施した場合、以下を影響分析レポートへ記載すること。

- 参照数
- caller 数（direct/indirect）
- callee 数（direct/indirect）
- 到達経路（主要経路の列挙）

「MCPを使ったが結果未提示」は不合格とする。

### 3.6.1 MCP証跡保存義務

Serena / CodeGraph の主要結果は影響分析レポートへ添付すること。

- 要約のみは禁止
- 主要結果（参照一覧、caller/callee一覧、経路結果）を再検証可能な形で保存する

### 3.7 影響範囲確定義務（和集合固定）

影響範囲一覧は以下の和集合で作成すること。

- grep/ripgrep 検出ファイル
- Serena 検出ファイル
- CodeGraph 検出ファイル
- 実装中に新規発見した関連ファイル

いずれかで検出されたファイルを除外してはならない。MCP未検出を理由に除外してはならない。

影響範囲一覧なしで実装開始不可。

### 3.7.1 修正対象確定義務

修正対象シンボルは AI が任意に決定してはならない。

- 影響範囲解析の結果、rev13 受け入れゲートへ影響するシンボルは修正対象候補として列挙すること
- 候補から除外する場合は除外理由を記録すること

### 3.7.2 rev13本文拘束規約

受け入れゲートのみを適合対象としてはならない。

rev13 本文に記載された以下も拘束条件とする。

- アーキテクチャ図
- シーケンス図
- ライフサイクル記述
- ownership 記述
- 責務定義
- 不変条件
- 禁止事項

ゲート適合を理由として本文との不整合を正当化してはならない。

### 3.7.3 リファクタリング禁止規約（新設）

rev13 に明記されていない以下を禁止する。

- 責務分割
- 責務統合
- Utility 抽出
- Helper 追加
- Manager 追加
- Facade 追加
- Adapter 追加
- Wrapper 追加

必要な場合は以下を提出すること。

- 追加理由
- rev13 該当箇所
- 既存実装で代替不可な理由

説明不能な新規抽象化は禁止する。

---

## 4. 設計準拠規約（rev13拘束）

AI は rev13 の受け入れゲート 1〜28 を設計不変条件として扱うこと。

特に以下は改修時の最優先拘束:

- Authority 単一 publish 経路
- RuntimeWorld immutable
- RT->Snapshot 到達経路 0
- Generation/WorldId 採番主体単一化
- RetireCoordinator の lifecycle owner 意味論
- GracePeriod single-reader 最適化規約

rev13 と矛盾する実装は禁止。

---

## 5. 部分修正禁止規約

AI は「必要箇所だけ修正」を行ってはならない。

修正対象が RuntimeWorld 系の場合、最低でも以下の整合を確認/更新すること。

- WorldBuilder
- Validation
- Publish
- Retire
- Observe
- Metrics
- Tests

説明不能な未修正箇所は禁止。

---

## 6. 新旧二重実装禁止規約

以下を禁止する。

- 旧経路残置（例: `publishOld()`）
- 暫定経路残置（例: `legacyPublish()`）
- 移行分岐残置（例: `if (useNewRuntime)`）

rev13 で不要となる経路は削除すること。

### 6.1 削除候補列挙義務

修正対象シンボルごとに、以下を一覧化すること。

- 削除した経路
- 削除不要と判断した経路

旧経路を残す場合は、残存理由を記載しなければならない。

### 6.2 シンボル削除規約（新設）

既存 public API の削除は禁止する。

削除する場合は以下を提出すること。

- caller 一覧
- indirect caller 一覧
- rev13 根拠
- 代替経路

提出なしの削除は禁止。

---

## 7. Authority一元化規約

許可される publish 経路は `RuntimeWorldAuthority` のみ。

禁止:

- direct publish
- direct retire
- direct free
- direct generation update
- direct sequence update

bypass 経路が 1 本でも残存する場合は完了不可。

### 7.1 Authority/Store/Retire 専用監査

`RuntimeWorldAuthority` / `RuntimeStore` / `RuntimeRetireCoordinator` について、以下を専用監査項目として必ず確認すること。

- publish 経路唯一性
- observe 経路唯一性
- retire 経路唯一性
- generation 更新唯一性
- publicationSequence 更新唯一性

### 7.2 Single Writer監査

以下について更新主体を全列挙すること。

- `RuntimeWorld`
- `generation`
- `worldId`
- `publicationSequence`

更新主体が複数存在しないことを証明すること。

---

## 8. RuntimeWorld不変条件規約

禁止:

- `mutable`
- `const_cast`
- 非 const accessor（publish 後更新に到達可能なもの）
- 遅延初期化（publish 後）
- publish 後変更

### 8.1 RuntimeWorld immutable 専用監査

以下を全検索し、検出結果を提出すること。

- `mutable`
- `const_cast`
- non-const accessor
- setter
- lazy initialize
- post publish update

### 8.2 Deep Immutability監査

`RuntimeWorld` が保持する全メンバ型を列挙すること。

保持型が内部状態変更可能である場合、以下を監査すること。

- mutable state
- cache
- lazy state
- synchronization state

`RuntimeWorld` 内部から状態変更可能な型の保持は禁止する。

### 8.3 Transitive Immutability監査（新設）

`RuntimeWorld` が保持する全型について、再帰的に到達可能な型を列挙すること。

以下を禁止する。

- mutable
- const_cast
- lazy initialization
- cache update
- synchronization state update

`RuntimeWorld` から到達可能な状態変更経路が存在しないことを証明すること。

---

## 9. RT安全規約

Audio Thread で以下を禁止:

- lock/mutex/recursive_mutex/shared_mutex
- condition_variable
- allocation/free
- blocking wait

追加導入禁止。

---

## 10. shared_ptr/RT規約

RT 側で以下を禁止:

- `shared_ptr` copy
- `shared_ptr` reset
- `shared_ptr` move

RT は `const RuntimeWorld*` のみ利用すること。

---

## 11. Validation規約

Validation 主体は `RuntimeWorldAuthority` のみ。

禁止:

- Builder validation（最終判定）
- UI validation
- Precheck validation による本判定代替
- Validation bypass publish

rev13 の Validation 必須項目をすべて満たすこと。

---

## 12. 改修漏れ防止規約

実装完了前に、AI は影響分析レポートを提出すること。

最低項目:

- 削除箇所（クラス/API/経路）
- 修正箇所（ファイル/理由）
- 未修正箇所（修正不要理由）

説明不能な未修正箇所は禁止。

### 12.1 改修漏れゼロ確認義務（最重要）

影響範囲一覧に含まれる全ファイルに対して、以下のいずれかを記録すること。

- 修正した
- 修正不要であることを確認した

未判定ファイルを残してはならない。

### 12.2 修正不要判定の証跡義務

修正不要と判定した場合、以下を提出すること。

- ファイル
- シンボル
- 調査方法
- 修正不要理由
- rev13 該当条項

### 12.3 シンボル単位判定義務

影響範囲一覧に含まれるファイルについては、主要シンボル単位で以下を記録すること。

- 修正
- 修正不要

ファイル単位のみの修正不要判定は禁止する。

---

## 13. 監査規約

### 13.1 grep監査義務（存在確認）

実装後に最低以下を全検索すること。

- `publish(` / `publishWorld(`
- `retire` / `retireWorld`
- `RuntimeWorld`
- `RuntimeSnapshot`
- `generation`
- `publicationSequence`
- `worldId`
- `epoch`
- `version`
- `authority`
- `observeWorldHandle`
- `shared_ptr`
- `weak_ptr`
- `atomic`
- `exchange`
- `compare_exchange`
- `store(`
- `load(`
- `fetch_add(`
- `delete` / `free` / `release` / `reset`

検索結果未確認で完了報告禁止。

### 13.2 否定検索義務

旧経路除去確認では、存在確認検索に加えて禁止シンボル検索を実施すること。

最低禁止シンボル例:

- `publishLegacy`
- `legacyPublish`
- `oldPublish`
- `useNewRuntime`
- `temporaryRuntime`
- `bridgeRuntimeOld`

検出された場合は、削除または残存理由記載を必須とする。

### 13.3 RuntimeWorld 破棄経路監査

`RuntimeWorld` 破棄経路を全列挙すること。

最低提出:

- 破棄関連シンボル（`delete`/`free`/`release`/`reset`）の検出結果
- `RuntimeWorld` に到達する破棄経路一覧
- `RuntimeRetireCoordinator` 管理外の破棄経路が存在しない証跡

### 13.4 RetireCoordinator 専用監査

以下を全列挙すること。

- retire 登録箇所
- retire 解除箇所
- free 判定箇所
- destroy 発生箇所

`RuntimeRetireCoordinator` 外から free 判定していないことを証明すること。

### 13.5 Generation更新監査

generation 更新箇所を全列挙し、`RuntimeGenerationGenerator` 以外の更新箇所が存在しないことを証明すること。

### 13.6 WorldId発番監査

worldId 代入/発番箇所を全列挙し、`RuntimeWorldIdGenerator` 以外から発番されていないことを証明すること。

### 13.7 publicationSequence監査（強化）

`publicationSequence` について以下を列挙し、rev13 規約との一致を確認すること。

- 発番箇所
- 更新箇所
- commit 箇所
- validation 失敗時に確定しないこと
- commit 前以外で採番しないこと
- publish 成功時のみ metadata へ反映されること

### 13.8 Observe長期保持監査

`observeWorldHandle()` の戻り値について、以下を全検索し検出結果を提出すること。

- メンバ格納
- static 保持
- lambda capture
- async capture
- container 保持
- cache 保持

### 13.9 MCP再監査義務（必須）

実装後に Serena/CodeGraph を再実行し、実装前との差分を確認すること。

確認事項:

- RuntimeWorld: 旧経路残存なし
- Publish: authority bypass なし
- Observe: 長期保持経路なし
- Snapshot: RT 到達経路なし
- Retire: direct free 経路なし
- Generation: 独自採番経路なし
- WorldId: 独自発番経路なし

### 13.10 未調査コード禁止

以下を禁止:

- 一部ファイルのみ読んで実装
- 定義だけ見て実装
- caller/callee 未確認で実装
- テスト未確認で実装

### 13.11 ISR Core監査

以下について専用監査を行うこと。

- `RuntimeWorldAuthority`
- `RuntimeStore`
- `RuntimePublicationCoordinator`
- `RuntimeRetireCoordinator`

監査項目:

- owner
- lifecycle
- publish 責務
- retire 責務
- validation 責務
- generation 責務

### 13.12 新規追加シンボル監査

追加された型・関数・namespace について、以下を記録すること。

- 追加理由
- rev13 対応箇所
- 既存機能との重複有無

既存責務と重複する新規型は禁止する。

### 13.13 Data Flow監査

以下について生成元から消費先まで全経路を列挙すること。

- `RuntimeWorld`
- `generation`
- `worldId`
- `publicationSequence`
- `RuntimeMetadata`

途中キャッシュ・ラッパ・View を経由する場合も列挙すること。生成元不明の更新は禁止する。

### 13.14 Ownership移譲監査

以下を全列挙すること。

- `unique_ptr` move
- `shared_ptr` 生成
- `shared_ptr` reset
- `shared_ptr` release
- raw pointer 化
- observer 取得

`RuntimeWorld` の所有権移譲は `Authority -> Store -> RetireCoordinator` 以外を禁止する。

### 13.15 Atomic監査

atomic 更新箇所について、以下を列挙すること。

- `memory_order`
- release/acquire 対
- publication 可視化保証

`memory_order` 変更時は理由を記載すること。

### 13.16 責務肥大化監査

既存クラスについて以下を列挙すること。

- 新規責務追加
- 権限追加
- ownership 追加

rev13 責務定義に存在しない責務追加は禁止する。

### 13.17 新規型追加制限規約（新設）

新規 class / struct / interface の追加は禁止する。

以下の場合のみ許可する。

- rev13 に明記
- テストコード
- コンパイル成立のための最小補助型

追加時は以下を提出すること。

- 追加理由
- 既存型で代替できない理由
- ownership
- lifecycle
- 責務

提出なしは不合格。

---

## 14. 到達経路監査規約

AI は次の主要経路以外を監査すること。

`RT -> RuntimeWorld -> Graph -> DSP`

特に、`RT -> Snapshot` 到達経路 0 を証明すること。

### 14.1 RT->Snapshot 到達経路0証明の提出要件

以下すべてを提出すること。

- grep 結果
- Serena 参照結果
- CodeGraph 到達経路結果
- 到達不能根拠

---

## 15. テスト更新規約

実装変更時は、既存テスト更新 + 新規テスト追加を行うこと。

最低テスト:

- Generation: rollback 拒否
- Publish: 再 publish 拒否
- Validation: invalid graph 拒否
- Retire: grace period
- Observe: RT observe 制約
- WorldId: 単調増加
- Generation: 単調増加
- publicationSequence: 単調増加
- validation 失敗時: publicationSequence 未確定
- RuntimeWorld immutable
- RT->Snapshot 到達不可
- observe 長期保持禁止
- RetireCoordinator 以外から free 判定不可

テスト未更新で完了報告禁止。

### 15.1 テスト品質規約

各新規テストは最低以下を含むこと。

- 正常系
- 異常系
- 境界値
- 失敗ケース

正常系のみは禁止する。

### 15.2 回帰試験義務（新設）

修正した不具合または受け入れゲート項目について、修正前は失敗し修正後は成功することを示す回帰試験を追加すること。

回帰試験が存在しない変更は完了不可。

---

## 16. 完了報告規約

完了時に以下を必ず提出すること。

1. 変更ファイル一覧（追加/変更/削除）
2. 影響範囲一覧（Runtime/Publish/Retire/Snapshot/Metrics/Tests）
3. rev13 受け入れゲート 1〜28 適合表（適合/非適合）
4. 残課題（存在時のみ）

### 16.1 rev13ゲートのコード証跡義務（強化）

rev13 受け入れゲート 1〜28 の各項目について、以下を提出すること。

- ファイルパス
- 行番号
- シンボル（関数/型/メンバ）
- 判定根拠

適合だけの記載は禁止。

### 16.2 ゲート反証確認義務

各ゲートについて、適合根拠だけでなく非適合となる可能性も記載すること。

反証不能な場合のみ適合と判定できる。

### 16.3 rev13本文適合表（新設）

以下について適合表を提出すること。

- アーキテクチャ図
- シーケンス図
- ownership 記述
- lifecycle 記述
- 責務定義
- 不変条件
- 禁止事項

各項目について以下を記載すること。

- 対応コード
- 行番号
- 適合根拠

---

## 17. 全体整合性監査規約

完了宣言前に、以下6領域の全体フローを再確認すること。

- RuntimeWorld
- Publish
- Observe
- Retire
- Generation
- Snapshot

局所確認のみで完了してはならない。

最低提出物:

- 全体フロー図（簡易テキスト図またはMermaid可）
- rev13 との一致確認結果

### 17.1 アーキテクチャ差分監査

実装前後の以下を比較し、差分を一覧化すること。

- publish 経路
- observe 経路
- retire 経路

差分項目:

- 経路追加
- 経路削除
- 責務移動

---

## 18. 部分完了宣言禁止規約（最重要）

rev13 受け入れゲートに影響する変更を行った場合、受け入れゲート検証が完了するまで、以下の宣言をしてはならない。

- 完了
- 実装完了
- 移行完了

Phase1完了等の局所完了表現は、全体ゲート未検証の状態では使用禁止とする。

### 18.1 完了類義語禁止（新設）

受け入れゲート未検証状態では、以下の表現を禁止する。

- 完了
- 実装完了
- 移行完了
- 主要完了
- 概ね完了
- 実質完了
- 完了見込み
- 問題なし

受け入れゲート検証前は「実装作業中」のみ使用可とする。

---

## 19. 完了宣言禁止条件

以下のいずれか該当時、AI は実装完了を宣言してはならない。

- grep監査未実施
- 否定検索未実施
- MCP再監査未実施
- MCP結果未提出
- MCP証跡未保存
- 影響分析未実施
- 影響範囲全ファイルの判定未完了
- 影響範囲主要シンボルの判定未完了
- テスト未更新
- 回帰試験未追加
- rev13 受け入れゲート未確認
- rev13ゲートのコード証跡未提出
- rev13ゲートの反証確認未提出
- rev13本文適合表未提出
- 旧経路残存
- bypass 経路残存
- RuntimeWorld mutable 残存
- RT で shared_ptr 操作残存
- RT->Snapshot 経路残存

---

## 20. 例外（Waiver）規約

やむを得ず規約逸脱が必要な場合、以下を明記した Waiver を作成しない限り実施不可。

- 逸脱条項
- 逸脱理由
- 影響範囲
- 代替安全策
- 期限付き解消計画

Waiver なき逸脱は不合格とする。

---

## 21. 最重要原則

AI はコードを書いたら完了ではなく、

rev13 受け入れゲート 1〜28 が全て満たされることを証明して初めて完了

と定義する。

また、改修対象クラスを修正した場合、そのクラスを参照する全コードを追跡し、影響範囲を確認しなければ完了してはならない。

これを改修漏れ防止の最上位規則とする。
