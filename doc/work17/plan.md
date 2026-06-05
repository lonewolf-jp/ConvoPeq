# ConvoPeq Practical Stable ISR Bridge Runtime 完全移行計画 v6.4 – 最終凍結完全版（全文・単独完結）

**対象バージョン**: v2.3  
**最終更新**: 2026-06-06  
**ステータス**: ⚡ 凍結・実装基準線・原則改訂終了 ⚡  

---

## 1. 基本方針

- **Authority Singularization**: 各権限は単一コンポーネントが保持する。
- **Semantic Single Source**: `RuntimePublishWorld` のみが権威情報源。
- **Fail-Closed**: 契約違反は即 reject／旧状態維持。
- **過剰なアーキテクチャ変更は行わない**（Singleton化、完全レイヤー分離は計画外）。
- **実運用で破綻しにくい最小改修**を徹底する。
- **監査の形骸化を防ぐ**: 監査レポートは必須フィールド＋内容確認＋ツール出力提出を必須とする。
- **静的ガバナンスと動的テストを分離**: AuthorityViolation 系は静的検査で行う。
- **API名・フィールド名に依存しない監査**: 実装の詳細な名前変更に強い設計とする。

---

## 2. 優先度分類

| 区分 | 意味 | CI必須 | 完了条件必須 |
|------|------|--------|--------------|
| **必須** | 移行完了に不可欠。コードに残存する問題を直接解決 | ✅ | ✅ |
| **推奨** | より堅牢にするが、未実施でも移行完了とみなす | ❌（警告またはレビュー対象） | ❌ |
| **再定義** | 過剰な制約を実用的な範囲に緩和 | 状況による | 一部のみ |

---

## 3. 未達成内容一覧（全項目・最終確定）

### 必須（P1～P20 のうち完了条件に必須のもの）

| ID | 未達成内容 | 対応種別 |
|----|-----------|----------|
| **P1** | **PublicationIntent 経路の残存（二段階移行）** | Phase1-A: 委譲ラッパ化、Phase1-B: 完全削除。**本質は publication lifecycle authority の排除（定義具体化済み）** |
| P2 | RuntimeBuildSnapshot 等を権威として使用 | Authority用途削除 |
| P3 | Retire Authority の二重化（新規コード禁止、既存警告） | 段階的統一（新規禁止、既存警告） |
| **P4** | **Generation / ActivationEpoch 契約（現行期間中）** | `generation` 増加時に `activationEpoch` は +1 以上増加。同一 `generation` での変更禁止。 |
| P5 | RuntimeWorld 不変性監査（freeze 必須ではない） | publish後変更がないことを静的解析＋レビューで確認 |
| **P6** | **Publication Failure Contract（全状態＋副作用ロールバック含む）** | 契約定義 + 実装（P20 に詳細） |
| P7 | Retire Ordering Contract 未確認 | **文書化のみ** |
| P8 | Authority Source Audit | Semantic decision は RuntimeWorld、Execution decision は Executor Local |
| P9 | Authority Regression Guard | CI + Authority Owner List（下表参照） |
| P10 | Projection Consistency Audit | RuntimeWorld → Projection の一致監査 |
| **P11** | **Observe Source Audit（監査レポート必須、内容確認）** | serena/codegraph 調査 + レポート必須フィールド |
| **P12** | **Crossfade Semantic Leakage Audit（監査レポート必須、内容確認）** | Call graph 解析 + レポート必須フィールド |
| P13 | External Semantic Dependency Audit（限定適用） | World構成を決定する値のみ対象 |
| **P14** | **Publication Unit Audit（監査レポート必須、完了条件必須）** | World 単位以外の部分公開禁止 |
| **P15** | **RuntimeStore Mutation Authority Audit（capability 発行主体ベース）** | `RuntimeStore` mutation の **capability 発行主体** が Coordinator のみ |
| **P20** | **Fail-Closed Rollback Audit（state + side effect + telemetry）** | 動的テスト + observable state + observable side effect（telemetry含む）のロールバック確認 |

### 推奨（完了条件に含めない）

| ID | 未達成内容 |
|----|-----------|
| P16 | RuntimeWorld Construction Audit（推奨） |
| P18 | RuntimeWorld Identity Audit（推奨） |

### 再定義（過剰制約を緩和）

| ID | 新定義 |
|----|--------|
| P17 | `freeze` 必須ではない。publish 後変更を監査で確認 |
| P19 | キャッシュや `mutable` は許容。Projection が authority にならないこと |

---

## 4. 必須項目の改修詳細（最終確定・文言飽和）

### P1 – PublicationIntent 二段階移行（publication lifecycle authority の排除）

**Phase1-A 前準備（必須）**:
- `publishWorld` 呼び出し箇所全列挙（`serena query "callers of publishWorld"`）
- `enqueuePublicationIntentForRuntimeCommit` 呼び出し元全列挙
- defer commit 経路全列挙（`appendPublicationIntentForCommitConsumer` 等）
- `RuntimePublishWorld` 生成箇所全列挙
- 現行コードの二重 publish 構造を把握した上で Phase1-A を開始する

**Phase1-A**（移行中）:
- `enqueuePublicationIntentForRuntimeCommit` を `coordinator.publishWorld()` への単純な委譲ラッパにする
- CI: 旧コードの存在を **警告**（エラーにはしない）

**Phase1-B**（移行完了）:
- `PublicationIntent`, `publicationLog`, 関連関数を完全削除
- 残骸メトリクス（`pendingIntentCount`, `publicationBacklog`, `intentBacklog` 等）も削除
- CI: 旧コードの存在を **エラー**
- **本質（具体定義）**: `publishWorld` 以外に **publication lifecycle authority** が存在しないこと
  - **publication lifecycle authority の定義**：
    - `RuntimePublishWorld` の **publish 実行タイミング** を決定できる権限
    - **publish 順序** を決定できる権限
    - **publish 保留状態** を保持できる権限
    - **publish 再試行状態** を保持できる権限
    - **publish バックログ状態** を保持できる権限
    - **publish 遅延状態** を保持できる権限
- 監査レポート `p1_phase1b_audit.md` に調査方法と結果を記載する

### P3 – Retire Authority 二重化（新規コード禁止、既存警告）

- **新規コード**: `EpochDomain::enqueueRetire` を直接呼び出さないこと（CI でエラー）。
- **既存コード**: 警告扱いとし、移行完了条件からは外す。
- **CI**: `git diff` で新規追加行のみチェック。

### P4 – Generation / ActivationEpoch 契約（increment policy 明文化）

**仕様**:
- `generation` は strict monotonic: `new.generation > lastCommitted.generation`
- `activationEpoch` は `generation` 増加時に **必ず増加する（+1 以上）**。`new.activationEpoch > lastCommitted.activationEpoch` を満たせば +1 でも +2 でもよい。
- 同一 `generation` での `activationEpoch` 単独変更は禁止
- **将来の分離可能性**: 現行移行期間中はこの契約を固定とする。将来、両者を独立した概念へ分離する必要性が生じた場合は別計画とする。

**実装**:
```cpp
if (new.generation <= lastCommitted.generation) return reject;
if (new.activationEpoch <= lastCommitted.activationEpoch) return reject;
```

### P11, P12, P14, P15 – 監査レポート必須（内容確認付き）

各監査レポート（`p11_audit.md`, `p12_audit.md`, `p14_audit.md`, `p15_audit.md`）に以下の必須フィールドを記入する：

```text
AUDIT_RESULT: PASS (または FAIL)
AUDIT_DATE: YYYY-MM-DD
AUDITOR: <name>
CHECKED_SYMBOLS: <カンマ区切りで調査したシンボル一覧（空禁止）>
FINDINGS: <発見された問題点、または問題なし>
SEARCH_COMMANDS: <使用した serena/codegraph/grep コマンド（空禁止）>
```

### P14 – Publication Unit Audit（完了条件必須）

- **禁止**: `RuntimePublishWorld` 全体以外の部分公開インターフェース（`publish(generation)`, `publish(dsp)` など）
- **検出方法**: `serena` または `codegraph` による確認を必須とする
- **監査レポート**: `p14_audit.md` として提出
- **完了条件**: 部分公開インターフェースが存在しないことを監査レポートで確認する（CI自動検出はしないが、完了宣言には必須）

### P15 – RuntimeStore Mutation Authority Audit（capability 発行主体ベース）

- **責務**: `RuntimeStore` の状態変更の **capability 発行主体** が `RuntimePublicationCoordinator` のみであることを確認する。
- **「authority owner」の明確化**: 
  - mutation を直接実行するコードの caller ではなく、**mutation を実行する capability（権限）を発行できる主体** が Coordinator のみであること
  - 例：Coordinator から `WriteAccess` を受け取ったオブジェクトが mutation を実行することは許容される（委譲）
  - しかし、Coordinator 以外が独自に `WriteAccess` を生成したり、Coordinator の関与なしに mutation を実行したりしてはならない
- **必須ツール**: `serena`, `codegraph`（grep は補助スクリーニングとしてのみ使用）
- **出力**: `audit/p15_serena.txt`, `audit/p15_codegraph.txt` として保存・提出（非空必須）

### P20 – Fail-Closed Rollback Audit（state + side effect + telemetry）

- **責務**: Publication reject 時にシステム状態と副作用が正しくロールバックされることを動的テストで保証する。
- **ロールバック対象**:
  - **observable state 全体**（状態機械、世代・シーケンス、バックログ、圧力カウンタ等）
    - 特に以下を含む：`generation`, `publicationSequenceId`, `publicationEpoch`, `mappedRuntimeGeneration` 等
  - **observable side effect** の absence：
    - retire callback の firing
    - publication notification の emission
    - `didPublishRuntimeNonRt` 等の publish hook の実行
    - `willRetireRuntimeNonRt` 等の retire hook の実行
    - validator hook の実行
    - **telemetry 更新**（`retireAuthorityCount` 等の統計値変更）
    - **evidence emission**（`emitEvidenceTickNonRt` 等）
- **`CoordinatorStateUnchanged` の定義**:
  - 全ての observable state が reject 前後で一致すること
  - 副作用（callback, notification, telemetry, statistics update）が reject 後に観測されないこと
- **動的テスト**: 各 observable state カテゴリと副作用カテゴリについて reject 前後で変化しないことを確認するテストを実装する

---

## 5. CI ワークフロー（最終版）

```yaml
name: ISR Authority Compliance (v6.4)

on: [push, pull_request]

jobs:
  # 静的ガバナンス検査
  authority-static-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      # P1 Phase1-A: 警告
      - name: Check legacy publication patterns (P1)
        run: |
          if grep -rqE "struct PublicationIntent|class PublicationIntent|enqueuePublicationIntentForRuntimeCommit\(|applyRuntimeCommitFromIntent\(" src/; then
            echo "⚠️ Warning: legacy commit patterns detected (Phase1-A)"
          fi
      
      - name: Check legacy metrics residues (P1)
        run: |
          if grep -rqE "pendingIntentCount|publicationBacklog|intentBacklog|setPendingIntentCount|setPublicationBacklogCount" src/; then
            echo "⚠️ Warning: legacy intent metrics found (Phase1-A)"
          fi
      
      # P14: 部分公開禁止（grep 補助、警告のみ、CI必須ではない）
      - name: Check partial publication interfaces (P14)
        run: |
          if grep -rqE "publish\(generation|publish\(dsp|appendPublicationIntentForCommit" src/; then
            echo "⚠️ Warning: possible partial publication interface detected (use serena for confirmation)"
          fi
      
      # P3: 新規コードのみエラー
      - name: Static AuthorityViolation: Direct EpochRetire (new code only)
        run: |
          if git diff origin/main...HEAD | grep -E "^\+\s*.*EpochDomain.*enqueueRetire" | grep -v "Coordinator"; then
            echo "❌ Error: new code uses direct enqueueRetire"
            exit 1
          fi
  
  # 動的テスト
  authority-runtime-tests:
    runs-on: ubuntu-latest
    needs: authority-static-checks
    steps:
      - uses: actions/checkout@v4
      
      - name: Build and run fail-closed governance tests (P20)
        run: |
          cmake -DCMAKE_BUILD_TYPE=Debug ..
          cmake --build . --config Debug
          ctest -R "Reject|Rollback|Unchanged|CoordinatorState|Callback|Notification|Telemetry|Statistics" --output-on-failure
  
  # 監査レポート検証
  audit-report-verification:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Verify audit reports (P11, P12, P14, P15)
        run: |
          for report in p11_audit.md p12_audit.md p14_audit.md p15_audit.md; do
            if [ ! -f "audit/$report" ]; then
              echo "❌ Error: $report not found"
              exit 1
            fi
            for field in AUDIT_RESULT AUDIT_DATE AUDITOR CHECKED_SYMBOLS FINDINGS SEARCH_COMMANDS; do
              if ! grep -q "$field:" "audit/$report"; then
                echo "❌ Error: $report missing $field"
                exit 1
              fi
            done
            if grep -q "CHECKED_SYMBOLS: *$" "audit/$report"; then
              echo "❌ Error: $report CHECKED_SYMBOLS is empty"
              exit 1
            fi
            if grep -q "SEARCH_COMMANDS: *$" "audit/$report"; then
              echo "❌ Error: $report SEARCH_COMMANDS is empty"
              exit 1
            fi
            if ! grep -q "AUDIT_RESULT: PASS" "audit/$report"; then
              echo "❌ Error: $report AUDIT_RESULT is not PASS"
              exit 1
            fi
          done
      
      - name: Verify P15 tool outputs (non-empty)
        run: |
          for output in p15_serena.txt p15_codegraph.txt; do
            if [ ! -f "audit/$output" ]; then
              echo "❌ Error: $output not found"
              exit 1
            fi
            if [ ! -s "audit/$output" ]; then
              echo "❌ Error: $output is empty"
              exit 1
            fi
          done
      
      - name: Verify P1 Phase1-B audit (optional)
        run: |
          if [ -f "audit/p1_phase1b_audit.md" ]; then
            if ! grep -q "AUDIT_RESULT: PASS" "audit/p1_phase1b_audit.md"; then
              echo "⚠️ Warning: P1 Phase1-B audit result is not PASS"
            fi
          fi
```

---

## 6. 完了条件（必須項目のみ・最終）

| ID | 完了条件 | 検証方法 | CI必須 | 完了条件必須 |
|----|---------|----------|--------|--------------|
| **P1** | **Phase1-B 完了（publication lifecycle authority 排除）** | CI エラー + 監査レポート | ✅ | ✅ |
| P2 | `RuntimeBuildSnapshot` 権威使用排除 | 静的解析 + CI | ✅ | ✅ |
| P3 | 新規退役コードは Coordinator 経由 | CI（新規コードエラー） | ✅ | ✅ |
| **P4** | **Generation / ActivationEpoch 契約テストがパス** | 単体テスト + CI | ✅ | ✅ |
| P5 | publish 後変更が無いことを監査 | 静的解析 + レビュー | ✅(警告) | ✅ |
| **P6** | **reject 時に全状態＋副作用（telemetry含む）がロールバック** | 単体テスト + CI（P20） | ✅ | ✅ |
| P7 | 退役順序文書化 | ドキュメント確認 | ❌ | ✅ |
| P8 | Semantic decision が RuntimeWorld のみ参照 | Authority Source Audit | ✅ | ✅ |
| P9 | Authority Owner List 遵守 | CI + コードレビュー（下表参照） | ✅ | ✅ |
| P10 | Projection と Authority の一致検証 | 単体テスト + CI | ✅ | ✅ |
| **P11** | **監査レポート提出** | 監査レポート + CI | ✅ | ✅ |
| **P12** | **監査レポート提出** | 監査レポート + CI | ✅ | ✅ |
| P13 | World構成決定値のみ RuntimeWorld に集約 | 静的監査 + CI警告 | ✅(警告) | ✅ |
| **P14** | **部分公開インターフェースなし（監査レポート提出）** | 監査レポート | ❌ | ✅ |
| **P15** | **RuntimeStore mutation capability 発行主体が Coordinator のみ** | 監査レポート + CI | ✅ | ✅ |
| **P20** | **動的テスト全てパス（state + side effect + telemetry ロールバック）** | ctest | ✅ | ✅ |

---

## 7. Authority Owner List（P9 の具体表）

実装時のレビュー・監査に使用する Authority Owner List：

| 權限 | 所有者 | 許容される委譲 | 禁止事項 |
|------|--------|----------------|----------|
| Publication Authority (World publish) | `RuntimePublicationCoordinator` | `publishWorld` の呼び出し元として任意のコードを許可 | Coordinator 以外が `RuntimeStore` の current world を直接書き換えること |
| Mutation Authority (Store 書き込み) | `RuntimePublicationCoordinator` | Coordinator から発行された `WriteAccess` 等を通じた mutation | Coordinator 以外が mutation capability を発行すること |
| Retire Authority | `RuntimePublicationCoordinator` | `enqueueRetire` の呼び出し元として任意のコードを許可（ただし新規コードは Coordinator 経由を推奨） | 既存の `EpochDomain::enqueueRetire` 直呼びを新規に追加すること |
| Validation Authority | `RuntimePublicationCoordinator` / `RuntimePublicationValidator` | 実装詳細に応じて委譲可能 | Coordinator 以外が `RuntimeStore` の内容を検証せずに書き換えること |

---

## 8. 実装着手指示（Execution Baseline）

### 8.1 実装着手の判断

**v6.4 を最終凍結版とし、実装作業を開始する。**

実装を妨げるレベルの設計欠陥はない。現行 ConvoPeq コードに残存する問題（`PublicationIntent` 系、`publicationLog` 系、`applyRuntimeCommitFromIntent` 系、defer commit 系）は計画の対象として確認済みである。

### 8.2 実装順序（最優先）

**P1 Phase1-A 前準備** から着手する。以下の全列挙を実施し、成果物を `audit/` ディレクトリに保存する。

1. `publishWorld` の呼び出し元全列挙 → `audit/p1_callers_publishWorld.md`
2. `enqueuePublicationIntentForRuntimeCommit` の呼び出し元全列挙 → `audit/p1_callers_enqueuePublicationIntent.md`
3. `RuntimePublishWorld` の生成箇所全列挙 → `audit/p1_runtimepublishworld_construction.md`
4. defer commit 経路（`appendPublicationIntentForCommitConsumer` 等）全列挙 → `audit/p1_defer_commit_paths.md`

これらの調査には `serena` と `codegraph` を必須とする。grep は補助的に使用する。

### 8.3 実装時の運用ルール

以下の運用ルールは計画の一部として扱い、実装チームは厳守すること。

- **成果物の固定**: 上記の命名規則で `audit/` に保存し、Phase1-B レビュー時に参照できるようにする。
- **Mutation Capability Graph**: P15 の監査では、可能な限り Coordinator → WriteAccess → RuntimeStore の授受関係をグラフとして残すことを推奨する。
- **副作用の解釈**: P20 の副作用は「rollback」ではなく「pre-commit 禁止」と解釈する。reject 時に副作用が一度も発生していないことを確認する。
- **Authority Owner List の共有**: 上記の Authority Owner List をチーム内で共有し、コードレビューの checklist として使用する。

---

## 9. 計画の確定と発効

**本計画をもって ConvoPeq Practical Stable ISR Bridge Runtime 完全移行計画を終了する。**

これ以降、計画の原則改訂は行わない。実装作業は上記の指示に従い進めること。ただし、実装過程で不可避な微調整（例：環境差異によるコマンドの詳細変更など）は「移行後メンテナンス計画」の範囲で許容する。原則を変更する改訂は新バージョンとして扱う。

実装作業は **P1 Phase1-A 前準備** から開始すること。

---

**計画策定者**: 監査チーム  
**最終承認日**: 2026-06-06  
**実装開始指示日**: 2026-06-06  
**凍結宣言**: 本計画は凍結され、実装作業は即時開始可能である。実装はこの計画に従い進めること。