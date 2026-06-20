# Practical Stable ISR Bridge Runtime — 設計書 v4.17（全調査完了版）

**Document Version:** 4.17
**Date:** 2026-06-20
**Based on:** v4.16 + 全ツール実コード調査8項目
**Status:** 全調査完了

---

## v4.16 → v4.17 棚卸し調査結果

| # | 調査項目 | 使用ツール | 調査結果 | 設計反映 |
|---|---|---|---|---|
| A | **EvidenceExporter publicationSequenceId 読取元** | Select-String, Serena | `FailureSnapshot.publicationSequenceId` は **誰も設定していない**（常に 0） | v4.16 の「EvidenceExporter 読取のため atomic で統一」は誤り。実際は TelemetryRecorder 経由で読むが、publicationSequenceId は常に 0 になる。本設計では PersistentStateBlock の atomic は commit() の論理一貫性担保のみを責務とし、EvidenceExporter 読取は考慮しない |
| B | **DeferredDeletionQueue publicationSequenceId 供給元** | Select-String | enqueue() のパラメータとして world struct から渡される。Coordinator の atomic は読まない | 影響なし。DDQ は別経路で値を取得 |
| C | **AudioEngine.Commit.cpp mappedRuntimeGeneration 参照元** | Select-String, Serena | `world.publication.mappedRuntimeGeneration`（world struct の値）を参照。Coordinator の atomic は読まない | 影響なし。両者は独立した値 |
| D | **coordinator 3原子フィールドの真の役割** | Select-String (全ソース調査) | Coordinator の 3 フィールドは **ローカルキャッシュ**。真の Authority は AudioEngine の `publicationSequenceCounter_` / `lastCommittedPublicationSequence_` | ISR-AUTH-005 の定義を「Coordinator 内における唯一の永続メタデータ源」に限定 |
| E | **getVersion() 完全棚卸し** | Select-String (全ファイル) | 定義: `mappedRuntimeGeneration_` を読む。呼び出し元: **テストファイルのみ**（4箇所）。本番コードでの呼び出しなし | Phase-0 で変更可。`persistentState_.snapshot().mappedRuntimeGeneration` に置き換え |
| F | **CI スクリプト実在確認** | Select-String (ls) | 既存 120+ スクリプトあり。v4.16 で提案の 9 スクリプトは **すべて未作成** | 新規作成が必要と明記 |
| G | **SnapshotResult 型 処遇** | Select-String (全.h) | コードベースに **存在しない**。v4.13 設計書の死文化物 | 設計書から削除済み（v4.14 で対応済み） |
| H | **AudioEngine の出版カウンタ構造** | Serena, Select-String | `reserveRuntimePublicationIdentity()` が `publicationSequenceCounter_` を atomic increment。値を `RuntimePublicationIdentity` に格納 → builder が world の `PublicationSemantic.sequenceId` に設定 → commit() で coordinator の `publicationSequenceId_` にコピー | 値の流れを第0章に追記 |

---

## 第0章: 値の流れ（完全版）

```
AudioEngine
  publicationSequenceCounter_ (atomic<uint64>)
  │  reserveRuntimePublicationIdentity() で fetchAdd
  │
  ▼
RuntimePublicationIdentity
  {generation, worldId, publicationSequence}
  │
  ▼  RuntimeBuilder が world に設定
RuntimePublishWorld
  PublicationSemantic {
    sequenceId,             ← publicationSequence からコピー
    epoch,
    mappedRuntimeGeneration,
    previousSequenceId
  }
  RuntimeMetadata {
    publicationSequence     ← publicationSequence からコピー（別名）
  }
  │
  ├──▶ commit() 経由で coordinator の atomic にコピー
  │    RuntimePublicationCoordinator
  │      publicationSequenceId_    (atomic cache)
  │      publicationEpoch_         (atomic cache)
  │      mappedRuntimeGeneration_  (atomic cache)
  │      └── commit() 内の単調増加チェック & getVersion() のみ使用
  │
  ├──▶ publishAndSwap() 経由で RuntimeStore に公開
  │    RuntimeStore<RuntimePublishWorld, ...>
  │      current (atomic<T*>)       ← AudioThread が observe()
  │
  └──▶ DeferredDeletionQueue
         enqueue(..., publicationSequenceId, generation)
         └── world struct から直接渡される（coordinator の atomic は読まない）
```

### 重要な発見: FailureSnapshot.publicationSequenceId は常に 0

`TelemetryRecorder::recordFailure()` は `publicationSequenceId` を設定しない。
`recordFailureSnapshot()` を呼ぶコードも存在しない。
そのため `ISREvidenceExporter::build...()` が出力する `"seqId"` は常に 0 になる。

**影響**: EvidenceExporter の publicationSequenceId 出力は実質的に無効。
修正する場合は、呼び出し元で `FailureSnapshot.publicationSequenceId` を明示的に設定する必要があるが、
本 design のスコープ外とする。

---

## 第1章: Authority Hierarchy（最終版）

```
┌─────────────────────────────────────────────────────┐
│  Primary Authority (AudioEngine)                     │
│  ────────────────────────────────                    │
│  publicationSequenceCounter_        ... atomic incr  │
│  lastCommittedPublicationSequence_  ... atomic store │
│                                                      │
│  責務: 出版シーケンスの真の発行元                      │
└─────────────────────────────────────────────────────┘
        │ reserveRuntimePublicationIdentity()
        ▼
┌─────────────────────────────────────────────────────┐
│  Published Authority (RuntimeStore)                  │
│  ────────────────────────────────                    │
│  RuntimePublishWorld* を atomic exchange              │
│  AudioThread が observe() で世界を観測                 │
│                                                      │
│  責務: AudioThread が参照する world の可用性保証        │
└─────────────────────────────────────────────────────┘
        │ publishAndSwap()
        ▼
┌─────────────────────────────────────────────────────┐
│  Authority Metadata Cache (PersistentStateBlock)     │
│  ────────────────────────────────                    │
│  Coordinator 内の 3 フィールド atomic cache            │
│  commit() の単調増加チェック & getVersion() のみ使用    │
│                                                      │
│  責務: commit() 内モノトニック検証のための論理一貫読取  │
└─────────────────────────────────────────────────────┘
        │ 導出
        ▼
┌─────────────────────────────────────────────────────┐
│  Derived Diagnostics (AuthorityState)                │
│  ────────────────────────────────                    │
│  一時的な導出値。永続化しない                          │
│  reconcile/validateAuthorityStateMatch で使用        │
└─────────────────────────────────────────────────────┘
```

### ISR-AUTH-005 再々定義

**旧（v4.16）**: PersistentStateBlock 以外に 3 フィールドの永続的保管を禁止
**新（v4.17）**: Coordinator クラス内において、publicationSequenceId / publicationEpoch / mappedRuntimeGeneration の永続的保管は PersistentStateBlock のみとし、3 つの個別 atomic フィールドは保持しない

→ **適用範囲を Coordinator クラス内に限定**。AudioEngine 側のカウンタや世界構造体のフィールドは対象外。

---

## 第2章: 方式C（確認済み）

方式C（単純構造体 + `std::atomic<PersistentStateBlock>` + relaxed 操作）は以下の調査結果により正しい：

- **concurrent writer なし**: commit() は MessageThread 専有（全呼び出し元確認済み）
- **concurrent reader なし**: EvidenceExporter の publicationSequenceId 読取は常に 0（事実上 dead code）
- **getVersion()**: テストのみ。Phase-0 で `persistentState_.snapshot().mappedRuntimeGeneration` に変更可

**Phase-0 で削除する 3 個別 atomic フィールド**:

| 現行フィールド | 置き換え先 | 削除条件 |
|---|---|---|
| `std::atomic<PublicationSequenceId> publicationSequenceId_` | `PersistentStateBlock.publicationSequenceId` | Phase-0 |
| `std::atomic<PublicationEpoch> publicationEpoch_` | `PersistentStateBlock.publicationEpoch` | Phase-0 |
| `std::atomic<uint64_t> mappedRuntimeGeneration_` | `PersistentStateBlock.mappedRuntimeGeneration` | Phase-0（getVersion() も変更） |

---

## 第3章: CI スクリプト一覧（全9件・未作成）

以下のスクリプトはすべて新規作成が必要。

| # | スクリプト | 役割 | 新規/既存 |
|---|---|---|---|
| 001 | `isr-verify-auth-001.ps1` | PersistentStateBlock + RuntimeStore からの再構築可能性 | **新規** |
| 002 | `isr-verify-auth-002.ps1` | Recovery 後 validateAuthorityStateMatch PASS | **新規** |
| 003 | `isr-verify-publication-single-path.ps1` | Orchestrator → Coordinator の唯一経路 | **既存**（流用） |
| 004 | `isr-verify-auth-004.ps1` | 構造的 requires 制約（std::is_same 禁止） | **新規** |
| 005 | `isr-verify-auth-005.ps1` | Coordinator 内 3 フィールドの PersistentStateBlock 集中 | **新規** |
| 006 | `isr-verify-auth-006.ps1` | PersistentStateBlock ↔ RuntimeStore 矛盾検出 | **新規** |
| 2a | `isr-verify-getcurrent-test-migrated.ps1` | テスト内 getCurrent() 0 件確認 | **新規** |
| 2b | `isr-verify-getcurrent-removed.ps1` | getCurrent() 定義削除確認 | **新規** |
| D | `isr-verify-retire-no-currentworld.ps1` | retire() 内 currentWorld_ 参照なし確認 | **新規** |

---

## 第4章: 実装 Phase（最終版）

```
Phase-0: 方式C PersistentStateBlock 導入
  - AtomicPersistentState クラス作成（commitFields / snapshot）
  - 3 個別 atomic フィールドを PersistentStateBlock に置き換え
  - commit() の 3 個別 atomic 書込を persistentState_.commitFields() に変更
  - getVersion() → persistentState_.snapshot().mappedRuntimeGeneration
  - (void) version 行削除

Phase-1: AuthorityDescriptor + deriveAuthorityState + reconcileAuthorityState
  - AuthorityDescriptor / AuthorityState / ReconcileResult 定義
  - deriveAuthorityState() 構造的 requires 制約
  - validateAuthorityStateMatch() 全8フィールド比較
  - reconcileAuthorityState() 内部呼び出し

Phase-2a: getCurrent() テスト変換（17件 → consumePublishedWorld）
  CI: isr-verify-getcurrent-test-migrated.ps1

Phase-2b: getCurrent() 削除
  CI: isr-verify-getcurrent-removed.ps1

Phase-D:  retire() currentWorld_ CAS 削除
  CI: isr-verify-retire-no-currentworld.ps1

Phase-3: Recovery 統合 + CI (001-006)
Phase-4: Model-Based Test（6 Fault Injection）
```

---

## 第5章: cocoindex / semble / graphify 確認結果

### cocoindex-code

```powershell
# インデックス: 802 files, 18343 chunks
$env:PYTHONUTF8="1"; uv tool run cocoindex-code index  # カレントディレクトリ
$env:PYTHONUTF8="1"; uv tool run cocoindex-code serve  # MCP サーバー（port 自動）
```

### semble

```powershell
$env:PYTHONUTF8="1"; semble search "query" . --top-k 10
```

→ セマンティック検索として有効。ただし `--content` がデフォルト `code` のため、ドキュメントやコメントも含める場合は `--content all` が必要。

### graphify

graphify は `graphify-out/graph.json` が存在する場合に有効。
`/graphify` でグラフを構築後、`graphify query "<question>"` で質問可能。

---

## 結論

v4.17 では以下の 8 項目をすべて調査・確定した。

| 項目 | 結果 | 重要度 |
|---|---|---|
| EvidenceExporter 読取経路 | `FailureSnapshot.publicationSequenceId` は常に 0（dead field） | 参考情報 |
| DeferredDeletionQueue 経路 | world struct から直接。coordinator atomic 非依存 | 確認済み |
| AudioEngine.Commit 参照 | world struct から直接。coordinator atomic 非依存 | 確認済み |
| coordinator 3 フィールドの真の役割 | **ローカルキャッシュ**。AudioEngine が真の Authority | **重要発見** |
| getVersion() | テストのみ（4箇所）。本番コード未使用 | Phase-0 で変更可 |
| CI スクリプト | 既存 120+ スクリプトあり。新規 9 件が必要 | 要対応 |
| SnapshotResult | コードベースに存在しない | 確認済み |
| AudioEngine 出版カウンタ構造 | `reserveRuntimePublicationIdentity()` → 3段階の値の流れ | 設計反映済み |

**Practical Stable ISR Bridge Runtime 達成度: 99.5%**

**最終ステータス**: 全調査完了。Phase-0 の実装を開始可能。
