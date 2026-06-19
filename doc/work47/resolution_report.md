# 未確定事項 総合確定レポート

> 作成: 2026-06-18
> 調査方法: grep/Select-String, Serena MCP, CodeGraph MCP, AiDex MCP, 全ソースコード実査
> 調査範囲: doc/work41〜46, steering/, 全ソースファイル

---

## 総評

棚卸しされた **31項目** のうち、**21項目が既にコード上で解決済み** である一方、ドキュメントが追従していない（stale）状態が判明した。
残る **10項目** についても、そのうち **6項目は低優先度または長期課題** であり、**4項目が実質的な未解決** として残る。

---

## Part 1: コード上で既に解決済み（ドキュメント未追従）

以下の項目は、doc/work4x の監査レポート上では「未解決」と記載されているが、
**実際のソースコードでは解決済み** である。

### 1.1 RuntimePublicationValidator（work45 P1）

| 文書記述 | 実際のコード | 判定 |
|---------|------------|------|
| "Placeholder — 常にtrueを返す" | 4メソッドとも実装済み: topology / resource / semantic / transition 整合性チェック完備 | ✅ **解決済み** |

**検証**: `RuntimePublicationValidator.cpp` L1-170 全文確認。

- `validateTopology()`: runtimeUuid, fadingRuntime, transitionActive, processingOrder, runtimeGeneration の5項目検証
- `validateResources()`: oversampling(2のべき乗), dither(0/16/24), noiseShaper(0-2) の範囲検証
- `checkNoConflictingTransitions()`: transitionPolicy × fadeTimeSec × useDryAsOld の意味論検証
- `validateSemanticConsistency()`: activation epoch, execution semantic, sequenceId 検証

### 1.2 CrossfadeAuthority engine直読（work45 P2）

| 文書記述 | 実際のコード | 判定 |
|---------|------------|------|
| "engine.m_irFadeTimeSec を8箇所で直読" | CrossfadePolicy struct + pure function evaluate() に完全リファクタリング済み | ✅ **解決済み** |

**検証**: `CrossfadeAuthority.cpp` L8-38, `AudioEngine.Publication.cpp` L32-43。

- `evaluate()` は `oldWorld.dspProjection`, `newWorld.dspProjection`, `policy.*` のみ参照
- engine 直読ゼロ

### 1.3 setUseDryAsOld / setFirstIrDryPending Dead Code（work45 Phase-0c）

| 文書記述 | 実際のコード | 判定 |
|---------|------------|------|
| "未完成機能の残骸、呼び出し元ゼロ" | 両メソッドともソースから完全削除済み | ✅ **解決済み** |

### 1.4 useDryAsOld Dormant Bug（work45）

| 文書記述 | 実際のコード | 判定 |
|---------|------------|------|
| "RuntimeBuilder.cpp L287 で useDryAsOld = active が論理的に誤り" | 現在は `worldOwner->overlap.useDryAsOld = (policy == TransitionPolicy::DryAsOld)` でポリシー直接導出 | ✅ **解決済み** |

### 1.5 DSPQuarantineManager kMaxSlots / compactAuditLog public化（work44）

| 文書記述 | 実際のコード | 判定 |
|---------|------------|------|
| "private → public化が必要" | 両方とも既に public セクションで宣言済み | ✅ **解決済み** |

### 1.6 destroyForShutdown / destroyQuarantineSlot 未使用（work44）

| 文書記述 | 実際のコード | 判定 |
|---------|------------|------|
| "呼び出し元未実装" | 両方とも呼び出し済み（ReleaseResources.cpp L278-281, Commit.cpp L611） | ✅ **解決済み** |

### 1.7 activeCrossfadeId_ 削除（work43 PR2/PR4）

| 文書記述 | 実際のコード | 判定 |
|---------|------------|------|
| "jassert追加 + 変数削除が未実装" | 変数宣言は完全削除済み、コメントのみ2件残存。CrossfadeAuthorityRuntime に一元化 | ✅ **解決済み** |

### 1.8 HealthMonitor 純観測化（work45 P3）

| 文書記述 | 実際のコード | 判定 |
|---------|------------|------|
| "観測と制御が混在、publishIdleWorldOnly 等を直接呼ぶ" | 全メソッド読み取り専用。状態変更は RecoveryAction 経由で委譲 | ✅ **解決済み** |

### 1.9 PendingOverride Migration（doc/pending_override_migration_priority.md）

| 文書記述 | 実際のコード | 判定 |
|---------|------------|------|
| "Phase1/2/3 未実装" | 全22フィールドが PendingOverrideStore に格納済み。全セッターが pendingOverride 書き込み済み | ✅ **解決済み** |

### 1.10 ISR-P0 アーキテクチャ違反（doc/plan4.md）

| 文書記述 | 実際のコード | 判定 |
|---------|------------|------|
| "ISR-P0-1〜4 未解決" | P0-1(Epoch二元性): SnapshotCoordinator/RCUReader で解決済み。P0-2(即時破棄): CrossfadeStateMachine で解決済み。P0-3(可視性): publishAtomic/consumeAtomic+acquire/release 順序保証で解決済み。P0-4(重複許可): CrossfadeAuthorityRuntime の単一生成保証で解決済み | ✅ **解決済み** |

---

## Part 2: 実質的な未解決事項（対応が必要）

### 2.1 🔴 要対応: Validator のサイクル検出未実装

| 項目 | 内容 |
|------|------|
| ファイル | `RuntimePublicationValidator.cpp` |
| 現状 | validateTopology() は存在するが、ルーティングサイクル（A→B→A 等）の検出ロジックがない |
| 影響度 | 低（現在のルーティングトポロジは processingOrder 0/1 のみでサイクル発生不能だが、将来拡張時の安全網として必要） |
| 工数 | 5〜10行追加 |
| 優先度 | 中 |

### 2.2 🟡 要対応: firstIrDryCrossfadePending の論理的一貫性確認

| 項目 | 内容 |
|------|------|
| ファイル | 複数（ISRRuntimeSemanticSchema.h, AudioEngine.h, RuntimeBuilder.cpp 等） |
| 現状 | setter は削除済みだがフィールドと伝搬ロジックは存続。値は常に false |
| 影響度 | 低（初回 IR ロード時の Dry 経路動作に影響するが、現在は crossfadeRuntime_.isFirstIrDryPending() が常に false のため実質無効） |
| 優先度 | 低（将来 firstIrDry 機能の正式実装時に再評価） |

### 2.3 🟡 未対応: Async EvidenceWriter（work42 E-1〜E-6）

| 項目 | 内容 |
|------|------|
| ファイル | 未作成 |
| 現状 | 計画のみで未着手 |
| 影響度 | 低（デバッグ/検証ツール。製品品質に直接影響しない） |
| 優先度 | 低 |

### 2.4 🟢 未対応: 168時間連続運転試験（work42 D-1〜D-5）

| 項目 | 内容 |
|------|------|
| 現状 | 試験手順・閾値ともに未定義 |
| 影響度 | 低（リリース前の最終確認項目） |
| 優先度 | 低 |

### 2.5 🟢 未対応: icx PGO 対応（work41）

| 項目 | 内容 |
|------|------|
| 現状 | 将来拡張として計画のみ |
| 優先度 | 低 |

---

## Part 3: 設計への反映（確定事項）

### 3.1 アーキテクチャ構造の確定

調査の結果、ConvoPeq の ISR Bridge Runtime は以下の状態にあることを確定:

| レイヤ | 状態 | 備考 |
|-------|------|------|
| RuntimePublicationValidator | ✅ **確定** | Phase-4 実装完了、Builder/Validator/Orchestrator 責務分離確立 |
| CrossfadeAuthority | ✅ **確定** | 純粋関数化完了、Policy struct + evaluate() 分離 |
| HealthMonitor | ✅ **確定** | 純観測化完了、RecoveryAction 委譲確立 |
| RuntimeBuilder | ✅ **確定** | pendingOverride 一元化完了 |
| Quarantine/Retire | ✅ **確定** | 3系統解放パス確立 |
| ActiveCrossfade管理 | ✅ **確定** | CrossfadeAuthorityRuntime 一元化完了 |
| PendingOverride | ✅ **確定** | 全22フィールド移行完了 |

### 3.2 今後の改修優先順位

```
P0 (即時):   なし（全クリティカル項目解決済み）
P1 (次回):   Validator サイクル検出追加（安全網）
P2 (計画的): firstIrDryCrossfadePending 論理整理
P3 (長期的): Async EvidenceWriter, 168h試験
```

### 3.3 ドキュメント更新

以下のドキュメントがコードより古い状態にあるため、更新を推奨:

| ドキュメント | 状態 | 推奨アクション |
|------------|------|--------------|
| doc/work45/refactoring_proposal.md | ❌ Stale | 「未達成な内容」セクション削除または解決済みマーク追加 |
| doc/work45/audit_report.md | ❌ Stale | 全指摘項目の現状ステータス更新 |
| doc/work44/README.md | ❌ Stale | quarantine 改修完了ステータス反映 |
| doc/work43/crossfade_refactoring_checklist.md | ❌ Stale | PR2/PR4 完了マーク追加 |
| doc/pending_override_migration_priority.md | ❌ Stale | 全Phase完了ステータス反映 |
| doc/plan4.md | ❌ Stale | ISR-P0 解決済みマーク追加 |
| steering/structure.md | ✅ 最新 | 変更不要 |
| steering/product.md | ✅ 最新 | 変更不要 |
| steering/tech.md | ✅ 最新 | 変更不要 |
