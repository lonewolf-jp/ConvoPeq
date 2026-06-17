# Work44: シャットダウンチリチリノイズ原因調査と修正設計

- **作成日**: 2026-06-17
- **起因**: 起動ログ `ConvoPeq.log` (2026-06-17 21:24) 分析による発見
- **関連Issue**: シャットダウン前の可聴ノイズ（チリチリ）

---

## 発見された3つの問題

| # | 問題 | 深刻度 | 設計書 |
|---|------|--------|--------|
| 1 | Quarantine滞留 — 解放パス欠落による10スロット蓄積 | **Critical** | [001_quarantine_reclaim_design.md](001_quarantine_reclaim_design.md) |
| 2 | シャットダウン時quarantine未解放 — `destroyForShutdown()` 未使用 | **High** | [002_shutdown_quarantine_cleanup.md](002_shutdown_quarantine_cleanup.md) |
| 3 | リタイアルータ保留2件 — 意図的だが監視強化が必要 | **Low** | [003_router_retire_investigation.md](003_router_retire_investigation.md) |

## ログ上の証拠

```
[ISR][Shutdown] Drain incomplete:
  pendingPub=0 pendingRetire=0 crossfade=0
  routerPendingRetire=2               ← 問題3
  maxDeferredAgeMs=0 deferred=0
  quarantine=10                        ← 問題1+2
  oldestAgeMs=378750                   ← ★ 最古約6.3分滞留
```

## 優先順位

1. **PR1**: Quarantine解放パス追加（問題1）— 通常動作中のメモリリーク解消
2. **PR2**: シャットダウン時quarantine全解放（問題2）— シャットダウン安全化
3. **PR3**: リタイアルータ保留の監視強化（問題3）— 情報収集・監視

---

## 用語定義

| 用語 | 説明 |
|------|------|
| Quarantine | リタイア処理中に解放条件未成立で隔離されたDSPスロット（3系統管理） |
| Case C | `onRuntimeRetiredNonRt()` 内の3分岐のうち、quarantineのみ行いreclaimしない経路 |
| RouterPendingRetire | `EpochDomain::deferredDeletionQueue` に滞留中のEBR削除待ちアイテム数 |
| `destroyForShutdown` | `DSPQuarantineManager` に定義済みだが**未使用**のシャットダウン解放関数 |

---

## 調査状況（2026-06-17 確定）

全3件の設計書について、以下のツールを用いたソースコード調査を完了：

| ツール | 用途 | 結果 |
|--------|------|------|
| grep / Select-String | API実装状況・呼び出し元確認 | `destroyForShutdown` の呼び出し元ゼロを確認 |
| Serena MCP (oraios) | コード構造・シンボル解析 | 3系統管理の全容を把握 |
| CodeGraph MCP | インデックス＋構造解析 | 主要ファイル間の依存関係確認 |
| AiDex MCP | プロジェクトインデックス | — |
| 手動ファイル読み取り | `EpochDomain`, `ISRRetireRuntimeEx` 等 | 全動作を確認 |

### 確定した未解決事項

| # | 事項 | 理由 |
|---|------|------|
| 1 | `DSPQuarantineManager::kMaxSlots` が private | public化が必要（PR1） |
| 2 | `DSPQuarantineManager::compactAuditLog()` が private | public化が必要（PR1/PR2） |
| 3 | `DSPHandleRuntime::destroyQuarantineSlot()` が未使用 | 呼び出し追加が必要（PR2） |
| 4 | `DSPQuarantineManager::destroyForShutdown()` が未使用 | 呼び出し追加が必要（PR2） |
| 5 | `routerPendingRetire=2` の残留 | 意図的な安全設計（PR3: 対応不要） |
