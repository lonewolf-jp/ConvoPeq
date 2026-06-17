# Work44 実装チェックリスト

- **作成日**: 2026-06-18
- **対象設計**: 001_quarantine_reclaim_design.md / 002_shutdown_quarantine_cleanup.md / 004_practical_stable_compliance.md
- **管理**: 各項目完了後 `✅` に更新

---

## Phase 0: 設計確認

- [x] 設計書001: PR1 再評価ループ設計
- [x] 設計書002: PR2 シャットダウン解放設計
- [x] 設計書003: PR3 調査結果
- [x] 設計書004: Practical Stable ISR 7条件適合性評価
- [x] ソースコード調査（Serena/grep/CodeGraph）完了

---

## Phase I: ISRDSPQuarantine 公開API整備

### I-1 `kMaxSlots` を public に移動

- [x] `ISRDSPQuarantine.h`: `kMaxSlots = 256` を private から public セクションに移動
- [x] 参照箇所の整合性確認

### I-2 `isActive()` 追加

- [x] `ISRDSPQuarantine.h`: 宣言追加
- [x] `ISRDSPQuarantine.cpp`: 実装追加

### I-3 `compactAuditLog()` を public に移動

- [x] `ISRDSPQuarantine.h`: private→public に移動

### I-4 `reclaimSlot()` generation=0 対応

- [x] `ISRDSPQuarantine.cpp`: `generation != 0` の場合のみ generation 一致確認

---

## Phase II: PR1 通常運用時 quarantine 再評価ループ

### II-1 `AudioEngine.Commit.cpp` 修正

- [x] `onRuntimeRetiredNonRt()` の `pendingIntents` ループ直後に再評価ブロックを追加
- [x] `DSPHandleRuntime::MAX_DSP_SLOTS` を使用
- [x] `isActive()` + `laneOf()` の二重チェック
- [x] grace完了確認
- [x] 3系統解放（reclaim → destroyQuarantineSlot → reclaimSlot）

---

## Phase III: PR2 シャットダウン時 quarantine 全解放

### III-1 `AudioEngine.Processing.ReleaseResources.cpp` 修正

- [x] `shutdownRuntime_.transitionTo(VerifyDrained)` 直前に quarantine 解放ブロックを挿入
- [x] GracefulDrain完了後であることを前提
- [x] 全256スロット走査
- [x] 3系統解放（destroyQuarantineSlot → destroyForShutdown → reclaim）
- [x] バッチ `compactAuditLog()`

---

## Phase IV: 検証・確認

### IV-1 静的検証

- [x] 既存コードとの整合性確認（grep）
- [x] コーディング規約準拠確認
- [x] デッドコード・未使用変数がないことの確認

### IV-2 ビルド確認

- [x] Debug ビルド成功（警告0）
- [x] リンクエラーなし

### IV-3 退行テスト

- [ ] 既存ユニットテスト全PASS
- [ ] CIゲートチェック通過

---

## 変更ファイル一覧

| # | ファイル | 変更内容 | ステータス |
|---|---------|---------|-----------|
| 1 | `src/audioengine/ISRDSPQuarantine.h` | kMaxSlots public化, isActive/compactAuditLog公開 | ✅ |
| 2 | `src/audioengine/ISRDSPQuarantine.cpp` | reclaimSlot gen=0対応, isActive実装 | ✅ |
| 3 | `src/audioengine/AudioEngine.Commit.cpp` | PR1再評価ループ追加 | ✅ |
| 4 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | PR2シャットダウン解放追加 | ✅ |
