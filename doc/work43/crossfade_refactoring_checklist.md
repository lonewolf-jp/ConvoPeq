# Crossfade Refactoring — 全体実装チェックリスト

- **作成日**: 2026-06-17
- **管理計画**: `doc/work43/PR1_*.md` 〜 `doc/work43/PR4_*.md`
- **関連ルール**: `doc/rule4-coding.md`, `doc/detailed_design_plan_rule_jp.md`, `doc/implementation_preflight_checklist.md`

---

## 0. アーキテクチャ原則（全 PR 共通）

1. **CrossfadeId 発行源は1箇所**: CrossfadeAuthorityRuntime のみ
2. **CrossfadeRuntime は ID を生成/管理しない**: Execution Cache として振る舞う
3. **crossfadeRecords_ は維持**: 複数 Crossfade 対応のためベクタ温存
4. **endCrossfade(CrossfadeId id) は維持**: ID→状態遷移の契約
5. **activeCrossfadeId_ は移設せず廃止**: 新しい中間状態を追加しない
6. **Authority の Record 集約は将来課題**: 現時点では二重管理を許容

---

## 1. PR1: beginCrossfade ID 統合 — チェックリスト

**ステータス**: ✅ 実装完了 (2026-06-17)
**詳細設計**: `doc/work43/PR1_beginCrossfade_ID統合.md`

### 1.1 実装前確認

- [x] 現在の `beginCrossfade` の呼び出し元を grep で確認（ゼロであること）
- [x] `nextCrossfadeId_` の全参照を grep で確認（`fetchAddAtomic` のみ）
- [x] `DSPLifetimeManager::beginCrossfade` の呼び出し元を grep で確認

### 1.2 実装項目

- [x] `ISRDSPHandle.h`: `beginCrossfade` シグネチャ変更
- [x] `ISRDSPHandle.h`: `nextCrossfadeId_` フィールド削除
- [x] `ISRDSPHandle.cpp`: `beginCrossfade` 実装変更（fetchAddAtomic → 引数 id）
- [x] `DSPLifetimeManager.h`: 引数 `id` 追加、戻り値 `void` 化
- [x] `DSPTransition.h`: `(void)xfadeId` → `beginCrossfade(from, to, xfadeId)`
- [x] `DSPTransition.h`: `publishAtomic(0)` → `publishAtomic(xfadeId)`

### 1.3 実装後確認

- [x] 全 `beginCrossfade` 参照が新シグネチャに一致していること
- [x] `nextCrossfadeId_` が全ソースコードから消滅したこと
- [x] ビルドが通ること（Debug）
- [x] generation-drift スクリプトのチェック2-a が PASS すること

### 1.4 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ISRDSPHandle.h` | `beginCrossfade(from, to)` → `(from, to, id)`、`nextCrossfadeId_` 削除 |
| `ISRDSPHandle.cpp` | `fetchAddAtomic` 削除、引数 `id` 使用 |
| `DSPLifetimeManager.h` | 引数 `id` 追加、戻り値 `void` |
| `DSPTransition.h` | `beginCrossfade` 呼び出し追加、`publishAtomic(0)` → `(xfadeId)` |

---

## 2. PR2: activeCrossfadeId_ 単一前提の表明 — チェックリスト

**ステータス**: 📋 設計済み（未実装）
**詳細設計**: `doc/work43/PR2_activeCrossfadeId_jassert.md`

### 2.1 実装前確認

- [ ] `activeCrossfadeId_` の全参照を grep で再確認
- [ ] `getActiveCrossfades()` が正常動作することを確認
- [ ] デバッグビルドで `jassert` が有効であることを確認

### 2.2 実装項目

- [ ] `AudioEngine.Timer.cpp` 完了検出経路に `jassert + getActiveCrossfades()` 追加
- [ ] `AudioEngine.Timer.cpp` タイムアウト回復経路に `getActiveCrossfades()` 全件ループ + jassert + diagLog 追加
- [ ] `.github/scripts/isr-verify-phase4-generation-drift.ps1`: `DSPTransition.h` を検査対象に追加
- [ ] `.github/scripts/isr-verify-phase4-generation-drift.ps1`: DSPTransition 用の `beginCrossfade` + `activeCrossfadeId_ publish` チェック追加

### 2.3 実装後確認

- [ ] デバッグビルドが通ること
- [ ] リリースビルドで `jassert` が消滅していること
- [ ] 通常 Crossfade 完了時に `jassert` が発火しないこと
- [ ] CI ゲートが PASS すること

### 2.4 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `AudioEngine.Timer.cpp` | 完了検出: `consumeAtomic` → `getActiveCrossfades()` + jassert |
| `AudioEngine.Timer.cpp` | タイムアウト: `consumeAtomic` → `getActiveCrossfades()` + jassert + 全件ループ |

---

## 3. PR4: activeCrossfadeId_ 完全削除 — チェックリスト

**ステータス**: 📋 設計済み（未実装）
**詳細設計**: `doc/work43/PR4_activeCrossfadeId_削除.md`

### 3.1 実装前確認

- [ ] PR2 が完了し、Timer が `getActiveCrossfades()` で ID を取得するようになったこと
- [ ] `activeCrossfadeId_` の全参照を grep で完全に棚卸し
- [ ] Timer の完了経路とタイムアウト経路に `publishAtomic(activeCrossfadeId_, 0)` が残っていること
- [ ] DSPTransition に `publishAtomic(activeCrossfadeId_, xfadeId)` が残っていること
- [ ] ReleaseResources.cpp に3箇所の `activeCrossfadeId_` 参照があること

### 3.2 実装項目

- [ ] `AudioEngine.h`: `activeCrossfadeId_` 宣言削除（L3608）
- [ ] `DSPTransition.h`: `publishAtomic(activeCrossfadeId_, ...)` 削除
- [ ] `AudioEngine.Timer.cpp` 完了検出: `publishAtomic(activeCrossfadeId_, 0)` 削除
- [ ] `AudioEngine.Timer.cpp` タイムアウト: `publishAtomic(activeCrossfadeId_, 0)` 削除
- [ ] `ReleaseResources.cpp` EmergencyDrain: `publishAtomic(0)` 削除
- [ ] `ReleaseResources.cpp` VerifyDrained: `publishAtomic(0)` 削除
- [ ] `ReleaseResources.cpp` shutdown counter: `consumeAtomic` → `isPending()`
- [ ] `AudioEngine.Threading.cpp`: 必要に応じて修正（`isPending()` 確認）
- [ ] `.github/scripts/isr-verify-phase4-generation-drift.ps1`: 検査対象を `DSPTransition.h` に更新、`AudioEngine.Commit.cpp` を削除
- [ ] `.github/scripts/isr-verify-phase4-generation-drift.ps1`: 全チェックを Authority→SPSC 版に更新
- [ ] `.github/scripts/isr-verify-phase4-generation-drift.ps1`: `activeCrossfadeId_` 全ソース消滅チェック追加

### 3.3 実装後確認

- [ ] ビルドが通ること（Debug + Release）
- [ ] `activeCrossfadeId_` が全ソースコードから消滅したこと
- [ ] 通常 Crossfade 完了が正しく動作すること
- [ ] タイムアウト回復が正しく動作すること
- [ ] shutdown が正しく動作すること
- [ ] generation-drift スクリプトが PASS すること
- [ ] CI ゲートが PASS すること

### 3.4 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `AudioEngine.h` | `activeCrossfadeId_` 宣言削除 |
| `DSPTransition.h` | `publishAtomic(activeCrossfadeId_, ...)` 削除 |
| `AudioEngine.Timer.cpp` | 完了検出 + タイムアウト: `publishAtomic(0)` 削除 |
| `ReleaseResources.cpp` | 3箇所の参照削除（`isPending()` 化含む） |
| `isr-verify-phase4-generation-drift.ps1` | チェック全面更新 |
| `AudioEngine.Threading.cpp` | 必要に応じて修正 |

---

## 5. 全体進捗サマリ

| PR | 内容 | 状態 | 影響ファイル数 |
|----|------|------|---------------|
| PR1 | beginCrossfade ID 統合 | ✅ **完了** | 4 |
| PR2 | activeCrossfadeId_ 単一前提表明 | 📋 設計済み | 1 |
| PR4 | activeCrossfadeId_ 完全削除（PR2 前提） | 📋 設計済み | 5〜6 |

### 凡例

- ✅ 完了
- 🔄 実装中
- 📋 設計済み（未実装）
- ⏳ 未着手

---

## 6. 依存関係

```
PR1 ──→ PR2 ──→ PR4
```

PR3 は削除（AudioThread 完了検出は採用せず、Timer 権威維持）。
PR2 完了後に PR4 を実施。PR4 のみで `activeCrossfadeId_` を完全削除する。

---

## 7. ロールバック方針

| ケース | 対応 |
|--------|------|
| PR1 失敗 | `git revert` で `beginCrossfade` + `nextCrossfadeId_` の変更を戻す |
| PR2 失敗 | `git revert` で jassert + getActiveCrossfades の変更を戻す |
| PR4 失敗 | `git revert` で activeCrossfadeId_ 削除の変更を戻す |

各 PR は独立したコミットまたは PR として管理し、部分的な切り戻しを可能にする。
