# PR1: beginCrossfade ID 統合 — 詳細設計書

- **作成日**: 2026-06-17
- **ステータス**: ✅ 実装完了
- **親計画**: `doc/work43/crossfade_refactoring_checklist.md`
- **関連ルール**: `doc/rule4-coding.md`, `doc/detailed_design_plan_rule_jp.md`

---

## 0. 目的

現在バラバラな CrossfadeId 発行源を **CrossfadeAuthorityRuntime** に一元化し、
`DSPHandleRuntime` の独自 ID 生成を排除する。これにより以下の不整合を解消する：

```
変更前:
  CrossfadeAuthorityRuntime: id=17（registerCrossfadeで発行→捨てられる）
  DSPHandleRuntime:          id=42（beginCrossfadeで独自発行→使われない）
  activeCrossfadeId_:        0   （DSPTransitionが常に0をpublish）

変更後:
  CrossfadeAuthorityRuntime: id=17（唯一権威）
  DSPHandleRuntime:          id=17（Authorityから注入）
  activeCrossfadeId_:        17  （正しいIDをpublish）
```

---

## 1. 変更ファイル一覧

| # | ファイル | 変更種別 |
|---|---------|---------|
| 1 | `src/audioengine/ISRDSPHandle.h` | 宣言変更 + フィールド削除 |
| 2 | `src/audioengine/ISRDSPHandle.cpp` | 実装変更 |
| 3 | `src/audioengine/DSPLifetimeManager.h` | インターフェース変更 |
| 4 | `src/audioengine/DSPTransition.h` | 呼び出し修正 |

---

## 2. 変更詳細

### 2.1 ISRDSPHandle.h

**宣言変更** (L120-121):

```cpp
// 変更前:
CrossfadeId beginCrossfade(DSPHandle from, DSPHandle to);

// 変更後:
void beginCrossfade(DSPHandle from, DSPHandle to, CrossfadeId id);
```

- 戻り値 `CrossfadeId` → `void`（呼び出し元が既に ID を保持）
- 引数に `CrossfadeId id` 追加（Authority 発行の ID を受け取る）

**フィールド削除** (L161):

```cpp
// 削除:
std::atomic<CrossfadeId> nextCrossfadeId_{1};
```

- このフィールドが二重権威の原因
- `CrossfadeAuthorityRuntime::nextId_` が唯一のカウンタ

### 2.2 ISRDSPHandle.cpp

**実装変更** (L58-69):

```cpp
// 変更前:
CrossfadeId DSPHandleRuntime::beginCrossfade(DSPHandle from, DSPHandle to)
{
    assert(!from.isNull() && !to.isNull());
    convo::publishAtomic(registry_[from.slot].state, DSPState::CrossfadingOut, ...);
    convo::publishAtomic(registry_[to.slot].state, DSPState::CrossfadingIn, ...);
    const auto id = convo::fetchAddAtomic(nextCrossfadeId_, 1u, ...);  // ← 独自生成
    crossfadeRecords_.push_back(CrossfadeRecord{ id, from, to, 0u, true });
    convo::publishAtomic(fadingRuntimeDSPHandle_, from, ...);
    return id;
}

// 変更後:
void DSPHandleRuntime::beginCrossfade(DSPHandle from, DSPHandle to, CrossfadeId id)
{
    assert(!from.isNull() && !to.isNull());
    convo::publishAtomic(registry_[from.slot].state, DSPState::CrossfadingOut, ...);
    convo::publishAtomic(registry_[to.slot].state, DSPState::CrossfadingIn, ...);
    crossfadeRecords_.push_back(CrossfadeRecord{ id, from, to, 0u, true });  // 外部ID
    convo::publishAtomic(fadingRuntimeDSPHandle_, from, ...);
}
```

- `fetchAddAtomic(nextCrossfadeId_, ...)` を削除（内部カウンタ不使用）
- `id` は引数で受け取った値をそのまま使用
- 戻り値削除（`CrossfadeId` → `void`）

### 2.3 DSPLifetimeManager.h

**インターフェース変更** (L26-30):

```cpp
// 変更前:
convo::isr::CrossfadeId beginCrossfade(convo::isr::DSPHandle from, convo::isr::DSPHandle to) noexcept
{
    return engine_.dspHandleRuntime_.beginCrossfade(from, to);
}

// 変更後:
void beginCrossfade(convo::isr::DSPHandle from, convo::isr::DSPHandle to, convo::isr::CrossfadeId id) noexcept
{
    engine_.dspHandleRuntime_.beginCrossfade(from, to, id);
}
```

- 引数 `id` 追加
- 戻り値 `CrossfadeId` → `void`

### 2.4 DSPTransition.h

**呼び出し修正** (L80-86):

```cpp
// 変更前（BUG）:
const auto xfadeId = engine_.crossfadeAuthorityRuntime_.registerCrossfade(oldHandle, newHandle);
(void)xfadeId;  // ★ 捨てている
engine_.publishAtomic(engine_.activeCrossfadeId_,
                     static_cast<convo::isr::CrossfadeId>(0u),  // ★ 常に0
                     std::memory_order_release);

// 変更後:
const auto xfadeId = engine_.crossfadeAuthorityRuntime_.registerCrossfade(oldHandle, newHandle);
engine_.dspHandleRuntime_.beginCrossfade(oldHandle, newHandle, xfadeId);  // ★ ID注入
engine_.publishAtomic(engine_.activeCrossfadeId_,
                     static_cast<convo::isr::CrossfadeId>(xfadeId),  // ★ 正しいID
                     std::memory_order_release);
```

---

## 3. 変更しないもの（設計契約の維持）

| 項目 | 理由 |
|------|------|
| `CrossfadeRecord` 型 | DSPHandleRuntime + AuthorityRuntime 両方で使用 |
| `crossfadeRecords_` ベクタ | 複数 Crossfade 対応のために維持 |
| `endCrossfade(CrossfadeId id)` | ID→状態遷移の明確な契約を保持 |
| `isSlotInCrossfade` | `crossfadeRecords_` ベースのロジック維持 |
| `activeCrossfadeId_` | PR4 で削除予定。PR2 で jassert 追加 |
| `CrossfadeRuntime` | ID 非管理を厳守（変更なし） |

---

## 4. 検証項目

- [ ] ビルドが通ること（Debug + Release）
- [ ] `generation-drift` スクリプトのチェック2-a (`beginCrossfade` 呼び出し) が PASS すること
- [ ] `DSPHandleRuntime::beginCrossfade` が `nextCrossfadeId_` を参照していないこと
- [ ] `activeCrossfadeId_` に 0 以外の値が publish されること（= xfadeId が正しく渡る）
- [ ] Timer 側の `consumeAtomic(activeCrossfadeId_)` で正しい ID が読めること（PR2 で検証容易化）
- [ ] DSPHandleRuntime の状態機械が正しく動作すること（CrossfadingOut / CrossfadingIn が設定される）

---

## 5. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| `DSPLifetimeManager::beginCrossfade` の呼び出し元が旧シグネチャを使用 | コンパイルエラー | grep で全呼び出し元を確認済み（DSPTransition.h のみ） |
| `nextCrossfadeId_` 削除によるリンクエラー | ビルド失敗 | 宣言＋唯一の使用箇所（fetchAddAtomic）を同時削除 |
| ID 値の一貫性崩れ | Authority=DSPHandle 間で不一致 | `beginCrossfade` に注入する ID は常に `registerCrossfade` の戻り値 |

---

## 6. 事後状態

```
DSPTransition (commit):
  1. CrossfadeAuthorityRuntime::registerCrossfade(from, to) → id (唯一権威)
  2. DSPHandleRuntime::beginCrossfade(from, to, id)        ← 同一ID
  3. publishAtomic(activeCrossfadeId_, id)                  ← 正しいID
```
