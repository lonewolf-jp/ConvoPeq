# Phase 4 代替設計: Builder/Runtime 二段階モデル

> 作成日: 2026-06-28
> ステータス: 設計案 (実装未)
> 根拠: RESOLUTION_REPORT.md §1.3

---

## 1. 設計の背景

### 1.1 課題の再確認

従来の `using RuntimePublishWorld = FrozenRuntimeWorld;` エイリアス置換は以下の理由で不適切:

| 課題 | 詳細 |
|------|------|
| C++ `operator->` 制約 | 生ポインタ `FrozenRuntimeWorld*` の `->` は `operator->()` を経由せず、`FrozenRuntimeWorld` のメンバを直接探索する |
| Builder mutable 競合 | Builder/Orchestrator は publish 前に `assertMutable()` で mutable アクセスを必要とする |
| 261箇所の変更 | `world.X` → `(*world)->X` への変換は Practical Stable の「小さい変更」思想に反する |

### 1.2 設計方針

```
変更の最小化:  Coordinator テンプレートの World パラメータは RuntimeState のまま維持
型安全性:      publish 境界で FrozenRuntimeWorld を使用（RAII + 型レベルの不変性保証）
互換性:        既存の world->field アクセス構文は一切変更しない
```

---

## 2. アーキテクチャ

### 2.1 全体構造

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Builder (mutable)                                  │
│                                                             │
│  RuntimeState (aligned_unique_ptr)                          │
│    ├── RuntimeBuilder::buildRuntimePublishWorld()            │
│    │     → フィールド設定 (mutable access)                  │
│    │     → freeze() は coordinator.publishWorld が実行      │
│    └── RuntimeBuilder::createBootstrapWorld()               │
│          → フィールド設定 + freeze() を自前で実行            │
│                                                             │
│  Coordinator template: World = RuntimeState                 │
│    → publishWorld(aligned_unique_ptr<RuntimeState>)         │
│    → sealRecursively() (RuntimeState::sealRecursively)     │
│    → release() → RuntimeState* を store に格納              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Runtime (immutable)                                │
│                                                             │
│  store 保持: RuntimeState* (const 経由でのみアクセス)       │
│    → consumeWorldHandle → const RuntimeState*              │
│    → world->field がそのまま動作 (変更不要)                 │
│                                                             │
│  Retire 経路:                                               │
│    → bridge.retireRuntimePublishWorldNonRt(world)           │
│    → ptr->unseal()  ← 新規追加                             │
│    → ptr->~RuntimeState()                                   │
│    → aligned_free(ptr)                                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 型の関係

```
RuntimeState (mutable, Builder専用)
  │
  ├── Coordinator テンプレート World = RuntimeState
  │     → store は RuntimeState* を保持
  │     → consumeWorldHandle → const RuntimeState* → world->field 動作
  │
  └── FrozenRuntimeWorld (RAII wrapper, builder/publisher 境界のみ)
        ├── aligned_unique_ptr<RuntimeState> を保持
        ├── デストラクタで unseal() + aligned_free()
        ├── const access のみ (operator-> / get / getRaw)
        └── Coordinator の World パラメータには使用しない
```

---

## 3. 変更箇所一覧

### 3.1 必須変更 (3箇所)

#### 変更1: Bridge retire に unseal() 追加

**ファイル**: `src/audioengine/AudioEngine.h`
**関数**: `retireRuntimePublishWorldNonRt`

```cpp
// 変更前:
void retireRuntimePublishWorldNonRt(RuntimePublishWorld* world, bool resetRevision) noexcept
{
    if (world == nullptr) return;
    engine_->enqueueDeferredDeleteNonRt(world, [](void* p) {
        auto* ptr = static_cast<RuntimePublishWorld*>(p);
        ptr->~RuntimePublishWorld();        // ← unseal なし
        convo::aligned_free(ptr);
    });
}

// 変更後:
void retireRuntimePublishWorldNonRt(RuntimePublishWorld* world, bool resetRevision) noexcept
{
    if (world == nullptr) return;
    engine_->enqueueDeferredDeleteNonRt(world, [](void* p) {
        auto* ptr = static_cast<RuntimePublishWorld*>(p);
        ptr->unseal();                      // ★ SealedObject の unseal
        ptr->~RuntimePublishWorld();
        convo::aligned_free(ptr);
    });
}
```

**根拠**: `RuntimeState` は `SealedObject<RuntimeState>` を継承しており、`unseal()` メソッドを持つ。Retire 時に unseal を呼ぶことで、publish 時に設定された Sealed/Sealed_Recursive 状態を解放する。これにより FrozenRuntimeWorld が行っていた RAII unseal と同じ安全性を確保する。

#### 変更2: Coordinator コメント更新

**ファイル**: `src/audioengine/AudioEngine.h`

```cpp
// 変更前:
// ★ Phase4 (DEFERRED): RuntimePublishWorld = FrozenRuntimeWorld に変更予定
//   ...
//   (DEFERRED の理由が長く書かれている)

// 変更後:
// ★ Phase4 (RESOLVED): Builder/Runtime 二段階モデル
//   Coordinator テンプレート World = RuntimeState (維持)
//   Builder: RuntimeState (mutable)
//   Runtime: const RuntimeState* (immutable, 既存 world->field 互換)
//   Retire時: bridge が unseal() を呼び出し (AudioEngine.h retireRuntimePublishWorldNonRt)
//   FrozenRuntimeWorld: builder 境界の RAII wrapper (coordinator 非使用)
```

#### 変更3: FrozenRuntimeWorld コメント更新

**ファイル**: `src/audioengine/FrozenRuntimeWorld.h`

用途を Coordinator 非依存であることを明確に文書化。

---

### 3.2 オプション変更 (推奨)

#### 変更4: FrozenRuntimeWorld を publish 境界で使用

Builder が FrozenRuntimeWorld を返すように変更し、Publisher が `RuntimeState*` を抽出して Coordinator に渡す。

```cpp
// RuntimeBuilder 変更案:
convo::aligned_unique_ptr<FrozenRuntimeWorld>
RuntimeBuilder::buildRuntimePublishWorld(...) noexcept
{
    auto state = RuntimeState::createForBuilder(token);
    // ... フィールド設定 (mutable) ...
    // freeze は caller (coordinator.publishWorld) が行う
    // FrozenRuntimeWorld でラップして返す
    return convo::aligned_make_unique<FrozenRuntimeWorld>(std::move(state));
}

// PublicationExecutor 変更案:
PublishResult PublicationExecutor::publish(
    AudioEngine& engine,
    convo::aligned_unique_ptr<FrozenRuntimeWorld> frozen) noexcept
{
    // FrozenRuntimeWorld から RuntimeState* を抽出
    // (const を外すのは正当: publish が所有権を coordinator に移譲するため)
    auto* rawState = const_cast<RuntimeState*>(frozen->getRaw());

    // aligned_unique_ptr<RuntimeState> でラップして coordinator に渡す
    // 注意: 所有権は coordinator に移るため frozen の解放は coordinator が担当する
    auto stateOwner = aligned_unique_ptr<RuntimeState>(rawState);
    frozen.release();  // 所有権移譲のため release

    auto coordinator = engine.makeRuntimePublicationCoordinator();
    return coordinator.publishWorld(std::move(stateOwner));
}
```

**この変更の効果**:
- Builder の戻り値型が `FrozenRuntimeWorld` になり、型レベルで「凍結済み」を表明
- 誤った mutable アクセスをコンパイル時に防止
- Coordinator は従来通り `RuntimeState*` で動作（互換性維持）

**コスト**:
- `const_cast` が必要（`FrozenRuntimeWorld::getRaw()` が `const RuntimeState*` を返すため）
- ただしこれは正当: 所有権移譲の過渡期であり、移譲後は coordinator が所有権を持つ

---

## 4. Coordinator テンプレートとの整合性

```cpp
// AudioEngine.h (変更なし)
using RuntimePublicationCoordinator = convo::RuntimePublicationCoordinator<
    RuntimeState,        // World = RuntimeState (不変)
    DSPCore*,            // Handle
    RuntimePublicationBridge  // Bridge
>;
```

`World = RuntimeState` のままなので:
- `publishWorld(aligned_unique_ptr<RuntimeState>)` ✅
- `consumeWorldHandle` → `const RuntimeState*` ✅
- `world->engine`, `world->graph`, `world->generation` ✅ (全261箇所)
- `bridge_.validatePublicationNonRt(const RuntimeState&)` ✅
- `bridge_.retireRuntimePublishWorldNonRt(RuntimeState*)` ✅ (unseal 追加後)

---

## 5. メリット・デメリット

### メリット
| 項目 | 詳細 |
|------|------|
| **既存コード変更ゼロ** | 全261箇所の `world->field` は変更不要 |
| **ビルド・テスト安定** | `using` 変更によりコンパイルエラーが発生しない |
| **段階的導入可能** | 必須変更3箇所のみ。オプション変更は後日 |
| **型安全性** | Builder→Runtime の境界が明確化 |
| **RAII 安全性** | unseal が retire 経路で確実に呼ばれる |

### デメリット
| 項目 | 詳細 |
|------|------|
| **コンパイル時不変性** | `const RuntimeState*` は runtime の不変性のみ保証 (FrozenRuntimeWorld の compile-time 不変性は無い) |
| **const_cast が必要** | オプション変更4で FrozenRuntimeWorld から RuntimeState* 抽出時に必要 |
| **FrozenRuntimeWorld の活用度** | Coordinator 非依存なので「型レベルのお守り」的役割に留まる |

### 不変性の保証レベル比較

| 方式 | コンパイル時 | ランタイム | コード変更量 |
|------|------------|-----------|------------|
| 現在 (`RuntimeState*`) | ❌ | ⚠️ (freeze 後の mutation は未チェック) | 0 |
| **二段階モデル** | ❌ (Runtime 側) | ✅ (unseal + freeze assert) | **3-7** |
| 旧 alias 置換 | ✅ | ✅ | 261+ |

**結論**: 二段階モデルは完全なコンパイル時不変性は提供しないが、ランタイムの不変性を維持しつつ、コード変更量を最小化する。

---

## 6. 実装ステップ

### Step 1: Bridge retire に unseal() 追加（最重要）
```cpp
// AudioEngine.h の retireRuntimePublishWorldNonRt
ptr->unseal();  // 追加
```
所要時間: 5分。リスク: 最小（unseal は冪等）。

### Step 2: コメント更新
- AudioEngine.h Phase4 コメント更新
- FrozenRuntimeWorld.h コメント更新

### Step 3: ビルド・テスト確認
```bash
.\build.bat Release nopause
cd build && ctest -C Release
```

### Step 4 (Optional): Builder 戻り値型変更
- `RuntimeBuilder` が `FrozenRuntimeWorld` を返すように変更
- `PublicationExecutor` で抽出ロジック追加

---

## 7. 設計判断の記録

| 判断 | 内容 | 日付 |
|------|------|------|
| DEFERRED | 旧 alias 置換は C++ 制約により不採用 | 2026-06-28 |
| 二段階モデル策定 | Coordinator World=RuntimeState 維持 + retire unseal | 2026-06-28 |
| FrozenRuntimeWorld | Coordinator 非依存の builder 境界 RAII として維持 | 2026-06-28 |

---

## 8. 付録: 実装前の冪等性確認

`SealedObject::unseal()` の実装を確認:

```cpp
// ISRSealedObject.h
void unseal() noexcept
{
    convo::publishAtomic(sealState_, SealState::Unsealed, std::memory_order_release);
}
```

`unseal()` は `SealState` を `Unsealed` に設定するのみで、以下の特性を持つ:
- **冪等**: 複数回呼んでも安全（最後の Unsealed が勝つ）
- **const 不要**: 非 const メソッドだが、retire 経路では所有権が coordinator にあり、他スレッドが参照していない
- **軽量**: atomic store 1回のみ

✅ `unseal()` 呼び出しは安全。
