# C4996 deprecation 警告: EngineRuntime → RuntimeSemanticSchema 移行ガイド

**調査日**: 2026-07-23
**対象**: `AudioEngine.Timer.cpp:469-472` の C4996 警告 4件

---

## 型の関係

deprecation メッセージは `RuntimeSemanticSchema` を推奨していますが、**実際のアクセス先は `RuntimeState::topology`** です。

```
RuntimeState（RuntimeWorld と呼ばれる Runtime Publish オブジェクト）
  ├── engine          ← EngineRuntime（deprecated, 削除予定）
  ├── topology        ← TopologySemantic（推奨）
  ├── routing         ← RoutingSemantic
  ├── execution       ← ExecutionSemantic
  └── ...（その他 Semantic フィールド）
```

`RuntimeSemanticSchema` は Semantic 群全体を表す設計上の名称であり、実際の実装は `RuntimeState` が保持する `topology`、`routing`、`execution` などの各 Semantic フィールドで構成されます。Timer.cpp では `runtimeWorld->topology` に直接アクセスします。

`EngineRuntime` は既存コードとの互換性維持のため現在も `RuntimeState` 内に保持されていますが、Authority は既に `TopologySemantic` へ移行済みであり、新規コードでは参照しません。

## 原因

`EngineRuntime` 構造体が `[[deprecated("Authority removed, use RuntimeSemanticSchema")]]` で宣言されている。

```cpp
// src/audioengine/RuntimeTransition.h:30
struct [[deprecated("Authority removed, use RuntimeSemanticSchema")]] EngineRuntime
{
    std::uint64_t currentRuntimeUuid = 0;
    std::uint64_t fadingRuntimeUuid = 0;
    std::uint64_t transitionCurrentRuntimeUuid = 0;
    std::uint64_t transitionNextRuntimeUuid = 0;
    // ... その他フィールド
};
```

---

## 対応関係

| 旧 API (`EngineRuntime`) | 新 API (`TopologySemantic`) | 備考 |
|--------------------------|---------------------------|------|
| `engine.currentRuntimeUuid` | `topology.runtimeUuid` | 直接対応 |
| `engine.fadingRuntimeUuid` | `topology.fadingRuntimeUuid` | 直接対応 |
| `engine.transitionCurrentRuntimeUuid` | `topology.runtimeUuid` | **現在の** RuntimeBuilder 実装では同値 |
| `engine.transitionNextRuntimeUuid` | `topology.fadingRuntimeUuid` | **現在の** RuntimeBuilder 実装では同値 |

---

## 定義ファイル

- **EngineRuntime**: `src/audioengine/RuntimeTransition.h:30`
- **TopologySemantic**: `src/audioengine/ISRRuntimeSemanticSchema.h:207-208`
- **RuntimeState**: `src/audioengine/AudioEngine.h:135`
- **FrozenRuntimeWorld**: `src/audioengine/FrozenRuntimeWorld.h:37`

---

## 設定元（RuntimeBuilder.cpp:238-239）

```cpp
worldOwner->topology.runtimeUuid = (current != nullptr) ? current->runtimeUuid : 0;
worldOwner->topology.fadingRuntimeUuid = (active && next != nullptr) ? next->runtimeUuid : 0;
```

---

## 移行コード（AudioEngine.Timer.cpp:469-472）

```cpp
// 変更前（deprecated）
const uint64_t currentUuid = (runtimeWorld != nullptr) ? runtimeWorld->engine.currentRuntimeUuid : 0;
const uint64_t fadingUuid = (runtimeWorld != nullptr) ? runtimeWorld->engine.fadingRuntimeUuid : 0;
const uint64_t transitionCurrentUuid = (runtimeWorld != nullptr) ? runtimeWorld->engine.transitionCurrentRuntimeUuid : 0;
const uint64_t transitionNextUuid = (runtimeWorld != nullptr) ? runtimeWorld->engine.transitionNextRuntimeUuid : 0;

// 変更後（TopologySemantic）— 簡潔版
const auto currentUuid =
    runtimeWorld ? runtimeWorld->topology.runtimeUuid : 0ULL;
const auto fadingUuid =
    runtimeWorld ? runtimeWorld->topology.fadingRuntimeUuid : 0ULL;
// RuntimeBuilder の現在の実装では transition UUID は
// current/fading UUID と同じ値になる
const auto transitionCurrentUuid = currentUuid;
const auto transitionNextUuid    = fadingUuid;
```

---

## 注意点

1. **transition UUID は実装依存**: 現在の `RuntimeBuilder` 実装では、`transitionCurrentRuntimeUuid` は `topology.runtimeUuid`、`transitionNextRuntimeUuid` は `topology.fadingRuntimeUuid` と同じ値が設定されるため、Timer.cpp では重複保持する必要はありません。ただし将来の RuntimeBuilder 実装変更に備え、必要に応じて個別に読み出すことも可能です
2. `TopologySemantic` は `RuntimeSemanticSchema` の一部。`RuntimeSemanticSchema` 全体を取得するには `RuntimeState`（`RuntimeWorld`）の `topology` フィールドに直接アクセス
3. `engine` フィールドは `RuntimeState` 内にまだ存在するが、将来的に削除予定
4. `AudioEngine.h:138` の `#pragma warning(disable : 4996)` は `RuntimeState` 定義内のみ。Timer.cpp では個別に警告が出る
5. この変更は単なる warning 修正ではなく、**Authority Single Source の徹底**（`EngineRuntime` → `TopologySemantic` への参照元一本化）として位置付ける
