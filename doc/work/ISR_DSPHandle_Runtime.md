# ISR DSPHandle Runtime 仕様

## 位置づけ

本書は **ISR 10層 architecture における DSP ownership 管理**の正本仕様である。

ConvoPeq の真の危険は **DSP lifetime ambiguity** である。
DSP ownership の source-of-truth を `DSPHandleRuntime` に統合しないと、
"unsafe runtime を綺麗に観測するだけ" になる。

位置づけ: `plan5.md` → REV2 未閉塞系統 B → 本書（詳細）

**注**: `ISR_DSPHandle_Allocator_Policy.md` はアロケータポリシーの正本であり、
本書は ownership / resolution / quarantine の runtime behavior を扱う補完正本である。

### REV3.2運用優先注記

- 本書の ownership/runtime 分解は設計参照表現として扱う。
- 実装運用は `plan5.md` REV3.2 を優先し、
  `runtime exposes evidence / CI validates evidence` を固定方針とする。
- stale handle の解釈衝突時は CI=Abort / Debug=Assert / Release=Quarantine+Silence を優先する。

---

## 問題の本質

### DSP ownership ambiguity

```text
現状:
  DSP インスタンスへのポインタが複数の runtime object に分散して保持されている
  crossfade 中の "from DSP" と "to DSP" の ownership が曖昧
  retire 後の DSP への参照が AudioThread 側に残留しうる
  reclaim timing が retire 完了確認なしに行われる可能性がある

要求:
  DSP lifetime の source-of-truth を単一の DSPHandleRuntime に統合する
  全 DSP 参照は DSPHandle (slot + generation) 経由に限定する
  stale handle は build別ポリシー（CI=Abort / Debug=Assert / Release=Quarantine+Silence）で閉塞する
```

---

## 定義

### DSPHandle

```cpp
struct DSPHandle
{
    uint32_t slot;        // レジストリスロット番号
    uint32_t generation;  // 世代番号（再利用スロット区別）

    bool isNull() const noexcept
    {
        return slot == 0 && generation == 0;
    }

    static DSPHandle null() noexcept
    {
        return { 0, 0 };
    }
};
```

### DSPState

```cpp
enum class DSPState
{
    Constructing,    // create 呼び出し後、Active 前
    Active,          // 通常使用中
    CrossfadingIn,   // crossfade 中（新 DSP 側）
    CrossfadingOut,  // crossfade 中（旧 DSP 側）
    Retired,         // retire 完了、grace period 中
    Quarantined,     // 問題検出によりアクセス禁止
    Reclaimed        // メモリ解放済み
};
```

### DSPRegistrySlot

```cpp
struct DSPRegistrySlot
{
    std::atomic<uint32_t> generation;  // ABA 防止世代番号
    DSPInstance*          instance;    // 所有インスタンス（nullptr if Reclaimed）
    DSPState              state;       // 現在状態
};
```

---

## DSPHandleRuntime

```cpp
class DSPHandleRuntime
{
public:
    // NonRT: DSP インスタンスを登録し DSPHandle を返す
    DSPHandle create(DSPFactoryResult factory);

    // RT/NonRT: handle を検証し、有効な参照を返す
    // stale handle（generation mismatch）は build別ポリシーで処理
    ResolvedDSP resolve(DSPHandle handle);

    // NonRT: crossfade 開始（from と to の state を更新）
    CrossfadeId beginCrossfade(DSPHandle from, DSPHandle to);

    // NonRT: crossfade 終了（from を Retired に遷移）
    void endCrossfade(CrossfadeId id);

    // NonRT: DSP を Retired に遷移（grace period 開始）
    void retire(DSPHandle handle);

    // NonRT: grace period 完了後のメモリ解放
    // retire() から一定 epoch 経過後のみ呼べる（RG-1）
    void reclaim(DSPHandle handle);

    // NonRT: 問題検出時に DSP を Quarantined に遷移
    void quarantine(DSPHandle handle);

    // スロット状態ダンプ（デバッグ・CI用）
    void emitOwnershipTrace(const std::filesystem::path& outputPath) const;
};
```

---

## CrossfadeAuthorityRuntime

crossfade 期間中の authority 管理を担う補完 runtime。

```cpp
using CrossfadeId = uint32_t;

struct CrossfadeRecord
{
    CrossfadeId id;
    DSPHandle   fromHandle;
    DSPHandle   toHandle;
    uint64_t    startEpoch;
};

class CrossfadeAuthorityRuntime
{
public:
    // beginCrossfade が呼ばれると record を登録
    CrossfadeId register_crossfade(DSPHandle from, DSPHandle to);

    // RT: 現在の crossfade record を取得
    const CrossfadeRecord* getActive() const noexcept;

    // NonRT: crossfade 完了を記録
    void complete(CrossfadeId id);
};
```

---

## DSPQuarantineRuntime

retire 後 grace period が完了するまで accidental reclaim を防ぐ runtime。

```cpp
class DSPQuarantineRuntime
{
public:
    // retire() 呼び出し時に quarantine entry を追加
    void enter(DSPHandle handle, uint64_t retireEpoch);

    // grace period 完了確認（true: reclaim 可能）
    bool isReclaimable(DSPHandle handle, uint64_t currentEpoch) const;

    // reclaim 直前に entry を削除
    void leave(DSPHandle handle);
};
```

---

## Invariants

| 識別子 | 内容 | 違反時アクション |
| --- | --- | --- |
| DSP-1 | 全 DSP 参照は DSPHandleRuntime 経由のみ | Abort |
| DSP-2 | stale handle（generation mismatch）の resolve 禁止 | CI: Abort / Debug: Assert / Release: Quarantine+Silence |
| DSP-3 | Retired 状態 DSP への RT 参照禁止 | Abort |
| DSP-4 | reclaim は retire epoch + grace period 後のみ許可 | Abort |
| DSP-5 | crossfade 中は from/to 両 handle のみ有効 | Abort |
| DSP-6 | Quarantined DSP への resolve 禁止 | Abort |
| DSP-7 | create は NonRT からのみ呼べる | Abort |
| GI-3 | all DSP lifetime routed through DSPHandleRuntime | Abort |

---

## DSP State 遷移規則

```text
Constructing  → Active          (create 完了)
Active        → CrossfadingOut  (beginCrossfade: from)
Constructing  → CrossfadingIn   (beginCrossfade: to)
CrossfadingIn → Active          (endCrossfade: from が Retired へ)
CrossfadingOut→ Retired         (endCrossfade)
Active        → Retired         (retire 直接呼び出し)
Retired       → Reclaimed       (reclaim: grace period 完了後)
任意          → Quarantined     (quarantine: 問題検出)
```

---

## 必須 artifacts

### dsp_ownership_trace.json

```json
{
  "schema": "dsp_ownership_trace_v1",
  "slots": [
    {
      "slot": 1,
      "generation": 3,
      "state": "Reclaimed",
      "events": [
        { "type": "created",   "epoch": 100 },
        { "type": "retired",   "epoch": 200 },
        { "type": "reclaimed", "epoch": 250 }
      ]
    }
  ],
  "active_crossfades": [],
  "invariant_violations": []
}
```

---

## Closed criteria

- [ ] 全 DSP インスタンスが DSPHandleRuntime に登録されている
- [ ] 全 DSP resolve が handle 経由でのみ行われている
- [ ] DSP-1 ～ DSP-7 の違反が buildポリシーに従って閉塞されている
- [ ] reclaim が grace period 確認なしに呼ばれる経路が存在しない
- [ ] crossfade 期間中の from/to 以外の DSP への RT アクセスが封止されている
- [ ] dsp_ownership_trace.json が emit される

---

## 関連文書

- `plan5.md`: REV2 未閉塞4系統 B 系統参照
- `ISR_DSPHandle_Allocator_Policy.md`: アロケータポリシー（64byte alignment 等）
- `ISR_10Layer_Implementation_Specification.md`: 修正版実装順序（ステップ 2）
- `ISR_RT_Execution_Frame.md`: RT thread からの DSP handle 参照
- `ISR_Retire_Authority_Graph.md`: retire lane 詳細
