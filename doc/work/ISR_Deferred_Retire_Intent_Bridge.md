# ConvoPeq ISR Deferred Retire Intent Bridge

## 目的

本書は **R14: Deferred Retire Intent Queue** の authoritative specification である。

RT（Audio Thread）が crossfade 完了等の retire トリガを検出した際に、
実際の retire/reclaim authority を NonRT RetireCoordinator へ安全に委譲するための
**RetireIntent Bus** を形式化する。

---

## 背景・動機

現実コードでは crossfade 完了は Audio Thread 側で検出される。
しかし ISR の原則は：

```
Audio Thread での reclaim/delete/free = 禁止
```

これを「運用規律」で守るのではなく、**構造として強制**するための bridge spec が本書の目的。

---

## Core 構造体定義

```cpp
enum class RetireReason
{
    CrossfadeComplete,     // crossfade が完了し old DSP が不要になった
    PublicationReplaced,   // 新 publish により旧 world が置き換わった
    ShutdownRequested,     // shutdown sequence による retire
    ErrorRecovery          // エラー回復による強制 retire
};

struct RetireIntent
{
    DSPHandle   handle;           // retire 対象の DSPHandle
    uint64_t    completionEpoch;  // retire を安全に実行できる最小 epoch
    RetireReason reason;          // retire の理由
};
```

---

## RT 側の義務（emit only）

Audio Thread は **intent の emit のみ** を行う。
reclaim / delete / free の実行は禁止。

```cpp
// Audio Thread 内から呼び出す唯一の許可操作:
void emitRetireIntent(const RetireIntent& intent) noexcept;
```

- `emitRetireIntent` は wait-free / lock-free SPSC queue への enqueue のみ
- queue が満杯の場合は **silent drop** ではなく overflow counter を increment

---

## NonRT 側の義務（authority execution）

`RetireCoordinator` が NonRT スレッドで実行する。

```cpp
class RetireCoordinator
{
public:
    // NonRT timer/event loop から呼び出す
    void processRetireIntents() noexcept;

private:
    void executeRetire(const RetireIntent& intent);
    void acknowledgeRetireIntent(const RetireIntent& intent);
};
```

- `executeRetire`: EpochDomain で epoch 条件を確認し、条件成立後に reclaim 実行
- `acknowledgeRetireIntent`: RT 側が確認できる完了フラグを設定（optional だが推奨）

---

## Intent Acknowledgement

NonRT が retire を実行完了した後、RT 側が参照確認できるよう
acknowledgement を提供することを推奨する（必須）：

```cpp
// NonRT が完了後に設定
void markRetireAcknowledged(DSPHandle handle, uint64_t completionEpoch);

// RT が参照確認（optional polling）
bool isRetireAcknowledged(DSPHandle handle) const noexcept;
```

---

## 禁止事項（CI static analyzer 対象）

以下は Audio Thread 内（`getNextAudioBlock` および呼び出し先 helper）での禁止：

| 禁止操作         | 代替                              |
| ---------------- | --------------------------------- |
| `reclaim(handle)` | `emitRetireIntent()` のみ         |
| `delete ptr`     | 禁止（NonRT 側 RetireCoordinator）|
| `free(ptr)`      | 禁止（NonRT 側 RetireCoordinator）|
| `mkl_free(ptr)`  | 禁止（NonRT 側 RetireCoordinator）|

---

## CI 強制ゲート

```text
Audio Thread コンテキスト内での
reclaim / delete / free / mkl_free 呼び出しが検出された場合:
→ CI fail（merge reject）
```

検出方法：

- `.github/scripts/check-src-atomic-dotcall.ps1` に RT retire rule を追加
- `ISR_Verification_Pipeline.md` V7 Retire Latency Audit と連携

---

## 関連正本

- `ISR_Retire_Authority_Graph.md` — retire authority 全体グラフ
- `ISR_Shutdown_State_Machine.md` — shutdown 時の RetireIngress 閉塞シーケンス
- `ISR_Verification_Pipeline.md` V7 — Retire Latency Audit ステージ
- `ISR_Formal_Guarantee_Package.md` P4 — 統合保証パッケージ参照

## Backlog 参照

- `ISR_Completeness_Risk_Backlog.md` R14 — Closed 最小検証項目

## ステータス

- Spec-Fixed: 2026-05-20
- Closed: 未完（実装・CI検証未実施）
