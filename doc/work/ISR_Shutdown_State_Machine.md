# ConvoPeq ISR Shutdown State Machine

## 目的

本書は **R15: Shutdown HB FSM** の authoritative specification である。

shutdown プロセスを形式的な State Machine として定義し、
phase 間の HB barrier を mandatory とすることで
shutdown 時の UAF（Use-After-Free）を構造的に排除する。

---

## 背景・動機

現状の shutdown は手続き的（procedural）であり、
各 phase の HB 関係が formal に保証されていない。

```text
informal shutdown
→ HB 保証なし
→ retire/reclaim 順序不定
→ UAF
```

本書は shutdown を **formal FSM + barrier** として定義することで
shutdown UAF を形式的に閉じる。

---

## ShutdownPhase 列挙

```cpp
enum class ShutdownPhase
{
    Running,              // 通常動作中
  AudioStopped,         // Audio callback 停止完了
  ObserverDrained,      // Observer のすべての read が完了
  RetireClosed,         // RT からの新規 RetireIntent ingress 閉塞完了
  EpochSettled,         // EpochDomain の grace period 完了
  ReclaimComplete,      // 全 reclaimable node の reclaim 完了
  ShutdownComplete      // 完全停止
};
```

**注意**: 現コードの `ShutdownPhase` 列挙とは名称が異なる場合がある。
本仕様に合わせて enum 名を変更すること。

---

## ShutdownBarrier

phase 遷移時に必ず ShutdownBarrier を実行すること。

```cpp
struct ShutdownBarrier
{
    ShutdownPhase from;
    ShutdownPhase to;
    // barrier を実行することで HB を確立する
    void execute() noexcept;
};
```

ShutdownBarrier の実行なしに phase を進めることは禁止。

---

## 必須 Barrier Rules

### S1: AudioStopped → ObserverDrained

```text
AudioStopped 完了
  HB barrier
    ObserverDrained 開始
```

AudioStopped が完了しないまま ObserverDrained を開始してはならない。

### S2: ObserverDrained → RetireClosed

```text
ObserverDrained 完了
  HB barrier
    RetireClosed 開始
```

ObserverDrained が完了しないまま RetireClosed へ遷移してはならない。
（観測中の old world への参照が残る状態で retire を止めると dangling read が発生する）

### S3: EpochSettled → ReclaimComplete

```text
EpochSettled 完了
  HB barrier
    ReclaimComplete 開始
```

全 epoch grace period が完了しないまま reclaim を実行してはならない。

---

## verifyShutdownFSM()

shutdown 完了後（または CI シミュレーション）に実行する検証関数：

```cpp
enum class ShutdownFSMVerificationResult
{
    Valid,
    PhaseSkipped,         // phase が順序通りに実行されなかった
    BarrierMissing,       // 必須 barrier が実行されなかった
    InvalidTransition,    // 不正な phase 遷移が発生
    ReclaimBeforeGrace    // grace period 未完了の reclaim（S3 violation）
};

ShutdownFSMVerificationResult verifyShutdownFSM(const ShutdownTrace& trace);
```

---

## shutdown_trace.json

shutdown の実行履歴を記録するトレースログ。
デバッグビルド + CI 検証時に生成すること。

```json
{
  "phases": [
    { "phase": "Running",            "timestamp_ns": 0 },
    { "phase": "AudioStopped",       "timestamp_ns": 1000000 },
    { "phase": "ObserverDrained",    "timestamp_ns": 2000000, "barrier": "S1" },
    { "phase": "RetireClosed",       "timestamp_ns": 3000000, "barrier": "S2" },
    { "phase": "EpochSettled",       "timestamp_ns": 4000000 },
    { "phase": "ReclaimComplete",    "timestamp_ns": 5000000, "barrier": "S3" },
    { "phase": "ShutdownComplete",   "timestamp_ns": 6000000 }
  ],
  "verificationResult": "Valid"
}
```

---

## AudioEngine.h 対応事項

現在の `ShutdownPhase` 列挙は本仕様（Layer 6 canonical naming）に一致させること：

| 本仕様（canonical） | 現コード（要確認） |
| ------------------- | -------------------|
| AudioStopped        | （確認・改名要）   |
| ObserverDrained     | （確認・改名要）   |
| RetireClosed        | （確認・改名要）   |
| EpochSettled        | （確認・改名要）   |
| ReclaimComplete     | （確認・改名要）   |
| ShutdownComplete    | （確認・改名要）   |

---

## 関連正本

- `ISR_Deferred_Retire_Intent_Bridge.md` — RetireClosed 時の RetireIntent 閉塞
- `ISR_Minimal_HB_Failure_Model.md` — HB 欠落時の failure catalog（HB-03 shutdown reclaim race）
- `ISR_Verification_Pipeline.md` V6 — Shutdown FSM Verification ステージ
- `ISR_Formal_Guarantee_Package.md` P5 — 統合保証パッケージ参照

## Backlog 参照

- `ISR_Completeness_Risk_Backlog.md` R15 — Closed 最小検証項目

## ステータス

- Spec-Fixed: 2026-05-20
- Closed: 未完（実装・CI検証未実施）
