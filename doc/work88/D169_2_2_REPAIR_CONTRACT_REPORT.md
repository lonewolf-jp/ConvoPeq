# D169-2-2 — Repair Contract Approval（Work Report）

```text
Task:    D169-2-2 — Duplicate-Prepare Collapse Repair Contract Approval
Date:    2026-09-07
Type:    read-only / contract freeze（production/test/CMake/build script/tool 変更 0）
Verdict: **APPROVED — 契約固定（候補 a「collapse を真の no-op 化」採用・候補 b 不採用）**
```

## 判定

D169-2-1（Case A）の修復契約として RC-D169-2-1〜7 を固定した。

- **RC-D169-2-1**: collapse（phase==Prepared && same SR/BS）は prepare transaction の
  開始ではなく duplicate request の吸収。ENTERED（Preparing → body → leavePrepare →
  Prepared）と COLLAPSED（Prepared 不変 → NO-OP → return）の 2 経路を明確分離。
- **RC-D169-2-2**: collapse 判定は `LifecycleIsolationRuntime` に残す（第二の
  lifecycle authority の作成禁止）。
- **RC-D169-2-3**: collapsed token は `leavePrepare()` に渡さない。
  `leavePrepare()` の `Preparing` 前提を緩和する修復は禁止。
- **RC-D169-2-4**: token identity の新設禁止（transaction ID / generation tag /
  epoch validation / validity flag / 新規 atomic・mutex / cancellation 全部）。
  collapse 判別には既存 `LifecycleToken::expectedPhase == Prepared`（collapse 経路
  のみが返す・一意判別子）を使用 — 既存 field の読み取りであり新規 state ではない。
- **RC-D169-2-5**: collapse 時は **prepare body 入口直後**（`PrepareToPlay.cpp:20`
  の `enterPrepare()` 帰還直後）で return。side effect（generation reset /
  pendingTask reset / publish / latency realloc / crossfade reset / analyzer re-init /
  placeholder creation / submitRebuildIntent / lifecycleState publish）を全て禁止。
  leavePrepare 直前 return では不十分（副作用が残る）ため入口が修復点。
- **RC-D169-2-6**: 非 collapse 経路（Uninitialized/Released/Prepared+SR/BS変更 →
  Preparing → Prepared）は完全維持。
- **RC-D169-2-7**: blocked-return の phase 残留（D169-2-1 §3.3 latent）は scope 外・
  同経路の変更禁止。

## 受入条件

RC-1〜RC-10（collapse 維持 / 真の no-op / leavePrepare 非呼出 / Preparing 前提不変 /
side effect 0 / 非collapse 不変 / 単一 authority / 新規 state 追加禁止 / blocked-return
scope 外 / 既存 protocol 不変）を契約文書に固定。D169-2-3 preflight への引き継ぎ項目
P1〜P4（判別子一意性・挿入位置の block/rollback 干渉・JUCE 契約上の no-op 安全性・
side effect list の網羅性）を明記。

## 成果物

- 正本: [evidence/D169/D169_2_2_REPAIR_CONTRACT.md](C:\VSC_Project\ConvoPeq\evidence\D169\D169_2_2_REPAIR_CONTRACT.md)
- 本報告: doc/work88/D169_2_2_REPAIR_CONTRACT_REPORT.md

## 変更範囲

production 0 / test 0 / CMake 0 / build script 0 / tool 0 / binary 0。

次: **D169-2-3 Preflight Source Audit（P1〜P4）→ D169-2-4 Minimal Implementation
（早期 return の挿入 1 箇所）→ D169-2-5 targeted collapse regression →
D169-2-6 device restart / stress → D169-2-7 full regression → D169-2 close**。
