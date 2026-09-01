# D155 — Recovery Transport / Coalesce 現状再監査（Work Report）

```text
Production/Test/CMake changes: 0 / Build: NOT RUN / CTest: NOT RUN / ST-1 再実行: NOT RUN（完全 read-only）
baseline: ConvoPeq.md Generated 2026-08-31 23:39:12（T3c Close Audit と同一）
詳細: evidence/D155_RECOVERY_TRANSPORT_COALESCE_REAUDIT.md
```

## 総合判定

> ## **A. NO CHANGE / DEFER**

**R1（MPSC 化）も coalesce 設計監査も、現時点で着手する根拠がない。** 実装・ソース変更は 0。

## 判定根拠（D155-A）

- `recoveryIntentQueue_` = `LockFreeRingBuffer<RecoveryIntent, 256>`（SPSC）。push 2 サイト（submitRecoveryRequest cpp:991 / redrive cpp:1257）は**全て CoordinatorLoop に収束**、pop 1 サイトは **RebuildThread 専用**（+ join 後 shutdown 単一スレッド）→ **SPSC invariant は構造的に成立**。
- **R1 のトリガ条件（「将来 Timer 等から直接呼ぶ」）は未発生** — Timer.cpp の recovery API 呼び出し = 0 件（RT affinity audit と整合）。
- `pendingIntentCount_` = reservation-before-push accounting（fetchAdd → push / 失敗 rollback / pop 消費）で `setPendingIntentCount` は TEST-ONLY（production 上書きは P2-1 §1.1.5 で廃止済み）。意味論は INV-ISR-02 整合。
- `pendingRecoveryAdmission_`（durable slot）は **state = atomic CAS プロトコル（D146）で独立保護**されており、R1 と同時変更する必要はない（writer 集合 {CL, RebuildThread} は R1 でも不変）。
- MPSC 化が必要になった場合の変更量は記録済み（queue 交換 + reservation invariant 再証明 + shutdown proof 再設計 + SPSC 契約/テスト再監査の 5 項目 — lifecycle domain には触れない）。

## 判定根拠（D155-B）

- `CoalesceIdentity = {quarantinedHandle, SemanticRecoveryTarget(5 値一致)}`（MUST-3）— **同一 handle・異なる target は区別される**（C4 実証）。「same handle → 最新で上書き」は構造的に不可能。
- **Superseded は未実装を確認**（`ResolvedSuperseded` 宣言のみ・cpp 遷移 0 件）— 意図的な Phase-II 凍結。
- **REPAIR_PLAN2-dash2 の「coalesce 将来対応」は stale** — 旧 P3 設計（`lastRecoveryHandle_`）は残存 0 件で、Phase-I coalesce は G-4.1→G-4.3 チェーンで**既に実装・監査・ストレス検証済み**（G-4.3 CAS-linearized coalesce + ST-1 T5 storm）。
- terminal → resubmit = NEW（T7 semantic・ST-1 F-2 実証）、durable overwrite は CAS NoAdmission 観測時のみ（無条件上書きなし）。

## D155-C: T3c との境界（確定）

```text
T3c CLOSED
   ├── lifecycle ownership domain（RecoveryLifecycleWord 16B・full-word CAS）＝ 変更禁止（本監査 0 触れ）
   └── recovery transport / admission（queue / pendingIntentCount_ / durable slot / redrive / coalesce protocol）＝ D155 対象
```

R1 / coalesce の将来変更が lifecycle CAS 層の再設計を含んではならないことを本監査で明示。

## 判定ルール適合

| 選択肢 | 判定 |
|---|---|
| **A. NO CHANGE / DEFER** | **● 該当** |
| B. R1 IMPLEMENTATION CANDIDATE | 複数 producer 不存在・SPSC 成立ゆえ不該当 |
| C. COALESCE DESIGN AUDIT REQUIRED | coalesce は G-4.x で完了・Phase-II Supersession は D18 凍結中ゆえ不該当 |
| D. BLOCKED | 安全性問題不存在ゆえ不該当 |

## 将来の着手条件（記録）

- Timer 等の第 2 producer 出現 → R1 実装前設計（変更対象 5 項目を evidence §2.5 に記録済み）
- Supersession が製品要件化 → Phase-II 設計監査（D18 規範）開始
