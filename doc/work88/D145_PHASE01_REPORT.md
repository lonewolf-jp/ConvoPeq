# D145 Phase 0-1 — I-HS2 実装前再導出 & 修約 3 証明（Work Report）

**Status: GO（Phase 2 進出可）。Production/Test source changes: 0。**
**詳細:** `evidence/D145-PHASE0-1_IHS2_PREDERIVATION.md` / **基準:** ConvoPeq.md `13:27:23`（D142 後版・ツリー無変更確認）

## Phase 0 の主要結果
1. **oblId lifecycle 確定**: `++nextId_` 単一採番（h:398・単一書込者 CL）、単調性で非再利用（h:417/C9 実証済）、durable slot の oblId は attach/overwrite 時のみ書込・reset は settle(false)/discard のみ。
2. **容量再導出**: L_logical_max=32（h:363）、Q=256（h:927）、durable=1、L_residency=257（I4:943/984）、K=4（h:369）。**E_max/O_max は Phase-I の実コードに存在しない**（episode 語彙は Phase-II deferred）と明記。
3. **★ D144 の 64+64=128 非 drop 証明は反証された**: `fillRecoveryQueue`（test:1076）が示す C16 再 push 設計では**同一 obligation の intent が 256 本 queue に載り得る**（cpp:961-968 NOTE 前例）。P1 後の Builder は pop 毎失敗=1 event なので、1 tick 窓の未処理上限は 128 でなく transport 256 + publish 失敗 ≤260 + durable spin。**指示 4 の分岐に従い容量再設計: primary 512 + fallback 128 + 両満杯時 HealthEvent 昇格**（drop しても obligation は Live 維持=safety 不変、非 silent 保証）。

## Phase 1（修約 3 の既存契約適合証明）
- **A ✓**: CL の payload 書込は `state.load(acquire)==NoAdmission` 観測時のみ → Building 中 overwrite は経路ごと消滅。
- **B ✓**: DurablePending 中同一 oblId は no-op + true（既存表現が有効）。
- **C ✓**: attach = payload 書込 → CAS(NoAdmission→DurablePending, release)。**CAS failure は live 中に発生不能**（NoAdmission からの唯一の遷移権限は CL 自身、discard は post-join）。万一の失敗時も残骸 payload は NoAdmission 下で誰にも読まれず次 attach が全面上書 → retry/rollback 不要（防御分岐のみ残す）。
- **D ✓**: 「same O ⇒ same target」を**前提でなく実コードで証明**: oblId の mint 専有性（単調採番 + identity immutable h:346）より、同一 O に到達する submit は同一 handle + 同一 SemanticRecoveryTarget（buildInput は target 構成要素で一致、build は現在設定を再読 cpp:809-811）。失われる更新分の消費を実測確認: **epoch は recovery 消費経路で読取 0 件**（grep 実測）、intentId は診断専用、recoveryGeneration は slot 不変値。→ no-op 化は意味論不変。D105-R23 との緊張は「episode 非依存の mint 専有性のみ」を用いて解消。

## 旧コメント/契約記述の洗い出し（実装時更新対象・今回は未修正）
h:937-946（SPSC 競合なし）/ h:968（plain・atomic 不要）/ h:318-324（delivery 単一書込者）/ h:960-961（coalesce 判定用・coalesce で更新）/ cpp:880-881（SPSC-safe）/ cpp:1059-1063（caller 記述）/ h:485-489（rearm limitation）— 一覧を evidence に固定。更新は安全性証明後（Phase 2-6 実装時）に実施。

## 判定
```text
Phase 0: 0-1/2 ✓、0-3 ✓、0-4 ✗→再設計（512+128+昇格）、0-5 ✓
Phase 1: A ✓ / B ✓ / C ✓ / D ✓
STOP 条件: 該当なし
→ GO: Phase 2（state atomic 化）へ進出可。採用容量=primary 512/fallback 128/昇格。
```
**本ターン指示範囲（Phase 0 → Phase 1）まで実施。実装 0。STOP — Phase 2 以降の指示を待つ。**
