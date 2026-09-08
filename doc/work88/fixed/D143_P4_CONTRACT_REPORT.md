# D143 — P4 Repair Contract Selection / Pre-Audit（Work Report）

**Status: 契約選定完了（I-HS）。Production/Test source changes: 0。**
**詳細:** `evidence/D143_P4_CONTRACT_PREAUDIT.md` / **基準:** ConvoPeq.md `Generated: 2026-08-31 13:27:23`（D142 後・同期確認）

## D143-1 で確定した現行 ownership（要点）
- delivery: writer W1-W8（W5=markTransientFailure cpp:1078 のみ RebuildThread、他 7 つは CoordinatorLoop）、reader 4 箇所全て CoordinatorLoop。
- durable slot: field 個別追跡で `state`/`recoveryObligationId`/payload 群が **CoordinatorLoop・RebuildThread・shutdown thread の 3 系統**接触と確定。
- **新規発見**: `rearmRecoveryRetry`（cpp:1100-1101、Orchestrator:413 経由）が RebuildThread から state+oblId を 2 段 plain 読し settle(true) で state に書く — D140 が数え落としていた第 3 の cross-thread 接触点。
- `hasPendingRecoveryAdmission` の production caller は 0（test 専用）と実測。

## D143-2 HB graph（要点）
- **CL→Builder**: 成立。P2 以降すべての付着/repair が rebuildMutex エッジ（submit 経路 / redrive wake）に続くため program order 経由で HB。
- **Builder→CL**: 非枯渇経路で不在（W5 の後の release 無し・CL の対応 acquire 無し）= 形式 UB。枯渇時のみ resolve CAS×state acquire で成立。
- **requestRebuild の wake**: MessageThread の task 書込に対する HB は実在するが、**CoordinatorLoop の durable-slot 書込とは別スレッド・別データで順序付けない**（「mutex+notify」だけでは SAFE 不成立と明示）。

## D143-3/7 候補比較と最終決定
```text
Candidate I    GO（単独では durable slot の P4-2/3/4 を満たさない — 随伴条件付き）
Candidate III  NO-GO（全面再設計はスコープ超過。個別 atomic 配置は state→oblId の
                      2 段 stale 読を消さないことをコードで示した）
Candidate II   NO-GO（反証: durable-slot race 残存・repair の stale 組合せ残存・
                      ownership protocol 不変・delivery は単一 byte で実害ゼロ）

Selected contract: Candidate I + state handshake（I-HS）
  (a) RecoveryFailure{oblId} を既存 intentQueue_（MPSC・実 producer 複数スレッド確認済）で
      CoordinatorLoop へ転送し、adjudication（delivery/counter/枯渇 resolve）を単一権限化。
      非 drop 保証（quarantine fallback の前例）+ exactly-once posting を契約に含む。
  (b) pendingRecoveryAdmission_.state を atomic release/acquire 化し、
      「payload/oblId 書込は state release に sequenced-before」規約で 3 値一貫観測を担保。
      take/settle の Consumer 所有は維持（lease 設計は不変）。
```
選定理由: I は既存 MPSC 輸送・既存 wake プロトコル・K=4 意味論をそのまま再利用でき、state ハンドシェイク addition で INV-P4-1..4 を同時に満たす最小契約。liveness も改善（adjudication が processIntent で redrive より前 = 同一 tick 完結）。

## D143-4/5（要旨）
- P3 repair は I-HS 下で stale strand モードが構造的に消滅（acquire した state と oblId が必ず対になる）。repair/fallback/repeat/latch の論理は不変。
- lost-event proof: 輸送喪失は pre-P1 stranding の再発になるため非 drop 保証が必須要件。duplicate は R20-3 exactly-once 違反 → 失敗サイト 1:1 post。stale は Live チェック no-op、ordering は CL 直列化、shutdown は ShutdownDiscarded と整合。

## D143-6 invariant 行列
evidence の表のとおり、I-HS のみが P4-1..10 すべてに ✓（現行文書の INV-X1-x / R17-4 / R20-3 と対応併記済み）。

## Remaining unresolved（D144 の入力）
1. RecoveryFailure 輸送の具体形と fallback/overflow 数値
2. exactly-once posting の 6 失敗サイトの証明義務
3. state atomic 化時の discard/shutdown release 順序
4. rearmRecoveryRetry の扱い（イベント化 vs state HS 下で Builder 実行のまま）
5. 回帰テスト設計

**STOP — 実装 0。D144（Implementation Plan / Proof Obligations）の指示を待つ。P5/P6 非着手。**
