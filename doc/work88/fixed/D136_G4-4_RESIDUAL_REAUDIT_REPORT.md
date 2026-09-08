# D136 — G-4.4 Residual Liveness / Representation-Race Re-audit（Work Report）

**Status: CONDITIONAL**（read-only。Production source changes: 0 / Test source changes: 0）
**詳細:** `evidence/D136_G4-4_RESIDUAL_REAUDIT.md`
**基準:** working tree（G-4.4 と無変更）＋ `ConvoPeq.md Generated: 2026-08-30 23:47:13`（現ツリー一致の最新生成版。ユーザー参照の 23:43:26 版ではない — 明示的前提として記録）。build 状態: `evidence/g44_ctest.log` Debug 40/40 + Release 40/40（同一ツリー、実測済み）。

## 判定一覧
```text
D136-A Builder wake gap:            FAIL
D136-B transient double representation: FAIL
D136-C Building overwrite:          CONDITIONAL
D136-D memory-order:                FAIL（contract defect として記録・未修正）
D136-E capacity/reservation conservation: PASS
D136-F test coverage:               GAP
Overall:                            CONDITIONAL
```

## 各決着の要点
- **A（FAIL 確定）**: rebuildCV.notify は全 4 サイト（requestRebuild task / stop / deferred-publish watchdog（`hasDeferredRequest` かつ 100 tick 限定 — F6-5 で毎 tick 通知は廃止、h:2714 コメントは陳腐化）/ submitRecoveryIntent の recoveryPending）。**redrive 付着（durable cpp:1158-1168 / transport cpp:1174-1176）を起こすものは存在せず**、`rebuildCV.wait` は無期限・retry timer なし・`recoveryRetryReady` は predicate 非参加（h:2726-2727）。よって「付着から有限時間内に Builder が観測・処理」のコード保証なし。
  **新規発見（決定論的）**: transport recovery の build 失敗（RebuildDispatch:996-1000）・warmup 失敗（1009-1023）は `continue` のみで `markTransientFailure` を呼ばない（durable 側 1075-1084/1102-1107 と非対称）。pop 済み obligation は Live+delivery=Transport に固定され redrive 候補（None のみ）から恒久除外 → **recovery 静永久停止**（台帳は無傷、shutdown 回収）。次 Gate 最優先候補。
- **B（FAIL 確定）**: interleaving 形式化 t0-t5（take→Building→settle(true)→markTransientFailure(delivery=None)→redrive→transport 併存）。spin 上限 break 後は窓が無期限。**単なる transient duplicate ではなく delivery uniqueness（§5/C16）および INV-X1-5（1 admission ≤ 1 reservation: durable 占有+transport 予約）の違反**として確定。INV-X1-6 はカウンタレベルでは成立（durable 非計上のまま）。台帳・resolve 冪等・世界整合は無傷（有界重複 build/publish、2 回目も同一 generation 採番で冗長 commit 到達）。
- **C（CONDITIONAL 決着）**: Building 中同一 oblId 上書きは**到達可能**（cpp:991 は sub-state 非照合、single-writer ordering では防げない）。ただし stranding に必要な RECOVERY-6 discard は**本番不到達と確定**: 本番 buildSource は `currentBuildSnapshot_`（AudioEngine.h:4456-4460）で、書込元は Commit.cpp:800 の sealed=true 値のみ、handle は ProcessIntent:137 で non-null ゲート。success/settle(true) 経路は無害・有界。→ 安全性が「payload 妥当性ゲート不変」に外部依存する点を次 Gate で契約化すべき。
- **D（FAIL 記録）**: 全 store/load 列挙の上、形式的結論: CoordinatorLoop 由来 wake（submit h:4500-4504 / watchdog Threading:292-295）の mutex エッジは program order 経由で redrive 書込も可視化するが、**MessageThread の requestRebuild で起床した Builder の take()（plain state 読 cpp:1195）と CoordinatorLoop の redrive 書込（cpp:1159）/上書き（cpp:998）の間には synchronizes-with が存在しない**。`take()` は `recoveryAdmissionPending_` を acquire しない（ペア cpp:1168↔1214 は存在するが消費経路が通らない）。C++ モデル上 UB → **memory-order contract defect として記録**（x86-TSO 実害は判定に用いない）。修正なし。
- **E（PASS）**: pendingIntentCount_ の recovery 関連全サイト（+1: 976/1174、−1: 983/1179/1261）と liveCount_（+1 h:408 / −1 h:430）を網羅追跡。`markTransientFailure→None→redrive→transport` 経路は +1/−1 対 1 回で**予約増殖なし**。32/256/1/257 の混同なし（二重住処も 257 枠内）。
- **F（GAP）**: T1（transient-failure×durable 保持×redrive 併存）未カバー（R18 系は take/settle 併用なし、C11-C16 は None×durable 窓を作らない）。T2 未カバー（単一スレッド API テストのみ）。T3 未カバーかつ**現ハーネスで構成不能**（std::thread 0 件）。テスト追加なし（指示）。

## Overall 理由
NO-GO 不成立: distinct obligation の delivery 喪失・台帳破壊・resurrection はいずれも成立せず。PASS 不成立: wake 保証欠如（A）・契約違反窓（B）・形式順序欠陥（D）がコード上確定の欠陥であり、うち transport build 失敗 stranding は決定論的到達。

## 次 Gate（G-4.4 実装）への入力候補（優先順）
1. transport recovery build/warmup 失敗 → markTransientFailure 配線（A 新規・決定論的）
2. redrive 付着時 Builder wake（A-1）
3. delivery uniqueness 修復方針の確定（B: markTransientFailure の durable 保持中 None 化禁止 or redrive same-oblId-busy transport 禁止）
4. durable 消費 memory-order 契約の確定（D: 3 案）
5. Building 中上書きの契約化（C）
6. T1/T2/T3 回帰テスト設計（F）

## STOP
D136 監査完了・verdict 報告済み。禁止事項（durable 設計変更・table rewrite・retry scheduler・wake 配線変更・markTransientFailure/overwrite/memory-order 修正・新規ソース作成）すべて遵守。**変更 0、実装に進まず指示待ち。**
