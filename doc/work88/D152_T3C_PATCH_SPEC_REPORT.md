# D152 — T3c Patch Specification（Work Report）

**Status: 仕様確定（G152-01..14 全項目固定）。変更 0・実装 0・build/CTest 非着手。**
**詳細:** `evidence/D152_T3C_PATCH_SPECIFICATION.md` / **基準:** ConvoPeq.md `15:32:58`。公式一次資料: MicrosoftDocs cpp-docs（intrinsic 実シグネチャ raw 取得）・MSVC STL atomic ヘッダー（lock pool 実測）。

## 固定した核心

1. **W exact 定義**: `alignas(16){id:u64, state:u8, pending:u8, adjudicated:u8, delivery:u8, pad[4]={}}` + static_assert 4 点（sizeof/alignof/trivially_copyable/standard_layout）。byte 格納・bitfield 不採用・pad 常時 0（CAS は 16B 全文比較）。
2. **wrapper 実シグネチャ確定**: `_InterlockedCompareExchange128(Destination, ExchangeHigh, ExchangeLow, ComparandResult*)` — **exchange は 2 スカラー、comparand は in/out で失敗時現在値が書き戻る**。`casLifecycle(expected&, desired&, dst&)` + `loadLifecycleAdvisory()` の 2 関数に固定。x64 基本形=lock cmpxchg16b=フルバリア → **追加 fence 不要**（acq_rel 超過）。
3. **advisory-load 規律**: 分断 load 自体は禁止しない。commit は全文 CAS のみ。失敗時は更新 comparand を再評価（再 load 不要の retry 規則）。偽成功生成不能。
4. **7 遷移の expected/new/retry/liveCount/telemetry を擬似コード固定**（T1 postSignal / T2 adjudicate / T3 resolve / T4 tryInsert / T5 coalesce / T6 submit attach / T7 redrive attach）。**「CAS 成功後に別 atomic 更新で意味を完成」構造は全経路に不在**。
5. **新規発見（R1）**: coalesce の全文 identity CAS は postSignal の pending 変化と衝突しうる → **1 発失敗=即 tryInsert は誤 NEW 化（ΔL 二重計上）**。contention 再試行ループ必須、terminal 判定のみ tryInsert 経路。D153 回帰検査に明記。
6. **tryInsert 順序完全固定**: (1) payload plain 書込（identity/handle/epoch/intentId/buildSource/recoveryGeneration — **署名拡張で呼び出し側 cpp:954-959 の post-insert 書込を関数内へ移動**）→ (2) W CAS `{old}→{N,Live,0,0,None}` → (3) liveCount++（勝者のみ）。
7. **markTransientFailure 3 責務分離**: production 6 サイト（RebuildDispatch:1006/1033/1091/1115 + Orchestrator:311/401）→ `postRecoveryFailureSignal` / CL `adjudicateRecoveryFailureSignals`（runCoordinatorPhase の processIntent 後・redrive 前）/ **TEST-ONLY ラッパ**（post+adjudicate、h:147 前例）。production 呼び出し 0 を D153 V1 に。
8. **delivery authority**: W 内フィールド・変更=CAS 経由のみ・書込 owner=CL。cpp:1071 の plain write は T2 の desired 内 None 化に吸収され消滅（V2 で機械検証）。
9. **thread affinity 決定**: **jassert 不採用**（ラッパがテストスレッドで実行するため両立不可。かつ安全性は CAS 主体でスレッド非依存 — affinity は決定論の規約）。代替強制=V7 grep。
10. **telemetry exact name**: exhausted は**枯渇 CAS 勝者のみ**（過計上不変条件）+ saturated/droppedStale/droppedTerminal/droppedInvalid の 4 新設（増加条件表で固定）。
11. **テスト**: 既存約 30 site はラッパで無変更（直接アクセス 0 実測済）。新規 NT-1..5 を exact 断言付きで固定（NT-5 は公開 API のみで 2 スレッド構成 — G-4.3-T T11 断念理由を解消）。
12. **D153 機械検証 V1..V10**（grep 条件・static_assert・call site 限定・CTest 40/40×2+NT）。

## 次アクション
**D153 — Read-only Implementation Audit** の指示を待つ（実装はまだ行わない）。file 別 patch 一覧（§12）とリスク R1..R7 を実装・監査の照合基準として固定。**Phase 2 実装凍結は D153 通過まで継続**。

**STOP — 実装 0。D153 の指示を待つ。P5/P6 非着手。**
