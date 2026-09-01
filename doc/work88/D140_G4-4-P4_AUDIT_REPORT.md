# D140 — G-4.4-P4 Memory-Order / Cross-Thread Delivery + Intermittent AV Audit（Work Report）

**Status: read-only 監査完了。Production/Test source changes: 0。**
**詳細:** `evidence/D140_G4-4-P4_MEMORY_ORDER_AV_AUDIT.md`
**基準:** ConvoPeq.md `Generated: 2026-08-31 09:28:16`（P3 完了版）+ git status = P3 時と同一 5 ファイル（余計な変更なし確認）

## 最終判定
```text
P4 audit: DATA RACE（B）
  - delivery: RebuildThread(W5=markTransientFailure cpp:1078) × CoordinatorLoop(R1-R4/W7/W8) の
    conflicting plain accesses、非枯渇経路に synchronizes-with エッジ無し = 形式 UB
  - 枯渇経路のみ resolve CAS(acq_rel)×state acquire で SAFE
  - durable-slot plain フィールド（state/oblId）も同型（D136-D 再確認）
  - 断続 AV との因果: UNPROVEN（H1/H2 不採用）
```

## 主要成果（10 点）
1. **delivery 全数**: W1-W8 / R1-R4 を列挙。h:324「CoordinatorLoop 単一書込」は **D105-R18 以降偽**と確定 — W5 の全 production caller（RebuildDispatch:1006/1033/1091/1115 + Orchestrator:311/401）が **RebuildThread** 実行であることを呼び出し連鎖（enqueuePublicationIntentForRuntimeCommit=Commit.cpp:782→submitPublishRequest、processDeferredAdmission=RebuildThread）で追証。
2. **HB グラフ**: 非枯渇 A→B はエッジ無し（fetch_add は別位置・state 非触达）。B→A は write-write 競合（W5×W7/W8）。到達 interleaving 実在。
3. **P3 repair の因果分離**: 新規 race クラスなし（同一フィールド・同一スレッド対）。ただし既存 durable-slot race の下で**新しい狭い失敗モード 1 件**（state→oblId の 2 段階 stale 読取 → 空 slot に delivery=Durable 再同期 → 実体無し strand、同一 key 再 submit か shutdown まで）。P3 の逐次正しさ（T-P3 合格）とは別問題として記録。
4. **redriveWakePending_**: writer/reader/clear 全て CoordinatorLoop（RebuildThread アクセス 0 件）→ **SAFE**。
5. **断続 AV**: 最終 P3 バイナリで **3/100 再現**（D139 の 20/20 は低頻度ゆえ未検出 — 訂正）。フォールト関数形状を 3 ビルドで一致確認（4 double の NaN ガード書込 +0x198..0x1B0、2 バイト [rdx+0x64]=1/[rdx+0x65]=2|6、r9 非 NULL だが不正宛先）。**この形状は delivery 経路と不一致**（delivery は単一 byte @0x148 付近）。Release は /DEBUG なしで PDB 非生成 → シンボル解決 487、**関数のソース名特定は未達**。
6. **H1-H4**: H1/H2 は証拠なしで不採用（排除はしない）。H3/H4 未証明。最有力は H4（既存 UB、P2 era の 1 回実行 CTest で未検出と整合）。
7. **修復契約候補**（実装禁止・次 Gate）: (i) adjudication の CoordinatorLoop 化（HB 問題が構造的消滅・推奨）> (iii) 状態の atomic ドメイン統合 > (ii) delivery のみ atomic 化（**§3 の 2 段階 stale 読取と durable-slot 残りを救えず単独不十分** — 「delivery だけ atomic」で全体が正しくなる保証なしとの明示）。
8. **P4 実装要否**: 形式 UB 是正として要。ただし契約の形は AV 診断（/DEBUG+LocalDumps または ASAN マイクロゲート）の結果に依存。

## 禁止事項遵守
delivery atomic 化・mutex 追加・memory_order 変更・redriveWakePending_ 変更・AV 修正・DSP lifetime 修正・queue/durable 変更・P5/P6 — **全て未実施（変更 0）**。一時診断スクリプト（WER 照会/dbghelp/disasm/AV ループ）は除去済み。

## STOP
D140 報告のみが成果物。**P4 実装・P5・P6 には進まない。** 推奨次アクション: AV 専用診断マイクロゲート（D141）→ その結果で P4 契約 (i)/(iii) を確定。
