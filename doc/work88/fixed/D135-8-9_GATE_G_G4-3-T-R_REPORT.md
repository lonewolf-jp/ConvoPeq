# D135-8/9 Gate G-4.3-T-R — Regression / Compile-Fix Audit（Work Report）

**Status: PASS**（read-only 監査。Production source changes: 0 / Test source changes: 0）
**詳細:** `evidence/D135-8-9_GATE_G_G4-3-T-R_AUDIT.md`
**ビルド evidence:** `evidence/g43tr_ctest.log`（本監査で新規取得）／`evidence/g43t_build.log`（G-4.3-T 当初）

## 目的
G-4.3-T で初めて露見した production compile-only 修正（`ISRRuntimePublicationCoordinator.cpp:822-825` の using 2 行＋コメント）が意味論不変であることを再確認し、regression coverage（T1-T11）と G-4.3 invariant を read-only で再監査した。

## A. compile-only fix → 意味論不変を確認
- `ObligationDomains` の宣言元は `RuntimeIntentCoordinator` の public ネスト enum（h:228-235、`std::uint8_t` bitmask None/IR/Conv/EQ/Config/OS）。最新 `ConvoPeq.md`（23:47:13 再生成版 line 57703）＋AiDex＋serena で三者一致確認。
- 修正 2 行は anonymous namespace（cpp:821-866、convo::isr 内・内部リンケージ）に閉じる純粋な**型名解決**:
  - `using convo::isr::RuntimeIntentCoordinator;` = 同一エンティティの再導入（曖昧化・遮蔽なし）
  - `using ObligationDomains = RuntimeIntentCoordinator::ObligationDomains;` = 型エイリアス（定義・ストレージ・ADL 不変）
- `computeDomainCoverage` の呼び出し箇所（cpp:892 の 1 箇所）・bitmask 導出・構築順序は G-4.2 報告記述と完全一致（不変）。戻り値型は `domainCoverage` フィールド型と同一で型変換なし。
- `domainCoverage` は src 全体で宣言＋コメントのみに出現。`operator==`（5 semantic 値）にも findByKey/tryInsert/capacity/resolve にも不使用 → **necessary-condition metadata only 維持**。
- 新警告 0（C4324 は D116 以前からの既存）。ODR/shadowing/型変換の問題なし。

## B. regression coverage → ソース・実行双方で成立
- 新規 5 本（T1 COALESCE+oblId 再利用 / T2 diff-target→NEW / T7 terminal 後同一 identity→NEW / T8 coalesce で generation 不変 / T10 snapshot-level metadata drift→COALESCE）を全件ソース確認。T10 は `RuntimeBuildSnapshot.sampleRate`（トップレベル、target 導出非関与）を操作しており、D18.8「同一 semantic target→coalesce / target 変化→distinct」の境界を正しく検証。
- T3=C4 / T4=C3 / T5=C2 は既存テストがカバー（新規重複なし）。T9 は静的検証（`recoveryGeneration = intent.intentId` 0 件、独立カウンタ）。T11 は production hook を追加せず**未実装（対象外として正しい）**。
- 実行: G-4.3-T 当初 exe 直接実行（DBG_EXIT=0/REL_EXIT=0）に加え、本監査で**フルビルド→CTest を両構成で実測**: Debug `100% tests passed out of 40`（DBG_CTEST_EXIT=0）/ Release `100% tests passed out of 40`（REL_CTEST_EXIT=0、#21=ISRSemanticValidationRejects=新規テスト含む exe、#40 AudioEngineHarness 15.85s）。

## C. G-4.3 invariant → 12/12 維持
CoalesceIdentity={handle,target} / same→findByKey→CAS Live→Live→ΔL=0 / diff→tryInsert→ΔL=+1・capacity≥32 reject / terminal→resolve() 同一 state atomic→ΔL=−1（−1 発火は cpp:1045/1087 の 2 サイトのみ）/ COALESCE 後 slot mutation 0（delivery bookkeeping は D105-R5-10 既存・単一書込者）/ generation 再発行なし / BuildGeneration 混同なし / episode production ref=0 / ResolvedSuperseded transition=0 / canSupersede path=0 / durable blind-overwrite 温存 / reservation 不変。

## D. Diff boundary
G-4.3-T デルタ＝テスト +99 行＋cpp compile-only fix 4 行のみ（h 変更は全て G-4.1/G-4.2 由来・既監査。他ファイル無変更）。
**教訓明記:** G-4.2/G-4.3 read-only 監査はビルド検証を含まず、型解決欠陥は G-4.3-T ビルドで初露見。今後は read-only 監査にも早期ビルド検証を組み込む（本監査は grep ではなく実測 build→CTest を証拠とした）。

## 非ブロッキング観察（次回編集窓推奨、今回変更なし）
1. h:960「rebuildRequestGeneration（coalesce 判定用）」コメントが G-4.2 の意味変更で陳腐化。
2. h:274-276 の「cpp:836-841」参照が 6-field 化で陳腐化。
3. ccc/semble/graphify はこの環境で起動失敗/未構築（代替スタックで監査完遂）。

## 判定: **PASS** → G-4.4 は開始しない（指示どおり停止）。次の設計 Gate は別途指示を待つ。
