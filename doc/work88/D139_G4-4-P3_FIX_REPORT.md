# D139 — G-4.4-P3 Double Representation 修復（Work Report）

**Status: 実装完了・AC-P3-1..10 成立・Debug/Release CTest 40/40×2 PASS**
**詳細:** `evidence/D139_G4-4-P3_DOUBLE_REP_FIX.md` / **ビルド:** `evidence/g44p3_ctest.log`

## 実装（唯一の変更点）
`redriveDeferredRecovery()` durable-busy 分岐に same-holder 検査を追加（cpp +19 行、うちコード 4 行）:
```cpp
if (pendingRecoveryAdmission_.recoveryObligationId == obligationId) {
    s.delivery = ObligationDeliveryState::Durable;   // metadata re-sync (NOT a new attach)
    redriveWakePending_ = true;
    return;
}
```
- 実体生成なし: queue push なし・durable 書込なし・tryInsert/resolve 非経由 → **ΔL=0・oblId 追加なし**（AC-P3-8）。
- different-holder は従来どおり transport fallback（AC-P3-4/C13 維持）。
- 二重住処の生成不能証明: redrive fallback 経路を遮断。submit 経路は opportunistic redrive（cpp:902）が同一スレッドで先に窓を repair → coalesce 早期 return（cpp:943-944）で push 到達不能。durable 上書き（cpp:998-1008）は実体を増やさない。**∎ container+durability+ordering の 3 点で成立**（AC-P3-1/2/3）。

## 実装前再確認の決着
- **same-holder 条件**: `recoveryObligationId` は CoordinatorLoop 付着/上書きのみが書き、take/settle(true) は保持、NoAdmission は oblId=0 にリセット → else 分岐では **oblId 一致単体で十分**（state 併記は冗長と確認）。
- **wake 必要性**: Builder predicate は durable slot 占有を反映しない。窓は spin 上限 break 後に Builder 就寝のまま残り得る → **latch 必要**（P2 同一プロトコル・実事象のみ発火で毎 tick notify 逆戻りなし）。ループ中の冗長 wake は no-op 1 回で無害。

## テスト（T-P3-1..6、public API のみ・+150 行）
T-P3-1（same-holder: transport 非生成+再同期+wake）/ T-P3-2（different-holder fallback）/ T-P3-3（二重不成立直接検証）/ T-P3-4（repeat redrive 増なし）/ T-P3-5（failure→redrive 循環 XOR + K=4 終端 + wake 源消滅）/ T-P3-6（free-slot 付着+P2 回帰）。既存 T-P2-1/5・C11-C16・全 40 テスト回帰なし。

## ★ 既存の断続 AV（P3 と無関係 — 混在させず報告）
負荷試験で Release テスト exe に低頻度（~1/9-1/15）の 0xC0000005 を観測。bisect で **P2+P3 テストをスキップしても再現**（この構成では same-holder repair 経路はデッドコード）→ **P3 実装・テストは無罪を確定**。WER は double 4 フィールドの NaN ガード付きコピー関数形状での不正書込（潜在的自由領域参照/ヒープ破損の遅延顕在化が疑われる）を指す。最終クリーンバイナリは Release 20/20・Debug 20/20・公式 CTest 40/40×2 合格。**P4（memory-order/潜在 UB）の診断対象として正式引き継ぎ**（データは evidence §5 添付）。P2 era の CTest が各 1 回のみで検出できていなかった低頻度事象と整合。

## 結果
```text
Debug:   full build OK → CTest 100% tests passed out of 40（DBG_CTEST_EXIT=0）
Release: full build OK → CTest 100% tests passed out of 40（REL_CTEST_EXIT=0）
ConvoPeq.md 再生成 Generated: 2026-08-31 09:28:16（G-4.4-P3 マーカー 9 件・ツリー同期確認）
```
一時診断スクリプト（probe/disasm/WER/flake）は除去済み。テストファイルに TEMP 残留 0 を確認。

## STOP
P3 完了。**P4/P5/P6 には進まない（指示待ち）。**
