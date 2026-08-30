# D119 — DESIGN-B 最小実装の試み → **NO-GO（実装全取消・安定状態に復帰）**

**Date:** 2026-08-29
**Frozen HEAD:** `a65ace1` + D117 観測トレース（macro-gated 2 ファイルのみ）
**結果:** production 修正は実装したが Gate 3 で破壊的挙動が判明 → **全取消（git checkout）**。現在の working tree は凍結状態に復帰済み（CTest 40/40 再確認済み）。

---

## 実施した実装（全て取消済み）

1. `retirePublishedDSP` 書換（D119-3 準拠: identity = CAS 取得 DSPCore* 一次、receipt は resolve による DSPCore* 一致検証、mismatch で quarantine/fatal 禁止・metadata 破棄）
2. Timer.cpp 3 箇所の fade-completion CAS サイトに `retirePublishedDSP(current)` 接続（submitObserve を authority から外す）
3. **発見①への対処として追加**: `RuntimePublishExecutor.h` Execution tail の oldDSP 復元（後に撤去）
4. **発見②への対処として追加**: `DSPTransition::onPublishCompleted` で publish 完成時に `dspHandleRuntime_.activate(newHandle)` を呼ぶ（後に撤去）

## Gate 結果

| Gate | 結果 |
| --- | --- |
| Gate 1 Compile | **PASS**（DIAG ON / production OFF 両方） |
| Gate 2 CTest | **PASS**（test21 単独 ×2 → 1回目のみ失敗・再現せず、full 40/40） |
| Gate 3 6-publish smoke | **FAIL** — 下記の通り |

## Gate 3 の二段階の失敗と、その過程で確定した新知見

### 発見① — D118 CAUSE-C1 のチェーンは**下流から既に遮断されていた**

`retirePublishedDSP` を Timer に接続しても `[D117_RETIRE]` は 0 発火。追求した結果:

- **`DSPHandleRuntime::activate()` は production コードから一度も呼ばれていない**（呼び出し元ゼロ、unit test のみ）→ `activeRuntimeDSPHandle_` が**常に null**
- → Orchestrator（RuntimePublicationOrchestrator.cpp:82）の `oldHandle = getActiveRuntimeDSPHandle()` が**常に null**
- → `decision.oldHandle = null` → Execution tail の `onPublishCompleted(new, nullptr, ...)` → **retire/crossfade 両分岐が到達不能**（claim も storeReceipt も beginCrossfade も実行されない）
- さらに fadeCompleted ブロック自体も `SnapshotCoordinator::tryCompleteFade()`（`m_fade`）にゲートされ、`m_fade.start()` は **GlobalSnapshot 経路（AudioEngine.Snapshot.cpp:145）でしか呼ばれない**ため、world publish シナリオでは**ブロック自体が一度も実行されない**

→ **D118 CAUSE-C1 は実在するが「休眠経路」の欠陥であり、観測された漏出の直接原因はより上流の「active handle 未公開 → oldHandle 常時 null」**。

### 発見② — 逐次修復は制御不能なフィードバックループを誘発

`activate()` を DSPTransition に配線すると（identity は正しく流れ `[D119_TAIL]/[D119_TRANS]` で oldDSP が実値で渡り、`[D117_RETIRE]`=4 / `[D117_DESTROY]`=5 発火）、同じ 6-burst 実行が:

- **DC live 104 / NUC live 206 / Priv 15.9GB / reclaim 試行 13,055,555 回**（50 秒で）
- `needsCrossfade=1` が評価される一方、**DSP crossfade の完了処理が構造的に到達不能**（m_fade ゲート）のため crossfade が完了せず、再構築要求が反復されるループ

→ crossfade 完了経路（`crossfadeRuntime_` と `SnapshotCoordinator::m_fade` の関係、`consumeCompletedFade` の駆動点）を含めた**設計レベルの再監査なしに逐次修正は不可能**と判断し、実装を全取消。

### 発見③ — **シャットダウン時の間欠クラッシュ（MSVC でも再現）**

revert 後の frozen ソースでも、8 秒 CLI 実行の終了時に **exit 139 (Access Violation) が約 2/3 の頻度で発生**（3 試行: 139/139/0。シナリオ自体は完走し "Auto-exit flush: shutting down" が最終行）。D116 restart cycles で 6/6 exit 0 だったことと合わず、**タイミング依存の既存 shutdown race** の可能性が高い。icx 0xc0000005 ×3（同一フォールトオフセット）と同族の可能性 → **D118-BLOCKER-1 に加え D119-BLOCKER-2 として凍結**（production 修正の前提条件として原因特定が必要）。

## 現状（復帰確認）

- production（`build\`, macro OFF）+ 診断（`build-diag`, macro ON）両方を reverted ソースから再ビルド
- diag 8s run: **exit 0**
- CTest: **40/40 PASS**（`evidence/D119_ctest_reverted.log`）
- production 8s run: シナリオ完走、ただし終了コード 139/139/0 の間欠（発見③）

## D119-7 GO/NO-GO 判定

**NO-GO — 実装全取消。**

理由: D118 の DESIGN-B は「fadeCompleted ブロックが実行される」「claim/receipt が成立する」ことを前提としていたが、その前提自体が production 経路では成立していない（発見①）。また部分的な配線は crossfade 完了の構造的欠落（発見②）と組み合わさって暴走を引き起こした。**本欠陥群は「1 箇所の修正」では閉じない。**

## 次段階への提言（D118-R: crossfade/fade-completion 経路の再設計監査）

修正対象を単点ではなく **publish → activate → crossfade → completion → retire の全体経路**として再監査する必要がある。確認すべき契約問答:

1. `DSPHandleRuntime::activate()` をどの経路でどのタイミングで呼ぶか（現在ゼロ）
2. DSP crossfade の完了を何が駆動するか（現行: m_fade = GlobalSnapshot 専用ゲート。world publish では誰も complete しない）
3. `consumeCompletedFade` / `endCrossfade` / fading slot CAS の駆動点と threading
4. Observe intent の epoch filter（stale 破棄）の意味論
5. シャットダウン間欠クラッシュ（発見③）の原因 — icx クラッシュと同族か

Phase-II 要素には一切進んでいない。commit 凍結は継続。

## 生成物

- `evidence/D119_diag_build.log` / `D119_diag2_build.log` / `D119_revert_diag_build.log` / `D119_revert_prod_build.log`
- `evidence/D119_gate3_6pub.log` / `D119_gate3b_6pub.log`（未発火証跡）/ `D119_gate3c.log`（暴走証跡: DC=104, NUC=206, Priv=15.9GB）
- `evidence/D119_probe4.log` / `D119_probe5.log` / `D119_revert_probe.log` / `D119_revert_prod_probe.log` / `D119_prod_retest*.log`
- `evidence/D119_ctest_diag.log` / `D119_ctest_reverted.log`
- `evidence/D119_diag_build2.bat`（独立 build-diag 手順）
- 本ファイル
