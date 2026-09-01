# D142 — Test Payload Contract Repair (Implementation + Verification Record)

**Date:** 2026-08-31 (+09:00)
**Type:** implementation（テストのみ）+ 回帰/AV stress 検証。**Production source changes: 0。**
**基準:** `ConvoPeq.md Generated: 2026-08-31 09:28:16`（P3 完了版・D141 と同一基準、ツリー一致確認済み）→ 修正後 `13:27:23` に再生成（`D142` マーカー 4 件 = 実ツリー同期確認）。
**Verdict: 全 AC 成立（下表）。D141 の断続 AV 消失を統計的に確認。**

## 1. 修正前の意図確認（指示 §4）
4 テストを全読した結果、いずれも `commit()` は **Bootstrapping→Ready への状態遷移の駆動**にのみ使用され、**payload（world）の内容は一切検証していない**ことを確認:
- `testCoordinatorDrainAndShutdownContract`: commit 後に drain 条件と shutdown→Bootstrapping 回帰を検証
- `testShutdownCompleteFailsWhenNotDrained`: commit 後に retire backlog 1 で Faulted を検証
- `testPressureStateNormalizationContract`: commit 後に Pressure→(swapPending 保持)→3 window→Ready を検証
- `testShutdownCompleteFailsWhenSwapPending`: commit 後に swapPending で Faulted を検証
→ payload は「実 RuntimeState であればよい」。bake 内容を検証する既存テスト（`testCoordinatorCommitAndMonotonicityContract` 等）は `createForTest()` を既に使用しており、本修正は同スタイルに揃えるもの。

## 2. 修正内容（4 箇所・テストファイルのみ）
```diff
-    int world = 1;
+    auto world = RuntimeState::createForTest();   // ★ D142: commit payload 契約 = 実 RuntimeState（旧 &int は commit の bake で stack BOF → D141）
     coordinator.commit(convo::isr::PublishAuthority::Granted,
                        convo::isr::RuntimeBoundary::NonRTWorld,
-                       &world,
+                       world.get(),
```
`reinterpret_cast`・バッファ拡張・commit 側の回避は一切使用せず、**payload 契約そのもの（実 `RuntimeState`）に適合**させた。旧パターン（`int world = 1;` / `&world,`）の残存 0 件を grep で確認。

## 3. 検証結果
| AC | 条件 | 結果 |
|---|---|---|
| D142-1 | 4 箇所全数修正 | ✅（置換 4+4、旧パターン 0） |
| D142-2 | payload が実 RuntimeState を指す | ✅（createForTest().get()） |
| D142-3 | production source 変更 0 | ✅（diff stat: 他 4 ファイルは P3 時と同値、変更は tests のみ） |
| D142-4 | P4 race 修正 0 | ✅（delivery/durable/memory-order 無touch） |
| D142-5 | Debug CTest 40/40 | ✅（DBG_CTEST_EXIT=0） |
| D142-6 | Release CTest 40/40 | ✅（REL_CTEST_EXIT=0） |
| D142-7 | Release AV stress で 0xC0000005 消失 | ✅ **200 反復 0 失敗**（D141: 13/200=6.5%。同一率なら 0/200 の確率 ≈1.5e-6 — 統計的に消失確定） |
| D142-8 | 4 テストの本来の検証意味を維持 | ✅（状態遷移駆動のみ、アサーション不変・全合格） |
| D142-9 | 一時診断コード/TEMP 残留なし | ✅（grep 0、d142_stress.bat 除去済） |
| D142-10 | ConvoPeq.md 再生成・同期確認 | ✅（13:27:23、D142 マーカー 4） |

## 4. 検証中のインシデント記録（透明性）
- 1 回目の Release build が C1033（vc140.pdb ロック）→ mspdbsrv 強制終了＋vc140.pdb 削除で復旧。
- 2 回目が LNK1143（kill 時に破損した juce_audio_formats.cpp.obj）→ kill 時刻（12:14 以降）の obj を削除して再ビルド → 3 回目で完全成功。
- いずれもビルド成果物の破損であり、ソース修正とは無関係。Debug/Release 公式結果は 3 回目の `evidence/d142_ctest.log` に記録。

## 5. 残存（次 Gate 入力）
- **D143（P4 Repair Contract Selection / Pre-Audit）**: D140 の delivery/durable-slot 形式 DATA RACE/UB の修復契約。D141 により AV との因果が切れたため、契約判断は (i) adjudication の CoordinatorLoop 集約を第一候補として read-only で確定可能。
- 参考: `tools/symtool.cs`（D141 の診断用・非追跡）が残置（bash 削除がフックに拒否されたため）。プロジェクトソースではない。

## STOP
D142 完了。**P4 実装（D143）には進まない。**
