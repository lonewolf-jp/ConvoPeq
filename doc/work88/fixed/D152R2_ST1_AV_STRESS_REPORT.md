# D152-R2 / ST-1 — AV stress 200（Work Report）

```text
Production source changes: 0 / Repo test source changes: 0 / CMake changes: 0
新規（リポジトリ外 instrument）: evidence/st1/D152R2_ST1_AVStress200.cpp + st1_build.bat + 実行ログ
使用した production オブジェクト: D154 検証時と同一（obj 再利用 — production byte-identical）
詳細: evidence/D152R2_ST1_AV_STRESS_200.md
```

**Status: ST-1 = PASS。Debug 200/200 + Release 200/200 サイクル、カウンタ整合違反 0。次工程 = RT affinity audit。**

## 結果サマリ

| チェック項目 | 結果 |
|---|---|
| liveCount_ < 0 / > 32 | 0 件（全フェーズ境界チェック・最終 0） |
| double terminal transition | 0 件（2 回目 resolve 常に false） |
| droppedInvalid | 0（厳密） |
| droppedTerminal / droppedStale / saturated | 期待意味論どおり（単調増加のみ・過剰計上違反 0） |
| recoveryRetryExhaustedCount | over-count 0（== Failed 終端数と厳密一致） |
| runtime `is_lock_free()` | **false** を両構成で記録（D152-R2 §4 expected = false の実行時証跡） |

## 実施内容

- **Part A**: D154 検証済み `ISRSemanticValidationTests.exe`（T1-T10 / NT-1〜NT-5 / RLOE C1-C16 / R21）を Debug/Release 各 200 反復 → **200/200 + 200/200 PASS・失敗 0**。
- **Part B**: リポジトリ外ストレスドライバ（D154 production obj を再リンク — production は byte-identical）で 1 サイクル = admission + T5 coalesce storm ×16 + T1/T2 実 2 スレッド競合 8ms（postSignal ∥ adjudicate）+ T6 delivery CAS churn ×64 + T3 terminal teardown、を 200 サイクル × 2 構成。数千万回規模の lifecycle CAS 競合で全不変条件維持。

## 発見事項（driver 設計訂正 — production は契約どおり）

- F-1: terminal 化時の delivery は**保持**される（delivery は state と独立・None-ing は非 terminal drain CAS のみ）— 初版 driver の期待誤り。
- F-2: 競合下で adjudicate が先に K 到達 → ResolvedFailed 化した場合、その後の同キー resubmit は**合法的に新規 obligation 再承認**（T7 semantic）し terminal slot を再利用し得る。driver は子 obligation を捕捉して teardown で双方 terminal 化・liveCount 基線復帰を検証。

いずれも T3c production の障害ではなく、契約意味論の stress 下実証として記録。

## 次工程（指示順序）

```text
ST-1 PASS（現在地）
   ↓
RT affinity audit（lifecycle W が ISR/AudioThread に触れていないことの再確認）
   ↓ PASS → T3c close candidate / FAIL → STOP・新規監査
   ↓
D154-F2（DSPHandle + T3c 側残存誤コメントの comment-only 別トラック）
```
