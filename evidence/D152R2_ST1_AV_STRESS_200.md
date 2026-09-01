# D152-R2 / ST-1 — AV stress 200 実施記録（lock-pool backend 下の contention / lifecycle pressure）

**Date:** 2026-08-31 (+09:00)
**Type:** 運用ストレス検証。**Production source: 0 変更 / Test source（リポジトリ内）: 0 変更 / CMake: 0。**
**対象バイナリ:** D154 検証時と同一の production オブジェクト（`build/CMakeFiles/ISRSemanticValidationTests.dir/{Debug,Release}/…` — obj mtime 20:40-20:52 > 最終ソース mtime 20:16、byte-identical 再利用）。

---

## 0. 判定（先出し）: **PASS — Debug 200/200 + Release 200/200 サイクル、カウンタ整合違反 0**

| ユーザー要求チェック項目 | 結果 |
|---|---|
| `liveCount_ < 0` | **0**（全フェーズで [0, 32] 境界チェック・最終 0） |
| `liveCount_ > capacity(32)` | **0**（同上） |
| double terminal transition | **0**（2 回目 resolve は常に false・`exhausted == Failed 終端数` と厳密一致） |
| droppedTerminal | **期待どおりの意味論**（terminal 化後の storm 投入のみ計上・単調増加・過剰計上ゲート違反 0） |
| droppedStale | **期待どおりの意味論**（id 消滅後の観測のみ計上・単調増加） |
| droppedInvalid | **0**（厳密一致 — obligationId==0 を投入しないため） |
| saturated | **期待どおりの意味論**（pending>=K の inert 投入計上・adjudicate が K で terminal 化） |
| recoveryRetryExhaustedCount | **over-count 0**（== Failed 終端数と厳密一致 — driver では 200/200 サイクルで 1 サイクル 1 件） |

加えて: **`std::atomic<RecoveryLifecycleWord>{}.is_lock_free() == false` を両構成で実行時に記録**（D152-R2 §4「expected MSVC result = false」の runtime 証跡）。

---

## 1. 実施方法

### Part A — 既存 D154 検証済みセマンティクス suite の 200 反復

- 対象: `build/{Debug,Release}/ISRSemanticValidationTests.exe`（T1〜T10 / NT-1〜NT-5（実 2 スレッド race 含む）/ RLOE C1〜C16 / R21 T1〜T3 等 — カウンタ整合は内部 assert で per-iteration 検証済み）
- 実行: 各 200 反復、exit code + stderr 集計
- **結果: Debug 200/200 PASS・Release 200/200 PASS・`TEST FAILED` 0 件**（evidence/st1/D152R2_ST1_partA_semantic200.log — 成功時無出力の suite のため 0 byte）

### Part B — リポジトリ外ストレスドライバ（カウンタ記録本体）

- driver: `evidence/st1/D152R2_ST1_AVStress200.cpp`（新設・リポジトリ外 instrument・CMake 未登録）
- 構成: **D154 ビルド済み production obj をそのままリンク**（production コードは D154 40/40 ゲート時と byte-identical。driver TU のみ新規 compile — `evidence/st1/st1_build.bat`、vcvarsall x64 + MSVC 14.51.36231、compile_commands.json と同一 flags）
- 1 サイクルの負荷パターン（単一 obligation に対し）:
  1. **admission**（新規 id — サイクル跨ぎで lifecycle 再利用）+ `{Live, Transport}` peek 検証
  2. **T5 coalesce storm** ×16（同キー再提出 → coalesce CAS Live→Live、ΔL=0、coalesced+1 厳密検証）
  3. **T1/T2 競合 window** 8ms — producer thread（postSignal storm = RebuildThread 役）∥ consumer thread（adjudicate storm = CL 役）
  4. **T6 delivery CAS churn** ×64（競合下での coalesce re-push → casDelivery attach）
  5. **T3 terminal**（teardown で残存 Live obligation を resolve — 単一 Completion Authority）
  6. **終端不変条件**: identity 安定・pending/adjudicated=0・2 回目 resolve=false・liveCount 基線復帰・カウンタ意味論一致
- 実行: **200 サイクル × {Debug, Release}**（×2 回で再現性確認）

---

## 2. 計測値（driver・200 サイクル集計）

| 構成 | run | elapsed | coalesced | exhausted | droppedTerminal | droppedStale | droppedInvalid | saturated | liveCount 最終 |
|---|---|---|---|---|---|---|---|---|---|
| Debug | 1 | 1845 ms | 15,937 | **200** | 25,401,916 | 488,875 | **0** | 358,133 | **0** |
| Debug | 2 | 1843 ms | 15,901 | **200** | 18,354,203 | 787,171 | **0** | 358,121 | **0** |
| Release | 1 | 1826 ms | 15,950 | **200** | 102,449,288 | 626,834 | **0** | 2,540,614 | **0** |
| Release | 2 | 1816 ms | 15,964 | **200** | 107,384,926 | 453,571 | **0** | 1,482,532 | **0** |

読み取り:

- **exhausted == 200（= Failed 終端数と厳密一致）**: 8ms の storm 内で adjudicate が K=4 到達 → ResolvedFailed 化が毎サイクル 1 回。**over-count 0**（2 重 terminal 化があれば 200 を超える）。
- **droppedTerminal / droppedStale / saturated の巨大値は期待意味論どおり**: storm producer が terminal 化後も投入を続けるため、terminal 語への投入 = dT、slot 再利用による id 消滅後の観測 = dS、pending==4 への追加投入 = sat。いずれも単調増加のみで過剰計上ゲート違反 0。
- **droppedInvalid == 0 厳密**。
- **coalesced ≈ 79.7/サイクル = 16 storm + churn 分** — churn の post-terminal 再提出は契約どおり **新規 obligation 再承認（G-4.3-T T7 semantic）** となり、以後は子 obligation への coalesce。driver は子 id を transport pop から捕捉して teardown で resolve。
- **liveCount は全サイクルで基線復帰（最終 0）** — terminal −1 は単一 authority（CAS winner）のみ。

---

## 3. driver 側で訂正した「期待値」（production 障害ではなく driver 設計誤り）

初版 driver の 2 つの想定誤りを訂正した（いずれも production は契約どおり動作 — 逆に契約意味論を実証した）:

| # | 初版の誤った期待 | 実際の契約挙動（実測） |
|---|---|---|
| F-1 | 「terminal 直後に delivery==None」 | delivery は ObligationState と独立（h:356）。**resolve / adjudicate は delivery を保持**し、None-ing は非 terminal drain CAS のみ（cpp:1140-1146）。terminal 語は delivery 値を保持する |
| F-2 | 「resolve が必ず勝ち、旧 terminal 語が残る」 | 競合下で adjudicate が先に K 到達 → ResolvedFailed 化し得る。その後の同キー resubmit は**合法的に新規 obligation 再承認**（terminal → resubmit NEW）し、terminal slot を再利用して上書きし得る。driver は pop から子 id を捕捉し、teardown で旧・子の双方を terminal 化して liveCount 基線復帰を検証する |

## 4. 判定

**ST-1 = PASS。** lock-pool backend（spinlock）下で数千万回規模の lifecycle CAS 競合（T1/T2 実 2 スレッド storm × 200 サイクル × 2 構成）を負荷しても、liveCount 境界・単一 terminal 化・exhausted 精密計上・dropped* 意味論・identity 安定性の全不変条件が維持された。**T3c の安全性要件（16B 原子的 CAS semantics + 非RT affinity）は stress 下でも充足。**

次工程: RT affinity audit（lifecycle W が ISR/AudioThread に触れていないことの再確認）。
