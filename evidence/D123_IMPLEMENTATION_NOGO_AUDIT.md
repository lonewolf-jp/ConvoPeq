# D123 — Minimal Lifecycle Repair Implementation → **NO-GO（実装全取消・安定状態復帰）**

**Date:** 2026-08-29
**Frozen HEAD:** `a65ace1` + D117 観測トレース（macro-gated 2 ファイルのみ — 現状に復帰済み）
**入力:** D122 PASS 12/12（`D122_REPAIR_CONTRACT_AUDIT.md`）

---

## 実施内容（段階実装・指示どおり）

### D123-A — Active Handle Publication（DSPTransition.h のみ）

- normal path / emergency path の双方で `lifetime.activate(newDSP)` 後に
  `DSPHandleRuntime::activate(registerDSPHandleForRuntime(newDSP))` を配線
- oldDSP == newDSP（同一 DSP 再 publish）を retire/crossfade 対象外とするガードを両分岐に追加

**Gate A: Compile PASS**（DIAG ON `build-diag` / production OFF `build\` 両方）。

### D123-A smoke → **暴走再現（D119 と同一）**

3-burst probe: **DC live 81 / NUC 160 / Priv 12.4GB / reclaim 試行 1,176万回**（publish 2回に対し）。
→ INV-XFADE-1（completion 実装なしでは start が永久保留）により、段階実装の中間状態は**構造的に不安定**であることを実測で確認。計画どおり D123-B に進行。

### D123-B/C/D — completion 独立駆動 + Observe 撤去 + shutdown 診断（CrossfadeRuntime.h / AudioBlock.cpp / Timer.cpp / ProcessIntent.cpp / MainApplication.cpp / MainWindow.cpp）

- RT: LinearRamp `remaining` 1→0 エッジ検出 → `notifyRampComplete()`（id 不使用の純シグナル）1 回 push
- Timer: snapshot-fade ゲートから**独立した** completion consume ブロック（active records 解決 → endCrossfade → fading slot CAS → `retirePublishedDSP`）
- `retirePublishedDSP` を D122-B 契約どおり書換（identity = CAS 取得 DSPCore* 一次、receipt は resolve 検証、mismatch で quarantine/fatal 禁止）
- Observe 経路から `retireByHandle` を撤去（ObserveIntentHandler / drainObserveDeferred — accounting invariant は維持）
- shutdown 診断: SHUTDOWN_BEGIN / mainWindow.reset() 完了 / LOGGER_DETACH / SHUTDOWN_END マーカー追加、Auto-exit 時の早期 logger 切断を撤去（teardown 全程をファイル記録可能に）

**Gate 1 Compile: PASS（両 config・error 0）**

### Gate 4（6-publish smoke）→ **NO-GO 検出**

| 指標 | 値 |
| --- | --- |
| [PUBLISH] | 2 回 |
| **DSPCore live** | **103**（publish 数に比例増加 — ユーザー判定基準「即 NO-GO」に抵触） |
| NUC live / Priv | 204 / **15.8GB** |
| **CONV_REBUILD** | **107 回**（約 2 回/秒の再構築ループ） |
| reclaim 試行 | 1,257 万回 |
| identity chain | **正常に発火**（`[D117_RETIRE] dsp=X retired=1 → enqueue → [D117_DESTROY] dsp=X` 同一ポインタ対合、shutdown も "mainWindow.reset() completed" + "LOGGER_DETACH / SHUTDOWN_END" まで完走） |

**identity chain（③retire→destroy 同一 identity 完結）と shutdown 観測性（D123-D）は成立したが、①② の成立を阻む再構築ループが新規に顕在化 → D123 規約により即時中止・全取消。**

## 新規発見 — crossfade 分岐の `setIRChangeFlag()` 自己言及ループ（D120-4 の具体化）

ループの因果（実コード確定）:

```text
DSPTransition crossfade 分岐（D123-A 配線により到達可能化）
  → engine_.setIRChangeFlag()                    DSPTransition.h:123（既存行）
  → m_pendingIRChange = true
  → Snapshot.cpp:95 promoteToStructural          「IR change promoted to structural path」
  → 構造 crossfade 経路へ昇格 → 構造 rebuild
  → 新 DSPCore → publish → crossfade 分岐 → setIRChangeFlag → ...
  ⇒ 約 2 回/秒の構造 rebuild ループ（CONV_REBUILD 107 回 / 50 秒）
```

- この分岐は旧コードでは到達不能（oldDSP 常に null）だったため、**ループは設計以来一度も実行されてこなかった潜在経路**
- `setIRChangeFlag` の本来の責務は「UI 側 IR 変更を rebuild 判定へ伝搬」だが、**rebuild 自身が起動した crossfade からも flag を設定するため自己言及**になる
- D122 の INV-XFADE-4（fading slot 占有が feedback loop を生成しない）は**ループ機構を不正確に想定**していた — 実際の駆動は fading slot ではなく **IR change flag の再昇格**。D122-H 契約の修正が必要

## 現状（復帰確認）

- production（`build\`）+ 診断（`build-diag`）両方を reverted ソースから再ビルド、**CTest 40/40 PASS**（`D123_ctest_reverted.log`）
- working tree は D117 観測トレース（macro-gated 2 ファイル）のみ

## D123 verdict: **NO-GO（全取消）**

ユーザー判定基準「DSPCore liveCount が publish 数に比例して増えた時点で即 NO-GO」に抵触。

## D122-R（契約修正）に持ち込む新規項目

1. **INV-IRFLAG-1（新設）**: crossfade 分岐（rebuild 起因の遷移）では `setIRChangeFlag()` を呼んではならない。flag の設定責務は UI 側 IR 変更のみに限定する。あるいは flag に「起因」タグを付与し promoteToStructural の再昇格を防止
2. promoteToStructural 昇格経路（Snapshot.cpp:95）と rebuild 起因 crossfade の関係の契約化 — 「昇格 → rebuild → crossfade → 昇格」の自己言及を構造的に排除
3. `setIRChangeFlag` の全 caller の列挙と責務分類（D123 実測: crossfade 分岐がループ源）
4. 暴走時の観測性: MEM_SNAP が ループ検出（DC live 急増 + CONV_REBUILD レート）を即座に可視化できた — HealthMonitor への rebuild-rate 監視追加を推奨

**D117/D119 から本監査までの累積知見**: 修復には (a) activate 公開、(b) completion 独立駆動、(c) **IR flag ループの遮断**、の 3 点が **単一パッチ**で必要。単点修正はいずれも別の欠陥を顕在化させる（D119: completion 不発で暴走、D123-A: 同一、D123-B: IR flag ループで暴走）。

## 生成物

- `evidence/D123A_prod_build.log` / `D123A_diag_build.log` / `D123_diag_build.log` / `D123_prod_build.log` / `D123_revert_*.log`
- `evidence/D123A_probe.log`（A 単体暴走）/ `D123_g4_6pub.log`（B 実装後: identity chain 発火 + ループ暴走の証跡）
- `evidence/D123_ctest_reverted.log`
- 本ファイル
