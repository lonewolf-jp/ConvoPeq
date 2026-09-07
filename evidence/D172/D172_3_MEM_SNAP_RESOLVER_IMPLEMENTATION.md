# D172-3 — MEM_SNAP RuntimeWorld Resolver Implementation + Regression Validation（evidence）

- 日付: 2026-09-08
- Type: implementation + regression validation（D172-2 contract 準拠）
- Authority stamp: 実装基準 = `ConvoPeq.md` Generated 2026-09-07 21:38:42
- Production diff: `src/audioengine/AudioEngine.Timer.cpp`（MEM_SNAP resolver 1 箇所 + コメント）/ `src/audioengine/AudioEngine.h`（R3 契約コメントのみ）
- Test source: 0 / CMake: 0
- Evidence logs: `evidence/D172/`（baseline / ctest ×3 config / diag run2〜4）

---

## D172-3.0 — baseline evidence（方式 α・変更前 diagnostic build）

- 実行: `build-diag/ConvoPeq_artefacts/RelWithDebInfo/ConvoPeq.exe`（CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON・IR reload storm 5 回・20s）→ exit 0x0
- ログ: `build-diag/ConvoPeq_artefacts/RelWithDebInfo/ConvoPeq.log`（115,439 行・MEM_SNAP 6,058 サンプル）
- **結果: `t_destroy(P) < t_MEM_SNAP(TRK≠0)` の決定的実例は未取得。全 6,058 サンプルで TRK=0.0（= slot null）**
- 原因（source 確定）: CLI flow では起動シーケンス内で gen1 publish が先行し、prepareToPlay の placeholder 生成分岐（`PrepareToPlay.cpp:283: if (!hasPublishedCurrent && !hasActiveRuntimeDSP())`）が不発（log `[DIAG] prepareToPlay: hasPublishedCurrent=1` 実測）→ W2（:287 `setActiveRuntimeDSP(placeholderRaw)`）が発動せず slot は null 維持
- **意味**: D172-1 の構造的 hazard（W2 発動 → 置換 destroy → 窓）は「placeholder 分岐が発動する経路」に限定的に開く。D169-2-5 の「timing 依存 flaky AV」記録と整合。D172-1 の source-level proof（W2 発動時の dangling 構造）は変更なく、ユーザー指示どおり implementation は継続
- 副次確認: 旧コードの TRK は D162-1P で「常時 0.0（計測未配線）」と文書化済み — TRK は本修正以前から意味ある観測値として機能していなかった（D172-2 P3 判定の補強）

## D172-3.1 — production 1 箇所 implementation

`AudioEngine.Timer.cpp` MEM_SNAP block（旧 :1079）:

```cpp
// 旧: auto* activeDSP = getActiveRuntimeDSP();
auto* activeDSP = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
if (activeDSP != nullptr) { auto stats = activeDSP->collectTrackedMemoryStatistics(); ... }
```

- 使用 authority は既存の `runtimeReadHandle`（callback 冒頭 :428 取得・callback 全体スコープ）と既存 resolver（AudioEngine.h:3398）のみ。新規 helper / atomic / queue / registry なし
- 非使用化したのは MEM_SNAP 内の `getActiveRuntimeDSP()` 1 箇所のみ（Timer.cpp 内の getActiveRuntimeDSP は 0 hits に）

## D172-3.2 — contract comment

1. **MEM_SNAP block**（Timer.cpp）: 「TRK source = RuntimeWorld current DSP / legacy `activeRuntimeDSPSlot`（non-owning placeholder mirror）を観測 authority として使用しない（slot は置換 destroy 後に dangling になり得る — D172-1）/ world 未公開時は TRK=0」を明記。**「slot を lifetime-safe 化した」という誤解を生む表現は不使用**（正: MEM_SNAP が slot を読むことをやめた）
2. **`logRuntimeTransitionEvent`**（AudioEngine.h:3842-3846）: dormant diagnostic（production caller 0 件）である旨 + 「復活・再利用時は `resolveActiveRuntimeDSPFromRuntimeWorldOnly` 経由に統一 — slot dereference の新規追加禁止」契約

## D172-3.3 — diff / authority audit — **全 6 項目 PASS**

`git diff` 実測:

| # | 確認 | 結果 |
|---|---|---|
| 1 | MEM_SNAP block に `getActiveRuntimeDSP()` 残存なし | PASS（0 hits） |
| 2 | MEM_SNAP が `runtimeReadHandle` 使用 | PASS（resolver 経由 1 箇所） |
| 3 | 新規 helper なし | PASS（diff = resolver 1 箇所 + comments のみ） |
| 4 | slot writer（W1-W4）変更なし | PASS（4 箇所とも現状維持） |
| 5 | destroy / retire path 変更なし | PASS（Retire.cpp / DSPLifetimeManager.cpp / ISRRetireRouter.cpp / Threading.cpp diff 0） |
| 6 | test / CMake 差分なし | PASS（diff 0） |

補足: `git diff` に `evidence/*.json`（epoch_reclaim_audit 等 6 ファイル）の差分があるが、これは baseline run が書き出した runtime telemetry の更新であり source ではない。

## D172-3.4 — build / CTest — **3 config 全 40/40 PASS（freshness gate 適用）**

| config | build dir | build | CTest | freshness gate（exe mtime > source mtime 23:55） |
|---|---|---|---|---|
| Debug（Ninja 単一） | build-msvc | RC=0・146/146 | **40/40**（63.7s） | exe 00:15 > 23:55 ✔ |
| Release | build-ci-check（--config Release） | RC=0・556/556 | **40/40**（43.8s） | exe 00:29 > 23:55 ✔ |
| RWDI（diagnostic） | build-diag（--config RelWithDebInfo） | RC=0・5/5 残差 | **40/40 ×2**（47.8s / 46.0s） | exe 01:11 > 23:55 ✔ |

**過程での 2 つの教訓（D169-2-6/7 再確認）**:
1. **build-diag は Ninja Multi-Config** — `--config` なしの `cmake --build` は Debug config を対象にする。初回 RWDI CTest は **変更前の stale exe**（21:10 build）で実行され、AudioEngineHarness が 1 回 SEGFAULT。この結果は D172-3 の検証として無効（旧コード上の既知 flake クラス — D169-2-5 記録の MEM_SNAP 付近 harness flake と整合）。`--config RelWithDebInfo` を明示して fresh rebuild 後、**40/40 ×2 連続 PASS**。
2. RWDI rebuild 中に mspdbsrv の強制終了起因で PDB 破損（C2471 / LNK1285 ×3）→ 破損 PDB 削除 + mspdbsrv restart で解消（D169-2-6 の LNK1285 教訓どおり failure pattern を直接走査）。

## D172-3.5 — diagnostic runtime validation（fresh exe）

実行 3 回（いずれも exit 0x0・`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON`）:

| run | workload | MEM_SNAP | 結果 |
|---|---|---|---|
| run1 | IR reload storm 6 回（同一ファイル） | 248 サンプル | **TRK=0.0 → 案A 適用後は resolver 経由。同一ファイル reload（hash 不変）は構造 rebuild 不発** — 旧コードの TRK=0.0 と区別するため fresh exe 再確認要（後述 run2-4 で解決） |
| run2 | --cli-ir 指定 + storm 6 回 | 246 サンプル | TRK=0.0 のまま → **調査結果: この exe は古い（RWDI rebuild 未反映）だったため無効**。fresh rebuild 後に再実施 |
| run3（fresh exe） | IR 内容 swap（irA→irB、+4s）+ storm 6 回 | 246 サンプル | **TRK: total=1.2（OS=0.0 EQ=0.2 AL=0.2 LT=0.3）が全サンプル非ゼロ** — resolver が world current DSP（uuid=3・live）を返し、旧コードでは不可能な実 member 読み取り。swap は telemetry suppression（`convolverParamsChanged: suppressed while CLI telemetry mode is enabled`）により構造 rebuild 不発 |
| **run4（fresh exe・AC-10 用）** | `--cli-rebuild` + IR swap（+4s） | 246 サンプル | **AC-10 実証成立**（下記） |

### run4 — AC-10 の時系列実証

```text
L~570   構造 rebuild dispatch（--cli-rebuild + IR swap 内容で gen4 → gen6）
L607    [DSP_DESTROY_FOOTPRINT] dsp=000001BA3D8E0080（旧 gen4 DSP・rebuild 置換）
L608    [DSP_FOOTPRINT_RELEASED] dsp=000001BA3D8E0080 remaining=0   ← 旧 DSP 物理破壊完了
L609    [MEM_SNAP] PUBLISH gen=6 | DC: live=1 | TRK: total=1.2      ← 破壊直後の MEM_SNAP が
L628    [MEM_SNAP] PUBLISH gen=6 | TRK: total=1.2                     新 world(gen6) current DSP の
  ...                                                          統計を出力・旧 DSP address への
L4955   [MEM_SNAP] PUBLISH gen=6 | TRK: total=1.2                     dereference 痕跡なし
最終     exit 0x0
```

- gen 分布: gen=4 ×10 → gen=6 ×236（`--cli-rebuild` による構造 rebuild で gen 前進・旧 DSP destroy を含む）
- **旧 DSP address（…3D8E0080）を MEM_SNAP が dereference した痕跡なし**（TRK は常に新 current の 1.2MB 系一貫値）
- TRK 非ゼロ（1.2MB）は「resolver が生存 world current を返した」ことの行動学的証明 — 旧コード（slot null）では構造的に不可能な値

## D172-3.6 — acceptance criteria

| AC | 条件 | 判定 |
|---|---|---|
| AC-1 | MEM_SNAP の DSP source が RuntimeWorld resolver に変更済み | **PASS**（diff 実測） |
| AC-2 | `activeRuntimeDSPSlot` writer / retire / destroy path 変更 0 | **PASS**（git diff 実測） |
| AC-3 | 新規 lifetime authority 0 | **PASS**（diff = resolver 1 箇所 + comments） |
| AC-4 | RuntimeReadHandle lifetime scope 不変 | **PASS**（:428 宣言・move なし・変更なし） |
| AC-5 | TRK semantics 変更をコメントで明記 | **PASS**（MEM_SNAP block コメント） |
| AC-6 | world 未公開時 TRK=0 | **PASS**（resolver null パス → TRK=0・構造保証） |
| AC-7 | Debug CTest 40/40 | **PASS** |
| AC-8 | Release CTest 40/40 | **PASS** |
| AC-9 | diagnostic build MEM_SNAP 継続 | **PASS**（fresh exe で 246 サンプル ×2 run・AV 0・exit 0x0） |
| AC-10 | rebuild → retire → destroy 後 stale slot dereference なし | **PASS**（run4 L608 destroy → L609+ MEM_SNAP gen=6 TRK=1.2・旧 address deref 痕跡なし） |
| AC-11 | dormant `logRuntimeTransitionEvent` に再導入禁止契約 | **PASS**（AudioEngine.h コメント実装） |
| AC-12 | Case 3 window を修正対象に拡大していない | **PASS**（EpochDomain / makeRuntimeReadHandle / handle 構造 diff 0） |

> ## **D172-3 PASS — AC-1〜12 全項目 PASS**

## 判定ロジックの遵守確認（ユーザー指示）

- 「ASan が落ちなかった」等の単一指標では判定せず、diff/authority audit → lifetime proof（D172-2）→ build → CTest → diagnostic runtime evidence の順序で判定
- TRK≠0 の継続は「placeholder の garbage」ではなく「world current DSP の統計」（案A による対象変更）として解釈 — run4 の destroy 直後サンプルが新 gen6 DSP の値を出していることで区別完了

## 限界・記録事項

1. **baseline 決定的実例未取得**: 方式 α の `t_destroy < t_MEM_SNAP(TRK≠0)` は baseline workload では発生せず（placeholder 分岐不発 → slot null）。hazard の構造は D172-1 source proof で確定済みであり、実装判断に影響なし
2. **RWDI 初回 SEGFAULT**: stale exe（変更前コード）上の flake。D172-3 の検証としては無効（freshness gate 教訓の再確認として記録）
3. **ConvoPeq.md 再生成必要**: 本実装により source が更新されたため、派生 snapshot（ConvoPeq.md）は次の監査監視 window で `output_sourcecode_markdown.py` による再生成が必要（D135-8/9 教訓）
4. debug 用 bat/ps1（d172_build_*.bat / D172_3_*.ps1）は evidence/D172/ に再現手順として保存
