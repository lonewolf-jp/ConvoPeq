# Phase 0/1 — D1〜D18 独立 commit / push 計画の確定

- Document: `doc/work113/P1-5-IR-P2_STG-11-D18_HISTORY-RECON_PLAN_20261001.md`
- Date: 2026-10-01
- 目的: `e565019` を起点に D1〜D18 を **D ごとに独立した commit** として `origin/main` へ push する計画の確定
- 本文書の時点で **commit / push は未実施**（Owner の Phase 1 確認待ち）
- Authority: リポジトリルート `ConvoPeq.md`（Generated `2026-10-01 19:25:35` / FRESH）

---

## 1. 現状の Git 状態（Phase 0 実測）

```
## main...origin/main [ahead 10]
```

`e565019..HEAD` の実履歴（`--reverse`、10 commits）:

| # | SHA | 日時 | subject |
| --- | --- | --- | --- |
| 1 | `e5e2f668` | 2026-09-27 21:01 | fix(eqprocessor): I-5-R1 atomic wrapper unification for rt*Shadow (29 sites, relaxed preserved) |
| 2 | `6dc03213` | 2026-09-27 23:17 | fix(audioengine): preserve staging gains across bypass toggles |
| 3 | `de07ba54` | 2026-09-28 07:11 | fix(audioengine): revalidate input headroom on mode transitions |
| 4 | `901a6c33` | 2026-09-28 10:37 | fix(convolver): reject stale async IR completion via request-fixed generation |
| 5 | `9afc71de` | 2026-09-28 11:41 | fix(convolver): report synchronous IR load failures |
| 6 | `29dbdca9` | 2026-09-28 14:33 | fix(audioengine): restore auto gain staging state |
| 7 | `b0b46941` | 2026-09-28 16:17 | fix(runtime): close recovery obligation lifecycle gaps |
| 8 | `13a80d66` | 2026-09-28 22:19 | fix(runtime): repair reclaim accounting across deferred retry |
| 9 | `f95f524c` | 2026-09-29 00:53 | fix(epoch): prevent quarantined live readers from being excluded |
| 10 | `bda43034` | 2026-10-01 19:22 | STG-11-D18-1: fix crossfade overlap publish producing phantom record and DSP resurrection |

---

## 2. pre-D1 範囲の分類（Owner の重点指示）

**`e565019`〜D1 直前の変更は「1 つの塊」ではなく、7 件の独立 commit として既に確定済みである。**
新規 commit の作成は不要で、**push のみ**が要る。

| SHA | 変更ファイル | 分類 | 備考 |
| --- | --- | --- | --- |
| `e5e2f668` | `ConvoPeq.md`, `src/eqprocessor/EQProcessor.Processing.cpp` | **D1 準備作業ではない** — 別 work item（I-5-R1、EQ atomic wrapper 統一） | D1 が触る `EQProcessor.Core.cpp` / `.h` とは別ファイル |
| `6dc03213` | `ConvoPeq.md`, `src/audioengine/AudioEngine.Parameters.cpp`, `src/tests/.../ConvolverStateRoundTripTests.cpp` | 別 work item（bypass toggle 時の staging gain 保持） | D10/D11 が触る `EQProcessor.Parameters.cpp` とは別ファイル |
| `de07ba54` | `ConvoPeq.md`, `AudioEngine.Parameters.cpp`, `AudioEngine.h`, `ConvolverStateRoundTripTests.cpp` | 別 work item（mode 遷移時の input headroom 再検証） | **注意**: `AudioEngine.h` は D3/D4/D5 も触るが、本 commit は既に確定済み |
| `901a6c33` | `ConvoPeq.md`, `ConvolverProcessor.h`, `convolver/ConvolverProcessor.{LoadPipeline,LoaderThread,Rebuild}.cpp`, `ConvolverProcessor.LoaderThreadInline.h`, `ConvolverStateRoundTripTests.cpp` | 別 work item（stale async IR completion 拒否） | D8/D9 が触る `ConvolverProcessor.Runtime.cpp` とは別ファイル |
| `9afc71de` | `ConvoPeq.md`, `convolver/ConvolverProcessor.LoadPipeline.cpp`, `ConvolverStateRoundTripTests.cpp` | 別 work item（同期 IR load 失敗の報告） | 同上 |
| `29dbdca9` | `ConvoPeq.md`, `AudioEngine.StateIO.cpp`, `ConvolverStateRoundTripTests.cpp` | 別 work item（auto gain staging 状態復元） | **注意**: `AudioEngine.StateIO.cpp` は D7 も触るが、本 commit は既に確定済み |
| `b0b46941` | `CMakeLists.txt`, `ConvoPeq.md`, `doc/.../STG-8-D1-D3_REPAIR-GATE_20260928.md`, `RuntimePublicationOrchestrator.cpp`, `DeferredPublicationTestAccess.h`, `PublishPipelineIntegrationTests.cpp`, `STG8RecoveryObligationTests.cpp` | **STG-8**（recovery obligation lifecycle 3 件の closure） | **注意**: `DeferredPublicationTestAccess.h` / `PublishPipelineIntegrationTests.cpp` / `CMakeLists.txt` は D1〜D11 も触る |

上記 7 件は **すべて既に GitHub へ push 済みではないが、commit としては確定・SHA 固定済み**。
禁止事項（rebase / amend / reset / force push）に該当しないため、
**pre-D1 は「新規 commit を作らず push のみ」として扱う**。

---

## 3. D1〜D18 の境界確定（hunk 単位の実測帰属）

未 commit の production/test 差分 11 ファイル（計 +1007/-110）を
`git diff -U0 HEAD` の hunk 単位で分割し、各 hunk 内の `STG-11-Dn` マーカーで帰属を確定した。

### 3.1 ファイル wholly-D（hunk 分割不要）

| ファイル | +/− | 帰属 |
| --- | --- | --- |
| `src/eqprocessor/EQProcessor.Core.cpp` | +37/−10 | **D1 のみ**（全 hunk に D1 マーカー） |
| `src/eqprocessor/EEQProcessor.h` → 実体は `src/eqprocessor/EQProcessor.h` | +39/−0 | **D1 のみ**（`#include "audioengine/ISRRetireRouter.h"` ＋ 2 マーカー付きブロック） |
| `src/core/SnapshotCoordinator.cpp` | +21/−3 | **D2 のみ** |
| `src/audioengine/AudioEngine.StateIO.cpp` | +38/−7 | **D7 のみ**（enum guard 6 箇所） |
| `src/audioengine/AudioEngine.Mmcss.cpp` | +125/−37 | **D5 のみ** |

### 3.2 hunk 分割が必要なファイル

| ファイル | +/− | 帰属 | 分割可否 |
| --- | --- | --- | --- |
| `src/eqprocessor/EQProcessor.Parameters.cpp` | +21/−0 | hunk@105=**D11**、@168/@199=**D10** | 領域が完全に分離。容易 |
| `src/convolver/ConvolverProcessor.Runtime.cpp` | +19/−0 | @931/@954=**D8**、@1172/@1197=**D9** | 完全に分離。容易 |
| `src/audioengine/AudioEngine.h` | +54/−0 | 2 hunk とも **D3+D4+D5** が同一ブロック内に混在 | マーカーコメント境界あり。分割可能 |
| `src/audioengine/AudioEngine.Timer.cpp` | +195/−23 | **D3 と D4 が 252〜314 で交互に混在**、315〜395=D3、396〜470=D4、1087〜1094=D3+D4、1095=D5 | **最も困難**。行単位のマーカーフィルタで再構成が必要 |
| `src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h` | +159/−0 | 1 hunk 内に `// ===== STG-11-D3/D4/D5 =====` 3 セクション | セクションマーカーで cleanly 分割可能 |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | +50/−0 | @65+10=宣言（D1〜D11 の 10 行）、@76+10=コメント、@1368+10=呼び出し、@1381+20=本文 | 1 行 = 1 D の対応で分割可能 |
| `CMakeLists.txt` | +140/−0 | @421+124=D1+D2 の standalone target、@867/@942/@2112=各 2 行=D1+D2、**@1932+10=AudioEngineHarness ソース 10 行が D1,D2,D3,D4,D5,D7,D8,D9,D10,D11 に 1 行ずつ対応** | 行単位で完全に分割可能 |

`AudioEngine.Timer.cpp` の D3/D4 交互領域の実測（現行ファイル行番号）:

```
252:  // ★ STG-11-D3: RT-safe observation only        → D3
264:  // STG-11-D4: RT-safe success observation only   → D4
292:  // ★ STG-11-D3: ...                            → D3
299:  // STG-11-D4: ...                              → D4
307:  // STG-11-D4: ...                              → D4
315:  // ★ STG-11-D3: RT-side failure recorder ...   → D3 (〜395)
396:  // ★ STG-11-D4: RT-side success recorder ...   → D4 (〜470)
1087: // ★ STG-11-D3: ... NonRT 診断                 → D3
1091: // STG-11-D4: ... NonRT 診断                    → D4
1095: // STG-11-D5: MMCSS ... NonRT 診断              → D5
```

### 3.3 新規 test TU（untracked → 各 D で新規追加）

| ファイル | 帰属 |
| --- | --- |
| `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | D1 |
| `src/tests/STG11D2SnapshotRetireTests.cpp` | D2 |
| `src/tests/AudioEngineHarness/STG11D3AffinityFailureTests.cpp` | D3 |
| `src/tests/AudioEngineHarness/STG11D4SuccessObservationTests.cpp` | D4 |
| `src/tests/AudioEngineHarness/STG11D5MmcssObservationTests.cpp` | D5 |
| `src/tests/AudioEngineHarness/STG11D7StateEnumGuardTests.cpp` | D7 |
| `src/tests/AudioEngineHarness/STG11D8IRLengthFiniteTests.cpp` | D8 |
| `src/tests/AudioEngineHarness/STG11D9TailFiniteTests.cpp` | D9 |
| `src/tests/AudioEngineHarness/STG11D10EQEnumGuardTests.cpp` | D10 |
| `src/tests/AudioEngineHarness/STG11D11TotalGainFiniteTests.cpp` | D11 |
| `src/tests/AudioEngineHarness/STG11D18CrossfadeTests.cpp` | **D18（commit 済み）** |

### 3.4 documentation（untracked → 各 D で新規追加）

| 群 | 件数 | 帰属 |
| --- | --- | --- |
| `P1-5-IR-P2_STG-11-D{1,2,3,4,5,7,8,9,10,11}_{REPAIR-CONTRACT-AUDIT,REPAIR-IMPLEMENTATION,POST-IMPLEMENTATION-GATE}` | 30 | 各 D に対応（1 commit 3 件） |
| `P1-5-IR-P2_STG-11-D6-REPAIR-CONTRACT-AUDIT_20260930.md` ＋ duplicate | 2 | D6（NO-GO・doc のみ） |
| `P1-5-IR-P2_STG-11-D{12,13,14,15,16,17}_FRESH-DISCOVERY_20261001.md` | 6 | 各 D（NO-GO・doc のみ） |
| `P1-5-IR-P2_STG-11_DISCOVERY_20260929.md` | 1 | STG-11 umbrella（**D1 の前置き**。pre-D1 か D1 かは Owner 判断） |
| `P1-5-IR-P2_STG-11_R4P3_REVERIFICATION_20261001.md` | 1 | R4/P3（NO-GO・doc のみ。**どの commit に入れるか Owner 判断**） |
| `P1-5-IR-P2_STG-11-D18-CLOSE_20261001.md` | 1 | **D18 commit は already 固定のため追記できない** → 別 doc commit or 未追跡保持 |
| `P1-5-IR-P2_STG-11-D19_FRESH-DISCOVERY_20261001.md` | 1 | **Owner 指示で commit/push 対象外**（未追跡保持） |

### 3.5 D1〜D18 に属さない未 commit / 未追跡（Owner 判断が必要）

| 種別 | 件数 | 内容 |
| --- | --- | --- |
| pre-STG-11 文書 | 68 | STG-1〜STG-10 / N1 / N2 / R3 / Phase-2A,3A,3B,3C / Step5EK-R1 / INCIDENT / PROCESS-DEFECT / INVENTORY ほか。**D1〜D18 ではない** |
| `doc/ConvoPeqMD/<ts>/ConvoPeq.md` | 19 | `output_sourcecode_markdown.py` が生成する監査用スナップショット |
| `.memories/repo/tool-versions-2026-09-26.md` | 1 | ツール構成メモ |
| `scripts/gen_clangdb.py` | 1 | clang-tidy 用 compile DB 生成 helper |
| `vc140.pdb` | 1 | **ビルド生成物**（commit 対象ではない。`.gitignore` 追加が妥当） |
| `AGENTS.md`（+95/−8） | 1 | **pre-existing worktree 変更**。これまでの工程で「触れない」と合意済み |
| `headroom-proxy-start.ps1`（+56/−14） | 1 | 同上 |
| `doc/work113/P1-5-IR-P2_STG-8-D1-D3_REPAIR-GATE_20260928.md`（+25） | 1 | **STG-8 の post-commit gate 記録**（commit `b0b46941` 用）。pre-D1 段階归结。過去 commit のため新規 commit には入らない |

---

## 4. 計画表（Owner 確認用）

| 順序 | 範囲 | 内容 | 現在の Git 状態 | commit 済み? |
| --- | --- | --- | --- | --- |
| pre-1 | `e5e2f668` | I-5-R1 atomic wrapper 統一（EQ） | commit 済・未 push | ✅ `e5e2f668` |
| pre-2 | `6dc03213` | bypass toggle 時の staging gain 保持 | commit 済・未 push | ✅ `6dc03213` |
| pre-3 | `de07ba54` | mode 遷移時の input headroom 再検証 | commit 済・未 push | ✅ `de07ba54` |
| pre-4 | `901a6c33` | stale async IR completion 拒否 | commit 済・未 push | ✅ `901a6c33` |
| pre-5 | `9afc71de` | 同期 IR load 失敗の報告 | commit 済・未 push | ✅ `9afc71de` |
| pre-6 | `29dbdca9` | auto gain staging 状態復元 | commit 済・未 push | ✅ `29dbdca9` |
| pre-7 | `b0b46941` | STG-8 recovery obligation lifecycle 3 件 closure | commit 済・未 push | ✅ `b0b46941` |
| pre-8 | `13a80d66` | STG-9-D1 reclaim accounting 終端 | commit 済・未 push | ✅ `13a80d66` |
| pre-9 | `f95f524c` | STG-10-D1 quarantined live reader の epoch 除外防止 | commit 済・未 push | ✅ `f95f524c` |
| **D1** | EQProcessor.Core.cpp / EQProcessor.h / STG11EQRetireTests.cpp / CMakeLists(standalone target + add_test + IPO + harness 1 行 + ASAN 1 行) / PPIT(宣言・コメント・呼び出し) / doc 3 件 | Candidate B: EQ 所有 member router | **未 commit** | ❌ |
| **D2** | SnapshotCoordinator.cpp / STG11D2SnapshotRetireTests.cpp / CMakeLists(standalone target + add_test + IPO + harness 1 行 + ASAN 1 行) / PPIT / doc 3 件 | R-A: `quarantineRetireSink()` 内で Q-full→E→T 昇格 | **未 commit** | ❌ |
| **D3** | AudioEngine.Timer.cpp(D3 領域) / AudioEngine.h(D3 領域) / DeferredPublicationTestAccess.h(D3 セクション) / STG11D3AffinityFailureTests.cpp / CMakeLists 1 行 / PPIT / doc 3 件 | Candidate A: RT → atomic record → NonRT point | **未 commit** | ❌ |
| **D4** | AudioEngine.Timer.cpp(D4 領域) / AudioEngine.h(D4 領域) / DeferredPublicationTestAccess.h(D4 セクション) / STG11D4SuccessObservationTests.cpp / CMakeLists 1 行 / PPIT / doc 3 件 | Candidate A を success path へ適応 | **未 commit** | ❌ |
| **D5** | AudioEngine.Mmcss.cpp / AudioEngine.Timer.cpp(D5 hook 1 箇所) / AudioEngine.h(D5 領域) / DeferredPublicationTestAccess.h(D5 セクション) / STG11D5MmcssObservationTests.cpp / CMakeLists 1 行 / PPIT / doc 3 件 | MMCSS observation のみ移設（OS API は残置・C-D5-OS1〜4 契約化） | **未 commit** | ❌ |
| **D6** | doc 2 件のみ | Fresh Discovery **NO-GO**（ownership 層） | **未 commit** | ❌ |
| **D7** | AudioEngine.StateIO.cpp / STG11D7StateEnumGuardTests.cpp / CMakeLists 1 行 / PPIT / doc 3 件 | enum 6 箇所の範囲ガード | **未 commit** | ❌ |
| **D8** | ConvolverProcessor.Runtime.cpp(@931/@954) / STG11D8IRLengthFiniteTests.cpp / CMakeLists 1 行 / PPIT / doc 3 件 | `setTargetIRLength`/`applyAutoDetectedIRLength` の有限性ガード | **未 commit** | ❌ |
| **D9** | ConvolverProcessor.Runtime.cpp(@1172/@1197) / STG11D9TailFiniteTests.cpp / CMakeLists 1 行 / PPIT / doc 3 件 | `setTailStartSec`/`setTailStrength` の有限性ガード | **未 commit** | ❌ |
| **D10** | EQProcessor.Parameters.cpp(@168/@199) / STG11D10EQEnumGuardTests.cpp / CMakeLists 1 行 / PPIT / doc 3 件 | `setBandType`/`setBandChannelMode` の 5 列挙値ガード | **未 commit** | ❌ |
| **D11** | EQProcessor.Parameters.cpp(@105) / STG11D11TotalGainFiniteTests.cpp / CMakeLists 1 行 / PPIT / doc 3 件 | `setTotalGain` の有限性ガード | **未 commit** | ❌ |
| **D12** | doc 1 件 | Fresh Discovery **NO-GO**（session-state 層） | **未 commit** | ❌ |
| **D13** | doc 1 件 | NO-GO（resource-lifecycle 層） | **未 commit** | ❌ |
| **D14** | doc 1 件 | NO-GO（async-generation 層） | **未 commit** | ❌ |
| **D15** | doc 1 件 | NO-GO（DSP-buffer 層） | **未 commit** | ❌ |
| **D16** | doc 1 件 | NO-GO（parameter-propagation 層） | **未 commit** | ❌ |
| **D17** | doc 1 件 | NO-GO（failure-state 層） | **未 commit** | ❌ |
| **D18** | DSPTransition.h / ISRDSPHandle.cpp / STG11D18CrossfadeTests.cpp / NormalRetireDSPHandleCompareTests.cpp / CMakeLists 1 行 / PPIT / doc 3 件 | F-A / F-B / F-C | commit 済・未 push | ✅ `bda43034` |

**順序の問題**: 現在は `bda43034`（D18）が HEAD にあり、D1〜D11 は未 commit のまま
worktree に残っている。Phase 2 を「D1 から順に commit → push」するには、
**HEAD を `f95f524c` に戻したうえで D1 を commit する**必要がある。
ただし `bda43034` は `f95f524c` の子であり、`git reset` 系は禁止されている。

### 4.1 禁止事項と整合する唯一の実現手段

```text
現状:  f95f524c ── bda43034(HEAD)
目標:  f95f524c ── D1 ── D2 ── … ── D11 ── bda43034
```

`bda43034` の SHA を変えない（= rebase / amend / reset 禁止）으면서
D1〜D11 を `bda43034` の**前**に置くには、
**新しい branch を `f95f524c` から作り、D1〜D11 を順に commit してから push**する
（既存 `main` と既存 10 commits は一切変更しない）のが禁止事項に抵触しない。

```text
git switch -c stg11-d1-d18 f95f524c     # 既存 main / 既存 SHA は不変
  → D1 commit → push → confirm
  → … → D11 commit → push → confirm
  → D18: 既存 bda43034 をこの系列のonto にして push
```

D18 の載せ方として 2 案ある（Owner 選択要）:

- **(A) cherry-pick 相当の新 SHA を避ける**: `bda43034` はそのままとし、
  D1〜D11 の push 完了後に「D1〜D11 の系列の onto を `bda43034` の**親**に
  付け替える」，还需要 history rewrite になり **禁止事項に抵触する**。
- **(B) 推奨**: D18 は `bda43034` のまま。 系列の push 完了後、
  `bda43034` の**祖先である `f95f524c` Already が origin にある**ことを利用し、
  D1〜D11 を含む新 branch の先端を `git push` したうえで、
  `bda43034` を **merge commit ではなく**その系列の続きとして置く。
  現実的には「D18 の変更内容を D1〜D11 の後に再適用した新 commit」を作る必要性が生じる。

**→ この点は Phase 1 の回答で Owner の明示指示が必要**（§6 質問 Q1）。

---

## 5. 各 D commit の作成手順（Phase 2 で実施予定）

1. `git status --short --branch` / `git diff --check` / `git diff --stat`
2. `git hash-object -w --path` + `git update-index --cacheinfo` で
   **当該 D の content のみ**を index に構築（D18 で採用した手法）。
   共有ファイル（`CMakeLists.txt` / `PublishPipelineIntegrationTests.cpp` /
   `AudioEngine.{h,Timer.cpp}` / `EQProcessor.Parameters.cpp` /
   `ConvolverProcessor.Runtime.cpp` / `DeferredPublicationTestAccess.h`）は
   HEAD content から当該 D の行だけを再構成。
3. 新規 test TU と doc を `git add`。
4. `git diff --cached` で当該 D 以外が混入していないことを確認。
5. commit（`conventional commit` 形式で `fix(...)` / `docs(...)`）。
6. `git log -1 --oneline --decorate` / `git status --short --branch` を確認。
7. `git push origin <branch>`。
8. `git fetch origin` → `git rev-parse HEAD` と `git rev-parse origin/<branch>` の一致確認。

---

## 6. Owner への確認事項

| # | 質問 | 推奨 |
| --- | --- | --- |
| **Q1** | D18 の `bda43034` を SHA 固定のまま D1〜D11 の**後**に置けない制約（rebase/reset 禁止）とどう整合させるか。上記 4.1 のとおり、既存 SHA を一切変更しない方法では D1〜D11 と `bda43034` を 1 本の線形系列に並べられない。 | (a) `bda43034` の**内容と SHA を保持**したまま、新 branch 上に D1〜D11 を作其後に `bda43034` を **`--no-ff` merge**（D18 は独立 commit のまま維持され SHA も変わらない。merge commit 1 件が加わる）。(b) D18 の変更を D1〜D11 の後に**同内容で再適用**し新 SHA を作る（D18 は「既存独立 commit 維持」の指示に反する）。→ **(a) を推奨** |
| **Q2** | pre-D1 の 7 commits + STG-9/10 の 2 commits（計 9）は push のみでよいか | はい。既に commit 済み・SHA 確定。`git push` の対象にする |
| **Q3** | `ConvoPeq.md` の扱い。履歴では毎 commit 同梱だが、D1〜D11 は累積値しか生成できない | 各 D の staged tree を temp worktree で checkout し、そこで `output_sourcecode_markdown.py` を実行して**その D 時点の正確な `ConvoPeq.md`** を生成・commit する（履歴の慣習を維持・11 回 × 約 80 秒） |
| **Q4** | D6 / D12〜D17（NO-GO・doc のみ）に独立 commit を作るか | 作る（Owner の「D ごとに独立 commit」に忠実） |
| **Q5** | `P1-5-IR-P2_STG-11_DISCOVERY_20260929.md`（STG-11 umbrella）と `STG-11_R4P3_REVERIFICATION_20261001.md` の帰属 | umbrella は D1 commit に同梱、R4P3 は D17 commit に同梱（最終監査成果物）を推奨 |
| **Q6** | `P1-5-IR-P2_STG-11-D18-CLOSE_20261001.md`（D18 commit 後に作成・未追跡） | D18 は amend 不可のため、**(a) D11 commit に同梱**（時系列が乱れる）または **(b) 最終 D18 push 後に doc のみ 1 commit**。→ (b) を推奨（Owner の「D18 は独立 commit 維持」に最も Bento 適合） |
| **Q7** | pre-STG-11 文書 68 件・`doc/ConvoPeqMD/` 19 件・`vc140.pdb`・`scripts/gen_clangdb.py`・`.memories/` の扱い | **D1〜D18 の commit には一切混入させない**（未追跡保持）。`vc140.pdb` のみ `.gitignore` 追加を提案 |
| **Q8** | `AGENTS.md` / `headroom-proxy-start.ps1` / `STG-8-D1-D3_REPAIR-GATE_20260928.md`（+25、STG-8 の post-commit 記録） | **D1〜D18 の commit には混入させない**（未 commit 保持）。STG-8 記録は過去 commit のため新規 commit には入らない |
| **Q9** | 各 D commit 前に Debug/Release build + CTest 45/45 を取るか（1 D あたり約 10 分、計約 2 時間） | **推奨: 取らない**。最終 D11 commit 後に Debug/Release 45/45 を 1 回取得し、途中は `git diff --cached` の構造確認のみ。ただし D3/D4/D5 は marker 分割がZI最も 복잡なため、**D3 commit 前だけ** Debug build で分割後 state の compile を確認したい |
| **Q10** | `P1-5-IR-P2_STG-11-D19_FRESH-DISCOVERY_20261001.md` | Owner 指示どおり **commit / push 対象外**（未追跡保持） |

---

## 7. 禁止事項の遵守確認

| 禁止 | 本計画での扱い |
| --- | --- |
| `git reset --hard` | 使用しない |
| `git clean` | 使用しない |
| `git rebase` | 使用しない（新 branch 生成 + 通常 commit のみ） |
| `git commit --amend` | 使用しない（D18 は現状のまま） |
| `git merge --squash` | 使用しない |
| `git push --force` / `--force-with-lease` | 使用しない（通常 push のみ。fast-forward でない場合は Q1 の指示を仰ぐ） |
| D19 の commit / push | 実施しない |
| D20 への 진행 | 実施しない |