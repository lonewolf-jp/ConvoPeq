# 全セッション残課題の最終棚卸し（2026-11-01）

- 対象リポジトリ: `C:\VSC_Project\ConvoPeq`（唯一の作業ディレクトリ）
- 作業セッション: `main-develop`（唯一）
- 基準コミット: `d24a5539`（`HEAD` = `origin/main`）
- 本作業は **読み取り専用調査＋文書化**。製品コード・合格条件・HOLD 状態は変更していない。

---

## 0. 方法（Owner 指示 A への適合）

### 0.1 未統合判定に使った証拠（単独使用の禁止を遵守）

| 証拠 | 何を測るか | 単独使用 |
|---|---|---|
| `git merge-base --is-ancestor <branch> origin/main` | ブランチが main の祖先か（＝完全に取り込み済みか） | 可（肯定的判定のみ） |
| `git rev-list --left-right --count origin/main...<branch>` | 先行／後退コミット数 | 補助 |
| `git cherry origin/main <branch>` | コミットの **patch-id 等価性**（`+` = 非等価） | 補助 |
| **追加行の一致率**（`git diff main...branch` の追加行が main のツリーに存在するか） | 内容が main に存在するか（squash 統合の検出） | 可（閾値 98%） |
| **該当コードの読解** | 意味的な整合（コメント・条件式・呼び出し順） | 可（最終確認として必須） |
| テスト登録の確認（登録行の存在） | テストが実際に走るか | 補助 |
| CI 結果（check-runs API） | 統合後の健全性 | 補助 |

**`git diff A...B`（三点）単独での判定は行っていない。** 三点比較は「merge-base からの差分」であり、
分岐後に main 側だけが進んだ内容も含むため、未統合判定には使えない（2026-10-31 の誤判定の原因）。

### 0.2 調査対象（12 情報源）

| # | 情報源 | 実測 |
|---|---|---|
| 1 | OpenChamber セッション | `opencode.db` の `session_v2` **165 件**（+ 旧 `session` 139 件） |
| 2 | GitHub PR | **76 件**（本 PR #96 を除く）= MERGED 66 / CLOSED（未マージ）10 |
| 3 | ローカルブランチ | **70 本**（`main` 含む）＝ 68 ＋ `main` ＋ 本作業ブランチ |
| 4 | リモートブランチ | **44 本**（`origin/HEAD` 含む） |
| 5 | 全 ref の監査 | **112 ref**（`main` と `origin/HEAD` を除く） |
| 6 | worktree | **8 個**（正本 1 ＋ セッション／残置 7） |
| 7 | stash | **0 件**（2026-10-31 に drop 済み） |
| 8 | 未追跡ファイル | **18 件**（意図的に追跡しないもの） |
| 9 | 残課題台帳 | `RESIDUAL-LEDGER-20261028.md`（第 28 次まで）ほか証拠文書 |
| 10 | main の実ソース | マーカー検索で存在確認（下記 §2.3） |
| 11 | セッション差分記録 | **取得不能**（§1.2） |
| 12 | CI | `d24a5539`: success 10 / skipped 1 / failed 0 |

---

## 1. セッション棚卸し

### 1.1 セッション数と分布

| ディレクトリ | セッション数 |
|---|---|
| `C:/VSC_Project/ConvoPeq`（正本） | 142 |
| `…/worktree/f34893/j195-notion-report` | 14 |
| `…/worktree/…/bright-falcon` | 7 |
| `…/worktree/…/crisp-zebra` | 1 |
| `C:/Users/user`（リポジトリ外） | 1 |
| **合計** | **165** |

- 既定タイトル 16 / 命名済み 149、アーカイブ済み 145、休止 1。
- 期間: 2026-07-02 〜 2026-10-31。

### 1.2 セッション単位の差分記録 — **取得不能（確定）**

| 項目 | 実測 |
|---|---|
| `session_v2.summary_files > 0` のセッション | **0 件**（165 件すべて `0`） |
| `session_v2.summary_diffs IS NOT NULL` のセッション | **0 件**（すべて NULL） |
| `~/.local/share/opencode/storage/session_diff/*.json` | **7 ファイル、すべて 2 B（`{}`）** |
| `session_message` | 71,585 行（会話記録はあるが差分ではない） |

**したがって、セッション単位の「何を変更したか」は復元できない。** 本棚卸しは、
ブランチ・コミット・PR・worktree・台帳・証拠文書から**発生元を再構成**した。
この制約は Owner 指示 3 の想定どおりであり、**記録として明示する**。

### 1.3 セッション → 成果物の対応（再構成）

| セッション群（タイトル） | 成果物（ブランチ / PR） | main 収載 |
|---|---|---|
| B2 Stage-1 / T1 計測 | `feat/b2-stage1-t1-measure-20261028` → PR #75 | 済 |
| H-2-R1 mixed tuple | `work/h2r1-mixed-tuple-20261026` → PR #38 | 済 |
| D8-2-B1/B2（T9 / T10 / T10-c） | `test/h3-t9-*`, `test/t10-*`, `test/t10c-*` → PR #46/#50/#44 | 済 |
| P2-1 resolve 契約 | `test/p2-1-resolve-states-20261027` → PR #56 | 済 |
| P0-1 / P0-2 / P1-E1 文書系 | `feature/p0-*`, `feature/p1-e1-*` | 済 |
| A2 プローブ | `test/a2-rt-allocation-probe`, `fix/a2-probe-lint-ae011` → PR #52/#53 | 済 |
| F-4 D2b 診断 | `test/f4-d2b-overwrite-redrive-diag` → PR #54 | 済 |
| STG-11-D18 修正 | `crisp-zebra`（内容一致 99.2%） | 済（F-A/F-B/F-C 現存） |
| bright-falcon 永続キャッシュ | `bright-falcon`（内容一致 99.2%） | 済 |
| R-46/47/48/49/50 の CI 系 | `fix/compile-blockers`, `ci/windows-msvc-build`, `ci/sanitizer-oneapi-r48`, `ci/r50-*` | 済（PR #74/#76/#78/#80/#83/#84/#85） |
| R-51（F1-b pin 保護） | `fix/r51-f1b-pin-protection-clean` → PR #89 | 済 |
| R-51 恒久テスト | `test/r51-two-engine-permanent-20261031` → PR #95 | 済 |
| R-06 U1 / U2 / U5 準備 | `feat/r06-u1-*` PR #90、`feat/r06-u2-*` PR #92、`feat/r06-u5-prep-*` PR #93 | 済 |
| F2 堅牢化 | `fix/f2-quarantine-destroy-lifo-20261031` → PR #94 | 済 |

---

## 2. ブランチ・PR 棚卸し（112 ref の機械判定）

### 2.1 判定結果

| 判定 | 件数 | 意味 |
|---|---|---|
| **MERGED** | 85 | main の祖先（完全取り込み） |
| **EQ-MERGED** | 11 | 先行コミットはあるが `git cherry` で `+` が 0（全 patch が main に等価） |
| **CONTENT-PRESENT** | 9 | squash 等で patch-id が異なるが、**追加行の一致率 ≥ 98%** |
| **REVIEW(generated-only)** | 2 | `ConvoPeq.md` のみ変更（生成物、陳腐化） |
| **REVIEW** | 5 | 要個別判定（下表） |
| 合計 | 112 | — |

### 2.2 要個別判定 7 件の結論

| ブランチ | 変更対象 | 一致率 | 判定 | 根拠 |
|---|---|---|---|---|
| `ci/r51-diag-only-20261030` | `AudioEngine.Threading.cpp` / `ISRDSPHandle.cpp` | 10.0% | **証拠専用（統合しない）** | `CONVO_R51_DIAG=1` のときだけ stderr へ出す**計測専用計装**。既定は無効、製品挙動の変更なし。R-51 の修理自体は PR #89 で main に統合済み |
| `ci/r51-two-engine-on-pr82` | 製品 6 ＋ テスト 2 ファイル | 78.9% | **証拠専用** | PR #82（REJECT）上に再現テストを載せたもの。恒久テストは PR #95 で main に統合済み |
| `ci/r51-two-engine-repro-20261030` | テスト 1 ファイル | 60.3% | **証拠専用** | 同上（再現用）。恒久版が main に存在 |
| `verify/r51-two-engine-on-merged-main-20261030` | テスト 1 ファイル | 60.3% | **証拠専用** | 同上（検証用） |
| `ci/h2r1-release-measurement-20261030` | `.github/workflows/h2r1-release-measurement.yml`（新規 235 行） | 46.1% | **証拠専用（測定専用 CI）** | main に H-2-R1 の測定ワークフローは無い。測定ビークル本体（PR #38）は統合済みで、本ワークフローは一回限りの計測用。**製品ソースを含まない** |
| `ci/r49-selected-tests-probe-20261029` | `.github/workflows/windows-msvc-build.yml`（+51） | 22.2% | **証拠専用（診断）** | 選択テスト実行＋ログ取得のプローブ。main の `windows-msvc-build.yml` には同等の選択実行は無いが、**診断専用**で製品に影響しない |
| `chore/convoPeqmd-023ae16f-20261029` | `ConvoPeq.md` | 0.0%（スタンプ行のみ） | **陳腐化（廃棄候補）** | 現在の正本は `d4716a2c` から生成したもの。旧スナップショットは `doc/ConvoPeqMD/` に保全されている |

### 2.3 main 側の実コード存在確認（台帳記述に依存しない検証）

| 対象 | main での確認結果 |
|---|---|
| R-01 B2 Stage-1 車両 | `runB2Stage1PositiveControls` 宣言＋呼び出し（`PublishPipelineIntegrationTests.cpp:1361/1368`）、`B2_STAGE1_MEASURE` 分岐あり |
| R-05 A2 プローブ | `namespace a2probe {`（`V11EngineRetireProtectionTests.cpp:78`）ほか 44 箇所 |
| R-06 U1 リング | `RtMeasureRing.h` を `AudioEngine.Init.cpp` / `AudioEngine.Processing.AudioBlock.cpp` が include |
| R-06 U2 集計 | `pollIfDue()` の定義と `AudioEngine.Threading.cpp:386` からの呼び出し |
| R-06 U5 準備 A1' | `Mode::TimeOnly` / `Mode::Full` の分岐、`RtMeasureRingTests.cpp:91` の期待値 |
| F2 堅牢化 | `F2 hardening` マーカー 5 箇所（`V11EngineRetireProtectionTests.cpp`） |
| R-51 恒久テスト | `checkR51TwoEngineOwnerCollision` 定義（:1563）＋登録（:2995） |
| R-51 修正（F-A/F-B/F-C） | `ISRDSPHandle.cpp:244/261`、`DSPTransition.h:110/125/148`、D18-D の claim→register→begin 順（L121 → L133 → L135） |
| R-03 D2b 診断 | `recSeq` を含む診断タプル（`STG8RecoveryObligationTests.cpp:351/467`） |
| R-04 H-2-R1 車両 | `measureH2R1MixedGainTupleWindow`（`PublishPipelineIntegrationTests.cpp:1265`） |
| V-11 候補修正 / R8 候補修正 | **main に存在**（`ISRDSPHandle.cpp:201-209` の Retired/DestroyPending 早期 return、`RuntimePublicationOrchestrator.cpp:694-697` の obligation terminalization） |
| V11 群の登録テスト数 | **23 件**（`if (!check...)` の登録行） |

### 2.4 PR 棚卸し

| 区分 | 件数 | 備考 |
|---|---|---|
| 総数 | **76**（本 PR #96 を除く） | PR 番号は #1〜#95（欠番あり） |
| MERGED | **66** | — |
| CLOSED（未マージ） | **10** | #18〜#25（タスク自動生成 PR・未マージで close）、#79（必須チェックの試験用・意図的）、#82（REJECT：重複・陳腐化） |
| OPEN | **0** | #82 を 2026-10-31 に CLOSE。現在の唯一の OPEN は本 PR #96 |

---

## 3. worktree 棚卸し

| worktree | HEAD | 追跡変更 | 未追跡 | main 先行 | 判定 |
|---|---|---|---|---|---|
| `C:\VSC_Project\ConvoPeq`（正本） | `d24a5539` | 0 | 18 | 0 | **正本** |
| `…/j195-notion-report`（本セッション） | `bda43034` | **5** | 752 | 1 | 下記 §3.1 |
| `…/j237-baseline` | `d79b8be4` | 8 | 0 | 2 | 内容は main に存在（99.2%） |
| `…/j237-baseline-clean` | `d79b8be4` | 0 | 0 | 2 | 同上 |
| `…/bright-falcon` | `d79b8be4` | 3 | 243 | 2 | 内容は main に存在（99.2%）。未追跡はスクリーンショット等 |
| `…/crisp-zebra` | `bda43034` | 0 | 0 | 1 | 内容は main に存在（99.2%） |
| `…/wt-g3` | `812abcc8` | 0 | 0 | 0 | 統合済み（PR #36） |
| `…/wt-r16` | `3f3225d7` | 0 | 0 | 0 | 統合済み（PR #58 系） |

### 3.1 `j195-notion-report` の追跡変更 5 件は **main に存在（統合済み）**

| ファイル | 変更 | 判定 |
|---|---|---|
| `src/audioengine/ISRDSPHandle.cpp` | `resolve()` で `Retired` / `DestroyPending` を解決不能に（V-11 candidate fix） | **main に同一コメント・同一条件で存在**（`ISRDSPHandle.cpp:201-211`） |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | `clearDeferredForShutdown()` で obligation を terminalize（R8 candidate fix） | **main に存在**（`:694-697`） |
| `AGENTS.md` / `CMakeLists.txt` / `opencode.json` | 環境・構成の記述差 | 追跡情報の差異。製品コードではない |

→ 全体の一致率 52.9% は、上記 3 つの構成ファイルの差（後続で別内容に更新済み）に引きずられた値である。
**製品ソースの未統合変更は 0**。

---

## 4. stash・未追跡

| 対象 | 実測 | 状態 |
|---|---|---|
| stash | **0 件** | 2026-10-31 に `fc2fb974` を drop（差分は生成時刻 1 行のみ） |
| 未追跡（`-uall`） | **18 件** | `doc/work113` 中間スナップショット 6、分割版 4、backup 6、`vc140.pdb` 1、`.memories` 1 |

---

## 5. 残課題レジスタ（Owner 指定の 12 項目）

各項目の「次に実施すべき作業 / 必要権限 / 完了条件」を明示する。

### RES-01 B2 全体（R-02）

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元セッション | A2/B2 監査系（`main`）、`j195-notion-report` |
| 2 | 当初の目的 | 診断カウンタによるメモリ計測の採否決定（Stage 1/2/3） |
| 3 | 実装済み | Stage 1 車両（R-01）＝ main 統合済み・計測 `VALID` / `b2_adoption=eligible` |
| 4 | main に存在 | 車両は存在。**全体設計・Stage 2/3 は未実装** |
| 5 | テスト・CI | 車両は CI 緑。Stage 2/3 のテストなし |
| 6 | 未完成 | Stage 2/3 の設計・実装・評価基準の確定 |
| 7 | 理由 | Stage 1 の結果を「採用可否」に翻訳する Owner 判断が未了 |
| 8 | 依存 | R-01（完了）、R-05（A2 本番適用）、R-06（計測） |
| 9 | Owner 判断 | **未**（採用可否） |
| 10 | 次の作業 | Stage 1 実測（`overall_result=VALID`）を根拠に、**採用/不採用の判断材料**を 1 枚に整理 |
| 11 | 必要権限 | 設計・文書化（承認済み）。実装は別承認 |
| 12 | 完了条件 | Owner が採用可否を決定し、採用なら Stage 2/3 の計画が承認される |

### RES-02 A2 本番適用（R-05）

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | A2 設計・プローブセッション |
| 2 | 当初の目的 | 本番経路で確保カウンタを観測 |
| 3 | 実装済み | テスト専用プローブ（PR #52/#53、main 統合済み） |
| 4 | main に存在 | プローブは存在。本番経路の計測は無い |
| 5 | テスト・CI | プローブはビルド・実行成立 |
| 6 | 未完成 | 本番適用（製品コード変更） |
| 7 | 理由 | **本番適用は未承認**（NO-GO 維持） |
| 8 | 依存 | RES-01（B2 全体） |
| 9 | Owner 判断 | **未**（NO-GO 継続） |
| 10 | 次の作業 | 本番適用の設計（変更点・RT 制約・オーバーヘッド・ロールバック）を作成して承認を求める |
| 11 | 必要権限 | 設計・文書化のみ（実装は別承認） |
| 12 | 完了条件 | 設計承認 → 実装 → RT 回帰なしの実証 |

### RES-03 R-06 U3 / U4

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | RT 計測検討セッション |
| 2 | 当初の目的 | U3=`operator new/delete` 置換の観測、U4=ロック計測 |
| 3 | 実装済み | なし（U1/U2/U5 準備のみ完了） |
| 4 | main に存在 | なし |
| 5 | テスト・CI | なし |
| 6 | 未完成 | 設計・実装・測定すべて |
| 7 | 理由 | **HOLD**（Owner 未承認） |
| 8 | 依存 | U5 の結論（測定系の妥当性） |
| 9 | Owner 判断 | **HOLD 継続** |
| 10 | 次の作業 | HOLD 理由と依存関係の整理のみ（実装しない） |
| 11 | 必要権限 | なし（文書化のみ） |
| 12 | 完了条件 | Owner が HOLD を解除し設計を承認 |

### RES-04 R-06 U5（`max` 指標・既定 OFF）

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | RT 計測検討セッション（`main-develop`） |
| 2 | 当初の目的 | 実時間計測モード（Full）の有効化可否判定 |
| 3 | 実装済み | A1'（モード 0/2/1・共通サンプル領域・リング drain）＝ PR #93 統合済み |
| 4 | main に存在 | する（`RtMeasureRing.h`、呼び出し点、`RtMeasureRingTests`） |
| 5 | テスト・CI | `RtMeasureRingTests` 登録済み。CI 緑 |
| 6 | 未完成 | **合格判定が得られていない**（p99 PASS / max FAIL） |
| 7 | 理由 | `max` が単発遅延事象で決まり、測定分解能（σ=158 µs）が許容値 +50 µs を上回る |
| 8 | 依存 | 測定構造の改善（RES-04-B）、環境統制 |
| 9 | Owner 判断 | **FAIL・既定 OFF・B-1/B-2 未承認・追加反復未承認**（2026-11-01） |
| 10 | 次の作業 | **測定構造の技術提案**（`U5-MAX-MEASUREMENT-PROPOSAL-20261101.md`）を提示し、採否を仰ぐ |
| 11 | 必要権限 | 設計・文書化のみ |
| 12 | 完了条件 | 測定構造の変更が承認され、現行基準のまま判別可能な測定が成立する、または「この環境では判定不能」と確定する |

### RES-05 F-4 / D2b 記録消失（R-03）

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | F-4 調査セッション |
| 2 | 当初の目的 | `overwrite/recSeq/redrive` の記録消失の再現・原因特定 |
| 3 | 実装済み | D2b 診断タプル（PR #54 統合済み） |
| 4 | main に存在 | 診断は存在 |
| 5 | テスト・CI | `[D2B-SELFCHECK] PASS` を確認（本セッションで複数回観測） |
| 6 | 未完成 | 再現せず（38 反復で `overwrite=1/recSeq=0/redrive=0` 一定） |
| 7 | 理由 | **Owner 判断で調査停止** |
| 8 | 依存 | 新証拠（R1 環境・時系列・ログ） |
| 9 | Owner 判断 | **停止継続**（新証拠が出るまで再開しない） |
| 10 | 次の作業 | なし（新証拠が出た場合のみ再開） |
| 11 | 必要権限 | 新証拠がある場合のみ調査 |
| 12 | 完了条件 | 再現手順の確立、または「再現しない」ことの合意 |

### RES-06 STG-8 群のテスト・フレーク（**本棚卸しで新規登録**）

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | 本セッション（`main-develop`）のローカル検証で 3 回観測 |
| 2 | 当初の目的 | （新規）STG-8 系テストの決定性確保 |
| 3 | 実装済み | `RuntimePublicationOrchestrator.cpp` の R8 candidate fix（obligation terminalization）は main に存在 |
| 4 | main に存在 | 上記 fix は存在。**フレーク自体は未解消** |
| 5 | テスト・CI | 観測: `checkSTG82OverwriteTerminalizesEvicted` → `[STG-8-D2] FAIL: unstable (L=0 hasDeferred=0)`、`checkSTG8ClearIsIdempotent` → `[R8-2] FAIL: first clear (L=0 hasDeferred=1)`。CI（リモート）では再現せず、ローカル Release で断続 |
| 6 | 未完成 | フレークの原因特定と決定化 |
| 7 | 理由 | 未調査（本タスクの範囲外だった） |
| 8 | 依存 | なし（ただしローカル検証の再現性に影響し、F2/R-51 の確認を妨げる） |
| 9 | Owner 判断 | **未**（調査・修正の承認が必要） |
| 10 | 次の作業 | 失敗ログの収集と、`L`（obligation 残数）と `hasDeferred` の遷移の追跡（テストのみの調査） |
| 11 | 必要権限 | 調査＋テスト修正（製品コード修正が必要になった時点で別承認） |
| 12 | 完了条件 | 同一条件で 20 回連続 PASS（フレーク 0） |

### RES-07 `doc/work113` の残り 6 件

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | P3 分類セッション |
| 2 | 当初の目的 | 証跡文書のリポジトリ収録可否 |
| 3 | 実装済み | **71 件を追跡**（`d4716a2c`） |
| 4 | main に存在 | する（71 件） |
| 5 | テスト・CI | 文書のみ（CI 影響なし） |
| 6 | 未完成 | `FRESH-DISCOVERY` 中間 6 件の扱い |
| 7 | 理由 | 中間スナップショット（後続記録で置換済み）のため保留 |
| 8 | 依存 | なし |
| 9 | Owner 判断 | 追跡 71 件は承認済み。**残り 6 件は未判断** |
| 10 | 次の作業 | 6 件の内容確認と、追跡／非追跡の最終決定 |
| 11 | 必要権限 | 追跡する場合は `git add` の承認 |
| 12 | 完了条件 | 6 件の扱いが決定し、未追跡件数が確定 |

### RES-08 未追跡 18 件（保全）

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | 複数（backup は Python 3.15 移行時） |
| 2 | 当初の目的 | 作業環境の保全 |
| 3 | 実装済み | 分類済み（`UNTRACKED-CLASSIFICATION-20261031.md`） |
| 4 | main に存在 | 追跡しない方針 |
| 5 | テスト・CI | 影響なし |
| 6 | 未完成 | — |
| 7 | 理由 | backup・デバッグ生成物・生成物・中間記録 |
| 8 | 依存 | RES-07 |
| 9 | Owner 判断 | **削除しない**（保全継続） |
| 10 | 次の作業 | 保全継続（削除・移動しない） |
| 11 | 必要権限 | 削除は未承認 |
| 12 | 完了条件 | なし（継続運用） |

### RES-09 他セッションの worktree 6 個

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | 各セッション |
| 2 | 当初の目的 | 並行作業 |
| 3 | 実装済み | 内容は main に統合済み（§3） |
| 4 | main に存在 | する |
| 5 | テスト・CI | — |
| 6 | 未完成 | 未追跡物（スクリーンショット等）の扱い |
| 7 | 理由 | 削除は未承認 |
| 8 | 依存 | なし |
| 9 | Owner 判断 | **読み取り専用・削除未承認** |
| 10 | 次の作業 | 削除する場合の事前確認（固有データの有無） |
| 11 | 必要権限 | 削除の承認 |
| 12 | 完了条件 | 削除可否の決定 |

### RES-10 陳腐化ブランチ（証拠専用 7・生成物 2・backup 1）

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | 各セッション |
| 2 | 当初の目的 | 計測・証拠・正本生成 |
| 3 | 実装済み | 製品ソースを含まない（§2.2） |
| 4 | main に存在 | 不要（証拠専用） |
| 5 | テスト・CI | — |
| 6 | 未完成 | — |
| 7 | 理由 | 証拠保全のため保持 |
| 8 | 依存 | なし |
| 9 | Owner 判断 | **削除未承認** |
| 10 | 次の作業 | 削除可否の判断材料（各ブランチの最終コミット・目的・参照証拠）を提示済み |
| 11 | 必要権限 | ブランチ削除の承認 |
| 12 | 完了条件 | 削除可否の決定 |

### RES-11 台帳の文書衛生（連番欠落・記録の二重管理）

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | 本セッション |
| 2 | 当初の目的 | 台帳の一貫性 |
| 3 | 実装済み | 第 28 次まで記録。修正差分も保存済み |
| 4 | main に存在 | 台帳は証拠ディレクトリ（リポジトリ外）にあり、**Git 追跡外** |
| 5 | テスト・CI | — |
| 6 | 未完成 | ①第 25 次が欠番（第 27 次に統合）②台帳が Git 管理外で、レビュー時に参照しづらい |
| 7 | 理由 | 記録は証拠ディレクトリに集約する運用のため |
| 8 | 依存 | RES-12 |
| 9 | Owner 判断 | 未（文書運用の変更は要承認） |
| 10 | 次の作業 | 欠番の注記、および台帳の一部を `doc/` へ要約移植する案の提示 |
| 11 | 必要権限 | 文書変更（承認済み範囲） |
| 12 | 完了条件 | 台帳の参照経路が文書化され、欠番が説明される |

### RES-12 文書の Git 追跡（本次調査の成果物）

| # | 項目 | 内容 |
|---|---|---|
| 1 | 発生元 | 本セッション（Owner 指示 A/B） |
| 2 | 当初の目的 | 残課題と技術提案をリポジトリでレビュー可能にする |
| 3 | 実装済み | `doc/residual-20261101/` に 4 文書を作成 |
| 4 | main に存在 | 本 PR で追加 |
| 5 | テスト・CI | docs のみ（ビルド影響なし） |
| 6 | 未完成 | — |
| 7 | 理由 | — |
| 8 | 依存 | なし |
| 9 | Owner 判断 | 文書 PR は条件付き承認済み |
| 10 | 次の作業 | CI 緑 → 差分確認 → main 統合 → `ConvoPeq.md` 再同期 |
| 11 | 必要権限 | 条件付き承認済み |
| 12 | 完了条件 | main に統合され、正本が再同期される |

---

## 6. 結論

1. **main に取り込むべき未統合の製品ソース変更は 0 件**（112 ref・8 worktree・76 PR の多証拠判定）。
2. 未統合に見えるものはすべて **(a) squash 済みの重複**、**(b) 計測・診断専用**、**(c) 生成物**である。
3. セッション単位の差分記録は**構造的に取得不能**（`summary_files=0` / `summary_diffs=NULL` / `session_diff` 空）。
   provenance はブランチ・PR・台帳から再構成した。
4. 未完成事項は **RES-01〜RES-12** の 12 件。うち Owner 判断が必要なのは
   **RES-01 / RES-02 / RES-03 / RES-06 / RES-07 / RES-09 / RES-10 / RES-11**。
5. 本棚卸しで**新規に発見**した実課題は **RES-06（STG-8 のフレーク）**である。
   これはローカル検証の再現性を損ない、F2/R-51 の確認を妨げていた。
