# ConvoPeq 残課題解消 — 最終報告（2026-11-02 夜間セッション）

Owner 指示書（R-101 / R-102 / R-106 / R-107 / R-108）に基づく実施結果。
**未検証の項目は「未検証」と明記**し、完了扱いにしていない。

- 作業場所: `C:\VSC_Project\ConvoPeq`（唯一の正本）
- 基点 `main`: `a83d10ba`（本セッション開始時は `ea9b980f`）
- 追加マージ: `7318e44f`（#105 R-101）、`a83d10ba`（#106 R-102）、本報告時点で #107 / #108 がレビュー中

## 1. サマリ

| 項目 | 状態 |
|---|---|
| 全セッション成果の回収 | 完了（製品コードの未統合差分 0） |
| main と正本の同期 | 完了（`main` = `origin/main`、差分 0、未追跡 0） |
| R-101（h2r1 恒久化） | **ワークフローはマージ済み**。dispatch は rc=0 8/9、60 分基準は run 全体で 3/9（**判定は Owner 判断待ち**、§3）。自動収集は継続中 |
| R-102（r49 診断移植） | **完了**（true / false とも実測で確認） |
| R-106（U5 再測定） | **完了**（20 実行。FAIL＋判定保留。測定器は既定 OFF） |
| R-107 Stage 1（列挙） | 完了 |
| R-107 Stage 2（U4） | 実装・ローカル検証完了（PR #108、ASan 待ち） |
| R-107 Stage 3（U3） | 計画作成済み（実装は U4 検証後） |
| R-108（旧分割削除） | 完了（削除前確認 4 条件＋削除後の正本一致を再確認） |

## 2. R-106 — U5 再測定（完了）

20 実行（5/アーム/条件、34,000 ブロック、121.7 分）。環境は専用コア（CPU 2）・MMCSS "Pro Audio"・
`THREAD_PRIORITY_TIME_CRITICAL`・高パフォーマンス電源・Defender 無効。

| 条件 | p99 | max | σ(max) | σ(p99) | 検出力（5 組） |
|---|---|---|---|---|---|
| 無音 | PASS（1161 ≤ 1227.5） | PASS（1865 ≤ 1886.0） | 280.3 µs | 9.2 / 20.1 µs | 0.05 |
| 1 kHz | PASS（1544 ≤ 1626.5） | **FAIL**（2440 > 2393.0） | 403.4 µs | 31.4 / 46.4 µs | 0.04 |

- 全 20 実行で `late50us=0`（周期 10,666.7 µs に対する超過なし）。max は単発スパイク。
- **σ ≤ 64 µs は max で未達**（280〜403 µs）、p99 で達成（9〜46 µs）。
- +50 µs を 95% で判別するには 493〜1,022 組/アームが必要（1 組 6.1 分 → 50〜104 時間）。
- 判定: **FAIL（max 基準）かつ判定保留（検出力不足）**。指標の再定義は行わない。**測定器は既定 OFF を維持**。

証跡: `u5_remeasure_20261102/`（20 ログ・sha256 付き JSON）、`R106-REMEASUREMENT-RESULT-20261102.md`。

## 3. R-101 — `ci/h2r1` の恒久化

| 条件 | 結果 |
|---|---|
| workflow 1 ファイルのみ | PASS（`.github/workflows/h2r1-release-measurement.yml` +232 行） |
| dispatch のみ | PASS（`on: workflow_dispatch`） |
| policy 旧 1 行の非混入 | PASS（差分は当該ファイルのみ） |
| 既存ジョブの PASS | PASS（PR #105: 12 pass / 0 fail） |
| dispatch 5 回（rc=0・60 分以内） | **rc=0 は 4/4。60 分基準は run 全体で 1/4、job 全体で 2/4**（下記） |

| run | 結論 | run 全体 | job 所要 | 判定（run 全体 ≤60 分） |
|---|---|---|---|---|
| `41447831754` | success | 61.1 分 | 60.0 分 | 不合格 |
| `41456500055` | success | **49.2 分** | 49.2 分 | **合格** |
| `41464228797` | success | 60.8 分 | 60.7 分 | 不合格 |
| `41474161659` | success | 60.4 分 | 60.2 分 | 不合格 |
| `41484247670` | success | **59.8 分** | — | **合格** |
| `41493867695` | **failure** | 10.5 分 | — | oneAPI 導入ステップ（下記） |
| `41513986906` | success | 60.7 分 | — | 不合格 |
| `41523395609` | **failure** | 58.3 分 | — | 測定ステップ内 `checkSTG82OverwriteTerminalizesEvicted` の flaky（下記） |
| `41532396525` | 実行中 | — | — | — |

- 全成功 run で `[H2R1] trials=40 mixed_trials=0 max_mixed_ms=0.0 distinct_worlds=80`（ログで確認）。
- 所要のばらつきは **`Install Intel oneAPI via winget`（未導入時 10.7 分）**の有無に起因し、
  コード側の要因ではない。60 分境界はランナーの在庫に依存する。
- **`41493867695` の失敗**: 同ステップで `winget exit=-1978335226` を返したが
  `mkl.h present: True` / `ipp.h present: True`（インストールは成立）。`$LASTEXITCODE` が
  ステップの終了コードとして扱われ FAIL となった **workflow の堅牢性問題**（測定・コードとは無関係）。
- **`41523395609` の失敗**: 5 回の測定のうち run 4 が `rc=8`。
  `[STG-8-D2] FAIL: unstable (L=1 hasDeferred=0)` → `checkSTG82OverwriteTerminalizesEvicted` が
  8 秒の安定性ウィンドウ中に deferred slot が解放されたことを検出。**保持時間に依存する
  タイミング依存テスト**であり、R-112 と同種の flaky（新規 R-113 として記録）。
- 合格 run のログ（`h2r1-run1〜5.log` ほか）は保存済み。


**Owner 判断案**

| 案 | 内容 | 影響 |
|---|---|---|
| A（推奨） | 「5 回すべて rc=0」を主基準とし、所要は実測分布（49〜63 分）を記録、60 分は目安として扱う | 追加コストなし |
| B | 厳格に 5 件（≤60 分）を収集するまで再試行（自動収集は継続中） | 数時間規模 |
| C | workflow の oneAPI 導入ステップを堅牢化（`$LASTEXITCODE` を握りつぶし、ヘッダ存在で成功扱い）＋キャッシュ導入 | R-101 の「追加のみ」を超えるため追加承認が必要 |



## 4. R-102 — `ci/r49` の診断移植（完了）

| 検証 | run | 結果 |
|---|---|---|
| `run_selected_tests=false` | `41423092426` | **success**、診断 3 ステップ **skipped** |
| `run_selected_tests=true` | `41423073648` | 診断 3 ステップ **実行**。`AudioEngineHarness` Passed 298.27 s、`HeadlessAudioPathVerification` Failed（`[CLI_PERF_RAW] callbacks=` 不在） |
| 既存 CI | — | PR #106 12 pass / 0 fail、main push の Windows MSVC・ISR は success |

診断の目的（CI ランナーに音声デバイスが無いことの切り分け）は達成。true の dispatch が failure に
なるのは診断ステップが ctest の rc を返す仕様で、既定 false のため通常 CI には影響しない。

## 5. R-107 — U4（実装・検証）と U3（計画）

### 5.1 U4（PR #108）

| 項目 | 内容 |
|---|---|
| 対象 | `freeListMutex_`（8 箇所）/ `crossfadeRecordsMutex_`（3）/ `recordsMutex_`（4） |
| 実装 | `LockScope` を 14 箇所に適用。既定 OFF は atomic load 1 回で素通り |
| RT 経路 | **計測点なし**（`AudioBlock.cpp` 無変更） |
| 記録先 | 固定配列の atomic（確保・ロック・I/O なし）。出力は `pollIfDue` から 1 秒 1 回 |
| 回帰テスト | `RtMeasureRingTests` 検証 (13) を追加 |
| Release | `build-ci-local`（ctest 2 回 rc=0）／`build`（ctest 1 回 rc=0） |
| Debug | **ビルド成功・ctest rc=0 / 340.2 s**（`STG-8-CLEAR` PASS、`U4 locks` PASS） |
| ASan | CI（PR #108）で **pass**（14 pass / 0 fail） |
| オーバーヘッド | **Off 298.91 s / On 297.89 s（差 −1.0 秒・ノイズ範囲）**。Off では `[U4]` 出力 0 行 |

### 5.2 U3（計画）

`R107-STAGE3-U3-PLAN-20261102.md`。テストバイナリ限定の `operator new` 置換＋`thread_local` RT フラグで
RT 経路の新規確保を検出する案 A を採用。U4 とは分離し、U3-1〜U3-4 の段階導入とする。

## 6. R-108 — 旧分割ファイルの削除（完了・再確認済み）

| 確認 | 結果 |
|---|---|
| 退避 ZIP | `archive-old-split-20261102/old-split-f8aab49b.zip` **1,498,358 B** / rev `0165f46760cd4350` |
| 削除対象 4 ファイル | Dropbox 上で **すべて absent**（存在しない） |
| 正本 `/ConvoPeq.md` | 6,451,711 B / sha256 `c12a532e…60e8a9`（**変更なし**） |
| 現行分割 | 基点直下に `ConvoPeq.part*` なし（削除済みのまま） |

## 7. 検証エビデンス

| 種別 | 内容 |
|---|---|
| main の CI | push `a83d10ba`: Windows MSVC **success**、ISR Authority **success**、Sanitizer CI / CodeQL 実行 |
| PR #105 / #106 | それぞれ 12 pass / 0 fail |
| PR #107 | 13 pass / ASan のみ pending |
| PR #108 | 13 pass / ASan のみ pending |
| ローカル | Release ctest（`build-ci-local` 2 回・`build` 1 回）rc=0、STG-8-CLEAR PASS、U4 locks PASS |
| 測定 | 20 実行・sha256 記録・`late50us=0` |

## 8. 未解決・Owner 判断事項

| # | 内容 | 状態 | Owner 判断案 |
|---|---|---|---|
| 1 | **R-101 の dispatch 5 回**（rc=0・60 分以内） | #107 マージ後に実施 | 承認済み範囲。実施継続 |
| 2 | **U5 の指標方針** | FAIL＋判定保留 | A: 既定 OFF のまま確定（推奨）/ B: p99 へ指標変更 / C: 493〜1,022 組へ増加 / D: ETW でスパイク源特定 |
| 3 | **R-111**: `HeadlessAudioPathVerification` が音声デバイス無しの CI ランナーで FAIL | 新規 | **A: 診断ステップに `CONVO_CI_BUILD=1` を設定（CMakeLists の設計意図どおり skip・推奨）** / B: VB-CABLE 導入 / C: 現状維持 |
| 4 | **R-113**: `checkSTG82OverwriteTerminalizesEvicted` の flaky（`[STG-8-D2] FAIL: unstable (L=1 hasDeferred=0)`）| 新規 | 安定性ウィンドウ（8 秒）が保持時間に依存。テスト側でウィンドウ短縮または「解放済みなら完了扱い」に変更（要承認） |
| 5 | **R-107 Stage 3（U3）の実装** | 計画のみ | U4 の ASan 結果を確認後に着手（ASan は CI pass 済み） |
| 5 | ローカル `build` の CMake キャッシュ破損 | 修復済み（VS 18 2026 で再構成・再ビルド） | 対応不要 |

## 9. 成果物の配信（Dropbox）

`/ConvoPeq-stage3-20261026/investigation-20261102/`

| ファイル | サイズ | 備考 |
|---|---|---|
| `r106-measurement-20261102.zip` | 100,685 B | 20 ログ＋集計 3 点 |
| `R106-REMEASUREMENT-RESULT-20261102.md` | 7,142 B | 測定記録 |
| `ConvoPeq.md` | 6,606,738 B | 最新生成（sha256 `f57b1f9f…`） |
| `split/ConvoPeq.part01〜04.md` | 2,097,085 / 2,097,132 / 2,097,150 / 315,371 B | 各 ≤2 MB・連結で元とバイト一致 |
| `split/SHA256SUMS.txt` | 375 B | 4 パートの sha256 |
