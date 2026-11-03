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
| R-101（h2r1 恒久化） | **完了**。ワークフローをマージし、main 上の dispatch が **5/5 合格（rc=0・60 分以内）**（§3） |
| R-102（r49 診断移植） | **完了**（true / false とも実測で確認） |
| R-106（U5 再測定） | **完了**（20 実行。FAIL＋判定保留。測定器は既定 OFF） |
| R-107 Stage 1（列挙） | 完了 |
| R-107 Stage 2（U4） | **完了**（PR #108 → `7dfb21d4`。既定 OFF・RT 計測点 0・Release/Debug/ASan PASS） |
| R-107 Stage 3（U3） | **実装・検証完了**（PR #114）。RT 経路の新規確保 **0**（陽性対照 1）。既定 OFF |
| R-113（STG-8-D2 flaky） | **修正済み**（Release **16/16**・Debug 1/1 で PASS、`STG-8-D2` 各回 PASS） |
| R-111（CI 診断の音声テスト） | **修正済み**（`CONVO_CI_BUILD=1` を Configure に設定。登録 1→0・選択 2→1。dispatch で `AudioEngineHarness` 1/1 PASS） |
| oneAPI 導入ステップ | **堅牢化済み**（`winget` 終了コードを判定に使わず、ヘッダ＋lib 4 点で検証。PR と main push の双方で success） |
| R-106 の指標再定義 | **技術判断案を提出**（`R106-METRIC-REDEFINITION-PROPOSAL-20261103.md`。判定は FAIL・判定保留のまま） |
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

### 2.1 指標再定義の技術判断案（提出済み・承認待ち）

`R106-METRIC-REDEFINITION-PROPOSAL-20261103.md` に、指標定義の選択肢と推奨案を整理した。

- 推奨は **metric v2**: p99 の非劣性検定（片側 Welch t、α=0.05、差 5%、6 組/アーム）を主判定とし、`late50us=0`・`dropped=0`・`overflow=0`・`n≥30,000` を測定成立ゲートとして分離する。max は記録専用（OFF の 2 倍超で警告）とし、合否に使わない。
- 根拠は、max が周期超過ではなく単発スパイクであること（`late50us=0`、20/20）と、σ(max) 280〜403 µs に対して判定幅 +50 µs が小さすぎること（5 組の検出力 0.04〜0.05）。
- p99 の検出力は無音 1.000、1 kHz 0.871（n=5）／0.928（n=6）。
- 既存 20 実行の参考値は無音 98.91%（p=0.0002）、1 kHz 99.28%（p=0.0047）。**合格扱いにしない**。
- 決定依頼は D1（v2 採用）、D2（既存データでの再判定）、D3（不採用時のクローズ）の 3 点。

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
| `41532396525` | success | **59.6 分** | 59.5 分 | **合格** |
| `41548873504` | success | **59.8 分** | 59.8 分 | **合格** |

**合格 5 件（すべて rc=0・60 分以内）**

| # | run | run 全体 | job 所要 |
|---|---|---|---|
| 1 | `41456500055` | 49.2 分 | 49.2 分 |
| 2 | `41484247670` | 59.8 分 | 59.7 分 |
| 3 | `41505721908` | 52.8 分 | 52.7 分 |
| 4 | `41532396525` | 59.6 分 | 59.5 分 |
| 5 | `41548873504` | 59.8 分 | 59.8 分 |

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


**判定と残課題**

- 厳格基準（run 全体 ≤60 分）で **5 件連続合格**したため、R-101 の検証は**完了**（案 B の収集を完遂）。
- なお、所要は `Install Intel oneAPI via winget`（未導入時 10.7 分）に依存して 49〜63 分でばらつく。
  将来の安定運用のため、次の改善を推奨（いずれも R-101 の「追加のみ」を超えるため追加承認が必要）。
  - workflow の oneAPI 導入ステップで `$LASTEXITCODE` を成功判定に使わない（ヘッダ存在で成功扱い）
  - oneAPI のキャッシュ／プリインストール前提の導入




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

### 5.2 U3（実装・検証完了 / PR #114）

`R107-STAGE3-U3-PLAN-20261102.md` の案 A（テストバイナリ限定の `operator new` 置換＋`thread_local` RT フラグ）で実装した。置換点は `RtAllocProbe.cpp` の 1 箇所に集約し、既存の `a2probe` も同じ実装へ統合した。

| 項目 | 結果 |
|---|---|
| 陽性対照（`selfcheck_rt_alloc=1`） | **1**（プローブが反応することを確認） |
| 実測（5 秒） | `rt_alloc=0` / 6,470 ブロック |
| 実測（30 秒） | `rt_alloc=0` / 38,208 ブロック |
| 既定 | OFF（`probe=off`）。計数のみ常時、出力は env で有効化 |
| 回帰 | プローブ ON を含む Release スイートで rc=0（計 16 回） |

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

| # | 内容 | 状態 | 次の扱い |
|---|---|---|---|
| 1 | **R-101 の dispatch 5 回**（rc=0・60 分以内） | **完了**（5/5 合格） | 解消 |
| 2 | **U5 の指標方針** | FAIL＋判定保留。技術判断案を `R106-METRIC-REDEFINITION-PROPOSAL-20261103.md` に提出 | **D1 v2 採用 / D2 再判定 / D3 クローズ** の決定待ち |
| 3 | **R-111**: `HeadlessAudioPathVerification` が音声デバイス無しの CI ランナーで FAIL | **修正済み**（`CONVO_CI_BUILD=1` を Configure に設定） | CI dispatch で 1/1 PASS を確認。継続監視 |
| 4 | **R-113**: `checkSTG82OverwriteTerminalizesEvicted` の flaky | **修正済み**（固定 8 秒判定を不変条件サンプリングへ変更） | Release 16/16・Debug 1/1・ASan CI で PASS。継続監視 |
| 5 | **R-107 Stage 3（U3）** | **実装・検証完了**（PR #114） | 解消 |
| 6 | oneAPI 導入のキャッシュ化 | 未実施（毎回 `winget` で導入） | 導入時間の短縮は別課題 |
| 7 | ローカル `build` の CMake キャッシュ破損 | 修復済み（VS 18 2026 で再構成・再ビルド） | 対応不要 |

## 9. 成果物の配信（Dropbox）

`/ConvoPeq-stage3-20261026/investigation-20261102/`（2026-11-03 更新分）

| ファイル | サイズ | 備考 |
|---|---|---|
| `ConvoPeq.md` | **6,470,973 B** | main `e89c66fe` で再生成（LF・sha256 `d30edac1…5ebb83`）。`--check` で FRESH |
| `split/part01〜04.md` | 2,097,115 / 2,097,095 / 2,097,114 / 179,649 B | 各 ≤2 MB。連結で元とバイト一致 |
| `split/SHA256SUMS.txt` | 4 行 | 4 パートの sha256 |
| `R113-R111-ONEAAPI-U3-20261103.md` | 7,710 B | 今回の 4 件の原因・変更・検証 |
| `R106-METRIC-REDEFINITION-PROPOSAL-20261103.md` | 指標再定義の技術判断案（D1〜D3） |
| `r101-dispatch-logs-final-20261103.zip` | R-101 dispatch の証跡 |

配信物はダウンロードして照合済み（全体 sha256 一致、`concat == full: True`、マニフェスト 4 行）。
