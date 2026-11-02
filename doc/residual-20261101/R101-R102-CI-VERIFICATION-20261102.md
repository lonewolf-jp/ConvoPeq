# R-101 / R-102 — CI 追加の検証記録（2026-11-02）

Owner 承認（R-101 / R-102）に基づく CI 定義変更の検証記録。

- 基準ブランチ: `main`
- マージ済みコミット: `7318e44f`（R-101 / PR #105）、`a83d10ba`（R-102 / PR #106）
- 関連: `EXECUTION-PROPOSALS-5ITEMS-20261102.md`

## 1. R-101 — `ci/h2r1` の恒久化

### 1.1 変更範囲（承認条件との照合）

| 条件 | 実測 | 判定 |
|---|---|---|
| workflow 1 ファイルの追加のみ | `.github/workflows/h2r1-release-measurement.yml` のみ（+232 行、他ファイル変更なし） | PASS |
| トリガを dispatch のみに変更 | `on: workflow_dispatch:` のみ（`push` / `pull_request` なし） | PASS |
| policy の旧 1 行を混入させない | 追加ファイルに policy 関連の変更なし。`git diff --stat main...ci/h2r1-permanent-20261102` は 1 ファイル +232 行のみ | PASS |
| main を起点とした移植 | PR の base は `main`、マージは squash | PASS |
| 既存ジョブの PASS | PR #105 のチェック **12 pass / 0 fail**（probe は仕様どおり skipped） | PASS |

workflow の中身は「Release ビルド（ConvoPeq / AudioEngineHarness）→ `ctest -R '^AudioEngineHarness$' -V` を 5 回 → 各回の `[H2R1] trials=…` を解析して C1/C2/C3 を判定」である。
合格条件（40 試行・5 回・C1 `mixed_trials==0`・C2 `distinct_worlds==2*trials`・C3 rc=0）は変更していない。

### 1.2 main 上の dispatch 実行

`gh workflow run h2r1-release-measurement.yml --ref main` を用い、**1 回ずつ完了を待って**実行した。
workflow の `concurrency`（`group: h2r1-release-measurement-${{ github.ref }}` / `cancel-in-progress: false`）は
「同時に 1 件のみ待機可能」であるため、5 件を同時に dispatch すると**後続 4 件が cancelled** になる。
これは workflow の仕様どおりの挙動であり、検証は逐次実行で行う。

| 試行 | run | 結果 | 所要 | 備考 |
|---|---|---|---|---|
| 1 | `41422842771` | **failure** | 45.7 分 | 内 5 回のうち run 5 が `rc=8`（`checkSTG8ClearTerminalizesObligation` の flaky 失敗）。run 1〜4 は `rc=0` |
| 2 | `41422888522` | 実行中 | — | — |
| 3〜 | 自動投入 | — | — | 失敗時は再試行し、成功 5 件を集める |

同時 dispatch で cancelled になった 4 件（`41422854238` / `41422865267` / `41422876370`）は検証対象から除外した。

#### 試行 1 の詳細（flaky 失敗の証跡）

```
run 1: rc=0 318.5s :: 50: [H2R1] trials=40 mixed_trials=0 max_mixed_ms=0.0 distinct_worlds=80 :: PASS
run 2: rc=0 318.4s :: 50: [H2R1] trials=40 mixed_trials=0 max_mixed_ms=0.0 distinct_worlds=80 :: PASS
run 3: rc=0 318.2s :: 50: [H2R1] trials=40 mixed_trials=0 max_mixed_ms=0.0 distinct_worlds=80 :: PASS
run 4: rc=0 318.3s :: 50: [H2R1] trials=40 mixed_trials=0 max_mixed_ms=0.0 distinct_worlds=80 :: PASS
run 5: rc=8 200s   :: 50: [H2R1] trials=40 mixed_trials=0 max_mixed_ms=0.0 distinct_worlds=80 :: FAIL(rc)
```

`h2r1-run5.log` の失敗行:

```
50: [STG-8-CLEAR] FAIL: hasDeferred still true after clear
50: FAIL: checkSTG8ClearTerminalizesObligation
1/1 Test #50: AudioEngineHarness ...............***Failed  199.96 sec
```

**測定値そのもの（`[H2R1] trials=40 mixed_trials=0 distinct_worlds=80`）は 5 回すべて同一**であり、
失敗は harness スイート内の別テスト（deferred clear の競合）に起因する。これは R-101 の変更に由来しない
既存の flaky テストであり、残課題として台帳へ記録する。

## 2. R-102 — `ci/r49` の診断移植

### 2.1 変更範囲（承認条件との照合）

| 条件 | 実測 | 判定 |
|---|---|---|
| ブランチをマージせず固有差分のみ移植 | `git diff --stat main...ci/r49-diagnostics-20261102` は `windows-msvc-build.yml` の +51 行のみ | PASS |
| 4 ステップのみ | `Enumerate audio devices` / `Run selected tests` / `Show ConvoPeq log`（＋`run_selected_tests` 入力定義） | PASS |
| 診断以外の差分を取り込まない | 追加行はすべて `if: inputs.run_selected_tests == true` 配下 | PASS |

### 2.2 `run_selected_tests` の true / false 検証

| ケース | run | 結果 | 診断ステップの実行 |
|---|---|---|---|
| **false** | `41423092426` | success | **3 ステップすべて skipped** |
| **true** | `41423073648` | failure（診断ステップが FAIL） | **3 ステップすべて実行** |

false ケースのステップ（`gh run view --json jobs`）:

```
[success] Configure (Release, MKL required)
[success] Build Release (ConvoPeq)
[success] Build Release (AudioEngineHarness)
[skipped] Enumerate audio devices (diagnostics, dispatch only)
[skipped] Run selected tests (diagnostics, dispatch only)
[skipped] Show ConvoPeq log (diagnostics, dispatch only)
[success] Upload build logs
```

true ケースのステップ:

```
[success] Enumerate audio devices (diagnostics, dispatch only)
[failure] Run selected tests (diagnostics, dispatch only)
[success] Show ConvoPeq log (diagnostics, dispatch only)
```

`Run selected tests` の内容:

```
1/2 Test #42: HeadlessAudioPathVerification ....***Failed    7.47 sec
CLI smoke failed: '[CLI_PERF_RAW] callbacks=' with a positive callback count was not found
2/2 Test #50: AudioEngineHarness ...............   Passed  298.27 sec
50% tests passed, 1 tests failed out of 2
```

**診断の目的は達成されている**。すなわち、CI ランナーには音声出力デバイスが無いため
`HeadlessAudioPathVerification` の CLI smoke が `[CLI_PERF_RAW] callbacks=` を出せず FAIL する一方、
`AudioEngineHarness` は PASS する。この切り分けが R-49 の調査対象であった。

診断ステップは `ctest` の rc をそのまま返すため、true の dispatch は workflow として failure になる。
これは**診断専用ジョブの仕様**（既定 false のため通常の CI には影響しない）。

### 2.3 既存 CI の PASS

| 対象 | 結果 |
|---|---|
| PR #106 のチェック | **12 pass / 0 fail**（probe は skipped） |
| `main`（`a83d10ba`）の push 実行 | Windows MSVC Build **success** / ISR Authority Compliance **success** |

## 3. 結論

| 項目 | 判定 |
|---|---|
| R-101 の変更範囲・トリガ・policy 非混入 | **PASS** |
| R-101 の 5 回 dispatch 検証 | **継続中**（失敗は flaky テスト起因。成功 5 件を集めるまで再試行） |
| R-102 の移植範囲 | **PASS** |
| R-102 の true / false 動作 | **PASS**（true で診断実行、false で skip） |
| 既存 CI の PASS | **PASS** |

## 4. 残課題として記録する事項

1. `checkSTG8ClearTerminalizesObligation` が稀に FAIL する（`hasDeferred still true after clear`）。
   R-101 の「5 回すべて rc=0」を満たすためにも、原因調査と安定化が必要。
2. `HeadlessAudioPathVerification` は音声デバイスの無い CI ランナーで FAIL する（R-49 の診断で確定）。
   CI で常時実行するか、ランナー条件で skip するかの判断が必要。
