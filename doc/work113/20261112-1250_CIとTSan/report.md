# push 後の確認：Remote CI の状態と TSan の実行可能性

作業日時: 2026-11-12 12:40〜12:50
対象: branch `ci-fix/173e6995`（commit `b237f233f5ddaab9ef471bf2afe3e036ceb089b5`）

## 結論

1. 対象ブランチの Remote CI は 1 件も走っていない。workflow のトリガ条件が `push` では `main` と `release/**` のみで、feature branch への push に反応しないため。CI を回すには PR 作成（pull_request イベント）か、`workflow_dispatch` 対応 workflow の手動実行が必要で、どちらも未承認のため行っていない。
2. TSan は WSL（Ubuntu 26.04 / clang 21.1.8）で configure から実行まで試した。3 ターゲットは TSan 計装で 0 警告、2 ターゲットは環境要因と既存の Clang 非互換で実行できなかった。

## 1. Remote CI

`gh run list --branch ci-fix/173e6995` の結果は 0 件（2026-11-12 12:46 時点）。

workflow のトリガは次のとおり。

| workflow | トリガ |
| --- | --- |
| audioengine-lint | pull_request、push（main） |
| list-compliance | pull_request、push（main） |
| isr-verification | pull_request（main）、schedule、workflow_dispatch |
| windows-msvc-build | pull_request（main）、push（main）、workflow_dispatch |
| sanitizer-ci | push（main / release/**）、pull_request（main） |

feature branch への push はどの `push` 条件にも一致しない。したがって「push したので CI が走る」状態ではない。

CI を回す選択肢（いずれも未承認のため未実施）:

- PR を作成する（`gh pr create --base main --head ci-fix/173e6995`）。pull_request イベントで lint・list-compliance・ISR Verification・Windows MSVC Build・Sanitizer が動く。
- `workflow_dispatch` を使う（`gh workflow run windows-msvc-build.yml --ref ci-fix/173e6995`、`gh workflow run isr-verification.yml --ref ci-fix/173e6995 -f verificationTier=standard`）。この 2 本だけ対応。lint・list-compliance・sanitizer は PR でしか動かない。

## 2. TSan（WSL での実測）

環境: Ubuntu 26.04（WSL2）、clang 21.1.8、ninja、cmake。`-DENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=Debug -DCONVOPEQ_ENABLE_ISR_TESTS=ON` で configure 成功（86.7 秒）。

| ターゲット | ビルド | 実行 | TSan 警告 |
| --- | --- | --- | --- |
| MpscBoundedRingTests | OK | ALL TESTS PASSED（10/10） | 0 |
| ScopedEpochReaderStructuralTests | OK | PASS（reader-safety contract） | 0 |
| ISRRuntimeIdentityTests | OK | 終了コード 0（出力なし） | 0 |
| ScopedEpochReaderTests | 失敗 | — | — |
| ISRSemanticValidationTests | 失敗 | — | — |

失敗の理由:

- `ScopedEpochReaderTests`: Clang が `ISRRetireLatency.h:1135` の `Classification cls = Classification{}` と `EpochDomain.h:767` の `static inline HoldWitness holdWitness_[kMaxReaders]{}` を拒否する（nested class の default member initializer を外側クラスの定義内で使えない、という Clang の制約）。**この 2 か所は基準 commit `173e6995` にも同じ形で存在する**（`git show 173e6995:...` で確認）。今回の変更で入ったものではない。
- `ISRSemanticValidationTests`: `ConvoPeq` ターゲットに依存し、Linux に MKL が無いため `mkl.h` で停止する。

CI の TSan ジョブ（`Debug TSan (Clang, best-effort)`）は `continue-on-error: true` で、configure が通らない場合や MKL/IPP が無い場合はスキップする作りになっている。この環境で実際に走った 3 ターゲットは警告 0 で、変更した `ScopedEpochReader`／ISR 周辺の並行性に TSan 由来の問題は出ていない。

TSan を本格的に全ターゲットへ広げる場合は、(a) Linux へ MKL を入れる、(b) 上記の Clang 非互換を直す、の 2 つが前提になる。どちらも今回の承認範囲外。

## 3. 変更していないもの

merge、release、tag 作成、PR 作成、`workflow_dispatch` による CI 実行、`main` への push は行っていない。main 作業ツリーは HEAD `173e6995`・変更 34 件のまま。

## 証跡

- `ci_status.txt`: ブランチの run 一覧（0 件）と workflow トリガの抜粋
- `tsan_configure.log`: WSL での TSan configure
- `tsan_build_app.log`、`tsan_build_ScopedEpochReaderTests.log`、`tsan_build_ISRSemanticValidationTests.log`: ビルド失敗の内容
- `tsan_run_MpscBoundedRingTests.log`、`tsan_run_ScopedEpochReaderStructuralTests.log`、`tsan_run_ISRRuntimeIdentityTests.log`: 実行ログ（警告 0）
- Dropbox: `/ConvoPeq-b237f233-commit/ci-and-tsan-20261112/`
