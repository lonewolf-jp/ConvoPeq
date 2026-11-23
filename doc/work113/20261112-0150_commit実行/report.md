# domain tag 実装の commit 実行

作業日時: 2026-11-12 01:35〜01:55
対象: domain tag 実装と、その検証・記録の最終化
commit: `173e69957b046290f7077797782f031ed8633fd0`（親 `c94d3767`）

## 結論

承認条件 1〜6 をすべて満たす範囲で commit した。対象は 49 ファイル（変更 43・新規 6）、差分は +5,455 / −215。`ConvoPeq.md` と資料再生成の成果物、`.github` の CI スクリプト、別系統の未コミット作業は含めていない。staged 状態を再現して Release ビルドと ctest 57/57 合格を確認してから commit した。

## 承認条件への対応

| 条件 | 対応 | 根拠 |
| --- | --- | --- |
| 1. HEAD が `c94d3767` から変化していない | commit 直前に `c94d3767` を確認。commit 後は `173e6995` | `git rev-parse` |
| 2. 対象ファイル・差分・新規追加の一覧確認 | 49 ファイルの一覧と差分を staged 状態で確認 | `git diff --cached --name-status` |
| 3. `ConvoPeq.md` と関連資料の扱いを明示 | すべて commit に含めない（下記） | `git show --name-only` に該当なし |
| 4. 未追跡 161 件を一括追加しない | 未追跡からは 6 ファイルだけを個別に追加 | 追加は 6 件（下記） |
| 5. 設計差分 v2.9 が不変 | 53,979 B・SHA-256 `C3AE5099…8427` のまま | `sha256sum` |
| 6. 承認範囲外の変更が混入しない | 除外 32 ファイルを staged に含めず、commit 後に作業ツリーへ残存することを確認 | 変更 33 件が未ステージで残る |

## 再検証（commit 後）

commit の内容と承認条件を、commit 後の状態からもう一度確認した。

- 条件 1: reflog で HEAD は 2026-11-09 の fast-forward 以降 `c94d3767` のままで、`173e6995` になったのは 2026-11-12 01:48:35 の commit だけ。commit の親も `c94d3767`。stash 操作による `reset: moving to HEAD` は HEAD を動かしていない。
- 条件 2: `git show --name-status HEAD` は 49 件（変更 43・新規 6）。内訳は上記「commit した範囲」のとおり。
- 条件 3: commit に `doc/`、`ConvoPeq.md`、`.github/` のパスは 1 件も含まれない。
- 条件 4: 新規追加は 6 件で、すべて `src/` 配下。未追跡は 161 件から 6 件を除いた 155 件になり、現在は本報告の 2 ファイルを加えて 157 件。
- 条件 5: `design_diff.md` は 53,979 B、SHA-256 `c3ae5099…56848427`。
- 条件 6: 除外した 32 ファイルは commit に 1 件も入っておらず、32 件すべてが変更として作業ツリーに残っている。commit 済みファイルで未ステージの変更が残るのは `CMakeLists.txt`（`DomainTagTests` 以外の定義）だけで、ほかに `AGENTS.md` も commit 後の記述修正として残る。
- 検証した内容と commit の同一性: staging 直後に控えた index の tree は `a1ec450a3ee95b7ba607e170b55ee64da8b726d2` で、commit の tree と一致する。Release ビルドと ctest 57/57 を実行した内容が、そのまま commit の内容である。

## commit した範囲（49 ファイル）

domain tag 本体と、その前提になっている退役遅延の測定・分類、および整合に必要なテスト・ビルド定義。

- 新規 6: `RetireDomain.h`、`ISRRetireLatency.h`、`RetireLatencyTeardownReporter.h`、`WorldLifetimeRecorder.h`、`ReaderMeasureRegistry.h`、`DomainTagTests.cpp`
- ドメイン刻印と集計: `DeferredDeletionQueue.h`、`EpochDomain.h`、`IRetireProvider.h`、`ISRRetireRouter.h/.cpp`、`RetireQuarantineStore.h`、`DeletionQueue.cpp`、`SnapshotRetireManager.h`、`ISRHB.h/.cpp`、`ISREvidenceExporter.h/.cpp`、`ISRDebugRuntime.h/.cpp`
- ドライバとゲート: `EQProcessor.h`／`.Core.cpp`／`.Processing.cpp`／`.Coefficients.cpp`、`AudioEngine.h`／`.Timer.cpp`／`.CtorDtor.cpp`／`.EQResponse.cpp`／`.RebuildDispatch.cpp`、`RCUReader.h`、`ScopedEpochReader.h`、`IReaderEpochProvider.h`、`SnapshotCoordinator.h/.cpp`、`ISRRuntimePublicationCoordinator.cpp`、`ConvolverProcessor.h`
- テスト 11: `D8_2_B_2_Tests.cpp`、`STG11EQRetireTests.cpp`、`DeferredDeletionQueueReclaimTests.cpp`、`V11ReaderReacquireTests.cpp`、`RetireGraceSemanticsTests.cpp`、`ScopedEpochReaderTests.cpp`、`STG10ReaderQuarantineTests.cpp`、`STG11D2SnapshotRetireTests.cpp`、`StuckReaderFallbackDrainTests.cpp`、`TerminalTelemetryContractTests.cpp`、`invariant_INV3_INV5.cpp`
- 記録・ビルド定義: `AGENTS.md`、`CMakeLists.txt`（`DomainTagTests` の定義 15 行のみ）

`EpochDomain` のコンストラクタが必須引数になったため、構築箇所を持つテストは範囲に含めざるを得ない。`D8_2_B_2_Tests.cpp` と `ConvolverProcessor.h`、`CMakeLists.txt` は、この整合のための最小限を含む。

## commit に含めなかったもの

### ConvoPeq.md と資料再生成の成果物（条件 3 の回答）

含めない。理由は 2 つある。

- 再生成した `ConvoPeq.md` は作業ツリー全体の写しで、今回 commit しない B 系の変更コードも含む。commit したソースと本文書が食い違う状態になる。
- リポジトリには `doc/work113/` の日時フォルダを 1 件も追跡していない（443 件の追跡ファイルはすべて単体のレポート）。分割版や `MANIFEST.md` を commit する慣行がない。

再生成した `ConvoPeq.md`、分割 4 ファイル、`report.md`、`MANIFEST.md` は作業ツリーと Dropbox に残した。`ConvoPeq.md` は変更 33 件の 1 つとして未ステージのまま残る。

### 除外した 32 ファイル

- `.github/scripts/isr-*.ps1` 6 件（実測証跡モードの CI 側）
- `CacheManager.cpp/.h`、`IRConverter.cpp/.h`、`NoiseShaperLearner.cpp/.h`、`MKLNonUniformConvolver.cpp/.h`、`ConvolverProcessor.LoadPipeline.cpp`／`LoaderThread.cpp`／`ResampleAndFallback.cpp`／`Runtime.cpp`（IR ロードとキャッシュの監査対応 B-01〜B-10）
- `MainWindow.cpp/.h`、`MainApplication.cpp/.h`、`DeviceSettings.cpp`、`ProgressiveUpgradeThread.cpp`、`ConvolverControlPanel.cpp`（UI・起動経路の B-05／B-13 系）
- `AudioEngine.Commit.cpp`（B-11 証跡モード）、`AudioEngine.Learning.cpp`、`AudioEngine.Reader.cpp`、`IRLoadAdmissionTests.cpp`、`TailLayerIntegrityTests.cpp`、`PublishPipelineIntegrationTests.cpp`
- `ConvoPeq.md`

未追跡 161 件のうち commit したのは 6 件で、残り 155 件（dated な作業記録フォルダ、別系統の新規ソース、分割版）は未追跡のまま残した。

## 検証

commit 予定の内容だけを作業ツリーに残すため `git stash push --keep-index` を使った。この状態でビルドとテストを実行し、終わったあとに stash を戻した。

- Debug: 714 ターゲットのビルド成功（`ConvoPeq.exe`、`DomainTagTests.exe` を含む）
- Release: 全ターゲットのビルド成功
- Release ctest: 57/57 合格（329 秒、`AudioEngineHarness` 302 秒を含む）
- 検証後に stash を戻し、commit 後の作業ツリーに除外分が残ることを確認

### 検証中に見つかった既存の不具合

Debug 構成では 2 件が落ちる。いずれも commit 前の作業ツリー全体でも同じ結果になり、今回の範囲分けに由来しない。

- `D8_2_B_2_Tests`: 終了コード `0xC00000FD`（スタック超過）。`EpochDomain` をスタック上に構築する既存のテスト構造による。Release では合格。
- `TailLayerIntegrityTests`: `RES-F` の CPU・メモリ予算が Debug の速度で超過。Release では合格。

Debug の 2 件は未修正。Release の検証は 57/57 で通っている。

## commit 後の状態

- HEAD: `173e6995`（親 `c94d3767`）。branch は `main` で、`origin/main` より 1 commit 進んでいる。
- 作業ツリー: 変更 34 件・未追跡 157 件・ステージ 0 件。除外した変更は失われていない。変更 34 件は、commit しなかった 32 ファイル、`CMakeLists.txt` の残り（`DomainTagTests` 以外の定義）、commit 後に `AGENTS.md` の「未コミット」記述を commit 済みへ直した 1 件。未追跡 157 件は、commit 後に残った 155 件に本報告の `report.md` と `commit-173e6995.patch` を加えた数。`ctest_release_57of57.log` は `.gitignore` の `*.log` に該当するため計数に入らない。
- push はしていない。merge と release も未実施。
- 設計差分 `doc/work113/20261111-1512_domain設計/design_diff.md` は 53,979 B・SHA-256 `C3AE5099…8427` で不変。

## 参照

- commit: `173e69957b046290f7077797782f031ed8633fd0`
- `doc/work113/20261112-0150_commit実行/commit-173e6995.patch`（commit の全差分）
- `doc/work113/20261112-0150_commit実行/ctest_release_57of57.log`（Release ctest の実行ログ）
- `doc/work113/20261112-0046_資料再生成/report.md`
- Dropbox: `/ConvoPeq-173e6995-commit/`
