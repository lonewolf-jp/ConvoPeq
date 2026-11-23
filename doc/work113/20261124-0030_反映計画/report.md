# 保護ツリーへの反映計画

作業日時: 2026-11-24 00:30
対象: `C:\VSC_Project\ConvoPeq`（保護ツリー）と main `75e64841`

この文書は読み取り専用の調査と反映計画である。保護ツリーには書き込んでいない（作業中の MCP 設定修正 1 行を除く。§2 に記す）。

## 1. 結論

1. 一括上書きはしない。追跡 43 ファイルのうち 36 は置換して問題ない（うち 6 は main とバイト一致）。残る 7 は、ローカル専用の ISR スクリプト 5 本、`opencode.json` の設定 1 行、`MainWindow.cpp` の ASIO 警告 1 箇所である。
2. 反映で失われるのは、いま除外と決めている系統だけである。probe 計測 81 行、B-07 102 行、B-11 20 行、reader 計測 13 行、B-12 と B-03 のコメント 12 行。加えて ISR スクリプト 5 本と ASIO 警告 12 行。いずれも今回の再保全から復元できる。
3. 反映先の基準は `75e64841` に固定してよい。製品コードは ConvoPeq.md の生成元 `8e523540` と同一で、差はドキュメント 6 ファイルだけだった。

## 2. 保護ツリーの現状

2026-11-24 00:36 に再取得した値は次のとおり。

| 項目 | 値 |
| --- | --- |
| HEAD | `4d476d5c65261fd975de2606f4d8b5c843cd86e1` |
| ブランチ | `main` |
| stash | なし |
| 未コミット | 398 エントリ（変更 43 / 未追跡 355） |
| シグネチャ | `8f226d032b644f25…` |
| 保全ブランチ | `wip/preserve-20261123`（`adfb16f6`、未変更） |

前回の保全時（396 エントリ、シグネチャ `6e346f32`）との差は 2 件だけである。

- `opencode.json` の 1 行（graphify の `protocol`）。2026-11-24 00:23 の MCP 修正で入れた。
- `doc/work113/20261124-0025_MCP復旧/report.md`（新規）。同じ修正の報告書。

残る 42 ファイルの WIP 差分は、前回の `tracked_changes.patch` と一致した（`index` 行の短縮を除いてセクション単位でバイト同一）。未追跡も、減ったものは無い。つまり統合作業のあとで保護ツリーは意図せず変わっていない。

再保全は凍結ブランチとは別に、新しいアーティファクトとして作った。

| ファイル | 内容 |
| --- | --- |
| `state.json` | HEAD・件数・シグネチャ・作成時刻 |
| `manifest.tsv` | 398 エントリの状態・サイズ・SHA-256 |
| `tracked_changes.patch` | 追跡変更 43 ファイルの差分 |
| `untracked_files.zip` | 未追跡 355 ファイル |

zip の中身 355 件は、すべて manifest の SHA-256 と一致した。置き場所は `%TEMP%\opencode\refl20261124\` と Dropbox の `/ConvoPeq-reflection-plan-20261124/` である。

## 3. 対象 main の確定

- main は `75e64841`（PR #271 のマージ `8e523540` と、その後の PR #272）。
- `8e523540..75e64841` の差分は 6 ファイルだけだった。`ConvoPeq.md`、統合報告書、`split/` の 4 本である。製品コードの変更は 0 行。
- したがって「ConvoPeq.md は `8e523540` から生成した」という記述と、最終 main は矛盾しない。報告書 §5 の記述はこの意味で正しく、反映先は `75e64841` で固定してよい。

## 4. 差分の全体像

| 範囲 | 値 |
| --- | --- |
| `4d476d5c..75e64841` | 232 コミット / 219 ファイル / +388,436 −1,846 行 |
| 内訳 | 追加 120 / 変更 99 / 削除 0 / 改名 0 |
| 保護ツリー側 | 追跡変更 43 / 未追跡 355（doc 339、src 13、その他 3） |

削除と改名が 0 なので、反映でパスが消えることはない。危険があるのは「ローカルの変更が main の内容で上書きされる」点だけである。

## 5. 対応表

分類の読み方。

- F0 一致: ローカルの内容が main とバイト一致。置換しても変化しない。
- F1 置換可: main が WIP の追加行をすべて含む。または残差が除外系統・コメントだけ。置換で意図は失われない。
- F2 要判断: 残差の一部が未決の改善。
- F3 ローカル専用: main に無い。反映すると失われる。
- F4 ローカル設定: main に無い設定差分。移植を推奨する。

### 5.1 追跡変更 43 ファイル

| 分類 | パス | 判定 |
| --- | --- | --- |
| F0 | `src/CacheManager.h` | main と同一 |
| F0 | `src/IRConverter.h` | main と同一 |
| F0 | `src/audioengine/AudioEngine.Learning.cpp` | main と同一 |
| F0 | `src/audioengine/ISRRetireLatency.h` | main と同一 |
| F0 | `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp` | main と同一 |
| F0 | `src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp` | main と同一 |
| F1 | `.github/scripts/isr-run-tiered-verification.ps1` | WIP の追加行はすべて main に在る |
| F1 | `AGENTS.md` | WIP の追加行はすべて main に在る |
| F1 | `CMakeLists.txt` | 残差は B-07 のテスト登録 8 行のみ |
| F1 | `ConvoPeq.md` | 生成物。main 版が最終 main 由来 |
| F1 | `src/CacheManager.cpp` | 残差は WIP 版の検証関数。main は C-05 + 2^21+1 で同等以上 |
| F1 | `src/ConvolverControlPanel.cpp` | WIP の追加行はすべて main に在る |
| F1 | `src/DeviceSettings.cpp` | WIP の追加行はすべて main に在る |
| F1 | `src/IRConverter.cpp` | 残差は置換で消えた旧コード。main が B-01 を含む |
| F1 | `src/MKLNonUniformConvolver.cpp` | 残差は B-12 の仕様コメント 4 行。実装は main に在る |
| F1 | `src/MKLNonUniformConvolver.h` | 残差は B-12 の仕様コメント 8 行 |
| F1 | `src/MainApplication.cpp` | 残差は B-11 の実測証跡モード 13 行（見送り） |
| F1 | `src/MainApplication.h` | 残差は B-11 のメンバ 3 行 |
| F1 | `src/MainWindow.h` | WIP の追加行はすべて main に在る |
| F1 | `src/NoiseShaperLearner.cpp` | 残差 85 行は B-07。B-05/B-09 は main に在る |
| F1 | `src/NoiseShaperLearner.h` | 残差 17 行は B-07 の宣言 |
| F1 | `src/ProgressiveUpgradeThread.cpp` | 残差 18 行は probe 計測のみ |
| F1 | `src/audioengine/AudioEngine.Commit.cpp` | 残差 4 行は B-11 の実測証跡 |
| F1 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | 残差 3 行は probe 計測のみ |
| F1 | `src/audioengine/AudioEngine.Reader.cpp` | 残差 13 行は reader 計測（LINT-AE-011 のため見送り） |
| F1 | `src/audioengine/AudioEngine.Threading.cpp` | 残差 4 行は probe 計測のみ |
| F1 | `src/audioengine/AudioEngine.h` | 差分の実体なし |
| F1 | `src/convolver/ConvolverProcessor.LoadPipeline.cpp` | 残差は probe と、main で decltype 化した 1 行 |
| F1 | `src/convolver/ConvolverProcessor.LoaderThread.cpp` | 残差 32 行は probe と、main が別形に整理した 3 行 |
| F1 | `src/convolver/ConvolverProcessor.Runtime.cpp` | 残差は B-03 のコメント。実装は main に在る |
| F1 | `src/convolver/ConvolverProcessor.StateAndUI.cpp` | 残差 7 行は probe 計測のみ |
| F1 | `src/core/EpochDomain.h` | WIP の追加行はすべて main に在る |
| F1 | `src/core/ReaderMeasureRegistry.h` | WIP の追加行はすべて main に在る |
| F1 | `src/core/ScopedEpochReader.h` | WIP の追加行はすべて main に在る |
| F1 | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | WIP の追加行はすべて main に在る |
| F1 | `src/tests/TailLayerIntegrityTests.cpp` | 残差はコメント 2 行。main 版が B-12 を含む |
| F2 | `src/MainWindow.cpp` | 残差 15 行のうち 12 行は ASIO 警告ダイアログ、3 行は B-14（main の C-16 が上位互換） |
| F3 | `.github/scripts/isr-generate-runtime-evidence.ps1` | main に未取込。反映すると失われる |
| F3 | `.github/scripts/isr-run-runtime-evidence.ps1` | main に未取込。反映すると失われる |
| F3 | `.github/scripts/isr-seed-evidence.ps1` | main に未取込。反映すると失われる |
| F3 | `.github/scripts/isr-verify-common.ps1` | main に未取込。反映すると失われる |
| F3 | `.github/scripts/isr-verify-v5.ps1` | main に未取込。反映すると失われる |
| F4 | `opencode.json` | MCP の protocol 1 行。main に無い（今回の修正） |

内訳は F0 が 6、F1 が 30、F2 が 1、F3 が 5、F4 が 1。

### 5.2 未追跡（doc 以外）16 ファイル

| 分類 | パス | 判定 |
| --- | --- | --- |
| U0 | `src/InputInspection.cpp` | main と同一 |
| U0 | `src/InputInspection.h` | main と同一 |
| U0 | `src/convolver/IRTrailingSilence.h` | main と同一 |
| U0 | `src/learner/LearnedStateMigration.cpp` | main と同一 |
| U0 | `src/learner/LearnedStateMigration.h` | main と同一 |
| U0 | `src/tests/AudioEngineHarness/NoiseShaperStateRaceTests.cpp` | main と同一 |
| U0 | `src/tests/AudioEngineHarness/RuntimeParameterGuardTests.cpp` | main と同一 |
| U0 | `src/tests/InputInspectionTests.cpp` | main と同一 |
| U0 | `src/tests/LearnedStateMigrationTests.cpp` | main と同一 |
| U1 | `src/tests/CacheManagerHeaderValidationTests.cpp` | 統合前の写し。main はフィクスチャ修正済み |
| U2 | `src/convolver/ConvolverStopProbe.h` | probe 計測（除外系統） |
| U2 | `src/learner/LearnerStateAuthority.h` | B-07（不採用） |
| U2 | `src/tests/LearnerStateAuthorityTests.cpp` | B-07（不採用） |
| U2 | `.github/scripts/isr-verify-evidence-consistency.ps1` | B-11 の証跡分離（除外） |
| U2 | `.openchamber/notes-todos.json` | ローカル状態 |
| U2 | `resf_raw.json` | 測定の生データ |

U0 は main が追跡しているファイルで、ローカルの未追跡コピーとバイト一致する。反映すると追跡ファイルになり、内容は変わらない。U1 は統合前の写しで、main 側がフィクスチャ不整合を直した版になっている。

### 5.3 未追跡 doc 339 ファイル

`doc/work113/` の過去報告書と `ConvoPeq.md` のスナップショットである。履歴であって現行ソースではない。main は `doc/` を追跡していないため、反映しても衝突しない。削除もしない。

## 6. 反映で失われるもの

| 系統 | 対象 | 行数 | 判断 |
| --- | --- | --- | --- |
| probe 計測 | 6 ファイルと `ConvolverStopProbe.h` | 81 | 2026-11-14 から除外を継続 |
| B-07 | `NoiseShaperLearner.h/.cpp`、`LearnerStateAuthority.h`、そのテスト、`CMakeLists.txt` の登録 | 102 | 不採用（upstream の C-03 を採用） |
| B-11 実測証跡 | `MainApplication.h/.cpp`、`AudioEngine.Commit.cpp` | 20 | 見送り（エクスポータが `not_measured` を設計上出す） |
| reader 計測 | `AudioEngine.Reader.cpp` | 13 | LINT-AE-011（`thread_local` 禁止） |
| ISR スクリプト | `.github/scripts/` の 5 本 | 差分全体 | 除外（§6 の再検討項目） |
| ASIO 警告 | `MainWindow.cpp` | 12 | 未決（§8） |
| 仕様コメント | `MKLNonUniformConvolver.h/.cpp`、`ConvolverProcessor.Runtime.cpp` | 12 | 実装は main に在る |

行数は「main に同じ行が無い」ものを数えた値である。行が無いことは機能が無いことを意味しない。個別に確認した範囲では、CacheManager の C-05 と 2^21+1、B-09 の非有限値の拒否、B-12 の層ゲイン、B-14 のロガー所有権（main は C-16 として実装）はいずれも main に在る。

いずれも `tracked_changes.patch` と保全ブランチから復元できる。probe 系は既定 OFF の診断なので、失われても製品の挙動は変わらない。

## 7. 反映で得られるもの

- 99 ファイルの更新と 120 ファイルの追加。B-01、B-04 から B-06、B-08 から B-10、B-13、B-15、BUG-3、C-03、C-05、C-16 が入る。
- 保護ツリーは 232 コミット遅れている。このままだと新しいテスト（`InputInspectionTests` など）を含まない。ただし今回の統合で取り込んだ分は、保護ツリーの未追跡コピーが main と一致している（§5.2 の U0）。
- 失うものより得るものが大きい。とはいえ、それは「一括で上書きしてよい」理由にはならない。順序を踏めば同じ結果を安全に得られる。

## 8. 反映手順（案）

予行と本体を分ける。どちらも「追跡ファイルを main に合わせ、未追跡は触らない」方式である。

### 8.1 予行

保護ツリーの複製を作り、同じ操作を先に通す。`git clone` は追跡内容だけを複製するので、巨大な `build/` を巻き込まない。

```
git clone -n C:\VSC_Project\ConvoPeq C:\VSC_Project\rehearsal
git -C C:\VSC_Project\rehearsal checkout 4d476d5c
git -C C:\VSC_Project\rehearsal apply %TEMP%\opencode\refl20261124\tracked_changes.patch
git -C C:\VSC_Project\rehearsal fetch --no-tags C:\VSC_Project\integ\ConvoPeq main
git -C C:\VSC_Project\rehearsal reset --hard FETCH_HEAD
git -C C:\VSC_Project\rehearsal status --porcelain -uall
```

未追跡の doc/ は複製に含まれないので、予行では件数が合わない。追跡ファイルの一致だけを見る。

### 8.2 本体（Owner 承認後）

1. 本章のアーティファクトが揃っていること、zip のハッシュが manifest と一致することを再確認する。
2. `git -C C:\VSC_Project\ConvoPeq fetch --no-tags C:\VSC_Project\integ\ConvoPeq main` を実行する。`.git` だけが変わり、作業ツリーは動かない。
3. `git -C C:\VSC_Project\ConvoPeq reset --hard 75e64841` を実行する。追跡ファイルが main の内容になる。未追跡は残るが、main が追跡する 10 本は main の内容で置き換わる（§5.2）。
4. `opencode.json` に `protocol` の 1 行を再適用する（または §9 の PR で main に入れる）。
5. `ConvolverStopProbe.h` など、参照されなくなった未追跡ヘッダはそのまま残す。ビルドには影響しない。
6. ビルドとテストを実行する（§9）。

`reset --hard` を使うのは、追跡ファイルの WIP を patch として既に保全しているためである。作業ツリーの未追跡 355 件は消えない。

### 8.3 別案

保護ツリーを凍結したまま、日常の作業を `C:\VSC_Project\integ\ConvoPeq`（main）へ移す方法もある。反映のリスクはゼロで、比較もできる。ただし保護ツリーは 232 コミット遅れたまま残るので、どちらを正とするかを決める必要がある。

## 9. 反映後の検証

- `git status --porcelain -uall` で、追跡変更が `opencode.json` の 1 件だけであること。未追跡は doc/ 339、ローカル専用 6 件、`MainWindow.cpp` の ASIO 警告を残した場合はその 1 件。
- `git diff --stat 75e64841` が `opencode.json` の 1 行だけであること。
- Release と Debug のビルド、`ctest` が 68/68 と 67/67（無負荷）。
- `grep -rn ConvolverStopProbe src/` が 0 件であること。
- `ConvoPeq.md` は main 版のまま。再生成は不要。

## 10. 要判断・残課題

1. ASIO 警告ダイアログ 12 行を main に載せるか。main は作成失敗をログしてユーザー領域へフォールバックするが、`loadFromFile` の失敗は無言である。
2. `opencode.json` の `protocol` 行を main の PR として入れるか。保護ツリーにだけ入れると、次の統合で再発する。
3. B-07 の残り検証（`setState` と公開スナップショットの同期）と B-11（実測必須 artifact の一覧）。
4. Issue #265（OOM 分岐）と #266（CodeQL 3 件）。
5. Debug の `v11_7` は負荷依存で落ちる。判定は無負荷で行う。
6. 反映方式（本ツリーで実施するか、作業環境を新クローンへ移すか）。

## 11. 付録

### 11.1 判定方法

`4d476d5c`（base）、作業ツリー（work）、`75e64841`（main）の 3 者をファイル単位で比較した。work と main がバイト一致なら F0。違う場合は base から work への追加行を 1 行ずつ取り出し、main に同じ行があるかを数えた。残差が除外系統かコメントだけなら F1、未決の改善を含むなら F2 とした。base と main が同一で work だけが違うものは F3・F4 である。

### 11.2 アーティファクト

| 種別 | 場所 | サイズ |
| --- | --- | --- |
| 再保全一式 | `%TEMP%\opencode\refl20261124\` | manifest 51,396 B / patch 759,420 B / zip 20,931,229 B |
| Dropbox | `/ConvoPeq-reflection-plan-20261124/` | 同上 |
| 対応表（機械可読） | `doc/work113/20261124-0030_反映計画/mapping.tsv` | 61 行（見出し 1 + 追跡 43 + 未追跡 16 + doc 1） |

