# D8 スタック修正の main マージ（Task A）

- 実施日時: 2026-11-14 17:30〜18:40 JST
- 対象: `src/tests/D8_2_B_2_Tests.cpp` のスタック使用量修正
- 種別: 分離・再検証・PR・main マージ
- 基準: `6e099932bcb4f122a72526f1c9786eecd7fc08b6`（作業開始時の `main` = `origin/main`）

---

## 0. 事前確認

| 項目 | 結果 |
| --- | --- |
| 作業開始時の `HEAD` / `origin/main` | `6e099932bcb4f122a72526f1c9786eecd7fc08b6` / 同値（`git ls-remote` でも一致） |
| 作業ツリーの未コミット変更 | modified 39 / untracked 136 / staged 0 |
| 対象ファイルの差分 | `src/tests/D8_2_B_2_Tests.cpp` のみ +28 / −8、SHA-256 `B55768FE…`（報告済みの値と一致） |
| `ConvoPeq.md` | 2026-11-14 15:27 生成（7,067,912 B）。**D8 修正前の内容**（`domStorage = std::make_unique` は 0 件、修正前の宣言が 4 件）で、現行ソースとは一致しない。資料ではなく現行ソースを正として作業した |

`ConvoPeq.md` はヘッダーに生成元コミットを持たないため、一致確認は内容で行った。D8 修正の適用後に再生成していない（本作業の対象外）。

---

## 1. 分離

作業ツリーには D8 修正以外に 39 件の未コミット変更があった。D8 修正だけを分離するため、`origin/main` から独立した worktree を作成した。

| 項目 | 値 |
| --- | --- |
| worktree | `C:\VSC_Project\d8-merge` |
| ブランチ | `fix/d8-stack-frame`（`origin/main` から作成） |
| 適用方法 | 作業ツリーの `src/tests/D8_2_B_2_Tests.cpp` を作業先へコピー（内容は同一） |
| 適用後の差分 | `src/tests/D8_2_B_2_Tests.cpp` のみ **+28 / −8** |
| 適用後の SHA-256 | `B55768FE7085F1312C3952F3AE0D229B0ECD3BD11E661EC98FE67C2518AE29CB`（報告済みの値と一致） |
| コミット | `62f12321` `fix(test): keep the D8 T11/T12/T13 epoch providers off the stack` |

差分の内容は、`EpochDomain` / `FullQueueEpochProvider` のローカル宣言 8 箇所を `std::make_unique` ＋ 参照エイリアスへ置き換えたものだけである。API・引数・アサーション・検証条件の変更はなく、`/STACK` も変更していない。

---

## 2. 独立 worktree での再検証

CMake を新規に構成し（`Ninja Multi-Config`、MSVC、MKL/IPP 検出あり）、Debug / Release をビルドした。

| 項目 | 結果 |
| --- | --- |
| Debug ビルド | 成功（154 ターゲット、129 秒） |
| Release ビルド | 成功（153 ターゲット、183 秒） |
| `D8_2_B_2_Tests`（Debug） | exit 0。T1〜T13 すべて PASS（T5 / T6 / T9 は設計上の DEFERRED） |
| `D8_2_B_2_Tests`（Release） | exit 0。同上 |
| 関連 ctest（Debug、11 件） | **11/11 PASS**（D8_1 / D8_2_B_2 / DeferredDeletionQueueReclaim / DomainTag / RetireGraceSemantics / ShutdownRetireIntentDrain / StuckReaderFallbackDrain / ScopedEpochReader / ScopedEpochReaderStructural / STG11EQRetire / STG11D2SnapshotRetire） |

フレームサイズ（Debug、`.pdata` の関数開始 RVA と `mov eax, imm32` ＋ `__chkstk` の即値から測定）:

| 関数 | 修正後 |
| --- | ---: |
| `test_T13_C2NonRtOnly` | 403,568 B |
| `test_T11_CompleteFadeRecords` | 165,728 B |
| `test_T12_FinalizeRetiresTargetAndCurrent` | 80,736 B |

stack reserve は `0x100000`（Debug / Release とも）。

---

## 3. PR と CI

| 項目 | 値 |
| --- | --- |
| PR | #210 `fix(test): D8 T11/T12/T13 の EpochDomain をスタックからヒープへ移す` |
| URL | https://github.com/lonewolf-jp/ConvoPeq/pull/210 |
| 変更ファイル | `src/tests/D8_2_B_2_Tests.cpp` のみ |
| CodeRabbit | pass（指摘なし: "No actionable comments were generated"） |

CI（PR 時点）は 15 件すべて success、1 件（dispatch 専用の probe）は skipping。

| チェック | 結果 |
| --- | --- |
| Windows MSVC / Release | pass（17m24s） |
| RelWithDebInfo ASan (MSVC, /MD) | pass（1h6m10s） |
| Debug TSan (Clang, best-effort) | pass |
| Analyze (actions / c-cpp / csharp / javascript-typescript) | pass |
| CodeQL | pass |
| CodeRabbit | pass |
| isr-verify / authority-static-checks / authority-runtime-tests | pass |
| audit-report-verification / compliance-list-md / lint-audioengine / Detect changed files | pass |

---

## 4. マージ

| 項目 | 値 |
| --- | --- |
| マージ方式 | merge commit（`gh pr merge --merge`） |
| マージコミット | `4d476d5c65261fd975de2606f4d8b5c843cd86e1` |
| マージ日時 | 2026-11-14 09:22:14 UTC |
| `origin/main` | `4d476d5c`（ローカル `main` と一致） |
| マージコミットの変更 | `6e099932..4d476d5c` で **1 ファイルのみ**（`src/tests/D8_2_B_2_Tests.cpp`） |

作業ツリーの更新: D8 ファイルのローカル変更は、マージ後の blob（`11d5b798`）と同一内容だったため、`git checkout -- src/tests/D8_2_B_2_Tests.cpp` で戻してから `git merge --ff-only origin/main` を実行した。更新後のファイル SHA-256 は `B55768FE…` で変わっていない。他の 38 件の未コミット変更には触れていない（staged 0、`reset --hard`・`clean` は未使用）。

---

## 5. マージ後の検証（main）

| 検証 | 結果 |
| --- | --- |
| `HEAD` / `origin/main` | `4d476d5c` / `4d476d5c` |
| `D8_2_B_2_Tests`（Debug、再ビルド後） | exit 0、T1〜T13 PASS |
| `D8_2_B_2_Tests`（Release、再ビルド後） | exit 0、T1〜T13 PASS |
| フレーム（Debug） | T13 **403,568 B**、T11 165,728 B（T12 80,736 B）。最大は T13 |
| stack reserve | `0x100000` |
| 全 ctest（Release、61 件） | **61/61 PASS** |
| 全 ctest（Debug、61 件） | 60/61 PASS。失敗は `TailLayerIntegrityTests`（RES-F）1 件のみ |

`TailLayerIntegrityTests` の失敗は本修正とは無関係で、Task B の調査対象である（D8 のテスト TU をリンクしない別ターゲット）。なお Release の全 ctest は、同じツリーで以前に実行したとき 60/61（同じ 1 件が失敗）だった。今回は 61/61 で、RES-F の判定が実行環境の負荷で変わることを裏づけている。

---

## 6. main 上の CI

マージコミット `4d476d5c` に対して 6 ワークフローが起動し、**すべて success** で完了した。

| ワークフロー | 結果 |
| --- | --- |
| Windows MSVC Build | success |
| Sanitizer CI（Debug TSan / RelWithDebInfo ASan） | success |
| Push on main（CodeQL: actions / c-cpp / csharp / javascript-typescript） | success |
| ISR Authority Compliance v6.4 | success |
| audioengine-lint | success |
| list-compliance | success |

---

## 7. 未解決事項

- Task B（RES-F の修正設計）は `doc/work113/20261114-1722_RES-F設計/report.md` に分離して提出済み。実装・閾値変更・CI 変更は未実施で、Owner 判断を待つ。
- 作業用 worktree `C:\VSC_Project\d8-merge` は残置（ブランチ `fix/d8-stack-frame` はマージ済み）。不要になった時点で撤去してよい。
