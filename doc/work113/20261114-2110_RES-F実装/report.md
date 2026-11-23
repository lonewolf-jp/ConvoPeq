# RES-F 仕様 §3.3 準拠修正の実装報告

- 実施日時: 2026-11-14 20:30〜22:00 JST
- ブランチ: `fix/resf-spec-compliance`（起点 `origin/main` = `4d476d5c65261fd975de2606f4d8b5c843cd86e1`）
- コミット: `4b796f6f`（実装）、`e32e7d05`（CodeRabbit 指摘対応）
- PR: #211 https://github.com/lonewolf-jp/ConvoPeq/pull/211
- 種別: テストコードとテスト構成の変更（Production コード・閾値・仕様は不変）
- 承認範囲: Owner 指示 1.1〜1.4 の範囲のみ

---

## 1. 変更したファイルと変更内容

| ファイル | 変更 | 差分（2 コミット合計） |
| --- | --- | --- |
| `src/tests/TailLayerIntegrityTests.cpp` | RES-F の CPU 判定を外し `[SKIP]` を明示。メモリ判定を `RES-M` として維持 | +17 / −5 |
| `src/tests/TailLayerPerfTests.cpp`（新規） | Release 専用の性能テスト（仕様 §3.3 準拠） | +300 |
| `CMakeLists.txt` | `TailLayerPerfTests` ターゲットと Release 限定の CTest 登録 | +62 |

合計 **+379 / −5**（内訳: 62 + 17 + 300 = 379 追加、5 削除）。数値は `git diff --numstat origin/main..HEAD` の実測値である。

上記 3 ファイル以外は変更していない（JUCE・r8brain-free-src・Production コードは不変）。

### 1.1 `TailLayerIntegrityTests.cpp`

- `reportSkip()` を追加し、`[SKIP]` 行を出す（`gFailures` を増やさない）
- `checkF()` の RES-F 判定を 2 行に分離
  - `report ("RES-M", r.memDeltaMB <= kMemLimitMB, ...)` … 32 MB の判定を Debug/Release 両構成で維持
  - `reportSkip ("RES-F-cpu", fmt ("... cpu_diag=%.3f ms/s (RES-F CPU is evaluated by TailLayerPerfTests in Release)"))`
- `kCpuLimitMsPerSec = 10.0` は値を変えず残す（`[[maybe_unused]]` とコメントで、判定は性能テスト側であることを明示）
- F/H/I の判定式・ケース表・許容値は無変更

### 1.2 `TailLayerPerfTests.cpp`（新規）

仕様 §3.3 の測定定義をそのまま実装した。

| 項目 | 実装 |
| --- | --- |
| 指標 | `Add` + `Get` の総壁時計時間 ÷ 処理音声秒数（`std::chrono::steady_clock`） |
| 入力生成 | 計測区間の外。先頭ブロック用（先頭のみ 1.0）と無音ブロック用の 2 本を事前に作る |
| ウォームアップ | 初回 1 回を実行して破棄（中央値に含めない） |
| サンプル数 | 5 回。ソートして添字 2 を中央値とする |
| 1 サンプルの長さ | 最低 5 音声秒（`repsPerSample = ceil(5.0 / audioPerRun)`） |
| 測定対象 | block 64〜32768 × IR 0.5 / 1.0 / 3.0 秒の 30 通り × **2 モード**（バイパス tailMode=2 / テール保持 tailMode=1）= **60 測定** |
| 外乱検出 (1) | 試行全体の壁時計 / プロセス CPU 時間 > 1.25（スケジューリング待ち） |
| 外乱検出 (2) | 5 サンプルの max/min > 1.25（競合による実行速度の変動） |
| 再測定 | 外乱時は最大 5 回まで。全回外乱なら FAIL（理由を出力） |
| 判定 | `cpu <= 10.0 ms/s` かつ `memDeltaMB <= 32.0`（**値は変更しない**） |
| Debug | `[SKIP] RES-F (Debug build: CPU budget is evaluated in Release by TailLayerPerfTests)` を出して 0 で終了 |
| 出力 | `[PASS]/[FAIL] RES-F mode=...`、`[DIAG]`（5 サンプル・中央値・音声秒・壁時計/CPU 比・spread・試行回数）、`[RETRY]`、`[SUMMARY]` |

2 モードを測るのは、旧 RES-F がバイパス（`checkF` は `tailEnabled=false`）だけを測っていたためで、CPU 回帰の被覆を落とさないためである（CodeRabbit 指摘への対応）。

### 1.3 `CMakeLists.txt`

`TailLayerIntegrityTests` のブロックと同構成で `TailLayerPerfTests` を追加し、次を設定した。

```cmake
add_test(NAME TailLayerPerfTests COMMAND TailLayerPerfTests CONFIGURATIONS Release)
set_tests_properties(TailLayerPerfTests PROPERTIES RUN_SERIAL TRUE LABELS "perf")
```

- Release 構成でのみ登録（Debug / RelWithDebInfo の ctest 一覧に現れない）
- `RUN_SERIAL` は仕様 §3.3「他プロセスの負荷を排除した状態で測る」に対応
- **CI の必須ゲートにはしていない**（Owner 判断 1 = 保留のため）

---

## 2. ビルド結果

独立 worktree（`C:\VSC_Project\resf-work`、`4d476d5c` から作成）で新規に CMake 構成した。

| 構成 | 対象 | 結果 |
| --- | --- | --- |
| Debug | `TailLayerIntegrityTests` / `TailLayerPerfTests` | 成功 |
| Release | 同上 | 成功 |
| Debug | 全ターゲット（回帰確認用） | 成功（545 ターゲット、443 秒） |
| Release | 全ターゲット（回帰確認用） | 成功（289 ターゲット） |

---

## 3. 機能テスト結果

`TailLayerIntegrityTests` の出力（30 ケース）。

| 構成 | 終了コード | `[SKIP] RES-F-cpu` | `[PASS] RES-M` | `[FAIL]` | 備考 |
| --- | --- | --- | --- | --- | --- |
| Debug | 0 | 30 | 30 | 0 | 従来は RES-F が 8 件 FAIL |
| Release | 0 | 30 | 30 | 0 | — |

F/H/I の判定は変更しておらず、全件 PASS（`[PASS]` 行 171 件）。

SKIP 行の例:

```
[SKIP] RES-F-cpu block=64 ir=24000 cpu_diag=22.422 ms/s (RES-F CPU is evaluated by TailLayerPerfTests in Release)
[PASS] RES-M block=64 ir=24000 mem=1.97 MB
```

`cpu_diag` は旧方式（ウォームアップを含む 5 回の中央値）の参考値で、**合否には使わない**。

---

## 4. 性能テストの測定値（Release、無負荷）

60 測定すべて PASS。実行時間 4.9〜5.5 秒。

| block | IR | バイパス CPU (ms/s) | テール CPU (ms/s) | メモリ最大 (MB) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 24000 | 4.514 | 4.445 | 1.98 |
| 64 | 48000 | 5.250 | 5.116 | 2.47 |
| 64 | 144000 | **7.818** | **7.720** | 7.92 |
| 128 | 24000 | 2.625 | 2.508 | 1.36 |
| 128 | 48000 | 3.840 | 3.885 | 3.51 |
| 128 | 144000 | 6.203 | 6.372 | 8.31 |
| 256 | 24000 | 1.629 | 1.697 | 1.22 |
| 256 | 48000 | 2.602 | 2.495 | 2.74 |
| 256 | 144000 | 4.628 | 4.304 | 10.96 |
| 512 | 24000 | 1.266 | 1.266 | 3.07 |
| 512 | 48000 | 1.693 | 1.662 | 3.59 |
| 512 | 144000 | 3.440 | 3.516 | 10.88 |
| 1024 | 24000 | 1.083 | 1.083 | 4.15 |
| 1024 | 48000 | 1.354 | 1.322 | 4.44 |
| 1024 | 144000 | 2.066 | 2.085 | 13.10 |
| 2048 | 24000 | 1.046 | 1.055 | 4.15 |
| 2048 | 48000 | 1.190 | 1.194 | 4.68 |
| 2048 | 144000 | 1.891 | 1.873 | 14.05 |
| 4096 | 24000 | 0.943 | 0.951 | 2.19 |
| 4096 | 48000 | 1.165 | 1.184 | 4.20 |
| 4096 | 144000 | 1.867 | 1.821 | 14.08 |
| 8192 | 24000 | 0.944 | 0.953 | 5.09 |
| 8192 | 48000 | 1.058 | 1.087 | 7.51 |
| 8192 | 144000 | 1.580 | 1.433 | 15.80 |
| 16384 | 24000 | 0.972 | 1.017 | 8.99 |
| 16384 | 48000 | 1.013 | 1.017 | 9.66 |
| 16384 | 144000 | 1.285 | 1.305 | 13.01 |
| 32768 | 24000 | 0.537 | 0.535 | 8.54 |
| 32768 | 48000 | 1.357 | 1.192 | 17.36 |
| 32768 | 144000 | 1.445 | 1.532 | 22.39 |

- 最悪は block=64 / IR 3.0 秒の **7.818 ms/s（バイパス）/ 7.720（テール）**。予算 10 に対し余裕 22〜28%
- メモリ最大は 22.39 MB（予算 32 MB）
- 3 回反復での最悪ケース: バイパス 7.261 / 7.477 / 7.349、テール 7.283 / 7.463 / 7.233（変動 ±3%）

中央値の算出確認（実例）:

```
[DIAG] mode=tail block=64 ir=144000 samples=[7.267 7.597 8.043 7.720 7.870] median=7.720 audio/sample=6.02s wall/cpu=0.989 spread=1.107 attempts=1
```

5 サンプルをソートすると `[7.267 7.597 7.720 7.870 8.043]` で、中央値（添字 2）は 7.720。出力の `median` と一致する。

ウォームアップの扱い: サンプル計測の前に `runOnce()` を 1 回実行して破棄している。`[DIAG]` に現れるのは 5 サンプルのみで、初回のコールド値は含まれない（コードレビューと出力の双方で確認）。

1 サンプルの音声長: `audio/sample` は 5.05〜6.04 秒で、全ケースで 5 秒以上。

---

## 5. 外乱検出・再測定の検証

| 条件 | 終了コード | PASS | FAIL | `[RETRY]` | 最終 DISTURBED |
| --- | ---: | ---: | ---: | ---: | ---: |
| 無負荷 3 回 | 0 / 0 / 0 | 60 / 60 / 60 | 0 / 0 / 0 | 10 / 16 / 11 | 0 |
| 負荷（論理 16 に対しビジーループ 14）※30 測定版で測定 | 1 | 29 | 1 | 8 | 0 |

- 無負荷では、一時的な外乱で `[RETRY]` が発生しても再測定で条件を満たし、60/60 PASS に収束する
- 負荷時は `[RETRY]` が発生し、`spread=2.023`、`wall/cpu=3.430` のような明確な外乱を検出して再測定した
- 負荷時の FAIL は 1 件（block=64 / IR 3.0 秒、10.642 ms/s）。この試行は `samples=[10.680 11.020 10.642 10.473 10.625]`、`wall/cpu=1.030`、`spread=1.052` で、**全サンプルが一様に遅い**ため外乱として検出できなかった

---

## 6. 実装中に判明した測定上の知見

1. **サンプル単位の壁時計 / CPU 比は使えない**: Windows のプロセス CPU 時間はタイマー刻み（既定 15.6 ms）で量子化される。20〜30 ms のサンプルでは比が 1.5 や 0.8 に飛び、無負荷でも外乱と誤判定した（実測: 22 件が FAIL）。試行全体（ウォームアップ + 5 サンプル、約 150 ms）の比に変更して解消した
2. **壁時計 / CPU 比だけでは競合を検出できない**: 負荷時は壁時計と CPU 時間が同じだけ増えるため、比は 1.0 前後になる。サンプル間の max/min（spread）を追加した。無負荷の最大 spread は 1.137、負荷時は 1.36 以上で、閾値 1.25 は両者を分離できる
3. **一様な競合低下は原理的に検出できない**: 全サンプルが同じだけ遅い場合、より遅い CPU と区別できない。この場合は CPU 超過として FAIL する（fail-closed）。仕様 §3.3 が求める「他プロセスの負荷を排除した状態」は実行環境側で担保する必要がある
4. **入力生成を計測区間に含めない**: 旧コードはブロックごとに入力バッファを書いてから `Add`/`Get` を呼んでいた。事前生成に変えて「Add + Get」だけを測るようにした（CodeRabbit 指摘への対応）

---

## 7. 全 ctest の結果

独立 worktree で全ターゲットをビルドして実行した（基準 `4d476d5c` のテスト登録は 57 件）。

| 構成 | 登録 | 結果 | 所要 | 失敗 |
| --- | ---: | --- | ---: | --- |
| Debug | 57 | **57/57 PASS** | 247.9 秒 | なし（従来は `TailLayerIntegrityTests` の RES-F 8 件で 56/57） |
| Release | 58 | **58/58 PASS** | 171.1 秒 | なし（`TailLayerPerfTests` を含む） |

- Debug の一覧に `TailLayerPerfTests` は**登録されない**（CPU を評価していないことが一覧から判別できる）
- Release の `TailLayerIntegrityTests` は 1.97 秒、`TailLayerPerfTests` は 4.56 秒で PASS

---

## 8. CI 結果

PR #211 の CI 結果は §12 に記載する。

---

## 9. D8 / ISR 回帰確認

全 ctest（Debug 57 件 / Release 58 件）に D8・ISR 系のテストが含まれており、すべて PASS した。

- `D8_2_B_2_Tests`、`D8_1_WrapperCacheTests`、`DomainTagTests`、`DeferredDeletionQueueReclaimTests`、`RetireGraceSemantics`、`ShutdownRetireIntentDrain`、`StuckReaderFallbackDrain`、`ScopedEpochReader`（Structural 含む）、`STG11EQRetire`、`STG11D2SnapshotRetire` 等
- 本変更はテスト 1 本の追加とテスト 1 本の判定分離であり、Production コードを含まない

---

## 10. 未解決事項・残存リスク

| # | 内容 | 状態 |
| --- | --- | --- |
| 1 | 一様な競合低下は外乱として検出できない（§6-3） | 設計上の限界。fail-closed で FAIL する。測定環境側の担保が必要 |
| 2 | CI 性能ゲートは未必須（Owner 判断 1 = 保留） | 解除条件は `doc/work113/20261114-2009_RES-F判断/report.md` §1 |
| 3 | 無負荷でも `[RETRY]` が 10〜16 回/60 測定発生する | 再測定で収束するため合否への影響はない。実行時間は 5 秒程度 |
| 4 | Debug の `cpu_diag` は旧方式の参考値 | 合否には使わない。`[SKIP]` 行に明示 |
| 5 | RelWithDebInfo 構成（ASan CI）では性能テストを実行しない | `CONFIGURATIONS Release` のため。仕様の基準構成は Release |
| 6 | 負荷時の検証は 30 測定版で実施した | 60 測定版は無負荷で 3 回確認。負荷時の挙動（外乱検出の作動）は同一コードで確認済み |

---

## 11. Git 作業ツリーの状態

| 項目 | 値 |
| --- | --- |
| 作業 worktree | `C:\VSC_Project\resf-work`（ブランチ `fix/resf-spec-compliance`） |
| コミット | `4b796f6f`（実装）、`e32e7d05`（CodeRabbit 対応） |
| 本体 worktree（`C:\VSC_Project\ConvoPeq`） | `main` = `4d476d5c`。他タスクの未コミット変更 38 件・untracked は**未変更**（staged 0） |
| 破壊的操作 | `reset --hard`・`clean`・他ファイルの checkout は未使用 |
| マージ | **未実施**（Owner 指示にマージの記載がないため、PR のまま） |

---

## 12. CI 結果（PR #211）

コミット `e32e7d05` に対する CI は **15 チェックすべて success**、1 チェック（dispatch 専用の probe）は skipping。

| チェック | 結果 |
| --- | --- |
| Windows MSVC / Release | pass（17m47s） |
| RelWithDebInfo ASan (MSVC, /MD) | pass（1h20m32s） |
| Debug TSan (Clang, best-effort) | pass（43s） |
| Analyze (actions / c-cpp / csharp / javascript-typescript) | pass（c-cpp 14m11s） |
| CodeQL | pass |
| CodeRabbit | pass（4m4s。指摘 2 件は §13 で対応） |
| isr-verify / authority-static-checks / authority-runtime-tests | pass |
| audit-report-verification / compliance-list-md / lint-audioengine / Detect changed files | pass |

PR の状態は `OPEN`・`mergeable`・`mergeStateStatus = CLEAN`。**マージは実施していない**（Owner 指示にマージの記載がないため）。

CI は性能テストを実行しない（`CONFIGURATIONS Release` のため ASan の RelWithDebInfo では登録されず、Windows MSVC / Release ジョブはビルドのみ）。したがって**性能ゲートは必須化されていない**（Owner 判断 1 = 保留のとおり）。

---

## 13. CodeRabbit の指摘への対応

| # | 指摘 | 対応 |
| --- | --- | --- |
| 1 | Major: 性能テストがバイパス構成（tailMode=2）の CPU を測らなくなり、旧 RES-F の被覆が落ちる | 30 通り × 2 モード（バイパス / テール保持）の 60 測定に変更。`mode=` を出力に追加 |
| 2 | Minor: 計測区間に入力生成が含まれ、仕様の「Add + Get」から外れる | 入力を事前生成し、計測は `Add` + `Get` のみに変更 |
