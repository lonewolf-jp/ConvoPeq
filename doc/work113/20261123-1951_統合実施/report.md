# 保護ツリー WIP の統合実施（2026-11-23）

`wip/preserve-20261123` の作業を main `f79012ef` に統合した。方針は「upstream を基本実装とし、WIP 固有の安全性・検証上の改善だけを移植する」。保護ツリー `C:\VSC_Project\ConvoPeq` は今回も変更していない。

統合コミットは `f2434c9c`（親 `f79012ef` と `adfb16f6`）。main との差は 36 ファイル、+7,179 / −522 行。

## 1. 決定と根拠

### 1.1 CacheManager の標本上限

**upstream の検査処理を採り、上限だけ WIP の 2^21+1 を維持した。**

2^21+1 は暫定値ではなく、`IRLoadAdmission.h` の `kMaxIRResampleOutputSamples`（= `MAX_IR_LATENCY` (2^21) + 1）である。R2 で凍結された 4 値のひとつで、FC-FORM-5「resample 出力長の上限」として定義されている。キャッシュが保持するのは準備済み IR（リサンプル後・トリム後）なので、この上限は対象と整合する。

upstream の 2^28 は 48 kHz で約 93 分に相当する。準備済み IR の上限としては緩く、WIP の 2^21+1（48 kHz で約 43.7 秒）で十分である。

実装は upstream の C-05 ブロック（dataSize / numChannels / numPartitions / fftSize / 積の一致 / 時間領域ディスクリプタ / 総サイズ）をそのまま使い、標本数の上限だけ差し替えた。WIP の `validateTimeDomainDescriptor` は同じ検査の重複になるため採用していない。読み出し直前の再検査は、関数を復活させず 1 つの条件式にまとめた。

### 1.2 学習器の状態保護

**upstream の C-03（`optimizerMutex`）を採用し、B-07 の `LearnerStateAuthority` は持ち込まない。**

ただし、C-03 が B-07 の想定した競合を包含するかを確認したところ、**包含していなかった**。

- C-03 が守るのは `NoiseShaperLearner` の optimizer（mean / covariance / sigma）である。
- BUG-1 の競合は別のオブジェクトで起きる。`AudioEngine::getAdaptiveNoiseShaperState()` が `bank.state`（560 B）をロックなしでコピーし、`setAdaptiveNoiseShaperState()` だけが `stateMutex` を取っていた。
- したがって upstream のままでは torn read が残る。

owner の指示どおり B-07 の設計には戻さず、**既存の `stateMutex` を使う最小の修正だけを移植した**。`AudioEngine.h` の `stateMutex` を `mutable` にし、読み取り側でも同じ mutex を取る。WIP の `NoiseShaperStateRaceTests`（opt-in、`CONVOPEQ_NS_STATE_RACE_TEST=1`）も移植し、この経路を検証できる状態にした。

B-07 が解こうとした「復旧要求と公開スナップショットの同期」は、C-03 と `stateMutex` で足りるか未検証の部分が残る。§6 に残課題として挙げた。

### 1.3 B-02 / B-03 の回帰テスト

**残し、実装に合わせて期待値を修正した。**

- `CacheManagerHeaderValidationTests` は、upstream の C-05 が要求する「`numPartitions * fftSize * 8 == dataSize`」に合わせてフィクスチャを直した（64 本 → 2048 本）。B-08（ディレクトリ剪定）と B-01/B-02（キー版数）の検査も含め、27 項目すべて通る。
- `RuntimeParameterGuardTests` は upstream の C-07（非有限値の拒否）の振る舞いを検証する形でそのまま通った。

upstream に同等のテストがある場合は重複を避けた。たとえば B-12 のテール整形は、upstream の `TailLayerIntegrityTests` に同じ検査（H-4 / H-7 / B12-G）が既に入っているため、WIP 側の重複は採用していない。

## 2. 実施内容

### 2.1 ブランチとマージ

`origin/main`（`f79012ef`）から `integ/wip-selective` を切り、`wip/preserve-20261123` を `--no-ff` でマージした（`merge.conflictStyle=diff3`）。

- 衝突は 14 ファイル。うち 5 は upstream 側をそのまま採用（`ConvoPeq.md`、`MKLNonUniformConvolver.cpp` / `.h`、`MainApplication.h`、`ConvolverProcessor.Runtime.cpp`）。
- 残る 9 は内容を読んで解決した（`AGENTS.md`、`CacheManager.cpp`、`IRConverter.cpp`、`MainApplication.cpp`、`MainWindow.cpp`、`NoiseShaperLearner.cpp` / `.h`、`ConvolverProcessor.LoaderThread.cpp`、`TailLayerIntegrityTests.cpp`）。
- 自動マージされた 17 ファイルも、後述の方法で両側の変更が残っていることを確認した。

### 2.2 除外したもの

| 対象 | 理由 |
| --- | --- |
| `ConvolverStopProbe.h` と 5 ファイルの計測行 | 既定 OFF の診断専用。2026-11-14 の統合で意図的に外した系統 |
| `src/learner/LearnerStateAuthority.h`、`LearnerStateAuthorityTests.cpp` | B-07 の設計。§1.2 のとおり不採用 |
| `doc/` の未追跡 338 ファイル | 過去の報告書と `ConvoPeq.md` のスナップショット。履歴であって現行ソースではない |
| `.openchamber/notes-todos.json` | ローカル状態。upstream も追跡していない |
| `resf_raw.json` | 測定の生データ。ルート直下に置かない |
| `.github/scripts/isr-verify-evidence-consistency.ps1` | B-11 の証跡分離。upstream の live evidence とは別系統 |
| 5 本の ISR スクリプトの WIP 差分 | 同上（§6 に再検討項目として記載） |

### 2.3 自動マージ 17 ファイルの意味レビュー

行単位で機械的に確かめた。各ファイルについて、WIP が追加した行と upstream が追加した行をそれぞれ取り出し、統合後のファイルに存在するかを数えた。除外した系統（probe、B-07、B-11 の代替実装）に属する行を除き、**取りこぼしは 0**。

ずれが出たのは、意図的に置き換えた箇所だけである。

- `IRConverter.cpp` の upstream 側 15 行は、ストリーミング実装に置き換わった旧コード。C-06 の 2^24 上限は `fileLength` に対して残した。
- `MainWindow.cpp` の upstream 側 3 行は、`created` 変数を使わない形に整理したもの。失敗時のログは残している。
- `CacheManager.cpp` の upstream 側 1 行は 2^28 の上限。§1.1 のとおり差し替えた。

## 3. 統合中に見つけて直した不具合

いずれも統合作業そのものに起因する。製品コードの既存不具合ではない。

1. **`CacheManager::validateCacheFile` が常に false を返していた。** 競合解消の編集で、`return true;` の直前に `return false;` が残っていた。全検査を通ったあとに落ちるため、キャッシュが一切読めず、`CacheManagerHeaderValidationTests` の C-1 が落ちて発覚した。
2. **`CacheManagerHeaderValidationTests` のフィクスチャが C-05 の積検査と矛盾していた。** `numPartitions=4` / `fftSize=512` に対してパーティション本体が 64 本しかなく、`4 * 512 * 8 != 512` で正当なファイルまで拒否されていた。2048 本に直した。あわせて、キャッシュファイルが見つからないときに空の `MemoryBlock` を `memcpy` して落ちる経路を早期 return に変え、出力を無バッファにした（中断時にどこまで進んだかを失わないため）。
3. **`NoiseShaperLearner.h` に B-07 の `pendingModeRaw`（atomic）が残っていた。** upstream の `.cpp` は `LearningMode pendingMode` を使うためコンパイルできなかった。`LearningMode pendingMode` に戻した。

## 4. 検証

### 4.1 ビルド

| 構成 | 結果 |
| --- | --- |
| Release 全ターゲット | 493 ステップ、エラー 0 |
| Debug 全ターゲット | 865 ステップ、エラー 0 |

### 4.2 テスト

- **Release `ctest`: 68/68 合格（194.04 秒）。**
- 新規・変更分の個別結果:
  - `CacheManagerHeaderValidationTests` 27/27
  - `InputInspectionTests` 13/13（X-1〜X-7、R-1〜R-4）
  - `LearnedStateMigrationTests` 12/12（M-15-1〜12）
  - `TailLayerIntegrityTests` 失敗 0（H-1〜H-7、B12-G 一式）
  - `AudioEngineHarness` 合格（Release 125.5 秒）
- **Debug `ctest`: 67/67 合格（272.97 秒、無負荷時）。**
- Debug は負荷に敏感である。ビルドや他の作業と並行して走らせた 1 回目は 66/67 で、`AudioEngineHarness` が `checkV11Concurrency` の `v11_7` で落ちた。これは `AGENTS.md` に既知の負荷依存フレーキーとして記録されている検査で、検出 0 件になると落ちる。同じ 2 回目は STG8 の teardown 検査の途中で終了した。どちらも無負荷で再実行すると合格しており、統合による恒久的な失敗ではない。**Debug の判定は無負荷で行うこと。**

Debug の既知失敗 2 件のうち、`D8_2_B_2_Tests`（スタック不足）と `TailLayerIntegrityTests`（CPU 予算）は**どちらも今回は合格した**。前者は 0.49 秒、後者は 14.55 秒。upstream が CPU 判定を `TailLayerPerfTests` へ移したことと、実行環境のスタック設定による。

## 5. ConvoPeq.md の再生成

統合後のソースから再生成した。生成元は統合コミット `f2434c9c` の作業ツリーである。

| 項目 | 値 |
| --- | --- |
| 生成日時 | 2026-11-23 19:51:06 |
| サイズ（生成直後） | 7,751,409 B（CRLF） |
| SHA-256（生成直後） | `edd307dd2f9200636bcc838e61970862ef7a70a9c0e5246fd834992c26a3d2ed` |
| 分割 | 4 ファイル（2,000,000 / 1,999,986 / 1,999,990 / 1,751,433 B）。再結合がバイト一致 |
| 秘密情報走査 | 0 件 |

リポジトリの `.gitattributes` は `*.md text eol=lf` なので、コミット後の blob は LF 正規化される（生成直後とはサイズ・ハッシュが変わる）。分割ファイルも同じ扱いで、リポジトリ内の LF 版でも再結合が一致することを確認した。

## 6. 残課題

1. **B-11 の証跡検証強化。** WIP は `isr-verify-common.ps1` に「`status=not_measured` / `not_run` を合格にしない」「`provenance=synthesized` を合格にしない」、`isr-verify-v5.ps1` に「`insufficient_observations` を合格にしない」を追加していた。いずれも fail-closed の改善だが、upstream の `ISREvidenceExporter` は未測定 artifact に `status=not_measured` を**設計上**書き出す。無条件に拒否すると ISR パイプラインが止まるため、今回は入れていない。どの artifact が実測必須かの一覧を決めてから入れる。
2. **B-07 が想定した競合の残り。** 学習中の `setState`（復旧）と公開スナップショットの同期は、C-03 の `optimizerMutex` と `stateMutex` で足りるかを検証していない。B-07 の設計を復活させるのではなく、必要なら独立 Issue で扱う。
3. **`checkV11Concurrency` の負荷依存。** Debug で負荷が高いと検出 0 件で落ちる。既知だが、統合でハーネスの検査が増えたぶん当たりやすくなっている可能性がある。
4. **保護ツリーへの反映。** 今回の統合は独立クローン内で完結している。保護ツリー `C:\VSC_Project\ConvoPeq` は `4d476d5c` のまま（未コミット 396 エントリ）で、反映は別途判断が要る。

## 7. 参照

- 比較の記録: `doc/work113/20261123-1842_統合比較/report.md`
- 保全の記録: `doc/work113/20261123-1831_WIP保全/report.md`
- 保全ブランチ: `wip/preserve-20261123`（`adfb16f6`）
- 統合コミット: `f2434c9c`（親 `f79012ef` + `adfb16f6`）
