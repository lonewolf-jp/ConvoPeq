# 保護ツリー WIP と upstream の三者比較（2026-11-23）

保護ツリー `C:\VSC_Project\ConvoPeq` の未コミット作業と、最新 main の変更を比較した。目的は、統合時にどちらを採るかをファイル単位で判断できる材料を作ることである。

**保護ツリーは今回も変更していない。** 比較は独立クローン `C:\VSC_Project\integ\ConvoPeq` で行い、保全ブランチ `wip/preserve-20261123` も凍結したまま触っていない。

## 1. 比較の基点

| 役割 | リビジョン | 内容 |
| --- | --- | --- |
| 共通基点 | `4d476d5c` | 保護ツリーの HEAD。ここから両者が分岐した |
| WIP | `adfb16f6` | `wip/preserve-20261123`。未コミット作業を 1 コミットにしたもの |
| upstream | `ce0ac515` | 比較時点の `origin/main` |

WIP 側の変更は 396 エントリ（追跡ファイルの変更 42、新規 354）。upstream は基点から 191 ファイルを変更している（うち新規 99）。

## 2. 比較方法

3 つの手段を併用した。

1. **変更範囲の比較**。`git diff --numstat` で両者の変更ファイルを出し、共通部分を抽出した。
2. **変更内容の比較**。ファイルごとに `difflib` で差分の編集領域を求め、両者の編集領域が基点の同じ行に重なるかを見た。あわせて、WIP が追加した行のうち upstream に存在しないもの（WIP 固有）と、その逆（upstream 固有）を数えた。
3. **マージ試行**。独立クローンで `gh/main` から `integ/merge-trial` を切り、`gh/wip/preserve-20261123` を `--no-ff` でマージして、テキスト衝突の実数を測った。試行後は `merge --abort` で破棄している。

**限界**: 2 の指標は行単位であり、意味の同一性までは判定しない。3 の「自動マージ成功」も、両側の変更が噛み合ったことしか意味しない。以下の分類は、差分の中身を読んだうえでの判断である。

## 3. 全体の結果

- 追跡ファイルの変更 42 のうち、**30 ファイルが upstream でも変更されている**。残る 12 は upstream が触っていない。
- 未追跡 354 のうち、upstream と重なるものは **0**。新規ファイルはすべて衝突しない。
- マージ試行の結果は **13 ファイルで衝突、競合ハンク 24 個**。残る 17 ファイルは自動マージされた。

前回の報告では重複を「27 ファイル」と書いた。これは main `685cacce` 時点の数で、`ce0ac515` までの 2 コミットで 3 ファイル増えた。今回の比較では **30** が正しい。

### 3.1 重要な発見

WIP の修正の多くは、**upstream に別の名前で既に入っている**。

| WIP の呼称 | upstream の呼称 | 状態 |
| --- | --- | --- |
| B-02（CacheManager の時間領域記述子検査） | C-05（ヘッダ全フィールドの検査） | 同じ脅威。上限値だけが違う |
| B-03（mixed 遷移周波数の非有限値拒否） | C-07（同じ箇所の同じ拒否） | 実装が一致。コメントだけ違う |
| B-07（学習器の状態所有権） | C-03（`optimizerMutex` による直列化） | 同じ競合を別設計で解いている |
| B-12（テールゲインを時間基準へ） | B-12（同名で upstream に存在） | 実装が一致。コメントだけ違う |
| BUG-1（`stateMutex` を mutable に） | C-03（`optimizerMutex`） | WIP 案は不要になった |
| B-11（live evidence の分離） | `--verify-runtime-evidence-live` の実装 | upstream が完成版を持つ |

したがって「WIP をそのまま入れる」と「upstream を採る」の二択ではなく、**WIP の独自部分だけを upstream に足す**のが基本方針になる。逆に、WIP 側の重複実装を持ち込むと、同じ処理が 2 系統になり、片方だけが直る事故につながる。

## 4. ファイル別の分類

分類は 3 つに分けた。

- **upstream 採用**: WIP 側の変更が upstream に包含されている、または設計ごと置き換わっている。
- **統合**: 両側に独立した有効な変更がある。片方を捨てると機能か修正が失われる。
- **WIP 採用**: upstream に変更が無く、WIP だけが変えている（30 ファイルの中には無い。§5 を参照）。

| # | ファイル | WIP 側の変更 | upstream 側の変更 | 分類 |
| ---: | --- | --- | --- | --- |
| 1 | `AGENTS.md` | 2026-11-13 の性能原因、domain tag 確定などの記録 | context-mode 2.x、headroom 0.45.0、icx `-O2` の知見 | 統合 |
| 2 | `CMakeLists.txt` | B-02/B-05/B-07/B-15 の 4 テスト、`learner/` と `InputInspection.cpp` の登録 | tsan-substrate、`SimplePeakLimiterKneeTests`、`DrainContractTests` など | 統合 |
| 3 | `.github/scripts/isr-run-tiered-verification.ps1` | B-11 の generate / run 分離（`-AllowSynthesis`） | `-RequireLiveRuntimeEvidence`、`-LiveRuntimeExitMs` | 統合 |
| 4 | `src/CacheManager.cpp` | B-02 時間領域記述子の整合検査 | C-05 ヘッダ全フィールドの検査 | 統合 |
| 5 | `src/ConvolverControlPanel.cpp` | B-10 実効位相モードの表示と警告 | C-20 テール倍率の 2 冪丸め | 統合 |
| 6 | `src/DeviceSettings.cpp` | B-05 安全な XML 解析、BUG-3 単一シリアライズと原子的置換 | C-14 入力ヘッドルーム上限と MMCSS 表示 | 統合 |
| 7 | `src/IRConverter.cpp` | B-01 admission、ストリーミング読み込み、RIFF 検査、末尾無音の除去 | C-06 標本数 2^24 上限、`channels > 8` 拒否、`try/catch` | 統合 |
| 8 | `src/MKLNonUniformConvolver.cpp` | B-12 テールゲインを IR 上の時間位置で定義 | B-12（同名）、BUG-013 の 2 冪丸め、C-19/C-20 | upstream 採用 |
| 9 | `src/MKLNonUniformConvolver.h` | B-12 の仕様コメント | B-12 の実装契約コメント（上位互換） | upstream 採用 |
| 10 | `src/MainApplication.cpp` | B-15 学習状態の明示移行 CLI、live evidence の分岐ガード | live evidence の完成実装、C-23 のサイズ上限ロガー | 統合 |
| 11 | `src/MainApplication.h` | B-14 のロガー保持、`liveEvidenceRequested` | `SizeCappedFileLogger` への置換 | upstream 採用 |
| 12 | `src/MainWindow.cpp` | B-05/B-06 XML 検査、B-13 ASIO ブラックリストのフォールバック | C-15 CLI 数値の厳密検査、C-16 ロガー所有権 | 統合 |
| 13 | `src/MainWindow.h` | B-14 の `juce::FileLogger` | C-16 の `SizeCappedFileLogger` | upstream 採用 |
| 14 | `src/NoiseShaperLearner.cpp` | B-07 状態所有権、B-09 非有限値、B-05 XML 検査 | C-03 `optimizerMutex`、C-28 nothrow、`safetyMargin` | 統合 |
| 15 | `src/NoiseShaperLearner.h` | B-15 の版数フラグ、`publishGenerationResult` の変更 | `[[nodiscard]] bool` 化と `safetyMargin` 追加 | 統合 |
| 16 | `src/audioengine/AudioEngine.Commit.cpp` | B-11 Release でも HB 違反レポートを出す | WORK113 の識別レジストリ更新 | 統合 |
| 17 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | ConvolverStopProbe の計測 3 行 | WORK113 の `finalizeRetireOnce`、drain 除外スコープ | upstream 採用 |
| 18 | `src/audioengine/AudioEngine.Reader.cpp` | ReaderMeasureRegistry の計測（atomic のみ） | WORK113 R1 の observerCount の説明 | 統合 |
| 19 | `src/audioengine/AudioEngine.Threading.cpp` | ConvolverStopProbe の計測 4 行 | WORK113 DQ-V2 の破棄観測 | upstream 採用 |
| 20 | `src/audioengine/AudioEngine.h` | `stateMutex` を `mutable` に（BUG-1） | WORK113 の DiagObjectRegistry、`uuidExhausted` | upstream 採用 |
| 21 | `src/audioengine/ISRRetireLatency.h` | 既定メンバ初期化の削除 | 同じ変更（一致） | upstream 採用 |
| 22 | `src/convolver/ConvolverProcessor.LoadPipeline.cpp` | B-01 失敗理由の末端までの伝搬、probe | WORK113 DIAG の pending / commit 記録 | 統合 |
| 23 | `src/convolver/ConvolverProcessor.LoaderThread.cpp` | B-04 RIFF 境界検査、末尾無音の除去、probe | C-04 世代照合（古いローダの誤操作防止） | 統合 |
| 24 | `src/convolver/ConvolverProcessor.Runtime.cpp` | B-03 非有限値の拒否（2 箇所） | C-07 同じ拒否、C-35 NaN の変換防止 | upstream 採用 |
| 25 | `src/convolver/ConvolverProcessor.StateAndUI.cpp` | probe、協調停止の記録 | `StateSwapOutcome` の導入 | 統合 |
| 26 | `src/core/EpochDomain.h` | 既定メンバ初期化の削除 | DrainCapability、テスト専用アクセス | upstream 採用 |
| 27 | `src/core/ReaderMeasureRegistry.h` | 既定メンバ初期化の削除 | C-29 スロットを initializing 経由で公開 | upstream 採用 |
| 28 | `src/core/ScopedEpochReader.h` | 既定メンバ初期化の削除 | LRU 8 エントリ、`<limits>` | upstream 採用 |
| 29 | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | B-03/BUG-1 テストのディスパッチ | #236 の一括復元テストのディスパッチ | 統合 |
| 30 | `src/tests/TailLayerIntegrityTests.cpp` | B-12 の参照エネルギー判定、524288 の音声確認 | CPU 判定を `TailLayerPerfTests` へ移設 | 統合 |

集計は **upstream 採用 12、統合 18、WIP 単独 0**。30 ファイルすべてに WIP 固有の行があるが、その大半はコメントの言い換えか、upstream が別設計で解いた重複である。

## 5. WIP が新規に追加したソースとテスト（13）

upstream に存在しないため、衝突しない。ただし内容の妥当性は別に確認する。

| ファイル | 対応 | 扱い |
| --- | --- | --- |
| `src/InputInspection.h` / `.cpp` | B-04/B-05/B-06（RIFF 境界、XML サイズ・深さ・DOCTYPE） | WIP 採用 |
| `src/convolver/IRTrailingSilence.h` | B-01（末尾無音の除去） | WIP 採用 |
| `src/learner/LearnedStateMigration.h` / `.cpp` | B-15（版数無しの学習状態を明示移行） | WIP 採用 |
| `src/learner/LearnerStateAuthority.h` | B-07（復旧要求と公開スナップショットの同期） | 要判断（§7-2） |
| `src/tests/CacheManagerHeaderValidationTests.cpp` | B-02 の回帰 | 要判断（§7-1） |
| `src/tests/InputInspectionTests.cpp` | B-04/B-05/B-06 の回帰 | WIP 採用 |
| `src/tests/LearnedStateMigrationTests.cpp` | B-15 の回帰 | WIP 採用 |
| `src/tests/LearnerStateAuthorityTests.cpp` | B-07 の回帰 | 要判断（§7-2） |
| `src/tests/AudioEngineHarness/NoiseShaperStateRaceTests.cpp` | BUG-1 の競合試験（opt-in） | 要判断（§7-2） |
| `src/tests/AudioEngineHarness/RuntimeParameterGuardTests.cpp` | B-03 の回帰 | WIP 採用（§7-3） |
| `.github/scripts/isr-verify-evidence-consistency.ps1` | B-11 の証跡整合検査 | WIP 採用 |

`src/convolver/ConvolverStopProbe.h` は §6 の理由で除外する。

## 6. 除外するもの

### 6.1 ConvolverStopProbe 系

`ConvolverStopProbe.h` と、それを参照する 5 ファイルの計測行（`LoadPipeline.cpp`、`LoaderThread.cpp`、`StateAndUI.cpp`、`Threading.cpp`、`ReleaseResources.cpp`）は**取り込まない**。

2026-11-14 の main 統合では、混在 3 ファイルに残っていた 6 ハンク（probe 系 5、BUG-1 1）を意図的に除外した記録が `AGENTS.md` にある。probe 系は同じ理由で全ファイルから外す。計測は既定 OFF の診断専用で、製品の挙動を変えない。混ぜると統合の差分が読みにくくなる。

### 6.2 リポジトリ直下の成果物

- `resf_raw.json` は測定の生データで、置き場所は `doc/` 配下が妥当。ルートには置かない。
- `.openchamber/notes-todos.json` は OpenChamber のローカル状態で、upstream も追跡していない。リポジトリに入れない。

### 6.3 `doc/` の未追跡 338 ファイル

過去の報告書と `ConvoPeq.md` のスナップショットである。**履歴として保全し、現行ソースとしては取り込まない**。とくに `doc/work113/20261111-0400_更新ソース/ConvoPeq.md` などの結合 Markdown は当時の生成物で、現行のソースではない。現行の正はリポジトリ直下の `ConvoPeq.md` と、その生成元コミットである（§8）。

## 7. 判断が必要な点

比較では決めきれない論点が 3 つ残った。いずれも「どちらも動く」ため、方針の指定が要る。

### 7-1. CacheManager の上限値

WIP の B-02 は、時間領域記述子の上限に `IRLoadAdmission` の凍結値（8ch / 2^21+1 標本）を使う。upstream の C-05 は 8ch / 2^28 標本で、部分的な記述子（channels だけ非 0 など）と全体サイズの整合も見る。

upstream の検査のほうが対象は広いが、標本数の上限は WIP のほうが厳しい。両方を足すと「upstream の広い検査 + WIP の厳しい上限」になる。ただし 2^21+1 を超える正当なキャッシュが存在し得るなら、WIP の上限は過剰になる。**既定案は upstream の検査を採り、上限だけ WIP の値に寄せる**。採否は実データの分布を見て決める。

### 7-2. 学習器の状態保護（B-07 と C-03）

同じ競合を 2 つの設計で解いている。

- WIP の B-07 は `LearnerStateAuthority` を新設し、復旧要求と公開スナップショットを 1 つの所有権で直列化する。ワーカーは世代境界で短時間だけロックを取る。
- upstream の C-03 は `optimizerMutex` を追加し、`getState` / `setState` / `applyPhaseParams` / `handleModeSwitch` を同じ mutex で囲む。

upstream の案は既に main で動いており、テストもある。WIP の案は設計としては広いが、未検証の新規クラスを伴う。**既定案は upstream（C-03）を採り、B-07 の新規クラスと専用テストは取り込まない**。B-07 が解こうとした競合が C-03 で塞がりきっていない場合は、別 Issue として起票する。

### 7-3. B-02 / B-03 の回帰テスト

`CacheManagerHeaderValidationTests` と `RuntimeParameterGuardTests` は、WIP の実装（上限値・非有限値の扱い）を前提に期待値を持つ。実装を upstream に寄せると、期待値の一部が合わなくなる。

テスト自体は有用なので、**実装を決めた後に期待値を合わせて残す**。捨てるのは、対応する実装を採らないと決めた場合だけにする。

## 8. ConvoPeq.md の再生成

最新 main から再生成し、生成元コミットを記録した。詳細は `ConvoPeq_provenance.md` に分けた。

| 項目 | 値 |
| --- | --- |
| 生成元コミット | `ce0ac515a57c0d6e04dd880d1f8fdda083f6c21d` |
| 生成日時 | 2026-11-23 18:38:41 |
| サイズ | 7,620,264 B（旧版は 6,578,349 B、2026-11-08 生成） |
| SHA-256 | `06a74951f9dd7bed60570c6897896362bc26068899b8e446085176654b39be18` |
| 分割 | 4 ファイル（最大 1,999,985 B）。再結合がバイト一致することを確認 |
| 秘密情報走査 | 0 件 |

生成は `origin/main` を checkout した独立クローンで行った。出力の見出しは作業ディレクトリ名を採るため、ディレクトリ名を `ConvoPeq` にしてから実行している。

## 9. 統合の進め方（次段階）

1. 保全ブランチ `wip/preserve-20261123` は復元専用として凍結を続ける。統合作業で書き換えない。
2. 統合は `4d476d5c` と `ce0ac515` を親に持つ一時ブランチで行い、保護ツリーには触れない。作業場所は使い捨ての worktree にする。
3. §4 の分類に従い、**upstream 採用 12 は upstream のまま**、**統合 18 は WIP 固有部分だけを移植**する。移植の単位はファイルではなく変更の意味とする。
4. §7 の 3 点を決めてから、`CacheManagerHeaderValidationTests` などの期待値を確定する。
5. 統合後は Release の全ターゲットビルドと `ctest`、Debug の既知 2 件、ASan 対象、CI の全ジョブを確認する。とくに自動マージされた 17 ファイルは、テキストが衝突しなかっただけで正しいとは限らないため、差分を読む。

## 10. 参照

- 保全の記録: `doc/work113/20261123-1831_WIP保全/report.md`
- 残課題の全体: `doc/work113/20261123-1815_残課題整理/report.md`
- 生成物の記録: `doc/work113/20261123-1842_統合比較/ConvoPeq_provenance.md`
- 保全ブランチ: `wip/preserve-20261123`（`adfb16f6`）
- 比較用クローン: `C:\VSC_Project\integ\ConvoPeq`（`ce0ac515`）
