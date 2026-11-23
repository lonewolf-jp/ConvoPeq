# 残り 13 件の実装と検証（B-03〜B-15）／B-16 判定変更

## §0 位置づけ

Owner の承認（B-03 ほか各段階の実装許可、B-10 案 B、B-11 第 2 段まで、B-08 は設計提示を条件に許可、B-05 は ConvoPeq 側限定、B-16 は判定変更のみ）に基づき、実装と検証を行った。

段階は提示時の区分に合わせた。

| 段階 | 対象 | 状態 |
|---|---|---|
| 提示 1 | B-03 | 実装・検証済み |
| 提示 2 | B-04、B-05、B-06、B-07、B-08、B-09、B-11 | 実装・検証済み（B-11 は第 2 段まで） |
| 提示 3 | B-10、B-12、B-13、B-14、B-15 | 実装・検証済み（B-10 は実行時検証で前提が否定され、判定を訂正） |
| 別枠 | B-16 | 判定変更のみ（コード変更なし） |

本書は「テストで確認したこと」と「未検証のこと」を分けて書く。実行時再現・メモリ実測を行っていない項目は、その旨を明記する。

変更は未コミットのまま作業ツリーに置いている。main へのマージ、コミット、リリース、配布、公開告知、README／RELEASE_NOTES／manual の変更、RES-06 の解除は行っていない。

Owner は本報告を受領した。実装作業は完了として扱い、最終受入れに必要な一部確認（TSan／LSan、B-10 と B-06 の GUI 目視、実行時障害の再現・メモリ使用量の実測、RES-06）は未完了として維持する。マージ・コミット・リリース等の権限は新たに発生していない。ctest と ASan の成功は TSan・LSan の代替証明にはならない。

## §1 B-03: mixed f1/f2 の非有限値拒否

変更は `src/convolver/ConvolverProcessor.Runtime.cpp` の 2 つの setter に、既存の `convo::numeric_policy::isFinite` ゲートを追加しただけである。非有限値は格納せず、以前値を維持する。`ConvolverProcessor.h` の宣言コメントに契約を追記した。

回帰テストは `src/tests/AudioEngineHarness/RuntimeParameterGuardTests.cpp`（新規）に置いた。NaN／±Inf を set して getter が以前値を返すこと、有効値が通ること、`setState`（セッション復元の経路）に NaN を流しても壊れないことを確認する。

## §2 提示 2

### 2.1 B-04: WAV メタデータの事前拒否

`src/InputInspection.cpp` / `.h` を新設し、RIFF/WAVE のチャンク宣言長が実ファイル残量を超える場合に拒否する `inspectRiffChunkBounds` を追加した。`createReaderFor` を呼ぶ 3 経路（`IRConverter::loadAudioFile`、`LoaderThread`、`ResampleAndFallback`）の直前に置いた。JUCE 本体は改修していない。

判定は「宣言長 > そのチャンクの残りバイト数」の 1 条件だけである。RF64 の `0xFFFFFFFF` は ds64 側の値を使うため対象外とし、RIFF 以外のファイルは素通しする（他フォーマットの正当な入力を落とさないため）。

### 2.2 B-05 / B-06: XML のサイズ・深さ・DOCTYPE 拒否

同じ `InputInspection` に `inspectXmlFile` と `parseXmlFile` を追加した。上限はサイズ 8 MiB、要素深さ 64、DOCTYPE／ENTITY の禁止である。

- サイズは、プリセット・セッション・学習状態の実測最大（数百 KiB）に対して十分な余裕として 8 MiB を置いた。
- 深さ 64 は、JUCE の `readChildElements` が無制限に再帰するため、アプリ側で先に止めるための値である。正当な ConvoPeq ファイルの深さは 1〜3 程度である。
- DOCTYPE／ENTITY の禁止は、外部実体展開（XXE 相当）と entity expansion を入口で断つためである。ConvoPeq が書く XML に DTD は無い。

適用箇所は 6 か所（`DeviceSettings.cpp` 4、`MainWindow.cpp` 1、`NoiseShaperLearner.cpp` 1）。`MainWindow` のプリセット読み込みでは、拒否時に `NativeMessageBox` で理由を表示し、ログにも残す（B-06 の「拒否時の挙動」）。

JUCE 本体の改修は行っていない。したがって、深さの厳密な制限は「事前スキャンで拒否する」方式であり、パーサ側の再帰に上限を入れたわけではない。

拒否時の挙動は経路で異なる。プリセット読み込みは `NativeMessageBox` で理由を表示し、ログにも残す。設定・学習状態の読み込みは、そのファイルを使わずに既定値で継続し、`Failed to parse noise shaper state file.` をログに残す（アプリは停止しない）。実行時の確認として、学習状態ファイルを差し替えて Release のアプリを起動し、DOCTYPE・ENTITY・深さ 200・壊れた XML の 4 条件が拒否され、正常 XML だけが受理されることを確認した（§5.2）。

### 2.3 B-07: 学習器保存のスナップショット固定

`NoiseShaperLearner::publishGenerationResult` で 1 回だけ `getState` を取り、保存ジョブと UI 通知へ値渡しする形に変えた。`saveLearnedState` は `const State&` を受けるようになり、ジョブ側での再読をやめた。これで保存される mean と bestCoefficients が同一世代になる。

データレースそのもの（`CmaEsOptimizer` に mutex が無いこと）は、保存経路から無ロック読みを消したことで当該経路では解消する。ただし学習スレッドの `update()` と他の `getState` 呼出（`AudioEngine.CtorDtor.cpp:441`）が並行する可能性は残る。ここは未検証である。

### 2.4 B-08: キャッシュ総量上限（設計提示）

Owner の指定 5 点に答える。

| 項目 | 内容 |
|---|---|
| 上限値と根拠 | 1 GiB（`convo::irload::kMaxIRLoadBytes` を再利用）。キャッシュは IR データの派生物であり、「同時に保持してよい IR 量」として製品が既に定めている値を超えてディスクに残す利点がない。新しい値を発明しない |
| 総量の算定方法 | キャッシュディレクトリ直下の `*.bin` のうち、命名が `<16 桁 hex>_<fft>.bin` に一致するものだけを対象に `getSize()` を合計する。他のアプリや他バージョンのファイルは数えない |
| 上限到達時の処理 | 最終更新の古い順に削除し、合計が上限以下になった時点で止める。`isEntrySafeToDelete` が拒否する使用中エントリは残す。Windows で mmap 中のファイルは `deleteFile` が失敗するため、巻き込まない |
| 既存キャッシュの扱い | `CacheManager` のコンストラクタで 1 回剪定する。したがって過去セッションで溜まった分は次回起動時に上限まで減る。冪等で、上限以下なら何もしない |
| 回帰テスト | 一時ディレクトリに正しい命名の 1 KiB ファイル 3 個と、命名が一致しない 2 個を置き、`pruneDirectory(1024)` で古い 2 個だけが消え、命名不一致が残ることを確認する（C-11a/b/c） |

上限を 1 GiB とした結果、典型的な 1 秒 IR（2ch）のエントリは約 1.5 MB なので数百件を保持できる。最大級の IR（2²¹+1 サンプル、2ch）は約 67 MB なので 15 件程度で上限に達する。セッション内の件数上限 10 件（`evictLRU`）はそのまま残した。

### 2.5 B-09: 適応係数の非有限値ゲート

受口 2 つ（`AudioEngine.Learning.cpp` の `setAdaptiveCoefficientsForSampleRate` と `...AndBitDepth`）で、非有限の要素をスキップして以前値を維持する。全要素が非有限なら書き込み自体を行わない。`NoiseShaperLearner::loadLearnedState` では、`learned_state.xml` の `BestCoefficients` と `CmaMean` の非有限値を中立値 0.0 に落とす（`clampCoeff` と同じ扱い）。

回帰テストは `RuntimeParameterGuardTests` に追加した。実エンジンを起動し、NaN／Inf を混ぜた係数集合を渡して、読み戻しが変化しないことを確認する。

### 2.6 B-11: ISR 証跡（第 2 段まで）

第 1 段として `FailureHandler::handleFailure` を、ファイルの存在ではなく内容で判断するように変えた。`hb_violation_report.json` の `status` が `ok` 以外（不明を含む）なら Quarantine、`shutdown_trace.json` の `verified` が `false` なら SafeMode とする。これで「同じ run で ok と Quarantine が併存する」状態は消える。

第 2 段として、実測の生成元を接続した。

- `ISRDebugRuntime` に `HBVerifierRuntime` を持たせ、`emitCIArtifacts` から `hb_violation_report.json` を生成する。内容は `status`（ok／violation）、`violations`、4 シナリオの `scenarioResults` で、CI の `isr-verify-v5.ps1` が要求する形と一致する。
- `hb_graph_trace.json` は既存の `HBTraceRuntime::emitTrace` が実イベントを書く。従来は CI モードだけで呼んでいたが、Release 以外では常に出すようにした。
- 実測の生成元が無い `payload_tier_report.json` と `retire_latency_report.json` は、`status:"not_measured"` と理由を書くようにした。定数の `violations:0` や `withinThreshold:false` を「測定結果」と誤読させないためである。
- 同様に、生成元が動かなかった場合の `closure_graph.json`、`retire_timeline.json`、`shutdown_trace.json`、`hb_graph_trace.json`、`hb_violation_report.json` の placeholder も `status:"not_measured"` と理由に変えた。既存ファイルがある場合は上書きしない（従来どおり）。
- `evidence_manifest.json` に `artifactDetail`（各 artifact の `source` と `measured`）を追加した。`artifacts` は従来どおり文字列配列なので、既存の検証器は壊れない。

CI 契約への影響は次のとおり。

| 対象 | 変更 | 影響 |
|---|---|---|
| `isr-verify-common.ps1` | `status` が `not_measured` / `not_run` の artifact を明示的に失敗させる | CI が生成する証跡はこの status を使わないため、CI の合否は変わらない。アプリ側の placeholder を「合格」と誤読する経路だけが塞がる |
| `isr-run-runtime-evidence.ps1` | `retire_timeline.json` の fallback に `rollbackFlags` を追加（generate 側と同じ形） | 既存の不整合の修正。Release のみのビルドではアプリが retire_timeline を出さず fallback が使われるが、`rollbackFlags` が無いため `isr-verify-v7.ps1` が「必須キー不足」で落ちていた。fallback 利用時の v7 が通るようになる |
| `isr-verify-v2..v10`、`isr-verify-evidence-provenance.ps1` | 変更なし | 必須キーと schema は変えていない |
| `isr-seed-evidence.ps1`、`isr-generate-runtime-evidence.ps1` | 変更なし | CI 側の証跡生成元であり、アプリの出力を入力にしていない |
| アプリの `evidence/` 出力 | 実測または `not_measured` | CI と同じディレクトリ・同じファイル名を書くため、アプリをリポジトリ直下で動かすと CI 生成物を上書きし得る。この衝突自体は今回の範囲外であり、未解決のまま残す |

CI 契約の変更は 2 ファイル・15 行の追加だけで、削除は無い。具体的な差分は次のとおり。

```diff
--- a/.github/scripts/isr-verify-common.ps1
+++ b/.github/scripts/isr-verify-common.ps1
@@ -127,5 +127,16 @@ foreach ($key in $RequiredKeys) {
     Assert-HasProperty -Object $data -Name $key
 }
 
+# ★ B-11: placeholder（未測定）を証跡として受理しない。
+#   アプリ側のエクスポータは、実測していない artifact に status=not_measured /
+#   not_run を付けて書き出す。これを合格にすると「生成できた = 安全」と誤読されるため、
+#   検証器は明示的に失敗させる。CI のシナリオ生成物はこの status を使わない。
+if ($data.PSObject.Properties.Name -contains 'status') {
+    $statusValue = [string]$data.status
+    if ($statusValue -eq 'not_measured' -or $statusValue -eq 'not_run') {
+        throw "Artifact '$ArtifactName' is a placeholder (status=$statusValue): evidence was not measured in this run."
+    }
+}
+
 Write-Host "[PASS] $ArtifactName schema=$Schema"
 return $data
```

```diff
--- a/.github/scripts/isr-run-runtime-evidence.ps1
+++ b/.github/scripts/isr-run-runtime-evidence.ps1
@@ -168,6 +168,10 @@ function New-FallbackArtifactObject {
                 epochMode        = "shared"
                 rollbackMode     = "shared"
                 rollbackReady    = $true
+                # ★ B-11: isr-verify-v7.ps1 は rollbackFlags を必須にする。
+                #   generate 側と同じ形に揃える（Release のみのビルドで fallback が
+                #   使われた場合に v7 が「必須キー不足」で落ちるのを防ぐ）。
+                rollbackFlags    = @{ global = $true; publicationOnly = $false; crossfadeOnly = $false; retirePathOnly = $true }
                 totalTransitions = 1
                 laneCounters     = @{ rtIntent = 1; coordination = 1; epoch = 1; reclaim = 1; quarantine = 0 }
                 generatedAtNs    = $GeneratedAtNs
```

`isr-verify-common.ps1` の追加は、全検証器が共有する入口に置いた。`status` を持たない artifact（CI のシナリオ生成物の一部）には影響しない。`isr-run-runtime-evidence.ps1` の追加は、`retire_timeline.json` の fallback が generate 側と形が違っていた既存の不整合を揃えるもので、B-11 の placeholder 化とは独立である。

「検証器が成功したこと」と「実測証跡が正しいこと」は別である。`status:"not_measured"` を導入したのは、生成に成功したことを安全性の証明と取り違えないためである。ISR 安全性の一般保証にはならない。

## §3 提示 3

### 3.1 B-10: 最小位相変換のスキップ通知（案 B）

実装は行ったが、**前提が成立しないことを実測で確認した**。

#### 実装した内容

- `ConvolverProcessor` に実効位相モード（`effectivePhaseModeForLastLoad`）と警告文（`lastPhaseWarning`、`lastErrorMutex_` 保護）を追加した。
- `LoaderThread` の変換分岐で、実際に適用したモード（Minimum／Mixed／AsIs）を記録し、要求と異なる場合に警告文を設定する。
- `applyComputedIR`（`IRConverter` とキャッシュの経路）では、位相変換を行わないため実効値を AsIs として公開する。
- `ConvolverControlPanel` の IR 情報ラベルに `Phase: <実効値>` を併記し、警告がある場合は橙色で表示する（ツールチップに詳細）。
- 回帰テスト PB-1／PB-2／PB-3 を追加した（§5.2）。

#### 前提が成立しないことの確認

`convertToMinimumPhase`（`ResampleAndFallback.cpp:382`）の `fftSize > 8388608` 分岐は、呼出元（`LoaderThread.cpp:903`）が渡すバッファが `targetLength` に切り詰められているため到達しない。

- `LoaderThread.cpp:769-771` は `targetLength = computeTargetIRLength(...)` を計算し、`stepTrimmed.setSize(channels, targetLength)` してから変換する。
- `computeTargetIRLength`（`StateAndUI.cpp:997-1023`）は `min(sr × targetIRTimeSec, MAX_IR_LATENCY)` を返し、`MAX_IR_LATENCY = 2097152`（2²¹）。
- したがって変換入力 N ≤ 2²¹、`nextPowerOfTwo(N × 4) ≤ 2²³ = 8388608`。分岐条件は `> 8388608` なので成立しない。
- 2²¹+1 サンプルの WAV を実際にロードする統合テスト（PB-2）でも、Minimum が適用され警告は出なかった。
- 混合位相側の 2 つの上限（`MixedPhase.cpp:171`、`:744`）も同じ理由で到達しない。`convertToMinimumPhase` の呼出元は `LoaderThread.cpp:903` の 1 か所だけである。

この結果を踏まえ、B-10 は「非バグ（前提不一致）」へ変更する。実装した `Phase:` 表示と警告は次の理由で残す。

- `Phase:` 表示は、要求モードと実際に適用されたモードを常に併記する観測情報であり、スキップの有無に関係なく UI と実態の一致に寄与する。
- 警告は、将来 `MAX_IR_LATENCY` を上げるか上限値を下げた場合に silent skip を防ぐ防御になる。到達不能であることは PB-2 が固定しており、条件が変われば同テストが落ちる。

防御的実装を残すか revert するかは Owner の裁量に委ねる。revert する場合の対象は、`ConvolverProcessor` の実効値・警告、`LoaderThread` の記録、`ConvolverControlPanel` の表示、`LoadPipeline` の AsIs 公開である。

### 3.2 B-12: テストの独立参照と音声ケース

`TailLayerIntegrityTests` の H 系に、IR 自身のエネルギーを独立参照とする絶対帯チェック（H-6、±6 dB）を追加した。従来の相対比較（block 256 の実測を基準）は残し、block 間の一貫性を見る役割に限定した。H-4 も `> -60 dB` だけでなく IR エネルギーとの比較を加えた。

F-5 には block 524288 の実音声ケース（`F-5-audio`）を追加した。形状だけでなく、出力テールのエネルギーが IR と一致することを見る。

### 3.3 B-13: ASIO ブラックリストの無言失敗

既定リストの作成に失敗した場合にユーザー領域（`userApplicationDataDirectory/ConvoPeq/asio_blacklist.txt`）へフォールバックし、それも失敗したら `NativeMessageBox` で通知するようにした。`loadFromFile` の戻り値も見る。

実行確認: 実行ファイルを `icacls /deny <user>:(W)` で書込不可にしたディレクトリへ置いて起動した。アプリは正常終了し（ダイアログで停止しない）、ユーザー領域に既定の 4 行が書かれた `asio_blacklist.txt` が用意され、exe 隣には作成されなかった。所要は初回 5.6 秒、強制再ビルド後のバイナリで 25.3 秒だった。テスト後にディレクトリとユーザー領域のファイルは元に戻した。

### 3.4 B-14: FileLogger の所有

`--cli-log-file` の `juce::FileLogger` を `MainWindow` のメンバで保持し、`release()` をやめた。デストラクタの先頭で `Logger::setCurrentLogger(nullptr)` を呼んでから破棄する。

### 3.5 B-15: 二重 tanh の解消

`publishGenerationResult` の再 `tanh` を削除し、引数を parcor（tanh 済み反射係数）として扱うようにした。非有限値だけは 0.0 に落とす。エンジンへ渡す `storeLearnedCoeffs` も同じ値になる。

既存の `learned_state.xml` は二重 tanh 済みの値を持つ。移行は行っていない（次回学習で上書きされる）。

### 3.6 段階を跨ぐ依存

- B-04 は B-01 が追加した admission の位置（`IRConverter::loadAudioFile` の `createReaderFor` 前）に相乗りする。B-01 が無い状態では、IRConverter 経路だけ検査位置が定まらない。
- B-06 は B-05 の共通ヘルパ（`convo::safeinput::parseXmlFile`）に依存する。B-05 を入れずに B-06 だけを入れると、プリセット読込だけが独自検査を持つ形になり、二重実装になる。
- B-11 の第 2 段は、`ISRDebugRuntime` の `HBVerifierRuntime` 接続と、`isr-verify-common.ps1` の placeholder 拒否が対になっている。片方だけでは「実測を出したが検証器が読まない」または「placeholder を検証器が合格させる」状態になる。
- B-12 の H-6 は B-12 自身の独立参照導入に依存する。既存の H-1/H-2（相対比較）はそのまま残した。
- B-10 の表示整合は B-10 の実効値公開に依存する。B-10 を外すと `applyComputedIR` の AsIs 公開も意味を失う。

## §4 判定の再評価（B-16 / B-10）

### 4.1 B-16: build_identity_gate の pathspec — 非バグ

git 2.56.0 で次の 3 コマンドを実行し、pathspec の挙動を実測した。

```
git diff HEAD --name-only -- 'src/*.cpp'            → src/convolver/... と src/tests/... を含む
git diff HEAD --name-only -- ':(glob)src/*.cpp'     → src/*.cpp 直下のみ
git diff HEAD --name-only -- ':(glob)src/**/*.cpp'  → 全階層を含む
```

git の pathspec は既定で `*` が `/` を跨ぐ。`/` を跨がせないのは `:(glob)` を付けたときである。したがって `src/*.cpp` はサブディレクトリを検出できており、「production src/ diff = 0 を誤って出す」という当初の不具合仮説は成立しない。

B-16 を「非バグ（前提不一致）」へ変更する。コード変更は行わない。N-01〜N-12 の判定は変更していない。

判定変更の記録は、元の検証報告書（`doc/work113/20261109-1855_バグ検証/report.md`）の B-16 節と 2 つの一覧表にも追記した。履歴を消さずに、変更日・根拠・適用範囲を併記する形にしている。

この実測は git 2.56.0 の挙動であり、他のバージョンへ一般化しない。

なお、`git diff HEAD` は未追跡ファイルを出さないため、新規の未追跡 `.cpp` は「diff = 0」に寄与しない。これは B-16 とは別の小さなギャップであり、必要なら別項目として起票する。

### 4.2 B-10: 最小位相の >2²¹ 無言スキップ — 非バグ

実行時検証で前提が否定された。詳細は §3.1 に書いたとおりで、変換入力が `targetLength`（≤ `MAX_IR_LATENCY` = 2²¹）に切り詰められてから `convertToMinimumPhase` へ渡るため、`fftSize > 8388608` の分岐には到達しない。2²¹+1 サンプルの WAV を実際にロードする統合テスト（PB-2）でも、Minimum が適用され警告は出なかった。

B-10 を「非バグ（前提不一致）」へ変更する。実装した `Phase:` 表示と警告は、観測情報および将来の防御として残す（revert の判断は Owner に委ねる）。

判定変更の記録は、元の検証報告書の B-10 節と一覧表にも追記した。

## §5 検証結果

### 5.1 全体（Release）

`ctest -C Release` で 58 件すべてが PASS した（B-04 の統合テスト RG-1／RG-2 と B-10 の実測 PB-1〜PB-3 を含む）。

最終確認として、変更した 26 ファイルの mtime を更新してから全 197 ターゲットを強制再ビルドし（コンパイルエラー 0）、`build/` で `ctest -C Release` を再実行した。結果は **58/58 PASS（375.58 s）** である。強制再ビルドに踏み切った理由は §5.7 に記載する。初回の実行は 349.45 s だった。

内訳のうち本作業に関係するものは次のとおり。

| テスト | 結果 | 所要 |
|---|---|---|
| InputInspectionTests（新規・B-04／B-05） | Passed | 0.20 s |
| CacheManagerHeaderValidationTests（B-02＋B-08） | Passed | 0.34 s |
| TailLayerIntegrityTests（B-12） | Passed | 3.57 s |
| AudioEngineHarness（B-01／B-03／B-04／B-09／B-10） | Passed | 324.45 s |
| HeadlessAudioPathVerification（B-13／B-14 の影響確認） | Passed | 6.53 s |

HeadlessAudioPathVerification は、作業中に一度だけ 30 秒のタイムアウトで落ちた。同じバイナリを直接実行すると 5.4〜6.0 秒で終了し、負荷のない状態の ctest でも 6.61 秒で通っている。落ちた回はいずれも短時間に複数回起動・強制終了を繰り返した直後で、ログは `prepareToPlay: enter sr=192000` の直前で約 10 秒止まっていた。音声デバイス初期化の待ちであり、コード変更に起因する再現性のある失敗ではないと判断した。ただし原因を特定したわけではないので、環境依存の可能性として残す。

### 5.2 項目別

| ID | 確認内容 | 結果 |
|---|---|---|
| B-03 | B03-1（初期値が有限）、B03-2a/2b（NaN 拒否）、B03-3a/3b（±Inf 拒否）、B03-4a/4b（有効値の受理）、B03-5（`setState` 経由の NaN で以前値維持） | すべて PASS |
| B-09 | B09-1（基準係数が有限）、B09-2（NaN／Inf を混ぜた係数集合が bank を変えない） | すべて PASS |
| B-04 | R-1（正常 WAV 受理）、R-2（data 宣言長の過大申告を拒否）、R-3（RIFF 以外は対象外）、R-4（bext 宣言長の過大申告を拒否） | すべて PASS |
| B-04（統合） | RG-1（非同期ローダ経路が細工 WAV を拒否し IR 未適用）、RG-2（プレビュー経路が同じ理由で拒否）。単体実行で `[B04] checkRiffGuardRuntime: PASS (loader / preview reject crafted RIFF)` を確認 | すべて PASS |
| B-05 | X-1（正常 XML 受理）、X-2（深さ 200 を拒否）、X-3（DOCTYPE 拒否）、X-4（ENTITY 拒否）、X-5（壊れた XML 拒否）、X-6（9 MiB を拒否） | すべて PASS |
| B-05（アプリ実行時） | `%APPDATA%\ConvoPeq\noise_shaper_learn.xml` を差し替えて Release のアプリを起動し、`DeviceSettings::loadNoiseShaperState` の拒否ログを確認。DOCTYPE＝拒否、ENTITY＝拒否、深さ 200＝拒否、壊れた XML＝拒否、正常 XML＝受理（対照） | 5 条件すべて期待どおり（アプリは各回 exit 0 で継続） |
| B-08 | C-11a（既定上限では何も消えない）、C-11b（1024 バイト上限で古い 2 件だけ削除）、C-11c（命名が一致しないファイルは残す） | すべて PASS |
| B-12 | H-1/H-2（block 間の相対一貫性）、H-4（3 秒 IR の絶対帯）、H-6（1 秒 IR の絶対帯 ±6 dB）、F-5-audio（block 524288 の音声一致） | すべて PASS |
| B-10 | PB-1（通常長の IR で Minimum が適用され警告なし）、PB-2（2²¹+1 サンプルでも cap により Minimum が適用され警告なし＝スキップ分岐は到達不能）、PB-3（同期 `loadIR` 経路は実効値 AsIs を公開） | すべて PASS（UI 描画は未検証） |
| B-13 | 実行ファイルを書込不可ディレクトリ（`icacls /deny <user>:(W)`）に置いて起動し、ユーザー領域へのフォールバックを確認。再ビルド後のバイナリで再測すると exit 0・25.3 秒（ダイアログで停止しない）、ユーザー領域の既定リストが存在し、exe 隣には作られなかった。テスト後は環境を元に戻した | PASS |
| B-06 | 専用テストは無い。プリセット読込が `parseXmlFile` を経由し、拒否時に通知とログへ出ることをコードで確認（検査自体は X-1〜X-6 で確認済み） | 回帰なし（通知ダイアログの表示は未検証） |
| B-07 | 専用テストは無い。保存ジョブが値渡しのスナップショットを使い、ジョブ内の `getState` 再読が無いことをコードで確認 | TSan と実行時の世代一致は未検証 |
| B-14 | ASan ビルドのアプリを `--cli-run --cli-exit-ms 1200 --cli-log-file <temp>` で実行。終了コード 0、stderr 0 バイト、ログ 1,116 バイト（`[D167] terminal-release intent` まで記録）、`AddressSanitizer` 報告 0 件。正の対照として `ASAN_OPTIONS=verbosity=1` で `AddressSanitizer Init done` を確認 | PASS（破棄順の安全性）|
| B-15 | 専用テストは無い。呼出元が parcor を渡し、再 tanh が無いことをコードで確認 | 公開値と評価値の数値一致は未検証 |
| B-01（再確認） | I-2／I-5 は B-04 の事前検査が先に弾くため、拒否理由として「memory limit」または「declared chunk size」のいずれかを受理する形に更新した | PASS |

B-04 を入れた結果、B-01 の I-2／I-5 が突く細工 WAV は、FC-FORM-1 の byte 予算に届く前に RIFF 検査で拒否される。バイト予算そのものの境界は、既存の `checkEFPredicateBoundaries`（純述語）が引き続き確認している。

### 5.3 測定上の注意

`TailLayerIntegrityTests` を ASan ビルドと並行して単体実行したとき、`RES-F`（CPU 14.3 ms/s）だけが落ちた。負荷のない ctest では通っている。`RES-F` は CPU 時間の閾値を見るテストであり、並行ビルド中の測定値は採用しない。本書の合否は ctest の結果を正とする。

### 5.4 ASan

`build-asan-local`（RelWithDebInfo + `/fsanitize=address`）で、B-02／B-04／B-05／B-08／B-12 のテストと `AudioEngineHarness` をビルドし、ASan ランタイム DLL を各 exe の隣に置いて実行した（B-10 の commit 直前移動を含む最終コードで再実行済み）。

| テスト | 結果 |
|---|---|
| `CacheManagerHeaderValidationTests`（B-02／B-08） | failures=0 |
| `InputInspectionTests`（B-04／B-05） | failures=0（X-1〜6、R-1〜4） |
| `AudioEngineHarness`（B-01／B-03／B-04／B-09／B-10） | 終了コード 0、全テスト PASS（`B03-1..5`、`B09-1/2`、`B04 RG`、`B10 PB` を含む） |
| `TailLayerIntegrityTests`（B-12） | 27 件失敗。すべて `RES-F`（CPU ms/s）。`RES-F` 以外の失敗は 0 件 |

いずれのログにも `ERROR: AddressSanitizer` の出力は無い。`TailLayerIntegrityTests` は CPU 時間とメモリ増分を閾値で判定する設計のため ASan 下では成立しない（`RES-F` の CPU が 10〜40 ms/s に増える）。同テストの機能判定（H 系・F 系）は Release で確認している。

§5.7 の強制再ビルドの後、ASan 側も変更 26 ファイルを touch して 4 ターゲットを再ビルドし、同じ 4 本を再実行した。結果は同一だった（`CacheManager` failures=0、`InputInspection` failures=0、`AudioEngineHarness` 終了コード 0、`TailLayer` は `RES-F` の 27 件のみで他 0 件、`ERROR: AddressSanitizer` 0 件）。

CMake 側は、ASan 有効時に `CacheManagerHeaderValidationTests`、`InputInspectionTests`、`TailLayerIntegrityTests` の 3 ターゲットへ同じフラグを適用するようにした（`ENABLE_ASAN=OFF` の Release 構成には影響しない）。

アプリ本体（`ConvoPeq` ターゲット）も ASan ビルドし、B-14 の終了経路を実行している（§5.6）。

### 5.5 CI 証跡経路の整合確認（B-11）

`isr-verify-common.ps1` を変更したため、CI 側の 3 経路（generate／seed／run）と検証器の組み合わせを、リポジトリ外の一時ツリーで確認した。リポジトリの `evidence/`（既存 1,105 件・JSON 148 件）を壊さないよう、`.github/scripts` と空の `evidence/` を一時ディレクトリへ複製して実行した。run 経路では `build/` へのジャンクションを用い、終了後に削除した。

| 経路 | 手順 | 結果 |
|---|---|---|
| generate | `isr-generate-runtime-evidence.ps1`（`CONVO_ISR_RUNTIME_RUN_ID=local-check-round2`）→ v2/v3/v4/v5/v6/v7/v9/v10 | 10 artifact を生成。8 本すべて exit 0 |
| generate（負例） | `hb_violation_report.json` の `status` を `not_measured` に差し替え → v5 | exit 1。`Artifact 'hb_violation_report.json' is a placeholder (status=not_measured): evidence was not measured in this run.` |
| seed | `isr-seed-evidence.ps1` → v2..v10（非 strict） | 8 本すべて exit 0 |
| seed（strict） | `ISR_REQUIRE_RUNTIME_EVIDENCE=1` で v5 | exit 1。`Strict evidence mode: seeded artifact is not allowed: hb_graph_trace.json`（既存の意図どおり） |
| run | `isr-run-runtime-evidence.ps1` → v2..v10 | 実行 exit 0。8 本すべて exit 0（下記の 1 件を修正した後） |

run 経路の初回実行で、`isr-verify-v7.ps1` だけが `Missing required property 'rollbackFlags'` で落ちた。原因は `isr-run-runtime-evidence.ps1` の `retire_timeline.json` fallback が `rollbackFlags` を持たないことで、generate 側には最初から入っていた。Release のみのビルドではアプリが retire_timeline を出さず fallback が使われるため、この欠落がそのまま v7 の失敗になっていた。fallback に generate と同じ `rollbackFlags` を追加し、再実行で v2..v10 すべて通ることを確認した。今回の B-11 変更とは独立の既存不整合である。

CI が生成する証跡は `generated`／`ok` を使うため、新しい placeholder 拒否には掛からない。アプリ側の `not_measured` だけが明示的に落ちる。

アプリの `--verify-runtime-evidence` 経路が何を出すかは、Release ビルドで実際に実行して確定させた。リポジトリ外の一時ツリーに `.github/scripts` と空の `evidence/` を置き、そのディレクトリを cwd にして起動した。出力は `evidence_manifest.json`（267 バイト）1 件だけで、`artifacts` は空配列だった。その出力をそのまま検証器に通すと、v5・v6・v7 はいずれも exit 1 になる。

| 検証器 | 結果 | メッセージ |
|---|---|---|
| `isr-verify-v5.ps1` | exit 1 | `Artifact not found: hb_graph_trace.json` |
| `isr-verify-v6.ps1` | exit 1 | `Artifact not found: shutdown_trace.json` |
| `isr-verify-v7.ps1` | exit 1 | `Artifact not found: retire_latency_report.json` |

現行アプリ出力の欠落は証跡上で隠していない。設計段階で挙げた 4 件（`hb_violation_report.json` の `scenarioResults` 欠落、`hb_graph_trace.json` の `eventCount:0`、`shutdown_trace.json` の `verified:false`、`retire_latency_report.json` の `withinThreshold:false`）は、テンプレートを書き出す経路（Debug の `debugArtifacts`）を前提にした指摘である。Release の実行経路では artifact 自体が出ないため、欠落は 4 件にとどまらず全件になる。

- `MainApplication.cpp:72-79` は `EvidenceExporter::exportEvidence(nullptr, 0)` を呼んで即終了する。エンジンを起動しないため、`ISRDebugRuntime::emitCIArtifacts`（B-11 で HB 違反レポートを実測生成する側）は動かない。実測した manifest も `minimalEvidence:true`、`artifacts:[]` で、この読みと一致する。
- `exportEvidence` は Release では manifest だけを書いて戻る（`isRelease` → `minimalEvidence`）。
- Debug では `debugArtifacts` の 8 種類（`closure_graph`、`mutation_fault_trace`、`hb_graph_trace`、`shutdown_trace`、`retire_timeline`、`publication_failure_log`、`publication_progress_log`、`deferred_health`）だけを書く。`hb_violation_report.json`、`retire_latency_report.json`、`payload_tier_report.json` は含まれない。Debug ビルドは作成していないため、この経路の出力はコードで確認した（実行時確認はしていない）。
- したがって `isr-run-runtime-evidence.ps1` は、アプリが出さなかった artifact を自前の fallback で補う。これは既存の設計であり、Release のみのビルドで「合成された証跡」になる理由でもある（スクリプト自身が警告を出す）。

この構造は「アプリの証跡生成モードが実測を伴わない」という意味で残存課題である。B-11 の範囲では、実測が必要な artifact に `not_measured` を明示し、検証器がそれを合格させないところまでを実装した。

### 5.6 B-14: FileLogger 破棄順の ASan 実行確認

`build-asan-local` の `ConvoPeq` ターゲット（RelWithDebInfo + ASan、11-10 00:39 ビルド）を `--cli-run --cli-exit-ms 1200 --cli-log-file <temp>` で実行した。B-14 の変更（`release()` をやめてメンバ保持にし、デストラクタの先頭で `Logger::setCurrentLogger(nullptr)` を呼ぶ）は、この終了経路で実際に通る。

| 確認 | 結果 |
|---|---|
| 終了コード | 0 |
| stderr | 0 バイト（`AddressSanitizer` の出力なし） |
| ログファイル | 1,116 バイト。`[CLI] Auto-exit flush: shutting down`、`[D167] terminal-release intent (app shutdown)` まで記録 |
| ASan が有効であることの対照 | `ASAN_OPTIONS=verbosity=1` で実行すると `AddressSanitizer Init done` を出力 |
| 実行ファイルの依存 | `dumpbin /dependents` に `clang_rt.asan_dynamic-x86_64.dll` |

したがって、ロガーがカレントのまま破棄される種類の不具合（use-after-free）は、この経路では再現しない。ただし MSVC の ASan に LSan 相当が無いため、**リークの有無は依然として未検証**である。

### 5.7 ビルドキャッシュの不整合と強制再ビルド

B-05 の実行時確認を始めたところ、Release のアプリは 9 MiB の XML を拒否したが、DOCTYPE・ENTITY・深さ超過を拒否しなかった。同じソースの ASan ビルドは DOCTYPE を拒否したため、原因はコードより先にビルドツリーにあると判断し、`loadNoiseShaperState` に一時的な診断ログ（`[B05DIAG]`）を入れて切り分けた。

診断の結果、原因は Release ツリーに残っていた**古いヘッダでコンパイル済みの `DeviceSettings.cpp.obj`** だった。`XmlLimits` の既定値が現在のヘッダと一致しない状態でリンクされており、サイズ上限（`maxBytes`）だけが有効で、`rejectDoctype` と `maxDepth` が効いていなかった。ソースの不具合ではない。インクリメンタルビルドがヘッダ依存を取りこぼしたことによる ODR 不整合である。

対処は次のとおり。

- 診断コードを削除した（ツリーの差分は元に戻した）。
- 変更した 26 ファイルの mtime を更新し、全 197 ターゲットを強制再ビルドした（コンパイルエラー 0）。
- `build/` で `ctest -C Release` を再実行し、58/58 PASS（375.58 s）。
- 再ビルド後のバイナリで B-05 の 5 条件、B-11 のアプリ出力、B-13 のフォールバックを再測した（§5.2）。

この不整合は、変更したヘッダ（`InputInspection.h`）と、それを取り込む既存の .cpp の組み合わせで起こり得る。最終検証は強制再ビルド後のバイナリで行っており、本報告の数値はそちらを正とする。

## §6 未検証事項・残存リスク

- 実行時再現とメモリ実測は行っていない。各修正はコード読解と入出力テストによる確認である。
- B-04 の効果は「細工した宣言長を拒否する」ことの確認であり、実際の巨大確保が起きないことのメモリ実測ではない。
- B-05 の深さ制限は事前スキャン方式である。JUCE の再帰自体に上限を入れたわけではないため、スキャンをすり抜ける構文があれば防御できない。DOCTYPE／ENTITY の禁止は入口の文字列検査であり、パーサ側の実体展開を無効化したわけではない。
- 検証中に、Release ツリーの古いオブジェクト（古い `XmlLimits` の既定値でコンパイル済み）が B-05 の DOCTYPE・深さ検査を無効化していた。強制再ビルドで解消し、最終検証は再ビルド後のバイナリで行った（§5.7）。インクリメンタルビルドでは、ヘッダを変えたときに同じ不整合が再発し得る。CI のクリーンビルドでは起こらない。
- B-07 のデータレースは、保存経路から無ロック読みを消しただけである。学習スレッドと `AudioEngine.CtorDtor.cpp:441` の `getState` が並行する可能性は残る。
- B-08 の剪定は起動時 1 回だけである。長時間起動したままのセッションでは、セッション内の件数上限（10 件）だけが効く。
- B-10 のスキップ分岐（`fftSize > 8388608`）は、変換入力が 2²¹ に切り詰められるため到達しない。B-10 は非バグ（前提不一致）へ訂正した。実装した警告は到達不能な防御コードであり、`Phase:` 表示だけが常時意味を持つ。
- B-10 の UI 表示（`Phase:` 行の追記と警告の橙色表示）は、画面での確認を行っていない。実効値の生成・公開・ラベルの文字列組み立てはコードで確認したが、描画結果は未検証である。
- B-10 の同期経路（`IRConverter`・キャッシュ）は位相変換自体を行わない。そこで Minimum を選んでも線形 IR が適用され、UI は Linear と表示する。表示は実態と一致するが、「Minimum を選んだのに最小位相にならない」という機能差は既存のままで、修正していない。
- B-11 は、アプリと CI が同じ `evidence/` に同じ名前で書く構造を残している。CI 契約の変更は「placeholder を合格にしない」1 点だけで、証跡の生成元の一本化は行っていない。アプリの `--verify-runtime-evidence` はエンジンを起動せずに証跡を書き出すため、Release では manifest のみ（実測: `artifacts:[]`）、Debug でも `debugArtifacts` の 8 種類（`hb_violation_report.json` を含まない）にとどまる。実測では、Release の出力を v5・v6・v7 に通すといずれも `Artifact not found` で exit 1 になった（§5.5）。この欠落を合格として扱う経路は塞いだが、埋めてはいない。
- B-12 で block 32768 の相対差（block 256 比で約 2.4 dB、524288 では約 13 dB）を観測した。これは DSP の tail 層のブロック依存であり、今回の範囲では修正していない。H-6 の絶対帯（±6 dB）では block 32768 は通るが、524288 の絶対値は IR と一致する一方で block 256 とは大きく異なる。原因は未調査である。
- B-14 は、破棄順の安全性だけを ASan で確認した（§5.6）。`--cli-run --cli-log-file` の終了経路で `AddressSanitizer` の報告は無く、use-after-free は再現しなかった。一方、リーク検査は実行していない。MSVC の ASan には LSan 相当が無く、この環境では検出できない。
- B-15 の係数変更は、既存の保存ファイルとの互換を保つ移行を行っていない。
- RES-06 は保留を維持している。`(32768, 524288]` の音声経路は B-12 の F-5-audio（524288）で 1 点だけ確認した。

## §7 変更ファイル

| ファイル | 内容 |
|---|---|
| `src/InputInspection.h` / `.cpp` | 新規。RIFF チャンク境界、XML のサイズ・深さ・DOCTYPE 検査 |
| `src/CacheManager.h` / `.cpp` | B-08 の総量上限と起動時剪定 |
| `src/convolver/ConvolverProcessor.Runtime.cpp`、`src/ConvolverProcessor.h` | B-03 の非有限値拒否、B-10 の実効位相モードと警告 |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | B-04 の事前検査、B-10 の実効値記録 |
| `src/convolver/ConvolverProcessor.LoadPipeline.cpp` | B-10 の同期経路での実効値（AsIs）公開 |
| `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp` | B-04 の事前検査 |
| `src/IRConverter.cpp` | B-04 の事前検査 |
| `src/DeviceSettings.cpp` | B-05 の XML 検査（4 か所） |
| `src/MainWindow.cpp` / `.h` | B-05/B-06 の XML 検査と通知、B-13 のフォールバック、B-14 の logger 保持 |
| `src/NoiseShaperLearner.cpp` / `.h` | B-05 の XML 検査、B-07 のスナップショット固定、B-09 の復元サニタイズ、B-15 の二重 tanh 解消 |
| `src/audioengine/AudioEngine.Learning.cpp` | B-09 の係数ゲート |
| `src/audioengine/ISRDebugRuntime.h` / `.cpp` | B-11 の HB 違反レポート生成 |
| `src/audioengine/ISREvidenceExporter.cpp` | B-11 の placeholder 化、manifest の詳細、FailureHandler の内容判定 |
| `src/ConvolverControlPanel.cpp` | B-10 の実効位相表示と警告表示 |
| `.github/scripts/isr-verify-common.ps1` | B-11 の placeholder 拒否 |
| `.github/scripts/isr-run-runtime-evidence.ps1` | B-11 の検証で見つかった `retire_timeline` fallback の `rollbackFlags` 欠落を修正（v7 との整合） |
| `src/tests/AudioEngineHarness/RuntimeParameterGuardTests.cpp` | 新規。B-03／B-09 |
| `src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp` | I-1〜I-6（B-01）、RG-1／RG-2（B-04 の統合）、PB-1〜PB-3（B-10 の実測）を追加 |
| `src/tests/InputInspectionTests.cpp` | 新規。B-04／B-05 |
| `src/tests/CacheManagerHeaderValidationTests.cpp` | C-11（B-08 剪定）を追加 |
| `src/tests/TailLayerIntegrityTests.cpp` | B-12 の独立参照と音声ケース |
| `CMakeLists.txt` | `InputInspectionTests` ターゲット、`RuntimeParameterGuardTests.cpp` の追加 |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | B-03／B-09 テストの登録 |

B-01／B-02 で触れた `src/IRConverter.h` と `src/ProgressiveUpgradeThread.cpp` は `doc/work113/20261109-2008_B01B02実装/report.md` に記載している。`src/convolver/ConvolverProcessor.LoadPipeline.cpp` は B-01（失敗理由の伝搬）と B-10（実効値の公開）の両方で変更している。

いずれも未コミットである。

### 7.1 差分の規模

`git diff --stat`（追跡ファイル）は 26 ファイル、1,541 行追加・219 行削除。新規ファイルは 5 件（`InputInspection.h` 47 行、`InputInspection.cpp` 270 行、`InputInspectionTests.cpp` 215 行、`RuntimeParameterGuardTests.cpp` 149 行、`CacheManagerHeaderValidationTests.cpp` 348 行）。

内訳の大きいものは、`IRConverter.cpp`（B-01 の admission と例外封じ）、`CacheManager.cpp`（B-02 の検証と B-08 の剪定）、`IRLoadAdmissionTests.cpp`（B-01 の I 系と B-04 追随）、`ISREvidenceExporter.cpp`（B-11 の placeholder 化と manifest）、`MainWindow.cpp`（B-05／B-06／B-13／B-14）である。

### 7.2 段階別サマリ

| 段階 | 成果物（コード） | 差分 | テスト | 残存リスク |
|---|---|---|---|---|
| 提示 1 | `ConvolverProcessor.Runtime.cpp`、`ConvolverProcessor.h`、`RuntimeParameterGuardTests.cpp` | 2 ファイル + 新規 1 | B03-1〜5 | なし |
| 提示 2 | `InputInspection.h/.cpp`、`CacheManager.h/.cpp`、`DeviceSettings.cpp`、`MainWindow.cpp/.h`、`NoiseShaperLearner.cpp/.h`、`AudioEngine.Learning.cpp`、`ISRDebugRuntime.h/.cpp`、`ISREvidenceExporter.cpp`、`isr-verify-common.ps1`、`isr-run-runtime-evidence.ps1`、`InputInspectionTests.cpp`、`CacheManagerHeaderValidationTests.cpp` | 11 変更 + 新規 3 | X-1〜6、R-1〜4、C-11a/b/c、B09-1/2、B-13 実行確認、B-05 アプリ実行時 5 条件、CI 3 経路、アプリ出力×v5/v6/v7 | B-07 の並行 `getState`、B-11 の evidence 衝突とアプリ出力の全件欠落 |
| 提示 3 | `ConvolverProcessor.LoaderThread.cpp`、`LoadPipeline.cpp`、`ConvolverControlPanel.cpp`、`TailLayerIntegrityTests.cpp`、`MainWindow.cpp/.h`、`NoiseShaperLearner.cpp` | 4 変更 | H-1/H-2/H-4/H-6、F-5-audio、PB-1/PB-2、B-14 ASan 実行、ctest 回帰 | B-10 の UI 描画（スキップ分岐は到達不能と確認）、B-12 の block 32768、B-14 のリーク検査、B-15 の数値一致 |
| 別枠 | （コード変更なし）B-16 の判定変更記録 | 0 | git 2.56.0 の pathspec 実測 | 実測は git 2.56.0 に限る |
