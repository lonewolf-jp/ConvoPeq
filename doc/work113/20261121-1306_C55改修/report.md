# 検証指摘 55 件の改修（2026-11-21 13:06 JST）

`doc/work113/20261121-1201_詳細検証/report.md` の C-ID 登録簿（C-01〜C-55）を、最新の `origin/main` `a9598d34` に対して改修した。対象は worktree `C:\VSC_Project\fix55`、ブランチ `fix/c01-c55-20261121`。既存の作業ツリー `C:\VSC_Project\ConvoPeq`（`4d476d5c`・未コミット多数）には触れていない。

## 1. 結論

55 件の内訳は次のとおりになった。

| 区分 | 件数 | 内容 |
| --- | ---: | --- |
| コードを修正した | 37 | 高 8、中 18、低 11 |
| 注記・文書で確定した | 5 | 中 3（C-18/C-19/C-21）、低 2（C-42/C-44） |
| 変更なしで確定した | 3 | C-22（判定不能）、C-24（要再測定）、C-39（条件付き） |
| 非バグ・設計リスク | 10 | C-26、C-47〜C-55。バグとして扱わない |
| **合計** | **55** | |

- 修正した: コードの挙動が変わる。
- 注記・文書で確定した: 設計制約をコード注記または利用者向け文書に明記した。
- 変更なしで確定した: 根拠を示して現状維持と判断した。

判定不能の C-22 は無理に直していない。

## 2. 作業の前提

- 起点は `origin/main` の `a9598d34751794c31d4da23048baed0c8e679e24`（2026-11-21 09:33）。
- 独立 worktree を作り、未コミット変更を持ち込んでいない。`git status` は作業開始時点で空だった。
- ビルドは MSVC ではなく Intel oneAPI 2026.2（icx）Release。`build.bat` ではなく cmake を直接呼び、`vcvars64` と `MKLROOT`／`IPPROOT`／`LIB` を明示した。
- コード改修のコミットは `712ecb55`、`07446a56`、`8c1668a9`、`cc1bee36`、`5f15bd8c`（親 `a9598d34`）。46 ファイル、1,100 行追加・180 行削除程度（報告書と AGENTS.md を含む）。

## 3. 高 8 件

| ID | 修正内容 | 主な変更箇所 |
| --- | --- | --- |
| C-01 | 公開処理の reader を呼出しローカルに統一し、`exit()` の成否フラグを所有者解放より先に落とした。フラグ自体も atomic にした | `core/RCUReader.h`、`audioengine/RuntimePublicationOrchestrator.{h,cpp}` |
| C-02 | 学習結果の公開で tanh を二重に掛けない（`toParcor()` の出力をそのまま公開） | `NoiseShaperLearner.cpp` |
| C-03 | 最適化器の内部状態へ触る 11 箇所を `optimizerMutex` で直列化した | `NoiseShaperLearner.{h,cpp}` |
| C-04 | 旧ローダーのフラグ解除に世代照合を入れた。新しいロードのフラグを消さない | `convolver/ConvolverProcessor.LoaderThread.cpp` |
| C-05 | mmap 前に全サイズ欄を検証する。`channels × samples × 8 == timeDomainSizeBytes` を含む | `CacheManager.cpp` |
| C-06 | 宣言長に上限（2^24 サンプル）を設け、確保失敗を `bad_alloc` で受ける | `IRConverter.cpp` |
| C-07 | mix／smoothing／mixed クロスオーバーの setter に非有限値の拒否を入れた | `convolver/ConvolverProcessor.Runtime.cpp` |
| C-08 | 学習状態 XML の属性を範囲検査してから公開する | `NoiseShaperLearner.{h,cpp}` |

C-01 は「共有 reader をやめる」と「フラグの公開順を直す」の両方を入れた。`trySubmitImpl` は元からローカル reader を持っていたが、admission には別のメンバ reader を渡していた。メンバを削除して同じローカル reader に揃えた。`exit()` は従来 `ownerThreadToken = 0` の公開後に `rootEnterSucceeded_ = false` を書いていた。解放直後に別スレッドが入場して true を書くと、旧所有者の false が後から上書きし、新しい所有者の入れ子判定が失敗する。フラグを先に落とし、型も `std::atomic<bool>` にした。

C-05 は、従来 `headerSize + dataSize + timeDomainSizeBytes` と実ファイル長の一致だけを見ていた。細工したファイルでは `channels × samples × 8` が `timeDomainSizeBytes` を超え、時間領域のコピーが mmap の終端を越える。`numPartitions × fftSize × 8 == dataSize`、`fftSize` の上限 2^20（積の回り込み防止）、`timeDomain` の整合も加えた。

## 4. 中 23 件

| ID | 修正内容 | 状態 |
| --- | --- | --- |
| C-09 | 容量超過時に deleter を即実行せず、破棄せずに落とす（UAF を避ける）。件数は `overflowDropCount()` | 修正 |
| C-10 | FMA intrinsic を `fmaPd` 系のラッパへ集約し、include 条件と使用条件を一致させた | 修正 |
| C-11 | `makeLPF` に `fc <= 0` の拒否を追加 | 修正 |
| C-12 | 保留 reclaim だけが残る場合も drain を再試行する | 修正 |
| C-13 | `oversamplingType` を範囲検査してから enum へ変換 | 修正 |
| C-14 | 優先度ボタンの初期文言を実設定に合わせ、ヘッドルーム上限の表示とクランプを同じ関数に統一 | 修正 |
| C-15 | CLI の整数・小数パーサを厳密化（空文字、`1+2`、`1.2.3` を拒否） | 修正 |
| C-16 | ブラックリスト書込の失敗をログに残し、CLI ロガーの所有権を `MainWindow` に持たせた | 修正 |
| C-17 | 診断リングの書き込み index を `wc % slots` に（読み出しと一致） | 修正 |
| C-18 | 最小位相変換の空戻りは線形 IR への意図的なフォールバックであると注記 | 注記 |
| C-19 | Bypass 中も L1/L2 を保持するのは意図的なメモリ交換であると注記 | 注記 |
| C-20 | 倍率を 2 の冪へ丸めて保存し、スライダーも同じ値へスナップ | 修正 |
| C-21 | Air Absorption の HF 減衰がブロック長依存である制約を注記 | 注記 |
| C-22 | 具体箇所を特定できず。修正しない | 判定不能 |
| C-23 | ログを書込み中も 8 MiB で打ち切り、超過時に 1 回ローテートする。キャッシュは既存の LRU で有界 | 修正 |
| C-24 | main に原因コードがなく、実測もし直していない | 要再測定 |
| C-25 | HB トレースのイベント数を 20 万件で上限化（超過時は古い半分を捨てる） | 修正 |
| C-27 | MSVC でも AVX2 高速路を有効化（`__FMA__` 未定義対策） | 修正 |
| C-28 | 学習器の確保を nothrow 版にし、OOM の失敗経路を到達可能にした | 修正 |
| C-29 | 集計スロットを「初期化中(2) → タグ書込 → ready(1)」の順にした | 修正 |
| C-30 | 出力先が 2 チャンネル未満なら処理せず clear（契約違反の書込みを防ぐ） | 修正 |
| C-31 | dry バッファが無いときはフェードを即完了させる（Float/Double 両方） | 修正 |
| C-32 | バンド有効化のリセット要求を状態交換の後に出した | 修正 |

C-09 は「容量超過で即解放」を「解放せずに捨てる」へ変えた。このキューは epoch 付きの遅延解放キューであり、enqueue 時点で reader が参照している可能性がある。即時解放は UAF になるため、リーク側へ倒した。この経路は現行構成では到達しない（本系統は `DeferredDeletionQueue`）。

C-20 は `TailLayerMultiplier.h` の丸め規則を setter でも使うようにした。従来はスライダーが 3 を表示してもエンジンは 2 で動いていた。

## 5. 低 14 件

| ID | 修正内容 | 状態 |
| --- | --- | --- |
| C-33 | 準備ブロック長を超える入力は先頭までに制限（作業バッファの範囲外書込みを防ぐ） | 修正 |
| C-34 | shutdown 中の早期 return より前に MMCSS 解除を実行 | 修正 |
| C-35 | `floorNoLibm` が NaN／Inf／巨大値を `long long` へ変換しない | 修正 |
| C-36 | equal-power 近似の結果を [0,1] にクランプ | 修正 |
| C-37 | リング容量の `static_assert` に `Capacity > 0` を追加 | 修正 |
| C-38 | モノラルの Mid で null へコピーしない | 修正 |
| C-39 | 非 2 冪ブロックは前提外。2 の冪では境界を跨がない | 条件付き |
| C-40 | TruePeak のブロック長を準備長へ制限（作業バッファの範囲外書込みを防ぐ） | 修正 |
| C-41 | 登録済みテストは CI の全件 ctest で実行される。A2 プローブの自己テストが最適化で消える問題を修正 | 修正 |
| C-42 | 日本語マニュアルの学習時間を実装値（10/30/60/120/300 秒）へ一致させた | 文書 |
| C-43 | ASan 対象に不足していた 4 テストを追加 | 修正 |
| C-44 | `Debug icx` の記述を非対応へ統一（ビルドガイド） | 文書 |
| C-45 | atomic のドット呼び出しをヘルパーへ置換 | 修正 |
| C-46 | 不要な `_mm256_zeroupper()` を削除（コンパイラが境界で出す） | 修正 |

C-42 は英語版が実装値と一致しており、日本語版だけが 5〜160 分と書いていた。差は 30〜60 倍になる。実装は `targetSeconds` が 10/30/60/120/300 秒である。

## 6. 非バグ・設計リスク 10 件

C-26、C-47〜C-55 はコードを変えない。C-26 はソース走査がすべて再帰であり、指摘の症状が再現しない。C-47〜C-54 はコードと指摘が合わない（訂正）。C-55 は ASIO ブラックリストが `const_cast` に依存する将来互換性のリスクで、現行の不具合ではない。いずれも登録簿の根拠をそのまま残す。

## 7. ビルドとテスト

### 7.1 新規の契約テスト

`Fix55ContractTests` を追加した。外部依存なしで決定的に検査できる 4 件を固定する。

| 対象 | 検査内容 |
| --- | --- |
| C-20 | 丸め結果が常に [2,16] の 2 の冪で、3→4、5→4、7→8 になる |
| C-29 | 集計スロットが空タグのまま ready にならない |
| C-36 | equal-power 近似が [0,1] に収まり、単調で、x=1 で 1 を超えない |
| C-37 | SPSC リングの容量・FIFO・pop 後の再 push |

MSVC 14.52.37025 で単体にコンパイルして実行し、16 項目すべて PASS（終了コード 0）を確認した。

### 7.2 静的ゲート

- `check-audioengine-lint.ps1`: pass（LINT-AE-001〜014）
- `check-src-atomic-dotcall.ps1`: pass
- `check-src-size-mul-cast.ps1`: pass

### 7.3 全ターゲットのビルド

icx 2026.2 Release で全 263 ステップをビルドし、エラー 0（終了コード 0）を確認した。途中で `-j 8` の並列ビルドが `0xc0000142` で落ちたため、`-j 4` で再実行した。これはツール側の並列度の問題で、コードのエラーではない。

### 7.4 ctest 全件

`ctest --test-dir build -C Release` を実行し、63/63 が成功した。新規の `Fix55ContractTests` も含む。最終実行は 181.16 秒である。

最初の実行では 61/63 だった。内訳は次の 2 件で、どちらもコードの退行ではない。

- `TailLayerPerfTests`: 判定が `wall/cpu=1.274 > 1.25` で「5 回とも外乱」と明示された。ビルド直後で他のプロセスが動いていたためである。負荷のない状態で単独実行すると 5.04 秒で成功した。
- `AudioEngineHarness`: A2 プローブの自己テストが `expected 1 allocation, got 0` で失敗した。原因は最適化である。icx `-O2` は、ポインタが外へ出ない `new int(1); delete p;` の対を削除する。2 TU の最小再現で `calls=0` を確認した。テスト側でポインタを volatile なグローバルへ逃がし、確保が消えないようにした。修正後は 145.58 秒で成功した。

### 7.5 レビュー指摘の反映

CodeRabbit のレビューで 10 件の指摘があり、すべて反映した（コミット `07446a56`）。

| 指摘 | 対応 |
| --- | --- |
| `handleLoadError` に世代照合がない | キューへ積む時点の世代を捕捉し、適用前に照合 |
| テール倍率が Air Absorption / Layer Tail Contouring で 2 のまま表示される | `effectiveTailL1L2Multiplier()` を単一規則にし、setter とモード切替で実効値を保存 |
| `FileLogger` の上限はコンストラクタ時だけ | `SizeCappedFileLogger` を追加し、書込み中も上限を適用して 1 回ローテート |
| Bypass のコメントが存在しない `setTailEnabled(false)` を案内 | 記述を削除 |
| `getState` がワーカー専用の非 atomic を別スレッドから読む | 公開済みの atomic（`progress.*`）を読む |
| `setState` が sample と update の間に入り得る | 状態 epoch を導入し、復元をまたいだ世代の update を破棄 |
| CMA mean を反射係数の範囲へ丸めていた | 非有限値だけ 0 にし、有限値は保持（±50 の飽和上限） |
| 公開係数が評価時の `coeffSafetyMargin` で丸められていない | 評価と同じ `clampCoeff` を公開時にも適用 |
| A2 プローブの volatile がポインタ先に付いていた | `const void* volatile` に修正 |
| TruePeak の上限が確保サイズで、再 prepare 後の履歴長と合わない | `stages[0].maxInputSamples` で制限 |

2 回目のレビューで 5 件の追加指摘があり、同じくすべて反映した（コミット `8c1668a9`）。

| 指摘 | 対応 |
| --- | --- |
| CLI ロガーを ~MainWindow で戻すため、~AudioEngine のログが CLI ファイルに残らない | 所有権を `MainApplication` へ移し、`mainWindow.reset()` の後まで保持 |
| 復元が update の後・公開の前に来ると、古い世代が復元値を上書きする | 公開を optimizer mutex 内で epoch 照合してから行う。message thread の bank 書込みも同じ epoch で判定 |
| 評価と公開で `coeffSafetyMargin` を別々に読む | 世代ごとに 1 回読み、評価と公開へ同じ値を渡す |
| 起動時に巨大な旧ログを丸ごと退避する | 末尾 `cap` バイトへ切り詰めてから使う |
| 退避に失敗すると上限を超えて追記し続ける | 退避失敗時はその場で切り詰め、それも失敗したら書込みを捨てる |

3 回目のレビューで 2 件あり、反映した（コミット `cc1bee36`）。

| 指摘 | 対応 |
| --- | --- |
| 公開が拒否されても、呼出し側が先に `bestScore` を下げている | 確定できたときだけ下げる。復元を検出したら復元後のスコアを基準へ戻す |
| 遅延 bank 書込みの epoch 照合と書込みの間に復元が入り得る | 照合と 2 つの engine 書込みを同じ mutex で囲む。autosave 要求はロック外（コールバックが学習器へ戻るため） |

4 回目のレビューで 2 件あり、反映した（コミット `5f15bd8c`）。

| 指摘 | 対応 |
| --- | --- |
| epoch と復元後スコアを mutex の外で別々に読み、新しい epoch と古いスコアの組を掴み得る | 読み取り・基準の更新・sampling を同一の臨界区間へ入れる |
| `latestScore` が改善時しか更新されず「Latest score」の表示が古いまま残る | 評価が有効な世代すべてで公開し、履歴は改善時のみとする。復元で無効になった世代は epoch 照合の下で除外 |

### 7.6 PR #229 の CI

PR https://github.com/lonewolf-jp/ConvoPeq/pull/229 の最新コミット `408c07e9` で、17 件のチェックがすべて成功した（1 件はディスパッチ専用の意図的な skip）。

| チェック | 結果 | 所要 |
| --- | --- | --- |
| RelWithDebInfo ASan (MSVC, /MD) | 成功 | 1 時間 16 分 |
| Debug TSan (Clang, substrate harness) | 成功（陽性対照も成功） | 26 秒 |
| Windows MSVC / Release | 成功 | 23 分 25 秒 |
| Analyze (c-cpp) | 成功 | 46 分 51 秒 |
| isr-verify | 成功 | 2 分 14 秒 |
| lint-audioengine、compliance-list-md、authority-static-checks、authority-runtime-tests、audit-report-verification | 成功 | 各 17〜43 秒 |
| CodeRabbit | 成功（4 回目のレビューで新規指摘なし） | 5 分 34 秒 |

マージ状態は `MERGEABLE` / `CLEAN` である。

## 8. C-ID ごとの検証根拠

| 根拠 | 対象 |
| --- | --- |
| 新規契約テスト（実行） | C-20、C-29、C-36、C-37 |
| 既存テスト（TailLayerMultiplierTests） | C-20 |
| 静的ゲート 3 本（実行） | 追加した atomic 呼出し、確保サイズ計算、禁止コメント語 |
| 全ターゲットのコンパイル（263 ステップ） | コードを変えた 37 件すべて |
| ctest 全件 63/63 | 退行の有無（全ターゲット） |
| コード読解 | C-01〜C-08、C-09〜C-17、C-23、C-25、C-27〜C-35、C-38、C-40、C-43、C-45、C-46 |

実行で確かめたのは、契約テスト、静的ゲート、コンパイル、ctest である。レースの再現（TSan）とメモリ使用量の実測は、この環境では行っていない。CI の Sanitizer ジョブに委ねる。

## 9. 変更ファイル

追跡ファイル 40 件を変更し、2 件を追加した（合計 42 件）。

- 製品コード: `CacheManager.cpp`、`ConvolverControlPanel.cpp`、`ConvolverProcessor.h`、`CustomInputOversampler.cpp`、`DeviceSettings.{h,cpp}`、`IRConverter.cpp`、`LoudnessMeter.cpp`、`MKLNonUniformConvolver.cpp`、`MainApplication.{h,cpp}`、`MainWindow.{h,cpp}`、`NoiseShaperLearner.{h,cpp}`、`OutputFilter.cpp`、`SizeCappedFileLogger.h`（新規）、`TailLayerMultiplier.h`、`TruePeakDetector.cpp`
- audioengine: `AudioEngine.Processing.AudioBlock.cpp`、`AudioEngine.Processing.BlockDouble.cpp`、`AudioEngine.Processing.DSPCoreDouble.cpp`、`AudioEngine.Processing.DSPCoreFloat.cpp`、`AudioEngine.Retire.cpp`、`ISRHB.{h,cpp}`、`RuntimePublicationOrchestrator.{h,cpp}`
- convolver: `ConvolverProcessor.LoaderThread.cpp`、`ConvolverProcessor.Runtime.cpp`
- core: `CommandBuffer.h`、`DeletionQueue.{h,cpp}`、`FadeEngine.h`、`RCUReader.h`、`ReaderMeasureRegistry.h`
- EQ: `EQProcessor.Parameters.cpp`、`EQProcessor.Processing.cpp`
- テスト: `Fix55ContractTests.cpp`（新規）、`PolyphaseGainCandidateRef.h`、`V11EngineRetireProtectionTests.cpp`
- ビルド・文書: `CMakeLists.txt`、`BUILD_GUIDE_WINDOWS.md`、`manual/ADAPTIVE_NOISE_SHAPER_LEARNING_WINDOW_JP.md`、`AGENTS.md`

## 10. 残り

- C-22: 「atomic のメモリ順序が弱い」という指摘に具体箇所がない。`AtomicAccess.h` と主要な公開点を読んだが、個別の欠陥として立証できなかった。修正しない。
- C-24: 学習 XML の再解析が遅いという指摘は、main に原因コードがない。修正前後を同一条件で測っていないため、要再測定のままにする。
- C-39: 非 2 冪ブロック長は設計の前提外である。前提を広げる場合は、遅延境界のブロック欠落を別途設計する必要がある。
- C-19／C-21: 挙動は変えず、制約をコード注記に残した。仕様として保証するなら、時間領域の減衰カーブへ移す設計変更が要る。
- TSan と ASan は手元では回していない。CI（PR #229）の TSan ジョブと RelWithDebInfo ASan ジョブで成功を確認した（§7.6）。手元で再現した結果ではない。

## 11. まとめ

高 8 件はすべてコードを直した。中は 18 件を直し、3 件は設計制約として注記した。低は 11 件を直し、2 件は文書、1 件は条件付きとして現状維持とした。判定不能の C-22 と要再測定の C-24 は、解決扱いにしていない。
