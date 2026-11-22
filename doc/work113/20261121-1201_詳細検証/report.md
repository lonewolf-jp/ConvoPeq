# 統合バグ・残課題リストの詳細検証（2026-11-21 12:01 JST）

対象は `lonewolf-jp/ConvoPeq` の `main` `a9598d34`。2026-11-21 11:43 版の統合リストに載せたコード項目 55 件を、該当ファイルとその呼出し元まで読んで確かめた。結論から書く。

## 1. 結論

55 件の状態は次のとおり。確認できたのが 36 件、条件付きが 4 件、一部だけ確認できたのが 3 件、判定不能が 1 件。残りは要再測定 1 件・訂正 9 件・設計リスク 1 件である。

| 区分 | 件数 | 前回との差 |
| --- | ---: | --- |
| 高 | 8 | 11 から 3 件を中へ移した |
| 中 | 23 | 21 から 3 件を受け入れ、1 件を非バグへ移した |
| 低 | 14 | 変更なし |
| 非バグ・設計リスク | 10 | 1 件を追加した |
| 作業・運用 | 7 | 変更なし |
| 要再確認 | 5 | 変更なし |
| **合計** | **67** | |

以後の評価はこの C-ID を正とし、旧 PRI 番号と H/M/L 番号は参照用に留める。

重要な変化は 3 つある。

1. 添付 HTML の「高」14 件のうち、3 件は指摘そのものがコードと合わない（C-48、C-49、C-50。C-49 の判定は末尾の追記で訂正した）。3 件は通常の経路では到達しないか、影響が 1 ブロックの過渡である（C-30、C-31、C-32）。残る 8 件が高になる。
2. 条件付きとしていた C-01（旧 H-01）と C-04（旧 H-04）は、スレッド構成と join の有無を確認し、競合と上書きの経路を特定した。
3. 未検証だった C-38（旧 L-07）は、モノラル入力の Mid バンドで null 書き込みになることを確認した。通常再生では DSPCore が 2 通道を渡すため到達しない。C-39 は 2 の冪ブロックで成立しない条件付き、C-26 は根拠なしと確定した。

## 2. 検証の方法と限界

`edc17cd9` から `a9598d34` の変更は 20 ファイルである。HTML の項目が参照するファイルのうち、この 20 件に入るのは `src/core/RCUReader.h` だけだった。他のファイルは `edc17cd` と同じ内容なので、HTML の記述が正しければそのまま当てはまる。今回はその記述自体を 1 件ずつ読んだ。

- 見たもの: 該当行、関数全体、呼出し元、スレッドの生成元、状態の公開順。
- 見ていないもの: 実行時の挙動。すべて静的読解である。レースの再現やメモリ使用量の実測はしていない。
- 判定の言葉: 「確認」はコード上で指摘の条件を特定できたもの。「条件付き」は前提が揃ったときだけ成立するもの。「一部確認」は複数の小項目の一部だけ確かめたもの。「訂正」はコードと指摘が合わないもの。

## 3. 重要度「高」（8 件）

| ID | 重要度 | 内容 | 根拠 | 出典 | 検証 | 備考 |
|---|---|---|---|---|---|---|
| C-01 | 高 | RCUReader の exit が所有者解放後に成否フラグを書く | `src/core/RCUReader.h:138,180,183` | PRI-01 | 確認 | 並行 submit で競合する |
| C-02 | 高 | 学習結果の公開で tanh を二重に掛ける | `src/NoiseShaperLearner.cpp:975,1465` | PRI-03 | 確認 | 評価値と適用係数がずれる |
| C-03 | 高 | getState が最適化器の内部配列を無ロックで読む | `src/NoiseShaperLearner.cpp:379-390` | PRI-04 | 確認 | 非アトミック配列の競合 |
| C-04 | 高 | 旧ローダーが新しいロードのフラグを消す | `ConvolverProcessor.LoadPipeline.cpp:46-55,96` | PRI-06 | 確認 | 旧スレッドを join しない |
| C-05 | 高 | キャッシュ復元で時間領域の寸法を照合しない | `src/CacheManager.cpp:180-184,269-289` | PRI-11 | 確認 | mmap の外を読み得る |
| C-06 | 高 | 音声ファイルの宣言長をそのまま確保する | `src/IRConverter.cpp:212-224` | PRI-12 | 確認 | 数 GB の確保を試す |
| C-07 | 高 | NaN が jlimit を通過して湿音を止める | `ConvolverProcessor.Runtime.cpp:1054-1098` | PRI-13 | 確認 | 状態復元から入り得る |
| C-08 | 高 | 学習状態の復元で値を検証しない | `src/NoiseShaperLearner.cpp:1422-1456` | PRI-14 | 確認 | 属性値をそのまま公開する |

## 4. 重要度「中」（23 件）

| ID | 重要度 | 内容 | 根拠 | 出典 | 検証 | 備考 |
|---|---|---|---|---|---|---|
| C-09 | 中 | レガシー削除キューの容量超過で即時削除する | `src/core/DeletionQueue.cpp:13-26` | MED-01 | 確認 | 容量超過時のみ |
| C-10 | 中 | EQ の SIMD 条件が include と使用で揃わない | `EQProcessor.Processing.cpp:15-17,27,34` | MED-02 | 確認 | 現行構成では実害なし |
| C-11 | 中 | makeLPF に fc <= 0 のガードがない | `src/OutputFilter.cpp:25-29` | MED-03 | 確認 | ゼロ以下の周波数を通す |
| C-12 | 中 | 保留 reclaim だけが残ると drain が再試行しない | `AudioEngine.Retire.cpp:68-73,113-119` | MED-04 | 条件付き | 判定に保留リストがない |
| C-13 | 中 | オーバーサンプリング種別を無検証でキャストする | `src/DeviceSettings.cpp:1236-1237` | MED-06 | 確認 | 範囲外 enum を許す |
| C-14 | 中 | 優先度の初期文言とヘッドルーム上限の表示が違う | `src/DeviceSettings.cpp:364,478,695` | MED-07 | 確認 | 表示と入力許容が不一致 |
| C-15 | 中 | tryParseIntOption が空文字や 1+2 を通す | `src/MainWindow.cpp:13-20` | MED-08 | 確認 | containsOnly が空文字を許す |
| C-16 | 中 | ASIO ブラックリストの書込失敗とロガーの所有権 | `src/MainWindow.cpp:267-273,408-409` | MED-09 | 確認 | 戻り値を見ず、解放もしない |
| C-17 | 中 | 診断リングの初回書き込みが 1 つずれる | `AudioEngine.Processing.AudioBlock.cpp:768-772` | MED-10 | 確認 | fetch_add の戻り値は加算前 |
| C-18 | 中 | 2^21 サンプル超の IR で最小位相変換が飛ぶ | `ConvolverProcessor.ResampleAndFallback.cpp:378-385` | MED-12 | 確認 | ログだけ残して空を返す |
| C-19 | 中 | バイパス時に L1/L2 を維持してメモリが増える | `src/MKLNonUniformConvolver.cpp:708-750` | MED-13 | 確認 | 設計意図の注記あり |
| C-20 | 中 | テール倍率のスライダーと実効値が一致しない | `src/ConvolverControlPanel.cpp:147-151` | MED-14 | 確認 | 2 の冪へ丸める |
| C-21 | 中 | 空気吸収の高域減衰がブロック長に依存する | `src/MKLNonUniformConvolver.cpp:717,759` | MED-15 | 確認 | コード内にも注記 |
| C-22 | 中 | atomic のメモリ順序と静止契約の検証が足りない | `src/audioengine/AtomicAccess.h` ほか | MED-16 | 判定不能 | 具体箇所がない |
| C-23 | 中 | ログ・XML・キャッシュが無制限に育つ | `src/MainApplication.cpp:143-146` ほか | MED-17 | 一部確認 | ログだけ確認 |
| C-24 | 中 | 学習 XML の再解析で起動と終了が遅い | `src/NoiseShaperLearner.cpp:1098,1424` | MED-18 | 要再測定 | main に原因コードがない |
| C-25 | 中 | HB トレースのイベント列に上限がない | `src/audioengine/ISRHB.cpp:33-38,51-55` | MED-19 | 確認 | 診断時のみ |
| C-27 | 中 | MSVC で decimate の FMA 経路が無効 | `src/CustomInputOversampler.cpp:210,662` | #215 | 確認 | 性能だけの問題 |
| C-28 | 中 | OOM 時の学習器の失敗経路が到達しない | `src/NoiseShaperLearner.cpp:55-76` | 自前 | 確認 | create() は例外を投げる |
| C-29 | 中 | 計測レジストリの初回 claim でタグ公開が逆 | `src/core/ReaderMeasureRegistry.h:133-150` | #225 | 確認 | 診断値だけに影響 |
| C-30 | 中 | 出力先のチャンネル数を検査せず 2ch 目へ書く | `AudioEngine.Processing.DSPCoreDouble.cpp:282-296` | PRI-07 | 確認 | 現行呼出しは 2ch を保証 |
| C-31 | 中 | ドライバッファ未確保時にバイパスのフェードを飛ばす | `AudioEngine.Processing.DSPCoreFloat.cpp:195-209,377-399` | PRI-08 | 条件付き | 確保失敗時だけ |
| C-32 | 中 | バンド有効化でリセットを状態交換より先に出す | `EQProcessor.Parameters.cpp:92-99` | PRI-09 | 条件付き | 1 ブロックの過渡 |

## 5. 重要度「低」（14 件）

| ID | 重要度 | 内容 | 根拠 | 出典 | 検証 | 備考 |
|---|---|---|---|---|---|---|
| C-33 | 低 | LoudnessMeter に容量検査がない | `src/LoudnessMeter.cpp:47-62` | LOW-01 | 確認 | 上流で保護 |
| C-34 | 低 | shutdown 中は MMCSS 解除に到達しない | `AudioEngine.Processing.AudioBlock.cpp:43-48,71-74` | LOW-02 | 確認 | 早期リターンが先 |
| C-35 | 低 | floorNoLibm が NaN と範囲外を変換する | `ConvolverProcessor.Runtime.cpp:51-58` | LOW-03 | 確認 | 未定義動作 |
| C-36 | 低 | equalPowerSinApprox が 1 をわずかに超える | `src/core/FadeEngine.h:13-18` | LOW-04 | 確認 | 1.0000035 程度 |
| C-37 | 低 | 容量 0 を許す static_assert | `src/core/CommandBuffer.h:34` | LOW-05 | 確認 | 0 & (0-1) が 0 |
| C-38 | 低 | モノラルの Mid バンドで null に書く | `EQProcessor.Processing.cpp:714,895` | LOW-07 | 確認 | 直列構成のみ |
| C-39 | 低 | 非 2 冪ブロックで遅延境界のブロックが欠ける | `src/MKLNonUniformConvolver.cpp:2032-2041` | LOW-08 | 条件付き | 2 の冪ブロックでは跨がない |
| C-40 | 低 | 補間読みと TruePeak の境界防御が足りない | `ConvolverProcessor.Runtime.cpp:556-577` | LOW-09 | 一部確認 | TruePeak だけ確認 |
| C-41 | 低 | テストと CI の 8 件 | `.github/workflows/isr-verification.yml` | LOW-10 | 一部確認 | 孤児テストだけ確認 |
| C-42 | 低 | マニュアルの秒と分が食い違う | `manual/ADAPTIVE_NOISE_SHAPER_LEARNING_WINDOW_EN.md` | LOW-11 | 確認 | 30〜60 倍の差 |
| C-43 | 低 | ASan の対象一覧に 4 テストがない | `CMakeLists.txt:2841-2860` | 自前 | 確認 | 4 件とも未登録 |
| C-44 | 低 | ICX Debug 非対応と build.bat の案内が違う | `BUILD_GUIDE_WINDOWS.md:214` | 自前 | 確認 | 案内だけ Debug icx |
| C-45 | 低 | atomic のドット呼び出しが残る | `src/ConvolverProcessor.h:886,922` | 自前 | 確認 | ヘルパーと等価 |
| C-46 | 低 | 不要な zeroupper が残る | `src/CustomInputOversampler.cpp:195` | 自前 | 確認 | 実害なし |

## 6. 非バグ・設計リスク（10 件）

| ID | 重要度 | 内容 | 根拠 | 出典 | 検証 | 備考 |
|---|---|---|---|---|---|---|
| C-47 | 非バグ | アナライザーの有効ボタンは既定でトグルする | `src/SpectrumAnalyzerComponent.h:188` | NOT-01 | 訂正 | JUCE が既定で設定する |
| C-48 | 非バグ | copyLatest の読み順が書き込み順と逆 | `src/AudioSegmentBuffer.h:103-133` | PRI-02 | 訂正 | 両方 writePosition → totalSamples |
| C-49 | 非バグ | リミッターのソフトニーが動作しない | `src/audioengine/SimplePeakLimiter.h:41-70` | PRI-05 | 訂正（追記で再訂正） | desiredGain は 1 未満（式の符号が誤り。追記参照） |
| C-50 | 非バグ | EQ 係数に isfinite(sr) がなく NaN を通す | `EQProcessor.Coefficients.cpp:151,185,446` | PRI-10 | 訂正 | 検査は揃っている |
| C-51 | 非バグ | 学習器の静的プールが終了時にデッドロックする | `src/NoiseShaperLearner.cpp:21,1492-1497` | MED-05 | 訂正 | WeakReference で寿命を確認 |
| C-52 | 非バグ | stopRequested と stop_token が食い違う | `src/NoiseShaperLearner.cpp:853-859` | MED-11 | 訂正 | どちらでも終了する |
| C-53 | 非バグ | EQ キャッシュに同一キーが二重に入る | `src/audioengine/AudioEngine.Cache.cpp:97-128` | 自前 | 訂正 | ロック内で再探索する |
| C-54 | 非バグ | ソフトクリップのニー条件が逆転する | `AudioEngine.Processing.DSPCoreDouble.cpp:479-482` | LOW-06 | 訂正 | 飽和量 0〜1 で逆転しない |
| C-26 | 非バグ | ゲートスクリプトのパスがサブディレクトリを跨がない | `.github/scripts/*.ps1` | MED-20 | 訂正 | ソース走査はすべて再帰。非再帰はトップレベル向け |
| C-55 | 設計リスク | ASIO ブラックリストが const_cast に依存する | `src/DeviceSettings.cpp:120-135` | RISK-01 | 設計リスク | 将来の互換性に依存 |

## 7. 作業・運用（7 件）

| ID | 内容 | 根拠 | 確認 |
|---|---|---|---|
| W-01 | 未マージの WIP | `C:\VSC_Project\ConvoPeq` は `4d476d5c` で 103 コミット遅れ。変更 42 件と未追跡 152 件 | `git status`、`git diff --stat` |
| W-02 | 陳腐化した統合ブランチ | `wip/integrate-4d476d5c` は R1 と DQ-V2 より前。main と 390 ファイル差 | `git diff` |
| W-03 | 未整理のローカルブランチ | 62 本が残る。内容が main に入った枝もある | `git for-each-ref` |
| W-04 | ハングの非再現 | `PublicationAdmissionTests` は 6 経路で正常 | 独立 worktree と CI |
| W-05 | ラベルの運用 | `needs-revalidation` は自動付与、解除は人手 | `isr-verify-pr-sla.ps1` |
| W-06 | Issue のラベルなし | #215 と #225 にラベルがない | `gh issue list` |
| W-07 | Intel の LIB | `vcvars64` だけでは `libircmt.lib` が解決しない | 実測 |

## 8. 要再確認（5 件）

| ID | 内容 | 記録 |
|---|---|---|
| Q-01 | N1 テール保持 | 未採用（保留）。再検討条件が 3 つ |
| Q-02 | IRState の C3 | 未達（deref 読み手 9 箇所に slot 0） |
| Q-03 | Portable Reader Safety | NOT PROVED（P1 は証明、P2 は未証明） |
| Q-04 | W1 違反 | SnapshotCoordinator に 1 件発見 |
| Q-05 | CR5 修正 | 未 commit で維持。保証範囲 F/H が未検出 |

## 9. 個別の検証記録

### 9.1 C-01 の競合経路

138 行で `nestingDepth` を減らし、180 行で `ownerThreadToken` を 0 に公開し、183 行で `rootEnterSucceeded_ = false` を書く（`RCUReader::exit()`）。57 行の CAS で所有権を取り、78 行で `rootEnterSucceeded_ = true` を書く（`enter()`）。

`RuntimePublicationOrchestrator::trySubmitImpl` は Message スレッド、RebuildThread、CoordinatorLoop のワーカーから呼ばれる（44-48 行のコメント）。この関数はメンバの `publicationReader` を 55 行で context に入れ、`AudioEngine::makeRuntimeReadHandle` が 3704 行で `observeCurrentRuntime(ctx.reader)` を呼ぶ。

スレッド A が 180 行と 183 行のあいだにいるあいだに、スレッド B が CAS に勝って 78 行を書く。すると A が B の true を false で上書きする。bool は非アトミックなのでデータ競合でもある。窓は狭いが、経路は実在する。他の reader は単一スレッドで使われる（ConvolverProcessor ほか 3 つ）。この競合は publication 用の 1 個に限られる。

### 9.2 C-04 の上書き経路

`LoadPipeline.cpp` の開始処理は 46 行で `isLoading = true` にし、53 行で旧ローダーへ終了を指示し、54 行で `loaderTrashBin` へ移す。join はしない。59 行で世代を進め、96 行で新しいローダーを開始する。

旧ローダーの `FlagResetter` は、デストラクタで `success` が偽なら `isLoading` と `isRebuilding` を false にする（`LoaderThread.cpp:73-79`）。`callAsync` に失敗した場合は直接書く（91-92 行）。ここに世代の照合はない。新しいロードが始まった後に旧ローダーが失敗すると、新規ロードのフラグが消える。`run()` のコメント（50-55 行）は commit 時の通貨判定に正当性を委ねているが、フラグの巻き戻しはその対象外である。

### 9.3 C-05 の範囲外読み

180-184 行で `validateCacheFile` が `headerSize + dataSize + timeDomainSizeBytes` と実ファイル長の一致を検査する。したがって `dataSize` はファイル長を超えない。CRC と `memcpy` は安全である。

問題は時間領域の復元にある。273 行は残りバイト数が `timeDomainSizeBytes` 以上かを確かめるだけである。275 行で `channels × samples` の `AudioBuffer` を作り、283-288 行で `samples × 8` バイトを `channels` 回コピーする。`channels × samples × 8` と `timeDomainSizeBytes` の一致を見ていないため、細工したファイルではコピーが mmap の終端を越える。確保に失敗すれば `bad_alloc` がそのまま伝播する。

### 9.4 C-17 のオフバイワン

`fetch_add(count, 1)` の戻り値は加算前の値である。最初のコールバックでは `wc = 0` になり、`(wc - 1)` が unsigned で回り込んで最後のスロットへ書く。2 回目以降は `wc = 1, 2, ...` なのでスロット 0、1 と進む。

読み出しは `Timer.cpp:1720-1725` で `i` を `[max(0, wc - slots), wc)` の範囲で回し、`idx = i % slots` を使う。最初の書き込みがスロット `slots - 1` にあるため、読み出しは空のスロット 0 を読み、最新の 1 件を表示しない。2 回目以降も 1 件ぶんのずれが残る。

### 9.5 C-30 の到達性

282 行で `numChannels = min(2, source.getNumChannels())` を出し、290-295 行で `destination.getWritePointer(ch, 0)` を書く（`processDoubleToBuffer`）。出力先のチャンネル数は見ていない。

呼出し元は `BlockDouble.cpp:409` の 1 箇所だけである。その手前の 385 行が `dspCrossfadeDoubleBuffer.getNumChannels() >= 2` を要求し、バッファは `PrepareToPlay.cpp:188` で 2 チャンネルに確保される。現行経路では 1 チャンネルの出力先が来ないため、範囲外書き込みは起きない。関数の契約としては穴が残る。

### 9.6 C-31 の条件

`requestedFullBypass` は 195 行で `eqBypassed && convBypassed` として決まる。ブレンドは 380-384 行で `oversamplingFactor == 1`、左右のドライバッファ、容量が揃うことを要求する。どれかが欠けるとブレンドを飛ばし、ウェットをそのまま出す。ドライバッファは `prepareToPlay` で確保されるため、欠けるのは確保失敗か、準備済みのブロック長を超える入力のときだけである。

### 9.7 C-32 の過渡

92 行で状態をコピーし、97 行で `requestBandReset` を出し、99 行で新しい状態を交換する（`setBandEnabled`）。音声スレッドは 605-621 行でリセット要求を読み、フィルタ状態を消す。

音声スレッドが交換前にリセットを消費すると、古い係数のまま状態だけが消える。次のブロックで新しい係数が適用される。影響は 1 ブロックの応答変化であり、メモリ安全性の問題ではない。

### 9.8 C-38 の null 書き込み

895 行で `blockR = (numChannels > 1) ? block.getChannelPointer(1) : nullptr` を受け取る（`processSerial`）。Mid バンドで `numChannels < 2` のとき、714 行が `FloatVectorOperations::copy(dataR, dataL, numSamples)` を実行する。`dataR` が null なので書き込み先が無い。並列構成は 810-822 行でモノラルを分岐しており、この問題は直列構成の Mid だけに存在する。DSPCore は常に 2 チャンネルのブロックを渡す（`DSPCoreFloat.cpp:203-204`）ため、通常再生では到達しない。

### 9.9 訂正の根拠

- C-48: 書き込みは 104 または 112 行で `writePosition` を release し、118 行で `totalSamples` を release する。読み出しは 128 行で `writePosition`、129 行で `totalSamples` を acquire する。順序は一致する。
- C-49: `clipStart = threshold - knee * 0.5`（41 行）。ニー区間は `clipStart < peak <= threshold` なので `threshold / peak < 1`、`kneeShape > 0` となり、`desiredGain < 1` になる。`desiredGain < envelope` は成立する。（この式は符号が誤りである。末尾の追記で訂正した。）
- C-50: `calcSVFCoeffs` は 151 行で `!std::isfinite(sr)` を検査する。185 行と 446 行でも `g` と `k` の有限性を見る。`computeEstimatedMaxGainComplex` の呼出し元（`RebuildDispatch.cpp:819-824`）は現在の `EQState` を渡すため、`getTotalGain()` と対象が一致する。
- C-54: `threshold = 0.95 - 0.45 * sat`、`knee = 0.05 + 0.35 * sat`。`sat` は `Parameters.cpp:112` で 0〜1 に制限される。`knee` の最大は 0.4、`threshold` の最小は 0.5 で、逆転しない。

### 9.10 判断できなかった項目

- C-22: 「atomic のメモリ順序が弱い」という指摘に具体箇所がない。`AtomicAccess.h` の全体を読んだが、個別の欠陥として立証できなかった。

### 9.11 追加で確定した 2 件

- C-26: `.github/scripts` の `Get-ChildItem` 52 箇所を確認した。ソースを走査するものはすべて `-Recurse` を持つ。再帰を付けていないのは `.github/isr-*.json` や `doc/work5` の直下ファイルが対象で、階層を跨ぐ必要がない。指摘の症状は再現しない。
- C-39: `outputDelaySamples` は先行層の総長である。`l0Len` は `l0Part` の倍数で、`l0Part = nextPowerOfTwo(max(blockSize, 64))`。2 の冪ブロック長なら `o_L` はブロック長の倍数になり、境界を跨ぐブロックは出ない。コードの注記（866-872 行）も非 2 冪ブロックを前提外としている。

## 10. 重要度の変更

| 新ID | 旧ID | 変更 | 理由 |
| --- | --- | --- | --- |
| C-30 | H-05 | 高から中 | 唯一の呼出し元が 2 チャンネルを保証する |
| C-31 | H-06 | 高から中 | ドライバッファの確保失敗か容量不足のときだけ |
| C-32 | H-07 | 高から中 | 影響が 1 ブロックの過渡で、メモリ安全性に触れない |
| C-01 | H-01 | 条件付きから確認 | 3 種のスレッドが同じ reader を使う経路を確認した |
| C-04 | H-04 | 条件付きから確認 | 旧ローダーを join せずに次を開始する実装を確認した |
| C-38 | L-07 | 未検証から確認 | 直列 Mid の null 書き込みを確認した |
| C-39 | LOW-08 | 未検証から条件付き | 2 の冪ブロックでは `o_L` が `l0Part` の倍数になる |
| C-26 | MED-20 | 中から非バグ | ソース走査はすべて再帰。非再帰はトップレベル向け |

## 11. 着手順

1. C-02 と C-03 を直す。学習器の中で閉じており、変更が小さい。
2. C-05 と C-06 を直す。外部ファイルの寸法と長さを信用しない検査を入れる。
3. C-01、C-07、C-08 を直す。reader のフラグ公開順、NaN の遮断、状態値の検証である。
4. C-04 を直す。フラグのリセットに世代照合を入れる。
5. 中の中で実害が読みやすい C-12、C-14、C-16、C-17 を直す。
6. W-01 の WIP を切り分ける。`InputInspection` 系と learner 系を分ける。

## 12. まとめ

高 8 件は、いずれも通常の操作か細工したファイルで成立する経路がある。HTML の高 14 件をそのまま信じると、3 件の誤りと 3 件の過大評価を抱えたまま着手することになる。今回の検証で、作業対象は 8 件に絞れた。残る不確実性は C-22 と、実測が必要な C-23/C-24/C-40/C-41 である。

## 追記（2026-11-22 07:30 JST）

本文は 2026-11-21 12:01 時点の記録である。その後の実測で状態が変わった項目と、判定の誤りをここに追記する。本文の解析はそのまま残し、追記を参照する印だけを 3 箇所に足した。

| ID | 本文の状態 | 現在の状態 | 根拠 |
| --- | --- | --- | --- |
| C-24 | 要再測定 | 維持（原因コード無し）。再測定では学習 XML の寄与を特定できなかった | PR #232（計画）、PR #234（実測）、`doc/work113/20261122-0618_C24計測/report.md` |
| C-27 | 確認 | 命令生成の受入条件は満たした。性能はカーネル単体で 1 回測っただけで、#231 が求める方法（4 タップ、N ≥ 20、中央値と IQR）は未達のため #231 を再開した。Issue #215 はクローズ | PR #233、`doc/work113/20261122-0526_231検証/report.md` |
| C-49 | 非バグ（訂正） | 判定を訂正する。ニー区間では `desiredGain >= 1` となり、ソフトニーは動作しない。指摘は正しく、「確認（ソフトニー無効）」が妥当 | #238、`src/audioengine/SimplePeakLimiter.h:41-70` |

- C-24 の実測では、設定読込（S2）の絶対値が 799.1 ms で固定ゲート 400 ms を超えた。内訳は測っておらず、原因は未分類である。この超過は #236 で追跡する。学習 XML の増分は S4 の 6.6 ms だけで、S2 の増分はドリフト以下だった。
- C-22 は判定不能のまま独立して維持する。C-39 は条件付きのまま変更しない。
- C-23、C-40、C-41 は未着手である。
- C-49 の判定は誤りだった。本文 §9.9 は `threshold / peak < 1` と書くが、`peak <= threshold` なので `threshold / peak >= 1` である。したがって `1 - threshold / peak <= 0` となり、`desiredGain = 1 - (1 - threshold / peak) * kneeShape >= 1` になる。ニー区間でゲインは下がらず、ソフトニーは動作しない。指摘そのものは正しく、分類は「非バグ」ではなく「確認（ソフトニー無効）」が妥当である。この件は #238 で追跡する。

## 追記 2（2026-11-22 08:50 JST）

- #231 の性能測定を、4 タップ・25 回の反復・中央値と IQR でやり直した（PR #239、マージ `6829d47d`）。受入条件 6 項目をすべて満たしたため、#231 はクローズした。
- 結果は 4 タップで 1.02 倍、32 タップで 1.97 倍、64 タップで 3.03 倍である（MSVC のスカラー経路との比較）。4 タップでは AVX2 の利点が出ない。
- C-49 の訂正は #238、設定読込の超過は #236 で追跡する。どちらもオープンである。

## 追記 3（2026-11-22 12:10 JST）

- C-49 を #238 で修正した。ソフトニーを閾値中心の対数振幅対称（全幅 1.0 dB、-2.0〜-1.0 dBFS）に置き、両端の値と傾きを満たす 3 次エルミートで接続した。開始点はゲイン 1・傾き 0、終端点は `T/peak` とその傾きに一致する。
- 旧実装はニー区間で `desiredGain >= 1` となり削減が起きず、終端も閾値（`T/peak = 1`）だった。旧 `kPLKnee = 0.108748` は「-1 dBFS 基準の 1 dB の線形差」で、ニー幅としては 0.58 dB しかなかった。
- 単体テスト `SimplePeakLimiterKneeTests` を追加した。静的カーブ（1.0 dB と 0.1 dB）、開始点・終了点の連続性と傾き、単調性、出力上限、左右チャンネル、右 nullptr、不正パラメーターを確認している。期待値は実装とは別の連立解から作ったリテラルである。
- 分類は「確認（ソフトニー無効）」から「修正済み」へ移す。報告書は `doc/work113/20261122-1135_238修正/report.md`。
- 幅を大きくするとニー区間内で出力が `T` を超えるため（8 dB で +1.16e-3、40 dB で +1.57）、6 dB を超える幅はハードニーへ落とす。超過が始まる境界は 6.12319128 dB。
- C-49 の範囲外として残すもの: `DSPCore::reset()` が `peakLimiter.reset()` を呼ばない点、`limitingEngaged` の閾値が -1.5 dBFS のままである点。
