# B-02 / B-01 実装と検証（2026-11-09）

## 1. 作業概要

Owner の実装許可に基づき、検証レポートでバグと確定した 16 件のうち B-02（CacheManager）と B-01（IRConverter）を実装し、回帰テストで確認した。

許可範囲は次のとおりで、いずれも逸脱していない。

| 項目 | 状態 |
|---|---|
| B-02 と B-01 の実装・検証 | 実施（本書） |
| 残り 14 件の実装 | 未実施（設計は `doc/work113/20261109-1940_修正設計/report.md`、段階提示は `doc/work113/20261109-2038_残14件提示/report.md`） |
| main へのマージ・リリース・配布・公開告知 | 未実施 |
| README / RELEASE_NOTES / manual の変更 | 未実施 |
| コミット・ブランチ操作 | 未実施（作業ツリー上のみ） |

B-02 を先に適用して単独で全テストを通し、その後 B-01 を加えた。この順序により、どちらの変更が回帰を起こしたかを切り分けられる。

## 2. 対象の識別子

| ID | 対象 | 重大度 | 内容（要約） |
|---|---|---|---|
| B-02 | `src/CacheManager.cpp` / `.h` | Low–Medium | キャッシュヘッダの時間領域ディスクリプタが未検証で、細工した `.bin` によりマップ外読み出しが起こり得る |
| B-01 | `src/IRConverter.cpp` / `.h`、`ConvolverProcessor.LoadPipeline.cpp` | Medium | 同期 IR ロード経路が admission（チャンネル数・長さ・byte 予算）を一切通らず、`lengthInSamples` をそのまま `int` へ窄めて全長を確保する。例外も捕捉しない |

旧番号との対応は #1 → B-02、#2 → B-01。

## 3. B-02: CacheManager の時間領域ディスクリプタ検証

### 3.1 変更内容

`src/CacheManager.cpp` に匿名 namespace の検査関数 `validateTimeDomainDescriptor` を追加した。上限は新設せず、`convo/IRLoadAdmission.h` の凍結値（8ch、2²¹+1 サンプル）を再利用する。

検査は次の順で行う。

1. 3 つのフィールド（channels / samples / sizeBytes）が「すべて 0」か「すべて非 0」であること。部分指定は拒否する。
2. すべて 0 なら時間領域なしとして受理（従来キャッシュとの互換）。
3. channels ≤ 8。
4. samples ≤ 2²¹+1。
5. `samples > SIZE_MAX / 8 / channels` なら拒否（乗算の前に溢れを判定する）。
6. `timeDomainSizeBytes == channels × samples × 8`。

この検査を 2 か所で適用する。`validateCacheFile`（読み込み時）と、`loadPreparedState` から分割した `loadPreparedStateFromFile` の時間領域復元直前である。ヘッダ検査と確保の間に値が変わり得ないとしても、確保の直前で再確認しておけば、将来ヘッダを組み立て直す変更が入っても確保側が無防備にならない。

総サイズ検査は符号付きの加算から、符号なしの減算前確認へ書き換えた。

- `fileSize < 0` を拒否。
- `headerSize > fileSize` を拒否。
- `dataSize > payloadSize` を拒否。
- `timeDomainSizeBytes > payloadSize - dataSize` を拒否。
- `dataSize + timeDomainSizeBytes != payloadSize` を拒否。

あわせて `numPartitions` と `numChannels` に `INT_MAX` 上限を設け、`fftSize` の比較を `uint64_t` に統一した。従来は `static_cast<int>(headerOut.fftSize)` で窄めてから比較していたため、`fftSize` が `INT_MAX` 超のときに負値へ化けて比較が成立しない余地があった。

時間領域の復元は、`mmap.getSize()` を符号なしへ移してから `tdOffset` と `expectedTdBytes` を減算前確認する形に変えた。1 チャンネル分のコピーは従来どおり `memcpy` で、strict-aliasing 違反は生じない。

不正と判定したエントリは Logger に 1 行残してファイルを削除する。キャッシュは再構築できるため、壊れたエントリを毎回読み直すより削除して再生成させるほうが安い。

### 3.2 テスト隔離

`CacheManager` に `explicit CacheManager(juce::File cacheDirectoryOverride = {})` を追加した。既定は従来どおり `userApplicationDataDirectory/ConvoPeq/IRCache` で、本番の挙動は変わらない。テストは一時ディレクトリを渡してユーザーのキャッシュを触らない。

### 3.3 回帰テスト

`src/tests/CacheManagerHeaderValidationTests.cpp` を新設し、`CMakeLists.txt` に `CacheManagerHeaderValidationTests` を追加した。テストは v1 / v2 のヘッダを実際に書き出し、必要に応じてバイト列を細工して読み込ませる。

| ID | 内容 | 期待 |
|---|---|---|
| C-0 / C-0b | キャッシュファイルの所在と基準の時間領域サイズ | 期待どおりの値 |
| C-1 / C-1b / C-1c | 正常な v2 キャッシュ | 受理・時間領域復元・パーティション長一致 |
| C-2 / C-2b | 時間領域サイズの不一致 | 拒否・ファイル削除 |
| C-3 | サイズの過大申告 | 拒否 |
| C-4 | ディスクリプタの部分指定 | 拒否 |
| C-5 | チャンネル上限超え | 拒否 |
| C-6 | 乗算オーバーフローを狙う値 | 拒否 |
| C-7 / C-7b | キー不一致 | 拒否・ファイル削除 |
| C-8 | ペイロード CRC 不一致 | 拒否 |
| C-9 / C-9b / C-9c / C-9d | v1（時間領域なし／あり） | 受理・復元 |
| C-10 / C-10b | v2（時間領域なし） | 受理・復元なし |

## 4. B-01: IRConverter の同期経路

### 4.1 実装前の設計確認

Owner 指定の 5 点は `doc/work113/20261109-1940_修正設計/report.md` §7 に確定させた。要点は次のとおり。

- byte 予算は「同時に生きるバッファの合計」に適用する。段階ごとに 8·ch·N（IR）、8·ch·(N+N′)（リサンプル入出力）、24·ch·N′（パーティション＋IR＋解析コピー）を既存の 1 GiB で制約する。新しい上限値は導入しない。
- 確保は `loadAudioFile`・`convertFile`・`loadIR` の 3 層で `std::bad_alloc` を捕捉し、メッセージスレッドの外へ例外を出さない（アップグレード経路のキャッシュ保存も同じ扱いとした。§4.2）。
- 失敗時は `applyComputedIR` を呼ばないため、音声の IR とエンジン状態は変わらない。`currentIrFile` は直前の値へ戻す。
- リサンプル前後に FC-FORM-5（2²¹+1）を評価する。
- セッション復元・CLI・UI はいずれも `requestConvolverPreset` → `loadIR` を通るため、失敗表示は既存の `getLastError()` に集約される。

### 4.2 変更内容

`loadAudioFile` は次の 4 点を変えた。

1. `reader->numChannels`（unsigned）と `reader->lengthInSamples`（int64）を窄める前に、FC-INV-9 の順で 5 述語を評価する（`admitChannelCountNonZero` → `admitFileLengthNonZero` → `admitFileLengthRepresentable` → `admitChannelCount` → `admitByteBudget`）。拒否理由は `IRLoadAdmission` の診断文言をそのまま返す。
2. 作業セット（8·ch·N ＋ 4·ch·chunk）を既存予算で検査する。
3. 読み込みを 256K サンプルのチャンクに分割する（`LoaderThread` の `kStreamChunk` と同一）。float 一時バッファはチャンク分に固定され、ファイル長に比例しない。
4. 確保と読み込みを `try` で包み、`std::bad_alloc` と `std::exception` を捕捉して理由付きで false を返す。

`convertFile` は次の 7 点を変えた。

1. 無条件コピー `juce::AudioBuffer<double> converted = ir;` を move に置き換えた。従来は同じ大きさのバッファを 2 本同時に持っていた。
2. リサンプル前に FC-FORM-5 を評価し、さらに 8·ch·(N+N′) を予算で検査する。`N′` は `ceil(N × targetSR / sourceSR) + 2` で見積もる。
3. リサンプル後の実長でも上限を再確認する（r8brain の丸め差のため）。
4. リサンプル成功後は入力を解放し、後段の同時確保量を減らす。
5. パーティション確保の前に、IR と解析コピー（16·ch·N′）＋パーティションバイトを予算で検査する。
6. `DIAG_MKL_MALLOC` の失敗を理由付きで返し、パーティションへ所有権を移した後の例外では二重解放しない。
7. 解析（`scaledIR` の生成と `IRAnalyzer`）を `try` で包む。

`ConvertConfig` を受け取る `convertFile` に `juce::String* errorOut = nullptr` を追加した。既定値があるため既存の呼び出しは変更不要で、`convertToHighRes` もそのまま動く。

`ConvolverProcessor::loadIR` は次の 6 点を変えた。

1. `currentIrFile` の直前値を控える。
2. キャッシュ読み出し（高解像度・低解像度の 2 経路）を `try` で包み、確保失敗はキャッシュミスとして変換へフォールバックする。
3. `convertFile` にエラー出力を渡し、失敗理由を保持する。
4. キャッシュ保存を `try` で包む。保存に失敗しても IR の適用は続ける（キャッシュは最適化であり、今回の IR の妥当性とは独立）。
5. 変換自体が失敗した場合に限り、現世代のままであることを確認して `currentIrFile` を戻す。
6. `handleLoadError` へ具体的な理由を渡す。理由がない場合のみ従来の「no FFT applied」文言を使う。

アップグレード経路（`ProgressiveUpgradeThread.cpp`）のキャッシュ保存も `try` で包んだ。`juce::Thread::run` の外へ例外が出るとプロセスが終了するため、確保失敗は当該ステップの失敗として扱う。

### 4.3 挙動の変化（意図したもの）

- 同期経路でも 9ch 以上の IR を拒否する。従来はプレビューだけが拒否し、本適用は通っていた。
- 同期経路でも 1 GiB 超の IR を拒否する。
- 末尾無音をトリムしない経路のため、FC-FORM-5 は「実際にリサンプルする長さ」に対して評価する。`LoaderThread` はトリム後の長さで評価するので、無音テールの長いファイルでは両者の判定が分かれ得る。この差は設計書 §7.6 に記載した。
- 変換に失敗した IR は `currentIrFile` に残らない。サンプルレート変更時のリビルドが同じ失敗を繰り返さない。

## 5. 検証結果

### 5.1 B-02 単独の状態

| 検証 | 結果 |
|---|---|
| `CacheManagerHeaderValidationTests`（Release） | 20/20 PASS |
| `CacheManagerHeaderValidationTests`（RelWithDebInfo + AddressSanitizer） | 20/20 PASS（ASan ランタイム DLL を exe 隣に配置して実行） |
| `ctest -C Release` 全件 | 57/57 PASS（334.24 s） |

ASan 実行でマップ外読み出しの報告は無かった。C-2（時間領域サイズ不一致）は、修正前なら「サイズが小さい」ヘッダで境界判定を通し、`channels × samples × 8` のコピーでマップ末尾を越えて読む入力である。

### 5.2 B-01 を加えた状態

| 検証 | 結果 |
|---|---|
| `AudioEngineHarness`（Release、単体実行） | 終了コード 0、`[B01] IRConverter admission: PASS (I-1..I-6)`（単体実行 20:45、ctest 内実行 20:23。いずれも同一バイナリ） |
| 既存 WORK102 の実行対象 | すべて PASS（J/K は従来どおり構造検証のみで未実行） |
| `ctest -C Release` 全件 | 57/57 PASS（346.00 s） |

I-1〜I-6 は次の経路を確認する。

| ID | 入力 | 期待した結果 |
|---|---|---|
| I-1 | 正常な 1 秒ステレオ 48 kHz | 変換成功 |
| I-2 | data チャンク長を 3e9 バイトと偽装 | 拒否（理由に "memory limit" を含む） |
| I-3 | 9ch | 拒否（理由に "too many channels" を含む） |
| I-4 | 8ch | 変換成功 |
| I-5 | 2ch・N = 67,108,865（1 GiB 境界の 1 サンプル上） | 拒否（理由に "memory limit" を含む） |
| I-6 | 8 kHz・350,000 サンプルを 48 kHz へ（43.75 s > 43.69 s） | 拒否（理由に "longer than the DSP" を含む） |

拒否系は null だけでなく理由の文字列まで照合する。最初の実行ではフィクスチャのファイル名に拡張子が無く、`AudioFormatManager::createReaderFor` が拡張子で形式を選ぶ仕様のため読み込み自体に失敗していた。このとき拒否系の 4 項目は「別の理由で null」でも合格していた。拡張子を付けて意図した経路を通し、理由の照合を足して再実行した結果が上表である。

### 5.3 切り分け

B-02 単独（57/57）→ B-01 追加（57/57）の順で、どちらの段階でも全件が通った。B-01 の追加によって既存テストの結果は変わっていない。

### 5.4 ASan 構成での回帰

`build-asan-local`（RelWithDebInfo + `/fsanitize=address`）で `AudioEngineHarness` をビルドし、`clang_rt.asan_dynamic-x86_64.dll` を exe の隣に置いて実行した。

| 検証 | 結果 |
|---|---|
| `AudioEngineHarness`（ASan） | 終了コード 0、全テスト PASS、AddressSanitizer の報告 0 件 |
| I-1〜I-6（ASan） | すべて PASS |

ログに `AddressSanitizer`、`LeakSanitizer`、`SUMMARY` の出力は無い。B-02 側の ASan 実行（20/20）は §5.1 のとおり。

## 6. 残存リスクと未検証事項

- B-01 / B-02 の実行時再現は行っていない。細工ファイルと細工ヘッダを与えた入出力の確認であり、実際のクラッシュ再現ではない。
- メモリ使用量の実測は行っていない。予算検査は「同時に生きるバッファの合計」を計算で制約するもので、実測のピークと一致する保証はない。
- `(32768, 524288]` のサンプル数域を通る音声経路は未検証のまま（前回報告から継続）。
- RES-06 のフレークは保留を維持している。
- 同期経路の FC-FORM-5 は、トリムを行わないため `LoaderThread` より厳しい。無音テールが長く、かつリサンプルが必要な IR は、プレビューが通っても本適用で拒否される場合がある。
- `juce::String` の確保失敗は捕捉していない。バッファ確保の失敗は捕捉するが、文字列組み立て中の `std::bad_alloc` は理論上残る。

## 7. 変更ファイル

| ファイル | 内容 |
|---|---|
| `src/CacheManager.h` | テスト用コンストラクタ、`loadPreparedStateFromFile` 宣言、`cacheDirectoryOverride` |
| `src/CacheManager.cpp` | `validateTimeDomainDescriptor`、総サイズ検査の書換、復元の二重検証・境界検査・不正削除 |
| `src/IRConverter.h` | `convertFile` の `errorOut`、`loadAudioFile` の `errorOut` |
| `src/IRConverter.cpp` | admission 適用、チャンク読込、予算検査、例外捕捉、move 化 |
| `src/convolver/ConvolverProcessor.LoadPipeline.cpp` | 失敗理由の伝搬、キャッシュ読み書きの例外封じ、`currentIrFile` 復元 |
| `src/ProgressiveUpgradeThread.cpp` | アップグレード時のキャッシュ保存の例外封じ |
| `src/tests/CacheManagerHeaderValidationTests.cpp` | 新規（C-0〜C-10） |
| `src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp` | I-1〜I-6 を追加 |
| `CMakeLists.txt` | `CacheManagerHeaderValidationTests` ターゲット、ASan 適用 |

いずれも未コミットである。
