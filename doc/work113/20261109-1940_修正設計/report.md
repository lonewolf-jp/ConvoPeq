# 修正設計 — B-01／B-02 の優先設計と残り 14 件の計画

- 対象: 2026-11-09 の検証報告（`doc/work113/20261109-1855_バグ検証/report.md` 訂正版）でバグと確定した B-01〜B-16
- 前提: 本指示ではソース変更・コミット・マージ・リリース・公開告知を行わない。本文は設計案であり、実装結果ではない
- 検証との区別: B-01／B-02 の不具合成立は**コード読解による確定**であり、クラッシュ・OOB の実行時再現とメモリ実測は未実施
- 表記: 行番号は現行 main `c94d3767` の実ファイル行

## §1 概要

| 区分 | 件数 | 内容 |
|---|---|---|
| 優先設計（本紙 §2・§3） | 2 | B-02 CacheManager の OOB、B-01 IRConverter の未捕捉 `bad_alloc` |
| 修正計画（§4） | 14 | 対象・優先度・依存・テスト方針のみ |
| 非バグ | 12 | N-01〜N-12。修正対象にしない |
| 保留 | 1 | RES-06（STG-8 のフレーク）。静的には判定しない |

実装順序の推奨は B-02 → B-01 → B-03 →（B-07／B-09／B-15）→（B-05／B-06／B-04）→ B-08 → B-11 → その他。B-02 を先にするのは、B-01 の修正後もキャッシュ経路が先に走るため、入口の堅牢化を先に閉じる方が検証しやすいからである。

## §2 B-02 CacheManager の時間領域サイズ不整合（優先設計）

### 2.1 現状

- `validateCacheFile`（`src/CacheManager.cpp:125`）は magic・key（:170）・fftSize（:173）・総サイズ（:180–183）を検査する。**`timeDomainSizeBytes` と channels×samples×8 の整合は見ない。**
- `loadPreparedState`（:219）は残量 ≥ `timeDomainSizeBytes` を確認して（:273）から `AudioBuffer<double>(channels, samples)` を確保し（:275–277）、`samples×8` をチャンネル分 memcpy する（:283–287）。
- checksum は partition データのみ（:236）。
- 保存側は正しい値を書く（:368）。

### 2.2 修正方針

**方針 A（採用案）：ヘッダ記述子の整合検査を追加する。** 形式（v2）は変えない。

1. `CacheManager.cpp` の anonymous namespace に検査関数を置く。

```text
validateTimeDomainDescriptor(const CacheHeader& h, juce::String& why) -> bool
  1) 部分指定の拒否
     (h.timeDomainChannels == 0) != (h.timeDomainNumSamples == 0)  -> 拒否
     (h.timeDomainSizeBytes == 0) != (h.timeDomainChannels == 0)    -> 拒否
  2) 上限
     h.timeDomainChannels    <= kMaxTimeDomainChannels (8, IRLoadAdmission と同一)
     h.timeDomainNumSamples  <= kMaxTimeDomainSamples  (2^21+1, 同 FC-FORM-5)
  3) 乗算のオーバーフロー回避
     h.timeDomainNumSamples > SIZE_MAX / sizeof(double) / h.timeDomainChannels -> 拒否
  4) 整合
     h.timeDomainSizeBytes == channels * samples * sizeof(double) でなければ拒否
```

2. `validateCacheFile` の末尾（総サイズ検査の後）で呼び、偽なら false を返す。v1／v2 の双方に適用する（v1 も同じフィールドを持つ）。
3. `loadPreparedState` でも確保前に同じ関数で再検査する（防御の二重化）。int へのキャストは検査通過後に限る。
4. 不正キャッシュの扱い：`loadPreparedState` で `validateCacheFile` が偽になった場合、`[CACHE] rejected corrupt entry ...` を Logger に出し、当該ファイルを削除する（キャッシュは再構築可能で、毎回の再試行を避ける）。削除が失敗しても処理は継続する。
5. 既存の正常キャッシュは保存側の式（:368）と一致するため、本検査で落ちない。

### 2.3 既存のアドミッション規約との整合

- 上限値は `src/convolver/IRLoadAdmission.h` の 8ch と 2²¹+1 を再利用し、IR 本体の受入とキャッシュの受入で同じ境界を使う。
- キャッシュは「アドミッションを通過した IR」から作られるため、8ch／2²¹+1 を超えるキャッシュは定義上不正である。新しい上限を発明しない。
- 1 GiB のバイト予算は IR 入力側の規定であり、キャッシュの時間領域（8 バイト/サンプル）には直接対応しない。ここでは channels×samples×8 の整合だけで十分（総サイズ検査がファイル長を拘束する）。

### 2.4 影響範囲

| 対象 | 変化 |
|---|---|
| `CacheManager::validateCacheFile` / `loadPreparedState` | 不正ヘッダを拒否。正常時は従来どおり |
| `ConvolverProcessor::loadIR` | キャッシュミスとして `convertFile` へフォールバック（挙動は従来のミス時と同じ） |
| `ProgressiveUpgradeThread` | save 側は不変。load 側は同じ検査を通る |
| 既存キャッシュ | 正常なものは無効化されない。不正なものだけ削除 |
| 性能 | O(1) の算術のみ。ロード時間への影響なし |

### 2.5 エラー処理

- 失敗は例外ではなく `nullptr`（既存契約）。
- 削除・ログは失敗しても致命的にしない。
- B-01 の修正後は、フォールバック先の `convertFile` もアドミッションを通るため、細工キャッシュ→巨大 IR の連鎖でも確保は bounded になる。

### 2.6 回帰テスト計画

新規ターゲット `CacheManagerHeaderValidationTests`（`juce_add_console_app` ＋ `add_test`、`src/tests/` 配下）を追加する。既存の `IRLoadAdmissionTests` の WAV 生成ヘルパ（`src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp:62`）は流用しない（キャッシュは WAV を経由しない）。

| ID | 内容 | 期待 |
|---|---|---|
| C-1 | `save()` → `loadPreparedState()` の正常往復 | 非 null、内容一致 |
| C-2 | tdSizeBytes=8、samples=1000、総サイズ一致 | null。ASan で OOB なし |
| C-3 | tdSizeBytes を +1／−1 | null |
| C-4 | channels=0、samples>0 | null |
| C-5 | channels=9 | null |
| C-6 | channels=UINT64_MAX（オーバーフロー狙い） | 算術せず null |
| C-7 | key 不一致 | null |
| C-8 | partition CRC 不一致 | null |
| C-9 | v1 ヘッダの正常系 | 受理 |
| C-10 | 不正ファイルの削除 | ファイルが消える |

C-2 は既存の Sanitizer CI（ASan ジョブ）で実行し、OOB が無いことを確認する。C-1/C-9 は通常の ctest で足りる。

### 2.7 代替案とリスク

- 代替 A：ヘッダと時間領域も CRC 対象にする v3 形式。検出力は上がるが、キャッシュ全消去とバージョン分岐が必要。**今回は採用しない**（将来の課題）。
- 代替 B：時間領域をキャッシュから外す。ロード時間が増えるため採用しない。
- リスク：正当なキャッシュを誤って拒否すると再構築コストが増える。境界式は保存側と同一なので誤判定は起きない。
- 未検証：実行時の削除失敗（ファイルロック）は静的には判定できない。

## §3 B-01 IRConverter のアドミッション迂回（優先設計）

### 3.1 現状

- `IRConverter::loadAudioFile`（`src/IRConverter.cpp:198`）は「存在」「1〜INT_MAX」「channels > 0」だけを検査し（:212–218）、`AudioBuffer<float>`（:220）と `AudioBuffer<double>`（:224）を確保する。
- 同じ IR を読む `LoaderThread`（`ConvolverProcessor.LoaderThread.cpp:417–451`）とプレビュー（`ResampleAndFallback.cpp:300–326`）は 5 述語の admission を通る。
- `loadIR`（`ConvolverProcessor.LoadPipeline.cpp:234`）の `convertFile` 呼出（:295）に try/catch が無い。
- `handleLoadError` は error 状態だけを設定し、engine／file／metadata を変更しない（:241 のコメント、定義は同ファイル）。UI は `getLastError()` を表示する（`ConvolverControlPanel.cpp:1448–1450`）。

### 3.2 修正方針

**方針 A（採用案）：同じ規約を IRConverter に適用し、例外を封じる。** 経路の統合（LoaderThread への一本化）は大規模なため今回は行わない。

1. `loadAudioFile` のシグネチャに任意のエラー出力を足す（既定 nullptr、既存呼出は無変更）。

```text
static bool loadAudioFile(const juce::File&, juce::AudioBuffer<double>&,
                          double& sampleRateOut, juce::String* errorOut = nullptr)
```

2. reader 取得後、**確保の前**に `IRLoadAdmission.h` の述語を FC-INV-9 の順で適用する。

```text
admitChannelCountNonZero(channels)
admitFileLengthNonZero(n)
admitFileLengthRepresentable(n)
admitChannelCount(channels)          // <= 8
admitByteBudget(channels, n)         // channels * n * 8 <= 1 GiB
```

いずれかが偽なら、`describe...` 系の既存メッセージ（`IRLoadAdmission.h:134/144/154`）を用いて `errorOut` を設定し false を返す。

3. 確保と read を `try { ... } catch (const std::bad_alloc&)` で包み、false ＋ 理由（"IR too large (Out of Memory)"）にする。`AudioBuffer` の確保失敗はここで止まる。
4. `convertFile`（:237）で、リサンプル前に `admitResampleOutput(n, sourceRate, config.targetSampleRate)` を評価する（`IRLoadAdmission.h:110` の除算形）。失敗なら nullptr。`IRDSP::resampleIR` も `bad_alloc` を捕捉する。
5. `loadIR` の `convertFile` 呼出を `try/catch (std::bad_alloc)` で包み、`handleLoadError("IR too large (Out of Memory)")` を呼ぶ。`prepared == nullptr` のときも `handleLoadError` で理由を出す（現状は Logger のみ、:302–305）。
6. `loadIR` の他の確保（`cacheManager->save` など）は対象外とする。必要なら別項目。

### 3.3 既存のアドミッション規約との整合

- 述語・上限値・順序は `IRLoadAdmission.h` と `LoaderThread` の実装（:417–451）をそのまま使う。**新しい上限を導入しない。**
- プレビュー経路（`ResampleAndFallback.cpp:300–326`）と同じ判定になるため、UI でプレビューが通ったファイルは `loadIR` でも通る（現状の「プレビュー成功 → 本ロードで落ちる」非対称が消える）。
- リサンプル上限は出力側（`2^21+1`）で評価する。入力長で近似しない。

### 3.4 影響範囲

| 対象 | 変化 |
|---|---|
| `IRConverter::loadAudioFile` / `convertFile` | 8ch 超・1 GiB 超・リサンプル上限超を拒否。例外を外に出さない |
| `ConvolverProcessor::loadIR` | 失敗時に `handleLoadError` で UI へ通知。前の IR は保持（契約どおり） |
| UI（`ConvolverControlPanel`） | プレビュー成功後は従来どおり。巨大 IR はプレビュー段階で既に拒否 |
| セッション復元（`StateAndUI.cpp:422`） | 巨大 IR を参照するセッションはエラー表示になり、クラッシュしない |
| CLI（`--cli-ir`） | 同上。終了コードは現行の自動化契約に従う（変更しない） |
| 正当な IR | 1 GiB／8ch／2²¹+1 以内は従来どおり |

### 3.5 エラー処理

- `bad_alloc` を含む例外を `loadIR` の外へ出さない。
- 失敗時は `setLastError` → `postCoalescedChangeNotification` の既存経路で UI に出す。文言は LoaderThread の "IR too large (Out of Memory)" に合わせる。
- `isLoading` を確実に false に戻す（`handleLoadError` が行う）。
- キャンセル（`shouldCancel`）の意味論は変えない。

### 3.6 回帰テスト計画

既存の `IRLoadAdmissionTests`（`src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp`、CMakeLists 登録済み）に、IRConverter 経由のケースを追加する。WAV 生成は同ファイルのヘルパ（:62）を再利用する。

| ID | 内容 | 期待 |
|---|---|---|
| I-1 | 1 秒ステレオ 48 kHz の正常 WAV | `loadAudioFile` が true、サイズ一致（回帰） |
| I-2 | data チャンク長を偽装した WAV（宣言 ≈2³¹ サンプル） | false、例外なし |
| I-3 | 9ch WAV | false（`admitChannelCount`） |
| I-4 | 8ch WAV | true（境界の受理） |
| I-5 | 1 GiB 予算の直上（ch×len×8 > 1 GiB） | false |
| I-6 | 44.1 kHz IR を 48 kHz へ（出力 > 2²¹+1） | convertible 前に false |
| I-7 | 正常 IR の `convertFile` | 非 null、従来と同じ値 |
| I-8 | ハーネスで細工 WAV を `loadIR` に渡す | クラッシュせず、`getLastError()` が非空、前の IR が保持 |

I-2 は「確保前に false」であることを、ASan 実行で確保が起きていないことと併せて確認する。I-8 は実行時間が長いため任意（nightly 相当）とし、通常 CI では I-1〜I-6 を回す。

### 3.7 代替案とリスク

- 代替 A：`loadIR` を廃止し `loadImpulseResponse`（LoaderThread）へ一本化。恒久策だが、同期 API の呼出元（セッション復元・CLI・テスト）と UI の待ち合わせが変わるため大規模。**将来の課題**。
- 代替 B：`loadAudioFile` だけを守り、`loadIR` の例外封じを省略。`AudioBuffer` 以外の確保（r8brain 内部）で落ちる余地が残るため不十分。
- リスク：1 GiB 超の IR を常用するユーザーはロードできなくなる。LoaderThread と同じ制限であり一貫するが、**リリースノートでの告知が必要**（互換性変更）。
- 未検証：`bad_alloc` の実際の発火、OS のメモリ枯渇時の挙動は実行で確認していない。

## §4 残り 14 件の修正計画

B-01／B-02 は §2・§3 で設計済みのため、本節は B-03〜B-16 の 14 件を扱う。優先度は P1（早期）／P2（次段）／P3（余力時）。依存は「先に済ませるべき項目」。

| ID | 対象 | 優先度 | 依存 | 修正の要点 | テスト方針 |
|---|---|---|---|---|---|
| B-03 | `ConvolverProcessor.Runtime.cpp:1054/1089` | P1 | なし | 2 setter の `jlimit` の前に `isFinite` 拒否を追加（D8／D9 と同型） | `STG11` 系に mixed 用の負例を追加 |
| B-04 | `juce_WavAudioFormat.cpp`（第三者） | P2 | B-01 | `createReaderFor` の前にファイルサイズ上限とチャンクヘッダ走査を入れる（アプリ側）。JUCE 改変はしない | 細工チャンク長の WAV で拒否、正常 WAV の回帰 |
| B-05 | JUCE パーサ＋アプリ 6 か所 | P2 | なし | 読込前のサイズ上限（共通ヘルパ）。深さは JUCE 側の制約として文書化 | 深い入れ子・巨大実体の XML で拒否、正常プリセットの回帰 |
| B-06 | `MainWindow.cpp:1648` | P2 | B-05 | B-05 と同じヘルパを chooser に適用（例 64 MB） | 巨大 XML の選択でエラー表示、正常系の回帰 |
| B-07 | `NoiseShaperLearner.cpp` | P2 | なし | 1490 のスナップショットをジョブへ渡し、プール側の再読をやめる | 決定論が難しいため TSan と反復実行で確認 |
| B-08 | `CacheManager.cpp:468` | P2 | B-02 | 起動時（または初回ロード時）にディレクトリを列挙し、総バイト上限で剪定 | 複数セッション相当のダミーで剪定を確認 |
| B-09 | `Learning.cpp:481`、`StateIO.cpp:130` | P2 | なし | setter に `isFinite` 拒否、復元経路にサニタイズ（`DeviceSettings` と同型） | 細工セッションで bank が汚染されないこと |
| B-10 | `ResampleAndFallback.cpp:381`、`LoaderThread.cpp:887` | P3 | 方針決定 | スキップ時に `setLastError`／UI 通知。エラーにするか警告にするかは仕様判断 | 2²¹+1 サンプル超の IR で通知を確認 |
| B-11 | `ISREvidenceExporter.cpp:299–309/433` | P2 | なし | 実測カウンタの注入と、存在のみ判定の廃止。CI 契約（seed／verify）の更新を伴う | CI で「ok と Quarantine の併存」が消えることを確認 |
| B-12 | `TailLayerIntegrityTests.cpp:307/320–330/377` | P3 | なし | 独立参照（素朴畳み込み）か絶対エネルギー基準へ変更。F-5 に音声ケースを追加 | テスト自体の負例（均一損失を注入して落ちること） |
| B-13 | `MainWindow.cpp:269–275` | P3 | なし | 作成・読込の失敗を検出し、UI 通知またはユーザーディレクトリへフォールバック | 書込み不可のディレクトリで除外が効くこと |
| B-14 | `MainWindow.cpp:406–407` | P3 | なし | `unique_ptr` をアプリ側で保持（`release()` をやめる） | CLI 実行後にリークが無いこと（ASan） |
| B-15 | `NoiseShaperLearner.cpp:1465` | P3 | なし | `publishGenerationResult` の再 tanh を削除し、呼出元の parcor 契約に統一 | 公開値と評価値の一致を数値で確認 |
| B-16 | `build_identity_gate.py:439` | P3 | なし | pathspec を `:(glob)src/**/*.cpp` 形式へ。未追跡も検出 | サブディレクトリ差分で FAIL することを確認 |

### 4.1 共通の留意点

- B-04／B-05／B-06 は「ファイル入力の堅牢化」として 1 つのヘルパ（サイズ上限・拡張子・チャンク走査）にまとめると重複が減る。ただし B-04 は JUCE 内部の確保順に関わるため、アプリ側の事前検査で代替する。
- B-07／B-09／B-15 は学習器サブシステムで、同じファイル群を触る。まとめて 1 つの変更単位にすると衝突が減る。
- B-12／B-16 は製品コードではなくテスト／検証ツール。リリース判断への影響は小さい。
- B-11 は CI の契約ファイル（`isr-*.json`、seed／verify スクリプト）と連動するため、単独で入れると CI が落ちる可能性がある。契約更新と同時に行う。

## §5 実装しないこと（本指示の範囲）

- ソースコードの変更、ビルド構成の変更、コミット、main へのマージ、リリース、配布、公開告知。
- 本紙は設計案であり、コードに適用していない。適用時は別途の承認を得る。

## §6 未検証のまま残る点

- B-01／B-02 の実行時再現（クラッシュ・OOB）とメモリ実測は未実施。設計はコード読解に基づく。
- B-01 の `bad_alloc` 発火条件（OS のメモリ状況）は環境依存で、静的には確定しない。
- B-02 の削除失敗（ファイルロック）時の挙動は実行でしか確認できない。
- RES-06 は保留を維持する。

## §7 B-01 実装前の設計確認（5 点）

Owner 指定の 5 点について、実装に先立って確定させた内容を示す。

### 7.1 byte budget が同時確保バッファの総量を制約する方法

現行の `convertFile` は、ロードした double IR（8·ch·N）を `converted` へ**無条件にコピー**し（+8·ch·N）、パーティション配列（+8·ch·N）と解析用 `scaledIR`（+8·ch·N）も確保する。したがって FC-FORM-1 の 8·ch·N だけでは総量を制約できない（実際のピークは最大 32·ch·N 級）。

修正では次の 3 点を組み合わせる。

1. 無条件コピーを move に置換する（`ir` は以降使わない）。これで 8·ch·N を消す。
2. 入力は 256K サンプルのチャンクで float→double 変換する（LoaderThread の `kStreamChunk` と同一）。float 側は 4·ch·chunk に固定される。
3. 段階ごとに、既存の `kMaxIRLoadBytes` だけを使って working set を検査する（除算形。新しい上限値は導入しない）。

| ステージ | 同時に生きるバッファ | 検査 |
|---|---|---|
| A: ロード | double IR 8·ch·N ＋ float chunk 4·ch·chunk | 8·ch·N ＋ 4·ch·chunk ≤ 1 GiB |
| B: リサンプル | 入力 8·ch·N ＋ 出力 8·ch·N′ | 8·ch·(N ＋ N′) ≤ 1 GiB |
| C: パーティション＋解析 | converted 8 ＋ partition 8 ＋ scaledIR 8 | 24·ch·N′ ≤ 1 GiB |

N′ は `ceil(N × targetSR / sourceSR) + 2` の見積りを使い、実際の長さは FC-FORM-5（2²¹+1）で別途制約する。ステージ A の chunk は `min(N, 262144)` サンプル。

この結果、非リサンプルかつ大容量の IR は従来より早い段階で拒否される（例：2ch では N ≒ 2,230 万サンプルが上限）。これは「実際に同時確保する量」に対する制約であり、FC-FORM-1 の値を変えるものではない。境界の差は §7.6 に明記する。

### 7.2 すべての確保経路で `std::bad_alloc` を捕捉する

- `loadAudioFile`：chunk 用 `AudioBuffer<float>` の構築、`out.setSize`、`reader->read` を try で包む。
- `convertFile`：`resampleIR`、パーティション配列の確保、解析用コピーを try で包む。
- `loadIR`：`convertFile` 呼出全体と `cacheManager->save` を try で包む。
- 捕捉は `std::bad_alloc` を基本とし、`loadIR` では `std::exception` も捕捉して別文言で `handleLoadError` に渡す。例外をメッセージスレッドの外へ出さない。

### 7.3 失敗時に現在の IR と Engine 状態を保持する

- `loadIR` は冒頭で `currentIrFile` を更新している。失敗時は `irFileLock` 下で previous を復元する（`currentIrFile` が失敗ファイルを指したままになると、以後のリビルドが同じ失敗を繰り返すため）。
- 音声の IR は `applyComputedIR` を呼ばないため不変。
- `isLoading` は `handleLoadError` が false に戻す。世代は進むが、これは in-flight の無効化であり状態破壊ではない。

### 7.4 リサンプル前後の上限と既存 Admission の整合

- 前段：`admitResampleOutput(N, sourceSR, targetSR)`（FC-FORM-5、2²¹+1）を `resampleIR` の前に評価する。
- 後段：戻り値の実長が上限内であることを確認する（r8brain の丸め差のため）。
- バイト予算は §7.1 のステージ B で評価する。

### 7.5 セッション復元・CLI・UI のエラー処理の一貫性

3 経路はいずれも `requestConvolverPreset` → `loadIR` を通る。失敗は `handleLoadError` → `setLastError` → `ConvolverControlPanel` の既存表示に集約される。プレビューは従来の admission のままで、プレビュー通過後に `loadIR` が拒否した場合も同じ表示になる（新しい不整合は作らない）。

### 7.6 既知の境界差（実装後に再評価）

- IRConverter の working set 検査は、LoaderThread の FC-FORM-1（8·ch·N）より厳しい。同じファイルがプレビューで受理され、`loadIR` で拒否される境界が生じ得る。これは「この経路だけが 3 本の double バッファを同時に持つ」事実に基づく意図的な差であり、エラー表示は共通である。
- 実装後、境界付近の値（2ch／8ch の N 上限）を実測して本節を更新する。
