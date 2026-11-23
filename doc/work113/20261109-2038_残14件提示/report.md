# 残り 14 件の段階提示（B-03〜B-16）

## §0 本書の位置づけ

B-01／B-02 を除く 14 件について、実装に入る前に Owner が判断できる材料を揃えた。実装は行っていない。本章の内容はすべてコード読解に基づき、実行時再現とメモリ実測は未実施である。例外は B-16 で、git コマンドの実測結果を §5 に示す。

N-01〜N-12 の非バグ判定は変更しない。B-16 についてのみ、後述の実測を根拠に判定の変更を提案する。

提示は 3 段階に分けた。各段階は独立に承認でき、前段の完了を待たずに次段を判断してよい。ただし B-05 → B-06、B-01 → B-04 の依存は残る。

| 段階 | 対象 | 件数 | 性格 |
|---|---|---|---|
| 提示 1 | B-03 | 1 | Medium。修正が 2 行規模で、実害の筋が最も明確 |
| 提示 2 | B-04、B-05、B-06、B-07、B-08、B-09、B-11 | 7 | 入力堅牢化・学習器・証跡。B-08 と B-11 は仕様判断を含む |
| 提示 3 | B-10、B-12、B-13、B-14、B-15 | 5 | 影響は限定的。B-10 は仕様判断が先行 |
| 別枠 | B-16 | 1 | 前提が実測で否定された。判定変更の提案 |

## §1 先に決めるべき 2 件

### 1.1 B-10 の仕様判断

`convertToMinimumPhase`（`src/convolver/ConvolverProcessor.ResampleAndFallback.cpp:370`）は、FFT サイズが `MAX_MINPHASE_FFT_SIZE = 8388608`（:380）を超えると空のバッファを返す（:381-385）。呼出側（`src/convolver/ConvolverProcessor.LoaderThread.cpp:884`）は戻り値を `validateBuffer` で判定し、偽なら置換しない（:887）。結果として線形位相の IR がそのままロードされ、エラーも警告も出ず、UI は Minimum のまま残る（:880-908）。Mixed も同じ形で、`convertToMixedPhase` の戻り値が無効なら置換しない（:904-905）。

判断が必要なのは「この状態をどう扱うか」であって、検出方法ではない。案は 3 つある。

| 案 | 内容 | 後方互換 | 実装量 | 評価 |
|---|---|---|---|---|
| A | ロード失敗にする（`handleLoadError`） | 壊れる（従来読めていた IR が読めなくなる） | 小 | 安全側だが利用者の選択肢を奪う |
| B | 警告を出し、線形位相で適用したことを UI に表示する | 保たれる | 中 | 推奨。表示と実処理の乖離だけを消せる |
| C | Minimum／Mixed 選択時に該当 IR を事前拒否する | 一部壊れる | 中 | A と同じ問題を早い段階で起こすだけ |

推奨は B。理由は、長い IR は「読めるが最小位相化できない」のであって、読めなくする理由がない。加えて B は、UI が「Minimum」と表示しながら線形で動く現状の矛盾を解消するという一点で完結する。

B を選ぶ場合の修正範囲は 3 ファイル程度になる。`LoaderThread` の分岐でフラグを立て、`IRState` に「実効位相モード」を持たせ、`ConvolverControlPanel` の表示に反映する。回帰テストは 2²¹+1 サンプル超の IR を Minimum と Mixed で読み、警告が出て線形として適用されることを確認する。

判断が付くまで B-10 は実装しない。

### 1.2 B-11 の CI 契約への影響

B-11 は「証跡が実測を表していない」問題だが、修正すると CI の契約に触れる。先に影響範囲を示す。

`src/audioengine/ISREvidenceExporter.cpp:298-313` は 11 個の artifact を定数文字列で書き出す。主なものは次のとおり。

| artifact | 現行の出力 |
|---|---|
| `closure_graph.json` | `nodeCount:0`、`edgeCount:0`、`descriptorCoverageComplete:true` |
| `hb_graph_trace.json` | `eventCount:0` |
| `hb_violation_report.json` | `status:"ok"`、`violations:[]` |
| `retire_timeline.json` | `totalTransitions:0`、lane／lifecycle カウンタは全て 0 |
| `payload_tier_report.json` | `violations:0`、families は固定 4 件 |
| `shutdown_trace.json` | `status:"template"`、`verified:false` |
| `retire_latency_report.json` | `status:"template"`、`withinThreshold:false` |

`FailureHandler::handleFailure`（:430-449）は `hb_violation_report.json` が**存在するか**だけで Quarantine を選ぶ（:433、:443）。ファイルの中身が `status:"ok"` でも Quarantine になる。これが「同じ run で ok と Quarantine が併存する」の正体である。

CI 側の検証器が要求する内容は次のとおり（`.github/scripts/` を実測）。

| 検証器 | artifact | schema | 必須キーと追加条件 |
|---|---|---|---|
| `isr-verify-v2.ps1` | `mutation_fault_trace.json` | v1 | `status` が `generated` か `ok`。`violations` があれば 0 |
| `isr-verify-v3.ps1` | `closure_graph.json` | v1 | `status`、`descriptorCoverageComplete`、`externalMutableDependencies` |
| `isr-verify-v4.ps1` / `v9` | `payload_tier_report.json` | v1 | `status`、`families` |
| `isr-verify-v5.ps1` | `hb_graph_trace.json` | `hb_trace_v1` | `eventCount`、`events`。`eventCount > 0`、件数一致、`ts` 単調、各要素に `ts/from/to/fromEpoch/toEpoch/mo/release/acquire` |
| `isr-verify-v5.ps1` | `hb_violation_report.json` | v1 | `status`、`violations`、`scenarioResults`。`status == "violation"` で失敗、`scenarioResults` は 4 件（`forced_reorder`、`epoch_lag`、`retire_delay`、`observe_race`）で各 `result` が `pass` |
| `isr-verify-v6.ps1` | `shutdown_trace.json` | v1 | 8 キー。`verified == true`、`sh5_lateCallbackCount` と `sh6_postStopEnqueueCount` が 0 |
| `isr-verify-v7.ps1` | `retire_latency_report.json` | v1 | `withinThreshold == true`。加えて `retire_timeline.json` の schema が v1 か v2 |
| `isr-verify-v10.ps1` | `runtime_budget_report.json` | v1 | `artifactTotalBytes` |
| `isr-verify-evidence-provenance.ps1` | 全般 | — | strict mode では seed 済み artifact を拒否（`isr-verify-common.ps1:101`） |

現行のアプリ出力をこのまま検証器に通すと、少なくとも 4 件が落ちる。

1. `hb_violation_report.json` に `scenarioResults` が無い（v5 が `Missing required property` で失敗）。
2. `hb_graph_trace.json` の `eventCount` が 0（v5 が「少なくとも 1 件必要」で失敗）。
3. `shutdown_trace.json` の `verified` が false（v6 が「FSM 検証失敗」で失敗）。
4. `retire_latency_report.json` の `withinThreshold` が false（v7 が「閾値超過」で失敗）。

> 実装後の追記: この 4 件は、テンプレートを書き出す経路（Debug の `debugArtifacts`）を前提にした指摘である。Release の `--verify-runtime-evidence` を実測すると出力は `evidence_manifest.json` 1 件のみ（`artifacts:[]`）で、v5・v6・v7 はいずれも `Artifact not found` により exit 1 になった。欠落は 4 件ではなく全件である。詳細は `doc/work113/20261109-2145_残件実装/report.md` §5.5。
CI の証跡は `.github/scripts/isr-generate-runtime-evidence.ps1`、`isr-seed-evidence.ps1`、`isr-run-runtime-evidence.ps1` が `evidence/` に生成しており、アプリの `artifactRoot()`（`ISREvidenceExporter.cpp:21-24`）も `current_path()/evidence` を指す。つまりアプリをリポジトリ直下で動かすと、CI と同じファイル名を上書きする。これが「Debug で検証すると落ちる」経路である。

修正の段階案は次の 2 つに分かれる。

| 段階 | 内容 | 契約への影響 |
|---|---|---|
| 第 1 段 | FailureHandler を存在判定から `status`／`violations` 判定へ変える。artifact の schema とキーは変えない | 小。検証器は無改変で通る |
| 第 2 段 | 各 artifact を実カウンタから生成する。`scenarioResults` を実行時に作る。ISR 非実行時は `status:"not_run"` を追加する | 大。`isr-verify-v3..v10` と provenance、seed／generate／run の 3 スクリプトを同時に更新する必要がある |

第 1 段だけで「ok と Quarantine の併存」は消える。第 2 段は「定数出力」を消すが、CI 契約の同時更新が前提になる。どちらまで今回の範囲に含めるかを決めてほしい。推奨は第 1 段のみを先行させ、第 2 段は CI の別作業として切り出すこと。

## §2 提示 1（P1）

### B-03 mixed f1/f2 の NaN 受入

| 項目 | 内容 |
|---|---|
| 事象 | `setMixedTransitionStartHz`（`src/convolver/ConvolverProcessor.Runtime.cpp:1054`）は `juce::jlimit(MIXED_F1_MIN_HZ, MIXED_F1_MAX_HZ, hz)` の結果を無条件に格納する（:1071-1072）。`jlimit` は NaN を素通しするため、NaN が `pendingOverride` に入る。`setMixedTransitionEndHz`（:1089）も同型。end 側は `currentEnd < clamped + 10.0f`（:1063）が NaN で常に偽になるため、補正も効かない |
| 実害 | 到達経路はセッション復元である。`AudioEngine::requestLoadState`（`src/audioengine/AudioEngine.StateIO.cpp:211-213`）は `Convolver` 子ノードを検証せず `uiConvolverProcessor.setState(convState)` に渡し、`ConvolverProcessor::setState`（`src/convolver/ConvolverProcessor.StateAndUI.cpp:370-371`）が `mixedF1Hz`／`mixedF2Hz` をそのまま setter へ流す。検証関数 `validateConvolverStateTreeForDebug`（`AudioEngine.Parameters.cpp:13`）は存在するが、プリセット側の `validatePresetStateTreeForDebug`（:74）は呼出元がゼロで、`setConvolverStateTree` の検証も `JUCE_DEBUG` のときだけ（:793-802）。Release では素通りする。NaN が入ると mixed 遷移周波数が非有限になり、再構築後の IR が非有限になる |
| 仕様変更 | 不要。同じファイルに「非有限は無視して以前値を維持する」規約が既にある（`ConvolverProcessor.Runtime.cpp:1181-1182`、`:1207-1208` の `convo::numeric_policy::isFinite` ゲート）。これに揃えるだけ。既存の検証関数を復元経路に接続する案は、適用範囲が 6 項目に広がるため別項目として扱う |
| 修正範囲 | 2 関数の冒頭に非有限の早期 return を追加（各行 1 行 + コメント）。`ConvolverProcessor.h:393-395` の宣言コメントに契約を追記 |
| 回帰テスト | `AudioEngineHarness` に負例を追加。NaN／+Inf／−Inf を set した後、getter が以前の値を返すこと。`requestLoadState` に NaN を含む `Convolver` ノードを与え、状態が壊れないことも確認する。有効値の set→get も同時に確認する |
| 依存・リスク | なし。NaN 許容に依存する呼出元が無いことをテストで確認する |

## §3 提示 2（P2）

### B-04 WAV メタデータのアドミッション前確保

| 項目 | 内容 |
|---|---|
| 事象 | JUCE の WAV リーダー（`JUCE/modules/juce_audio_formats/codecs/juce_WavAudioFormat.cpp:1348-1390`）は、`bext`／`smpl`／`inst`／`cue ` の宣言長 `length`（uint32）で `calloc(jmax((size_t) length + 1, sizeof(...)))` を実行する。クランプは無い。`axml`／`iXML` も宣言長で `readIntoMemoryBlock` する（:1379-1390） |
| 実害 | 数 KB の細工 WAV で最大 4 GiB の確保要求になる。`createReaderFor` の内側なので、アプリ側の admission より前に走る。B-01 の修正後も、`LoaderThread`（`ConvolverProcessor.LoaderThread.cpp:402`）と `ResampleAndFallback`（`:285`）の 2 経路は未捕捉のまま |
| 仕様変更 | 不要。B-01 と同じ「admission の前に確保しない」原則の適用 |
| 修正範囲 | アプリ側に RIFF チャンク走査ヘルパを追加し、3 か所の `createReaderFor` 前に「宣言長がファイル残量を超えるチャンクがあれば拒否」を入れる。JUCE は改変しない（vendored 4,359 ファイルを上流に追従させるため） |
| 回帰テスト | `bext`／`smpl`／`inst`／`cue ` の宣言長を実体より大きくした WAV を 3 経路に与え、拒否されること。正常な WAV（メタデータ付きを含む）が従来どおり読めること |
| 依存・リスク | B-01 の admission 配置に依存する。走査を厳しくしすぎると正当な WAV を落とすため、「宣言長 > ファイル残量」の 1 条件だけに限定する |

### B-05 XmlDocument の無制限再帰・実体展開

| 項目 | 内容 |
|---|---|
| 事象 | `XmlDocument::readChildElements`（`JUCE/modules/juce_core/xml/juce_XmlDocument.cpp:503`）は深さ引数を持たず、:453 で自己再帰する。外部実体の展開（:804-895）にも上限が無い。アプリ側 6 か所（`DeviceSettings.cpp:854/917/1002/1085`、`MainWindow.cpp:1648`、`NoiseShaperLearner.cpp:1424`）はサイズも深さも検査せずに `parse` する |
| 実害 | 深い入れ子の小さい XML（数十 KB）でスタックが枯渇し、プロセスが終了する。外部実体参照を含む XML ではローカルファイルの読み出しが成立し得るが、入力はユーザーが選んだファイルに限られる |
| 仕様変更 | 不要（入力堅牢化）。ただし後述の「どこまで直すか」は判断が要る |
| 修正範囲 | (a) アプリ側の共通ヘルパでファイルサイズ上限と最大深さの事前スキャンを実装し、6 か所に適用する。(b) 深さ上限を厳密にするには JUCE の `readChildElements` に深さ引数を追加する改変が必要で、vendored ツリーに上流差分が残る |
| 回帰テスト | 深い入れ子 XML の拒否、6 か所それぞれの正常読込の回帰、外部実体参照を含む XML の拒否 |
| 依存・リスク | (a) だけでは実体展開を止められない。厳密に止めるなら (b) が要る。どちらを採るかは Owner 判断 |

### B-06 プリセット XML のサイズ上限なし

| 項目 | 内容 |
|---|---|
| 事象 | `src/MainWindow.cpp:1646-1653` は拡張子 `.xml` だけを確認し、:1648 で `XmlDocument::parse(file)` を呼ぶ。サイズの事前検査が無い |
| 実害 | B-05 と同じ。入口がユーザーのファイル選択なので独立に計上している。巨大 XML で UI スレッドが停止し、深さ次第で終了する |
| 仕様変更 | 不要 |
| 修正範囲 | B-05 のヘルパを chooser の分岐に適用し、拒否時に `setLastError` 相当の表示を出す（1 か所） |
| 回帰テスト | 上限超の XML を選ぶとエラー表示になり、`requestLoadState` が呼ばれないこと。正常 XML の回帰 |
| 依存・リスク | B-05 のヘルパに依存 |

### B-07 学習器保存の無同期読取

| 項目 | 内容 |
|---|---|
| 事象 | `getState`（`src/NoiseShaperLearner.cpp:379-390`）が `optimizer.serializeTo`（`src/CmaEsOptimizer.h:73`）を無ロックで読む。`CmaEsOptimizer` に mutex も atomic も無く、学習スレッドの `update()` と同じ配列を並行して触る。`publishGenerationResult` は :1489-1490 でスナップショットを取るが、:1493 のジョブはそれを渡されず、`saveLearnedState` の :1415 で読み直す |
| 実害 | 保存される mean が世代混交になる。ウォームスタートの初期値が「ある世代の一部」になり、再開後の収束が不定になる。データレース自体は C++ の未定義動作 |
| 仕様変更 | 不要 |
| 修正範囲 | `saveLearnedState` に `const State&` を受ける経路を追加し、ジョブへスナップショットを値渡しする。`NoiseShaperLearner.cpp` の 2 か所とヘッダの宣言 |
| 回帰テスト | 学習中の保存で、保存された mean の全要素が単一世代に由来することを確認する（保存直前に世代番号を記録して照合）。TSan 構成で当該テストを 1 回実行する |
| 依存・リスク | スナップショットを渡すと保存内容がわずかに古くなる。これは許容する |

### B-08 IR キャッシュの跨セッション成長

| 項目 | 内容 |
|---|---|
| 事象 | `evictLRU`（`src/CacheManager.cpp:580-609`）は `cacheMap` だけを見る。`cacheMap` は当セッションで `touch` したエントリのみ（:527-547）で、ディレクトリは走査しない。上限 10 件（`src/CacheManager.h:54`）はセッション内でしか効かない |
| 実害 | `IRCache` が起動ごとに増え続ける。1 セッション 10 件 × 数 MB〜数十 MB が積み上がり、ディスクを圧迫する。`clear()` を手で呼ぶまで減らない |
| 仕様変更 | 要判断。保持件数とは別に「総バイト上限」と「剪定のタイミング」を決める必要がある。推奨は総バイト 512 MB、起動後の初回ロード時に 1 回走査、超過分は mtime の古い順に削除 |
| 修正範囲 | `CacheManager` に `pruneDirectory(maxBytes)` を追加し、呼出を 1 か所（初回ロード）に置く。`isEntrySafeToDelete` と整合させる |
| 回帰テスト | ダミー 20 ファイルを置いて起動し、上限以下に剪定されること。使用中エントリが消えないこと。v1／v2 の壊れたファイルは B-02 の経路で消えること |
| 依存・リスク | B-02 の検証済みヘッダ判定に依存。他バージョンの `.bin` を消さないよう、magic と version で対象を限定する |

### B-09 適応係数の無ゲート保存・復元

| 項目 | 内容 |
|---|---|
| 事象 | 受口 2 つ（`src/audioengine/AudioEngine.Learning.cpp:481-501`、`:535-556`）は nullptr と個数だけを検査し、`stagedCoefficients[i] = coeffs[i]` で逐語コピーする。復元は `AudioEngine.StateIO.cpp:130 → :137` と `NoiseShaperLearner.cpp:1439-1453`（`learned_state.xml`）。`DeviceSettings.cpp:1164/1196` はサニタイズ済みで、ここだけ抜けている |
| 実害 | NaN／Inf が bank に入る。適用点の `clampCoeff`（`LatticeNoiseShaper.h:113-120`）でクランプされるため音響被害は無いが、getter が NaN を返すと UI 表示と比較が壊れ、保存→再読で NaN が固定化する |
| 仕様変更 | 不要（`DeviceSettings` と同型のサニタイズ） |
| 修正範囲 | 受口 2 か所に非有限の拒否（以前値維持）、復元 2 経路に同じ判定を追加。計 4 点、各 3〜5 行 |
| 回帰テスト | 細工したセッションと `learned_state.xml` を読み、bank が汚染されないこと。有効値の復元が従来どおりであること |
| 依存・リスク | なし |

### B-11 ISR 証跡の定数出力・存在のみ判定

| 項目 | 内容 |
|---|---|
| 事象 | `src/audioengine/ISREvidenceExporter.cpp:298-313` が 11 個の artifact を定数で書き出す。`FailureHandler::handleFailure`（:430-449）は `hb_violation_report.json` の存在だけで Quarantine を選ぶ（:433、:443） |
| 実害 | 証跡が実測を表さない。同じ run で `status:"ok"` と `Quarantine` が併存する。アプリの出力は CI 検証器の必須キーを満たさず、そのまま検証すると v5・v6・v7 で 4 件落ちる（§1.2） |
| 仕様変更 | 要。「証跡は実測のみ」「存在判定をやめる」は運用契約の変更にあたる |
| 修正範囲 | 第 1 段: FailureHandler を `status`／`violations` 判定へ（1 ファイル、契約無改変）。第 2 段: 各 artifact を実カウンタから生成し、`scenarioResults` を実行時に作る（同一ファイル + 検証器 9 本前後 + seed／generate／run の 3 本） |
| 回帰テスト | `isr-verify-v3..v10` をローカルで実行し、生成→検証が通ること。`ok` と `Quarantine` が同一 run で併存しないこと。provenance の strict mode が通ること |
| 依存・リスク | 第 2 段は CI を赤にし得る。契約更新と不可分。推奨は第 1 段のみ先行 |

## §4 提示 3（P3）

### B-10 最小位相の >2²¹ 無言スキップ

§1.1 の判断後に設計を確定する。判断が B 案なら、修正範囲は `LoaderThread` の分岐、`IRState` の実効位相モード、`ConvolverControlPanel` の表示の 3 点になる。

| 項目 | 内容 |
|---|---|
| 実害 | Minimum／Mixed を選んでも実際は線形位相で動く。レイテンシとプリリンギングの特性が選択と一致しない。音は出るため気付きにくい |
| 仕様変更 | 要（§1.1） |
| 回帰テスト | 2²¹+1 サンプル超の IR を Minimum と Mixed で読み、警告が出て線形として適用されること。閾値以下の IR では従来どおり最小位相化されること |
| 依存・リスク | 仕様決定が前提 |

### B-12 T-1／T-2 のテスト自己参照

| 項目 | 内容 |
|---|---|
| 事象 | `src/tests/TailLayerIntegrityTests.cpp` の H 系は block 256 の実測（同じ被検コード）を基準に比較する（:377、許容差 0.1 dB／2.5 dB）。H-4 は `eFixedDb > -60` のみ（:307）。F-5 は形状だけを見て音声を流さない（:320-330） |
| 実害 | 全ブロック共通の均一損失（例: 全帯域 −6 dB）を検出できない。テストが緑のまま品質が落ちる。block (32768, 524288] の経路は未検証のまま |
| 仕様変更 | 不要（テストの妥当性の問題） |
| 修正範囲 | 基準を独立参照（素朴畳み込み、または入力エネルギーからの解析値）に変更する。F-5 に実音声ケースを追加する。1 ファイル |
| 回帰テスト | テスト自身の負例。均一損失を注入したときに落ちること |
| 依存・リスク | 基準を変えると既存の許容差を再調整する必要がある |

### B-13 ASIO ブラックリストの無言失敗

| 項目 | 内容 |
|---|---|
| 事象 | `src/MainWindow.cpp:267-276` で `blacklistFile.replaceWithText(...)` の結果を見ず、`asioBlacklist.loadFromFile(...)` の戻り値も見ない。`AsioBlacklist.h:15-19` はファイル不在で false を返す |
| 実害 | exe が書込不可の場所（Program Files 配下など）にあると、既定ブラックリストが作られないまま空リストで起動し、ASIO4ALL や BRAVO-HD が除外されない。通知も無い |
| 仕様変更 | 不要 |
| 修正範囲 | 失敗時にユーザーディレクトリへフォールバックし、それも失敗したら UI に通知する。`MainWindow.cpp` 1 か所とヘッダ |
| 回帰テスト | 書込不可ディレクトリを模擬し、フォールバック先が使われること。両方失敗時に通知が出ること |
| 依存・リスク | なし |

### B-14 `--cli-log-file` の FileLogger リーク

| 項目 | 内容 |
|---|---|
| 事象 | `src/MainWindow.cpp:406-407` が `fileLogger.release()` で所有権を捨てる。`juce::Logger::setCurrentLogger` は logger を所有しない |
| 実害 | CLI 実行で 1 オブジェクトとファイルハンドルが終了まで残る。終了時に OS が回収するため実害は僅少だが、リーク検査で検出される |
| 仕様変更 | 不要 |
| 修正範囲 | `juce::FileLogger` をアプリ側のメンバで保持し、`setCurrentLogger(ptr)` に変える。1 ファイルとヘッダ 1 メンバ |
| 回帰テスト | ASan／LSan 構成で CLI を実行しリーク 0。ログ出力内容の回帰 |
| 依存・リスク | なし |

### B-15 学習器の二重 tanh 公開

| 項目 | 内容 |
|---|---|
| 事象 | 呼出元（`src/NoiseShaperLearner.cpp:975 → :979`）は `CmaEsOptimizer::toParcor`（`src/CmaEsOptimizer.h:194-198` で tanh 済み）を渡す。`publishGenerationResult`（:1459-1468）が再度 `std::tanh` を掛ける |
| 実害 | 公開・保存される `bestCoefficients` が評価した parcor と一致しない（絶対値が小さく出る）。`learned_state.xml` 経由で再開すると初期値もずれる |
| 仕様変更 | 不要（公開契約を「parcor を受ける」に統一する） |
| 修正範囲 | `publishGenerationResult` の tanh を削除し、引数名を parcor に合わせる（1 関数）。呼出元は 1 か所だけで、既に parcor を渡している |
| 回帰テスト | 公開値と評価値の数値一致。`learned_state.xml` の保存→読込の往復 |
| 依存・リスク | 既存の保存ファイルは二重 tanh 済みの値を持つ。読込時に変換しないなら、移行の要否を判断する必要がある |

## §5 別枠 — B-16 の判定変更提案

B-16 は「`src/tools/build_identity_gate.py:438-439` の pathspec `src/*.cpp` がサブディレクトリに一致せず、production src の差分を見落とす」という指摘だった。これを git 2.56.0 で実測した結果、前提が成立しない。

```
$ git diff HEAD --name-only -- 'src/*.cpp'          → src/convolver/... と src/tests/... を含む
$ git diff HEAD --name-only -- ':(glob)src/*.cpp'   → src/*.cpp 直下のみ
$ git diff HEAD --name-only -- ':(glob)src/**/*.cpp'→ 全階層を含む
```

git の pathspec は既定で `*` が `/` を跨ぐ。`/` を跨がせないのは `:(glob)` を付けたときである。したがって現行の pathspec はサブディレクトリを検出できており、「production src/ diff = 0 を誤って出す」は起こらない。

| 項目 | 内容 |
|---|---|
| 実害 | なし（前提が誤り） |
| 仕様変更 | 不要 |
| 修正範囲 | なし |
| 回帰テスト | 上記 3 コマンドの出力を記録し、将来 git の挙動が変わった場合に検出できるようにする |

なお、`git diff HEAD` は未追跡ファイルを出さないため、新規の未追跡 `.cpp` は「diff = 0」に寄与しない。これは B-16 とは別の小さなギャップであり、必要なら別項目として起票する。

B-16 は「非バグ（前提不一致）」へ判定を変更することを提案する。番号は維持し、N-01〜N-12 には触れない。

## §6 実施条件（承認後）

- 1 項目 1 変更単位とし、項目ごとに既存テスト全緑を確認してから次へ進む。
- 実行時再現とメモリ実測を行っていない項目は、報告でも「未実施」と明記する。
- B-04／B-05／B-06 は入力堅牢化の共通ヘルパを 1 つにまとめ、3 項目を続けて実施する。
- B-07／B-09／B-15 は学習器サブシステムで同じファイル群を触るため、続けて実施する。
- B-11 の第 2 段は CI 契約の更新と同時に行い、単独では入れない。
- B-10 は仕様決定後に着手する。
- B-12／B-16 は製品コードではなくテスト／検証ツールなので、リリース判断への影響は小さい。

## §7 見積り

| 項目 | 変更ファイル数（見込み） | 追加テスト | 仕様判断 |
|---|---|---|---|
| B-03 | 2 | 1 群 | 不要 |
| B-04 | 4（ヘルパ + 3 経路） | 2 群 | 不要 |
| B-05 | 2 +（JUCE 改変なら 1） | 3 群 | 深さ対応の範囲 |
| B-06 | 1 | 1 群 | 不要 |
| B-07 | 2 | 1 群 + TSan | 不要 |
| B-08 | 2 | 1 群 | 上限値と時期 |
| B-09 | 2 | 1 群 | 不要 |
| B-10 | 3 | 1 群 | 要（§1.1） |
| B-11 | 1（第 1 段）／約 14（第 2 段） | CI 実行 | 要（§1.2） |
| B-12 | 1 | 負例 1 群 | 不要 |
| B-13 | 2 | 1 群 | 不要 |
| B-14 | 2 | 1 群 | 不要 |
| B-15 | 1 | 1 群 | 移行の要否 |
| B-16 | 0 | 記録のみ | 判定変更 |
