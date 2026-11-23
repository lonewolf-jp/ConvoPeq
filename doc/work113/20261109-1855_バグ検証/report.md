# 検証レポート記載のバグ候補 — ソース検証（訂正版・識別子確定）

- 対象レポート: `C:\Users\user\Desktop\ConvoPeq 構造・妥当性検証レポート.html`（Part 20、HEAD `c94d376`）
- 検証対象: `C:\VSC_Project\ConvoPeq` main `c94d3767`（作業ツリーの変更なし）
- 検証: 2026-11-09、静的コード精読のみ（ビルド・実行なし）
- 範囲: 新規候補 7 件と継続項目 21 件（計 28 件）
- 訂正: 2026-11-09 19:40、識別子を **B-01〜B-16（バグ）/ N-01〜N-12（非バグ）** に確定し、表・本文・修正計画の対応を一致させた

## §0 識別子の確定

以前の版は「表の連番」と「§2 の #1〜#7」「§3 の 3.1〜3.12」が混在し、同じ `#1` `#2` が文脈によって別項目を指していた。本版では種別ごとの固定 ID を用いる。

- **B-01〜B-16**：バグと確定した 16 件。重大度と影響範囲を考慮した順。
- **N-01〜N-12**：非バグと確定した 12 件。
- 旧番号との対応は §1 の対応表に示す。本文と修正計画はこの ID のみを使う。

## §1 判定一覧（確定）

| ID | 候補 | 対象 | 重大度／区分 |
|---|---|---|---|
| B-01 | IRConverter が IR アドミッションを迂回 | `IRConverter.cpp`、`LoadPipeline.cpp` | Medium |
| B-02 | CacheManager の時間領域サイズ不整合 | `CacheManager.cpp` | Low–Medium |
| B-03 | mixed f1/f2 の NaN 受入 | `ConvolverProcessor.Runtime.cpp` | Medium |
| B-04 | WAV メタデータのアドミッション前確保 | `juce_WavAudioFormat.cpp` | Low |
| B-05 | XmlDocument の無制限再帰・実体展開 | `juce_XmlDocument.cpp`、アプリ 6 か所 | Low |
| B-06 | プリセット XML のサイズ上限なし | `MainWindow.cpp:1648` | Low |
| B-07 | 学習器保存の無同期読取 | `NoiseShaperLearner.cpp`、`CmaEsOptimizer.h` | Low |
| B-08 | IR キャッシュの跨セッション成長 | `CacheManager.cpp` | Low |
| B-09 | 適応係数の無ゲート保存・復元 | `AudioEngine.Learning.cpp`、`StateIO.cpp` | Low |
| B-10 | 最小位相の >2²¹ 無言スキップ | `ResampleAndFallback.cpp`、`LoaderThread.cpp` | Low → **非バグ（2026-11-09 判定変更）** |
| B-11 | ISR 証跡の定数出力・存在のみ判定 | `ISREvidenceExporter.cpp` | Low–Medium |
| B-12 | T-1／T-2 のテスト自己参照 | `TailLayerIntegrityTests.cpp` | Low |
| B-13 | ASIO ブラックリストの無言失敗 | `MainWindow.cpp`、`AsioBlacklist.h` | Low |
| B-14 | `--cli-log-file` の FileLogger リーク | `MainWindow.cpp:406–407` | Low |
| B-15 | 学習器の二重 tanh 公開 | `NoiseShaperLearner.cpp:975/1465` | Low |
| B-16 | build_identity_gate の pathspec | `build_identity_gate.py:439` | Minor → **非バグ（2026-11-09 判定変更）** |
| N-01 | FileLogger の無制限成長 | `MainApplication.cpp:83` | 実害僅少 |
| N-02 | 群遅延 120 の未開示 | `MixedPhase.cpp:322` | 文書 |
| N-03 | HB トレースの無制限増大 | `ISRHB.cpp:30` | 実害なし |
| N-04 | (F) の文書化欠落 | `README.md`、`RELEASE_NOTES.md` | 文書 |
| N-05 | Bypass のメモリ増大 | `MKLNonUniformConvolver.cpp:841–844` | 設計上の特性 |
| N-06 | ISR Verification が merge commit で未実行 | CI 運用 | 運用 |
| N-07 | C-1 CI の空転ステップ | `isr-verification.yml:372–379` | 意図した方針 |
| N-08 | D-1 コメントの陳腐化 | `MKLNonUniformConvolver.h:129` | 文書 |
| N-09 | V-1 警告の不可視 | `MKLNonUniformConvolver.cpp:660–664` | 可視性 |
| N-10 | E-1 閾値の緩和 | `PolyphaseGainFidelityTests.cpp:675–677` | 意図的ゲート |
| N-11 | HOLD コメントの陳腐化 | 同ファイル `:299` | 文書 |
| N-12 | README のファイル数 385 | `README.md` | 文書 |

集計：**バグ 16 件（B-01〜B-16）、非バグ 12 件（N-01〜N-12）**。重複なし。

### バグ 16 件の対応関係

| ID | 対象ファイル | 説明 | 重大度 | 実害 | 修正対象 |
|---|---|---|---|---|---|
| B-01 | `IRConverter.cpp`、`LoadPipeline.cpp` | IR アドミッションの迂回 | Medium | 未捕捉 `bad_alloc` で終了 | `loadAudioFile`／`convertFile`／`loadIR` |
| B-02 | `CacheManager.cpp` | 時間領域サイズの不整合 | Low–Medium | マップ外読み出し（クラッシュ） | `validateCacheFile`／`loadPreparedState` |
| B-03 | `ConvolverProcessor.Runtime.cpp` | mixed f1/f2 の NaN 受入 | Medium | NaN が pendingOverride へ | `setMixedTransitionStartHz`／`EndHz` |
| B-04 | `juce_WavAudioFormat.cpp` | メタデータの事前確保 | Low | admission 前の確保スパイク | アプリ側の事前検査 |
| B-05 | `juce_XmlDocument.cpp`、アプリ 6 か所 | 無制限の再帰・実体展開 | Low | スタック枯渇で終了 | パース前のサイズ・深さ検査 |
| B-06 | `MainWindow.cpp:1648` | プリセット XML のサイズ無制限 | Low | 巨大 XML で終了 | chooser のサイズ検査 |
| B-07 | `NoiseShaperLearner.cpp` | 保存の無同期読取 | Low | 保存 mean の世代混交 | `publishGenerationResult` |
| B-08 | `CacheManager.cpp` | 跨セッションのキャッシュ成長 | Low | ディスクの単調増加 | `evictLRU`／起動時走査 |
| B-09 | `Learning.cpp`、`StateIO.cpp` | 適応係数の無ゲート保存・復元 | Low | bank に NaN（適用時クランプ） | `setAdaptiveCoefficients*`／`requestLoadState` |
| B-10 | `ResampleAndFallback.cpp`、`LoaderThread.cpp` | 最小位相の無言スキップ | Low → **非バグ**（2026-11-09 判定変更） | 表示と実処理の乖離 | なし（入力が 2²¹ に cap され分岐に到達しない） |
| B-11 | `ISREvidenceExporter.cpp` | 証跡の定数出力・存在のみ判定 | Low–Medium | 証跡が実測を表さない | 定数出力の実測化 |
| B-12 | `TailLayerIntegrityTests.cpp` | テストの自己参照 | Low | 均一損失を検出不能 | 参照の独立化、F-5 に音声 |
| B-13 | `MainWindow.cpp`、`AsioBlacklist.h` | ブラックリストの無言失敗 | Low | 除外が効かない | 戻り値検査と通知 |
| B-14 | `MainWindow.cpp:406–407` | FileLogger のリーク | Low | CLI 時に 1 個リーク | `unique_ptr` の保持 |
| B-15 | `NoiseShaperLearner.cpp:1465` | 二重 tanh の公開 | Low | 未評価の係数を公開 | 再 tanh の削除 |
| B-16 | `build_identity_gate.py:439` | pathspec の不一致 | Minor → **非バグ**（2026-11-09 判定変更） | 誤った証跡 | なし（前提が実測で否定された） |

### 旧番号との対応

| 旧 | 新 | 旧 | 新 | 旧 | 新 | 旧 | 新 |
|---|---|---|---|---|---|---|---|
| 1 | B-02 | 8 | B-03 | 15 | N-04 | 22 | B-13 |
| 2 | B-01 | 9 | B-09 | 16 | N-05 | 23 | B-14 |
| 3 | B-04 | 10 | B-10 | 17 | N-06 | 24 | B-15 |
| 4 | B-05 | 11 | N-02 | 18 | B-12 | 25 | N-10 |
| 5 | B-07 | 12 | B-11 | 19 | N-07 | 26 | N-11 |
| 6 | B-08 | 13 | N-03 | 20 | N-08 | 27 | B-16 |
| 7 | N-01 | 14 | B-06 | 21 | N-09 | 28 | N-12 |

## §2 バグの詳細

### B-01 IRConverter が IR アドミッションを迂回 — バグ（Medium）

**レポートの主張**：IRConverter 経路だけが 8ch／1 GiB／リサンプル上限を持たず、全デコードをメモリに実体化する。

**確認した事実**

- `IRConverter::loadAudioFile`（`src/IRConverter.cpp:198`）の検査は「存在」「lengthInSamples が 1〜INT_MAX」「channels > 0」のみ（:212–218）。その後 `AudioBuffer<float>`（:220）と `AudioBuffer<double>`（:224）を確保する。8ch 上限・1 GiB バイト予算・出力 2²¹+1 サンプル上限は無い。
- 同じ IR を読む `LoaderThread` は 5 段の admission を通し（`src/convolver/ConvolverProcessor.LoaderThread.cpp:417–451`）、プレビューも同じ契約を適用する（`ResampleAndFallback.cpp:300–326`）。上限値は `IRLoadAdmission.h:39–52`。
- `AudioBuffer` は確保失敗で `std::bad_alloc` を投げ（`juce_HeapBlock.h:46`）、`loadIR` の `convertFile` 呼出（`LoadPipeline.cpp:295`）に try/catch は無い。

**到達性**：`requestConvolverPreset` は `loadIR` を直接呼ぶ（`AudioEngine.Parameters.cpp:212–215`）。呼出元は UI（`ConvolverControlPanel.cpp:1185`、プレビュー成功後）、CLI（`MainWindow.cpp:796/838`）、セッション復元の irPath（`StateAndUI.cpp:422`）。UI はプレビューで軽減されるが、セッションと CLI は素通りする。

**実害**：申告長の大きい WAV で float／double の巨大確保を試み、失敗すれば未捕捉例外でプロセスが終了する。

### B-02 CacheManager の時間領域サイズ不整合 — バグ（Low–Medium）

**主張**：`timeDomainSizeBytes` と channels×samples×8 の整合を検証せず、checksum も時間領域を覆わないため、細工キャッシュで過大確保＋マップ外 memcpy に至る。

**確認した事実**

- `validateCacheFile`（`src/CacheManager.cpp:125`）は magic・key（:170）・fftSize（:173）と総サイズ（:180–183）だけを検査する。channels×samples×8 との照合は無い。
- checksum は partition データのみ（:236）。ヘッダと時間領域は対象外。
- `loadPreparedState`（:219）は残量 ≥ timeDomainSizeBytes を確認して（:273）から `AudioBuffer<double>(channels, samples)` を確保し（:275–277）、`samples×8` で memcpy する（:283–287）。宣言を小さくすればマップ外を読む。
- 正常保存側は ch×samples×8 を正しく書く（:368）。

**到達性**：`loadIR` は `loadPreparedState` を 2 つのキーで試す（`LoadPipeline.cpp:287/298`）。キーは公開アルゴリズム（:84）で計算できるため、IR とキャッシュをセットで配布されると一致する。書込み先は `%APPDATA%\ConvoPeq\IRCache` で、ローカル書込みが前提。

**実害**：アクセス違反によるクラッシュ（DoS）。読み出した値は後段の有限性検査に掛かるため、情報漏えいの実用性は低い。

### B-03 mixed f1/f2 の NaN 受入 — バグ（Medium）

セッターの `setMixedTransitionStartHz`（`src/convolver/ConvolverProcessor.Runtime.cpp:1054`）は `juce::jlimit` の結果を無条件に格納する（:1071–1072）。`jlimit` は NaN をそのまま返すため、NaN が pendingOverride に入る。`setMixedTransitionEndHz`（:1089）も同型。到達経路はセッション復元（`StateAndUI.cpp` の mixedF1Hz／mixedF2Hz）。

### B-04 WAV メタデータのアドミッション前確保 — バグ（Low）

WAV の `juce_WavAudioFormat.cpp:1354–1375` は `bext`／`smpl`／`inst`／`cue ` を宣言長で `calloc(jmax(length+1, sizeof(...)))` する。`createReaderFor` の内部であり、アプリの admission より前に走る。`axml`／`iXML` も宣言長で読む（:1379–1386 付近）。失敗すれば B-01 と同じ未捕捉 `bad_alloc` に至る。

### B-05 XmlDocument の無制限再帰・実体展開 — バグ（Low）

`readChildElements` の再帰（`juce_XmlDocument.cpp:453 → 503`）に深さ上限がなく、外部実体も無制限（:819、:906）。アプリ側 6 か所（`DeviceSettings.cpp:854/917/1002/1085`、`MainWindow.cpp:1648`、`NoiseShaperLearner.cpp:1424`）にサイズ・深さの検査が無い。深い入れ子の小さい XML でスタック枯渇 → 終了。

### B-06 プリセット XML のサイズ上限なし — バグ（Low）

`MainWindow.cpp:1646` は拡張子だけを確認し、`:1648` で `XmlDocument::parse(file)` を呼ぶ。ファイルサイズの事前検査が無い。B-05 と同根だが、入口が別（ユーザーが開くプリセット／セッション）なので独立に計上する。

### B-07 学習器保存の無同期読取 — バグ（Low）

オプティマイザ側では `CmaEsOptimizer` に mutex・atomic が無い（`src/CmaEsOptimizer.h:73/107/132`）。`getState` は `serializeTo` を無ロックで読み（`src/NoiseShaperLearner.cpp:379–390`）。`publishGenerationResult` は 1490 でスナップショットを取るがジョブへ渡さず、ジョブ側が 1415 で再読する。保存 mean が世代混交する。

### B-08 IR キャッシュの跨セッション成長 — バグ（Low）

削除処理の `evictLRU`（`src/CacheManager.cpp:468`）は `cacheMap` のみを対象にし、ディレクトリを走査しない。`cacheMap` は `touch`（:409）されたエントリのみ。上限 10 件（`CacheManager.h:51`）はセッション内でしか効かず、IRCache は跨セッションで増え続ける（手動 `clear()` のみ）。

### B-09 適応係数の無ゲート保存・復元 — バグ（Low）

係数の受け口 `AudioEngine::setAdaptiveCoefficientsForSampleRate`（`src/audioengine/AudioEngine.Learning.cpp:481`）は nullptr と個数のみを検査し、`stagedCoefficients[i] = coeffs[i]`（:498）で逐語コピーして bank へ渡す（:500）。セッション復元（`StateIO.cpp:130 → :137`）が無検証、`learned_state.xml` も無検証（`NoiseShaperLearner.cpp:1424/1442/1453`）。`device_settings.xml` はサニタイズ済み（`DeviceSettings.cpp:1164/1196`）。適用点の `clampCoeff`（`LatticeNoiseShaper.h:113–120`）でクランプされるため音響被害は無い。

### B-10 最小位相の >2²¹ 無言スキップ — バグ（Low）

変換関数 `convertToMinimumPhase`（`ResampleAndFallback.cpp:370`）は `MAX_MINPHASE_FFT_SIZE = 8388608`（:380）超で空を返す（:381–384）。呼出側（`LoaderThread.cpp:884`）は `validateBuffer` が偽なら置き換えず（:887）、元の IR のままロード成功する（:880–908）。表示は Minimum／Mixed のまま残る。

**【判定変更 2026-11-09】非バグ（前提不一致）へ変更。**

実行時検証で、この分岐に到達しないことを確認した。`LoaderThread.cpp:769-771` は `targetLength = computeTargetIRLength(...)` を計算し、`stepTrimmed.setSize(channels, targetLength)` してから変換へ渡す。`computeTargetIRLength`（`StateAndUI.cpp:997-1023`）は `min(sr × targetIRTimeSec, MAX_IR_LATENCY)` を返し、`MAX_IR_LATENCY = 2097152`（2²¹）である。したがって変換入力 N ≤ 2²¹、`nextPowerOfTwo(N × 4) ≤ 2²³ = 8388608` となり、`> 8388608` の条件は成立しない。2²¹+1 サンプルの WAV を実際にロードする統合テストでも、Minimum が適用され警告は出なかった。混合位相側の上限（`MixedPhase.cpp:171`、`:744`）も同様に到達しない。`convertToMinimumPhase` の呼出元は 1 か所だけである。

B-10 は修正対象から除外する。詳細は `doc/work113/20261109-2145_残件実装/report.md` §3.1／§4.2 に記載した。

### B-11 ISR 証跡の定数出力・存在のみ判定 — バグ（Low–Medium）

証跡の生成側 `ISREvidenceExporter.cpp` は `closure_graph.json`（:299）、`hb_graph_trace.json`（:301）、`hb_violation_report.json`（:302）、`retire_timeline.json`（:306）、`payload_tier_report.json`（:309）を、実測を伴わず定数の JSON として書き出す。`shutdown_trace.json`（:307）と `retire_latency_report.json`（:308）は template。FailureHandler は `hb_violation_report.json` の存在だけで Quarantine を記録する（:433/:443）。CI／Debug では「ok」と「Quarantine 要」が同じ run で併存する。

### B-12 T-1／T-2 のテスト自己参照 — バグ（Low）

`TailLayerIntegrityTests.cpp` の H 系は block 256 の実行結果（同じ被検コード）を基準に比較し（:377）、許容差は 0.1／2.5 dB（:377）、H-4 は `eFixedDb > −60` のみ（:307）。全ブロック共通の均一損失を検出できない。F-5 は形状のみで音声を流さない（:320–330）ため、block (32768, 524288] の経路は未検証。

### B-13 ASIO ブラックリストの無言失敗 — バグ（Low）

ブラックリストの生成は `MainWindow.cpp:269–275` で、`replaceWithText` の結果も `loadFromFile` の戻り値も見ない。`AsioBlacklist.h:15` はファイル不在で false を返すため、書込み不可の場所では空リストのまま除外が効かない。UI 通知も無い。

### B-14 `--cli-log-file` の FileLogger リーク — バグ（Low）

`MainWindow.cpp:406–407` が `fileLogger.release()` で所有権を放棄する。JUCE の `setCurrentLogger` は logger を所有しないため、CLI 実行時に 1 オブジェクトとハンドルがプロセス終了まで残る。

### B-15 学習器の二重 tanh 公開 — バグ（Low）

`publishGenerationResult` の呼出元は `toParcor` の結果（`CmaEsOptimizer.h:194–198` で tanh 済み）を渡す（`NoiseShaperLearner.cpp:975 → 979`）。関数内で再度 `tanh`（:1465）。評価した係数集合と公開値が一致しない。

### B-16 build_identity_gate の pathspec — バグ（Minor）

`src/tools/build_identity_gate.py:439` の `src/*.cpp` はサブディレクトリに一致しない（git の pathspec は `/` を跨がない）。`:440` の「production src/ diff = 0」を誤って出し得る。

**【判定変更 2026-11-09】非バグ（前提不一致）へ変更。**

git 2.56.0 で実測した結果、`git diff HEAD --name-only -- 'src/*.cpp'` は `src/convolver/...` と `src/tests/...` を含んだ。git の pathspec は既定で `*` が `/` を跨ぎ、跨がせないのは `:(glob)` を付けた場合である（`:(glob)src/*.cpp` は直下のみ、`:(glob)src/**/*.cpp` は全階層）。したがって「サブディレクトリに一致せず production src/ diff = 0 を誤って出す」という前提は成立しない。

この実測は git 2.56.0 の挙動であり、他のバージョンへ一般化しない。コード変更は行っておらず、B-16 は修正対象から除外する。根拠と経緯は `doc/work113/20261109-2145_残件実装/report.md` §4 に記載した。

## §3 非バグの詳細

| ID | 判定理由 |
|---|---|
| N-01 FileLogger | 起動毎に削除され（`juce_FileLogger.cpp:44 → 69–73`）、RT のログは診断マクロ内。実測 32 KB 級で機能障害なし |
| N-02 群遅延 120 | `maxAllowedGD = 120.0`（`MixedPhase.cpp:322`）は意図的制約。UI・マニュアルに記載が無いのは文書ギャップ |
| N-03 HB トレース | 上限は無いが、呼出は publish／retire の 3 箇所のみ（`Commit.cpp:384/479`、`ReleaseResources.cpp:775`）。増加は無視できる |
| N-04 (F) 文書化 | README は遷移（20 ms）のみで保持変更に触れない。コードの欠陥ではなく文書の欠落 |
| N-05 Bypass メモリ | L1/L2 が IR 全長分を確保するのは機能の副作用。入力 1 GiB 予算は展開後を制限しないが、不具合ではない |
| N-06 ISR Verification | merge commit では未実行だが、PR ヘッド `ba3604f`／`d1e35b9` で success。検証の実体はある |
| N-07 C-1 CI 空転 | 非ブロッキングは `isr-verification.yml:372–379` に方針として明記。意図された設計 |
| N-08 D-1 コメント | `MKLNonUniformConvolver.h:129` の記述が実装と不一致。動作に影響なし |
| N-09 V-1 警告 | 32768 超は Logger のみで UI 非表示。可視性の改善点で、実害なし |
| N-10 E-1 閾値 | 旧値 −9.542425 は旧規約固有。−9.442 は「悪化しない」境界で、DC は別途 1e-9 と bitwise で固定 |
| N-11 HOLD コメント | `PolyphaseGainFidelityTests.cpp:299` が現状と矛盾。動作に影響なし |
| N-12 README 385 | `git ls-files src` は 387。文書の陳腐化 |

## §4 レポート記述の訂正

| レポートの記述 | 本検証の結果 |
|---|---|
| `requestConvolverPreset` の dispatch 先が未確認 | `loadIR`（同期）で確定（`AudioEngine.Parameters.cpp:212–215`） |
| プリセット／セッション XML の上流パース箇所を特定できず | `MainWindow.cpp:1648` で確定 |
| ファイル数 388（Part 16） | 387 |
| `loadPreparedState` の int キャスト下流が未追跡 | 実体はサイズ整合の欠落による OOB（B-02） |
| ISR Verification が未実証 | PR ヘッド 2 件で success、merge commit では未実行 |

## §5 検証上の制約（維持）

- 実行時再現は未実施。B-01／B-02 のクラッシュ・OOB は**コード上の成立**までで、実行で再現したものではない。
- メモリ使用量の実測は未実施。B-01 の「8 GB／16 GB」は確保サイズの計算値である。
- RES-06（STG-8 のフレーク）は**保留**のまま。
- 非バグ 12 件の判定は変更しない。修正対象に読み替えない。
- 本報告は検証結果であり、実装結果ではない。

## §6 修正計画

2 件の優先設計と残り 14 件の計画は `doc/work113/20261109-1940_修正設計/report.md` に分離した。本指示ではソース変更・コミット・マージ・リリース・公開告知を行っていない。
