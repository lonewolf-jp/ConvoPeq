# 外部監査バグ主張の詳細検証（第2次）

- 実施: 2026-11-06
- 対象: `ConvoPeq 構造・妥当性検証レポート.html` のバグ候補 #1〜#12 と N1（全 13 件）
- 調査 HEAD: `da32ae6370303439bf51f18edd4df79bfec4c6be`（`main`、想定と一致）
- 作業ツリー: 追跡ファイルの変更 0 件。未追跡は既存の `doc/work113/` 配下のみ
- 制約: 調査と報告書作成のみ。ソース、テスト、CMake、CI は変更していない。コミットと PR も作成していない

## A. 調査概要

### A.0 用語と表記

外部監査の文言を引用する際は、原文の表現を尊重した。現行コードの識別子へ言い換えた箇所では、対応するファイル名と行番号を併記した。数値は独自に再計算し、近似や丸めを含む場合はその旨を明記した。判定は、機構の存在、到達可能性、影響の大きさを分けて記述した。

本調査では、観測できた事実と推論を分離した。観測は、ソース閲覧、式の再計算、既存バイナリの実行に限られる。観測できない事項は、未判定として残した。推論は、条件と前提を添えて記した。過大な一般化を避けるため、反例と境界例を探した。結果として、複数の主張で当初の深刻度を下げた。


### A.1 実施した検証手段

| 手段 | 内容 |
|---|---|
| 静的解析 | 対象関数の呼び出し元・呼び出し先、状態変数の設定と読み出し、分岐とガードを現行ソースで追跡 |
| 数値検証 | ハッシュ衝突の再現、SVF 係数の応答計算、最小位相スペクトルのアンラップと MixedPhase 後段の再現 |
| テスト実行 | `build/Release/PolyphaseGainFidelityTests.exe`（2026-11-04 ビルド）を実行 |
| 資料照合 | 外部監査 HTML から原文を抽出し、主張単位で現行コードと対照 |
| 差分確認 | `git log 771804d..HEAD -- <file>` で監査時点からの変更を確認 |
| 環境確認 | `build/CMakeCache.txt` のオプション値、`git status` の作業ツリー状態 |

### A.2 実施していない検証手段

- ASan／TSan の実行
- 新規ビルド（既存バイナリのみ使用）
- 実音声出力の測定、聴感評価
- 実機での IR 読込・MixedPhase 変換の実行
- 新規テストの作成と実行
- 外部監査の原資料（第4パス報告書本文、notes/pass4-01〜03）

### A.3 調査対象ファイル

`src/eqprocessor/EQProcessor.ProcessingCache.cpp`、`src/audioengine/AudioEngine.Cache.cpp`、`src/audioengine/AudioEngine.Snapshot.cpp`、`src/audioengine/AudioEngine.h`、`src/convolver/ConvolverProcessor.MixedPhase.cpp`、`src/convolver/ConvolverProcessor.Internal.h`、`src/convolver/ConvolverProcessor.StateAndUI.cpp`、`src/convolver/ConvolverProcessor.Runtime.cpp`、`src/convolver/ConvolverProcessor.LoadPipeline.cpp`、`src/convolver/ConvolverProcessor.LoaderThread.cpp`、`src/CustomInputOversampler.cpp`、`src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`、`src/audioengine/OversamplingPolicy.h`、`src/eqprocessor/EQProcessor.Parameters.cpp`、`src/eqprocessor/EQProcessor.Core.cpp`、`src/eqprocessor/EQProcessor.Coefficients.cpp`、`src/audioengine/AudioEngine.Parameters.cpp`、`src/MKLNonUniformConvolver.cpp`、`src/ConvolverProcessor.h`、`src/audioengine/SimplePeakLimiter.h`、`src/TruePeakDetector.cpp`、`JUCE/modules/juce_core/text/juce_CharacterFunctions.h`、`manual/IR_ADVANCED_WINDOW_EN.md`、`CMakeLists.txt`

## B. 主張別の判定一覧

| ID | 原文の主張 | 判定 | 重大度 | 発生機構 | 到達条件 | 再現結果 | 影響範囲 | 根拠 | 未検証事項 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ハッシュの符号衝突で係数キャッシュが誤再利用され、最悪 96 dB の誤差 | 妥当 | Major | 符号ビットを落としたハッシュ一致のみでキャッシュを返す | 同一バンドのゲインを符号だけ変えて再設定（通常 UI） | ハッシュ一致と応答差 12／96 dB を数値確認 | 音声（帯域中心）、状態 | `ProcessingCache.cpp:10-15,24-54`、`AudioEngine.Cache.cpp:51-128`、`AudioEngine.h:4370-4376` | 実音声での聴感、他パラメータの符号到達 |
| 2 | 片方向アンラップで 2π の不連続が残り、群遅延目標を誤る | 条件付き | Minor | 巻き戻しの累積を保持せず、上方向補正もない | Mixed 選択、かつ位相が 1 ステップで π を超えて回る急峻な IR | 生配列は最大 653 rad 不一致。後段リミッター後は帯域内 0.12 rad 以下 | 音声（Mixed 時のみ、限定的） | `MixedPhase.cpp:281-292,331-346` | 実 IR での聴感、大規模 IR の掃引 |
| 3 | ポリフェーズ利得が段あたり −2.50 dB、補正は既定 OFF | 妥当 | Major | 中心タップの ×2 がマクロ内のみ。補償経路なし | OS 有効時（Auto で 48 kHz→×8）、SoftClip 時は OS=1 でも局所 2 倍 | テストで 0.75／0.5625／0.421875 を確認 | 音声（レベル） | `CustomInputOversampler.cpp:549-552`、`CMakeLists.txt:134`、`DSPCoreDouble.cpp:354,534` | 実聴感、他 SR での実測 |
| 4 | バンドの freq/gain/Q セッターに値検証がない | 妥当 | Minor | バンド番号のみ検証し、値をそのまま格納 | 破損プリセット／セッションの復元（XML の nan/inf も可） | 上位層に事前検証なしを確認 | 状態、係数（無言バイパス） | `EQProcessor.Parameters.cpp:19-74`、`EQProcessor.Core.cpp:588-609`、`AudioEngine.Parameters.cpp:139-141` | UI 経由の非有限値到達 |
| 5 | setNonlinearSaturation が NaN を公開する | 妥当 | Minor | `jlimit` が NaN を素通しし、変化検出も発火しない | 破損セッションの復元 | `jlimit` の実装で確認 | 状態、表示（飽和は無効化） | `EQProcessor.Parameters.cpp:227-247`、`juce_MathsFunctions.h:526-528` | UI 表示の実害 |
| 6 | Convolver setState の手動分岐が D8 ガードを迂回し、未定義動作になる | 一部妥当 | Minor | 手動分岐が jlimit のみで `autoDetectedIRLengthSec` に格納 | 破損セッション（手動オーバーライド＋NaN） | 迂回は確認。float→int の UB は不成立 | 状態、表示のみ | `StateAndUI.cpp:336-343,993-1019`、`Runtime.cpp:921-947` | 表示系の実害 |
| 7 | 不正な filterStructure が無言で Serial に変換される | 妥当 | Minor | 不正値を Serial に強制 | 破損セッション | コードで確認 | 音声（直列／並列の変化） | `EQProcessor.Parameters.cpp` の setFilterStructure | 聴感差 |
| 8 | フォールバックの位相回転にパディングがない | 妥当（NOT） | Info | 円環シフトとして正確。近似誤差は有界 | 2 の冪長の IR、フォールバック時のみ | コードで確認 | 限定的 | `MixedPhase.cpp:749,837` | 誤差量の実測 |
| 9 | SetImpulse に stderr プローブが残る | 妥当 | Info | 無条件の `fprintf` | IR 読込・再構築のたび | コードで確認 | ログのみ（非 RT） | `MKLNonUniformConvolver.cpp:1059-1092` | 出力行数の実測 |
| 10 | LoaderThread の世代チェックが未実装 | 妥当（NOT） | Info | コメントのみ陳腐化。commit 側ガードが実在 | — | ガード 7 箇所を確認 | なし | `LoadPipeline.cpp:59,223,253,297,404,827`、`LoaderThread.cpp:642-654` | なし |
| 11 | 既定テール整形が文書と異なる | 妥当（NOT、文書問題） | Info | 既定は Layer Tail Contouring、マニュアルは Air Absorption | 既定設定 | コードとマニュアルを確認 | 文書のみ | `ConvolverProcessor.h:86`、`manual/IR_ADVANCED_WINDOW_EN.md:74` | なし |
| 12 | リミッター／TruePeak 等の内部状態が未監査 | 一部解消 | 未判定 | リミッターは NaN を保持しない。TruePeak と FixedDelay は未確認 | — | 一部確認 | 未判定 | `SimplePeakLimiter.h:44-82`、`DSPCoreDouble.cpp:692-715` | TruePeak 履歴、FixedDelay 再割当て |
| N1 | Bypass でテールが切断される | 未判定 | 未判定 | 報告書に機構の記載なし | — | 検証不能 | — | — | 原文・根拠コード |

## C. 詳細検証

### C.1 #1 ハッシュ符号衝突

**監査の主張。** 符号ビットを落とすため +x と −x が同一ハッシュになり、係数キャッシュが誤って再利用される。最悪 ±48 dB で 96 dB の誤差。

**該当箇所。** `EQProcessor.ProcessingCache.cpp:10-15` が `bits & 0x7FFFFFFF` を返す。ここでは `computeParamsHash`（`:24-54`）が 20 バンドの frequency/gain/q/enabled/type/channelMode、totalGainDb、agcEnabled、nonlinearSaturation、filterStructure、sampleRate、maxBlockSize を混ぜる。

**処理経路。** `AudioEngine.Snapshot.cpp:56-58` が `getOrCreate` を呼び、返った `paramsHash` をスナップショットへ格納する。RT 側は `AudioEngine.h:4373-4376` で `pinByHash(eqCoeffHash)` を呼び、ハッシュだけで係数を解決する。この関数（`AudioEngine.Cache.cpp:51-128`）は `map.find(hash)` のみで、パラメータの比較はない。`:111-116` で map をコピーして追記するため、エントリはエンジン稼働中に残る。

**衝突条件。** ハッシュは各値の「符号を除いたビット列」の結合である。したがって、対象フィールドが符号だけ異なり、他の全フィールドが一致すれば衝突する。値域のクランプ後に符号が残るのはバンドゲインと totalGainDb である。frequency と Q は負値が下限へクランプされ、同じ値になるため係数も同じになる。nonlinearSaturation は 0 未満が 0 に落ちる。よって音響的に意味のある衝突はバンドゲインである。

**数値確認。** ハッシュ関数を同じ定数で再実装した結果、±6、±12、±48 dB がそれぞれ一致した。

```
+6  -> 96a202e4c8d12609 / -6  -> 96a202e4c8d12609
+48 -> a536013d977aa7d5 / -48 -> a536013d977aa7d5
```

**係数差の定量化。** `calcPeakingSVF`（`EQProcessor.Coefficients.cpp:477-513`）の式を再実装し、48 kHz・1 kHz・Q=1 で応答を測った。

| 意図した設定 | 再利用される係数 | 帯域中心の実応答 | 誤差 |
|---|---|---|---|
| −6 dB | +6 dB の係数 | +6.000 dB | 12.000 dB |
| −12 dB | +12 dB の係数 | +12.000 dB | 24.000 dB |
| −48 dB | +48 dB の係数 | +48.000 dB | 96.000 dB |

「96 dB」は、意図した −48 dB と、再利用された +48 dB 係数の中心周波数応答との差である。帯域全体で一定の 96 dB ではない。中心から離れると差は縮む。

**成立条件。** 同じバンドのゲインを、符号だけ変えて再設定する。UI のスライダーで +6 dB から −6 dB へ動かすだけで成立する。キャッシュは消えないため、プリセットを跨いでも残る。

**影響。** 表示と実際の応答が一致しない。カット表示でブーストされるため、聴感上の不整合が大きい。

**前回との差異。** 前回は衝突の存在と 96 dB の算出を静的に示した。今回は SVF 応答を実装して 12／24／96 dB を実測し、符号が残るパラメータと残らないパラメータを切り分けた。

**判定。** 妥当、Major。修正は符号ビットを残すことと、ヒット時の全パラメータ比較。

### C.2 #2 位相アンラップ

**主張（原文）。** 片方向補正のため 2π の不連続が残り、群遅延目標を誤り、MixedPhase の結果が歪む。major（オプトイン）。

**実装。** `ConvolverProcessor.MixedPhase.cpp:281-292`。

```
delta = raw - phi[k-1]
if (delta > π) phi[k] = phi[k-1] + delta - 2π
else           phi[k] = phi[k-1] + delta
```

**数式上の差。** 上の式は `phi[k] = raw - 2π`（分岐内）または `phi[k] = raw`（分岐外）に簡約される。つまり、巻き戻しが起きるたびに累積オフセットを捨てる。標準のアンラップは補正量を累積するため、位相が −2π、−4π と下がり続ける場合に差が開く。上方向（delta < −π）の補正がない点も原文のとおりである。正しい実装は `ConvolverProcessor.Internal.h:33-46` にあり、フォールバック経路（`:837`）はこちらを使う。

**後段。** `:294-325` で重み付け（F1 以下は最小位相、F2 以上は線形位相）と、線形スペクトルの振幅 1e-10 未満の無効化を行う。`:331-346` で隣接ビンとの差が 120 サンプル相当を超えると前の値へ固定する。`:350-362` で群遅延を計算し、移動平均と EMA を経て目標とする。

**再現。** 最小位相スペクトルをケプストラム法で作り、`fftSize = 4 × 2^k` で位相を計算した。コードのループと `numpy.unwrap` を比較し、その後に重み付けとスロープリミッターを通した。

| 入力 | 生の不一致（最初のビン） | 帯域内の目標位相差（既定 200/1000） |
|---|---|---|
| 2 次低域通過 100 Hz | なし | 0 |
| 4 次低域通過 100 Hz | 22.2 kHz（使用帯域外） | 0 |
| 4 次低域通過 500 Hz | なし | 0 |
| 2 次高域通過 30 Hz | 229 Hz | 0.123 rad |
| 4 次高域通過 30 Hz | 229 Hz | 0.083 rad |
| 2 次高域通過 50 Hz | 229 Hz | 0（両者が同じ位置で固定） |

生の配列では最大 653 rad の差が出る。一方、スロープリミッターが 2π の跳びを「120 サンプル超の群遅延」として検出し、前の値で固定するため、目標位相に残る差は小さい。今回の入力群では帯域内で 0.13 rad 未満だった。

**成立条件。** Mixed を選び、最小位相の位相が 1 ステップで π を超えて回る IR を使う。急峻な低域特性や長い群遅延が条件になる。既定の遷移帯域（200〜1000 Hz）では、通常の低域通過 IR は使用帯域内で不一致が出なかった。

**影響。** 目標位相の誤差は限定的で、2π の跳びがそのまま音響へ出るわけではない。原文の「群遅延目標が誤り最適化結果の IR が歪む」は、少なくとも目標段階では最大 0.13 rad の差にとどまる。

**前回との差異。** 前回は生の配列の不一致だけを示した。今回は後段を通した目標位相まで追跡し、リミッターが差を吸収することを示した。重大度は major ではなく minor（条件付き）が妥当である。

**判定。** 条件付き、Minor。修正は `unwrapPhaseRadians` への置換で足りるが、優先度は低い。

### C.3 #3 ポリフェーズ利得

**原文はこう述べる。** 段あたり −2.50 dB の損失。補正は既定 OFF。SoftClip 有効時は OS=1 でも発生。

**コード。** `CustomInputOversampler.cpp:549-552`。畳み込み側は常に `*= 2.0`、中心側の `*= 2.0` は `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN` 内のみ。`decimateStage`（`:565-`）に 2 倍も補償もない。`CMakeLists.txt:134` の option は OFF で、`build/CMakeCache.txt` も `OFF`。マクロは 4 箇所の generator expression で対象ターゲットへ伝播する。

**本番経路。** `AudioEngine.Processing.DSPCoreDouble.cpp:354` の `processUp` から `:534` の `processDown` の間に、段数分の補償はない。あるのはユーザー設定の `outputMakeupGain` とコンボルバ入力トリムのみ。`:496-507` の局所 2 倍 OS も同じ部品を使う。

**実測。** `PolyphaseGainFidelityTests.exe` を実行した（1.58 秒、PASS=54、FAIL=0）。DC 利得は次のとおり。

```
r=2 dcBase=0.750000000000  dcCand=1.000000000000
r=4 dcBase=0.562500000000  dcCand=1.000000000000
r=8 dcBase=0.421875000000  dcCand=1.000000000000
production_flag = undefined/off
```

20·log10(0.75) = −2.498 dB。原文の −2.50／−5.00／−7.50 dB は正しい。

**テストの検出能力。** このテストは Shadow 参照モデルとの bitwise 一致を見る「Shadow fidelity gate」であり、production の正しさを判定する gate ではない（テスト冒頭 `:14-17` に明記）。PASS=54 は「実装が参照モデルと一致し、既知の 0.75 則を再現した」ことを示す。利得規約が正しいことや、聴感上のレベルが適正だとは限らない。production の補正は HOLD のままである（`:291-293`）。

**影響。** OS を切り替えるとレベルが段数に応じて下がる。補償がないため、ユーザーがメイクアップで合わせ直す必要がある。

**判定。** 妥当、Major。ただし開発側が認識済みの既知欠陥で、修正は Phase 0 契約テストの更新を伴う。

### C.4 #4 バンドセッターの入力検証

**外部監査の指摘。** freq/gain/Q に isfinite も jlimit もない。上位層の事前検証が未読のため LIKELY。

**現行の実装。** `EQProcessor.Parameters.cpp:19-74`。3 セッターともバンド番号の範囲だけ確認し、値をそのまま `EQState` に格納する。兄弟の `setTotalGain`（`:103-116`）は isFinite 拒否と ±48 dB クランプを行う。

**呼び出し経路。** UI は `EQControlPanel.cpp:323,331,339` から `AudioEngine` のインライン関数（`AudioEngine.h:1327-1329`）を経て同じセッターへ来る。復元は `EQProcessor::setState`（`EQProcessor.Core.cpp:588-609`）が ValueTree の値を直接渡す。`AudioEngine.Parameters.cpp:139-141` の事前検証は EQ 子ノードの型だけを見て、数値は見ない。

**非有限値の出所。** JUCE の `readDoubleValue`（`juce_CharacterFunctions.h:252-278`）は `nan`、`inf`、`-inf` の文字列を明示的に解釈する。セッション XML の属性にこれらの文字列が入ると、NaN／Inf がそのままセッターへ届く。

**状態変化。** NaN は格納される。係数計算では `validateAndClampParameters`（`:91-103`）の jlimit が NaN を素通しするが、各 SVF 関数の isfinite 防御がバイパス係数を返す。音声は NaN にならず、当該バンドが無言でフラット化する。`getState`（`Core.cpp:551-`）は生値を書き戻すため、保存と再読込で自己永続する。

**影響。** 音声の破綻ではない。設定の欠落と、復元後の再現性の欠如である。

**前回との差異。** 前回は上位層が未読で LIKELY だった。今回は事前検証がないことを確認し、BUG 相当と判断した。

**判定。** 妥当、Minor。

### C.5 #5 飽和処理と NaN

**該当箇所。** 対象は `EQProcessor.Parameters.cpp:227-247`。ここでは `juce::jlimit(0.0f, 1.0f, value)` のみを使う。この `jlimit` は `juce_MathsFunctions.h:526-528` で `value < lower ? lower : (upper < value ? upper : value)` と実装され、NaN は両方の比較が false のためそのまま返る。変化検出も NaN では false になり、早期 return しない。

**伝播。** NaN は atomic（`:234`）と `EQState`（`:240`）の両方へ入る。RT 側の `processBand` は `saturation > 0.0` が false になるため、飽和を無効化するだけである。`getState` が生値を書き戻すので、プリセットへ往復する。

**入力経路。** セッション XML の `nan`（C.4 と同じ）、または公開 API への NaN 渡し。UI スライダーは有限値しか出さない。

**影響。** 飽和が無言で無効になる。音声は壊れない。状態とハッシュには残る。UI 表示が `nan` になる可能性がある。

**判定。** 妥当、Minor。

### C.6 #6 D8 バイパス

**監査の主張。** 手動オーバーライド分岐が isfinite ガードを迂回し、NaN が `computeTargetIRLength` の float→int 変換へ達して未定義動作になる。

**実装。** 対象は `ConvolverProcessor.StateAndUI.cpp:336-343`。ここは `jlimit` のみで `pendingOverride.autoDetectedIRLengthSec` に格納する。この関数（`ConvolverProcessor.Runtime.cpp:921-947`）は isFinite 拒否を行うが、この分岐は通らない。

**データフロー。** `computeTargetIRLength`（`StateAndUI.cpp:993-1019`）が読むのは `pendingOverride.targetIRLengthSec` である。手動分岐は同じブロックで `setTargetIRLength(v.getProperty("irLength"))` を呼び（`:346-349`）、こちらは isFinite ガードを持つ。`autoDetectedIRLengthSec` の読み出しは `ConvolverControlPanel.cpp:1389` の表示ラベルだけである。

**帰結の判定。**

| 原文の帰結 | 成立するか |
|---|---|
| ガードの迂回 | 成立する |
| NaN の格納と状態への永続化 | 成立する |
| float→int の未定義動作 | 成立しない（target は別フィールド） |
| 過大確保・OOB | 成立しない（`computeTargetIRLength` は target のみ使用） |

**影響。** セッションに NaN が残り、表示に `nan` が出る。メモリ安全性の問題ではない。

**前回との差異。** 前回は同じ結論だったが、`autoDetectedIRLengthSec` の読み出し先を列挙して裏付けた。

**判定。** 一部妥当、Minor。

### C.7 #7 setFilterStructure

**コード。** `EQProcessor.Parameters.cpp` の `setFilterStructure` は、`Serial` でも `Parallel` でもない値を `Serial` へ強制変換する。兄弟の `setBandType` と `setBandChannelMode` は不正値を拒否して現状維持する。

**経路。** `EQProcessor.Core.cpp:594-595` が ValueTree の int を cast して呼ぶ。通常 UI は 0／1 しか出さない。破損セッションでのみ到達する。

**影響。** 意図した並列構成が直列になり、複数バンドが重なる帯域の応答が変わる。無通知である。

**判定。** 妥当、Minor。

### C.8 #8 フォールバックの位相回転

**現行の実装。** 対象は `MixedPhase.cpp:749`。`fftSize = nextPowerOfTwo(numSamples)` で、×4 パディングはない。`:837` で `unwrapPhaseRadians` を適用し、`:839-` で bin ごとのフェーザ乗算を行う。

**評価。** 周波数領域のフェーザ乗算は円環シフトとして正確に実装される。線形遅延としてのテール回り込みは、長さが 2 の冪のときに生じる有界な近似誤差である。フォールバックはオールパス設計が空を返した場合のみ使われる（`:45-50`）。

**判定。** 妥当（NOT）、Info。

### C.9 #9 SetImpulse の fprintf

**該当箇所。** `MKLNonUniformConvolver.cpp:1059-1092`。`li == 0` のたびに `[L0_WRITE]` を 3 行、形状を 1 行、最大 40 行のスロットピークを stderr へ出す。`#if` ガードはない。

**呼び出し。** `ConvolverProcessor.h:862,867` が L/R のエンジン構築時に呼ぶ。IR の読込・再構築のたびに、最大 2 × 44 行が出る。RT スレッドではない。

**影響。** ログの汚染のみ。製品コードへの診断混入。

**判定。** 妥当、Info（cosmetic）。

### C.10 #10 B-6 世代チェック

コメントは `LoaderThread.cpp:78-80` に残るが、ガードは `LoadPipeline.cpp:59`（bump）、`:223,404,827`（isCurrent）、`:253,297`（キャンセル）、`LoaderThread.cpp:642-654`（自分の世代）に実在する。commit 直前の確認があるため、古い結果が新しい要求を上書きする経路は静的には成立しない。

**判定。** 妥当（NOT）、Info。

### C.11 #11 既定テール整形

`ConvolverProcessor.h:86` の既定は `LayerTailContouring`。`manual/IR_ADVANCED_WINDOW_EN.md:74` は「Default: Air Absorption」と書く。マニュアル `:73` は L1/L2 を「減衰のみ」と説明するが、`MKLNonUniformConvolver.cpp:701-708` は L1 に 1.31〜1.56 倍、L2 に 1.03〜1.18 倍のゲインを乗算する。コードは意図的な音作りで、誤りは文書側にある。

**判定。** 妥当（NOT、文書問題）、Info。

### C.12 #12 リミッター／TruePeak 等

`SimplePeakLimiter.h:44-82` の状態は `envelope` のみ。更新は `desiredGain < envelope` のときだけなので、NaN の比較が false になり、NaN を保持しない。最終段の NaN スクラブ（`DSPCoreDouble.cpp:692-695`）は TruePeak（`:703`）とリミッター（`:715`）の前にある。

TruePeak の履歴と FixedDelay（`DSPCoreIO.cpp:299-331`）の再割当ては未確認である。前者は履歴バッファへ非有限値が入る可能性、後者は再 prepare 時のクリア有無が残る。

**判定。** 一部解消、未判定。

### C.13 N1 Bypass テール切断

外部監査の HTML には、N1 が LIKELY に分類された事実と「詳細は報告書本文に未収録」という注記しかない。機構・トリガー・影響の記述は本文にも注記にも存在しない。原文が確認できないため、推測で補完しない。

参考として、コード上はバイパス時にウェットをドライへクロスフェードし（`DSPCoreDouble.cpp:510-530`）、テールを保持したまま鳴らす経路は見当たらない。挙動としてはテールがフェードで切れる。これをバグとするかは仕様判断である。

**判定。** 未判定。必要な資料は第4パス報告書の該当節（§9 以降）と、N1 の調査ノート。

## D. 修正候補

いずれも提案であり、実装はしていない。

| ID | 対象 | 修正方針 | 副作用・互換性 | 必要テスト | 優先度 | 着手判断に必要な情報 |
|---|---|---|---|---|---|---|
| 1 | `floatToCanonicalBits`、`getOrCreate` | 符号ビットを保持し、ヒット時に全パラメータを比較 | キャッシュのヒット率がわずかに下がる | 符号違いの連続設定、プリセット跨ぎ | P1 | なし（現行 UI で再現） |
| 3 | `CustomInputOversampler`、CMake | 補正の既定 ON、または decimate に段数分の補償 | 既存プロジェクトのレベルが変わる。Phase 0 契約テストの期待値更新が必要 | 全段数・全 SR の DC／周波数応答、既存トラックの差分 | P2 | 既定 ON 化の可否（音作り判断） |
| 4 | `setBandFrequency/Gain/Q` | isFinite 拒否と範囲クランプ | 復元時に不正値が無視される（現状は格納） | 破損セッションの復元、境界値 | P2 | なし |
| 5 | `setNonlinearSaturation` | isFinite 拒否 | 不正値の無視 | 同上 | P2 | なし |
| 6 | `ConvolverProcessor::setState` の手動分岐 | `applyAutoDetectedIRLength` と同じ isFinite 拒否 | 不正値の無視 | 破損セッション、表示 | P3 | 表示仕様の確認 |
| 7 | `setFilterStructure` | 不正値を拒否して現状維持 | 復元時に既定へ落ちる | 破損セッション | P3 | なし |
| 9 | `SetImpulse` | プローブをコンパイル時ガード下へ | ログが出なくなる | なし | P3 | なし |
| 2 | `MixedPhase.cpp` のループ | `unwrapPhaseRadians` へ置換 | 目標位相が変わる（最大 0.13 rad） | Mixed の null 比較、聴感 | P3 | 実 IR での影響確認 |

## E. 未解決事項

| 区分 | 内容 |
|---|---|
| 再現できなかった主張 | N1（原文なし）、#2 の「通常の IR で 2π が残る」（典型的な低域通過では帯域内で再現せず） |
| 根拠資料が不足 | 第4パス報告書の本文（§9 以降）、notes/pass4-01〜03 |
| 静的解析だけでは判断できない | #1 の聴感上の目立ち方、#2 の実 IR での影響、#3 の聴感、#7 の応答差 |
| 実音声測定が必要 | OS 切替時のレベル変化、Mixed の位相誤差、破損セッションの復元 |
| ASan／TSan が必要 | なし（今回の対象は数値と状態の経路） |
| N1 | 原文と根拠コードの提供が必要 |
| Owner の判断 | #3 の既定 ON 化、#1 の修正着手、#4〜#7 の検証強化の範囲 |

## F. 検証の再現手順

| 目的 | 手順 |
|---|---|
| HEAD と作業ツリー | `git rev-parse HEAD`、`git status --porcelain` |
| 対象ファイルの未変更 | `git log 771804d..HEAD -- <file>` |
| ポリフェーズ | `build/Release/PolyphaseGainFidelityTests.exe` を実行 |
| ハッシュ衝突 | 符号マスクと `hashCombine` を同じ定数で再実装し、±6／±12／±48 dB を比較 |
| 係数応答 | `calcPeakingSVF` の式を再実装し、SVF の再帰で正弦波応答を測定 |
| 位相アンラップ | 最小位相スペクトルをケプストラム法で生成し、コードのループと `numpy.unwrap` を比較。重み付けとスロープリミッターも再現 |
| 入力経路 | `juce_CharacterFunctions.h:252-278` の nan／inf 解釈、`EQProcessor.Core.cpp:588-609` の直接呼び出し |

## G. 変更なしの確認

- 追跡ファイルの変更: 0 件（`git status --porcelain` で `??` のみ）
- ソース、テスト、CMake、CI の変更: なし
- コミット、PR: なし
- 本報告書のみを新規作成
