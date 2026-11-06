# ConvoPeq 構造・妥当性検証レポートのバグ主張を検証

- 実施: 2026-11-06
- 対象: `ConvoPeq 構造・妥当性検証レポート.html`（第4パス、HEAD `771804d`）のバグ候補 13 件と新規候補 N1
- 検証 HEAD: `41702b60`（本検証時点の `main`）
- 方法: 現行ソースの静的読解、既存テストの実行、ハッシュと位相アンラップの数値再現
- 結論: 5 件の BUG のうち 3 件はそのまま妥当、1 件は条件付きで妥当、1 件は影響の記述が過大。LIKELY 4 件は 3 件が妥当で、1 件は帰結の記述が誤り。NOT 3 件は妥当。N1 は判断できない。

## 1. 前提

| 項目 | 内容 |
|---|---|
| 監査時点の HEAD | `771804d91637dabec8b18aad6e2aef5a23c5c428` |
| 検証時点の HEAD | `41702b60` |
| 対象ファイルの変更 | 報告が名指しした 8 ファイルはいずれも `771804d..HEAD` で変更 0 件 |
| ビルド | 既存の `build/Release`（MSVC、Release）を使用。新規ビルドは行っていない |
| 実行したテスト | `build/Release/PolyphaseGainFidelityTests.exe`（1.58 秒、PASS=54 FAIL=0） |

対象ファイルに変更がないため、報告の所見は現行コードにもそのまま当てはまる。ただし報告の行番号は現行ファイルと 5〜10 行ずれる。内容の一致は確認したため、本記録では現行の行番号を示す。

## 2. 判定の要約

| # | 報告の区分 | 検証結果 | 補正 |
|---|---|---|---|
| 1 | BUG / major | 妥当 | 重大度も妥当。±48 dB の衝突を数値で確認 |
| 2 | BUG / major | 一部妥当 | 実装欠陥は確認。到達条件付きで、major は過大 |
| 3 | BUG / major | 妥当 | 既存テストの実行で 0.75 の段数を確認 |
| 4 | LIKELY / minor | 妥当（BUG 相当） | 上位層に事前検証はなく、到達性の推論リンクは解消 |
| 5 | LIKELY / minor | 妥当 | `jlimit` が NaN を素通しすることを実装で確認 |
| 6 | LIKELY / minor | 一部妥当 | ガード迂回は事実。float→int の未定義動作はこの経路では成立しない |
| 7 | BUG / minor | 妥当 | 不正値を Serial へ無言変換 |
| 8 | NOT | 妥当 | 回転は円環演算として正確。パディングなしの記述も一致 |
| 9 | BUG / cosmetic | 妥当 | 無条件の `fprintf` を確認 |
| 10 | NOT | 妥当 | 世代ガードは複数実在。コメントのみ陳腐化 |
| 11 | NOT（文書問題） | 妥当 | 既定は Layer Tail Contouring、マニュアルは Air Absorption と記載 |
| 12 | UNKNOWN | 一部解消 | リミッターは NaN を保持しない。TruePeak と FixedDelay は未監査 |
| N1 | LIKELY | 判断不能 | 報告書に機構の記載がない。バイパス時のテールフェードはコードで確認 |

## 3. 各項目の検証

### 3.1 #1 EQ 係数キャッシュのハッシュ符号衝突 — 妥当

報告の主張は「符号ビットを落とすため +x と −x が同一ハッシュになり、係数キャッシュが誤って再利用される」。

| 確認点 | 結果 |
|---|---|
| 符号ビットの除去 | `EQProcessor.ProcessingCache.cpp:10-15` で `bits & 0x7FFFFFFF` |
| ハッシュ対象 | 同 `:24-54`。band の frequency/gain/q、totalGainDb、nonlinearSaturation を含む |
| 再利用判定 | `AudioEngine.Cache.cpp:51-128`。`map.find(hash)` のみで、パラメータの全比較はない |
| キャッシュの寿命 | 同 `:111-116` で map をコピーして追記するため、エンジン稼働中は保持される |
| 係数の利用 | `EQProcessor.Processing.cpp:139-155` は係数 `m0/m1/m2` を使用。バンドゲインは係数にのみ反映される |

ハッシュ関数を同じ定数で再実装し、バンドゲインだけを変えて計算した結果は次のとおり。

```
gain   +6.0 dB -> hash 96a202e4c8d12609
gain   -6.0 dB -> hash 96a202e4c8d12609
gain  +48.0 dB -> hash a536013d977aa7d5
gain  -48.0 dB -> hash a536013d977aa7d5
```

報告の「+6→−6 で 12 dB、最悪 ±48 で 96 dB」は、帯域中心のゲイン差として妥当である。totalGainDb の衝突は、係数が totalGain に依存せず実値が別経路で適用されるため無害という報告の整理も正しい。

結論: **妥当**。修正は符号ビットを残すことと、ヒット時の全パラメータ比較の追加。

### 3.2 #2 MixedPhase の位相アンラップ — 一部妥当

コードは `ConvolverProcessor.MixedPhase.cpp:281-292` にあり、次の処理を行う。

```
delta = raw - phi[k-1]
if (delta > π)  phi[k] = phi[k-1] + delta - 2π
else            phi[k] = phi[k-1] + delta
```

`delta < −π` の分岐がないため、位相が増加して +π を跨ぐ場合と、1 ステップで 2π を超えて回る場合に追従できない。一方、同リポジトリの `ConvolverProcessor.Internal.h:33-46` には両方向補正と累積を備えた `unwrapPhaseRadians` があり、フォールバック経路（`MixedPhase.cpp:837`）はそちらを使っている。実装の非対称は事実である。

報告の「通常の IR では 2π の不連続が残る」は、数値実験では条件付きだった。

| 入力（最小位相に変換） | 標準 unwrap との差 | 1300 Hz 未満の不一致 |
|---|---|---|
| 2 次低域通過 | 0 | 0 |
| 4 次低域通過（30〜500 Hz） | 0 | 0 |
| 4 次高域通過（30〜500 Hz） | 0 | 0 |
| 8 次低域通過 30 Hz | 2π（最大 12.57 rad） | 108 bin |
| 8 次低域通過 100 Hz | 2π | 102 bin |
| 8 次高域通過 500 Hz | 2π（最大 25.13 rad） | 108 bin |
| ランダムな室特性 | 0 | 0 |

- 4 次以下と室特性では、最小位相スペクトルの位相は単調減少し、片方向の補正で足りる。
- 8 次の急峻なフィルタでは、位相が 3π を超えて回り、1 ステップの差分が 4π になる箇所で取りこぼす。不一致は 1300 Hz 未満にも現れる。
- MixedPhase の遷移上限は `ConvolverProcessor.h:232` の 1300 Hz で、これを超える帯域では最小位相の重みが 0 になる。既定の遷移（200〜1000 Hz）では、今回の実験範囲で不一致は使われない帯域にあった。

評価: **一部妥当**。実装欠陥は確認できるが、「通常の IR で残る」「major」は過大。到達条件は急峻な振幅特性と遷移上限付近の設定で、可聴影響は未定量である。修正は `Internal.h` の `unwrapPhaseRadians` への置換で足りる。

### 3.3 #3 ポリフェーズ利得誤差 — 妥当

| 確認点 | 結果 |
|---|---|
| 補正マクロ | `CustomInputOversampler.cpp:549-552`。畳み込み側は常に ×2、中心側の ×2 は `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN` 内のみ |
| 既定 | `CMakeLists.txt:134` の option は OFF |
| 本番経路の補償 | `AudioEngine.Processing.DSPCoreDouble.cpp` の processUp（:354）から processDown（:534）の間に、段数分の補償ゲインはない。あるのはユーザー設定の `outputMakeupGain` とコンボルバ入力トリムのみ |
| 局所 2 倍 OS | 同 `:496-507`。SoftClip 有効かつ OS=1 のとき、2 倍の up/down を通る |

既存テストを実行した結果（PASS=54、FAIL=0）では、DC 利得は次のとおりだった。

```
r=2: dcBase=0.750000000000  dcCand=1.000000000000
r=4: dcBase=0.562500000000  dcCand=1.000000000000
r=8: dcBase=0.421875000000  dcCand=1.000000000000
[INFO] production_flag = undefined/off のまま（HOLD 継続）
```

20·log10(0.75) = −2.498 dB。報告の「−2.50／−5.00／−7.50 dB」は正しい。テスト自身が production フラグ未定義を確認しており、出荷構成で補正が効いていないことも裏付けられる。

まとめ: **妥当**。重大度 major も妥当。ただし開発側が HOLD 中の既知欠陥であり、新規の不具合ではない。

### 3.4 #4 EQ バンドセッターの無検証 — 妥当（BUG 相当）

| 確認点 | 結果 |
|---|---|
| セッター本体 | `EQProcessor.Parameters.cpp:19-74`。freq/gain/q はいずれもバンド番号の範囲のみ確認し、値をそのまま格納 |
| 比較対象 | 同 `:103-116` の `setTotalGain` は isFinite 拒否と ±48 dB クランプを行う |
| 復元経路 | `EQProcessor.Core.cpp:588-609` の `setState` が ValueTree の値を直接セッターへ渡す |
| 上位層の事前検証 | `AudioEngine.Parameters.cpp:139-141` は EQ 子ノードの型のみ確認。数値は検証しない |

報告は「上位層の事前検証が未読」として LIKELY に留めていたが、そのリンクは解消した。手製・破損プリセットから到達できる。

影響は報告の整理どおりで、NaN のバンドは SVF 係数側の isfinite 防御で無言のバイパスになり、NaN は状態とハッシュに残る。音声が NaN になるわけではない。

判断: **妥当**。区分は LIKELY から BUG に上げてよいが、影響は minor のまま。

### 3.5 #5 setNonlinearSaturation の NaN 公開 — 妥当

- `EQProcessor.Parameters.cpp:227-247` は `juce::jlimit(0.0f, 1.0f, value)` のみ。
- `juce::jlimit` は `juce_MathsFunctions.h:526-528` で `value < lower ? lower : (upper < value ? upper : value)` と実装され、NaN はどちらの比較も false のためそのまま返る。
- 変化検出 `std::abs(clamped - previous) < 1e-9` も NaN では false になり、早期 return しない。
- 公開先は atomic と EQState の両方（`:234-241`）。

RT 側は `EQProcessor.Processing.cpp:291` の `saturation > 0.0` が NaN で false になり、飽和が無効化されるだけである。状態には残り、`getState` が生値を書き戻すためプリセットへ往復する。

結論: **妥当**（minor）。

### 3.6 #6 Convolver setState の D8 バイパス — 一部妥当

ガードの迂回は事実である。`ConvolverProcessor.StateAndUI.cpp:336-343` の手動オーバーライド分岐は `jlimit` のみで `autoDetectedIRLengthSec` に格納し、isfinite を確認する `applyAutoDetectedIRLength`（`ConvolverProcessor.Runtime.cpp:921-947`）を通らない。`jlimit` は NaN を通す。

ただし報告が挙げた「float→int 変換の未定義動作、過大確保」は、この経路では成立しない。

| 確認点 | 結果 |
|---|---|
| `computeTargetIRLength` の入力 | `StateAndUI.cpp:993-1019` は `targetIRLengthSec` を読む。`autoDetectedIRLengthSec` は読まない |
| 手動分岐の target | 同じ分岐内で `setTargetIRLength(v.getProperty("irLength"))` を呼ぶ（`:346-349`）。こちらは isFinite ガード付き |
| `autoDetectedIRLengthSec` の用途 | `ConvolverProcessor.h` の getter と、`ConvolverControlPanel.cpp:1389` の表示ラベルのみ |
| 状態への影響 | `getState` が書き戻すため NaN はセッションに残る |

したがって影響は「NaN の状態汚染と表示の `nan`」に限られ、メモリ安全性の問題ではない。報告も最終的に minor としているが、未定義動作という理由付けは取り下げるべきである。

評価: **一部妥当**。

### 3.7 #7 setFilterStructure の無言 Serial 変換 — 妥当

`EQProcessor.Parameters.cpp` の `setFilterStructure` は、`Serial` でも `Parallel` でもない値を `Serial` へ強制変換して処理を続ける。兄弟の `setBandType`・`setBandChannelMode` は不正値を拒否して現状維持するため、契約が不統一である。復元経路は `EQProcessor.Core.cpp:594-595` から到達する。

まとめ: **妥当**（minor）。

### 3.8 #8 フォールバック FFT 位相回転のパディングなし — 妥当（NOT）

- `ConvolverProcessor.MixedPhase.cpp:749` で `fftSize = nextPowerOfTwo(numSamples)`。×4 パディングはない。
- 同 `:837` で `unwrapPhaseRadians` を適用しており、位相回転は bin ごとのフェーザ乗算である。
- 周波数領域のフェーザ乗算は円環シフトとして正確に実装される。線形遅延としてのテール回り込みは、2 の冪ちょうどの長さでのみ生じる有界な近似誤差で、フォールバック経路に限られる。

判断: **妥当**。報告の「本質的に円環演算として正確」という表現は、線形遅延の近似という但し書き付きで正しい。

### 3.9 #9 SetImpulse の fprintf 残留 — 妥当

`MKLNonUniformConvolver.cpp:1059-1092` に `[L0_WRITE]` の `fprintf` がある。`#if` ガードはなく、`li == 0` の SetImpulse ごとに最大 40 行超を stderr へ出す。RT 経路ではなく、ローダー／メッセージスレッドから呼ばれる。

結論: **妥当**（cosmetic）。

### 3.10 #10 B-6 世代チェック — 妥当（NOT）

コメントは `ConvolverProcessor.LoaderThread.cpp:78-80` に残るが、世代ガードは複数実在する。

| 箇所 | 内容 |
|---|---|
| `LoadPipeline.cpp:59` | 要求開始時に `bumpGeneration()` |
| `LoadPipeline.cpp:223, 404, 827` | `isCurrentGeneration` による確認 |
| `LoadPipeline.cpp:253, 297` | キャンセル判定 |
| `LoaderThread.cpp:642-654` | 自分の世代との一致確認 |

commit 直前の確認（`:827`）があるため、古い結果が新しい要求を上書きする経路は静的には成立しない。コメントが陳腐化しているだけである。

評価: **妥当**。

### 3.11 #11 既定テール整形 — 妥当（コードのバグではない）

- `ConvolverProcessor.h:86` と `:1082` の既定は `TailMode::LayerTailContouring`。
- `manual/IR_ADVANCED_WINDOW_EN.md:74` は「Default: Air Absorption」と記載する。
- 同マニュアル `:73` は Layer Tail Contouring を「L1/L2 へ減衰のみ」と説明するが、コードは L1 のゲインを増幅する（strength の下限 1.25）。

まとめ: **妥当**。コードは意図的設計で、誤りはマニュアル側にある。

### 3.12 #12 残存未監査の内部状態 — 一部解消

| 対象 | 結果 |
|---|---|
| SimplePeakLimiter | `src/audioengine/SimplePeakLimiter.h`。状態は `envelope` のみで、更新は `desiredGain < envelope` のときだけ。NaN 比較は false のため、NaN を保持しない |
| 最終段の NaN スクラブ | リミッターの前段にあり、非有限値を 0 に置換する（報告のチェーン記述どおり） |
| TruePeakDetector | reset 系はあるが、NaN 入力時の内部履歴の挙動は未確認 |
| FixedDelay | 未確認 |

判断: **一部解消**。リミッターについては報告の懸念は当たらない。TruePeak と FixedDelay は未監査のまま。

### 3.13 N1 Bypass テール切断 — 判断不能

報告書本文が途切れており、機構の記載がない。コードでは `AudioEngine.Processing.DSPCoreDouble.cpp:510-530` がバイパス時にウェットをドライへクロスフェードし、テールを保持したまま鳴らし続ける経路は見当たらない。挙動としてはテールがフェードで切れる。これをバグとするかは仕様判断で、静的調査だけでは決められない。

## 4. 報告書の記述についての補足

- 行番号は現行ファイルと一致しない（例: `floatToCanonicalBits` は報告 L16-26、現行 L10-15）。対象ファイルに差分がないため、報告側の参照方法のずれである。
- #6 の「float→int の未定義動作」は、`autoDetectedIRLengthSec` ではなく `targetIRLengthSec` の話であり、手動分岐は後者を設定しない。理由付けの取り違えである。
- #2 の「複数回の巻き戻しを蓄積できない」は正確ではない。下方向の巻き戻しは累積する。欠けているのは上方向（位相増加）と 1 ステップ 2π 超の補正である。
- #4 の「到達性の推論リンク」は、上位層を読むことで解消した。

## 5. 再現手順

| 目的 | 方法 |
|---|---|
| 対象ファイルの未変更確認 | `git log --oneline 771804d..HEAD -- <file>` が 0 件 |
| ポリフェーズ利得 | `build/Release/PolyphaseGainFidelityTests.exe` を実行し、`dcBase` と `production_flag` を確認 |
| ハッシュ衝突 | 符号マスクと `hashCombine` を同じ定数で再実装し、±6／±48 dB のハッシュを比較 |
| 位相アンラップ | 最小位相スペクトルをケプストラム法で生成し、コードと同じ片方向ループと `numpy.unwrap` を比較 |

## 6. 未確認・限界

- TSan や ASan を用いた実行時検証は行っていない。
- ハッシュ衝突の可聴影響は、係数の再利用を静的に確認したもので、実際の音声出力は測定していない。
- MixedPhase の不一致が最終的な音響へ与える影響は、スロープリミッターと重み付けを含めた数値評価が必要である。
- TruePeakDetector と FixedDelay の NaN 挙動は未確認。
- 報告書 N1 の機構は報告書に記載がなく、判断できない。
- 本検証は `build/Release` の既存バイナリを使用した。ソース変更は行っていない。

## 7. 最小修正の候補（実装はしない）

| # | 修正 |
|---|---|
| 1 | `floatToCanonicalBits` から符号マスクを外す。ヒット時に全パラメータを比較する |
| 2 | 当該ループを `unwrapPhaseRadians` に置換する |
| 4 | `setBandFrequency`／`setBandGain`／`setBandQ` に isFinite 拒否と範囲クランプを追加する |
| 5 | `setNonlinearSaturation` に isFinite 拒否を追加する |
| 6 | 手動分岐でも `applyAutoDetectedIRLength` と同じ isFinite 拒否を行う |
| 7 | 不正な `FilterStructure` を拒否して現状維持する |
| 9 | `fprintf` をコンパイル時ガード下へ移す |
| 3 | 補正の既定 ON 化は Phase 0 契約テストの更新を伴うため、別判断 |
