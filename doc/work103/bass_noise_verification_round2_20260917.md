# WORK103-R2 — 低音ジジジ要因の再検証報告（round-1 結論の妥当性検証）

- **作成日**: 2026-09-17
- **種別**: Verification（round-1 報告 `bass_noise_investigation_20260917.md` の各要因を源码レベルで再検証）
- **結論の変化**: 主因Aは**条件付きで維持**（公称ピーク0.71で単独では届かず、3つの未モデル化ゲインが余裕1.4dBを消す構造と修正）。欠陥Bは**2件の確定欠陥に分割**（うち1件は現行IRには無影響の潜伏欠陥と訂正）。副因Cは維持。誤記2件を訂正（§7）。

```text
判定: A=条件付き維持 / B=B1確定(影響限定)+B2確定(潜伏) / C=維持 / 誤記2件訂正
```

---

## 1. 再検証サマリー

| # | round-1 主張 | 再検証結果 | 扱い |
|---|---|---|---|
| A | 出力安全鎖に当たる（試算1.05） | EQ totalGain −2dB（`smoothTotalGain`）の適用を見落とし。正しくは**公称0.71**。ただし余裕は1.4dBしかなく、(a)サンプル間ピーク(b)allpass過渡再生育(c)空気帯域の高温が余裕を消す | 条件付き維持（§2） |
| B | Tukey窓でconvBoost過小→AutoGain不足 | 窓クラッシュ自体は数値確定（−53dB〜完全消去）。しかし減衰型IRでは`max(0,負)=0`でplan不変＝**現行IRのbuzzには寄与しない**。別途、NUC経路で値自体が未伝搬の確定欠陥を発見（潜伏・ブースト型IRで発火） | B1（窓・影響限定）＋B2（未伝搬・潜伏）に分割（§3） |
| C | 超音波→IMD | OutputFilterは処理レート384kで正しくprepare（2倍ズレ仮説は死亡）。空気帯域+4dBの実測でHF側も高温と追加確認 | 維持（§4） |
| — | cmaesRestarts等はMixed-phaseの安全装置 | 誤記。`coeffSafetyMargin/cmaesRestarts/enableStabilityCheck`は**NoiseShaperLearner系**の属性（rgで7ファイル特定、convolver系0件） | 訂正（§7） |
| — | Mixed-phaseがピーク再生育の主因候補 | 20×2次allpassで61sampleのGD段差を実現＝1段約3sampleで温和。rho上限0.995（`AllpassDesigner.cpp:283,553,641`）。当該IRでは無害と除名 | 訂正（§7） |

---

## 2. 主因Aの再計算（厳密化）

### 2.1 確定したゲイン流

- Planner入力：`spec.analysis.{eqMaxGainDb,eqMaxQ,irFreqPeakGainDb}`（`RuntimeBuilder.cpp:304-308`）。EQ側は`computeEstimatedMaxGainComplex`を**処理レート**で測定し`max(measured,upperBound)`でcollapse（`AudioEngine.RebuildDispatch.cpp:697-732`）。Conv側は`getIrFreqPeakGainDb()`（同:743）。
- AutoGain有効時はplan値で上書き（`RuntimeBuilder.cpp:319-324`）＋`[AUTO_GAIN_PLAN]`ログ出力あり。XMLの−8/+3dBは不使用。
- EQ `totalGain=-2dB`は`smoothTotalGain`ランプとして帯域処理**後段**に適用（`EQProcessor.Processing.cpp:961-975`）。
- Limiterは無条件常時動作（`processOutputDouble:703-710`）、release 100msはホストレート基準（`DSPCoreLifecycle.cpp:228`、`releaseCoeff≈0.999948`＠192k＝50Hzの周期20msを跨いで抑圧が残る）。

### 2.2 RBJ厳密値（fs=384k）

- 低域スタック（25 shelving +3／40 +3／63 +3、Q0.707）：50Hzで **+5.19dB（×1.818）**、40Hz +4.92dB。
- 空気帯域スタック（12.5k +0.5〜19.5k +1×4）：14〜20kHzで **+3.8〜+4.0dB（×1.55〜1.59）**。
- plan値（eqBoost≈+5.2と推定）：input≈−5.2dB（×0.549）、makeup≈+5.2dB（×1.82）。

### 2.3 公称ピーク（0dBFS正弦波・impulse.wav conv帯域gain 0.49）

- 低域50Hz：`1.0×0.549×0.49×1.818×0.794×1.82 ≈ 0.71`。Limiter 0.841まで**余裕わずか1.4dB**。
- 高域16kHz：`1.0×0.549×0.51×1.589×0.794×1.82 ≈ 0.64`。こちらも高温。
- 余裕を消す3要素（いずれも源码・文献で裏付け）：(a)**サンプル間ピーク**（再構成で数dB超過、tonalux／iZotope準拠の定説。TruePeakDetectorが計測のみで保護に使われていない点も確認）、(b)**キック等の広帯域過渡＋位相回転によるピーク再生育**（allpass自体は温和だがconvolver＋EQの線形位相回転の総和）、(c)**リリース100msの抑圧残留による波形整形の連鎖**。
- リミッター無 lookahead＋無OS＋即時アタックの低域歪特性は文献で再確認（Melda「0msアタックはsharp edge」、SageAudio「lookahead 0.1ms＋4xOSが定石」、tonalux「reactive limiterは数sampleクリップ後にしか抑圧できない」）。**本機のリミッターは3要素をすべて欠く**。
- 追加の歪源としてEQ Parallel saturation：`accum += band処理(含tanh)×20band − src`（`EQProcessor.Processing.cpp:751-791`）のため、**全band 0dBでも0dBFS信号には20band分のtanh色付け**が残り、ブーストbandでは増大。Conv→EQ時の低音に上乗せされる。バイパス時は完全早期return（同:525）で無害。

→ Aは「単独で常時クリップ」ではなく「**余裕1.4dBの剃刀設計＋3要素で余裕消滅→低域で最も可聴な歪**」と修正して維持。

## 3. 欠陥Bの分割と確定

### 3.1 B1：Tukey窓クラッシュ（数値確定・影響は限定）

`IRAnalyzer.cpp:71-99`（Tukey α=0.5、`kMaxAnalysisWindow=65536`）の窓ゲイン実測：

| 経路 | ピーク位置 | 窓ゲイン |
|---|---|---|
| 384k AsIs（peak@488） | 先頭テーパー内 | **0.0022（−53dB）** |
| Minimum／Mixed（peak@0） | 端点 | **0.0（完全消去）** |
| 48k RCU（peak@61） | 先頭テーパー内 | **0.0022（−53dB）** |

`windowMean≈0.75`の補正（同:152）は+2.5dBにしかならず形状復元不能。デルタ状IR（99％がピーク1点）は周波数特性が20〜50dB過小評価される。**ただし減衰型IRでは`max(0,負)=0`でplan不変**のため、現行IRのbuzzへの寄与はなしと訂正。`applyClampProtection`の周波数クランプ（`IRConverter.cpp:112-119`、上限+3dB）も同窓で測定されており、先頭集中IRには発火しない（過大スケールの防御欠如＝残留リスク）。

### 3.2 B2：NUC経路で`irFreqPeakGainDb`が未伝搬（新規確定・潜伏欠陥）

- `executePendingCommit` Phase 1（`LoadPipeline.cpp:799-808`）：`updateIRState(loadedIR, sampleRate)`を**デフォルト引数（0.0dB）のまま**呼ぶ。`PendingCommit`構造体（`ConvolverProcessor.h:955-962`）に周波数ゲインの field 自体が存在せず、`applyNewState`のシグネチャにもない。
- `LoaderThread::doTransformStep`（`LoaderThread.cpp:711-719`）は`scaleFactor`のみ算出し周波数ピークを計算しない。
- よって**現行のNUCロード経路では`IRState::irFreqPeakGainDb`は常に0.0**（新規）またはstale（rebuild時は更新なし）。`RebuildDispatch.cpp:743`→Planner `convBoost=0`固定。
- 4種の独立検索で一致確認：rg／AiDex exact／**serena MCP `search_for_pattern`**（`LoadPipeline.cpp:564`のみが値付きwriter）／**tgrep search**（writerは`IRConverter.cpp`＋`LoadPipeline.cpp:564`＋定義のみ）。
- 影響：現行の減衰型IRでは無害だが、**REW／rePhase系のブースト型IR（null補償+9dB等）を読んだ瞬間にheadroom不足→確定的クリップ**。room-correction用途の製品として潜伏地雷。修正案：`doTransformStep`で窓なし（矩形）ピークホールド測定を追加し`PendingCommit`経由で伝搬する。

## 4. 副因Cの維持と補強

- 死亡仮説：OutputFilterのprepareレート誤り→`outputFilter.prepare(processingRate)`（`DSPCoreLifecycle.cpp:216`）で384k正規。HC 22k／LP 24kは設計通り。
- 維持根拠：IRの20kHz帯リンギング±0.1（scaled −31dBFS級）＋空気帯域EQ +4dBの二重奏で、192k出力の20〜96kHz残留が無視できない。SoftClip（384k域動作）の高調波も加わる。Voicemeeter内部SRC→DAC→アンプでのIMD／折り返しは文献 consistent（Gearspace／RME／Xiph）。低音振幅に追従する点も畳み込みの線形性から説明可能。
- 除名済み：Mixed-phase高Qリンギング（§1表の通り温和と確認）。

## 5. 除外の再確認（今回追加分）

- パーティション／ブロック不整合：L0 part=2048＝処理ブロックで整合。firecrawl開発者検索（overlap-add／saveの正規条件）とも矛盾なし。
- DCブロッカ：3箇所ともレート別init（`DSPCoreLifecycle.cpp:204`）。IR用1Hz 2段（`UltraHighRateDCBlocker`）のデルタ応答は3e-05級で非可聴。
- Tukey非対称窓＋80ms fade、r8brain Linear 8倍のプリリンギング（−100dB級サイドローブ）はいずれも線形・微小で非可聴と確定。
- ドライ遅延クロスフェード：定常時は整数遅延読み（`Runtime.cpp:611-620`）、retarget時のみ補間。持続buzzの原因にならない。

## 6. 静的解析・ツール実行記録

- **cppcheck**（`--language=c++ --std=c++20 --enable=warning,performance`）：`IRAnalyzer.cpp`＋`UpperBoundEstimator.cpp`＝機能欠陥0件。`EQProcessor.h`等の未初期化メンバ警告のみ（スタイル級）。
- **clang-tidy**：rootの`compile_commands.json`は配列でないためDB認識不可（先頭が`{`の4.3MB単一object）。有効な`compile_commands_clang.json`（3TU）を別dirに`compile_commands.json`として配置し、`RuntimePublicationOrchestrator.cpp`（＝Planner入力受渡しTU）に`clang-analyzer-core.NullDereference,bugprone-use-after-move`で実行＝**警告0件**。
- **tgrep**：`C:\Windows\System32\`に実在（小文字パスでは不可）。`count-files`＝src内341件。`search irFreqPeakGainDb`でB2を裏付け。
- **serena**：使用法をGitHub READMEで学習。`serena.exe`（uv導入済み）でHTTP-MCPサーバ（:8123）を起動しJSON-RPCで`initialize→initial_instructions→search_for_pattern→read_memory`を実行。symbol系はプロジェクトのpython LS（pyright）初期化失敗によりマネージャ全体が停止するため使用不可（C++ clangd単独でも不可）。非LSP系でB2を第三者確認＋`global/mandatory-tools-workflow`メモリで3層パイプライン遵守を再確認。使用後に自起動分のみ停止。
- **semble／ccc／graphify／AiDex**：round-1に続き使用。cccは`--project`非対応のため素の`search`で実行。graphifyは61029ノードからBFS。
- **rg／ast-grep 0.44／fdfind 10.3／ag 2.2／fzf／sed／awk**（WSL）：`cblas_dscal`適用確認、`PendingCommit`構造、`coeffSafetyMargin`消費元特定等。fdfind／fzfはWindows側PATHになくWSL経由で実行。
- **Obscura**（native exe確認、`fetch --dump text`でiZotope頁取得に参加）：docker版不使用。
- **DDGS／trafilatura**（pip導入済み確認）：DDGSでlookahead文献5件→tonalux頁をtrafilaturaで抽出（6819字）し主因Aの文献根拠を補強。**Crawl4AI**はimport確認のみ（単頁抽出には過剰のため未起動）。
- **context7**：JUCEをresolve（`/juce-framework/juce`）し`dsp::Oversampling` latencyをquery。返却はoboe系の無関係文書のみで本件への寄与なし。
- **firecrawl**：web検索（Voicemeeter SRC実態）＋developer検索（partitioned convolution正規条件）を実行。
- **brave**：round-1のlimiter／IMD文献に使用。
- **Dr.Memory**：ASIO実機再生を要する動的検査のため未実行（`build/`にDebug／Releaseツリーあり。運転時の外形観測つきで実施する後続課題）。

## 7. 訂正記録（round-1からの変更）

1. 公称ピーク1.05→**0.71**（EQ totalGain −2dBの適用漏れ）。主因Aの機序は「常時接触」から「**余裕1.4dB＋3要素で消滅**」へ修正。
2. `cmaesRestarts／coeffSafetyMargin／enableStabilityCheck`はMixed-phaseではなく**NoiseShaperLearner系**。buzzとの因果関係なし。
3. 欠陥Bのplan影響は現行IRでは**なし**（減衰型のため）。真の危険はB2の潜伏欠陥（ブースト型IRで発火）。
4. Mixed-phase allpassは当該IRでは温和（20×2次で61sample段差）と確認し除名。

## 8. 恒久対策（優先順）

1. リミッターにlookahead（1〜4sample）＋リリースの低域適応化、または天井−1.5dB化（§2.3）。
2. B2修正：`doTransformStep`で矩形窓ピークホールド測定→`PendingCommit`拡張→`updateIRState`伝搬（§3.2）。
3. B1修正：窓掛け前ピークサーチ／ピーク中央シフト／矩形窓併用（§3.1）。
4. SoftClip後段の超音波除去LPF（Sharp 20kHz）または`convHCMode=Sharp`既定化（§4）。
5. Dr.Memory＋実機ログ（`[AUTO_GAIN_PLAN]`のeqMaxGainDb／irFreqPeakGainDb実値）による最終確定。
