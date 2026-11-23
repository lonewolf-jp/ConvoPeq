# 残存バグ・残存課題の整理（2026-11-23 18:15 時点）

判定リビジョン: `main` = `8eaeb93d`。GitHub のオープン Issue は **1 件（#252）**、オープン PR は **0 件**。main の直近 40 回の workflow 実行は**すべて success** である（Windows MSVC Build と CodeQL で、doc のみのマージは解析をスキップしている）。

前回の全件報告（`doc/work113/20261115-2115_残課題一覧/report.md`、判定 `296ecfeb`）から 8 日経っている。本稿は**その後の解消を反映し、今回の作業で見つかったものを追加した**ものである。各項目に「本日確認」か「前回記載のまま」かを付けた。

## 1. 結論

| 区分 | 件数 | 内容 | 優先度 |
| --- | ---: | --- | --- |
| 未修正バグ（main） | 3 | NoiseShaperLearner の OOM 分岐が到達不能、CodeQL アラート 3 件（自作コード）、Debug 構成の既知失敗 2 件 | P1〜P2 |
| 未検証・未計測 | 6 | icx の `/fp:fast` による数値差、NUPC 幾何、オーバーサンプリング、qwen 指摘の網羅検証、新規 3 テストの ASan 未登録、S2 の長期再現性と ASIO 初期化 | P2 |
| 低優先・衛生 | 4 | 冗長な `vzeroupper`、atomic のドット呼び出し、`tailL1L2Multiplier` の丸め、EQ キャッシュの重複挿入（要再確認） | P3 |
| リポジトリ運用・文書 | 5 | 保護ツリーの未統合 WIP 194 件、未追跡の報告書 138 件、`ConvoPeq.md` の陳腐化、CodeQL の JUCE 32 件、`OPENCODE_CONFIG_CONTENT` 警告 | P1〜P2 |
| 環境ツール | 3 | context-mode の他エージェント未設定、Graphify Cloud の無料枠上限、headroom MCP の復帰手順 | P3 |

## 2. 未修正バグ（main に存在する欠陥）

### 2.1 NoiseShaperLearner の OOM 分岐が到達しない（本日確認）

`src/NoiseShaperLearner.cpp` のコンストラクタで `segmentBuffer = AudioSegmentBuffer::create();` の直後に `if (!segmentBuffer)` がある。しかし `create()` は失敗時に `std::bad_alloc` を投げ、nullptr を返さない。したがってこの分岐は到達せず、OOM では `errorMessage` と `Status::Error` に落ちずに例外が伝播する。同じ関数の直前では nothrow 版を nullptr で判定しており、**1 つの関数の中で失敗時の意味論が 2 通り**になっている。

修正案は 2 つある。`create()` を `try/catch` で包んで nullptr 契約に戻すか、呼出側を `try/catch` にして `errorMessage` へ落とすかである。どちらかに統一する。2026-11-15 の指摘から変更は無い。

### 2.2 CodeQL アラートのうち自作コードの 3 件（本日確認）

オープンなアラートは 35 件ある。32 件は `JUCE/` 配下（編集禁止のベンダーコード）で、残る 3 件が自作コードである。

| # | ルール | 深刻度 | 場所 |
| ---: | --- | --- | --- |
| 54 | `cpp/overflowing-snprintf` | high | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp:2443` |
| 55 | `cpp/overflowing-snprintf` | high | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp:2449` |
| 52 | `actions/missing-workflow-permissions` | medium | `.github/workflows/soak-ci.yml:38` |

54 と 55 は計測用ハーネスの文字列整形である。出力先の長さを `snprintf` の戻り値で見るか、`std::string` へ置き換えれば消える。52 は workflow に `permissions` を書くだけで閉じられる。

JUCE の 32 件は、CodeQL の設定で `JUCE/**` を解析対象から外せば一覧から消える。advanced setup へ移した際にこの除外を入れていないので、毎回同じ 32 件が出続ける。

### 2.3 Debug 構成の既知失敗 2 件（構成依存・前回実測）

`ctest -C Debug` は 2 件落ちる。`D8_2_B_2_Tests` は 0xC00000FD（スタック不足）で、`EpochDomain` が約 230 KB をスタックに置くことが原因である。`TailLayerIntegrityTests` は RES-F の CPU 予算（10 ms/s）超過で落ちる。どちらも Release では通る。本日、main の `CMakeLists.txt` に `D8_2_B_2_Tests` のスタック拡張（`/F`）が入っていないことを確認した。ロジックの誤りではなく構成の差である。

## 3. 未検証・未計測

| # | 項目 | 現状 | 必要な作業 |
| ---: | --- | --- | --- |
| 1 | icx の `/fp:fast` による数値差（BUG-029） | main は icx Release に `/fp:fast` を残している（本日、`CMakeLists.txt` で確認）。MSVC は `/fp:precise` | 同一入力で icx と MSVC の出力を比較し、許容差を測る |
| 2 | NUPC 幾何（BUG-010〜012, 014） | コード上のガードの存在確認まで | パーティション長・層境界・整除条件の全域検証 |
| 3 | オーバーサンプリング（BUG-025, 026） | 同上 | 比・段数・フィルタ長の組合せ検証 |
| 4 | qwen の非バグ 34 件 | 記載理由のスポット検証まで | 幾何・数値系の網羅検証 |
| 5 | 新規 3 テストの ASan 未登録 | `TailLayerMultiplierTests` / `AudioSegmentBufferTests` / `AllpassDesignerCmaesTests` が `CONVOPEQ_ASAN_TEST_TARGETS` に無い（本日確認） | 一覧へ追加し、ASan でリーク検査を効かせる |
| 6 | S2 の長期再現性・ASIO 初期化（#252） | 48 時間あけた独立再測定は未実施。ASIO 初期化の内訳は ETW が取れず未確定 | #252 の完了条件を参照 |

## 4. 低優先・衛生

| # | 項目 | 状態 |
| ---: | --- | --- |
| 1 | 冗長な `_mm256_zeroupper()`（`src/CustomInputOversampler.cpp`） | 残存（本日確認）。MSVC と icx は自動で `vzeroupper` を生成する |
| 2 | atomic のドット呼び出し（`src/audioengine/AudioEngine.h:894` の `liveCount.fetch_sub(`） | 残存（本日確認）。`convo::fetchSubAtomic` と等価だが規約違反 |
| 3 | `tailL1L2Multiplier` の丸め | 値は 2〜16 を受理し、実効値は最も近い 2 の冪へ丸められる。**UI からは参照が無くなっている**ため、表示と実音の不一致はもう起きない見込み。API の受理範囲と丸めだけが残る |
| 4 | TruePeakDetector の容量ガード | **解消**。`preparedMax` で `numSamples` を clamp し、`sanitizedCapacity >= numSamples` を確認している（本日確認） |
| 5 | EQ キャッシュの重複挿入 | 要再確認。`AudioEngine.Cache.cpp` は写しへ `emplace` して publish する形で、前回の「同一キーが 2 個」という説明を裏付ける経路は見当たらなかった |

## 5. リポジトリ運用・文書

### 5.1 保護ツリーの未統合 WIP（最重要）

`C:\VSC_Project\ConvoPeq` は `4d476d5c` のままで、**main より 213 コミット遅れ**ている。そこに未コミットの作業が **194 エントリ**残っている。

- `src`: 追跡ファイル 32 件の変更（+2,234 / −352）と未追跡 11 件。内訳は Convolver の読み込み経路（`LoadPipeline.cpp` +109、`LoaderThread.cpp` +182）、`InputInspection.{h,cpp}`、`src/learner/`（3 ファイル）、`IRTrailingSilence.h`、`ConvolverStopProbe.h`、テスト 6 本（`CacheManagerHeaderValidationTests` / `InputInspectionTests` / `LearnedStateMigrationTests` / `LearnerStateAuthorityTests` / `NoiseShaperStateRaceTests` / `RuntimeParameterGuardTests`）。
- `CMakeLists.txt` は +143 行で、上記テストのターゲットを 4 本追加している。main には無い。
- **衝突の下地**: ローカル変更 45 件のうち **27 ファイルが upstream でも変更済み**である（`DeviceSettings.cpp`、`NoiseShaperLearner.*`、`AudioEngine.*`、`ConvolverProcessor.*`、`EpochDomain.h` など）。統合には手作業の解決が要る。
- 帰属が未確認の作業であり、放置すると失われる危険がある。**最優先で「保全（ブランチ化）」か「破棄の判断」を決めるべき**である。

### 5.2 未追跡の報告書 138 件

`doc/` に未追跡の作業フォルダが 138 件ある（`20261115-*` 以降の報告書）。GitHub 上の記録と食い違わないよう、まとめて 1 本の PR にするか、`.gitignore` で対象を絞るかを決める必要がある。

### 5.3 `ConvoPeq.md` の陳腐化

`ConvoPeq.md` は 2026-11-08 のスナップショットで、以降のソース変更を反映していない。保護ツリーでも `M` になっている。次回のソース監査前に再生成が必要である（2 MB 分割版も含む）。

### 5.4 `OPENCODE_CONFIG_CONTENT` の警告

17:09 以降の OpenCode サーバログに `configuration normalization diagnostic ... source=OPENCODE_CONFIG_CONTENT kind=invalid action="rejected malformed JSON or JSONC document"` が 12 回出ている。設定ファイル 2 つ（`~/.config/opencode/opencode.json`、`C:\VSC_Project\ConvoPeq\opencode.json`）はいずれも JSON として妥当だった。アプリが環境変数で注入する設定の話で、原因は未解明である。

## 6. 環境ツール

1. **context-mode**: `context-mode doctor` は opencode の行だけ PASS し、claude-code / codex / gemini / qwen / vscode は未設定で FAIL する。本環境（OpenChamber）の不具合ではなく、他エージェントへ展開していないだけである。
2. **Graphify Cloud**: 無料プランは 1 リポジトリ 25,000 ノードまでで、本リポジトリは 71,126 ノードのため収まらない。PR チェックの `Graphify` / `Graphify Formal Verification` は Cloud 側の機能で動いている。
3. **headroom MCP**: プロセスを止めると自動では戻らない。OpenChamber を完全終了し、`serve --service` の opencode サーバを止めて起動し直すと戻る（手順は `AGENTS.md` に記載）。

## 7. 前回（2026-11-15）から解消したもの

| 項目 | 結果 |
| --- | --- |
| #215（MSVC `/arch:AVX2` で FMA decimate 経路が無効） | **解消**。#231 の逆アセンブル確認と性能測定を経て 2026-11-21 に両方クローズ |
| TruePeakDetector の容量ガード欠落 | **解消**（本日確認、§4 の 4） |
| ARCHITECTURE §9 の bypass crossfade 記述 | 解消済み（前回確認） |

## 8. 次にやるなら（推奨順）

1. **保護ツリーの WIP を保全する**（ブランチ化して push）。213 コミット遅れのまま未コミット 194 件を抱える状態が、いま最も損失リスクが高い。
2. **CodeQL の自作コード 3 件を閉じる**。`snprintf` 2 件と workflow の `permissions` 1 件で、いずれも小さな修正である。あわせて `JUCE/**` の除外を CodeQL 設定に入れ、32 件の常時ノイズを止める。
3. **NoiseShaperLearner の OOM 分岐をどちらかの意味論へ統一する**（§2.1）。
4. **新規 3 テストを ASan の対象へ追加する**（§3 の 5）。
5. 未検証の数値系（icx の `/fp:fast`、NUPC 幾何、オーバーサンプリング）は、音響出力の正しさに直結するため、まとめて 1 回の計測で片付けるのが効率的である。
6. `doc/` の未追跡 138 件と `ConvoPeq.md` の再生成は、文書の一貫性のため次の監査前にまとめる。

## 9. 参照

- 前回の全件報告: `doc/work113/20261115-2115_残課題一覧/report.md`
- S2 の残課題: Issue #252、`doc/work113/20261122-1944_236ゲート改定/report.md`
- ツール更新: `doc/work113/20261123-1657_ツール更新/report.md`
- icx ビルド: `doc/work113/20261123-1740_icxビルド/report.md`
- CodeQL の切替: `.github/workflows/codeql.yml`
