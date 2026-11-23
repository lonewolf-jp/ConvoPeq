# main ブランチへのマージ確認と検証

- 実施: 2026-11-09 17:44〜17:55（`ctest` は 17:46:56〜17:52:29）
- フォルダ名 `20261109-2230` は作業ラベルであり、実時刻は上記のとおり
- 指示: Owner「main ブランチのソースコードへのマージはコーダーが実施」「対象は今回の検証に対応する既存の実装変更に限定」
- 対象 HEAD: `c94d376772ee22916f8350cdaa33f8d3787d16bd`
- 結論: 対象実装は **すでに main にマージ済み**。追加のマージコミットは作成せず、ソースコードの変更は 0 件

## §1 マージ状況

| 項目 | 値 |
|---|---|
| PR #205（`impl/part8-a-g`） | MERGED 2026-11-09T06:02:57Z、merge commit `e240d680b67304b54030d0212dcf208381ccb205` |
| PR #206（`impl/part8-e-f-h-i`） | MERGED 2026-11-09T06:03:29Z、merge commit `c94d376772ee22916f8350cdaa33f8d3787d16bd` |
| ローカル main | `c94d3767` |
| origin/main | `c94d3767`（ローカルと同一） |
| ブランチの取り込み | `git branch --merged main` に `impl/part8-a-g` / `impl/part8-e-f-h-i` を含む。未マージのローカルブランチは無し |
| 差分 | `git diff origin/impl/part8-a-g main` と `git diff origin/impl/part8-e-f-h-i main` はいずれも 0 行（main は両ブランチの内容をそのまま含む） |
| マージコミットの親 | `e240d680` は `2aecd73e` と `d1e35b94`、`c94d3767` は `e240d680` と `ba3604f0` を親とする |
| main の CI（`c94d3767`） | audioengine-lint / Sanitizer CI / ISR Authority Compliance v6.4 / list-compliance / Windows MSVC Build / CodeQL = すべて success |

- 未マージのリモートブランチは 2026-10-26〜11-03 の旧系列（`origin/docs/*`、`origin/fix/r5x-*` など）のみで、Part 8 と本検証に対応する実装変更は含まない
- 対象実装（A: ローダ診断ゲート、E: ポリフェーズ利得補正、F: Bypass の IR 保持、H: テール幾何、I: 先頭パーティション欠落）は `e240d680` と `c94d3767` で main に取り込み済み

## §2 追加マージを行わなかった理由

- 指示の対象は「今回の検証に対応する既存の実装変更」であり、その全てがすでに main に存在する
- 内容の変わらない空のマージコミットを積むと、承認済みの履歴だけが増え、検証対象と main の内容が変わらない
- 未マージの旧ブランチは本検証と無関係なため、スコープ外として取り込まない

## §3 マージ後の検証

ビルドツリー: `build`（Release、`CONVOPEQ_CORRECT_POLYPHASE_GAIN=ON`、`CONVOPEQ_ENABLE_LOAD_CHAIN_DIAGNOSTICS=OFF`）

| 検証 | 結果 |
|---|---|
| ビルドの鮮度 | 最新ソース `TailLayerIntegrityTests.cpp` 15:03:39 に対し、テスト実行ファイルは 15:08（`build/Release/PolyphaseGainMetrics.exe` など）。マージ後のソースでビルド済み |
| `ctest --test-dir build -C Release` | **56/56 passed**（100%）、333.11 秒、exit 0 |
| 主要テスト | `AudioEngineHarness` 301.90 秒で Passed、`PolyphaseGainFidelityTests` 1.51 秒、`PolyphaseGainMetrics` 0.80 秒、`TailLayerIntegrityTests` を含む 56 件すべて Passed |
| マージ前の CI | PR #205 / #206 のチェックはすべて pass（Windows MSVC Release、RelWithDebInfo ASan、Debug TSan、CodeQL、audit-report-verification、CodeRabbit ほか） |
| Part 8 実装時の受入（2026-11-09） | ctest 56/56、A の stderr 0 バイト、G の clang-tidy 監査 PASS |
| 数値検証（42 WAV） | 受入基準を満たす（`doc/work113/20261109-2000_リリース検証/report.md` §4） |

## §4 ソースコードの追加変更

- `git diff HEAD` は空、追跡ファイルの変更は 0 件
- 追加されたのは `doc/` の報告書のみ（未追跡）

## §5 README / RELEASE_NOTES / manual

- 本指示では**変更していない**（`git diff` 空、最終変更は `5a871d3b` のまま）
- 参考: これらの文書には Part 8 E の実装コミット `5a871d3b` で互換性告知がすでに入っている
  - `README.md`「Compatibility notice — oversampling gain (next release)」
  - `RELEASE_NOTES.md`「Unreleased」の英文・和文
  - `manual/AUDIO_SETTINGS_WINDOW_EN.md` / `_JP.md`
- 現行記載は公称値（0.75 / 0.5625 / 0.421875、−2.50 / −5.00 / −7.50 dB）を用いており、確定版（実測値 2.4988 / 4.9975 / 7.4963 dB と未測定範囲の明示を含む）とは表現が異なる。置き換えは Owner の作業範囲
- 確定済みの最終文面は `doc/work113/20261109-2000_リリース検証/report.md` §5.2（英文）と §5.3（和文）に保持

## §6 リリース・配布・公開告知

- リリース、配布物の作成・配布、公開告知、README / RELEASE_NOTES / manual への反映は**実施していない**
- 本マージ確認は、これらの実行許可を意味しない

## §7 追加の問題

- なし
