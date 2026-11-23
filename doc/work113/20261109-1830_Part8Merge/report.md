# Part 8 PR の merge と merge 後検証

- 実施: 2026-11-09 18:30（merge は 06:03 UTC）
- 対象: PR [#205](https://github.com/lonewolf-jp/ConvoPeq/pull/205)（A+G）、[#206](https://github.com/lonewolf-jp/ConvoPeq/pull/206)（E/F/H/I）
- 結果: 2 PR を指定順（#205 → #206）で merge。ローカル統合検証と main の CI を確認
- 未実施: release・配布・互換性告知の公開（Owner 判断により別途）

## §1 merge 結果

| 順 | PR | merge commit | 備考 |
|---|---|---|---|
| 1 | #205 | `e240d680b67304b54030d0212dcf208381ccb205` | レビュー指摘（Trivial）対応込み |
| 2 | #206 | `c94d376772ee22916f8350cdaa33f8d3787d16bd` | #205 merge 後に実施 |

- `main` の HEAD は `c94d3767`。両 PR のコミットは履歴に残り、fast-forward の取り込みも確認
- ブランチは削除していない（`impl/part8-a-g`、`impl/part8-e-f-h-i`）

## §2 レビュー指摘への対応

CodeRabbit の指摘は 2 件で、いずれも **Trivial**（重大な指摘なし）。

| PR | 指摘 | 対応 |
|---|---|---|
| #205 | 診断 OFF でも argmax 走査 2 回とテールエネルギー走査が走る | `d1e35b94` で走査・エネルギー計算を同じマクロの内側へ移動。既定 0 バイト、ON は 192 行/29,940 B を再確認 |
| #206 | I-2 が幾何のみで、先頭 L1 パーティションの出力を見ていない | `ba3604f0` で先頭 L1 パーティションのゲイン（block 512 で 3.15 dB）を block 256 の参照（3.15 dB）と比較する判定を追加 |

## §3 merge 後のローカル検証（main `c94d3767`）

| 項目 | 結果 |
|---|---|
| 全ターゲットビルド（Release） | 成功 |
| `ctest` | **56/56 PASS** |
| A の既定動作（`--n1-analysis`） | stderr **0 バイト**、`ALL PASS`、exit 0 |
| G の監査ゲート（exhaustive/enforce） | `[PASS] clang-tidy audit gate verified` |
| 統合状態 | A の `CONVOPEQ_ENABLE_LOAD_CHAIN_DIAGNOSTICS`（既定 OFF）と E の `CONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` が共存 |

## §4 merge 後の CI（main）

| ワークフロー | 結果 |
|---|---|
| Windows MSVC Build | success |
| ISR Authority Compliance v6.4 | success（初回は Ninja 配布物取得の HTTP 504 で失敗。インフラ要因として失敗ジョブを再実行し成功） |
| CodeQL | success |
| audioengine-lint | success |
| list-compliance | success |
| Sanitizer CI | success（ASan: RelWithDebInfo MSVC、TSan: Debug Clang best-effort とも success） |

- 504 は `ashutoshvarma/setup-ninja@v1.1` が GitHub releases から v1.10.0 を取得する段階で発生。コード変更に起因しない
- 再実行で解消したため、ワークフロー変更は行っていない

## §5 未実施・別途判断

- release 作成、配布、製品公開、互換性告知の最終公開
- block 32768 超の性能保証
- 3.0 秒 IR の block 64/128 における聴感評価
- I の根本原因の追加調査（倍率上限 4 は回避策）

## §6 参考: 実行コマンド

```
git checkout main && git pull --ff-only
cmake -S . -B build -DCONVOPEQ_ENABLE_LOAD_CHAIN_DIAGNOSTICS=OFF -DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure      # 56/56
build/Release/AudioEngineHarness.exe --n1-analysis <dir> post
pwsh .github/scripts/isr-verify-clang-tidy-audit.ps1 -Tier exhaustive -RequireClangTidy
gh run list --branch main
```

