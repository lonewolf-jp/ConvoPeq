# PR #163（#1B）のマージと ConvoPeq.md の再生成

- 実施: 2026-11-07
- 基準: `main` = `8e54a53a`（マージ前）
- 承認: PR #163 のマージ、および必要に応じた `ConvoPeq.md` の再生成・SHA-256 検証

## 1. PR #163 のマージ

| 項目 | 値 |
|---|---|
| PR | #163 |
| マージ方式 | squash |
| マージ commit | `7faf9fa9` |
| 変更 | 10 ファイル（+271 / −14） |
| 内容 | #1B の実装（無効バンドのキー正規化）、単体・統合テスト、記録の更新 |
| PR の CI | 15 チェック中 14 pass + 1 skip（意図的失敗の dispatch のみ）、`CLEAN` |
| main の CI（`7faf9fa9`） | Windows MSVC・Sanitizer・CodeQL・lint・list-compliance・ISR Authority のすべてが success |
| main の CI（`568aa10e`） | Windows MSVC・Sanitizer・CodeQL・ISR Verification・ISR Authority のすべてが success |

`main` は `8e54a53a` から `7faf9fa9` へ進んだ。

## 2. `ConvoPeq.md` の再生成

#1B でソースが変わったため、`7faf9fa9` を基準に再生成した。

| 項目 | 値 |
|---|---|
| 生成元 `main` | `7faf9fa9` |
| 生成日時 | 2026-11-07 01:40:48 |
| 自動保存されたスナップショット | `doc/ConvoPeqMD/20261107014048/ConvoPeq.md` |
| 鮮度検査 | `NEWER_SRC_COUNT: 0`、`STATUS: FRESH` |
| サイズ | 6,534,569 バイト（LF） |
| SHA-256 | `e320b2b972ee3640ec683b291c10c91e4d868de0c14bf9779564852bcf37d4f9` |

生成直後は CRLF だったため、リポジトリの `*.md text eol=lf` に合わせて LF に正規化した。

### 分割版（1 ファイル 2 MiB 以下）

| ファイル | サイズ（バイト） | SHA-256 |
|---|---|---|
| `ConvoPeq.part01.md` | 2,097,124 | `d35ca56fd319f907a7f9fb73626c8986dbb29992bc3e8608997d407d04b23b39` |
| `ConvoPeq.part02.md` | 2,097,097 | `97a1685cc467044679b1f90901fed6f6255a19dd18ad555b00e6fb73ca0eff52` |
| `ConvoPeq.part03.md` | 2,097,143 | `7b07b3165872e4014488ea1d53dc1f58c8c168896961113a5cce82c93c5b2134` |
| `ConvoPeq.part04.md` | 243,205 | `46fdf47d0c939dbdab6b48eb764e2c78238854843cfe9aa2ecfd1592e0d43f84` |

4 分割。最大は 2,097,143 バイトで 2 MiB 以下。結合すると元ファイルと一致する。

## 3. 記録の更新

- `01-residual-register.md`: #1B を「実装済み（PR 未マージ）」から「マージ済み（`7faf9fa9`）」へ更新。
- `00-owner-decision.md`: #1B の対応状況をマージ済みへ更新。

## 4. 状態

| 項目 | 状態 |
|---|---|
| `main` | `7faf9fa9` |
| PR #163 | MERGED |
| #1B | 解消済み |
| `ConvoPeq.md` | 再生成して反映（SHA-256 `e320b2b9…d4f9`） |
| #1A | 保留を維持 |
| #12 | 修正しない |
| N1 | Issue #158 のまま（実装未承認） |
