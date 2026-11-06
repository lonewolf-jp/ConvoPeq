# PR #159 のマージと `ConvoPeq.md` の再生成

- 実施: 2026-11-06
- 承認: PR #159 のマージ、マージ後の `ConvoPeq.md` の再生成
- 制約: 再生成はソースを変更しない。既存の設計・判断資料は変更しない

## 1. PR #159 のマージ

| 項目 | 値 |
|---|---|
| PR | #159 |
| マージ方式 | squash |
| マージ commit | `48c9b726` |
| 変更 | 9 ファイル、+592 / −34 |
| 内容 | #9 の修正（`MKLNonUniformConvolver.cpp`）と、決定・分類・設計・条件・起票の記録 8 件 |
| PR の CI | コードコミット `fa49ca5f` で全項目 pass（Windows 17 分 31 秒、ASan、TSan、isr-verify、CodeQL、lint、authority、compliance） |
| main の CI（`48c9b726`） | 確認時点で Windows・Sanitizer・CodeQL は実行中。lint・authority・compliance は success |

`fa49ca5f` 以降のコミットはドキュメントのみである。マージ時点で ASan は実行中だったが、変更内容は文書に限られ、コード部分は `fa49ca5f` の CI で確認済みである。

## 2. `ConvoPeq.md` の再生成

| 項目 | 値 |
|---|---|
| 生成元 `main` | `48c9b726` |
| 生成日時 | 2026-11-06 22:03:10 |
| 自動保存されたスナップショット | `doc/ConvoPeqMD/20261106220310/ConvoPeq.md` |
| 鮮度検査 | `NEWER_SRC_COUNT: 0`、`STATUS: FRESH` |
| サイズ | 6,529,199 バイト（LF、147,940 行） |
| SHA-256 | `e4b3e0b9950b6dabd0462a92d2082c123d93c47307374dd1dd105bd20bb5e07f` |
| 前回（`f641a187`）との差 | #9 の削除とヘッダの生成時刻のみ |
| 反映 | PR #160（squash）、`main` = `a7923564` |
| 検証 | `main` の blob と作業ツリーが一致。差分は 4 ファイル（`ConvoPeq.md` と記録 3 件）のみ |

### 分割版（1 ファイル 2 MiB 以下）

| ファイル | サイズ（バイト） | SHA-256 |
|---|---|---|
| `ConvoPeq.part01.md` | 2,097,097 | `2cbaee76f214aa3b35fd1a399e8d4608e4a53789297c6455f9e97e4c0d3f35bf` |
| `ConvoPeq.part02.md` | 2,097,103 | `ee553367c2238ec24d6fa321d4067f9a2f8e5b360edb3b4663815890a55bc905` |
| `ConvoPeq.part03.md` | 2,097,106 | `143accc188ff11e46cf3309800721332ddac37f1b58c2cff5fdb718d18b26f78` |
| `ConvoPeq.part04.md` | 237,893 | `0cc92c3da9ec3553344d73a57f7fb99ea8bf35dbbbe87a83aff19e032de41f08` |

4 ファイルを順に結合すると元ファイルと一致する。

## 3. 補足

- 生成直後のファイルは CRLF だった。リポジトリの `*.md text eol=lf` に合わせて LF へ正規化してから分割した。正規化後の内容はコミットされる blob と一致する。
- 既存の設計・判断資料の変更は、`00`・`01`・`04` の状態更新と追記に限る。

## 4. 状態

| 項目 | 状態 |
|---|---|
| `main` | `a7923564` |
| PR #159 | MERGED（`48c9b726`） |
| PR #160 | MERGED（`a7923564`） |
| #9 | 解消済み |
| `ConvoPeq.md` | `a7923564` のソースに対応（FRESH） |
| #1B の実装 | 未承認 |
| soak・TX | 条件決定待ち |
