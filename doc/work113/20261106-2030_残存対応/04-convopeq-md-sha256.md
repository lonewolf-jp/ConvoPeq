# `ConvoPeq.md` の再生成記録と SHA-256

- 基準 `main`: `f641a187`
- 生成日時: 2026-11-06 19:41:43
- 生成ツール: `output_sourcecode_markdown.py`
- 生成後の `main`: `f7dcdf7a`（PR #157 を squash マージ）

## 1. 生成条件

| 項目 | 値 |
|---|---|
| 生成元の commit | `f641a187`（ソース変更前のクリーンな状態） |
| 作業ツリーの変更 | なし（生成時点で tracked 変更 0） |
| 生成コマンド | `python output_sourcecode_markdown.py` |
| 自動保存されたスナップショット | `doc/ConvoPeqMD/20261106194143/ConvoPeq.md` |
| 鮮度検査 | `python output_sourcecode_markdown.py --check` → `NEWER_SRC_COUNT: 0`、`STATUS: FRESH` |

## 2. 生成物

| 項目 | 値 |
|---|---|
| ファイル | `ConvoPeq.md` |
| サイズ | 6,530,681 バイト |
| 行数 | 147,973 行 |
| 改行 | LF のみ（CRLF 0） |
| SHA-256 | `b44dc68af3fd61a7720faf0b62288c533787753815decb5be83e18212b141cdb` |

## 3. 分割版（1 ファイル 2 MB 以下）

| ファイル | サイズ（バイト） | SHA-256 |
|---|---|---|
| `ConvoPeq.part01.md` | 2,097,117 | `a8df71b9b73fb960934efe96cda8571017ced70f5efb2041eb7247fc45d5c2ed` |
| `ConvoPeq.part02.md` | 2,097,121 | `2fac40a6961ddc52bcc1640afb82478c06df6ea2feb7cdc4ffa4dc27f5662aae` |
| `ConvoPeq.part03.md` | 2,097,082 | `67b54d25efef6ba803bf14e0e811befbf0c600818fcfa6acb536231f2a0eb9e5` |
| `ConvoPeq.part04.md` | 387,334 | `249b42005ff14f405ecd350399d547cd54fcba29839a881203506bf85272f892` |

- 4 分割。最大は 2,097,121 バイトで、2 MiB（2,097,152 バイト）以下。
- 4 ファイルを順に結合すると元ファイルと一致する（`sum==source`）。
- 分割版はリポジトリには置かず、Dropbox へ納品した。

## 4. リポジトリへの反映

| 項目 | 値 |
|---|---|
| PR | #157 |
| マージ方式 | squash |
| マージ commit | `f7dcdf7a` |
| 差分 | `ConvoPeq.md` のみ、622 行追加・107 行削除 |
| ソース変更 | なし（`git diff f641a187 f7dcdf7a` は `ConvoPeq.md` の 1 ファイルのみ） |
| マージ後の blob | 6,530,681 バイト、SHA-256 `b44dc68a…b141cdb`（作業ツリーと一致） |

## 5. 整合性の確認

1. 生成直後の `--check` が `FRESH`（新しいソース 0 件）。
2. `f641a187` から `f7dcdf7a` までの差分は `ConvoPeq.md` の 1 ファイルのみで、ソースは変わっていない。したがって `main` の `ConvoPeq.md` は `main` のソースに対応する。
3. 既存の設計・判断資料（`doc/` 配下）は PR #157 で変更していない。
4. マージ後の blob の SHA-256 が、生成物の SHA-256 と一致する。

## 6. Dropbox 納品

| ファイル | 結果 |
|---|---|
| `ConvoPeq.md` | アップロード後に再ダウンロードし、SHA-256 の一致を確認 |
| `ConvoPeq.part01.md` 〜 `part04.md` | 同上。4 件すべて一致 |

## 7. 注意

- `doc/ConvoPeqMD/20261106194143/ConvoPeq.md` は生成時の控えで、改行が CRLF のため SHA-256 は `3ee9eed739cd2e4b57282f6cd0d625b03eb863a96849ac7ad51417b79e2e0c7a`（6,678,654 バイト）になる。CRLF を LF に正規化すると本体と完全に一致する（同一 SHA-256）。リポジトリの `ConvoPeq.md` とは改行だけが異なる。
- `#9` の修正（PR #159）のマージ後、`main` = `48c9b726` を基準に再生成した（§8）。

## 8. 2 回目の生成（#9 マージ後）

| 項目 | 値 |
|---|---|
| 生成元 main | `48c9b726`（#159 のマージ後） |
| 生成日時 | 2026-11-06 22:03:10 |
| 自動保存されたスナップショット | `doc/ConvoPeqMD/20261106220310/ConvoPeq.md` |
| 鮮度検査 | `NEWER_SRC_COUNT: 0`、`STATUS: FRESH` |
| サイズ | 6,529,199 バイト（LF） |
| SHA-256 | `e4b3e0b9950b6dabd0462a92d2082c123d93c47307374dd1dd105bd20bb5e07f` |
| 前回との差 | #9 の削除とヘッダの生成時刻のみ |

### 分割版

| ファイル | サイズ（バイト） | SHA-256 |
|---|---|---|
| `ConvoPeq.part01.md` | 2,097,097 | `2cbaee76f214aa3b35fd1a399e8d4608e4a53789297c6455f9e97e4c0d3f35bf` |
| `ConvoPeq.part02.md` | 2,097,103 | `ee553367c2238ec24d6fa321d4067f9a2f8e5b360edb3b4663815890a55bc905` |
| `ConvoPeq.part03.md` | 2,097,106 | `143accc188ff11e46cf3309800721332ddac37f1b58c2cff5fdb718d18b26f78` |
| `ConvoPeq.part04.md` | 237,893 | `0cc92c3da9ec3553344d73a57f7fb99ea8bf35dbbbe87a83aff19e032de41f08` |

4 分割。最大は 2,097,106 バイトで 2 MiB 以下。結合すると元ファイルと一致する。

生成直後のファイルは CRLF だったため、リポジトリの `*.md text eol=lf` に合わせて LF へ正規化してから分割した。正規化後の内容は、コミットされる blob と一致する。
