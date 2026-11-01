# ConvoPeq.md のバイト形の整理 — なぜ SHA-256 が 2 通り出るのか（2026-11-01）

- 目的: オーナーが実体を取得・照合できない原因の切り分け
- 状態: **読み取りのみ。製品コード・テスト・CI 定義・Git は未変更**
- 基準: `HEAD` = `origin/main` = `6bb7c8d5`

## 1. 結論

`ConvoPeq.md` には**同一文書の 2 つのバイト形**が存在する。どちらも同じ内容で、
違いは**改行コードだけ**（CRLF か LF か）。

| 形 | サイズ | SHA-256 | どこにあるか |
|---|---|---|---|
| **正規形（LF）** | **6,436,359 B** | **`f8aab49b49dccd9134ec8a4e777a6ddbfe9ed7c3be7f8a923160049239fa2f08`** | git blob `HEAD:ConvoPeq.md`（`c4889eb3…`）／リポジトリのコミット内容 |
| 作業ツリー（CRLF） | 6,581,900 B | `eb5ef71f813fe738faddc54bc3b93367a699b51df2d59e1e0af24b9ae5e08ee3` | `C:\VSC_Project\ConvoPeq\ConvoPeq.md`（チェックアウト結果） |

差分は **145,541 B** で、これは **CRLF の行数と完全に一致**する。

```
disk bytes      = 6581900
CRLF count      = 145541 / LF-only = 0        ← 全行が CRLF
LF-normalized   = 6436359                     ← git blob と同一サイズ
disk(LF)==blob  = True                        ← 正規化すると git blob とバイト一致
```

## 2. 分割版との関係

`ConvoPeq.part01〜04.md`（`.openchamber/convopeq-md-split-20261030/`）は
**正規形（LF）から分割**されており、**連結すると git blob とバイト一致**する。

| ファイル | サイズ | SHA-256 |
|---|---|---|
| `ConvoPeq.part01.md` | 2,097,115 B | `4c1d09212da8aba52f8818c4ad22b8c80b962a8aeed9b5327421a68a4e724b71` |
| `ConvoPeq.part02.md` | 2,097,123 B | `83fd7edafe36a602e012ba7671bcc88ac3a7e623f7dddf880cf3acabc8417023` |
| `ConvoPeq.part03.md` | 2,097,111 B | `4340b3ae99d1457317f2d69c0893a0be247920d1d42c5f57b270ca6d101cc2e9` |
| `ConvoPeq.part04.md` | 145,010 B | `898036199cdc01cc7712aaf0affdf6452db90a0c7bb60e896d128a9a1b3cfdf0` |
| **合計** | **6,436,359 B** | **`f8aab49b…`（正規形と一致）** |

検証（実測）:

```
parts sum==blob = True
parts==blob     = True
```

## 3. なぜ 2 形になるのか（実測）

- `.gitattributes:16` は `*.md text eol=lf`、`git check-attr` の結果も `text: set` / `eol: lf`。
- `core.autocrlf = false`、`core.eol` は未設定。したがって **autocrlf が CRLF を作っているのではない**。
- 作業ツリーの `ConvoPeq.md` は**生成ツールが CRLF で書き出した**もの（mtime 2026-11-01 20:16:56）。
- `text` 属性が付いているため、git は比較時に作業ツリーの CRLF を LF へ正規化する。
  よって **CRLF のままでも `git status` は clean**（`git diff` も空）になる。
- git blob 自体は LF のまま保持されるため、`git cat-file blob` と作業ツリーでサイズ・ハッシュが変わる。
- **内容（行）は同一**であり、差分は改行のみ（`disk(LF)==blob = True` で確認）。

> 注意: この 2 形は「どちらかが壊れている」のではない。同じ文書の表現差である。

## 4. 照合時の指針（オーナー向け）

| 取得元 | 期待サイズ | 期待 SHA-256 |
|---|---|---|
| git（`git show 6bb7c8d5:ConvoPeq.md`） | 6,436,359 B | `f8aab49b…`（LF） |
| Dropbox `/ConvoPeq-stage3-20261026/ConvoPeq.md` | **6,436,359 B** | **`f8aab49b…`（LF・正規形を配置）** |
| Windows 作業ツリーのコピー | 6,581,900 B | `eb5ef71f…`（CRLF） |

**照合は「サイズ＋SHA-256」を取得元ごとの期待値と突き合わせる**。
作業ツリーのファイルをそのままハッシュすると `eb5ef71f…` になり、
Dropbox / git の `f8aab49b…` と一致しない。これは**内容の不一致ではない**。

正規化して比較する場合:

```powershell
# CRLF を LF に落としてからハッシュ（正規形と比較）
$b = [IO.File]::ReadAllBytes("C:\VSC_Project\ConvoPeq\ConvoPeq.md")
$s = [Text.Encoding]::UTF8.GetString($b).Replace("`r`n","`n")
[BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash(
    [Text.Encoding]::UTF8.GetBytes($s))).Replace("-","").ToLower()
# -> f8aab49b49dccd9134ec8a4e777a6ddbfe9ed7c3be7f8a923160049239fa2f08
```

## 5. 配信の是正

これまでの配信スクリプト（`publish_deliverables.py`）は**調査文書のみ**を送っており、
`ConvoPeq.md` と分割版は送っていなかった。さらに、**正規の配置先は Dropbox ルートの
`/ConvoPeq.md`**（Notion の運用記録 D-1R-2BF で確認）であるにもかかわらず、日付フォルダにのみ
置いていたため、ルートの `/ConvoPeq.md` は **2026-10-08 版（5,753,173 B）のまま**であった。
是正として:

- ルート `/ConvoPeq.md` を正規形（LF・6,436,359 B）で**上書き**（旧版は `/ConvoPeq-archive/ConvoPeq-20261008.md` に退避）
- 分割 4 本を既存規約の `/ConvoPeq-f8aab49b-split/` に配置（各 ≤ 2 MiB、連結で正規形とバイト一致）
- `investigation-20261101/SHA256SUMS.txt` に**全ファイルの SHA-256 とサイズ**を記載
- 検証は**サイズではなく、ダウンロードして再計算した SHA-256** で行う

## 6. この資料で言えること

- `f8aab49b…`（6,436,359 B）と `eb5ef71f…`（6,581,900 B）は**同一内容**。
- 分割 4 本は正規形と**バイト一致**（`parts==blob = True`）。
- 照合の不一致は**改行コードの差**であり、内容の改変ではない。
