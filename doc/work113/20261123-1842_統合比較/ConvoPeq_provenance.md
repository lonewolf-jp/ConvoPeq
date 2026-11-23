# ConvoPeq.md の生成記録（main `ce0ac515`）

| 項目 | 値 |
| --- | --- |
| 生成元コミット | `ce0ac515a57c0d6e04dd880d1f8fdda083f6c21d`（`origin/main`） |
| 生成元ブランチ | `integ/main-27`（`gh/main` を追跡。作業ツリーは独立クローン） |
| 生成日時 | 2026-11-23 18:38:41 |
| 生成スクリプト | `output_sourcecode_markdown.py`（main の HEAD 版、`--no-archive`） |
| 対象 | `src/`、`build.bat`、`CMakeLists.txt` |
| ファイルサイズ | 7,620,264 B |
| SHA-256 | `06a74951f9dd7bed60570c6897896362bc26068899b8e446085176654b39be18` |
| 分割 | `split/ConvoPeq_part01.md`〜`part04.md`（1,999,982 / 1,999,985 / 1,999,969 / 1,620,328 B） |
| 分割の再結合 | 4 分割の連結が元ファイルとバイト一致することを確認済み |
| 秘密情報走査 | 0 件（AWS / GitHub / `sk-` / Slack のトークン、秘密鍵ヘッダ） |

## 生成方法

`origin/main` を checkout した独立クローン（`C:\VSC_Project\integ\ConvoPeq`）で実行した。保護ツリーは使っていない。

```
python output_sourcecode_markdown.py --no-archive
```

3.9 秒で完了した。出力の見出しは作業ディレクトリ名を採るため、ディレクトリ名を `ConvoPeq` にしてから実行している（`integ27` のままだと `# Project Extract & Source Code: integ27` になる）。

## 前版との差

リポジトリに追跡されていた `ConvoPeq.md` は 2026-11-08 10:08:09 生成の 6,578,349 B で、3 週間分の変更を含んでいなかった。今回の版は main `ce0ac515` の内容を反映している。

## 注意

`doc/` 配下の未追跡ファイルには、過去の `ConvoPeq.md` スナップショットが複数ある（`doc/work113/20261111-0400_更新ソース/ConvoPeq.md` など）。これらは当時の生成物であり、**現行ソースとして取り込まない**。現行の正はリポジトリ直下の `ConvoPeq.md` と、その生成元コミットである。
