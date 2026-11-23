# ASIO ブラックリスト失敗時の警告（#274）

作業日時: 2026-11-24 01:40
対象: main `1712983d`（統合クローン `C:\VSC_Project\integ\ConvoPeq`）

## 1. 目的

ASIO ブラックリストの用意に失敗したとき、従来はログだけ残して無言で起動していた。除外が効かないまま動き続けるため、利用者が理由に気づけない。Issue #274 として起票し、GUI では警告を出し、CLI 自動実行では出さない形に直した。

WIP にあった警告 12 行は、CLI でもダイアログが出るためそのままは取り込めない。判断を純関数に切り出し、テストで固定する方針にした。

## 2. 変更

| ファイル | 内容 |
| --- | --- |
| `src/AsioBlacklistSetup.h`（新規） | 既定テキスト、`ensureBlacklistFile`、`resolveBlacklistFile`、`isAutomationInvocation`、`shouldShowBlacklistWarning` |
| `src/MainWindow.cpp` | コンストラクタの ASIO 初期化を関数呼出しに置換。読み込み失敗時に警告（自動実行を除く） |
| `src/tests/AsioBlacklistSetupTests.cpp`（新規） | 6 ケース・24 項目 |
| `CMakeLists.txt` | `AsioBlacklistSetupTests` の登録 |

挙動の差分は 3 点だけである。

1. `loadFromFile` の戻り値を確認するようになった（従来は無視）。
2. 失敗時に `[MainWindow] ASIO blacklist could not be created or read:` を残し、GUI なら警告ダイアログを出す。
3. 両方の場所で用意できなかったときに `[MainWindow] ASIO blacklist unavailable in both locations` を残す。

C-16 のログ文言とフォールバックの順序は変えていない。

## 3. テスト

| ID | 内容 |
| --- | --- |
| A-1 | 実行ファイル隣接に無ければ既定の内容で作る（`ASIO4ALL` を含む） |
| A-2 | 既存ファイルは上書きしない（`MyDriver` が残り、既定値が混ざらない） |
| A-3 | 実行ファイル隣接に書けなければユーザー領域へ退避する（`.../ConvoPeq/asio_blacklist.txt`） |
| A-4 | どちらにも用意できなければ `unavailable`。診断用パスは実行ファイル隣接 |
| A-5 | 通知の条件（失敗かつ GUI のときだけ true） |
| A-6 | 自動実行の判定（`--cli-run`、`--cli-ir=...`、`--verify-runtime-evidence(-live)`） |

A-3 と A-4 は、実在しないディレクトリと「ファイルを親にしたパス」で作成失敗を再現している。ACL は使わない。

## 4. 検証

- Release ビルド: `ConvoPeq` と `AsioBlacklistSetupTests` がエラー 0 でリンク。
- `AsioBlacklistSetupTests`: 24/24 PASS（`[SUMMARY] failures=0`）。
- `ctest -C Release`: 69/69 PASS（189.73 秒）。新規テストは 0.06 秒。

## 5. 実 GUI / CLI での確認（2026-11-24 追記）

マージ後に、実際のバイナリで条件を確かめた。2 か所のブラックリストパスを「同名のディレクトリ」に置き換えて作成失敗を再現している（元のファイルは退避し、確認後に復元した）。

| 起動 | 結果 |
| --- | --- |
| GUI（引数なし） | タイトル `ASIO blacklist` のウィンドウが表示された |
| CLI（`--cli-run --cli-exit-ms 2000`） | ダイアログは 0 件。終了コード 0 |

どちらの起動でもログは同じ 3 行だった。

```
[ASIO] blacklist file could not be created: ...\asio_blacklist.txt
[MainWindow] ASIO blacklist unavailable in both locations
[MainWindow] ASIO blacklist could not be created or read: ...\asio_blacklist.txt
```

復元後、実行ファイル隣接の `asio_blacklist.txt` は既定内容に戻り、ユーザー領域にはファイルが無い状態（確認前と同じ）に戻した。

## 6. CI

PR #275 の CI は 19 項目中 17 項目 success、2 項目 skip（Release / RelWithDebInfo ASan / Debug TSan / CodeQL / lint / ISR 系）。マージ後の main でも `Windows MSVC Build` と静的ゲートは success。
