# domain tag 実装の最終状態確認と ConvoPeq.md の再生成

作業日時: 2026-11-12 00:46〜00:55
対象: domain tag 実装（未コミット）、`ConvoPeq.md`、分割版、Dropbox 配布

## 結論

Owner が提示した最終状態のうち、リポジトリから機械的に確認できる項目はすべて一致した。`ConvoPeq.md` と分割版は domain tag 実装より古く、実装を含んでいなかったため、Owner の指示で再生成し、分割版の作成と Dropbox への配布まで完了した。

## 一致を確認した項目

| 項目 | 実測値 | 判定 |
| --- | --- | --- |
| HEAD | `c94d376772ee22916f8350cdaa33f8d3787d16bd`（短縮 `c94d3767`） | 一致 |
| 作業ツリー（00:46 時点） | 変更 75 件・未追跡 154 件 | 一致 |
| 設計差分 v2.9 | `doc/work113/20261111-1512_domain設計/design_diff.md` 53,979 B・SHA-256 `C3AE5099…56848427` | 一致（不変） |
| 実装ファイル | `src/core/RetireDomain.h`（1,848 B・11/11 23:15）、`IRetireProvider.h`（3,528 B・23:17）、`SnapshotRetireManager.h`（1,982 B・23:30） | 存在 |
| commit／merge／release | 未実施（HEAD が変わっていない） | 一致 |

ctest 61/61 と 15-c-1・15-c-2 は本確認では再実行していない。Owner の記録をそのまま最新状態として扱う。

## 再生成前の不整合

`ConvoPeq.md` は追跡ファイルで、内容は 7,125,894 B、更新は 2026-11-11 19:07。domain tag 実装のファイル更新（23:15〜23:30）より約 4 時間早かった。

- `ConvoPeq.md` 内の `RetireDomain` / `retireDomainId` / `sanitizeDomain` の出現回数はいずれも 0。
- 分割版 `doc/ConvoPeqMD/split_20261111/ConvoPeq_part01..04.md`（各 1.95 MB 前後・更新 11/11 07:12）も 4 ファイルすべて 0。本体よりさらに古かった。
- 直近の生成スナップショット `doc/ConvoPeqMD/20261111190530/ConvoPeq.md` も 19:07 の同一内容。

このままでは次回のコード検討で参照する「最新 `ConvoPeq.md`」に実装が載らない。Owner の判断で即時再生成とした。

## 再生成の結果

- 生成本体: `ConvoPeq.md` 7,170,936 B / 159,191 行、生成日時 2026-11-12 00:48:16。
- スナップショット: `doc/ConvoPeqMD/20261112004816/ConvoPeq.md`（生成スクリプトが自動保存）。
- 作業記録: `doc/work113/20261112-0046_資料再生成/`（本体の複製、`MANIFEST.md`、`split/`）。
- 出現回数は `RetireDomain` 182、`retireDomainId` 17、`sanitizeDomain` 5、`DomainTagTests` 10、`invalidDomainAssignments` 11。実装が反映されている。

分割版は行境界で機械的に 4 分割し、各 2 MB 未満、4 ファイルの連結が本体とバイト一致することを確認した。

| ファイル | サイズ (B) |
| --- | --- |
| ConvoPeq_part01.md | 1,949,939 |
| ConvoPeq_part02.md | 1,949,999 |
| ConvoPeq_part03.md | 1,949,945 |
| ConvoPeq_part04.md | 1,321,053 |

Dropbox の `/ConvoPeq-c94d3767-split/` に本体・分割 4 件・`MANIFEST.md`・`report.md` の 7 ファイルを配置し、アップロード後のサイズがローカルと一致することを確認した。

## 作業ツリーの現在値

- 変更 75 件（今回から `ConvoPeq.md` を含む）、未追跡 161 件（00:46 時点の 154 件に、本作業で作った `doc/work113/20261112-0046_資料再生成/` の 7 ファイルを加えた数）。
- 変更件数が 76 件になっていない点は、00:46 時点で変更として計上されていた 1 件が git の index 更新で外れたためと見られる。初回のファイル一覧を残していないため、その 1 件は特定していない。
- `doc/ConvoPeqMD/` は `.gitignore` の対象なので、スナップショットはこの計数に入らない。

## 未実施

- commit／merge／release（Owner の別途承認が必要）
- 非 Windows ビルドの実測
- 実装後ソースの第三者による独立再監査

## 参照

- `ConvoPeq.md`
- `doc/work113/20261112-0046_資料再生成/MANIFEST.md`
- `doc/work113/20261111-1512_domain設計/design_diff.md`
- `doc/ConvoPeqMD/20261112004816/ConvoPeq.md`
