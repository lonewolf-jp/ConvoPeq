# 作業ベースの移行記録（2026-11-24 決定）

作業日時: 2026-11-24 01:45
対象: `C:\VSC_Project\integ\ConvoPeq`（main）と `C:\VSC_Project\ConvoPeq`（保護ツリー）

## 1. 決定

2026-11-24 の Owner 判断を記録する。保護ツリーへの一括反映は行わず、統合クローンを今後の作業ベースにする。

| 対象 | 決定 |
| --- | --- |
| 日常作業の移行先 | `C:\VSC_Project\integ\ConvoPeq`（main `1712983d` 以降） |
| 保護ツリー `C:\VSC_Project\ConvoPeq` | 凍結。読み取り専用。反映しない |
| 保全ブランチ `wip/preserve-20261123` | `adfb16f6` で凍結維持 |
| 保護ツリーへの一括反映 | しない（反映計画は `doc/work113/20261124-0030_反映計画/report.md` に記録済み） |
| ASIO 警告 12 行 | 条件付きで採用。独立 PR にする |
| `opencode.json` の `protocol` 行 | main には載せない。ローカル設定として維持 |
| B-07 の残り検証 | 独立課題として継続 |
| B-11 | 実測必須 artifact の一覧を先に確定 |
| Issue #265 / #266 | #265 を先行 |

## 2. 実施したこと

| 項目 | 状態 |
| --- | --- |
| ASIO 警告（#274） | PR #275。判断を純関数に切り出し、24 項目のテストを追加した。Release の `ctest` は 69/69 |
| #265（OOM 経路） | PR #276。`create()` の `std::bad_alloc` を境界で受け、nullptr 判定を到達可能にした |
| #266（CodeQL 3 件） | PR #277。`snprintf` の戻り値加算をクランプし、`soak-ci.yml` に最小権限を追加した |
| B-11 の一覧 | `doc/work113/20261124-0200_B11一覧/report.md`。11 種の artifact と live 1 種を、実測主体・条件・必須区分で整理した |
| B-07 の残り検証 | Issue #278 として起票した |

## 3. 実務上の変更

- ビルドとテストは統合クローンの `build/`（Release）を使う。保護ツリーの `build/` は使わない。
- 保護ツリーは参照専用とする。WIP の復元は次の 3 経路。
  1. 保全ブランチ `wip/preserve-20261123`（`adfb16f6`）
  2. `%TEMP%\opencode\refl20261124\` の `tracked_changes.patch` / `untracked_files.zip` / `manifest.tsv`
  3. Dropbox `/ConvoPeq-reflection-plan-20261124/`（上記と同内容）
- `opencode.json` の graphify `protocol` 行は保護ツリーにだけ存在する。統合クローンで graphify を使うときは同じ 1 行を入れる。main の設定には入れない（環境依存のパスを含むため）。
- セッションの作業ディレクトリはツール側の制約で保護ツリーを指したままである。編集・ビルドはパスを明示して統合クローンで行う。

## 4. 移行しなかったもの

保護ツリーの 398 エントリのうち、統合で採用しなかった系統（probe 計測、B-07、B-11 の代替実装、5 本の ISR スクリプト、reader 計測）は統合クローンへ持ち込んでいない。必要になった時点で、上記の保全から個別に取り出す。

## 5. 関連

- `doc/work113/20261124-0030_反映計画/report.md`（ファイル単位の対応表。実施しない判断の根拠）
- `doc/work113/20261124-0140_ASIO警告/report.md`
- `doc/work113/20261124-0220_265修正/report.md`
- `doc/work113/20261124-0240_266修正/report.md`
- `doc/work113/20261124-0200_B11一覧/report.md`

## 6. 復元訓練（2026-11-24）

保全アーティファクトから保護ツリーの状態を戻せることを、使い捨てのクローンで確かめた。

1. `git clone --shared -n` で複製し、`4d476d5c` をチェックアウトする。
2. `tracked_changes.patch` を `git apply` する（43 ファイルすべてクリーンに適用）。
3. `untracked_files.zip` を展開する（355 ファイル）。
4. 結果は **398 エントリ・シグネチャ `8f226d032b644f25`** で、保護ツリーと一致した。
5. ファイル単位では 394 件が完全一致し、4 件（`ConvoPeq.md`、`MainWindow.cpp`、`LoadPipeline.cpp`、`StateAndUI.cpp`）だけ改行が CRLF → LF に正規化されていた。`.gitattributes` の `text` 正規化によるもので、内容は同一である。

改行まで含めて戻せるように、追跡 43 ファイルの作業ツリー版を `tracked_modified_files.zip` として追加した。これを展開すると **398 ファイルすべての SHA-256 が manifest と一致**する（確認済み）。

限界として、`git apply` だけでは追跡ファイルの改行バイトが揃わないため、厳密な復元には `tracked_modified_files.zip` を併用する。
