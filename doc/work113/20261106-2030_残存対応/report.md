# 残存課題への対応（#9 修正・#1B 設計・N1 起票・ConvoPeq.md 更新）

- 実施: 2026-11-06
- 基準: `main` = `f7dcdf7a`（`ConvoPeq.md` 更新後）
- 対象: #9 の修正、#1 残存 B の設計検討、N1 の起票、`ConvoPeq.md` の再生成、soak・TX の条件整理
- 制約: #9 は修正・テスト・差分レビュー・PR 作成まで。マージは未実施。設計と条件整理は文書のみ

## 1. `ConvoPeq.md` の再生成

| 項目 | 結果 |
|---|---|
| 生成元 | `f641a187`（ソース変更前のクリーンな状態） |
| 生成結果 | 6,530,681 バイト（約 6.2 MiB）、`--check` で FRESH |
| 分割 | 4 件（2.00 MiB × 3、0.37 MiB × 1）。結合すると元ファイルと一致 |
| 統合 | PR #157（squash）、`main` = `f7dcdf7a` |
| SHA-256 | `b44dc68af3fd61a7720faf0b62288c533787753815decb5be83e18212b141cdb`（6,530,681 バイト） |
| Dropbox | `ConvoPeq.md` と分割 4 件を納品し、SHA-256 照合一致 |

`#9` の修正は別ブランチにあるため、`main` の `ConvoPeq.md` は `f7dcdf7a` のソースに対して FRESH のままである。`#9` をマージした時点で再生成が必要になる。

## 2. #9 SetImpulse の stderr プローブ削除

### 変更

`src/MKLNonUniformConvolver.cpp` の `SetImpulse` にあった `[L0_WRITE]` の `fprintf` 4 箇所を削除した。内容は gen 3 行、geom 1 行、slot 最大 40 行で、いずれも診断用の数値トレースである。係数計算と状態には触れていない。

### 差分レビュー

| 確認項目 | 結果 |
|---|---|
| 変更ファイル | `src/MKLNonUniformConvolver.cpp` の 1 ファイルのみ |
| 差分 | 34 行削除、1 行（削除の記録コメント）追加 |
| 削除した `fprintf` | 4 箇所（gen 2、geom 1、slot 1） |
| `fprintf` の残存 | 0 箇所 |
| `<cstdio>` | 元から include なし。削除後も未使用の include は増えていない |
| `<cmath>` | `SetImpulse` の外（L419）で `std::sqrt` を使用しているため残す |
| `irLen`・`impulse` | 削除ブロックの外でも使用中（`irLen` は 13 箇所）。未使用引数の警告は出ない |
| RT 経路 | 変更なし。削除したのは読み取り専用のトレースのみ |

### 検証

| 項目 | 結果 |
|---|---|
| `AudioEngineHarness` ビルド | 成功（exit 0） |
| IR 読込を含むハーネス実行（修正後） | exit 0、321.9 秒、`all publish pipeline tests PASS`、`[L0_WRITE]` **0 行**（stdout・stderr とも） |
| 同じハーネス実行（修正前のバイナリ） | exit 0、322.3 秒、`all publish pipeline tests PASS`、`[L0_WRITE]` **456 行**（stderr） |
| 検証に使った実行ファイル | `build\Release\AudioEngineHarness.exe`、SHA-256 `bf946e1b980d177619824c9b76ba8350fff4afc6b4846accf3855850997d4ed5` |
| 全ターゲットの Release ビルド | 成功（exit 0） |
| `ctest -C Release` | **54 / 54 PASS**、327.8 秒 |

修正前と修正後は、同じ IR を読み込む同じハーネスを、削除ブロックの有無だけを変えて実行した。実行中に IR 読込が起きていることは、`irLen=48000` と `SR01Load768k3s: irLen=2097151` の記録で確認した。stderr を分離して保存し、`[L0_WRITE]` が 1 行も出ないことを確認している。

`IR_LOAD` の 1 行要約は `diagLogNonRt` 経由でコード上に残している。既定の実行では出力されず、#9 が問題にした 44 行×2 の出力とも無関係である。

### 状態

コミット `fa49ca5f`（ソース）と文書コミット、PR #159。マージは未実施。

## 3. #1 残存 B の設計（無効バンドのキー正規化）

詳細は `02-issue1b-design.md` に分離した。要点は次のとおり。

- `makeCacheKey` は無効バンドの周波数・ゲイン・Q もキーに入れるため、値ごとに別キャッシュができる。
- 係数生成は無効バンドをバイパスにし、処理側は `bandActive` を先に確認する。値は係数にも出力にも影響しない。
- 無効バンドの `frequencyBits`・`gainBits`・`qBits` を 0 に固定すれば、同じ有効設定が同じキーになる。出力は変わらない。
- `paramsHash`（= `eqCoeffHash`）の値が変わるため、期待値を持つテストや記録は実装時に洗い出す。

コード変更とテスト追加は未承認のまま。

## 4. N1 の起票

改善要望として Issue #158 を作成した。

- タイトル: 改善要望: バイパス時のテール保持とノークリック遷移（N1）
- 現行の即時ドライ切替、テール切断、状態凍結、クロスフェードなしを記載。
- クロスフェード、テール保持、解除時挙動の明確化を候補として提示。
- 実装は未承認であることを明記。

## 5. soak・TX の条件整理

詳細は `03-soak-tx-conditions.md` に分離した。

soak は実行責任者、ランナー、費用、頻度、証拠の保存先が未決定である。ランナーは oneAPI が必要なためセルフホストを推奨し、頻度はリリース前 1 回＋週次を推奨する。合否は exit 0、`ALL PASS`、`rejected=0`、`peakBacklog=0`、メモリの `converged` で判定する。

TX 実機は、採用済みの測定計画（`doc/residual-20261101/RES2-TX-MEASUREMENT-PLAN-20261106.md`）に対し、機材、計測系、実施責任者、時期、証拠の保存先が未決定である。

いずれも実行は未承認。条件が決まった後に実行計画を作成する。

## 6. PR と CI

| 項目 | 結果 |
|---|---|
| ブランチ | `fix/nuc-trace-log` |
| コミット | `fa49ca5f`（ソース）、文書コミット（本記録一式） |
| PR | #159（OPEN） |
| Windows MSVC / Release | pass（17 分 31 秒） |
| ASan（RelWithDebInfo） | pass（コードコミット `fa49ca5f`） |
| TSan（best-effort）、isr-verify、CodeQL、lint、authority、compliance | pass |
| マージ | 未実施 |

文書コミットの CI も起動する。変更はドキュメントのみのため、プロジェクトの運用規則に従い完了を待たない。

## 7. 成果物

| ファイル | 内容 |
|---|---|
| `00-owner-decision.md` | Owner 決定の記録（9 項目の判断と実行承認、対応状況） |
| `01-residual-register.md` | 残存課題の分類レジスタ（未修正バグ／非バグ既知課題／解消済み／コード外残課題） |
| `report.md` | 本報告 |
| `02-issue1b-design.md` | #1 残存 B の設計案、等価性、端ケース、テスト案、リスク |
| `03-soak-tx-conditions.md` | soak・TX の未決定事項、合否基準、実行手順、決定依頼表 |
| `04-convopeq-md-sha256.md` | `ConvoPeq.md` の生成条件、本体と分割 4 件の SHA-256 |
| `05-n1-issue-record.md` | N1 の起票記録（Issue #158） |
| `06-issue1a-hold.md` | #1 残存 A の保留判断と再審査条件 |

## 8. 状態

| 項目 | 状態 |
|---|---|
| main | `f7dcdf7a` |
| #9 | 修正済み。`fa49ca5f`（ソース）、PR #159（マージ未実施） |
| #1 B | 設計案のみ（未承認） |
| N1 | Issue #158 |
| soak・TX | 条件整理のみ（実行未承認） |
| TSan | best-effort を継続 |
| `ConvoPeq.md` | `f7dcdf7a` に対して FRESH。#9 のマージ後に再生成が必要 |
