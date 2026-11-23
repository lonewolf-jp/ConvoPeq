# 提出物の取得経路（MANIFEST）

作成: 2026-11-11 23:50 JST（v2.9） / 対象: domain tag 設計差分と 2026-11-11 版ソース（reason2 スナップショット）

**受理**: 2026-11-11、Owner が v2.9 と 15-c-1／15-c-2 の検証方法を承認した。設計差分 v2.9 は
53,979 B・SHA-256 `C3AE50991D9AB0A50BF09C9D8BDBD0B8919AC4F04732F567D93AE76656848427` で確定とし、以後変更しない。
受理記録と実装着手前の干渉チェックは `/ConvoPeq-domain-design-20261111/acceptance/` にある。

## 1. アップロード先アカウント（実測）

| 項目 | 値 |
|---|---|
| Dropbox アカウント | `masaki.tachibana@gmail.com` |
| account_id | `dbid:AABX_ob…`（先頭 12 文字） |
| home_namespace_id | `15370767795` |
| ルート | 通常のユーザールート（team space ではない） |

`files/list_folder` と `files/get_metadata` で**存在を確認済み**（2026-11-11 22:45 JST に再確認）。

## 2. ファイル（実パスとサイズ）

| 実パス | サイズ | 更新（UTC） |
|---|---|---|
| `/ConvoPeq-domain-design-20261111/design_diff.md` | 53,979 B（v2.9: 15-c の条件行列（5 条件）を実測して §3.2 に追加・15-c-2 のデバッガ設定に Microsoft Learn の `sxd` の根拠と「実測／文献」の区別を明記。設計は v2.6 の B-1 のまま） | 2026-11-11T13:24:12Z |
| `/ConvoPeq-domain-design-20261111/MANIFEST.md` | 本ファイル | — |
| `/ConvoPeq-domain-design-20261111/v26/report.md` | v2.6 の報告（Release 契約の確定・版表記の修正） | — |
| `/ConvoPeq-domain-design-20261111/v26/evidence/contract_check.cpp` | v2.6 の契約ハーネス。`jassertfalse` を含まないため **#15-a 相当のみ**を検証している | — |
| `/ConvoPeq-domain-design-20261111/v26/evidence/contract_check_build_run.txt` | v2.6 の Debug／Release ビルドと実行ログ | — |
| `/ConvoPeq-domain-design-20261111/v26/evidence/source_state.txt` | ソース現状（未実装の再確認・生成箇所） | — |
| `/ConvoPeq-domain-design-20261111/v27/report.md` | v2.7 の報告（#15 の分割・証跡の区別・サイズと取得経路の修正） | — |
| `/ConvoPeq-domain-design-20261111/v27/evidence/assert_probe.cpp` | 15-c の方法実証プローブ（`jassertfalse` を含む。製品 ctor の検証ではない） | — |
| `/ConvoPeq-domain-design-20261111/v27/evidence/assert_probe_build_run.txt` | プローブの Debug／Release ビルドと実行ログ（いずれも PASS） | — |
| `/ConvoPeq-domain-design-20261111/v27/evidence/size_consistency.txt` | 設計差分の実サイズと、報告・MANIFEST の記載一致の確認 | — |
| `/ConvoPeq-domain-design-20261111/v28/report.md` | v2.8 の報告（15-c を JUCE の実条件へ修正・3 条件の実測） | — |
| `/ConvoPeq-domain-design-20261111/v28/evidence/assert_probe.cpp` | JUCE 8.0.12 の実物ヘッダを使うプローブ（`jassertfalse` は実マクロ） | — |
| `/ConvoPeq-domain-design-20261111/v28/evidence/mini_debugger.cpp` | 15-c-2 用の最小デバッガ（`DEBUG_ONLY_THIS_PROCESS`・初回例外をアプリへ渡す） | — |
| `/ConvoPeq-domain-design-20261111/v28/evidence/assert_probe_build_run.txt` | 3 条件（Debug 非接続／Debug 接続／Release）のビルドと実行ログ（すべて PASS） | — |
| `/ConvoPeq-domain-design-20261111/v28/evidence/size_consistency.txt` | 設計差分の実サイズと、報告・MANIFEST の記載一致の確認 | — |
| `/ConvoPeq-domain-design-20261111/v29/report.md` | v2.9 の報告（15-c の条件行列・実測と文献の区別） | — |
| `/ConvoPeq-domain-design-20261111/v29/evidence/assert_probe.cpp` | JUCE 実ヘッダを使うプローブ（ガード強制スイッチ付き） | — |
| `/ConvoPeq-domain-design-20261111/v29/evidence/mini_debugger.cpp` | 15-c-2 用の最小デバッガ | — |
| `/ConvoPeq-domain-design-20261111/v29/evidence/condition_matrix.txt` | 5 条件（Debug／Release × デバッガ有無 × ガード強制）のビルドと実行ログ・要約表 | — |
| `/ConvoPeq-domain-design-20261111/v29/evidence/size_consistency.txt` | 設計差分の実サイズと、報告・MANIFEST の記載一致の確認 | — |
| `/ConvoPeq-domain-design-20261111/acceptance/report.md` | v2.9 の受理記録と着手前チェック（承認条件の反映・証拠区分の維持・干渉 20 ファイル） | — |
| `/ConvoPeq-domain-design-20261111/acceptance/overlap_analysis.md` | 実装が触る 20 ファイルの状態と干渉評価・着手前 SHA-256 | — |
| `/ConvoPeq-domain-design-20261111/acceptance/baseline_git.txt` | 読み取りのみのベースライン（HEAD `c94d3767`・変更 63 件・未追跡 149 件・`git diff --stat`） | — |
| `/ConvoPeq-source-20261111-reason2/ConvoPeq.md` | 7,125,894 B | 2026-11-11T10:07:52Z |
| `/ConvoPeq-source-20261111-reason2/ConvoPeq_part1.md` | 1,899,936 B | 2026-11-11T10:07:54Z |
| `/ConvoPeq-source-20261111-reason2/ConvoPeq_part2.md` | 1,899,989 B | 2026-11-11T10:07:55Z |
| `/ConvoPeq-source-20261111-reason2/ConvoPeq_part3.md` | 1,899,984 B | 2026-11-11T10:07:56Z |
| `/ConvoPeq-source-20261111-reason2/ConvoPeq_part4.md` | 1,426,010 B | 2026-11-11T10:07:58Z |
| `/ConvoPeq-stage3-20261111/report.md` | 12,457 B | 2026-11-11T05:26:41Z |

## 3. 共有リンク（誰でも取得可能・rlkey 付き）

| 対象 | URL |
|---|---|
| 設計差分フォルダ | https://www.dropbox.com/scl/fo/rmswtz9z379thgdqscoek/AN5mdlh4GQAewVluBC-DUi4?rlkey=ywg542lfe5261c8c7bq465rws&dl=0 |
| 2026-11-11 版ソース（reason2・`ConvoPeq.md` + 分割 4 本） | https://www.dropbox.com/scl/fo/953eiuteh7m0dutpq86lm/AMZh6X5BUlgBzUgXnCMVtnI?rlkey=2q59oy4erc40qj2ticeg90cab&dl=0 |
| 段階③の報告・証跡 | https://www.dropbox.com/scl/fo/0iikem05m1w2ma6o0aw3t/APjGzI5thLYh9EE_N7YmfXg?rlkey=sxayezi912vd0tu3l40f1jl1l&dl=0 |

## 4. 取得できない場合の代替（ローカル）

- 設計差分: `C:\VSC_Project\ConvoPeq\doc\work113\20261111-1512_domain設計\design_diff.md`
- 2026-11-11 版ソース（reason2）: リポジトリ直下の `ConvoPeq.md`（7,125,894 B・2026-11-11 19:07 JST 生成）が Dropbox の reason2 と同一サイズ。
  分割 4 本は Dropbox の `/ConvoPeq-source-20261111-reason2/` にある。ローカルの
  `doc/work113/20261111-1245_段階3分類/ConvoPeq_part1..4.md` は 14:24 生成の **stage3 版**（`ConvoPeq.md` 7,124,929 B に対応）であり、
  reason2 の分割としては使わない。
- 再生成: `C:\Python315\python.exe output_sourcecode_markdown.py`（リポジトリ直下で実行）
