# domain tag 設計 v2.7 — #15 の Debug 検証の分離と証跡の整理

Owner レビュー（2026-11-11・3 回目）の指摘 3 点を反映した。設計そのもの（B-1）は v2.6 から変えていない。**製品コードは未変更**で、実装は承認済みだが #15-c の方法確認を待って着手する。

## §0 結論

- **指摘 1（#15 の Debug 検証）**: §7.2 #15 を **15-a（共通変換ロジック）／15-b（Release の実 ctor）／15-c（Debug の実 ctor と加算順序）** に分けた。v2.6 の証跡 `contract_check.cpp` は `jassertfalse` を含まないため、**15-a しか検証していなかった**と明記した。前回「§3.2 のコードをそのまま写した」と書いたのは誤りで、これも訂正した。
- **15-c の方法は実証済み**。`jassertfalse`（`JUCE_DEBUG` のとき `__debugbreak()`）を含むプローブで、Debug では `EXCEPTION_BREAKPOINT`（0x80000003）を捕捉でき、**捕捉後のカウンタが +1**（＝ assertion より前に加算）であることを確認した。Release では捕捉されず `Unknown` に落ちてカウンタが +1 になる。ただしこれは**方法の実証**であり、製品 ctor に対する 15-b／15-c は実装後に実施する。
- **指摘 2（サイズの不一致）**: 本版の実サイズ **48,891 B** を報告と MANIFEST の両方に書いた。一致は機械的に確認している（§2）。
- **指摘 3（取得経路）**: MANIFEST のソース一覧を `/ConvoPeq-source-20261111-reason2`（2026-11-11T10:07Z）に更新した。
- 承認範囲は Owner の 6 項目（`RetireDomainId`、明示ドメイン ctor、`sanitizeDomain()` と不正値カウンタ、provider／router の伝播、`DeletionEntry` の刻印、§7.2 の 15 項目）に限る。Reader レジストリ統合・ASan・commit／merge／release は未承認のままである。

## §1 #15 の分割と検証方法

| 副検証 | 対象 | 合格条件 | v2.6 の証跡で確認できていたか |
|---|---|---|---|
| 15-a | 共通変換ロジック（両構成） | 判定子の真偽、`sanitizeDomain()` の `Unknown` への変換、許可値の採用 | **できていた**（`contract_check.cpp`） |
| 15-b | Release の実 ctor | 製品 `EpochDomain` に不正値を渡すと `domainId() == Unknown`、カウンタが 1 件増える、停止しない | できていなかった（実 ctor が未実装） |
| 15-c | Debug の実 ctor | `jassertfalse` が発火し、**その前にカウンタが加算される** | できていなかった（プローブに `jassertfalse` が無い） |

15-c の手順は、`__try` で `EpochDomain bad(static_cast<RetireDomainId>(5));` を構成し、`__except` で `EXCEPTION_BREAKPOINT` を捕捉したうえで、捕捉後に `invalidDomainAssignments()` を読む。加算が assertion より前であることは、この読み値でしか確かめられない。assertion を発火させずフォールバックだけを見る確認は 15-a であり、15-c の代用にしない。

### 1.1 方法の実証（プローブ）

`evidence/assert_probe.cpp` は、§3.2 の ctor と同じ形（`jassertfalse` を含む）を再現したプローブである。JUCE 本体は持ち込まず、`jassertfalse` の定義だけを同じ形（`JUCE_DEBUG` のとき `__debugbreak()`）で写している。

| 構成 | ビルド | 結果 |
|---|---|---|
| Debug | `clang++ -std=c++17 -O0 -fms-extensions -DJUCE_DEBUG=1 -Wall -Wextra` → 警告なし | `EXCEPTION_BREAKPOINT`（0x80000003）を捕捉。捕捉後のカウンタが +1。`PASS` |
| Release | `clang++ -std=c++17 -O2 -DNDEBUG -fms-extensions -Wall -Wextra` → 警告なし | 捕捉なし。`domainId() == Unknown`、カウンタ +1。`PASS` |

このプローブで言えるのは「方法が成立する」ことまでで、製品 ctor の検証ではない。実装後に同じ手順を製品 `EpochDomain` に対して実行し、15-b／15-c の証跡とする。

## §2 サイズと取得経路

| 項目 | 値 |
|---|---|
| 設計差分の実サイズ | **48,891 B** |
| 報告と MANIFEST の記載 | どちらも 48,891 B（`evidence/size_consistency.txt` で機械確認） |
| 最新ソース | `/ConvoPeq-source-20261111-reason2`（2026-11-11T10:07Z） |

reason2 の内訳（Dropbox のメタデータで実測）:

| ファイル | サイズ | 更新（UTC） |
|---|---|---|
| `ConvoPeq.md` | 7,125,894 B | 2026-11-11T10:07:52Z |
| `ConvoPeq_part1.md` | 1,899,936 B | 2026-11-11T10:07:54Z |
| `ConvoPeq_part2.md` | 1,899,989 B | 2026-11-11T10:07:55Z |
| `ConvoPeq_part3.md` | 1,899,984 B | 2026-11-11T10:07:56Z |
| `ConvoPeq_part4.md` | 1,426,010 B | 2026-11-11T10:07:58Z |

## §3 変更した箇所（設計差分 v2.7）

| 節 | 変更 |
|---|---|
| 冒頭・§0.1 | v2.7 の行を追加し、v2.6 の「本版」を外した |
| §0.2 | 指摘 3 点の対応表に差し替え、承認範囲 6 項目を明記 |
| §3.2 | 「証跡の区別（v2.7）」を追加（旧ハーネスは 15-a のみ／プローブは方法の実証／実装後に 15-b・15-c） |
| §7.2 | #15 を 15-a／15-b／15-c に分割し、15-c の手順を注記。「3 副検証がそろって初めて #15 の完了」と明記 |
| §9 | #1 を v2.7 に、#2 を「実装は承認済み・15-c の方法確認後に着手」に更新 |
| §11 | v2.6 と v2.7 の証跡パスを追加 |

## §4 成果物

| 項目 | 内容 |
|---|---|
| 設計差分 | `doc/work113/20261111-1512_domain設計/design_diff.md`（v2.7・48,891 B） |
| 取得経路 | `doc/work113/20261111-1512_domain設計/MANIFEST.md` |
| 証跡 | 本報告の `evidence/`（プローブ本体・ビルド実行ログ・サイズ一致の確認） |
| コード変更 | なし |
