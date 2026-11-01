# ConvoPeq 統合報告（2026-11-02）

- 状態: **統合済み・配信済み（本ファイルは確定値）**
- 作業場所: `C:\VSC_Project\ConvoPeq`（唯一）
- 統合先: `lonewolf-jp/ConvoPeq` の `main`

## 1. 統合した変更

| 単位 | 内容 | コミット |
|---|---|---|
| B-2 | 診断 sink の所有権をエンジン／DSPCore へ移す（型付き化・`DiagSink`・attach/detach・14 push 経路・旧グローバル撤去） | `cfb0f477` |
| A2→A3-a | assert を CAS 結果へ移動（真の異常のみ検出）＋ deferred をスロット単位マークへ | `d501afd8` |
| 資料 | 設計・検証・統合台帳・STATUS/INDEX/VERIFICATION | `682dd700` |
| CI 規約 | atomic ドット呼び出しを `convo::` ヘルパーへ | `66068da8` |
| 資料追記 | CI 状況とポリシー同期提案 | `2d4f5d5a` |
| ポリシー同期 | `residencyContainerAllowlist` の `lineRegex` **1 行のみ** | `c9e74603` |
| `j237` ガード（セッション横断） | `ISRShutdown.h` のテスト専用 friend を `#if defined(CONVOPEQ_UNIT_TESTS)` で囲う（+4 行） | `0dfa39e3` |

## 2. 検証結果

| 構成 | 既定スイート | B-2 専用 | A2→A3-a 専用 |
|---|---|---|---|
| Release（診断 OFF） | PASS 298.60 s（2 回）＋ 299.91 s（3 回目） | PASS ×3 | PASS ×3 |
| Debug（診断 ON・assert 有効） | PASS 337.04 s | PASS ×3 | PASS ×3 |
| RelWithDebInfo + ASan | 未実行 | PASS | PASS |
| `j237` ガード単独（Release） | PASS 298.62 s | — | — |

- 実行環境: MSVC 19.51（VS 18 Enterprise）、`build-ci-local`（Ninja / Release）、`build-diag`（Ninja / Debug / 診断 ON）、`build-asan-local`（Ninja / RelWithDebInfo / ASan）
- 実行コマンドは `doc/residual-20261101/B2-IMPLEMENTATION-20261101.md` §検証に記載
- テストの削除・回数削減・timeout 緩和は行っていない

### 2.1 M1〜M7（B-2 実装後の測定）

| # | 診断 OFF（Release） | 診断 ON（Debug） |
|---|---|---|
| M1 `sizeof(DSPCore)` | 4,624,576 B | 4,657,600 B |
| M2 `diagOwnerId_` offset | 4,624,552 | 4,657,584 |
| M3 `diagSink_` offset | 4,624,560 | 4,657,592 |
| M4 `is_standard_layout_v` | false | false |
| M5 `atomic<DiagSink*>` lock-free | true / true | true / true |
| M6 診断マクロ | 0 | 1 |
| M7 コンパイラ | msvc 1951 | msvc 1951 |

測定は記録のみで、合否・安全性判定には使用していない。

### 2.2 既存 CI（GitHub Actions）

| ジョブ | 結果 |
|---|---|
| CodeQL（actions / c-cpp / csharp / javascript-typescript） | PASS |
| RelWithDebInfo ASan (MSVC, /MD) | PASS |
| Debug TSan (Clang, best-effort) | PASS |
| Windows MSVC / Release（必須チェック） | PASS |
| compliance-list-md | PASS |
| isr-verify | PASS（ポリシー同期後） |
| lint-audioengine / authority 2 件 / audit-report-verification | PASS |

> 最終的な 17 項目の結果は統合時点で確定させる。

## 3. 全セッション調査（第 1 段階）

`doc/residual-20261101/SESSION-CONSOLIDATION-LEDGER-20261101.md` に記録。

| 実測 | 値 |
|---|---|
| ローカルブランチ | 76（祖先比較の未統合 19、origin のみ 10） |
| `git cherry` の未取込ユニークコミット | 15 |
| 機能実装として未取込 | **0**（main に別 SHA で取込済み／測定専用／CI／文書） |
| worktree | 8（クリーン 4、未コミット 3） |
| 有効な未コミット変更 | **1 件**（`j237` の `ISRShutdown.h`） |

## 4. 未統合の課題（統合後に残るもの）

| # | 課題 | 状態 | 次の作業 |
|---|---|---|---|
| 1 | `j237` `ISRShutdown.h`（テスト専用 friend の `#if` ガード） | **PR #104 作成済み**（`e8e83fc5`、ローカル Release ビルド＋回帰 PASS） | CI 完了 → main 統合 |
| 2 | `j195` `AGENTS.md` の運用メモ 17 行 | 製品コードではない | 文書として反映するか判断 |
| 3 | CI 系ブランチ 5 本（r51 測定・r49・r50・h2r1） | CI 定義変更に該当 | 別途承認 |
| 4 | 陳腐化ブランチ 8 本 | 削除せず保持 | Owner 判断 |
| 5 | `j195` / `bright-falcon` の未追跡ファイル（763 / 243 件） | 証拠・ログ・スナップショット | 内容と由来の調査（削除しない） |

## 5. 残る承認事項

| 項目 | 状態 |
|---|---|
| U5 | **FAIL・既定 OFF を維持**（許容値 +50 µs、p99 基準は不変） |
| U3 / U4 | **HOLD を維持** |
| リリース・外部公開 | 未承認 |
| CI 定義変更（ワークフロー・検査スクリプト） | **未変更**（ポリシーの allowlist 1 行のみ同期） |
| 既定 OFF 機能の有効化 | 未承認 |

## 6. ソース一致

| 対象 | 値（統合後に記入） |
|---|---|
| 機能統合の merge commit | `1f258686`（B-2 / A2→A3-a）、`0dfa39e3`（j237 ガード） |
| ローカル `HEAD` / `origin/main` | 一致（`git rev-parse HEAD` と `origin/main` が同一。以降は資料のみのコミットでソースは不変） |
| 作業ツリーの追跡差分 | 0（`ConvoPeq.md` 再生成コミット後） |
| Dropbox `/ConvoPeq.md` の SHA-256 | `c12a532e70770ab40430d34052af87407291e0a12c2a9e68f10229fd5760e8a9`（6451631 B・LF 正規形） |
| 分割版 | `/ConvoPeq-c12a532e-split/`（各 ≤ 2 MiB、連結で正規と一致） |
