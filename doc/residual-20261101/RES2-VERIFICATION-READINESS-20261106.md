# RES2 検証準備・限定実走の記録（2026-11-06）

- 基準: `main` = `60854ea8`（作業開始時）。作業ブランチは作らず、記録のみを作成した。
- 目的: RES2-11（TSan）、RES2-12（soak）、RES2-13（TX 実機）の検証可能性を確認し、実走の可否と遮断要因を記録する。
- 結論: TSan の限定実走は**構造上できない**（テスト対象に TSan が付かない）。soak は承認条件により未実行。TX 実機は機材・シナリオ・実施者の指定が無く未実施。

## 1. 共通の前提

| 項目 | 値 |
|---|---|
| 基準 commit | `60854ea8` |
| 変更対象 | なし（本記録は文書のみ） |
| RT 経路・Publish・Retire・Epoch・Crossfade・DSP | 未変更 |
| 既定値 | 未変更 |
| CI workflow | 未変更 |
| waiver | 未適用 |

## 2. RES2-11（TSan）

### 2.1 環境構成の確認

| 項目 | 結果 |
|---|---|
| WSL | Ubuntu 26.04 |
| clang | 21.1.8 |
| cmake | 4.4.4 |
| ninja | 1.13.2 |
| pkg-config | **無し** |

### 2.2 configure の結果

`cmake -S . -B /tmp/tsan-build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DENABLE_TSAN=ON -DCONVOPEQ_REQUIRE_MKL=OFF -DCMAKE_BUILD_TYPE=Debug` は、JUCE のモジュール解決で停止した。

> Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)

### 2.3 構造上の遮断要因

| # | 事実 | 影響 |
|---|---|---|
| 1 | `ENABLE_TSAN` は `ConvoPeq` ターゲットにのみ `-fsanitize=thread` を付ける（`CMakeLists.txt:2101-2114`） | `ISRSemanticValidationTests` は TSan で計装されない |
| 2 | テストターゲットは `ConvoPeq` に依存する（`add_dependencies`） | Linux では JUCE GUI の全依存（X11/ALSA 等）が必要 |
| 3 | WSL に pkg-config と JUCE 依存が無い | configure が通らない |

- したがって、現行の CMake 構成のままでは、TSan の診断実走を行っても**テストは TSan で動かない**。TSan の証拠を得るには、テストターゲットへ `-fsanitize=thread` を付ける変更（ソース変更）が必要で、本承認の範囲外である。
- CI の skip（`[TSan] Skipped (no build configured)`）は、この構造と整合する。

### 2.4 評価条件 1〜5 の充足状況

| # | 条件 | 状態 |
|---|---|---|
| 1 | 対象テストの固定と再評価条件 | 未充足。対象リストは資料のみ |
| 2 | skip の理由・発生率・環境依存性の記録 | 未充足。skip は success |
| 3 | 連続成功率と失敗分類 | 未充足。success は skip |
| 4 | 実行時間と CI コスト | 未充足。実測は skip の 0.4〜1.1 分のみ |
| 5 | 利用可能／不可能な環境の分離 | 未充足 |

### 2.5 実走の結論

- 限定診断実走は**未実施**。理由は 2.3 の構造要因と、2.2 の環境不足である。
- 実行可能になる条件: テストターゲットへの TSan 付与（CMakeLists の変更）、WSL への pkg-config と JUCE 依存の導入。どちらも追加承認が必要。
- 実測実行時間: 取得できず（skip 時のジョブ時間 0.4〜1.1 分のみ）。

## 3. RES2-12（soak）

### 3.1 ランナー構成の確認

| 項目 | 結果 |
|---|---|
| 第一候補 | oneAPI 導入済みセルフホストランナー |
| この開発機の oneAPI | あり（`C:\Program Files (x86)\Intel\oneAPI`、MKL/compiler） |
| 既存の手動実行 | `workflow_dispatch`（`scenario=all`）またはローカルの `AudioEngineHarness.exe --soak --scenario=all` |
| ローカル実測 | 799.9 秒（13.3 分）、ALL PASS |
| CI 実行 | 0.6 分で skip（oneAPI 不在） |

### 3.2 実運用に必要な条件（案）

| 項目 | 内容 |
|---|---|
| oneAPI | Base Toolkit（MKL と compiler）を導入済みであること |
| ドライバー | ASIO は不要（soak はオーディオデバイスを使わない） |
| 権限 | ランナーサービスの実行ユーザーがビルドと実行を行えること |
| 環境変数 | `MKLROOT`、`PATH` に compiler/MKL の bin |
| 復旧 | 失敗時にランナーを再起動する手順、ログの保存先 |
| 費用 | マシンの電気代と保守。GitHub の分数課金は無い（セルフホスト） |
| 運用責任 | ランナーの更新、ディスク、セキュリティの担当 |

- 費用と運用責任は、導入前に確定する必要がある（Owner 条件）。
- 週次化・`schedule` 追加・soak の実行は、いずれも未承認のため行っていない。

## 4. RES2-13（TX 実機）

### 4.1 測定データの取得口

| 指標 | 取得方法 |
|---|---|
| ブロック処理時間 | `CONVOPEQ_RT_MEASURE=1` で `RtMeasureRing` が `[RTM] ... max=... us` を stdout に出す（`RtMeasureRing.h:1004`） |
| XRUN | `xRunBuffer` に積まれ、非 RT 側（`AudioEngine.Timer.cpp:1621`）が pop して診断ログに出す |
| 最終値・再構築回数 | 既存のハーネス／診断ログ |

### 4.2 実機測定の実施条件（未充足）

| 項目 | 状態 |
|---|---|
| 実機（ASIO デバイス）の指定 | **未指定**（Owner が承認時に指定する必要がある） |
| シナリオと実施者 | **未指定** |
| 測定計画 | 採用済み（`RES2-TX-MEASUREMENT-PLAN-20261106.md`） |
| 実施 | 未実施 |

- 機材・シナリオ・実施者が指定されていないため、測定は開始していない。条件を独自に決めて実施することは、承認条件に反する。

## 5. 成果物と未確認事項

### 5.1 成果物

| 種別 | 内容 |
|---|---|
| TSan 環境構成記録 | 本記録 §2.1〜2.3 |
| 評価条件 1〜5 の充足状況 | 本記録 §2.4 |
| 対象テスト一覧 | `RES2-TSAN-EVIDENCE-20261106.md` §3（現行 1＋候補 5） |
| 実走結果 | 未実施（遮断要因を記録） |
| 実測実行時間 | 未取得（skip 時のジョブ時間のみ） |
| soak ランナー構成記録 | 本記録 §3 |
| 実測時間 | ローカル 13.3 分（既存測定） |
| 費用・保守・復旧の評価 | 本記録 §3.2（定性） |
| TX 測定手順と取得口 | `RES2-TX-MEASUREMENT-PLAN-20261106.md`、本記録 §4.1 |

### 5.2 未確認事項

- TSan の検出能力、実実行時間、失敗分類（実走できていない）。
- セルフホストランナーの実運用（時間・安定性・費用）。
- 実機での 1.0 ms 達成、XRUN の有無、最終値一致。
- waiver の妥当性（未判断）。

## 6. 追加承認が必要な事項

| 事項 | 内容 |
|---|---|
| TSan の計装 | 実施済み（2026-11-06 の受理対象） |
| TSan 環境 | 実施済み（WSL に pkg-config・JUCE 依存を導入） |
| TSan の MKL・IPP Linux 対応 | 未承認。必要性と実現可能性を別途判断 |
| soak の実行 | 手動 1 回は実施・受理済み。追加実行は未承認 |
| soak のランナー運用 | 未承認。登録・費用・責任者は未決定 |
| TX の実施 | 実機・シナリオ・実施者・時期の指定待ち |
