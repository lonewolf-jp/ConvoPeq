# RES2 検証準備と実走の可否（2026-11-06）

- 実施: 2026-11-06 03:00–03:30（Windows 11 x64、WSL Ubuntu 26.04）
- 基準: `main` = `60854ea8`（作業開始時）
- 結論: TSan の限定実走は構造上できない。soak は承認条件により未実行。TX 実機は機材・シナリオ・実施者の指定が無く未実施。

## 1. TSan（RES2-11）

### 試行したこと

WSL で次を実行した。

```
cmake -S . -B /tmp/tsan-build -G "Ninja Multi-Config" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DENABLE_TSAN=ON -DCONVOPEQ_REQUIRE_MKL=OFF -DCMAKE_BUILD_TYPE=Debug
```

### 結果

`Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)` で停止した。

### 構造上の遮断要因

| # | 事実 | 影響 |
|---|---|---|
| 1 | `ENABLE_TSAN` は `ConvoPeq` にのみ `-fsanitize=thread` を付ける（`CMakeLists.txt:2101-2114`） | `ISRSemanticValidationTests` は計装されない |
| 2 | テストは `ConvoPeq` に依存する（`add_dependencies`） | Linux では JUCE GUI の全依存が必要 |
| 3 | WSL に pkg-config と JUCE 依存が無い | configure が通らない |

- このため、現行構成で実走してもテストは TSan で動かない。証拠を得るには、テストターゲットへの TSan 付与（CMakeLists の変更）と、WSL への依存導入が必要で、いずれも承認範囲外である。
- 実走は未実施。実測時間も未取得（skip 時のジョブ時間 0.4〜1.1 分のみ）。

## 2. soak（RES2-12）

| 項目 | 結果 |
|---|---|
| 第一候補 | oneAPI 導入済みセルフホストランナー |
| この開発機 | oneAPI あり（MKL・compiler） |
| ローカル実測 | 799.9 秒（13.3 分）、ALL PASS |
| CI | 0.6 分で skip（oneAPI 不在） |
| 週次・schedule | 未承認のため未実施 |

- 実運用に必要な条件（oneAPI、権限、環境変数、復旧手順、費用、運用責任）を整理した。
- 費用はマシンの電気代と保守。GitHub の分数課金は無い。

## 3. TX 実機（RES2-13）

| 指標 | 取得方法 |
|---|---|
| ブロック処理時間 | `CONVOPEQ_RT_MEASURE=1` で `[RTM] ... max=... us` を stdout に出す |
| XRUN | `xRunBuffer` を非 RT 側（`AudioEngine.Timer.cpp:1621`）が pop して診断ログに出す |
| 最終値・再構築回数 | 既存のハーネス／診断ログ |

- 実機（ASIO デバイス）、シナリオ、実施者が未指定のため開始していない。条件を独自に決めて実施しない。

## 4. 更新した資料

| 資料 | 更新 |
|---|---|
| `RES2-VERIFICATION-READINESS-20261106.md` | 新規。3 件の準備状況、遮断要因、成果物、追加承認 |
| `RES2-TSAN-EVIDENCE-20261106.md` | 実走の試行結果と構造上の問題（§9） |
| `RES2-SOAK-RUNNER-OPTIONS-20261106.md` | 環境確認と必要要件（§7） |
| `RES2-TX-MEASUREMENT-PLAN-20261106.md` | データ取得口の確認（§8） |

## 5. 検証

| 確認 | 結果 |
|---|---|
| 表の列数 | 4 ファイルで不一致 0 |
| 偽漢字 | 0 |
| 日本語 lint | 新規資料 0 |
| 文書のみ | `src`・workflow・既定値の変更なし |
| PR | #148（`ce386431`）。doc のみのため CI を待たずに管理者マージ |
| `main` / `origin/main` | `ce386431` / `ce386431`（一致） |
| `ConvoPeq.md` | FRESH |
| Dropbox | 14 資料を納品し、再取得照合すべて一致 |

## 6. 追加承認が必要な事項

| 事項 | 内容 |
|---|---|
| TSan の計装 | テストターゲットへ `-fsanitize=thread` を付ける変更 |
| TSan 環境 | WSL への pkg-config・JUCE 依存の導入、または専用 Linux 環境 |
| soak の実行 | セルフホストランナーの構築と、1 回の手動実行の承認 |
| TX の実施 | 実機・シナリオ・実施者の指定と、測定開始の承認 |
