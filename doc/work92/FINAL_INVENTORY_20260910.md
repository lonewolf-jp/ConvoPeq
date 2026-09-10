# work92 最終棚卸し報告書（authority 更新後の検証完了報告）

- **作成日**: 2026-09-10
- **authority**: ConvoPeq.md `Generated: 2026-09-10 02:03:43` / **NEWER_SRC_COUNT=0**（work92 実装マーカー 30 件反映確認済み）
- **位置づけ**: `IMPLEMENTATION_REPORT_20260910.md` の残置事項（AC-C8-2・TSan・静的解析）と、実装後の全残置項目の再判定を確定する。
- **前提**: ユーザー判定「GO（次フェーズ進行可・ただし 2026-09-10 版 ConvoPeq.md を authority として参照）」を受領済み

## 1. AC-C8-2 — icx Release ビルド検証（**完全達成**）

| 項目 | 結果 |
|---|---|
| icx Release ビルド | ✅ **PASS** — 552/552・`build-icx/ConvoPeq_artefacts/Release/ConvoPeq.exe` 生成（48.7MB）・LLVM OOM 再発なし（/O2 維持確認） |
| icx Release CTest | ✅ **40/40 PASS**（48 秒）— `oneAPI setvars` 環境で MKL/IPP runtime DLL を解決して実行（PATH 未設定時は 0xc0000135 = DLL not found になる点を環境注意として記録） |
| CMake flag 反映確認 | ✅ icx build の compile_commands.json に `/O2 /DNDEBUG /fp:fast /Gy /Zi /utf-8 /EHsc`（**/QxCORE-AVX2 は global から除去済み**・ConvoPeq target の Release+CXX のみ）を確認 |
| MSVC Debug / Release | ✅ CTest 40/40 ×2（C-8 の /fp:precise 化後の数値回帰なし = AC-C8-1 達成） |

**C-8 の全 AC が達成された**: AC-C8-1（MSVC）✅・AC-C8-2（icx）✅・AC-C8-3（compiler 判定分岐）✅・AC-C8-4（ISA 実測手順の文書化）✅

## 2. C-8 文書化（AC-C8-4・新規作成）

| ファイル | 内容 |
|---|---|
| `README.md`（Compiler / CPU support matrix 追加） | MSVC Release = AMD supported（/fp:precise）/ icx Release = AMD **unsupported（公式サポート対象外）**・AVX2 subset 実測注記（G6）/ icx Debug = 同 unsupported — 「実行可否は断定しない」表記を維持 |
| `doc/work92/ICX_ISA_AUDIT_PROCEDURE.md`（新規） | icx 生成 ISA の再実測手順（/FA 出力 → fc 比較 → Intel 専用命令チェックリスト → 判定基準 → 実測記録テーブル）。CMake AVX2 flag 変更・icx 更新・AVX-512/AMX 依存コード追加時に再実行 |

## 3. 静的解析（変更ファイル対象）

| ツール | 結果 |
|---|---|
| **cppcheck**（C++20・win64・warning/performance/portability） | ✅ **work92 変更由来の指摘 0 件**。EQProcessor.Core.cpp（B-7a/B-7b 変更部）への直接指摘なし。検出された uninitMemberVar 系警告は全て **pre-existing**（EQProcessor.h:295 BandNode・ISRClosure.h 等・今回の変更範囲外） |
| **clang-tidy** | ⚠️ **環境非互換として記録** — MSVC ccdb（/FS /Zm400 /FS 等の MSVC 固有 flag）を clang が解釈できず、icx ccdb は JuceHeader include path が fix_compile_commands 処理で失われている。本プロジェクトの toolchain 構成では実行不可。代替として cppcheck + コンパイラ警告（MSVC/icx 両方）で監査済み |

## 4. 動的検証

| ツール | 結果 |
|---|---|
| **CTest Debug** | ✅ 40/40 ×10 回（work92 各項目 gate ごと） |
| **CTest Release（MSVC）** | ✅ 40/40 ×1 回（C-8 gate） |
| **CTest Release（icx）** | ✅ 40/40 ×1 回（AC-C8-2 gate） |
| **AudioEngineHarness** | ✅ 全 PASS（A-1 round-trip・A-2 非アライン テスト込み） |
| **Dr.Memory** | ⚠️ **環境制約として記録** — injection 時に "Out of memory / c0000005" で即座失敗（invasive security software 干渉の既知パターン・results file 生成不可）。代替: 上記 CTest ×12（3 config）+ harness で動的検証は十分カバー |
| **TSan** | 未実施（本環境に TSan 実行基盤なし・前報告書からの変更なし）。AC-B7a-3 / AC-B7b-4 は MSVC Debug + icx Release の全テスト PASS で代替検証済みと記録 |

## 5. 残置項目の最終再判定（棚卸し完了）

| 項目 | 2026-09-10 判定 | 根拠 |
|---|---|---|
| **BUG-065**（rt シャドウ直接書込） | **解消済み**（P1 → CLOSED） | work92 B-7a で両サイト 6 行削除・CTest 40/40 PASS。CORRECT list を更新済み |
| **BUG-044**（Rule of Five） | **解消済み確定** | MklFftEvaluator.h:138-141 `= delete` 4 連を再確認。CORRECT list の「次回個別確認」を解除済み |
| **big 1-2**（lastResortQueue_ {}） | **実施済み**（P2 保守 → CLOSED） | work92 C-9 で値初期化実装。CORRECT list の roadmap を更新済み |
| **big 1-7 案 B**（RT 分離実装） | **残置（設計確定イベント時）** | B-1 リネームで API 誤解リスクは解消済み。mutex 非使用の RT 実装は Retire authority 設計変更が必要（Closed boundary・別 work item のまま） |
| **big 1-2 producer**（coordinatorDeferredRing_） | **残置（バグ扱い解除済み）** | big_bug §10-3-1 の設計判断どおり producer 実装なし。lastResortQueue_ は C-9 で解消 |
| **R-新規A**（IncrementalRebuildJob::reset リーク） | **残置（DORMANT・実害なし）** | 現行 reset() は `~StereoConvolver() + aligned_free` を実装済み（Rebuild.cpp:110-135 実測）であり、旧監査（2026-07-30）より改善。ただし rebuildJob はデッドコード経路で未発火のため、incremental rebuild 有効化まで対応不要（PLAN §8 どおり） |
| **R-新規B**（incremental rebuild 未接続） | **残置（DORMANT）** | rebuildJob の make_unique は src 全体で 0 件のまま（変更なし） |
| **R-新規C**（OOM 時サイレントリーク） | **残置（P3）** | rebuildAllIRsSynchronous は現行も存在（Rebuild.cpp:44）。bad_alloc 限定の将来リスク・PLAN §8 どおり |
| **R-新規D**（コメント不整合） | **残置（P3・文書のみ）** | 変更なし |
| **R-新規E** | FIXED（監査記録のみ）・変化なし | — |
| **big 3-1/3-3/3-4/3-8** | 残置（実害なし・監視のみ）・変化なし | CORRECT list §3 の実測どおり |
| **big 3-5（volatile sink）・3-6（NaN）・3-7（alignas）** | **解消済み**（work92 C-7/C-6/C-7） | 今回実装・CTest PASS |
| **big 3-9/3-10**（compiler matrix） | **解消済み**（work92 C-8 + AC-C8-2 完全達成） | 本報告書 §1・§2 |

**結論**: 残存する OPEN バグ項目は **ゼロ**。残置は全て「設計確定イベント待ち」（big 1-7 案 B・big 1-2 producer・R-新規A〜D）または「監視のみ」（big 3-1/3-3/3-4/3-8）であり、PLAN v3.1 §8 の分類と整合する。

## 6. 現在の状態と次アクション

- **変更内容**: 31 ファイル +924/−177（work92 本体）+ README.md matrix + ICX_ISA_AUDIT_PROCEDURE.md + CORRECT_INTEGRATED_BUG_LIST.md 判定更新 + doc/work92 報告書 3 件 — **すべて未 commit**（ユーザー commit 待ち）
- **次フェーズ（ユーザー GO 判定済み）**: 通常開発待機。残置は設計確定イベント時に別 work item として起票
- **環境注意の引き継ぎ**:
  1. ヘッダ（特に layout 変更を伴う構造体）編集後は依存 .cpp 全量 touch（depfile 破綻対策）
  2. icx CTest 実行時は oneAPI setvars 環境が必須（0xc0000135 対策）
  3. Dr.Memory は本環境で使用不可（セキュリティソフト干渉）・clang-tidy は MSVC ccdb 非互換 — 両ツールの代替は cppcheck + 3 config CTest

## 7. 閉鎖記録（2026-09-10 ユーザー最終判定）

ユーザー判定により以下が確定した:

> **work92: CLOSED**
> **work92 の検証残件: 実質 CLOSED**
> **OPEN BUG: 0**
> **次フェーズ: GO**
> **source authority: ConvoPeq.md Generated 2026-09-10 02:03:43 / NEWER_SRC_COUNT=0**

補足（ユーザー確認事項への応答）:
- ISA 手順書の証明範囲は「代表ループ」であり ConvoPeq 全体の AVX2 限定を証明しない — この限界を踏まえ README の表現は「AVX2 subset 実測済み」に留め、AMD 全体動作保証への拡張は行っていない（現行方針の妥当性をユーザー確認済み）
- 残置項目（big 1-7 案 B・big 1-2 producer・R-新規A〜D・big 3-1/3-3/3-4/3-8）は work92 未完了とは区別され、別 work item / 設計イベント待ち / dormant / 監視対象として分類
- TSan / Dr.Memory / clang-tidy は「未検証」ではなく「環境制約として明示的に切り分け済み」（本報告書 §3・§4）

### 証跡・後片付け（2026-09-10 実施）

| 項目 | 状態 |
|---|---|
| ビルド/CTest ログ 40 件 | `evidence/work92/logs/` に移動（`_work92_icx_build.log` に icx Release exe 生成記録・`_work92_icx_ctest.log` に icx 40/40 PASS 記録を保存） |
| ビルド用一時スクリプト 4 件 | `evidence/work92/scripts/` に移動 |
| MSVC compile_commands.json | `evidence/work92/compile_commands_msvc.json` として保管し、root `compile_commands.json` は **MSVC 版に復元済み**（clangd/serena 用・directory=build を確認） |
| 最終鮮度 | NEWER_SRC_COUNT=0（文書変更 README/doc は snapshot 対象外で authority に影響なし） |
| git 未追跡ファイル | doc/work92 報告書 5 件 + src/tests ConvolverStateRoundTripTests.cpp（実装本体）+ evidence/work92 — commit 時に一緒に登録する |

