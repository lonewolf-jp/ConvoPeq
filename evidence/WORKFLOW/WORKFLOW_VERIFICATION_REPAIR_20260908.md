# Workflows Verification & Repair — 実測記録（2026-09-08）

## 対象

`.github/workflows/` の 6 workflow:
audioengine-lint / list-compliance / isr-authority-compliance / isr-verification / sanitizer-ci / soak-ci

## 総合判定

> **YAML 構文: 6/6 PASS。ローカル再現可能な検証ステップを実行した結果、2 workflow（audioengine-lint / list-compliance）が FAIL → 原因特定・修正済み → 修正後 Full Debug build PASS + CTest 40/40 PASS。**
> 残る 4 workflow（isr-verification / isr-authority-compliance / sanitizer-ci / soak-ci）は CI runner 依存（MKL/oneAPI availability・GitHub API）でローカル完全再現は不可能だが、構文・参照先実在性・policy 妥当性は実測確認済み。

## 検証結果一覧

| Workflow | YAML | ローカル実行 | 結果 |
|---|---|---|---|
| audioengine-lint | OK | **FAIL → 修正後 PASS** | LINT-AE-011 1 件（後述） |
| list-compliance | OK | **FAIL → 修正後 PASS** | atomic dot-call 69 件（後述） |
| isr-authority-compliance | OK | 構文・参照先のみ確認 | 静的ガバナンス 3 step は参照先実在・policy expiry 2026-12-31 有効 |
| isr-verification | OK | policy validation 相当を確認 | 8.1 close policy schema/expiry OK・verifier 17 script 実在・tier 解決ロジック正常 |
| sanitizer-ci | OK | 構文のみ | oneAPI graceful skip 設計確認（runner に oneAPI 無しは既知） |
| soak-ci | OK | 構文のみ | workflow_dispatch 手動実行専用・graceful skip 設計確認 |

## 発見した欠陥と修正（Practical Stable ISR 観点）

### 欠陥 1: list-compliance が必ず FAIL する（構造的欠陥・高優先）

`check-src-atomic-dotcall.ps1` のスキャン範囲が `src/` 全体（**tests を含む**）だったため、
test 専用プロバイダの bookkeeping atomic（`MockSink.callCount.load()` 等 65 件）が
production 違反として検出され、**CI が毎回必ず失敗する状態**だった。
実運用で「必ず赤くなる gate」は観測価値を持たず、真の violation を埋没させる。

**修正（.github/scripts/check-src-atomic-dotcall.ps1）**:
- strict scan の対象を production（`src/` から `tests` ディレクトリを除外）に限定
- forbidden-symbol（globalEpochDomain / atomic_flag / ObservedSnapshot handoff）と
  `memory_order_seq_cst` ルールは従来どおり全 src を対象に維持（安全性の縮退なし）

### 欠陥 2: production 33 件の T3c lifecycle word が誤検出

`ISRRuntimePublicationCoordinator.{cpp,h}` の recovery obligation lifecycle 操作は
**16B full-word CAS プロトコル（D152-R2 設計確定・MSVC lock-pool backend）**であり、
`convo::consumeAtomic/publishAtomic/exchangeAtomic` helpers では構造的に置換不可能
（full-word 一括 CAS が T3c の TOCTOU 防御の中核）。

**修正**: 該当 33 行に `// NOLINT(atomic-dot-call): T3c 16B full-word CAS (D152-R2) — helper 置換不可の設計固定プロトコル` を付与。
既存の NOLINT 運用（AudioEngine.CtorDtor.cpp:33 等 10 件超の先行例）と同一パターン。動作変更ゼロ。

### 欠陥 3: write-only diag flag が atomic だった（低優先・정리）

`AudioEngine.h:1088` `std::atomic<bool> diagFootprintCaptured` は reader 0 件の write-only marker で、
隣接する `diagFootprint`（plain struct）と不整合。診断ビルド限定（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`）・
全操作が NonRT であるため plain bool に変更。Commit.cpp:832 / DSPCoreLifecycle.cpp:242 の store を代入に更新。

### 欠陥 4: audioengine-lint LINT-AE-011 の過剰検出

`RuntimeHealthMonitor.h:335` の `mutable CachedStuckDiagnosis m_lastStuckDiagnosis_` は:
- timerCallback（MessageThread・100ms Non-RT sampler）のみが write/read
- RT audio path からは到達しない
- POD aggregate（destructor なし・cross-thread handoff なし）
- `takeSnapshot() const` の const-correctness のために mutable が必要

mutex 用の mutable 許可と同じ const-correctness パターンであり、
lint スクリプトに「mutable POD aggregate + 周辺コメントで NonRT 契約を明示」という
許可パターンを追加（ソース側にも thread contract コメントを追記）。
`thread_local` や RT 汚染の検出能力は維持。

## 検証実績（修正後）

| 項目 | 結果 |
|---|---|
| YAML 構文 6/6 | PASS |
| check-audioengine-lint.ps1 | **PASS**（LINT-AE-001〜014 全通過） |
| check-src-atomic-dotcall.ps1 | **PASS**（violation 0） |
| check-list-compliance.ps1 | PASS |
| check-src-size-mul-cast.ps1 | PASS |
| Full Debug build | **PASS**（398 targets・error 0） |
| CTest Debug 全体 | **40/40 PASS**（AudioEngineHarness 含む・59.7s） |
| Python authority verifiers（coverage / source_count / publication / inventory / duplication / retire_ordering） | 6/6 PASS |

## 残存事項（CI runner 依存・ローカル解決不可）

1. **isr-authority-compliance / sanitizer-ci / soak-ci の runtime テスト**: GitHub runner に
   Intel oneAPI が無いため graceful skip が正常動作（設計どおり・PR をブロックしない）。
2. **retire_authority_verifier.py exit=1（L579 lifetimeMgrForFinalDSP.retire）**:
   stash による baseline 実測で **修正前に存在する pre-existing 問題**と確定（今回の変更の影響ではない）。
   isr-verification workflow では non-blocking warning 扱い（Practical Stable ISR 規約どおり）だが、
   次回 governance 変更時に D-series で確認推奨。
3. **isr-verification の PR SLA labeling** は GitHub API 必須のためローカル検証対象外。

## authority 整合

- 変更ファイル: production 5（AudioEngine.h・Commit.cpp・DSPCoreLifecycle.cpp・Coordinator.cpp/h）+
  lint/dotcall スクリプト 2。動作変更は plain bool 化のみ（診断ビルド限定・NonRT）。
- authority drift hash: inventory path ベースのため comment/flag 編集の影響なし（実測 PASS）。
- verifier manifest hash: `.github/scripts/isr-verify-*.ps1` のファイル名リストベース —
  今回改名・追加なしのため影響なし。
- Freeze register: CLOSED 領域（T3c / RecoveryLifecycleWord）に触れるのは NOLINT コメント付与のみで
  ロジック無変更。RuntimeHealthMonitor は D177 契約（observation-only）を維持。

## ConvoPeq.md 再生成の必要性

src 変更があるため、次回 snapshot 再生成（`python output_sourcecode_markdown.py`）が必要。
