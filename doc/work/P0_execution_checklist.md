# P0 実装実行チェックリスト（R18 → R16 → R25）

- 作成日: 2026-05-21
- ベース: `doc/work/R11-R25_Closed判定監査表_2026-05-21.md` の「差分監査追補」
- 対象範囲: **P0のみ**（merge blocker 直結）
  - R18 CI Verification Pipeline
  - R16 HB Failure Spec + Reorder Simulation
  - R25 DebugRuntime CI限定化

---

## 実行方針（固定）

- 1PR = 1R を原則とする（`R18` → `R16` → `R25`）。
- 各PRで以下を必須実施:
  1. 実装
  2. ローカル検証（script/build）
  3. 証跡更新（監査表または本書）
- 完了判定は `ISR_Completeness_Risk_Backlog.md` の Closed最小検証項目に従う。

---

## 共通事前チェック（着手前）

- [x] `Strict Atomic Dot-Call Scan` が Pass
- [x] `check-list-compliance.ps1` が Failures=0
- [x] Release build が成功
- [x] 既存 `evidence/` の主要 artifact を確認（上書き/欠落の挙動把握）

---

## PR-1: R18 CI Verification Pipeline 固定

### PR-1 目的

V2〜V10 の **ステージ名と実体**を一致させ、artifact契約違反を確実に merge blocker 化する。

### PR-1 実装チェックリスト

- [x] `.github/workflows/isr-verification.yml` を更新
  - [x] V2〜V10 が仕様と一致する呼び分けになっている
  - [x] V10 が Ownership Cycle Detection と整合する検証を呼ぶ
  - [x] verify/scan 失敗時にジョブ失敗（非0終了）
- [x] `.github/scripts/isr-verify-v*.ps1` を整理
  - [x] 各ステージの artifact/schema/required key が仕様に一致
  - [x] `isr-verify-common.ps1` の missing/schema mismatch/parse error が fail になる
- [x] CI証跡保存
  - [x] verify結果（ログ/レポート）をアーカイブ対象へ追加
  - [x] 成功時/失敗時のどちらでも参照可能な出力導線を用意

### PR-1 完了判定（DoD）

- [x] artifact missing / schema mismatch / parse error のいずれでも CI fail
- [x] V2〜V10 が仕様名と一致（監査で説明可能）
- [x] 成功時に証跡（レポート/ログ）が保存される

### PR-1 変更候補ファイル

- `.github/workflows/isr-verification.yml`
- `.github/scripts/isr-verify-v2.ps1` ～ `.github/scripts/isr-verify-v10.ps1`
- `.github/scripts/isr-verify-common.ps1`

---

## PR-2: R16 HB Failure Spec + Reorder Simulation 固定

### PR-2 目的

`simulateReorderScenario()` を「実効シナリオ」を持つ検証へ強化し、required HB の有無で pass/fail が分かれる状態にする。

### PR-2 実装チェックリスト

- [x] `src/audioengine/ISRHB.cpp` を更新
  - [x] `ForcedReorder` の失敗条件を実装
  - [x] `EpochLag` の失敗条件を実装
  - [x] `RetireDelay` の失敗条件を実装
  - [x] `ObserveRace` の失敗条件を実装
- [x] `HBVerifierRuntime::runScenarioSuite()` の結果がシナリオ依存で変化
- [x] `hb_violation_report.json` に scenario結果が反映
- [x] CIでシナリオ実行を確認（PR-1の pipeline 上で動くこと）

### PR-2 完了判定（DoD）

- [x] required HB 欠落ケースで fail が再現
- [x] required HB 適用ケースで pass が再現
- [x] 4シナリオの結果がレポートに保存される

### PR-2 変更候補ファイル

- `src/audioengine/ISRHB.h`
- `src/audioengine/ISRHB.cpp`
- （必要に応じ）`.github/scripts/isr-verify-v5.ps1`

---

## PR-3: R25 DebugRuntime CI限定化 固定

### PR-3 目的

Release で proof負荷を最小化し、Debug=partial / CI=full を実コードとCIで一致させる。

### PR-3 実装チェックリスト

- [x] `src/audioengine/ISRDebugRuntime.cpp` を確認/調整
  - [x] Release で proof出力を抑制
  - [x] Debug は partial
  - [x] CI は full
- [x] `src/audioengine/ISREvidenceExporter.cpp` を確認/調整
  - [x] Release で不要 artifact 常時生成を抑制
  - [x] build mode 別の proof level が運用定義と一致
- [x] 呼び出し境界整理
  - [x] `AudioEngine.Commit.cpp` の evidence emit が build profile と整合
- [x] RuntimeReductionGate 相当の CIゲートを追加
  - [x] 新規 runtime object 追加時の審査フック（lint/scan）を導入

### PR-3 完了判定（DoD）

- [x] Release=proof off（または最小）
- [x] Debug=proof partial
- [x] CI=proof full
- [x] 上記差分が CI 上で自動検証される

### PR-3 変更候補ファイル

- `src/audioengine/ISRDebugRuntime.h`
- `src/audioengine/ISRDebugRuntime.cpp`
- `src/audioengine/ISREvidenceExporter.h`
- `src/audioengine/ISREvidenceExporter.cpp`
- `src/audioengine/AudioEngine.Commit.cpp`
- `.github/workflows/isr-verification.yml`
- （新規）`.github/scripts/isr-verify-runtime-reduction-gate.ps1` など

---

## PRごとの検証テンプレート（記録欄）

### PR-1（R18）

- 実行日:
- 実行者: GitHub Copilot
- 主変更: V2〜V10スクリプト再マッピング、V10 ownership script、RuntimeReductionGate導入、workflow artifact upload整備
- 検証結果:
  - Atomic Dot-Call: Pass
  - list-compliance: Failures=0（Warnings=9）
  - ISR verify: V2〜V10 + RuntimeReductionGate Pass
- 証跡パス: `.github/workflows/isr-verification.yml`, `.github/scripts/isr-verify-v2.ps1`〜`v10-ownership-cycle.ps1`, `isr-verify-runtime-reduction-gate.ps1`
- 判定: [x] Pass  [ ] Fail

### PR-2（R16）

- 実行日:
- 実行者: GitHub Copilot
- 主変更: `HBVerifierRuntime` のシナリオ別判定実装（ForcedReorder/EpochLag/RetireDelay/ObserveRace）と違反レポート強化
- 検証結果:
  - HB scenario suite: 実装反映済み
  - ISR verify: V5 Pass
- 証跡パス: `src/audioengine/ISRHB.h`, `src/audioengine/ISRHB.cpp`
- 判定: [x] Pass  [ ] Fail

### PR-3（R25）

- 実行日:
- 実行者: GitHub Copilot
- 主変更: `ISREvidenceExporter` を Release/Debug/CI で出力分離、RuntimeReductionGate CI追加
- 検証結果:
  - Release/Debug/CI profile: Release=minimal / Debug=partial / CI=full 反映
  - ISR verify: V2〜V10 + RuntimeReductionGate Pass
- 証跡パス: `src/audioengine/ISREvidenceExporter.cpp`, `.github/scripts/isr-verify-runtime-reduction-gate.ps1`
- 判定: [x] Pass  [ ] Fail

---

## P0完了ゲート（最終）

- [x] R18 DoD 全達成
- [x] R16 DoD 全達成
- [x] R25 DoD 全達成
- [x] `doc/work/R11-R25_Closed判定監査表_2026-05-21.md` へ P0再判定を反映
- [x] Backlog（`ISR_Completeness_Risk_Backlog.md`）の該当R判定更新に必要な証跡が揃っている

---

## 注意事項

- 本書は **P0実行計画**であり、正本仕様の置換ではない。
- 仕様解釈に衝突がある場合は `plan5.md` REV3.2 と `ISR_Completeness_Risk_Backlog.md` を優先する。
