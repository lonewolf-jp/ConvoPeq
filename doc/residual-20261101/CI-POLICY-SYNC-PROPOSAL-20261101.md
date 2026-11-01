# CI ポリシー同期の提案（2026-11-02）

## 1. 何が起きているか

PR #103 の `isr-verify` が 1 件の violation で失敗する。

```
[WARN] [CI-RESIDENCY-001] src/audioengine/AudioEngine.h:5290 residency container declaration is not allowlisted
ISR v7.3 residency/telemetry checks failed. violations=1
```

- 実行スクリプト: `.github/scripts/isr-verify-v73-residency-telemetry.ps1`
- 判定: `residencyContainerAllowlist` に一致しない「residency コンテナ宣言」を違反として扱う。
- 該当宣言: `std::vector<DSPCore*> deferredDestroy_;`（B-2 で型付き化したもの）

## 1.1 CI の現況（PR #103 / 再実行後）

| ジョブ | 結果 |
|---|---|
| CodeQL（actions / c-cpp / csharp / javascript-typescript） | PASS |
| RelWithDebInfo ASan (MSVC) | PASS（56m35s） |
| Debug TSan (Clang, best-effort) | PASS |
| Windows MSVC / Release | PASS（32m14s） |
| compliance-list-md | **PASS**（atomic ドット呼び出し修正後） |
| lint-audioengine | PASS |
| authority-static-checks / authority-runtime-tests | PASS |
| audit-report-verification | PASS |
| **isr-verify** | **FAIL — CI-RESIDENCY-001 の 1 件のみ** |

`isr-verify` の失敗はこの 1 件だけで、他の検査は PASS している。

## 2. 原因

allowlist のエントリが**旧型を前提**にしている。

```json
{
  "pathRegex": "^src/audioengine/AudioEngine\\.h$",
  "lineRegex": "std::vector<void\\*>\\s+deferredDestroy_\\s*;",
  "owner": "audio-runtime",
  "issue": "BRIDGE-ISR-AI-GOV-001",
  "rationale": "AudioEngine deferred destroy residency container (F1/-b deferred destroy when a pin is retained)",
  "expiry": "2027-12-31"
}
```

（`.github/isr-ai-governance-policy.json:252-259`）

B-2 では Owner 指示「DSPCore と証明できない `void*` を `static_cast` しない」に従い、
`std::vector<void*>` → `std::vector<DSPCore*>` に変更した。結果として `lineRegex` が一致しなくなった。

## 3. 提案（1 行の差し替え）

```diff
-            "lineRegex": "std::vector<void\\*>\\s+deferredDestroy_\\s*;",
+            "lineRegex": "std::vector<DSPCore\\*>\\s+deferredDestroy_\\s*;",
```

他のフィールド（owner / issue / rationale / expiry）は**変更しない**。

## 4. 影響範囲

| 対象 | 影響 |
|---|---|
| 実行時挙動 | **なし**（JSON は検証スクリプトの入力のみ） |
| ワークフロー定義（`.github/workflows/*.yml`） | **変更なし** |
| 他の allowlist エントリ | 変更なし |
| isr-verify の他の検査 | 変更なし |

## 5. 代替案（採らない理由）

| 代替 | 評価 |
|---|---|
| B-2 の型付き化を撤回（`std::vector<void*>` に戻す） | Owner 指示（`void*` の `static_cast` 禁止）に反する。**不採用** |
| 宣言を別の型エイリアスで書いて検出を回避 | 検査をすり抜ける目的の変更であり、統制上不適切。**不採用** |
| ポリシーを変更せず `isr-verify` を赤のまま放置 | 「検証未完了の変更を main へ統合しない」に反する。**不採用** |

## 6. 立場

`.github/isr-ai-governance-policy.json` は CI ゲートの入力であり、
Owner 指示「CI 定義変更は今回の承認に含めない」に該当すると判断した。
したがって**本提案は未適用**で、承認後に適用する。

## 7. 承認後にやること

1. 上記 1 行を適用（コミット）。
2. `powershell -File .github/scripts/isr-verify-v73-residency-telemetry.ps1` をローカル実行し violations=0 を確認。
3. push して CI の再実行を確認。
4. CI 全緑を確認して main へ統合。

## 8. 事前検証（一時適用 → 即時 revert）

承認判断の材料として、**コミットせずに** 1 行だけ適用して検証し、直後に元へ戻した。

```
（一時適用）
[INFO] wrote evidence/isr_v73_residency_telemetry_report.json
[PASS] ISR v7.3 residency/telemetry checks passed
probe exit = 0
（revert 後、git status に差分なし）
```

- 効果: `CI-RESIDENCY-001` の violation が 0 になる。
- 副作用: 他の検査結果に変化なし（スクリプト全体が PASS）。
- リポジトリの状態: `git checkout -- .github/isr-ai-governance-policy.json` で完全に復元済み
  （`git status --porcelain` の該当ファイル差分 0）。
