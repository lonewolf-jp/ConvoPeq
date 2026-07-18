# Bug #2 / #5 / #8 改修計画書（実装完了版）

**作成日**: 2026-07-18（v3: 実装完了＋レビュー反映）
**対象バグ**: Bug#2 (Retireキュー枯渇), Bug#5 (デッドコード), Bug#8 (Use-After-Free)
**ベース**: commit `8956ca7`（Bug#1/3/4/6/7 修正済みの最新状態）→ 実装完了

**構成**: 本ドキュメントの前半（#1〜#3）は設計・計画を、後半（付録A）は実装の詳細を記載する。
**実装状況**: 全ての改修は完了し、Debug / Release (icx) ビルド成功、CI チェック全件 PASS。

**レビュアー指針**: ISR Runtime 設計思想「Authority Singularization」を尊重し、Retire Authority は `ISRRetireRouter` 一箇所に集約する。

---

## Appendix: 実装サマリ

### Bug #8 — NoiseShaperLearner Use-After-Free

| 変更 | ファイル | 内容 |
|------|---------|------|
| `shutdownWorkerThread()` 追加 | `.h` / `.cpp` | 停止シーケンスを共通化 (`request_stop + notify_all + join`)。self-join 防止 (`jassert + terminate`)、`noexcept`。 |
| デストラクタ修正 | `.cpp` | 従来の手動 join ループを `shutdownWorkerThread()` に置換。`Idle` 遷移なし。 |
| `stopLearning()` 修正 | `.cpp` | 従来の非対称な停止 (`request_stop` + `stopEvaluationWorkers`) を `shutdownWorkerThread()` に統一。 |
| `startLearning()` 修正 | `.cpp` | 従来の `stopLearning()` 後の二重 `workerThread.join()` を削除。 |

### Bug #5 — DeferredDeletionQueue デッドコード

| 変更 | ファイル | 内容 |
|------|---------|------|
| `reclaim()` else 節 | `DeferredDeletionQueue.h` | 到達不能コード (`if (canDelete)`, `scanPos++`, `++scanned`) を削除し、単純な `break;` に置換。FIFO 理由のコメントを追加。 |

### Bug #2 — Retireキュー枯渇・監視強化

| 変更 | ファイル | 内容 |
|------|---------|------|
| Timer 診断ログ | `AudioEngine.Timer.cpp` | `overflowCount` / `deltaOverflow` / `queueUsage` / `epochGap` の診断ログを追加。 |
| Coordinator 委譲 | `ISRRuntimePublicationCoordinator.cpp` | `enqueueRetire()` → `router.enqueueWithRetry()` に委譲（リトライロジックを Router に集約）。 |

### Timer.cpp static → インスタンスメンバ

| 変数 | 変更前 | 変更後 | 分類 |
|------|--------|--------|------|
| `lastOverflow` | `static uint64_t` | `lastOverflowCount_` | インスタンス状態 |
| `lastZeroAlloc` | `static uint32_t` | `lastZeroAllocCount_` | インスタンス状態 |
| `s_prevPageFaults` | `static uint64_t` | `pageFaultPrev_` | インスタンス状態 |
| `s_pfEwmaAvg` | `static double` | `pfEwmaAvg_` | インスタンス状態 |
| `s_pfSampleCount` | `static int` | `pfSampleCount_` | インスタンス状態 |
| `lastMemLogUs` | `static uint64_t` | 現状維持 | レートリミッタ |
| `lastCbSummaryUs` | `static uint64_t` | 現状維持 | レートリミッタ |
| `s_cachedMemInfo` | `static` struct | 現状維持 | プロセスキャッシュ |

### CMakeLists.txt

- `DeferredDeletionQueueReclaimTests` ターゲット追加（`add_executable` / `add_test` / `target_compile_features` / `target_include_directories`）

### CI チェック

- `check-list-compliance.ps1`: Failures 0, Warnings 5（手動レビュー項目のみ）
- `check-src-atomic-dotcall.ps1`: PASS（テストファイルの18件違反を修正）
- `check-audioengine-lint.ps1`: PASS
- `check-src-size-mul-cast.ps1`: PASS
- `check-work21-epochdomain-gates.ps1`: ALL PASSED
- `check-authority-boundary.ps1`: PASS
