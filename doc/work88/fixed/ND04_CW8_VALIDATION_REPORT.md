# ND-04 — CW-8 Targeted Validation / Build / CTest（Work Report）

```text
ND-04 — CW-8 PublishedWorldObservation Targeted Validation

Production source: 1 file changed（src/audioengine/RuntimeWorldAuthority.h・+63/−0）
Test source: 1 file changed（src/tests/ISRSemanticValidationTests.cpp・+162/−0）
CMake: 0
Build: RUN（Debug + Release・configure 再実行含む）
CTest: RUN（Debug 40/40・Release 40/40）
stress: 0（指示どおり保留 — CW-8 closure 後に必要性判断）

baseline:
ConvoPeq.md Generated 2026-09-01 18:46:22（テスト実行時のコード基準）
→ 最終 FRESH 2026-09-01 19:31:33（dependent-type 修正を反映した再生成・NEWER_SRC_COUNT=0）
```

## 総合判定

> ## **ND-04 = PASS / CW-8 CLOSED / Case A CLOSED**
>
> 受入条件 17 項目すべて成立（§3 チェックリスト）。

---

## 1. Snapshot 再生成（指示 1）

- 実装後の初回再生成: `Generated: 2026-09-01 18:46:22`・`--check` = **FRESH / NEWER_SRC_COUNT=0**。対象 2 ファイルの収録確認（`PublishedWorldObservation` 19 hits・`testCW8_PublishedWorldObservation` 2 hits）。
- テスト結果の evidence コード基準は 18:46:22（Debug targeted build は 18:46:22 snapshot 源源のソースで実施）。

## 2. ND-03 diff 再監査（指示 2）+ 実装修正 1 件

14 項目チェックリストを全て再確認（定義 1 箇所 / private ctor / friend 1 型 / private members / const pointer accessor / default 構築不可 / aggregate 不可 / 独立構築不可 / identity 導出はコード 1 箇所のみ（:245・もう 1 件はコメント）/ observe() 7 件中 factory 1 回 / null deref 保護あり / 新 atomic 0 / authority 追加 0 / 既存 API 変更 0 — 削除行 0 で構造保証）。

**member template 必要性の再確認 → 必要（維持）。** `struct RuntimeState;` は本ヘッダ :21 の前方宣言のみで、AudioEngine.h include は循環回避のため既存制約として不可能。`&world->publication` の member access には完全型が必須。

### ビルドで発見・修正した実装バグ（1 件）

初回 targeted build で **C2027（RuntimeState 不完全型）** が全 Authority include TU で発生:

- 原因: factory 本体で `const RuntimeState* world = ...` と**非依存型**で宣言したため、member access `world->publication` が **テンプレート定義時点（phase 1）で意味検査**された（2-phase lookup）。遅延 instantiation の前提が誤りだった。
- 修正: world の型を**依存型 `const StateT*`** に変更し、member access の意味検査を instantiation まで先送り。修正は RuntimeWorldAuthority.h 内の factory 本体 3 行のみ（呼び出し契約・他コードへの影響なし）。修正後コメントに根拠を実測込みで記録。
- この修正を含む最終コード基準 = `Generated: 2026-09-01 19:31:33`（FRESH）。

## 3. Build（指示 3・4）

| 工程 | 結果 |
|---|---|
| configure（Ninja Multi-Config・vcvarsall x64 + oneAPI include） | **CONFIGURE_EXIT=0**（impl-*.ninja 欠損からの再構築） |
| Debug targeted build（ISRSemanticValidationTests） | **DBG_TARGET_EXIT=0** |
| Release targeted build | **REL_TARGET_EXIT=0** |
| Debug full build（全 495 target） | **DBG_FULL_EXIT=0** |
| Release full build（全 494 target） | **REL_FULL_EXIT=0** |

Logs: evidence/nd04_build_targeted.log・nd04_build_full.log

## 4. Targeted test（指示 3・4）

`ISRSemanticValidationTests.exe` 単独実行（throw 型・全 assertion 成立が exit 0 の意味）:

| Config | 結果 | 含まれる検証 |
|---|---|---|
| **Debug** | **DBG_RUN_EXIT=0** | T-CW8-1/2/3 統合（null 観測 → publish {1,1,1} → pair address identity → publish {2,2,2} → pair 進行・旧 observation 凍結）PASS |
| **Release** | **REL_RUN_EXIT=0** | T-CW8-4 black-box（bake == observed publication == observed identity）PASS |

**T-CW8-7 の test TU での成立**: static_assert は compile-time であり、**Debug/Release 両 targeted ビルドが exit 0 で完了した事実そのものが**、テスト TU で以下が成立したことの証明:
- `!std::is_aggregate_v<PublishedWorldObservation>` ✓
- `!std::is_default_constructible_v<PublishedWorldObservation>` ✓
- `!std::is_constructible_v<PublishedWorldObservation, const RuntimeState*, const PublicationSemantic*>` ✓（非 friend TU からの独立 pair 構築は compile error になる）
- `is_trivially_copyable_v` / `is_nothrow_copy_constructible_v` ✓

shutdown/null semantics: 未 publish null → no-op（nullptr 返却）→ shutdown clear 後 `{nullptr, nullptr}` を同一テスト内で検証 PASS。

Logs: evidence/nd04_run_targeted.log

## 5. 既存 3 suite 回帰（指示 5）— CTest target 名で照合・実行

ND-02/03 で参照した suite の現行 CTest target 名を実測（`ctest -N`）:

| ND 文書上の名称 | 実行 target（CTest #） | Debug 結果 |
|---|---|---|
| ISRSemanticValidationTests | **ISRSemanticValidationRejects（#21）** | **Passed** |
| ObservePathSingleSourceTests | **ObservePathSingleSource（#29）** | **Passed**（0.07s） |
| RuntimeWorldAuthorityProjectionTests | **RuntimeWorldAuthorityProjectionContract（#33）** | **Passed**（0.35s） |

`ctest -R "ISRSemanticValidationRejects|ObservePathSingleSource|RuntimeWorldAuthorityProjectionContract"` → **3/3 passed**。CW-8 追加による既存契約（read-source singularization・projection 契約・ISR semantic 検証）への regression = 0。

## 6. 全体 CTest（指示 6）— 実行時の現行登録数を基準

| Config | 結果 |
|---|---|
| **Debug** | **100% tests passed, 40/40**（DBG_CTEST_EXIT=0） |
| **Release** | **100% tests passed, 40/40**（REL_CTEST_EXIT=0） |

- 登録数は実行時の現行 CMake/CTest 出力を基準（本日実測: 40 tests — 過去の「40/40」期待値の固定ではなく現行出力で確認）。
- AudioEngineHarness（#40）Debug 17.71s PASS 等、全 suite 正常。
- Logs: evidence/nd04_ctest_debug.log・nd04_ctest_release.log

## 7. forbidden diff 再確認（指示 7・CTest 後実測）

| 項目 | 実測 |
|---|---|
| Production 変更 | `RuntimeWorldAuthority.h`（+63/−0）**のみ** — その他 production source diff 0（AudioEngine.h / RuntimeStore.h / Coordinator .h/.cpp / RuntimePublishExecutor.h / Retire.cpp / ISRShutdown.h / ISRSealedObject.h を個別実測で diff=0 確認） |
| Test 変更 | `ISRSemanticValidationTests.cpp`（+162/−0）**のみ** |
| CMake | **0** |
| new atomic | **0**（diff 内 atomic 追加 0 件） |
| ownership authority | **0**（WriteAccess/LifetimeState/OwnerChannel 追加 0 件） |
| retire/EBR | **0** |
| RuntimeStore | **0** |
| Coordinator / PublishExecutor | **0** |
| stress | **0**（指示どおり保留） |

## 8. ND-04 受入条件チェックリスト（全 17 項目）

```text
[x] ConvoPeq.md = FRESH（19:31:33・最終コード基準）
[x] NEWER_SRC_COUNT = 0
[x] Debug build PASS（targeted + full）
[x] Release build PASS（targeted + full）
[x] T-CW8-1/2/3 PASS（Debug/Release 両方）
[x] T-CW8-4 PASS（Debug/Release 両方）
[x] T-CW8-6 PASS（型 static_assert・両 config compile）
[x] T-CW8-7 PASS（型 static_assert・両 config compile）
[x] null/shutdown semantics PASS
[x] 既存 3 suite PASS（ISRSemanticValidationRejects / ObservePathSingleSource / RuntimeWorldAuthorityProjectionContract）
[x] Debug 全体 CTest PASS（40/40）
[x] Release 全体 CTest PASS（40/40）
[x] forbidden production diff = 0
[x] CMake diff = 0
[x] new atomic = 0
[x] ownership authority change = 0
[x] retire/EBR change = 0 / RuntimeStore change = 0
[x] stress = 0
```

> ## **ND-04 = PASS / CW-8 CLOSED / Case A CLOSED**

## 9. 遷移

```text
ND-01 GO → ND-02 GO → ND-03 実装 → ND-04 PASS   ← 本報告（CW-8 closure）
   ↓
ND-06 — CR-α BuildError retry contract audit（read-only・次工程）
   ↓
CR-α 実装判断
```

ND-06 では指示どおり `BuildError → classification → RetryDecision → backoff → retry scheduling → RT/NonRT boundary` の到達可能性と ownership/liveness contract を read-only で再監査する（BE-8 契約: BuildError/retry 判定は rebuildThreadLoop 限定を規範として扱う）。
