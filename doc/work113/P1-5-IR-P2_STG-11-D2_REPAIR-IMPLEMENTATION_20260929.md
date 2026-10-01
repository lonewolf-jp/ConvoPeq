# STG-11-D2 Repair Implementation（2026-09-29）

> **Verdict: STG-11-D2 Repair Implementation = PASS**
> **COMMIT = NOT AUTHORIZED / PUSH = NOT AUTHORIZED（未実施）**
> 本書は Owner GO（R-A 正式採用）に基づく最小実装と全 validation の実施記録である。
> D1 の変更は revert していない。D3 / D4 には進んでいない。

---

## 1. Authority

Owner 指定: `ConvoPeq(20260929-105350).md == ConvoPeq.md`（統一 authority）。

| 項目 | 実装開始時 | 全 validation PASS 後の再生成 |
| --- | --- | --- |
| HEAD | `f95f524c` | `f95f524c`（不変） |
| SHA-256 | `CDAFA18B...5C38B1` | **`9565A3D5...FF4F8F5BE`** |
| size | 5,690,145 B | **5,710,250 B** |
| Generated | 2026-09-29 19:35:52 | **2026-09-29 21:16:18** |
| NEWER_SRC_COUNT | 0 | **0** |
| STATUS | FRESH | **FRESH**（`--check` exit 0） |

完全 SHA（実測）: `9565A3D546BF256F75C1A2B97FB1E9B99CFF8D6DED11A3673642EADFF4F8F5BE`

---

## 2. Production changes

### 2.1 変更ファイル（R-A のみ）

**`src/core/SnapshotCoordinator.cpp` — `quarantineRetireSink()` 1 関数のみ**
（+24/-10 相当。header・Router・Epoch・RT・Coordinator・authority・EQ 系は不変）。

```cpp
void SnapshotCoordinator::quarantineRetireSink(...) noexcept
{
    if (m_retireSink == nullptr || ptr == nullptr || deleter == nullptr)
        return;
    // Stage 1: Q。格納成功で ownership 移転成立。
    if (m_retireSink->quarantineRetire(...))
        return;  // Q owns ptr
    // Stage 2: E。Q-full 時のみ到達。
    if (m_retireSink->emergencyQuarantine(..., 0, 0))
        return;  // E owns ptr
    // Stage 3: T。growable のため常に受領（ownership は必ず移転）。
    (void) m_retireSink->terminalReclaim(...);
    // T owns ptr
}
```

- 旧 `assert(false && ...)` は除去した（到達不能になったため。
  Release の silent loss を assert で検出する設計への依存を排除 —
  Owner §4 の禁止事項に準拠）。
- `#include <cassert>` は残存するが実行 assert は 0 件（static 検証 W6）。
- E/T は既存 public API（`ISRRetireRouter.h:312,322`）をそのまま使用。
  `ISRRetireRouter.*` の意味変更なし、新規 authority なし。
- 全段は同一 engine domain / 同一 router（epoch provenance 維持）。
- 同一 ptr の複数 container 格納なし（各段は transfer 成立でのみ return）。

### 2.2 変更禁止の遵守

| 禁止対象 | 結果 |
| --- | --- |
| `SnapshotCoordinator.h` | diff 0 |
| `ISRRetireRouter.*` | diff 0 |
| `EpochDomain.*`（STG-10-D1 維持） | diff 0 |
| `RCUReader.*` | diff 0 |
| `EQProcessor.*`（D1 維持） | D1 のみ。D2 による追加 0 |
| AudioEngine の RT path | diff 0 |
| Coordinator / authority boundary | diff 0 |
| Q-full 時の silent drop | 解消（T が常時受領） |
| assert 消滅依存の設計 | 解消（value ベースの終端解決） |
| caller 側への escalation 分散 | なし（sink 内に完結） |
| 新規 retire authority | なし |
| coordinator 委譲後の同一 pointer 再 enqueue（D1 型） | なし（sink は再 enqueue しない） |
| RT への retire/reclaim/delete 追加 | なし |
| raw `std::atomic` API 追加 | 0 件 |

---

## 3. Test changes

### 3.1 新規 TU

`src/tests/STG11D2SnapshotRetireTests.cpp`（新規）。
TD2-0 的補助なし。TD2-1 / TD2-2 / TD2-3 の 3 件（決定論的・単一スレッド）:

| ID | 内容 | oracle（正確） |
| --- | --- | --- |
| TD2-1 | D(4096)+Q(512) pre-fill → coordinator 駆動 3 回 | **E == 3** |
| TD2-2 | D+Q+E(512) pre-fill → coordinator 駆動 3 回 | **T == 3** |
| TD2-3 | `finalizeShutdown(false)`＋既存 drain で完全解決 | 全 counts 0・alive 0 |

- pre-fill は counted local object、coordinator 駆動分は実 `GlobalSnapshot`
 （`SnapshotFactory::create`）。両者を混ぜない。
- fill 中は reader を滞留させて minReader を固定する
  （coordinator retire が内部 `publishEpoch()` するため。固定しないと
  pre-fill が reclaim されて決定論的 fill が崩れることを実測で確認し対処した）。
- fixture は heap 確保（TD1 の教訓。`EpochDomain` 約 213KB）。
- harness 配線: `PublishPipelineIntegrationTests.cpp` に宣言＋呼び出し（+11）。
- standalone CTest target: `STG11D2SnapshotRetireTests`（`add_test(NAME STG11D2SnapshotRetire ...)`）。
  TU 共有し `STG11D2_STANDALONE_MAIN` でのみ `main()` を定義（TD1 と同一方式）。

### 3.2 既存 oracle の改変 = 0。test hook 追加 = 0。

---

## 4. CMake changes

test-infra のみ（全量実読で確認）:

| 要素 | 内容 |
| --- | --- |
| standalone target | `STG11D2SnapshotRetireTests`（TU＋D8_2 と同一の最小リンク集合） |
| add_test | `STG11D2SnapshotRetire` 1 件 |
| IPO-OFF／ASAN | 新 target のみ |
| harness TU 追加 | `STG11D2SnapshotRetireTests.cpp` 1 行 |

production target の optimization policy・既存 target の compiler/linker behavior・
既存 CTest の実行条件の変更は 0 件。

---

## 5. Test results

### 5.1 TD2-1 / TD2-2 / TD2-3

| 項目 | Debug standalone | Release standalone |
| --- | --- | --- |
| TD2-1（E == 3 正確） | PASS | PASS |
| TD2-2（T == 3 正確） | PASS | PASS |
| TD2-3（全 0・alive 0） | PASS | PASS |
| exit | 0 | 0 |

### 5.2 TD2-4（既存非回帰）

| 項目 | Debug | Release |
| --- | --- | --- |
| CTest 全件 | **45/45 PASS**（196.06 s） | **45/45 PASS**（150.21 s） |
| harness 直接実行 | exit 0・全 PASS | exit 0・全 PASS |
| `D8_2_B_2_Tests` T2（D→Q 成功経路） | PASS（CTest の一部） | PASS |
| `StuckReaderFallbackDrainTests` | PASS（同上） | PASS |
| harness STG-1〜9＋TD1＋TD2 | 全 PASS | 全 PASS |

CTest 件数が 44 → 45 になったのは `STG11D2SnapshotRetire` の新規登録による
（既存 44 件は全て PASS のまま）。

### 5.3 Negative control（Owner §6）

R-A の E/T 昇格を一時的に無効化（Q-only legacy）した隔離ビルドで:

```text
TD2-1: E residency=0 expected 3 → FAIL（Release: silent loss を検出）
TD2-2: T residency=0 expected 3 → FAIL
TD2-3: PASS（shutdown path は影響なし）
```

**value oracle のみで旧欠陥を検出できることを実証**した後、production を R-A に
完全復元した（`NEGCTL` 残存 0 件を `rg` で確認。最終 tree に negative-control 用
変更は残っていない）。復元後に Release TD2 全 PASS を再確認した。

---

## 6. TD1 regression（Owner §7）

| 項目 | Debug | Release |
| --- | --- | --- |
| TD1 standalone（TD1-0/1/2/3/4a/4b） | exit 0・全 PASS | exit 0・全 PASS |
| TD1 harness 内 | PASS | PASS |
| TD1 oracle（Q=304／D/Q/E/T=4096/512/512/280／total 5400） | 変更なし | 変更なし |

INV-D1-1〜INV-D1-6 の破壊なし。D1 の double-ownership 修正・member router・
INV-D1-6 は維持されている（D2 diff は EQ 系に触れないことを static W3 で確認）。

---

## 7. ASAN（Owner §8）

**`ASAN = INCONCLUSIVE / environment blocked`**（D1 Gate と同一事象）。

| # | 試行 | 結果 |
| --- | --- | --- |
| 1 | `build-asan` Debug で D2 target をビルド | 成功 |
| 2 | 実行 | `0xC0000139`（pre-main ローダ失敗。D1 と同一署名） |
| 3 | RelWithDebInfo での代替 | 未実施（D1 で同一失敗を確認済みのため） |

代替 evidence（D1 Gate §6 と同一方針）:

1. **正確な counts**: TD2-1（E=3）/ TD2-2（T=3）/ pre-fill 合計一致。
   全オブジェクトが正確に 1 箇所に所有されているため double-free は構造的に起こり得ない。
2. **全 drain 完了**: release 後に全 counts 0・alive 0（TD2-3）。
3. **Negative control**: guard 除去で E==0/T==0 を検出。
4. **複数回実行**: Debug／Release standalone＋harness 両方で exit 0。
5. **destructor 実行**: 全 test が scope exit で破棄を完了し正常終了。

---

## 8. Static validation

| # | 検査 | 結果 |
| --- | --- | --- |
| 1 | production 変更 = D1 維持分＋`SnapshotCoordinator.cpp` のみ | PASS |
| 2 | D2 の実コード差分 = sink 内の Q→E→T 昇格のみ | PASS |
| 3 | 禁止 14 ファイル diff 0（EQ 系は D1 維持分のみで D2 追加なし） | PASS |
| 4 | RT path files diff 0 | PASS |
| 5 | raw `std::atomic` 追加 0 | PASS |
| 6 | sink 内の実行 assert 0（コメント・未使用 include のみ残存） | PASS |
| 7 | NEGCTL 残存 0 | PASS |

---

## 9. Build

| 項目 | 結果 |
| --- | --- |
| Debug build | PASS |
| Release build | PASS |

---

## 10. Remaining Risks

| # | 項目 |
| --- | --- |
| 1 | C1∧C2∧C3 の同時成立は二重 exceptional（4608 entry 滞留）のため実運用の発火頻度は極めて低い。EBR 破綻 scenario では到達しうる（設計上予定されていた escalation の欠落を塞いだ） |
| 2 | E/T 昇格後も engine epoch が前進しなければ entries は滞留する（timeliness のみ。既存 Q と同一条件） |
| 3 | ASAN 実行は D1 と同一の環境 block。§7 の代替 evidence で対応 |
| 4 | N4（submit 系 overflow）は STG-8 との非交差確認が別途必要。本 STG の scope 外 |
| 5 | `resetFadeStateAndRetireTarget` は dead code のまま（raw `enqueueRetire`＋bool 破棄）。将来の呼び出し追加時は sink 経路を使うべき旨を Audit に記録済み |

---

## 11. 最終報告

```text
STG-11-D2 = IMPLEMENTATION COMPLETE

production changes =
  src/core/SnapshotCoordinator.cpp (quarantineRetireSink 1 関数: Q→E→T 昇格)
  + D1 維持分 (EQProcessor.h / EQProcessor.Core.cpp)
test changes =
  src/tests/STG11D2SnapshotRetireTests.cpp (新規)
  + harness wiring (+11) + TD1 維持分
CMake changes =
  STG11D2SnapshotRetireTests target + add_test + IPO-OFF + ASAN
  + TD1 維持分 (test-infra only)

TD2-1 = PASS (E == 3 exact; Debug + Release)
TD2-2 = PASS (T == 3 exact; Debug + Release)
TD2-3 = PASS (all zero; Debug + Release)
TD2-4 = PASS (CTest 45/45 Debug + Release; harness both; D8_2/StuckReader included)

TD1 regression = PASS (standalone + harness, Debug + Release; oracle unchanged)
INV-D1-1..6 = intact (D2 diff touches no EQ files)

ASAN = INCONCLUSIVE / environment blocked (same signature as D1 gate;
       substitute evidence in §7)
ConvoPeq authority =
  SHA = 9565A3D546BF256F75C1A2B97FB1E9B99CFF8D6DED11A3673642EADFF4F8F5BE
  size = 5,710,250 B / Generated 2026-09-29 21:16:18
NEWER_SRC_COUNT = 0
FRESH = yes

production source = working tree (D1 + D2 uncommitted)
staged = 0
commit = 0
push = 0 (NOT AUTHORIZED)

Remaining Risks = §10 (trigger frequency / timeliness / ASAN env / N4 / dead code)
Verdict = READY FOR POST-IMPLEMENTATION GATE
```

**COMMIT / PUSH は未実施。Owner へ差し戻す。**
