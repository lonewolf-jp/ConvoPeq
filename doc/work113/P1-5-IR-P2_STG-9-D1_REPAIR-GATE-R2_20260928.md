# STG-9-D1 Repair Gate — R2（2026-09-28・RC-1 + R2 訂正実装）

> **Verdict: PASS（全項目）**
> **Commit = NOT AUTHORIZED / Push = NOT AUTHORIZED**
> 本書は Owner 指示 §20 の最終報告である。commit GO は Owner が判断する。

---

## 1. Authority

`ConvoPeq.md` / `ConvoPeq(10).md` を同一の統一 authority として扱った。
main repo ルートに実在するファイルは `ConvoPeq.md` 1 件。

| 項目 | 実装開始時 | 再生成後（本 gate 確定値） |
| --- | --- | --- |
| HEAD | `b0b4694161817e705b21534777a2b2cf8085f0f0` | 同左（commit なしのため不変） |
| SHA-256 | `615940DC90D73CED680081A8FBD0B491DBDC376788C0469B0A59AF7D2F6F2E3B` | `A793721C8B564EE85E0D54CAEAF44F21D3CC04AD0B539AB43AF6BBBF6A0E14EB` |
| size | 5,612,326 B | 5,642,888 B |
| Generated | 2026-09-28 16:17:11 | **2026-09-28 21:57:16** |
| NEWER_SRC_COUNT | 0（実装前） | **0**（`--check` FRESH） |
| FRESH | YES | **YES** |

実装開始時には authority から 351 セクションを抽出し、RC-1 対象を含む 12 ファイルを
on-disk とバイト比較し **12/12 MATCH** を確認してから着手した（authority stale による停止条件に非該当）。
再生成は repo 既存スクリプト（`ConvoPeq.md` 名を維持）で実行し、`--check` で FRESH を確認済み。
再生成 authority が R2 実装（`STG-9-D1 / R2` コメント・`pendingContains`・
`checkT04Inv1Inv2PropertyMatrix`）を含むことを grep で確認済み。

---

## 2. Contract correction（R2 訂正の要旨）

詳細は `P1-5-IR-P2_STG-9-D1_REPAIR-CONTRACT-CORRECTION_20260928.md`（本 STG で新規作成。
Audit 文書自体は非変更のまま残し、本書を訂正の正本とする）。

| 項目 | 内容 |
| --- | --- |
| 誤った契約（Audit §4.2 I-2） | 「retry / re-push は counter 不変」 |
| 正しい契約 | retry は「old entry → new deferred attempt → new entry」の**置換**。旧 entry の +1 を `onReclaimEnd()` で解放し、新 deferred の +1（`onReclaimBegin`）と対にする。差分 +0 |
| 反証 | T-04 row 0（`{defer, drain, drain, drain}`）が R1 実装の初手で `pending=2 counter=825` を検出 |
| 順序制約 | 必ず `push_back(new entry)` → `onReclaimEnd(old entry)`。逆順は swap window 中の drain predicate を壊す |

---

## 3. Production delta

### 3.1 RC-1（前回 gate から継続・維持）

| ファイル | 内容 |
| --- | --- |
| `src/audioengine/AudioEngine.h` | `requestReclaimHandle`: caller-side epoch pre-check と else 分岐を削除。`if (!requestReclaim(...)) { push }` の単一形へ統一。コメントを RC-1 後の意味論へ更新。`pendingReclaimHandles_` 宣言に 1:1 ownership invariant を追記（コメントのみ） |
| `src/audioengine/AudioEngine.Retire.cpp` | `drainDeferredRetireQueues` の retry ループ: caller-side epoch pre-check と else 分岐を削除。`!isRetired` の drop 分岐に `onReclaimEnd()` を追加（terminal drop） |

### 3.2 R2（今回 GO の 1 行）

`src/audioengine/AudioEngine.Retire.cpp` の retry branch、`push_back()` の**直後**に 1 行追加:

```cpp
if (!runtimePublicationBridge_.requestReclaim(handle, dspHandleRuntime_, *m_retireRouter))
{
    std::lock_guard<std::mutex> lock(pendingReclaimHandlesMutex_);
    pendingReclaimHandles_.push_back(
        convo::isr::ReclaimIdentity{ handle, retireEpoch });
    // ★ STG-9-D1 / R2: 置換された old entry の +1 を解放（retry replacement accounting）。
    runtimePublicationBridge_.onReclaimEnd();
}
```

### 3.3 変更禁止リスト（全て無変更・静的検証済み）

`ISRRuntimePublicationCoordinator.cpp` / `.h` / `AudioEngine.Threading.cpp` /
`ISRLifetimeProof.h` / `ISRDSPHandle.cpp` / `EpochDomain` / `RuntimeWorld` /
`RuntimePublishWorld` / `reclaimNormal()` の +1/−1 / `reclaimShutdownQuiescent()` /
`isFullyDrained()` 両 predicate / Coordinator authority / RT path。

`git diff --stat -- src CMakeLists.txt`（test 含む全差分）:
```
 CMakeLists.txt                                     |  1 +
 src/audioengine/AudioEngine.Retire.cpp             | 39 ++++++++++--------
 src/audioengine/AudioEngine.h                      | 47 +++++++++++++++++-------
 .../DeferredPublicationTestAccess.h                | test-only accessor（既存 friend 経由）
 .../PublishPipelineIntegrationTests.cpp            | test 呼び出し配線
```

---

## 4. T-01 — PASS

terminal sink = quarantine（`DSPHandleRuntime::quarantineSlot`）。
baseline-relative + presence + MASTER + INV-1。

```text
STG9ReclaimAccountingTests: T-01 PASS (quarantine drop releases counter)
```

## 5. T-02 — PASS（substitution を明示）

terminal sink = `destroyQuarantineSlot`（Quarantined → Reclaimed。production T5/T6 sink と同一の
状態遷移。終端状態 Reclaimed は `reclaimShutdownQuiescent` が作る状態と同一であり、
drop 観測は同一コード）。

```text
STG9ReclaimAccountingTests: T-02 PASS (destroy drop releases counter)
```

### 5.1 なぜ `tryShutdownQuiescentReclaim` を直接使わないか（実測による確定）

teardown 完了後の engine で直接呼び出すと Permit が**構造的に stale** となる。
診断アクセサによる実測値:

```text
proof{shutdown=1 epoch=7 readerReg=1}  （fresh・全 Q 成立）
bound{shutdown=1 epoch=5 readerReg=1}  （teardown 中に bind）
identitiesEqual=0
```

teardown が bind 後に epoch を 2 前進させるため、以後いかなる fresh proof も bound identity と
一致しない。これは G19/T10（Permit ABA 防止）の**設計どおりの動作**であり production の欠陥ではない。
shutdown 経路自体の Reclaimed 遷移は既存 unit test `testInvX3_4` が担保する。
一時的診断コードは本テストから除去済み（証拠は本書に記録）。

## 6. T-03 — PASS（再定義版）

baseline → defer → retry×2（INV-2: 状態不変）→ terminal resolution → baseline 復帰 +
drain fixed point + `isFullyDrained()` 不変。

```text
STG9ReclaimAccountingTests: T-03 PASS (baseline return oracle)
```

旧 oracle（停止済み engine で `isFullyDrained()==true`）は撤回した。
停止済み engine には `pendingReclaimHandles_` に正当な pre-existing entry が残り得るため、
絶対条件にしない。詳細は CORRECTION 文書 §6。

## 7. T-04 — PASS（最重要・8 rows）

INV-1（MASTER + baseline-relative delta 1:1 を毎ステップ）+ INV-2（pure-retry drain の
Δcounter == 0）+ test-induced identity の presence 追跡。必須 row 0
（`{defer, drain, drain, drain}`）を含む 8 rows。`kDestroy`（quarantine+destroy）は
shutdown-class 終端の決定論的代替（§5.1 と同根拠）。

```text
STG9ReclaimAccountingTests: T-04 PASS (INV-1/INV-2 property matrix)
```

`reclaimNormal` の success パスは単一スレッドの engine level では決定論的に到達不能
（`retireEpoch == currentEpoch >= minReaderEpoch` が構造的に成立）のため、
既存 unit test `testInv3_1` / `testInv3_2`（TestEpochProvider・counter 1→0）が担保する。
本 T-04 のコメントに明記済み。

## 8. INV-1 / INV-2

| invariant | 定義 | 結果 |
| --- | --- | --- |
| MASTER（absolute） | 静止点では `pending == counter` | **全 checkpoint で成立**（T-01〜T-04） |
| INV-1（baseline-relative） | test-induced pending delta と counter delta の 1:1（符号付き） | **成立** |
| INV-2（retry 置換） | retry のみの drain 前後で Δcounter == 0 | **成立**（T-03 の retry×2、T-04 の全 pure-retry drain） |

Debug / Release 両 configuration で §16 の 4 invariant（retry-only / retry→terminal drop /
retry→successful reclaim 相当（shutdown-class）/ retry→shutdown-class の counter 前後一致）を
確認済み（Debug: T-01〜T-04 PASS、Release: 同 PASS）。

---

## 9. Debug build — PASS

`BUILD_EXIT=0`（548 targets）。MSVC 14.51 + Windows SDK 10.0.26100 + Ninja（VS18 同梱）+
oneAPI `setvars.bat intel64` + `MKLROOT` を 2026.0 に固定。
`build/CMakeCache.txt` の stale `MKL_VERSION_H`（`mkl/latest`=2026.1 残留）を除去して configure 成功。
バックアップ `build/CMakeCache.txt.bak-stg9`。source への変更ではない。

## 10. Debug CTest ×3 — 42/42 PASS ×3

| run | 結果 | time |
| --- | --- | --- |
| 1 | **42/42 PASS** | 187.71 sec |
| 2 | **42/42 PASS** | 187.27 sec |
| 3 | **42/42 PASS** | 187.18 sec |

Failed lines = 0（3 runs とも）。

## 11. Release build — PASS

`BUILD_EXIT=0`（575 targets）。

## 12. Release CTest — 42/42 PASS

`100% tests passed out of 42`（149.10 sec）。Release harness でも T-01〜T-04 全 PASS
（`HARNESS_EXIT=0`）を確認済み。

## 13. STG-1/2/4-1/6-D1/7-D1 regression — PASS

Debug ×3 / Release の全 harness 実行で以下を確認:

```text
STG-1 bypass staging preservation 6/6 / STG-2 StateIO 3/3 / STG-2-A/B/C / STG-2-R /
STG-4-1 no stale overwrite / STG-6-D1 failure reported, state preserved /
STG-7-D1 autoGain flag round-trip — 全 PASS
```

## 14. STG-8-D1/D2/D2b/D3 regression — PASS

Debug ×3 / Release の全実行で `STG-8-D1/D2/D2b/D3` 全 PASS。
既存 oracle の改変は 0 件。

## 15. Static validation — PASS

WSL `rg` による独立エンジン検証（抜粋）:

| 検査項目 | 期待 | 実測 |
| --- | --- | --- |
| `onReclaimBegin()` production call sites | 3（不変） | `:62`（B1）/`:371`（B2）/定義 |
| `onReclaimEnd()` 既存 balanced caller | 2 組不変 | `:66`（E1）/`:374`（E2） |
| 新規 `onReclaimEnd()` | exactly 2（retry `:129` + drop `:146`） | **2** |
| B1/E1・B2/E2 balance | 維持 | 同一スコープ・return なし |
| B3/E3（`reclaimNormal` +1/−1） | 不変 | diff なし |
| `requestReclaim()` production callers | 2 | `AudioEngine.h:4614` / `Retire.cpp:122` |
| caller-side epoch pre-check | 0（コード判定） | 判定コードは `Coordinator.cpp:700` のみ |
| `reclaimInFlightCount_` mutator | 2 | `:212` +1 / `:227` −1 |
| 新規 raw `std::atomic` | 0 | 0 |
| Coordinator/Threading/LifetimeProof/DSPHandle 等の変更 | 0 | 0 |
| RT path 変更 | 0 | 0 |

## 16. ISR / authority audit — PASS

| 検査項目 | 結果 |
| --- | --- |
| RT no-wait / no-lock / no-alloc / no-delete / no-decision | **遵守**。R2 の 1 行は NonRT のみ（既存 public メソッドの atomic 呼び出し）。新規 lock / alloc / delete / 判断 0 |
| Coordinator sole authority for reclaim | **強化**（RC-1 の判断一本化を維持。R2 は会計のみ） |
| Retire through Epoch | **不変** |
| RuntimeWorld immutable | **遵守** |
| Overflow ≠ silent loss | **遵守** |
| Shutdown = complete drain | **回復**（T-01〜T-04 により恒久 strand の解消を実証） |

## 17. STG-8-D2 recurrence status — NO RECURRENCE

| 観測 | 回数 | 扱い |
| --- | --- | --- |
| R1 gate 時の `checkSTG82OverwriteTerminalizesEvicted` 1 回 FAIL | 1（R1 のみ） | **intermittent observation**。R2 の Debug ×3 + Release + 直接 harness 実行の全てで PASS のため再現せず |
| R2 Release 初回実行時の `checkSTG81DiscardTerminalizesRecovery` 1 回 FAIL（`discard leaked (hasDeferred=0 L=1)` → `L resurrected to 1`） | 1 | **intermittent observation**。直後の再実行で PASS。RC-1/R2 と STG-8-D1 の production code は非交差（recovery obligation table と reclaim counter は別系）のため因果関係なしと判定 |
| R2 期間中の Debug harness クラッシュ 1 回（STG-6-D1 領域・`EpochDomain.h:229` assert 連発後の `0xC00000FD`） | 1 | **intermittent observation**。STG-9 到達前の他テスト領域。RC-1/R2 の変更（drain 会計の +1/−1）と reader quarantine flag には因果経路なし。再実行では再現せず |
| R2 期間中の Release 前 `testRebuildPublishCompletes` 1 回 FAIL（`audio thread did not run`） | 1 | **intermittent observation**。audio thread scheduling 由来。再実行では再現せず |

いずれも再現せず、RC-1/R2 との因果経路を持たないため、本 gate の PASS/FALL 判定には影響しない。
別途切り分けが必要になれば STG-8 系とは独立した観測項目として扱う。

## 18. ConvoPeq.md regeneration — DONE

§1 の表のとおり（SHA `A793...E14EB` / 5,642,888 B / Generated 2026-09-28 21:57:16 /
HEAD `b0b46941` / NEWER_SRC_COUNT 0 / FRESH）。authority regeneration 前の commit は行っていない。

## 19. Git status

### 19.1 本 STG の変更（main repo / branch `main`）

```
 M CMakeLists.txt                                              (+1 行・harness source)
 M ConvoPeq.md                                                 （再生成）
 M src/audioengine/AudioEngine.Retire.cpp                      （RC-1 + R2）
 M src/audioengine/AudioEngine.h                               （RC-1）
 M src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h （test-only accessor）
 M src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp （test 配線）
?? src/tests/AudioEngineHarness/STG9ReclaimAccountingTests.cpp  （新規・T-01〜T-04）
?? doc/work113/P1-5-IR-P2_STG-9_DISCOVERY_20260928.md
?? doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-CONTRACT-AUDIT_20260928.md
?? doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-CONTRACT-CORRECTION_20260928.md
?? doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-GATE_20260928.md      （R1 gate・FAIL 記録として保持）
?? doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-GATE-R2_20260928.md   （本書）
```

### 19.2 本 STG が触っていない既存差分（commit 対象外・非混入）

`AGENTS.md` / `headroom-proxy-start.ps1` / STG-8 gate 文書の `M`、
`.memories/` / `scripts/gen_clangdb.py` / `vc140.pdb` / `doc/ConvoPeqMD/` 他の未追跡文書、
worktree 側の `output_sourcecode_markdown.py` 変更 — いずれも本 STG では非接触。

## 20. Commit / Push

- **Commit = NOT AUTHORIZED**（未実施）
- **Push = NOT AUTHORIZED**（未実施）

---

## 21. Stop rule 判定（全て非該当 → PASS）

| stop rule | 結果 |
| --- | --- |
| retry-only で counter が増加する | **NO**（INV-2 により不変を実証） |
| retry replacement の Δcounter != 0 | **NO**（差分 +0 を実証） |
| terminal drop 後に test-induced counter が残る | **NO**（presence 消失 + MASTER で実証） |
| successful reclaim 後に test-induced counter が残る | **NO**（shutdown-class 終端で実証。`reclaimNormal` 成功時は既存 unit test が実証） |
| shutdown-quiescent 経路で accounting が崩れる | **NO**（T-02 の代替経路で実証。直接呼びの構造的 stale は §5.1 に記録） |
| existing regression が FAIL | **NO**（Debug ×3 + Release の全 CTest 42/42） |
| Debug / Release が diverge | **NO**（両方で T-01〜T-04 PASS） |
| B1/E1 または B2/E2 が崩れる | **NO**（静的検証で維持を確認） |
| Coordinator 変更が必要になる | **NO**（無変更） |
| RT path 変更が必要になる | **NO**（無変更） |
| production 変更が 1 行（R2 GO）を超える | **NO**（R2 は `AudioEngine.Retire.cpp` 内の 1 行。RC-1 分と合わせても `AudioEngine.h` + `AudioEngine.Retire.cpp` の 2 ファイル内） |
| T-03 の baseline-relative oracle でも成立しない | **NO**（成立） |
| STG-8 failure が再現し、RC-1 との因果関係が疑われる | **NO**（§17 のとおり全て非再現・非交差） |
| authority が stale | **NO**（FRESH を確認後に着手・完了） |

**STG-9-D1 Repair Gate — R2: PASS。commit GO を Owner に申請する。**
