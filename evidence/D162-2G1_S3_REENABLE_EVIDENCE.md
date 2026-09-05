# D162-2-G1 Evidence — S3 Staged Re-enable (S3=ON / V-D=OFF)

- Work item: D162-2-G1（D162-2-G0 PASS を受けての S3 再有効化・ステージ 1/4）
- Date: 2026-09-04
- 基準ソース: ConvoPeq.md `Generated: 2026-09-04 18:29:10` working tree に G1 変更を適用
- production 変更: **1 ファイル**（`src/audioengine/RuntimePublicationOrchestrator.cpp`・
  `clearDeferredForShutdown()` のみ）。V-D（ReleaseResources.cpp）は**未接触**。
- 判定: **PASS**（§4 判定表）

---

## 1. G1 diff（production 変更の全体）

`clearDeferredForShutdown()` 内・`deferredSlot_.reset()` 直前に挿入（実ファイル行 618-633）:

```cpp
        // ★ D162-2-G1 (S3 re-enable): slot reset の前に、slot が保持していた handle 登録済み
        //   DSPCore を authority（EBR・INV-D162-8 単経路）で disposition する。
        //   midrun 経由（drainDeferredClearIfRequested → E-4d retire 済み）では map 不在で
        //   no-op となるため二重 retire は構造的に発生しない（INV-D162-3）。
        if (deferredSlot_.has_value())
        {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            juce::Logger::writeToLog(juce::String::formatted(
                "[D162-2E_DEFERRED] event=CLEAR_SHUTDOWN_DISPOSITION gen=%llu dsp=%p",
                (unsigned long long)deferredSlot_->request.generation,
                (void*)engine_.resolveDSPHandle(deferredSlot_->request.newDSP)));
#endif
            retireRegisteredDSP(
                deferredSlot_->request,
                "shutdown-clear");
        }
        deferredSlot_.reset();
```

加えて同関数内の 2 点のみ修正:
- 旧 `event=CLEAR gen=%llu dsp=%p (no disposition — pre-existing mismatch)` の文言を
  `event=CLEAR gen=%llu dsp=%p` に更新（mismatch 解消済みのため誤記表記を除去）。
- D162-2-B 残課題コメントを D0/F 経緯込みの G1 状態に更新（挙動変更なし）。

**触れていないもの**（指示どおり）: ReleaseResources.cpp / V-D `if (false &&)` /
destroyRolledBackDSP / tryShutdownQuiescentReclaim / EBR 実装 / AudioSegmentBuffer.h /
AlignedAllocation.h / DSPLifetimeManager / E-3 assertion / S1/S2/S4 / テストコード。

## 2. G1-1 Build（build-diag・既存レシピ踏襲）

| config | 結果 | 証跡 |
| --- | --- | --- |
| Debug（全ターゲット 236） | **EXIT=0** | evidence/D162-2G1_build_debug.log |
| Release（ConvoPeq 16） | **EXIT=0** | evidence/D162-2G1_build_release.log |
| RelWithDebInfo（138） | **EXIT=0** | evidence/D162-2G1_build_rwdi.log |

## 3. G1-2 CTest（build-diag）

| suite | 結果 | 備考 |
| --- | --- | --- |
| Debug | **40/40 PASS**（36.95s） | AudioEngineHarness 含め全 PASS |
| Release | **39/40**（15.52s） | 唯一の失敗 = AudioEngineHarness `0xC0000374` — D162-1P/B/E/F と同一の pre-existing Release DIAG 起動クラッシュ → **従来どおり除外で 39/39 相当 PASS** |

## 4. G1-3/4/5 Soak 結果一覧

実行順は指示どおり Debug 6-gen → RWDI 6-gen → RWDI 60-gen（ladder を崩さず実施）。

| 指標 | Debug 6-gen | RWDI 6-gen | RWDI 60-gen |
| --- | --- | --- | --- |
| exit code | **0x00000000** | **0x00000000** | **0x00000000** |
| crash dump（本 run 由来） | **0 件** | **0 件** | **0 件** |
| `CLEAR_SHUTDOWN_DISPOSITION` | **1** | **1** | **6** |
| `event=CLEAR` | 1 | 1 | 6 |
| `CLEAR_MIDRUN_DISPOSITION` | 1 | 1 | 6 |
| `shutdown-clear` retire 呼出 | 1（全て `retired=0` no-op※） | 1（同※） | 6（同※） |
| D117_DESTROY / FOOTPRINT_RELEASED | 7 / 7 | 7 / 7 | **61 / 61** |
| residual（enqueued − destroyed 補正後） | **0** | **0** | **0** |
| EBR pend 最終 / 最大 | 0 / 1 | 0 / 1 | 0 / 2（全て運転中一時・運転内消化） |
| EBR overflow | 0 | 0 | 0 |
| `INV-D162-8` DIAG（E-3 違反ログ） | **0 件** | **0 件** | **0 件** |
| Signature A（`AudioSegmentBuffer` crash chain） | **0 件** | **0 件** | **0 件** |
| Signature B（`detectStuckReaders` jassert） | **0 件** | **0 件** | **0 件** |
| Signature C（新規） | **0 件** | **0 件** | **0 件** |
| XRUN | 2（Callback ≤3.44ms） | 4（≤1.69ms） | 9（baseline: F=15 / E=17 を下回る） |
| E-4 会計 | 2434 = 2430+3+**1** | 2673 = 2669+3+**1** | **28456 = 28409+40+6+1** |
| shutdown sequence complete | 1 | 1 | 1 |
| `[FAULT] coordinator Faulted` | 1 | 1 | 1（**pre-existing**: B/C/D0/E/F の全過去ログに同数存在） |

※ shutdown-clear retire は全例で **E-4d（timer-clear-midrun）retire の直後に同一 DSP へ
二重呼び出しされたケース**であり、1 回目の map erase により `retired=0`（no-op）で
帰還 — INV-D162-3（二重処分禁止）の構造的保証を実行時 8 回検証した。

## 5. disposition 対応（CREATE → CLEAR_SHUTDOWN_DISPOSITION → retire → EBR → destroy）

### 5.1 Debug 6-gen（gen=14・dsp=000001E2F062D0C0）

```text
event=CREATE gen=14
event=CLEAR_MIDRUN_DISPOSITION gen=14            ← Timer C2/C3/C4 → requestDeferredClear → latch
  → [D162-2B_RETIRE] origin=timer-clear-midrun   ← E-4d: retired=1（map erase + EBR enqueue）
  → [D117_RETIRE] enqueue=0 (Success) epoch=32   ← EBR 破壊権取得
event=CLEAR gen=14                               ← clearDeferredForShutdown（midrun 終端）
event=CLEAR_SHUTDOWN_DISPOSITION gen=14          ← ★ G1 S3 block 実行
  → [D162-2B_RETIRE] origin=shutdown-clear       ← retired=0（map 不在 → no-op・INV-D162-3）
  ...
[D117_DESTROY] dsp=000001E2F062D0C0              ← EBR digest（dtor body 内）
[DSP_FOOTPRINT_RELEASED] dsp=000001E2F062D0C0 remaining=0
```

### 5.2 RWDI 60-gen（6 件全 closure）

| gen | dsp | retired=1 (E-4d) | enqueue | shutdown-clear | D117_DESTROY | RELEASED |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 000001CF40313080 | ✓ | 0=Success (ep 148) | no-op | ✓ | remaining=0 |
| 46 | 000001CF34D1B080 | ✓ | 0=Success (ep 164) | no-op | ✓ | remaining=0 |
| 48 | 000001CF2C90E080 | ✓ | 0=Success (ep 170) | no-op | ✓ | remaining=0 |
| 54 | 000001CF42747080 | ✓ | 0=Success (ep 192) | no-op | ✓ | remaining=0 |
| 59 | 000001CF413E2080 | ✓ | 0=Success (ep 216) | no-op | ✓ | remaining=0 |
| 64 | 000001CF41370080 | ✓ | 0=Success (ep 234) | no-op | ✓ | remaining=0 |

**6/6 で CREATE → disposition（S1 authority）→ EBR（Success）→ destroy → footprint released
remaining=0 の完全閉包。** residual 0。

### 5.3 F baseline との突合（S3 効果の定量化）

| run | destroys | 備考 |
| --- | --- | --- |
| F 60-gen（S3 OFF） | 60 | shutdown 時 deferred DSP 1 件が無処分（S3 無効の既知残留） |
| **G1 60-gen（S3 ON）** | **61** | **無処分 1 件が解消**（= E closure 水準 61 と一致・S3 の no-op 安全性も 8 回実行時検証） |

## 6. G1-2/3/4/5 の crash dump 分類

| dump | 時刻 | 由来 | 分類 |
| --- | --- | --- | --- |
| ConvoPeq.exe.3668.dmp | 19:16 | Release CTest **HeadlessAudioPathVerification** が起動した **旧 build-icx Release binary**（8/18 ビルド・F/G1 修正を含まない）の 0xC0000005 static-teardown — cli-smoke-test.ps1 が明示 toleration する既知挙動（F-era dump 24808 と同一クラス） | **pre-existing・G1 無関係** |
| AudioEngineHarness.exe.30912.dmp | 19:17 | Release CTest AudioEngineHarness 0xC0000374（D162-1P/B/E/F 文書済み pre-existing） | **pre-existing・G1 無関係** |
| G1 Debug 6-gen / RWDI 6-gen / RWDI 60-gen | — | dump **0 件** | — |

Signature A/B/C は全 run で **0 件**。

## 7. 観測可能性条件の充足判定（G0 §7.3）

- 実行時検証条件 `count(CLEAR_SHUTDOWN_DISPOSITION) > 0`: **充足**（1 / 1 / 6）。
- ただし全例が「E-4d midrun retire 済み DSP への二重呼び出し（no-op）」経由であり、
  **EmergencyDrain 直接 / C1 direct で slot 保持 DSP が残り、S3 retire が『唯一の
  disposition 実行者』（retired=1）になるケースは今回の soak では未観測**。
  → G1 の gate 条件自体は充足（S3 block は実行時に到達・安全に帰還）するが、
  この变異は **G3 開始前の追加観測事項**として記録する（G2 には影響しない）。

## 8. その他の観測（全て pre-existing 確認済み）

| 観測 | G1 | 過去ログ照合 |
| --- | --- | --- |
| `[FAULT] coordinator in Faulted state after markShutdownComplete` | 1 件/run | D162-2B_soak6 / D162-2D0_dbg6 / D162-2E_soak60b / D162-2F_soak60 に各 1 件 — pre-existing |
| `drain timeout reached, performing safe tryReclaim`（releaseResources waitForDrain 2000ms） | 1 件 | E/F 60-gen とも 1 件 — pre-existing（dtor D5/D8 が消化し E-3 assert 通過） |
| `RECOVERY execute action=3` | 発生 | F dbg6=1 / F soak60=6 — pre-existing |
| Priv 最終 | 593MB（382→593） | F=445MB / E=461MB — 同オーダー（NUC live=4・DC live 1-3 収束・leak なし） |

## 9. G1 判定

| 状態 | 該当 | 判定 |
| --- | --- | --- |
| **S3 発火 + crash なし + residual 0 + EBR closure** | **該当** | **G1 PASS** |
| Signature A | 0 件 | F regression なし |
| Signature B | 0 件 | 記録対象すら出現せず |
| Signature C / E-3 jassert | 0 件 | なし |
| residual > 0 / EBR pend 恒常増加 / XRUN 新規 | なし | — |

**D162-2-G1 = PASS。** G2（V-D-b: authority `retire()` 化）への進行条件を満たす。
V-D は本ステージでは一切有効化していない。
