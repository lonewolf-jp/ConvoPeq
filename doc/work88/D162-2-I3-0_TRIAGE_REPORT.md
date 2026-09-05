# D162-2-I3-0 Remaining Issues Triage Report

```text
Date:     2026-09-05
Type:     read-only triage report（production/test/CMake/build 0 変更）
Evidence: evidence/D162-2I2/I3_0_TRIAGE.md（詳細版）
Input:    I1 Final PASS 確定 + D162-2-I2 完了記録の残課題 5 項目
判定:     **P0（reconfigure publication）= Case C（契約曖昧）→ ADR 先行・実装禁止**
```

---

## 1. I1 Final 正式確定

```text
D162-2-I1-A  PASS
D162-2-I1-B  PASS
D162-2-I1-C  PASS
D162-2-I1-D  PASS（I2 修復後 6/6）
D162-2-I2    PASS
I1 Final     PASS（正式確定・2026-09-05）
```

I2 で確立した「未登録 DSP → CallerDestroy → destroyRolledBackDSP」の ownership closure は
I1 lifecycle closure として完了扱い。**この経路を不用意に再変更しないこと**（ユーザー指示）。

## 2. I3-0-A/B — reconfigure 後 Active=0 の契約判定（P0・最重要）

### 4 lifetime 分離評価（要旨）

| lifetime | 範囲 | reconfigure 後の状態 |
| --- | --- | --- |
| L1 AudioEngine | ctor→dtor（MainWindow member・再生成なし） | Prepared（re-prepare 成功） |
| L2 ShutdownRuntime admission | ctor→closeAdmission→**Closed 永久**（INV-LIFE-9・reopen API 不存在・packedState_ は phase_ と別 word） | **Closed 永久** |
| L3 RuntimeIntentCoordinator | markShutdownComplete を release 毎に実施（:684）→ Bootstrapping 復帰あり | 再 bootstrap 準備済み（L2 と非対称） |
| L4 JUCE prepare/release | device 変更のたび反復（JUCE 契約） | re-prepare 完了・publish 不可能 |

### Case 判定: **Case C（契約曖昧）**

- **Case B 寄りの凍結契約**: T8「Path B resurrection → rejected」は dash2 A2 の機械的
  acceptance 項目（REPAIR_PLAN2-dash2.md:2811）・INV-LIFE-9 は第五者レビュー #31 Phase 0
  凍結対象（同書:3423）・isShutdownInProgress は「OR 判定を永久維持」（AudioEngine.h:1525）・
  publishIdleWorldOnly に明示 guard。
- **Case A 寄りの実装**: prepareToPlay は完全 re-entrant（Unprepared→Preparing 第一級遷移・
  rebuild thread 再起動・generation reset）・:217-220 comment は re-prepare 後も publish が
  機能する前提・markShutdownComplete の release 毎実施。
- **決定的欠落**: dash2 設計文書は JUCE host 契約（同一 engine reconfigure 反復）と
  admission lifetime の関係に**触れる記述がゼロ**。
- **実務的帰結**: `AudioEngine` は MainWindow member（再生成なし）のため interactive で
  device 変更 1 回目から publication が永久不能（bypass 継続）。Case B を「意図」と
  呼ぶには機能的に重い。
- **結論**: 実装禁止（reopen・C/D 案着手禁止）。**次工程 = ADR**:
  (i) engine lifetime = admission lifetime 正式化（Case B・reconfigure は engine 再生成を
  host 契約とする）か (ii) reconfigure-aware admission 設計（Case A・INV-LIFE-9 再審）かを
  文書決定するまで、どの実装も着手しない。

## 3. I3-0-C — Debug double-release crash（P1）

- crash 区間: 2 回目 releaseResources の `ABOUT_TO_EXIT_SCOPE`（ReleaseResources.cpp:710）
  直後 〜 `~AudioEngine: enter`（CtorDtor.cpp:106）直前。I2FileLogger（1 行 flush）で
  engine diag を全捕獲済み（teardown_diag.log 357 行・最終行が ABOUT_TO_EXIT_SCOPE）。
- SIGSEGV（139）であり `leaveRelease` の `std::abort()`（exit 3）ではない。
- double-processed 状態の確定:
  - isr ShutdownRuntime FSM の逆行（ShutdownComplete→AudioStopped）は
    `transitionViolations_++` で握り潰し（ISRShutdown.cpp:145-148）— crash 箇所ではない。
  - LifecycleIsolationRuntime の Released→Preparing→Released は遷移表どおり合法。
  - slot/handle/registry/world/deferred は I2 後整合（BISECT で修復無起因も確認済み）。
  - 残候補: coordinator 停止下の DeferredDeletionQueue/OwnerChannel drain、
    ~AudioEngine の re-prepare 後 teardown。
- 確定手段: x64dbg attach / WER dump 解析 → **I3-1（crash audit）**。I2 影響なし
  （BISECT 済み）。

## 4. I3-0-D — Release harness 0xc0000374（P2）

- G2〜I1-R2 の全 Release CTest が `AudioEngineHarness 0xc0000374`（3.5-5.0s）で 39/40。
  I2 後は即時 SEGFAULT に変化（リンク順による layout 差と推定・未検証）。
- production/test revert bisect で **I2 無関係確定**。帰属確定は I3-1（Debug crash audit と
  x64dbg を共用）。

## 5. I3-0-E — ownership=None conservation（P3）

6 返却点の全表（evidence 版参照）から:

- 既存 registered DSP 経路（pubResult1/Timer/Transition）: rollback CAS 失敗 → 登録温存 →
  registry ownership 保持 = conservation 成立。
- 新規 placeholder 経路: rollback 成功 → 未登録 → caller が owner だが `None` は
  破壊義務を通知しない = **latent ownership hole**（I2 分岐を素通りして leak 成立）。
- ただし None を返す #3（world==nullptr）/#4（seqId==0）は production 不発
  （build 失敗=OOM のみ・seqId は常に非 0）。
- **判定**: 修復は enum cleanup ではなく **ADR で None の契約を確定**（「ownership 変更なし・
  caller が事前状態維持」+ 新規 DSP を渡す caller の契約明確化）してから I2 分岐拡張の
  可否を判断。緊急度低（現行不発）。

## 6. I3-0-F — MMCSS / FFTBackendTests（P4/P5）

### MMCSS err=1552（確定）

- **1552 = `ERROR_THREAD_ALREADY_IN_TASK`**（winerror.h:10219）。コード comment の
  「1552 (ERROR_NO_MORE_ITEMS)」は誤り（`ERROR_NO_MORE_ITEMS` = 259L・winerror.h:1927）。
  `AudioEngine.Mmcss.cpp:143` の expected 分岐（5/183/259）が 1552 を catch しないため
  FAILED に到達 — 実態は「thread が既に MMCSS task 所属」= 正常系 success 扱いが意図だった。
- **ASIO tag の根因**: `setAudioDeviceTypeName` は session 初期化時のみ呼ばれる
  （DeviceSettings.cpp:1133/1260）→ device switch 後に policy が再 publish されず残値
  （SelfManagedProAudio）が使用される。diag 表示のみで gate 影響なし。
- 修復候補（I3-1・小）: :143 へ 1552 追加（+ comment 修正）+ DEV_SWITCH 後の
  setAudioDeviceTypeName 再呼び出し。

### FFTBackendTests CMake（確定）

- RWDI full build 失敗の原因: **`/utf-8` 未付与 target での CP932 誤解釈**
  （C4819 実測）。FFTBackendTests（CMakeLists.txt:614-627）には
  `/utf-8` も `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`/`JUCE_DSP_USE_INTEL_MKL` の
  compile definition も未付与（AudioEngineHarness は :1876 で `/utf-8` あり）。
  BOM 無し UTF-8 日本語コメントが CP932 で parse 破損 → `DIAG_MKL_MALLOC` 未定義
  （C3861）+ 仮引数解決失敗（C2065）。
- **stale-obj マスキング**: Debug/Release は過去 .obj が残存していたため I2-3 で通過した
  ように見えたが、RWDI は初回 compile で顕在化。**clean rebuild でも Debug/Release が
  壊れる潜在 build 破損**。修復は `/utf-8` 追加が最小（P5・lifecycle 系から分離）。

## 7. 仕分け結果と I3-1 の推奨着手順

```text
1. ADR: reconfigure × admission lifetime 契約（P0・Case C 解消 — 文書のみ）
2. I3-1 crash audit: Debug double-release + Release 0xc0000374（x64dbg 共用・P1/P2）
3. ownership=None ADR（P3・P0 の ADR に併記可）
4. 小修正群（P4 MMCSS 分類 / P5 FFTBackendTests /utf-8 — それぞれ独立・小）
```

本 triage は read-only 完了。I1 Final / I2 の成果に変更なし。
