# D162-2-I1-Final — 再監査記録（I2 修復後・I1-A/B/C/D + Final）

```text
Date:     2026-09-05
Type:     I1 Final re-audit（I2 修復込み binary での D profile 再試験結果に基づく）
Input:    I1-A/B/C/D 実測 + D162-2-I2 完了記録（evidence/D162-2I2/I2_COMPLETION.md）
判定:     **I1-A PASS / I1-B PASS / I1-C PASS / I1-D PASS（I2 後 6/6）/
          D162-2-I2 PASS / I1 Final = PASS（正式確定・2026-09-05 ユーザー承認）**
```

---

## 1. I1 Final 再判定条件の達成

指示の再判定条件は「DSP lifecycle: constructed = destroyed が **pointer identity で 6/6**」。

I2 修復 binary（RWDI Sep 5 10:12）での Profile D 再試験:

| 項目 | I1-D（修復前） | I2 後再試験 |
| --- | --- | --- |
| constructed = destroyed（pointer identity） | 6/6 **FAIL**（orphan 6/6） | **6/6 PASS**（2 construct = 2 destroy・完全一致） |
| placeholder orphan | 6/6 検出（~108MB 未処分） | **0/6**（D117_DESTROY → DSP_DESTROY_FOOTPRINT → DSP_FOOTPRINT_RELEASED remaining=0 の chain 6/6） |
| crash 系 12 gate | 全 PASS | 全 PASS（劣化なし） |
| E-4 / zone / XRUN / stale / Signature | 全 PASS | 全 PASS（劣化なし） |
| shutdown 時 `retired=0`（dangling slot 証拠） | 6/6 出力 | **0 件**（slot 事前 null 化） |

I1-D は A′ 修復自体の regression ではなく、独立した reconfigure 経路の ownership gap
（D162-2-I1-D-R0 で root cause 確定）による lifecycle gate FAIL であった。I2 でその gap を
最小単位（PrepareToPlay.cpp 1 箇所）で閉じ、同一条件の再試験で lifecycle gate が
成立したため、**I1-D の STOP を解除し PASS に移行する**。

## 2. I1 Final 判定

```text
D162-2-I1-A  PASS（plain ×8: exit 0x0・dump 0・zone clean・V-D closure・gen 1:1・XRUN 全 startup transient）
D162-2-I1-B  PASS（IR+burst ×6: E-4 収支 6/6・S3 retired=1 6/6 再現・residual 0・V-D-b 防御 6/6）
D162-2-I1-C  PASS（IR+rebuild ×6: IR+rebuild 1 回/run・E-4 完全収支・gen 1:1・S3 no-op 正常）
D162-2-I1-D  PASS（device cycle ×6・I2 修復後: orphan 0/6・constructed=destroyed 6/6・全 gate 維持）

D162-2-I2    PASS（orphan ownership repair・最小単位実装・実機 6/6 証明）

I1 Final     PASS（正式確定・2026-09-05）

A′ deferred terminal disposition: proven effective（B 6/6 retired=1・C/D no-op 冪等・D 再試験でも劣化なし）
独立発見（reconfigure orphan）: D162-2-I2 により修復・実証済み。以後この経路（未登録 DSP →
CallerDestroy → destroyRolledBackDSP）を不用意に再変更しないこと（ユーザー指示）。
```

I1 の契約（I0 測定 reconciliation で固定）: exit 0x0 / dump 0 / zone 3 行 / residual 0 /
EBR 0 / E-3 0 / direct destroy 0 / stale HIT 0 / Signature 0/0/0 / dup destroy 0 /
shutdown-window XRUN 0 / E-4 収支 / generation 1:1 — **全 profile 全項目で成立**。

## 3. 残課題（I1 Final の判定に影響しない・I3 以降）

1. reconfigure 後の publication 不可（Active=0・bypass 継続）— admission reopen を伴う
   設計課題（R0 §7-C/D）。
2. Debug: reconfigure 二重 release（prepare→release→prepare→release）の pre-existing segfault。
3. Release: AudioEngineHarness pre-existing crash（0xc0000374・G2 以来）。
4. RWDI full build: FFTBackendTests の macro 定義不足（target_compile_definitions）。
5. MMCSS-ASIO err=1552 の分類不正（AudioEngine.Mmcss.cpp:143 の定数比較）。
6. `commitRuntimePublication` の `ownership=None` 返り値（world==nullptr / seqId==0）の
   caller 契約明確化。

## 4. 成果物一覧

- I2: `evidence/D162-2I2/I2_0_PREFLIGHT.md`・`I2_COMPLETION.md`・再試験ログ
  `evidence/D162-2-I1/D2-D7.log`（I2 binary 版・上書き）
- I1-D-R0: `evidence/D162-2-I1/D_R0_RECONFIGURE_ORPHAN_OWNERSHIP_AUDIT.md`
- 本記録: `evidence/D162-2-I1/I1_FINAL_REAUDIT.md`
- production diff: `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`（1 箇所 + include）
- test diff: `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp`・
  `AudioEngineHarness.h/.cpp`（seam 追加）
