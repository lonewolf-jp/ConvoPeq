# STG-11-D5 Post-Implementation Gate

- Document: `doc/work113/P1-5-IR-P2_STG-11-D5_POST-IMPLEMENTATION-GATE_20260930.md`
- Work item: **STG-11-D5** — MMCSS RT 到達経路の残存 RT 契約違反修正
- Predecessor: `P1-5-IR-P2_STG-11-D5_REPAIR-CONTRACT-AUDIT_20260930.md`（GO） /
  `P1-5-IR-P2_STG-11-D5_REPAIR-IMPLEMENTATION_20260930.md`
- Date: 2026-09-30
- HEAD: `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- Branch: `main...origin/main [ahead 9]`, staged = 0

---

## 判定

```
READY FOR COMMIT
```

**commit / push は未実施。** Owner の commit GO を待つ。

---

## 1. Authority 再確認

### 1.1 着手時

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `678B5BA15658A7AEDCED97D0E6A59EC2D3AF4C9BBE107350EF4710CC9B3DF57C` |
| size | 5,805,842 B |
| Generated | `2026-09-30 21:09:22` |
| NEWER_SRC_COUNT / `--check` | 0 / FRESH（exit 0） |

### 1.2 完了後（再生成後）

| 項目 | 値 |
| --- | --- |
| SHA-256 | `4014C78F3B2095981817275CC8EE220B6FC2570A04C8A2F1357C751B` |
| size | 5,854,098 B（着手時比 +48,256 B） |
| Generated | `2026-09-30 22:17:20` |
| NEWER_SRC_COUNT / `--check` | 0 / FRESH（exit 0） |

D5 の production 2+1 ファイル・test TU・test access が反映されていることを確認
（`recordMmcssEventObserved` / `reportMmcssEventIfRecorded` / `mmcssObserved_` 28 件）。
D1 / D2 / D3 / D4 も引き続き反映済み。`ConvoPeq.md` は編集していない。

---

## 2. Core repair の判定

| 項目 | 判定 | 根拠 |
| --- | --- | --- |
| M1〜M4 の RT logging 除去 | **PASS** | 旧文言 4 種が RT 関数内に不在。registration body に backend token なし |
| M5 の RT logging 除去 | **PASS** | 旧文言が revert 関数内に不在。revert body に backend token なし |
| OS API の実行場所維持 | **PASS** | D5-T4（Av* site 数不変、same-thread revert 維持） |
| `thread_local` lifetime 維持 | **PASS** | 3 変数の宣言不変（D5-T4） |
| Message Thread は flag のみ | **PASS** | ReleaseResources に Av* なし（D5-T4） |
| failure / return / fallback semantics 不変 | **PASS** | D5-T5（error 分類・fallback 順序・return 数・once-guard） |
| 新規 thread / timer / queue / authority なし | **PASS** | authority audit 0 件 |

---

## 3. Atomic rule の判定

| 項目 | 結果 |
| --- | --- |
| D5 追加行の raw `std::atomic` API | **0 件** |
| D5 の atomic 操作 | **全件** `convo::` wrapper（21 箇所） |
| memory-order | 既存 convention（`release` / `acquire` / count は `acq_rel`） |
| 新設 synchronization 構造 | **0 件** |

D5 atomic member は **8 件**（Contract §4.5 どおり。D3 の 6 件・D4 の 7 件とは別物）。

---

## 4. 不変条件の判定

| ID | 判定 | 実測 |
| --- | --- | --- |
| INV-D5-1 / INV-D5-2 | **PASS** | M1〜M5 `RT-safe (backend-free, record-only)` |
| INV-D5-3 | **PASS** | record 5 件が guard 外。report 発行部が guard 内。OFF 時の直接 lock なし |
| INV-D5-4 | **PASS** | 新 authority 0。輸送物は整数のみ |
| INV-D5-5 順序 | **PASS** | `payload -> count(fetchAdd/acq_rel) -> observed(publish/release)` |
| INV-D5-5 読順 | **PASS** | `observed(fast-out) -> count(acquire) -> snapshot(acquire)` |
| INV-D5-5 誤確定なし | **PASS** | 5 ラウンド全 PASS。`reportedCount` は毎回ちょうど +1 |
| C-D5-OS1〜OS4 | **PASS** | D5-T4（site 数・same-thread・flag-only） |
| D1 / D2 / D3 / D4 invariant | **PASS** | TD1 / TD2 / D3-T1〜T5 / D4-T1〜T6 全 PASS |

---

## 5. Diagnostics ON / OFF の判定（D5-T3）

| 象限 | 判定 | 根拠 |
| --- | --- | --- |
| **OFF** | **RT-safe** | body 内の残存 backend token なし。直接 lock なし |
| **ON** | **RT-safe + 既存文言で診断** | D5-T1/T2（RT 側不在）＋ D5-T3（NonRT report が原文言＋guard 内） |

### Negative control（D5-T8）

M5 を旧 `diagLog` へ**一時的に復元**し再 build:

```
D5-T1: old wording '[MMCSS] reverted on Audio Thread' still inside RT function
FAIL: D5-T1/D5-T2 MMCSS sites backend-free
harness rc = 1
```

→ oracle は欠陥を正確に捕捉する。復元後の再 build・再実行で全 PASS に戻ることを確認
（record 5 件の存在を実ファイルで再確認）。

---

## 6. Regression の判定（D5-T7 / T9 / T10）

### 6.1 D5 結果（Debug / Release、harness 直接実行、rc=0 両方）

```
PASS (D5-T1/D5-T2/D5-T3/D5-T4/D5-T5/D5-T6)
```

### 6.2 D4 / D3 regression

```
STG11D4SuccessObservationTests: PASS (D4-T1/D4-T2/D4-T3/D4-T4/D4-T5/D4-T6)
STG11D3AffinityFailureTests: PASS (D3-T1/D3-T2/D3-T3/D3-T4/D3-T5)
```

### 6.3 D1 regression

```
TD1-1 PASS (Q=304) / TD1-2 PASS (T=280) / TD1-3 / TD1-4a / TD1-4b PASS   rc=0
```

### 6.4 D2 regression

```
TD2-1 PASS (E=3) / TD2-2 PASS (T=3) / TD2-3 PASS   rc=0
```

### 6.5 full Debug CTest（D5-T9）

```
100% tests passed out of 45
```

### 6.6 full Release CTest（D5-T10）

```
100% tests passed out of 45
```

### 6.7 raw std::atomic audit（D5-T11）

D5 追加行に raw API **0 件**。

### 6.8 RT lock/allocation/backend audit（D5-T12）

`recordMmcssEventObserved` 本体 **0 件**。

### 6.9 authority audit（D5-T13）

新規 authority / 同期構造 **0 件**。ownership 輸送なし。

### 6.10 静的解析

| ツール | 対象 | 結果 |
| --- | --- | --- |
| clang-tidy | `AudioEngine.Mmcss.cpp` | finding **0** |
| cppcheck 2.22 | project DB | project 全体 135 件は既存ノイズ。**Mmcss.cpp 0 件** |

### 6.11 ASAN

環境 block（`0xC0000139`）。**別問題として記録し、PASS と扱わない。**
代替 evidence: 構造的 oracle（D5-T1/T4/T5/順序）＋ negative control ＋
Debug/Release 45/45 ＋ exact-count oracle（D1/D2 不変）＋ deterministic 契約（5 ラウンド）＋静的解析。

---

## 7. Diff scope の判定

### 7.1 変更ファイル一覧

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/audioengine/AudioEngine.Mmcss.cpp` | 125 | 37 | **D5** |
| `src/audioengine/AudioEngine.Timer.cpp` | 195 | 23 | D3 + D4 + **D5**（hook 1 行） |
| `src/audioengine/AudioEngine.h` | 54 | 0 | D3 + D4 + **D5**（20 行） |
| `src/tests/AudioEngineHarness/STG11D5MmcssObservationTests.cpp` | 新規（untracked） | — | **D5** |
| `src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h` | 159 | 0 | D1 + D3 + D4 + **D5**（56 行） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | 26 | 0 | D1〜D4 + **D5**（5 行） |
| `CMakeLists.txt` | 135 | 0 | D1〜D4 + **D5**（1 行） |
| `src/core/SnapshotCoordinator.cpp` | 21 | 3 | D2（既存・保持） |
| `src/eqprocessor/EQProcessor.Core.cpp` | 37 | 10 | D1（既存・保持） |
| `src/eqprocessor/EQProcessor.h` | 39 | 0 | D1（既存・保持） |
| `ConvoPeq.md` | 4805 | 227 | 再生成 |

### 7.2 禁止領域の変更なし

`ISRRetireRouter` / `EpochDomain` / `ISRRuntimePublicationCoordinator` /
`ISRCoordinatorLoop` / `PublicationAdmission` / `RuntimeStore` /
`Crossfade` / `Publish` / `Retire` / `Recovery` / `diagLog` backend — **すべて未変更**。
D1〜D4 の既存修正の巻き戻しなし。D3/D4 の ordering・test contract 変更なし。

### 7.3 pre-existing 変更

`AGENTS.md` / `headroom-proxy-start.ps1` / STG-8 gate 文書には**触れていない**。

---

## 8. 観測事項（未修正）

### OBS-D5-1: `revertMmcssPriorityOnAudioThread()` は caller なしの dead code

`finalizeMmcssShutdown()`（NonRT）と役割重複の legacy alias と見られる。
削除は scope 外。次回 Discovery 候補。

### OBS-D5-2: 既存 test の intermittent（R8）

D5 作業中に flake による失敗は観測されなかった（復元直後の実行を含む全 run が rc=0）。
D3 Gate の OBS-5 / D4 Gate の OBS-D4-3 は引き続き未解決の既知 flake として残る。

---

## 9. Gate 判定

| Gate 項目 | 判定 |
| --- | --- |
| Authority 再確認（着手時・再生成後） | PASS |
| D5-1 call chain audit | PASS |
| D5-2 OS API audit（移動不可能性の立証） | PASS |
| Core repair（RT-safe observation のみ） | PASS |
| OS API 実行場所・thread 契約の維持 | PASS |
| failure / return / fallback semantics 不変 | PASS |
| Atomic rule（raw 追加 0） | PASS |
| NonRT side（既存 execution point のみ） | PASS |
| Diagnostics ON/OFF 双方 | PASS（negative control で実証） |
| INV-D5-1 .. INV-D5-5 / C-D5-OS1〜OS4 | PASS |
| D1 / D2 / D3 / D4 invariant | PASS |
| D5-T1 .. T6（T7 は §6、T9〜T13 は §6.5〜6.9） | PASS（Debug / Release） |
| Negative control（D5-T8） | PASS |
| full Debug / Release CTest | PASS 45/45 |
| audits（T11/T12/T13）＋静的解析 | PASS |
| ConvoPeq regeneration + freshness | PASS |
| Diff scope（禁止領域 0 変更） | PASS |
| D1〜D4 の保持 | PASS |
| commit / push | **未実施** |

```
READY FOR COMMIT
```

---

## 10. commit 時に Owner が確定すべき境界（D4 Gate §10 を継承・更新）

現在の作業ツリーには **D1 + D2 + D3 + D4 + D5** の未 commit 変更が同時に乗っている。

| 選択肢 | 内容 |
| --- | --- |
| A | 5 件を 1 つの commit にまとめる（同一 STG の連続修正。wiring を共有するため分割コストが高い） |
| B | 複数コミットに分割する（中間状態で build 不能になる共有ファイルがあるため非推奨） |
| C | D1 + D2 を先に commit し、D3 + D4 + D5 を別 commit にする |

Owner が指示を出すまでは、追加 commit を行わない。

**push は引き続き一切禁止。D6 へは進まない。**
