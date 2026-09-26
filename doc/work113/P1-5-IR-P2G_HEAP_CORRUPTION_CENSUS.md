# P1-5-IR-P2-G — heap corruption source census（read-only）

- **作成**: 2026-09-22 / work113 P1-5-IR Phase 2-G
- **性質**: read-only source/provenance audit（production/test 変更 0・実行なし）
- **対象**: `AudioEngineHarness+0x203556d`（free caller）の symbol/source 解決と、free 対象 block の provenance census
- **基準**: HEAD `1e9e63e3`・binary `F6C49980920C1D8E`・cdb attribution 済（`P1-5-IR-P2F3_SIGSEGV_ATTRIBUTION_CDB.md`）

---

## 1. 状態固定（3-G-0）

| 項目 | 値 |
| --- | --- |
| HEAD | `1e9e63e3` |
| production source / CMake / settle / protocol / commit | 0 / 0 / 0 / 0 / 0 |
| binary SHA-256[:16] | `f6c49980920c1d8e` |
| link PDB | **build tree 内に存在しない**（compile PDB `vc140.pdb` のみ: `build/CMakeFiles/AudioEngineHarness.dir/Release/vc140.pdb` 59MB） |

## 2. Symbol resolution（3-G-1）= 不可能を確定

- link 時 PDB（`AudioEngineHarness.pdb`）が build tree に存在しない（`find build -name "*.pdb"` → compile PDB のみ）。
- cdb による `.reload /f` + `lmvm` → **`(no symbols)`**。vc140.pdb を `AudioEngineHarness.pdb` にリネームして隣接配置しても解決しない（compile PDB は GUID/RVA contribution を link と共有しないため）。
- **`AudioEngineHarness+0x203556d` → function/source/line = 解決不能（symbol unavailable）**。

## 3. free caller の disassembly 特徴づけ（3-G-2・部分）

crash log の stack で `free_base` の return address = image+0x203556d。その前後の disassembly:

```text
+0x2035560:  mov  rcx,qword ptr [img+0x2829e70]      ← cached pointer slot
+0x2035567:  call qword ptr [img+0x2220948]          ← indirect call（free 経路・IAT 風 slot）★ free_base の呼び出し
+0x203556d:  mov qword ptr [img+0x2829e70],r15       ← slot を NULL 化（r15=0）
+0x2035574:  mov rcx,r14 / mov dword ptr [img+0x2829e78],r15d
+0x20354eb:  sub edi,1 / jns 0x2035470               ← atexit テーブル逆順ループ（execute_onexit_table inline）
+0x2035470:  mov rbx,[r14+rdi*8] / lea rcx,[img+0x2829e84] / call +0x2040d60   ← entry 実行
```

**特徴づけ**:
- free caller（+0x2035567 の呼び出し側）は、**atexit テーブル（execute_onexit_table の inline copy）から呼ばれる静的 teardown handler 内**で、**cached global pointer（slot img+0x2829e70）を free した直後に null-out する**典型パターン。
- `free` は **IAT 風 indirect slot（img+0x2220948）経由の call** → ucrtbase!free_base へ到達し、そこで c0000374 を raise。

## 4. atexit caller の分類（3-G-3）

| 分類 | 判定 |
| --- | --- |
| H1: harness 固有 static destruction | **可能** — `static BuzzLogger logger`（BassBuzzMeasurement.cpp:1540・juce::Logger 派生の function-local static・atexit に登録される）とその位置づけが一致 |
| H2: harness atexit 登録処理 | 部分一致（上記の静的 teardown wrapper） |
| H3: production object destruction | **可能** — production AudioEngine も同一テーブルに登録されるが、`~AudioEngine` は main 終了前（`[DIAG] ~AudioEngine` 行）に完了済みのため、**exit 時 atexit とは別フェーズ** |
| H4: JUCE/CRT/third-party | **強い候補** — free 経路が indirect IAT call・JUCE shutdown（`ScopedJuceInitialiser_GUI` :1542 + JUCE shutdownJuce_GUI 関連の static teardown）と整合 |
| H5: 未解決 | **function/line は未解決**（link PDB 欠落） |

**結論**: caller は **atexit 登録の静的 teardown（H1/H4 境界）**。実際の destructor 実体の line 特定は symbol なしでは不能。

## 5. Provenance / ownership census（3-G-4）

### 5-1. capture / tap lifetime
- Session は stack object・tap は `clearTap()` 済みで capture 後 runCapture を抜ける → crash run の SIGSEGV はこの経路の候補から後段に移動（exit 時点では既に Session 破棄済み）。

### 5-2. AudioEngine lifetime（両 run に共通して観測される shutdown 異常）
- os=1 smoke・no-os 対照 **両方**で:
  ```text
  [ISR][Shutdown] Drain incomplete: pendingPub=0 pendingRetire=0 crossfade=0 routerPendingRetire=2 … (observation only)
  [FAULT] ~AudioEngine: coordinator in Faulted state after markShutdownComplete
  ```
  - producer: `AudioEngine.Processing.ReleaseResources.cpp:704-724` / `AudioEngine.CtorDtor.cpp:306-311`。
  - `routerPendingRetire=2`（ISRRetireRouter 滞留 item 2 件・`RuntimeDrainAudit.h:30`）が shutdown 時に残存 → coordinator Faulted。
- **この shutdown 異常は OS=1 固有ではない**（no-os run も同じ出力・exit 127 = 異常終了）。→ **exit 時 heap corruption は OS=1 と無関係に共通 teardown 経路で発生している可能性が高い**。

### 5-3. IR swap / generation
- crash run: conv gen=9/IR gen=14/seq=8 — swap は完了していた。provenance として断定する証拠はまだない。

### 5-4. raw allocation/ownership census
- deferred-free 系: `src/DeferredFreeThread.h`・`src/RefCountedDeferred.h`（ISR bridge の lifetime 隔離）が shutdown 時の free 対象 block provenance の最有力候補領域。ただし「存在」だけでは defect としない（指示どおり）。

## 6. 判定

```text
Step 3-G = PARTIAL
heap corruption source candidate = shutdown/static-teardown path（H1/H4 境界・実体未解決）
root cause = UNKNOWN（block provenance 未特定）
P2 matrix = STOP
```

- symbol 解決は **link PDB 欠落により不能**（§2）→ source line 特定はできず。
- disassembly で確定できたのは「**atexit 内静的 teardown handler が cached global pointer を free した時点で corruption 検出**」まで。
- **追加の重要な観察**: exit 時異常（c0000374 相当）は **OS=1 と OS=2 の両 run で発生**（no-os run も exit 127）→ corruption は OS 条件に依存しない shutdown 経路の問題の可能性が高い。OS=1 capture SIGSEGV とは**別の独立した既存状態**の可能性が高い。

## 7. 次ステップ（承認待ち）

- **3-H**: ASan build（`build.bat Debug msvc -DENABLE_ASAN=ON` 相当）による corruption 発生位置の直接特定 — 候補が shutdown/teardown path に絞られたため、3-G census を前提に実行価値が上がった。
- または **link PDB を再生成する build**（同一 source・PDB only 再 link は code layout を変えるため attribution 無効 — 非推奨）。
- ISR-bridge shutdown drain（routerPendingRetire=2・coordinator Faulted）を別チケット化する判断。
