# D101-31-D 実施報告書 — AdmissionPackedState Defensive Hardening + Race Verification

- **実施日**: 2026-08-25
- **前提監査**: D101-31-C (CONDITIONAL PASS) の条件付き項目の解消
- **判定**: **PASS**（D1〜D10 全ゲート充足）
- **基準ドキュメント**: ConvoPeq.md 2026-08-25 09:16 再生成版（本作業内で再生成・修正反映済み @L58898）

---

## 1. 目的

D101-31-B の契約を一切変更せず、C で残った条件付き項目のみを解消する。

1. `closeAdmission()` の version overflow 修正
2. version wrap regression test 追加
3. `tryAdmit() ↔ closeAdmission()` の G-H race 専用テスト追加
4. release underflow / double-release 契約の明文化テスト
5. Debug / Release ビルド + 全 CTest 回帰

---

## 2. 変更内容

### 2.1 D-2: version overflow 修正（`src/audioengine/ISRShutdown.cpp`）

**問題**（C 監査指摘の通り）:

```cpp
// 修正前 (ISRShutdown.cpp:441-443)
const uint32_t desired = (static_cast<uint32_t>(AdmissionState::Closing)
                    | ((version + 1) << kVersionShift)   // ← version==63 で bit8 に漏出
                    | (count << kReservationShift));
```

`version == 63` のとき `64 << 2 = 0x100` となり、version 領域 `[2:7]` から
reservationCount 領域 `[8:31]` へビットが侵入し、outstanding() が +1 誤認識される。

**修正**:

```cpp
// 修正後 (ISRShutdown.cpp:439-447)
const uint32_t nextVersion = (version + 1u) & kVersionMask;
const uint32_t desired = (static_cast<uint32_t>(AdmissionState::Closing)
                    | (nextVersion << kVersionShift)
                    | (count << kReservationShift));
```

### 2.2 テスト用 seam（production authority 不変）

既存リポジトリパターン（`AudioEngine.h` の `DeferredPublicationTestAccess`、
`#if defined(CONVOPEQ_UNIT_TESTS)` + friend、Production バイナリ無変更）に完全準拠。

| ファイル | 変更 |
|---|---|
| `src/audioengine/ISRShutdown.h` | `#if defined(CONVOPEQ_UNIT_TESTS)` 下に `struct AdmissionPackedStateTestAccess;` 前方宣言 + クラス内 `friend struct convo::isr::AdmissionPackedStateTestAccess;`。Production ビルドでは両方ともコンパイルされずバイナリ無変更 |
| `src/tests/AdmissionPackedStateTestAccess.h` | **新規**。`load(const ShutdownRuntime&)` / `store(ShutdownRuntime&, uint32_t)` のみ。テスト setup 時（シングルスレッド）の version 注入専用 |
| `CMakeLists.txt` | `target_compile_definitions(AdmissionPackedStateTests PRIVATE CONVOPEQ_UNIT_TESTS=1)` を追加 |

seam 採用前の監査結果:
- public API 経由で version=63 へ到達不可（closeAdmission は Open→Closing のみで version bump、Closed→Open は INV-LIFE-9 により存在しない）
- ShutdownRuntime の再初期化 API / packedState_ の既存テストアクセス手段はいずれも不存在
- → production runtime API を追加しない friend seam が唯一かつ最小の選択

### 2.3 新規テスト3件（`src/tests/AdmissionPackedStateTests.cpp`）

#### Test 10: `testVersionWrapDoesNotCorruptReservationCount`（D-3）

```
packedState_ ← Open | version=63 | count=5 (=0x5FC)   [TestAccess で注入]
↓ closeAdmission()
厳密語比較: raw == Closing | version=0 | count=5 (=0x501)
  （修正前バグなら 0x601 — bit8 漏出で outstanding()==6 になる）
→ outstanding()==5 不変、release(5)、joinProducers()==true、Closed 到達
```

#### Test 11: `testConcurrentTryAdmitCloseAdmission`（D-4 / G-H race）

- 100 rounds × 構成: spinner 4 スレッド（`go` gate 後 `tryAdmit(1)/release(1)` ループ）
  + メインスレッドが `closeAdmission()` 実行
- **どちらが先に linearize するかを期待値固定しない**。Case A/B は情報出力のみ:

```
[info] race rounds: tryAdmit-first(Case A)=1, close-first(Case B)=99
```

- 各 round 共通の不変条件検証:
  - `outstanding() == 0`（exactly-one release）
  - closeAdmission 後の `tryAdmit(1)` は必ず false（post-close admission 不可能）
  - `joinProducers() == true` → `admissionState() == Closed`
  - resurrection なし
- Case A（tryAdmit CAS 先）: admit は全て Open 下で成立・解放済み → join 成功
- Case B（closeAdmission CAS 先）: 以降の tryAdmit 全敗 → join 成功
- **両方が成功して矛盾状態になることは単一 atomic word CAS により不可能** — D101-30 G-H linearization の executable evidence

#### Test 12: `testDoubleReleaseDoesNotUnderflow`（D-5）

- 契約の明文化（挙動変更なし）: `release(n)` where `count < n` → silent no-op（state/count 不変）
- `tryAdmit(1); release(1); release(1)` → `outstanding()==0`（`0x00FFFFFF` へ wrap しない）
- `tryAdmit(1); release(2)` → `outstanding()==1` 不変
- 1000 回過剰 release ストレス → clamp 維持、その後 close/join 正常動作（FSM 無傷）

---

## 3. コード監査結果（D-4 / D-5 / D-8 関連の再確認）

| Invariant | 確認結果 |
|---|---|
| `AdmissionState != Open ⇒ tryAdmit() == false` | ✅ ISRShutdown.cpp tryAdmit 冒頭 state check（Open 以外は即 return false） |
| `joinProducers() == true ⇒ Closing ∧ count==0 → Closed` | ✅ ISRShutdown.cpp:477 付近。count!=0 は false（caller retry 契約） |
| Q1 Proof sealing: `admissionState()==Closed` 必須 | ✅ tryMakeQuiescenceProof 内 q1 条件（変更なし） |
| Retire 経路に tryAdmit なし | ✅ 呼び出し箇所は正確に3: RebuildDispatch.cpp:319 (Recovery) / AudioEngine.h:4443 (Build) / RuntimePublicationOrchestrator.cpp:69 (Publication)。Retire はゼロ |

---

## 4. 検証結果

### 4.1 ユニットテスト（AdmissionPackedStateTests.exe 単体実行）

```
12 passed, 0 failed out of 12 tests
    [info] race rounds: tryAdmit-first(Case A)=1, close-first(Case B)=99
```

既存9テスト + 新規3テスト全 PASS。

### 4.2 ビルド

| Config | 結果 |
|---|---|
| Debug (MSVC, Ninja Multi-Config) | ✅ PASS（`[4/4] Checking build artifacts...` 完了。警告は既存分のみ） |
| Release | ✅ PASS（同上、error 0） |

### 4.3 CTest（Debug, full）

```
100% tests passed out of 38
```

- **CTest 実数は 38/38 のまま**（ユーザー指示通り 39 に固定せず実数を記録）。
  新規3テストは既存 ctest エントリ `#21 AdmissionPackedState` の内部 test 数を
  9→12 に増やす形のため、ctest エントリ総数は不変。
- `#21 AdmissionPackedState` 単体再実行: 内部 12/12 PASS 確認。

---

## 5. Gate 判定一覧

| Gate | 条件 | 判定 |
|---|---|---|
| D1 | version overflow が 6bit 内に収まる | ✅ `(version+1u) & kVersionMask` |
| D2 | version wrap で reservationCount を破壊しない | ✅ 厳密語比較テスト (0x501) |
| D3 | tryAdmit ↔ closeAdmission race test PASS | ✅ 100 rounds PASS |
| D4 | linearization が単一 CAS word で維持 | ✅ post-close tryAdmit 全敗を全 round で確認 |
| D5 | release underflow で state/count 破壊なし | ✅ clamp 契約テスト |
| D6 | exactly-one release 維持 | ✅ race test で outstanding==0 を全 round 確認 |
| D7 | Q0 → Proof sealing 維持 | ✅ joinProducers/Q1 条件コード監査 |
| D8 | Retire に tryAdmit なし | ✅ 呼び出し3経路のみ |
| D9 | Debug/Release build PASS | ✅ 両 config |
| D10 | 全 CTest PASS | ✅ 38/38（実数記録） |

---

## 6. 制約遵守確認（変更禁止項目）

以下は一切変更していないことを確認:

- `AdmissionReservation` の責務範囲 ✅
- Publication / Recovery / Build の 3-path ✅
- Retire への `tryAdmit()` 追加なし ✅
- `packedState_` の authority（layout・単一CAS word 契約）✅
- `tryAdmit → enqueue → release` の reservation lifetime ✅
- Q0 の `outstanding()` ✅
- `joinProducers()` の shutdown semantics ✅
- `CoordinatorState` / `pendingIntentCount_` / `publicationIntentResidencyCount_` / `retireBacklogCount_` ✅

本タスクは **hardening + executable evidence の追加のみ**（契約変更なし）。

---

## 7. 補足・運用メモ

- **ビルド環境要件**: `build.bat` / `ctest` の前に
  `vcvarsall.bat x64` + Intel oneAPI `setvars.bat intel64` が必要
  （素のコマンドプロンプトでは cl / mkl.h が見えず C1083）。`run_ctest.bat` と同一の手順。
- **ConvoPeq.md**: 本作業中に `output_sourcecode_markdown.py` で再生成
  （2026-08-25 09:16 版）。修正は L58898 付近に反映済み。
  今後の D 系判断は本版を基準にすること。
- **clangd 診断について**: 編集時に LSP が `AtomicAccess.h not found` /
  `JuceHeader.h not found` 等を報告するが、これは compile_commands.json 未解決による
  既知の clangd ノイズであり、MSVC/CMake 実ビルドには影響しない（今回も実ビルドで確認）。

---

## 8. 次のステップ

D101-31-C の CONDITIONAL PASS は本報告書をもって解消。
次の大きな Admission/Shutdown 設計変更へ進行可能：

- Phase B2: external setter elimination plan（15 site / 5 file の除去順序設計）
- §1.8 / Blocking-8 Phase D: BuildResult → RetryDisposition wiring
- §2.2.1.2 Path B gate: enqueuePublicationIntent への ShuttingDown gate 追加検討
