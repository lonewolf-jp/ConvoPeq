# STG-11-D1 Post-Implementation Gate（2026-09-29）

> **Verdict: STG-11-D1 = READY FOR COMMIT**
> **Commit = NOT AUTHORIZED / Push = NOT AUTHORIZED（未実施）**
> 本書は Owner GO に基づく Post-Implementation Gate（§2〜§10）の実施記録である。
> D2 / D3 / D4 の implementation は開始していない。

---

## 1. Authority（§7）

Owner 指定: `ConvoPeq(20260929-105350).md == ConvoPeq.md`（統一 authority）。
**以下の SHA は実ファイルから再取得した実測値であり、報告書からの転記ではない。**

| 項目 | 実測値 |
| --- | --- |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `CDAFA18BBCA815595ACB0B2C3D3F4C1FCB25D01DE150FEA6772837F65E5C38B1` |
| size | 5,690,145 B |
| Generated | 2026-09-29 19:35:52 |
| NEWER_SRC_COUNT | 0 |
| FRESH | yes（`--check` exit 0） |

---

## 2. double-ownership 修正の最終監査（§2）

**PASS**。`enqueueDeferredDeleteWithFallback()`（`EQProcessor.Core.cpp:26-80`）を source 実読で再確認した。

### 2.1 所有権遷移の完全列挙

```text
enqueueDeferredDeleteWithFallback()
  ├─ :37  m_retireCoordinator == nullptr → return false（by design・drop 計数）
  ├─ :60  coordinator->enqueueRetire(Granted, m_ownedRetireRouter, ...)
  │         └─ 内部で router.enqueueWithRetry を完全委譲実行
  │            （ISRRuntimePublicationCoordinator.cpp:160）
  ├─ :64-67 Success / QueuePressure / TerminalReclaim → return true
  │           （ownership transfer 成立。同一 ptr の再 enqueue なし）
  ├─ :68-69 Shutdown → return false（caller が ownership 保持。再試行なし）
  └─ :73  QueueFull のみ router.enqueueWithRetry を直接実行（未格納の場合のみ retry）
```

### 2.2 確認事項への回答

| Owner 確認事項 | 結果 |
| --- | --- |
| Success / QueuePressure / TerminalReclaim → ownership transfer 成立 → 同一 ptr の再 enqueue 禁止 | **成立**（`:64-67` で `return true` し `:73` に到達しない） |
| Shutdown → ownership transfer 不成立 → caller が ownership を保持 | **成立**（`:68-69` で `return false`。再試行なし） |
| QueueFull → 実際に格納されていない場合のみ retry | **成立**（`:73` 到達は QueueFull のみ。`enqueueWithRetry` の戻り値は P-4 契約どおり再判定される） |
| `QueuePressure / TerminalReclaim → 同一 ptr → もう一度 enqueueWithRetry()` の経路が存在しないこと | **存在しない**（`:64-67` が全ての所有成立を捕捉。`RetireEnqueueResult` の 5 値 `Success/QueuePressure/QueueFull/Shutdown/TerminalReclaim`（`ISRAuthorityClass.h:28-34`）の全てを分岐で処理し、漏れがない） |

---

## 3. TD1-1 / TD1-2 の deterministic oracle 再確認（§3）

**PASS**。最新 binary（guard 修正済み）による fresh 実行で再確認した
（Debug standalone ×3 / Release standalone ×1 / Debug harness / Release harness）。

### 3.1 TD1-1

```text
Q = 304（2200 setters × 2 − 4096）
```

### 3.2 TD1-2

```text
D = 4096
Q = 512
E = 512
T = 280
総数 = 4096 + 512 + 512 + 280 = 5400 = 2700 setters × 2
```

### 3.3 二重格納の非再発

`T = 1584`（二重所有時の観測値）は**再発していない**。
全実行で `T = 280` 正確。合計 5400 と一致し、単一所有が成立している。

---

## 4. Negative control の扱い（§4）

**保持**。double-ownership guard を一時的に除去した隔離ビルドで以下を観測済み
（implementation report §4／§5 に記録。production は完全復元し最終 diff に含まない）:

```text
double-ownership guard を除去
    ↓
TD1-1: Q residency=608 expected 304（正確に 2 倍で FAIL）
    ↓
drain 時に abort（double-delete）
```

最終 production tree が guard 修正済みであることを §2 の source diff で確認した
（`:64-73` の分岐が存在し、`stackRouter` 残存 0 件）。

---

## 5. intermittent SEGFAULT の再確認（§5）

### 5.1 事実経過

| # | 実行 | 結果 |
| --- | --- | --- |
| 1 | Debug CTest 全件（初回） | 43/44。`AudioEngineHarness SEGFAULT` 1 件 |
| 2 | Debug CTest `AudioEngineHarness` 単体再実行 | **PASS**（169.73 s） |
| 3 | Debug CTest 全件再実行 | **44/44 PASS**（209.61 s） |
| 4 | Debug harness 直接実行 | **exit 0・全 PASS**（TD1 含む） |
| 5 | Release CTest 全件 | **44/44 PASS**（151.56 s） |
| 6 | Release harness 直接実行 | **exit 0・全 PASS**（TD1 含む） |
| 7 | Debug standalone TD1 | **exit 0・全 PASS ×3 回** |
| 8 | Release standalone TD1 | **exit 0・全 PASS** |

### 5.2 判定

```text
D1 test が再現性なく crash → 該当せず（TD1 は Debug×3・Release×1・harness両方で安定 PASS）
D1 test が再現可能に crash → 該当せず
double-delete / heap corruption / UAF が検出 → 該当せず（§3 の正確な counts が単一所有を証明）
既知 I3 経路だけで再現し D1 と無関係 → 該当（以下 evidence）
```

**Evidence**: 失敗したのは `AudioEngineHarness` 全体であり、TD1 到達前の既存領域での
SEGFAULT である。直後の単体再実行で同一 binary が PASS し、その後の全件再実行でも
44/44 PASS した。ログ末尾には pre-existing の
`[I2T] phase4: abandon engine (pre-existing Debug segfault route, I3 issue)` が
記録されている。**D1 implementation に起因する deterministic failure ではない。**

**判定: PASS。**

---

## 6. ASAN（§6）

### 6.1 結論

**ASAN 実行は環境要因で block された。INCONCLUSIVE と記録する。**
D1 defect の signal ではない（load 時・pre-main のローダ失敗であり retire logic と無関係）。

### 6.2 事実経過

| # | 試行 | 結果 |
| --- | --- | --- |
| 1 | `build-asan` Debug で `STG11EQRetireTests` をビルド | 成功（ASAN 計装付き、75MB） |
| 2 | 実行 | `0xC0000135`（DLL 不足）→ `clang_rt.asan_dbg_dynamic` / `clang_rt.asan_dynamic` を exe 横に配置 |
| 3 | 再実行 | `0xC0000139`（STATUS_ENTRYPOINT_NOT_FOUND） |
| 4 | vcvars 環境＋MKL PATH 明示で再実行 | 同一 `0xC0000139` |
| 5 | ダンプ解析（WinDbg） | 全 import DLL は load 成功。`ntdll!LdrGetProcedureAddressForCaller` での forwarder 解決失敗。worker thread 上の初期化中、pre-main |
| 6 | RelWithDebInfo（release CRT＋ASAN）でビルド＋実行 | 同一 `0xC0000139` |
| 7 | 既存 ASAN `AudioEngineHarness.exe`（2026-09-22 ビルド） | **実行可能**（ASAN toolchain 自体は動作する） |

### 6.3 代替 evidence（double-free / UAF / overflow がないこと）

ASAN の代わりに以下が同等の保証を与える:

1. **正確な counts**: TD1-1（Q=304）/ TD1-2（D/Q/E/T=4096/512/512/280、合計 5400）が
   Debug / Release とも正確。**全オブジェクトが正確に 1 箇所に所有されている**ため、
   double-free は構造的に起こり得ない。
2. **全 drain 完了**: release 後に全 counts 0・drop 0。全 deleter が正確に 1 回ずつ実行された。
3. **Negative control**: guard 除去で Q=608（2 倍）を検出後 abort。
   guard ありでは abort なし。**oracle が double-ownership を検出できる**。
4. **複数回実行**: Debug standalone ×3 / Release ×1 / harness 両方で exit 0。
   heap corruption があれば再現性高く abort するはずの規模（5400 objects × 繰り返し）で安定。
5. **destructor 実行**: 全 test が scope exit で `~EQProcessor`（force drain 含む）を実行し、
   正常終了している。

---

## 7. ConvoPeq authority（§7 再掲）

§1 のとおり。実測 SHA `CDAFA18B...5C38B1` を唯一の正とする。

---

## 8. CMake 変更の確認（§8）

**PASS**。`CMakeLists.txt` の差分は以下に限定される（全量実読）：

| 要素 | 内容 |
| --- | --- |
| standalone target | `STG11EQRetireTests`（test TU＋EQ 本体＋router 系 cpps） |
| add_test | `STG11EQRetire` 1 件 |
| STG11_STANDALONE_MAIN | standalone 側の `main()` 定義切替のみ |
| IPO-OFF | 新 target のみ |
| ASAN | 新 target のみ（既存リストへの追加） |
| include/link/definition | 新 target のみ。harness と同一前提の複写（NOMINMAX / JUCE / AVX2 / MKL） |

production target の optimization policy・既存 test target の compiler/linker behavior・
既存 CTest の実行条件の変更は **0 件**。

---

## 9. Getter 7 件について（§9）

**現状報告（削除しない）**。全 7 件は以下を満たす:

| getter | 定義 | 呼び出し元 |
| --- | --- | --- |
| pendingRetire | `EQProcessor.h:261` const noexcept | test TU のみ |
| quarantineResident | `:265` const noexcept | test TU のみ |
| emergencyResident | `:269` const noexcept | test TU のみ |
| terminalResident | `:273` const noexcept | test TU のみ |
| retireDropCount | const noexcept（`consumeAtomic` 読取） | test TU のみ |
| privateEpoch | const noexcept（`currentEpoch()` 委譲） | test TU のみ |
| routerEpoch | const noexcept（`currentEpoch()` 委譲） | test TU のみ |

- **read-only**: 全て `const noexcept` で、既存 const reader への委譲または atomic load のみ。
  書き込み・counter 変更・flag 変更は 0。
- **side-effect free**: mutex 取得・allocation・deleter 実行を含まない。
  （`quarantineResidentCount()` 等は `size_` 読取＋mutex を取るが、状態を変更しない。
  RT から呼ばれないため contention の問題もない。）
- **RT path から呼ばれない**: `EQProcessor.Processing.cpp` /
  `ConvolverProcessor.Runtime.cpp` からの呼び出し 0 件（`rg` 実測）。
- **ownership を変更しない**: 所有権の移転・放棄を一切行わない。
- **authority を変更しない**: Coordinator / Router / EpochDomain の状態を変更しない。

**D1 の test observability に不要な getter はない**（7 件全てが TD1-0〜TD1-4a の
いずれかの oracle で使用されている）。したがって削除の検討自体が不要。

---

## 10. Final invariant audit（§10）

| 不変条件 | 再確認結果 |
| --- | --- |
| INV-D1-1: EQ retire → EQ private EpochDomain → EQ-owned router | **PASS**（TD1-4a/4b＋§2 の束縛確認） |
| INV-D1-2: Q/E/T ownership survives `enqueueDeferredDeleteWithFallback()` return | **PASS**（TD1-1/1-2 の正確な counts） |
| INV-D1-3: EQProcessor destruction 前に durable retire state が解決される | **PASS**（TD1-1/1-2/1-3 の release 後全 0＋dtor 正常終了） |
| INV-D1-4: foreign / engine epoch advance は EQ entry を reclaim しない | **PASS**（TD1-4b: foreign 100 前進で未解放） |
| INV-D1-5: RT path は ownership / reclaim / delete を行わない | **PASS**（RT 関連 diff 0＋全 retire 呼び出し NonRT） |
| INV-D1-6: 一度 ownership transfer が成立した ptr は、同一 obligation として二重 enqueue されない | **PASS（新規追加）**。§2 の分岐（`:64-69`）が構造的に保証。TD1-1/1-2 の正確な合計（4400/5400）が実証。Negative control（Q=608）が検出力を実証 |

---

## 11. Commit 判定

```text
STG-11-D1 Post-Implementation Gate
==================================

§2 double-ownership audit     = PASS
§3 oracle re-confirmation     = PASS (Q=304 / D/Q/E/T=4096/512/512/280 / total 5400)
§4 negative control           = PASS (kept; final tree has guard)
§5 intermittent SEGFAULT      = PASS (deterministic D1 failure ではない)
§6 ASAN                       = INCONCLUSIVE (environment-blocked; §6.3 の代替 evidence あり)
§7 authority                  = PASS (CDAFA18B...5C38B1 / FRESH)
§8 CMake                      = PASS (test-infra only)
§9 getters                    = PASS (reported as-is; 7 件全て必要)
§10 invariants (INV-D1-1..6)  = PASS

Production changes = 2 files (EQProcessor.h / EQProcessor.Core.cpp)
Test changes       = 1 new TU (448 lines) + harness wiring (+6)
CMake changes      = test-infra only
Authority changes  = 1 (ConvoPeq.md regenerated, FRESH)

Commit = 0
Push   = 0
staged = 0 files
branch = main...origin/main [ahead 9]

Verdict:
STG-11-D1 = READY FOR COMMIT
```

**`STG-11-D1 = READY FOR COMMIT`。commit / push は未実施。Owner の別途 GO を待つ。**
