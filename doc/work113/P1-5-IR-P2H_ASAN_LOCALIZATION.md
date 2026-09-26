# P1-5-IR-P2-H — ASan heap corruption localization report

- **作成**: 2026-09-22 / work113 P1-5-IR Phase 2-H
- **性質**: read-only localization（production/test 変更 0・ASan build は既存 mechanism + 専用 cache `build-asan` で実施）
- **条件**: OS=1 smoke 条件そのまま（quiet 既定・settle 無変更）
- **ASan exe**: `build-asan/RelWithDebInfo/AudioEngineHarness.exe`（`4f1af5d6b940c64d`・74MB ASan 計装済み）
- **ASan runtime**: `VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\clang_rt.asan_dbg_dynamic`（MSVC 同梱 DLL・LLVM 配下 DLL は非互換）
- **raw log**: `tmp/p1_5_ir_p2_raw/asan_smoke_os1_20260922.log`（exit 1）

---

## 1. ASan が捕捉した error 全文（要旨）

```text
==14204==ERROR: AddressSanitizer: attempting double-free on 0x12e7c6ea1160 in thread T0:
  access stack:
    #0 free (asan_malloc_win.cpp:705)
    #1 juce::ArrayBase<juce::DeletedAtShutdown*, DummyCriticalSection>::setAllocatedSize(int)
       JUCE/modules/juce_core/containers/juce_ArrayBase.h:234
    #2 juce::Array<juce::DeletedAtShutdown*,…>::clear
    #3 juce::DeletedAtShutdown::deleteAll(void)      juce_DeletedAtShutdown.cpp:96
    #4 juce::shutdownJuce_GUI(void)                  juce_MessageManager.cpp:469
    #5 juce::ScopedJuceInitialiser_GUI::~ScopedJuceInitialiser_GUI()  :477
    #6 'runBassBuzzMeasurement'::'2'::dynamic atexit destructor for 'juceInit'
    #7-12 ucrtbase!execute_onexit_table（atexit テーブル）
    #13 __scrt_common_main_seh exe_common.inl:295

  freed by thread T0 here:   ← 1 回目の free
    #1 juce::HeapBlock<juce::DeletedAtShutdown*,0>::{dtor}
    #2 juce::ArrayBase<juce::DeletedAtShutdown*, …>::~ArrayBase   juce_ArrayBase.h:71
    #3 juce::Array<juce::DeletedAtShutdown*, …>::{dtor}
    #4 'juce::getDeletedAtShutdownObjects'::'2'::dynamic atexit destructor for 'objects'
       （静的 objects 配列の atexit dtor）

  previously allocated by thread T0 here:
    realloc → ArrayBase::setAllocatedSize → ensureAllocatedSize → add
    ← juce::DeletedAtShutdown::DeletedAtShutdown  juce_DeletedAtShutdown.cpp:49
    ← juce::ShutdownDetector ctor ← SingletonHolder<ShutdownDetector>::get
    ← juce::Timer::TimerThread ← SharedResourcePointer ← juce::Timer::Timer
    ← AudioEngine::AudioEngine(void)  src/audioengine/AudioEngine.h:2145
    ← AudioEngineHarness::AudioEngineHarness  AudioEngineHarness.cpp:11
    ← runBassBuzzMeasurement  BassBuzzMeasurement.cpp:1704
    ← main  PublishPipelineIntegrationTests.cpp:1127

SUMMARY: AddressSanitizer: double-free
  juce_ArrayBase.h:234 in ArrayBase<DeletedAtShutdown*, DummyCriticalSection>::setAllocatedSize(int)
==14204==ABORTING
```

## 2. 取得できた項目（3-H-3 の要求充足）

| 項目 | 値 |
| --- | --- |
| 1. error type | **double-free** |
| 2. invalid address | `0x12e7c6ea1160` |
| 3. access type | **FREE** |
| 4. size | 64-byte region（region 先頭 0 バイト内） |
| 5. source file | `JUCE/modules/juce_core/containers/juce_ArrayBase.h` |
| 6. source line | **:234**（`setAllocatedSize` の delete 経路） |
| 7. allocation stack | 取得済み（`DeletedAtShutdown::DeletedAtShutdown` ← `AudioEngine ctor` ← `runBassBuzzMeasurement:1704`） |
| 8. deallocation stack | 取得済み（**'juce::getDeletedAtShutdownObjects'::objects' の atexit dtor** が先に free） |
| 9. access stack | 取得済み（`shutdownJuce_GUI → deleteAll` が 2 回目の free） |

**provenance の直接結合（3-G UNKNOWN → 解決）**:

```text
allocation:  juce::DeletedAtShutdown::DeletedAtShutdown (:49)  — AudioEngine ctor 経由で
             Timer/ShutdownDetector が objects[] 配列に登録
                   ↓
free #1:     atexit dtor of 'objects'（getDeletedAtShutdownObjects の静的 Array）
             — 静的 ArrayBase が exit 時に dtor で block を解放
                   ↓
free #2:     DeletedAtShutdown::deleteAll()（ScopedJuceInitialiser_GUI dtor →
             shutdownJuce_GUI → deleteAll → Array::clear → setAllocatedSize(:234)）
                   ↓
= double-free 検出（同一 block の 2 度 free）
```

## 3. 3-G の PARTIAL 結果との照合

- 3-G の「検出地点 = atexit 内静的 teardown / cached global pointer free」は、**この double-free の free #1（atexit dtor of 'objects'）と free #2（ScopedJuceInitialiser_GUI → deleteAll）の順序逆転**で説明がつく: 静的 `objects` Array の atexit dtor が先に block を解放し、その後 `juceInit` の dtor → `shutdownJuce_GUI` → `deleteAll` が既に解放済みの block をもう一度 free しようとした。
- **構造的因果**: `runBassBuzzMeasurement` が **function-local `static juce::ScopedJuceInitialiser_GUI juceInit`**（:1542）を持つ一方、JUCE の DeletedAtShutdown objects 配列は **namespace-scope の atexit dtor** で解放される。両者の **atexit 破棄順序が逆転**（objects の atexit dtor が juceInit の dtor より先に走った）ことが直接の double-free 経路。
- この anomaly は **OS 条件に依存しない共通 teardown 経路**（3-G §5-2 の観察と整合）。
- production DSP（Convolver/RuntimeWorld）・IR swap・Session/tap capture は stack に一切出現していない。

## 4. 判定

```text
Step 3-H = PASS
heap corruption origin = identified（double-free・JUCE DeletedAtShutdown 静的配列の
  atexit dtor と ScopedJuceInitialiser_GUI dtor の破棄順序逆転）
P2 matrix = STOP（修正はまだしない）
```

- classification: **H-B(=3-H-4 の shutdown 関連)ではなく H-A(= heap corruption evidence)** として成立。ただし stack に RetireRouter/DeferredFree は出現していないため、`routerPendingRetire=2`（H-B チケット候補）とは**切り離して別チケット**とする（分離の根拠 = ASan stack が両者を結ばないため）。
- 3-H-5 の「何も検出しない」経路は採用されず。

## 5. 3-G-3 分類への照合（旧 H1〜H5 → 本 ASan 結果）

- 3-G で H1/H4 境界と推定した caller は、ASan stack の #6/#4（`juceInit` dtor と `objects` dtor の dynamic atexit destructor）で確定: **H4(JUCE/CRT static teardown) が主経路**。3-G の disassembly 特徴づけ（cached slot free → null-out）と ASan stack は整合する。

---

## 附録

- ASan raw log: `tmp/p1_5_ir_p2_raw/asan_smoke_os1_20260922.log`（error block :762-827）
- ASan build: `build-asan/RelWithDebInfo/AudioEngineHarness.exe`（SHA[:16] `4f1af5d6b940c64d`・専用 cache・既存 ENABLE_ASAN mechanism）
- runtime DLL: `VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\clang_rt.asan_dbg_dynamic-x86_64.dll`（MSVC 標準配置・LLVM 配下 DLL は非互換）
