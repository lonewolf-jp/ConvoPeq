# bug.md 改修計画書（v6 — 最終決定版）

**策定日**: 2026-07-23
**ベース**: `doc\work82\bug-verification-report.md`（再検証版）+ ソースコード照合 + Intel SDM CPUID 確認

---

## 0. v5→v6 修正点（本検証で発見した重大な誤り）

| 項目 | v5 | v6 修正 | 理由 |
|------|-----|---------|------|
| **CVMD-005** | `modeId=0` を `setSelectedId(0)` | **危険 — 撤回** | JUCE ComboBox の Item ID は 1〜4 のみ。`setSelectedId(0)` で ComboBox が空表示になる。1〜4 を維持し Bypassに ID=5 を新設 |
| **H-5 FMA 位置** | 修正案内で leaf 7 EBX[12] を FMA と誤記 | **leaf 1 ECX[12]** に修正 | Intel SDM: CPUID leaf 7 EBX[12] = RDT-M。FMA3 は leaf 1 ECX[12] |
| **H-5 既存コードのバグ発見** | なし（未認識） | **既存コードに FMA 位置の誤り** | `CpuFeatureCheck.cpp:44` が leaf 7 EBX[12] を FMA と誤認 |

---

## 1. P0: 即時修正（2件）

### 1-1. CVMD-005: 完全 bypass 時の UI 表示誤り

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/MainWindow.cpp:1190-1197`（表示）、`src/MainWindow.cpp:1261-1264`（ComboBox 項目定義）、`src/MainWindow.cpp:1349-1373`（選択処理） |
| **ソース確認** | ✅ `eqBypassed && convBypassed` 時に `modeId = 3`（Conv->Peq）のまま |

**重要な制約**: `orderModeBox` は JUCE ComboBox。既存の Item ID は 1〜4 で、`setSelectedId(0)` は「選択なし」（空表示）となる。Bypass 表示には **Item ID=5** を新設する。これにより既存の 1〜4 の構造を維持しつつ、`orderModeBoxChanged()` での mode 5 ハンドリングも追加。

**同期調査結果（ソース確認済み）**: MainWindow の `orderModeBox` は唯一的な入力手段。DeviceSettings には ComboBox がなく、`updateGainStagingDisplay()`（`:658`）で AudioEngine の状態を読み取りラベル表示するのみ。`DeviceSettings.cpp:669` で既に `convBypassed && eqBypassed` → `"Bypass"` を正しく処理。MainWindow の ID=5 追加は DeviceSettings の表示ロジックに影響しない。

**UI 仕様の確認**: `orderModeBox` の tooltip は既に `"Processing mode"`（`:1267`）。Bypass を選択肢に追加しても tooltip と矛盾しない。mode==5 選択時は `setProcessingOrder()` を呼ばず bypass フラグのみ設定するため、**以前の ProcessingOrder は保持される**。Bypass 解除後は元の順序に戻る。

**実装時の注意**: ID=5 追加時に、プロジェクト全体で `orderModeBox.getSelectedId()` の参照の参照箇所を検索し、ID を 1〜4 のみと仮定している箇所がないことを確認する。

**修正案（ComboBox 項目追加）**:
```cpp
// MainWindow.cpp:1261-1264 — ComboBox 項目定義に ID=5 "Bypass" を追加
orderModeBox.addItem("Conv", 1);
orderModeBox.addItem("Peq", 2);
orderModeBox.addItem("Conv->Peq", 3);
orderModeBox.addItem("Peq->Conv", 4);
orderModeBox.addItem("Bypass", 5);   // ★ 新規追加
```

**修正案（表示ロジック — MainWindow.cpp:1190-1198）**:
```cpp
int modeId;
if (eqBypassed && convBypassed)
    modeId = 5; // Bypass（新設）
else if (!eqBypassed && convBypassed)
    modeId = 2; // Peq
else if (eqBypassed && !convBypassed)
    modeId = 1; // Conv
else if (!eqBypassed && !convBypassed
      && audioEngine.getProcessingOrder() == AudioEngine::ProcessingOrder::EQThenConvolver)
    modeId = 4; // Peq->Conv
else
    modeId = 3; // Conv->Peq
orderModeBox.setSelectedId(modeId, juce::dontSendNotification);
```

**修正案（選択処理 — MainWindow.cpp:1349-1373 に mode 5 追加）**:
```cpp
void MainWindow::orderModeBoxChanged()
{
    const int mode = orderModeBox.getSelectedId();
    switch (mode)
    {
        case 1: /* Conv */
            audioEngine.setConvolverBypassRequested(false);
            audioEngine.setEqBypassRequested(true);
            break;
        case 2: /* Peq */
            audioEngine.setConvolverBypassRequested(true);
            audioEngine.setEqBypassRequested(false);
            break;
        case 3: /* Conv->Peq */
            audioEngine.setProcessingOrder(AudioEngine::ProcessingOrder::ConvolverThenEQ);
            audioEngine.setConvolverBypassRequested(false);
            audioEngine.setEqBypassRequested(false);
            break;
        case 4: /* Peq->Conv */
            audioEngine.setProcessingOrder(AudioEngine::ProcessingOrder::EQThenConvolver);
            audioEngine.setConvolverBypassRequested(false);
            audioEngine.setEqBypassRequested(false);
            break;
        case 5: /* Bypass */
            audioEngine.setConvolverBypassRequested(true);
            audioEngine.setEqBypassRequested(true);
            break;
        default:
            jassertfalse; /* 異常 ID 検出 */
            break;
    }
```

---

### 1-2. H-5: CpuFeatureCheck の FMA ビット位置誤り + XSAVE チェック

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/CpuFeatureCheck.cpp:38-49` |

**既存コードのバグ（本検証で発見）**: `CpuFeatureCheck.cpp:44` で `constexpr int kFMABit = 12;` を leaf 7 EBX に対してチェックしているが、Intel SDM によると:
- **leaf 7, EBX bit 12 = RDT-M**（Resource Director Technology Monitoring）
- **leaf 1, ECX bit 12 = FMA3**

**影響**: この誤りは現在動作している（RDT-M は AVX2 世代以降の CPU でほぼ常に有効）が、将来の CPU や VM 環境で FMA のみ有効で RDT-M が無効の場合に誤って false を返し、アプリケーションが起動できない。

**修正案（Method 2 の完全な書き直し — AVX 実行可否確認）**:
```cpp
// Method 2: AVX2 実行可否確認（MaxLeaf + OSXSAVE + AVX + FMA + AVX2 + XGETBV）
{
    // Step 0: CPUID(0) で最大 leaf を確認
    // CPUID leaf 7 は最大 leaf >= 7 の CPU でのみ有効。
    // 現代 CPU では不要に近いが、フォールバック実装として堅牢性を高める。
    int leaf0[4] = { 0 };
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    __cpuid(leaf0, 0);
#elif defined(__GNUC__) || defined(__clang__)
    __get_cpuid(0, &leaf0[0], &leaf0[1], &leaf0[2], &leaf0[3]);
#endif
    if (static_cast<unsigned>(leaf0[0]) < 7u)
        return false;  // leaf 7 未対応 → AVX2 不可

    // Step 1: leaf 1 で OSXSAVE + AVX + FMA を確認
    int leaf1[4] = { 0 };
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    __cpuid(leaf1, 1);
#elif defined(__GNUC__) || defined(__clang__)
    __get_cpuid(1, &leaf1[0], &leaf1[1], &leaf1[2], &leaf1[3]);
#endif
    // bit 27: OSXSAVE — XGETBV 発行前に必須（なければ #UD）
    constexpr int kOSXSAVEBit = 27;
    if ((leaf1[2] & (1u << kOSXSAVEBit)) == 0)
        return false;
    // bit 28: AVX — AVX 命令使用に必須（Intel SDM: CPUID.1:ECX[28]）
    constexpr int kAVXBit = 28;
    if ((leaf1[2] & (1u << kAVXBit)) == 0)
        return false;
    // bit 12: FMA3（★ leaf 1 ECX、leaf 7 EBX ではない）
    constexpr int kFMABit = 12;
    if ((leaf1[2] & (1u << kFMABit)) == 0)
        return false;

    // Step 2: leaf 7 で AVX2 を確認
    int leaf7[4] = { 0 };
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    __cpuidex(leaf7, 7, 0);
#else
    __cpuid_count(7, 0, leaf7[0], leaf7[1], leaf7[2], leaf7[3]);
#endif
    constexpr int kAVX2Bit = 5;
    if ((leaf7[1] & (1u << kAVX2Bit)) == 0)
        return false;

    // Step 3: XGETBV で OS の YMM 保存を確認
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    const unsigned int xcr0 = static_cast<unsigned int>(_xgetbv(0));
#else
    unsigned int eax = 0, edx = 0;
    __asm__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    const unsigned int xcr0 = eax;
#endif
    // XCR0[1] = XMM enabled, XCR0[2] = YMM enabled
    if ((xcr0 & 0x6u) != 0x6u)
        return false;

    return true;
}
```

**#else ブランチの修正（`:58-60`）**:
```cpp
#else
    // 未対応コンパイラ: AVX2 を使用できないと判断して安全側に倒す
    return false;
#endif
```

**注意**: Method 1（`IsProcessorFeaturePresent(PF_AVX2)`、Windows 8.1+）は変更不要。

**補足（GCC/Clang）**: Method 2 の `__get_cpuid` / `__cpuid_count` は `<cpuid.h>` ヘッダが必要（GCC/Clang のみ）。MSVC では `<intrin.h>` で利用可能。ConvoPeq は Windows/MSVC 専用のため実害はないが、コード例として残す場合は注記推奨。

---

## 2. P1: 次回リリース前（3件）

### 2-1. CVMD-007: cachedTailLength のスレッド安全性

（v5 から変更なし）

```cpp
std::atomic<double> cachedTailLength { 0.0 };
return convo::consumeAtomic(cachedTailLength, std::memory_order_relaxed);
convo::publishAtomic(cachedTailLength, newValue, std::memory_order_relaxed);
```

### 2-2. M-3: DeferredRetireFallbackQueue の totalPushCount_ 未実装

（v5 から変更なし）

### 2-3. Bug5: prepareSingleStage の noexcept 見直し

**修正内容（ユーザー修正済み）**:
- `prepareStage()` を `void` → `bool` に変更し、内部で `makeAlignedArray_nothrow` を使用
- `prepareSingleStage` で `prepareStage` の戻り値を確認し、失敗時は `return false`
- 全 `makeAlignedArray` を `makeAlignedArray_nothrow` に統一

**設計意図**: `noexcept` 関数内で `std::bad_alloc` が発生しても `std::terminate` にならないよう、全確保経路を non-throw に統一。

---

## 3. P2: リファクタリング（3件）

### 3-1. Bug4: lastError のスレッド安全性

**修正内容（ユーザー修正済み）**:
- `ConvolverProcessor.h` に `lastErrorMutex`, `setLastError()`, `clearLastError()` を追加
- `LoadPipeline.cpp` の `lastError` 操作を `setLastError()` / `clearLastError()` に統一

**重要な注意**: `IncrementalRebuildJob::lastError` は `ConvolverProcessor::lastError` と**完全に別物**です。`Rebuild.cpp` での `job.lastError` 操作は mutex 不要（RebuildThread 専用）のためそのまま。

### 3-2. Bug12: progressCallback のマーシャリング

（v5 から変更なし。ただし「必須条件」を「API契約」に修正）

**API契約**: `progressCallback` を受け取る側（呼び出し側）は、callback 内で GUI Component を参照する場合、**必ず `juce::Component::SafePointer` または同等の寿命管理**を使用すること。AllpassDesigner 側からは強制できないため、これは API 利用者への契約である。

### 3-3. Bug14: CacheManager の一時ファイル残存

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/CacheManager.cpp:366-382` |
| **ソース確認** | ✅ `save()` 関数で `temp.createOutputStream()` 失敗時に `temp.deleteFile` なし、`temp.moveFileTo(file)` 失敗時も削除なし |

**修正案**:
```cpp
std::unique_ptr<juce::FileOutputStream> out(temp.createOutputStream());
if (!out) {
    temp.deleteFile();
    return;
}
// ...
if (!out->flush())
{
    // flush 失敗: データが不完全なため move しない
    out.reset();
    temp.deleteFile();
    return;
}
out.reset();
if (!temp.moveFileTo(file)) {
    temp.deleteFile();
}
```

---

## 4. 設計原則との整合性

全修正案で「RT スレッドは所有権を持たない」「RT 内で mutex/heap 禁止」「AtomicAccess ラッパー統一」の各原則を満たす。

### 変更履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| v1 | 2026-07-23 | 初版 |
| v2 | 2026-07-23 | ユーザー評価反映（M-2/M-1 削除、Bug4/Bug12 見直し） |
| v3 | 2026-07-23 | ソースコード照合で確定（H-5 Method 2 限定、M-3 確認） |
| v4 | 2026-07-23 | ユーザー評価 93→97 反映（enum class, Dispatcher, AtomicAccess） |
| v5 | 2026-07-23 | v4 誤り訂正（Bug4 enum 撤回、Bug12 過剰設計撤回） |
| **v6** | **2026-07-23** | **CVMD-005 modeId=0 の JUCE ComboBox 不具合訂正 + H-5 FMA ビット位置誤り訂正・既存コードのバグ発見** |
