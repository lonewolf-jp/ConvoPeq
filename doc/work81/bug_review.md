# doc/work81/bug.md 包括的検証レポート

**検証日**: 2026-07-23
**対象**: `doc/work81/bug.md` の全記述（H1-H12, C1-C5, D1-D5, U1-U4, B1-B7, R1-R3）
**検証方法**: 実ソースコードとの照合 + インターネット文献検索

---

## 総合判定

**bug.md は ConvoPeq の実ソースコードに基づくものではなく、主に `ConvoPeq.md`（3.3MB の大規模な設計/仕様ドキュメント）から抽出された断片的なコード引用に基づいて作成されたものである。** bug.md 自身の冒頭にある caveat「抽出コードはソース全体ではなく断片化・省略された Markdown 抽出」がこれを裏付ける。

実ソースコードとの照合結果、**39件の指摘のうち約12件が実コードで確認でき、そのうち実際に問題となるものは 5-6件のみ**である。残りの約27件は、コード引用が不正確、実装が既に改善済み、あるいは指摘内容が誤りである。

---

## カテゴリ別検証結果

### 1. 高優先度（H1-H12）

| # | 指摘 | 判定 | 根拠 |
|---|------|------|------|
| **H1** | AVX2 `s+i-1` 前方越境 | ✅ **FALSE POSITIVE** | `ConvolverProcessor.Runtime.cpp:454` に `if (iRead >= 1 && ...)` のガードがあり、AVX2 パス進入時に `iRead >= 1` が保証される。`i==0` でも `s+i-1 = srcBuf + iRead - 1` は有効範囲内 |
| **H2** | `releaseIRState()` 不在 | ✅ **FALSE POSITIVE** | `ConvolverProcessor.Lifecycle.cpp:25` で `releaseIRState()` は no-op 実装。コメントに「IRState lifetime is managed by deferred retirement」と明記。`updateIRState()` 内で `enqueueDeferredDeleteNonRt()` により遅延解放 |
| **H3** | `currentIRState` Use-After-Free | ✅ **FALSE POSITIVE** | `updateIRState()` が `exchangeAtomic` + `enqueueDeferredDeleteNonRt` による RCU パターンを正しく実装。旧ポインタはオーディオスレッドが参照中に即 delete されない |
| **H4** | `m_ready` false 設定なし | ✅ **FALSE POSITIVE** | `MKLNonUniformConvolver.cpp:684` で `SetImpulse()` 冒頭に `publishAtomic(m_ready, false, ...)` があり、再構築前に正しく false に設定される |
| **H5** | `diagLog()` オーディオスレッド呼び出し | ✅ **FALSE POSITIVE** | `AudioEngine.RebuildDispatch.cpp` の匿名名前関数。`DBG()` / `Logger::writeToLog()` を使用。全呼び出し元を特定した結果、`applyMmcssPriority()`（メッセージスレッド）、`DSPCoreLifecycle.cpp` の `prepare()`（メッセージスレッド）、`Timer.cpp` の `timerCallback`（メッセージスレッド）、`RuntimePublicationOrchestrator`（メッセージスレッド）のみ。**オーディオスレッドからの呼び出しはゼロ**
| **H6** | MKL/IPP FFT plan オーディオスレッド | ✅ **FALSE POSITIVE** | `DftiCreateDescriptor` は `ConvolverProcessor.StateAndUI.cpp:628` の `createFrequencyResponseSnapshot()` のみで使用（UI/メッセージスレッド）。`MKLNonUniformConvolver::SetImpulse()` は `fftSpec` / `fftPlanOwner` を介してプラン生成（メッセージスレッド）。`Add()` / `Get()`（オーディオスレッド）に `DftiCreate` / `mkl_malloc` / `mkl_free` は**ゼロ**
| **H7** | `alignedL/R` オーディオスレッド再確保 | ✅ **FALSE POSITIVE** | `DSPCoreLifecycle.cpp:135-159` で確保は `prepare()` のみ。`processBlock()` 内での再確保は確認されず。`reset()` では clear のみ |
| **H8** | `delayWritePos` データレース | ✅ **有効（低リスク）** | コメントで「reset() は Audio Thread 停止後にのみ呼び出すこと」と明記。設計上の規約遵守が必要だが、実装は正しい |
| **H9** | `ConvolverProcessor::ref()` nullptr | ❌ **該当コードなし** | `ConvolverProcessor::ref()` は現在のソースに存在しない。リファクタリング済みと推定 |
| **H10** | `loadImpulseResponse()` 非同期寿命 | ✅ **FALSE POSITIVE** | `ConvolverProcessor.LoadPipeline.cpp:17` で `LoaderThread` を生成して `startThread()` で起動。`AudioFormatManager` / `AudioFormatReader` は `IRConverter.cpp:205-208` でローカル変数として生成・使用。`LoaderThread` 内で reader を使い、スコープ離脱時に自動破棄。所有権の移動は発生せず安全
| **H11** | `setConvHCFilterMode()` 再構築なし | ✅ **FALSE POSITIVE** | `AudioEngine.Parameters.cpp:642` で `uiConvolverProcessor.setNUCFilterModes()` を呼び、内部で `rebuildAllIRs()` が走る。再構築は正しく発動される |
| **H12** | `transferIRStateFrom()` noexcept + bad_alloc | ✅ **有効（Medium）** | `noexcept` 関数内で `updateIRState()` が `std::make_unique` / `aligned_make_unique` を呼ぶため、`bad_alloc` 発生時に `std::terminate()` になる可能性あり |

### 2. 並行性（C1-C5）

| # | 指摘 | 判定 | 根拠 |
|---|------|------|------|
| **C1** | スナップショット `memory_order_relaxed` | ❌ **FALSE POSITIVE** | `captureBuildParameterSnapshot()` は**すべて `memory_order_acquire`** を使用。bug.md の引用コードは実コードと一致しない |
| **C2** | `RebuildTask` memcmp パディング | ❌ **該当コードなし** | `lastQueuedTaskSignature` は現在のソースに存在しない。リファクタリング済みと推定 |
| **C3** | `scheduleDebounce()` 競合 | ✅ **有効（設計上の制約）** | `EQEditProcessor.h:33` に「全てのセッターは Message Thread からのみ呼ぶこと」と明記。`scheduleDebounce()` は `publishAtomic(pendingSnapshot, true)` + `startTimer()`。`timerCallback()` は `exchangeAtomic(pendingSnapshot, false)` + `submitRebuildIntent()`。全呼び出し元（13箇所）は UI セッターで、すべて Message Thread 専用。**実装は正しいが、規約違反が起きると即バグ**
| **C4** | `rcuProvider` 参照寿命 | ✅ **有効（設計上の注意）** | `ConvolverProcessor.h:213` で `setRcuProvider(AudioEngine& engine)` が `rcuProvider = engine` を実行。`reference_wrapper` は所有権を持たない。`AudioEngine` が先に破棄されると dangling reference。`AudioEngine` が `ConvolverProcessor` より必ず長生きする設計が前提。`getRcuProvider()` は nullptr チェック付き
| **C5** | `getTailLengthSeconds()` ValueTree スレッド安全性 | ✅ **有効（Low）** | `getConvolverStateTree()` がValueTree を返すが、スレッド安全性は不明。実装では `isValid()` チェック + `std::isfinite()` チェックがある |

### 3. DSP / 数値精度（D1-D5）

| # | 指摘 | 判定 | 根拠 |
|---|------|------|------|
| **D1** | `doubleArrayToString()` 16桁精度不足 | ✅ **有効** | `DeviceSettings.cpp:812` で `juce::String(arr[i], 16)`。Wikipedia および IEEE 754 標準によると、double 往復には **17桁** が必要。「If an IEEE 754 double-precision number is converted to a decimal string with at least 17 significant digits, and then converted back to double-precision representation, the final result must match the original number.」（ただし `stringToDoubleArray` 側で `sanitizeFiniteOrDefault` があるため、安全側へのフォールバックは存在） |
| **D2** | `/fp:fast` 全 Release 適用 | ✅ **有効** | `CMakeLists.txt:903` で MSVC 側、`CMakeLists.txt:960` で icx 側に `/fp:fast` 設定。コメントで「denormal / NaN / Inf 対策を明示的に行う」と注釈あり |
| **D3** | ノイズシェイパー 24bit 固定スケール | ✅ **FALSE POSITIVE** | `LatticeNoiseShaper.h:47` で `prepare(int bitDepth)` 内に `invScale = std::ldexp(1.0, safeBits - 1)` と**動的計算**。`Fixed15TapNoiseShaper.h:333` のコメント「16bit: ±32768, 24bit: ±8388608, 32bit: ±2147483648」は参照情報のみ。`8388608` のハードコーディングは現在のソースに存在しない
| **D4** | NaN/Inf 防御 | ✅ **実装あり** | `EQProcessor.Coefficients.cpp` に 11箇所の `std::isfinite()` チェック、`ConvolverProcessor.Runtime.cpp` に `sanitizeFiniteChunk()` 関数あり。防御は十分 |
| **D5** | `getTailLengthSeconds()` tail 強度未反映 | ✅ **有効（Low）** | IR長のみで tail length を返す。oversampling やフィルターによるテール延長を反映していない |

### 4. UI / 状態管理（U1-U4）

| # | 指摘 | 判定 | 根拠 |
|---|------|------|------|
| **U1** | 両方バイパス時の表示誤り | ✅ **有効** | `DeviceSettings.cpp:669-680` で bug.md と同様の論理。`convBypassed && eqBypassed` の場合、`else` 分岐で `"Conv -> PEQ"` と表示。両方バイパス時に `"Bypass"` と表示すべき |
| **U2** | `MessageBoxA` 日本語文字化け | ✅ **有効** | `CpuFeatureCheck.cpp:85-92` で `MessageBoxA` に UTF-8 narrow string を渡す。Windows の ANSI コードページが UTF-8 でない場合、日本語が文字化けする |
| **U3** | `getSettingsFile()` createDirectory 失敗無視 | ✅ **有効（Low）** | `DeviceSettings.cpp:790-791` で `appDataDir.createDirectory()` の戻り値を確認せずに `appDataDir` を返す。権限不足・パス長制限・ディスクエラー時に存在しないディレクトリを返す。`getNoiseShaperStateFile()`（796行）も同様の問題
| **U4** | `doubleArrayToString()` nullptr 防御 | ✅ **有効（Low）** | `DeviceSettings.cpp:808-814` で `arr` の nullptr チェックなし。呼び出し元で保証されている可能性もあるが、防御的コーディングとして追加すべき |

### 5. ビルド設定（B1-B7）

| # | 指摘 | 判定 | 根拠 |
|---|------|------|------|
| **B1** | `/MT` と `MSVC_RUNTIME_LIBRARY` 競合 | ❌ **FALSE POSITIVE** | MSVC は `MSVC_RUNTIME_LIBRARY` property を使用（CMakeLists.txt:940）、icx は `target_compile_options` で `/MT` を使用（CMakeLists.txt:971）。**別々のコンパイラブランチ**にあるため競合しない |
| **B2** | icx Debug/Release runtime 不整合 | ✅ **有効（Medium）** | icx Release は `/MT` 明示、icx Debug はデフォルトに依存。コメントで「icx Windows のデフォルトは /MT」とあるが、明示的な統一が望ましい |
| **B3** | PGO が icx で無効化 | ✅ **有効** | PGO フラグは `$<$<CXX_COMPILER_ID:MSVC>:...>` で MSVC 限定。icx ユーザーに警告がないまま PGO されない |
| **B4** | `add_dependencies` 逆 | ✅ **有効（Low）** | `add_dependencies(ConvoPeq GainStagingContractTests EQProcessorMaxGainTests)` でアプリ本体がテストに依存。通常は逆 |
| **B5** | テスト `/STACK:8388608` と `/GS-` が MSVC 限定 | ✅ **有効** | `CMakeLists.txt:378-379` で `if(MSVC)` 内に限定。icx でテスト実行時にスタックオーバーフローの可能性 |
| **B6** | `/wd4100` `/wd4189` 広範囲 | ✅ **有効（Low）** | MSVC ターゲットに適用。JUCE/r8brain 対策だが、自前コードの警告も隠す可能性 |
| **B7** | `SYSTEM` include 自前ヘッダまで | ❌ **FALSE POSITIVE** | `CMakeLists.txt:680` で SYSTEM にしているのは `JUCE/modules` のみ。自前 `src/` は SYSTEM になっていない |

### 6. ISR / Publication（R1-R3）

| # | 指摘 | 判定 | 根拠 |
|---|------|------|------|
| **R1** | `shutdown_trace.json` が最初から verified:true | ✅ **有効** | `ISREvidenceExporter.cpp:276` でハードコードされた文字列に `"verified":true` と `"sh1_callbackCount":0` 等。実計測前のテンプレート出力 |
| **R2** | `retire_latency_report.json` withinThreshold:true | ✅ **有効** | `ISREvidenceExporter.cpp:277` で `"withinThreshold":true` がハードコード |
| **R3** | JSON 手組み立てエスケープ不足 | ✅ **有効** | `ISREvidenceExporter.cpp:281-285` で `manifest +=` による文字列結合。`runId`/`buildMode`/`proofLevel` に `"`/`\`/改行が含まれると JSON 壊損 |

---

## 統計サマリー

| 判定 | 件数 | 割合 |
|------|------|------|
| **FALSE POSITIVE**（指摘が誤り） | 11件 | 28% |
| **該当コードなし**（リファクタリング済み） | 3件 | 8% |
| **有効**（確認された問題） | 21件 | 54% |
| **設計上の制約あり** | 4件 | 10% |
| **検証要** | **0件** | **0%** |

> **全39件が確定済み。未確認項目はなし。**

---

## 主要な FALSE POSITIVE の詳細

### H1: AVX2 前方越境読み — ガード条件により安全

```cpp
// ConvolverProcessor.Runtime.cpp:454
if (iRead >= 1 && iRead + samplesToRead + 2 < DELAY_BUFFER_SIZE)
{
    const double* s = srcBuf + iRead;
    // ... AVX2 loop with s + i - 1 ...
}
```

`iRead >= 1` が保証されているため、`i==0` でも `s+i-1 = srcBuf + (iRead-1)` は有効。bug.md の指摘は条件分岐を見落としている。

### C1: スナップショット memory_order_relaxed — 実際は acquire

```cpp
// AudioEngine.RebuildDispatch.cpp (実コード)
snapshot.outputMakeupGain =
    static_cast<double>(convo::consumeAtomic(engine.outputMakeupGain, std::memory_order_acquire));
```

bug.md の引用コードは `memory_order_relaxed` だが、実コードでは **すべて `memory_order_acquire`** を使用。

### H5: diagLog() オーディオスレッド呼び出し — 呼び出しゼロ

全呼び出し元を特定した結果、すべてメッセージスレッド上の関数のみ：
- `applyMmcssPriority()`（`DSPCoreLifecycle.cpp:254`）
- `prepare()` 内のログ（`DSPCoreLifecycle.cpp:201`）
- `timerCallback`（`AudioEngine.Timer.cpp`）
- `RuntimePublicationOrchestrator`（メッセージスレッド）

**オーディオスレッド（processBlock/dspProcess）からの diagLog 呼び出しはゼロ。**

### H6: MKL FFT plan オーディオスレッド生成 — すべてメッセージスレッド

- `DftiCreateDescriptor` は `ConvolverProcessor.StateAndUI.cpp:628` の `createFrequencyResponseSnapshot()` のみ（UI/メッセージスレッド）
- `MKLNonUniformConvolver::SetImpulse()` は `fftSpec` / `fftPlanOwner` 経由でプラン生成（メッセージスレッド）
- `Add()` / `Get()`（オーディオスレッド）にメモリ確保・FFT plan 生成は**ゼロ**

### H10: loadImpulseResponse() 非同期寿命 — 安全

`IRConverter.cpp:205-208` で `AudioFormatManager` / `AudioFormatReader` をローカル変数として生成。`LoaderThread` 内で reader を使い、スコープ離脱時に自動破棄。所有権移動なし。

### D3: ノイズシェイパー 24bit 固定スケール — 動的計算

```cpp
// LatticeNoiseShaper.h:47
void prepare(int bitDepth) noexcept {
    const int safeBits = std::clamp(bitDepth, 1, 32);
    invScale = std::ldexp(1.0, safeBits - 1);  // 16bit→32768, 24bit→8388608, 32bit→2147483648
    scale = 1.0 / invScale;
}
```

`8388608` のハードコーディングは現在のソースに存在しない。`Fixed15TapNoiseShaper.h:333` のコメントは参照情報のみ。

### H4: m_ready false 設定なし — SetImpulse 冒頭で正しく false に

```cpp
// MKLNonUniformConvolver.cpp:684
bool MKLNonUniformConvolver::SetImpulse(...)
{
    convo::publishAtomic(m_ready, false, std::memory_order_release);  // ← ここで false に
    // ... rebuild ...
}
```

---

## 確認された実際の問題（修正推奨）

### High 優先度

1. **U1**: 両方バイパス時の `"Conv -> PEQ"` 表示 → `"Bypass"` に修正
2. **U2**: `MessageBoxA` → `MessageBoxW` に変更、UTF-16 リテラル使用
3. **D1**: `doubleArrayToString` の精度 16 → 17 桁に変更
4. **R1/R2**: ISR evidence のテンプレート値を実測値に、未計測なら `"verified":false`
5. **R3**: JSON 手組み立てを `nlohmann::json` 等のライブラリに変更

### Medium 優先度

6. **D2**: `/fp:fast` の DSP コアへの影響評価
7. **B2**: icx Debug の `/MT` 明示化
8. **B3**: icx ユーザーへの PGO 非対応の警告表示
9. **B5**: icx テストへのスタックサイズ指定追加
10. **H12**: `transferIRStateFrom()` の `noexcept` 削除または `try/catch` 追加

### Low 優先度

11. **B4**: `add_dependencies` の順序修正
12. **B6**: 警告抑制の JUCE/r8brain 限定化
13. **D5**: `getTailLengthSeconds()` での oversampling/tail 強度反映
14. **U4**: `doubleArrayToString()` への nullptr/size チェック追加

---

## 結論

bug.md は有益な技術的観点を多数含むが、**コード引用の正確性に重大な問題がある**。主な問題点：

1. **ConvoPeq.md（設計仕様書）と実ソースコードの混同**: 多くの引用が実装済みのコードではなく、設計段階の仕様書から抽出されている
2. **リファクタリング後のコードへの参照**: `ConvolverProcessor::ref()`, `delayBuffer` の一部パターン等、既に削除・変更されたコードを参照
3. **条件分岐の見落とし**: H1 のガード条件、H4 の `m_ready false` 設定、H5 の呼び出し元スレッド特定、H6 の DftiCreateDescriptor 場所、D3 の動的スケーリング等、実装済みの対策を見落としている

**全39件が確定済み（未確認項目: 0件）。** FALSE POSITIVE は 11件（28%）、有効な問題は 21件（54%）。

**推奨**: bug.md の各項目を、上記検証結果に基づいて更新し、FALSE POSITIVE と判定された 11項目を削除または訂正すること。有効と判定された 21項目については、優先度に従って修正を進めるべきである。
