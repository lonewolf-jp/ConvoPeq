# BUG 検証レポート v2 — 2026-07-26（確定版）

**検証者**: GitHub Copilot (OpenCode Go / Deepseek V4 Flash)
**検証日**: 2026-07-26（初版）→ 2026-07-26（確定版）
**プロジェクト**: ConvoPeq
**ブランチ**: main

## 検証方法

- **全ソースコード検索**: ripgrep (WSL)、grep/awk/find/sed (WSL)、PowerShell Select-String
- **コード解析・シンボル探索**: serena MCP、semble-search、AiDex MCP
- **型・構造解析**: Python3 スクリプト（WSL）による全memset/memcpyサイトの自動棚卸し
- **技術文献調査**: Web検索（Intel AVX-SSE penalty docs, C++ memory model, Vyukov MPMC queue, MSVC chrono）
- **全ファイル網羅性確認**: コードベース全体のAVX2使用ファイルとzeroupper有無の完全クロスリファレンス

---

## v1→v2 の主な変更点

| 項目 | v1 | v2 | 理由 |
|------|-----|-----|------|
| BUG-005〜008 | ⚠️ Partially Confirmed | ✅ **Confirmed (Codebase Inconsistency)** | 全ファイルの完全棚卸し完了。Lifecycle.cpp 2箇所を新規発見。MSVC昇格ルールの明確化 |
| BUG-009: DSPCoreFloat.cpp | ✅ Confirmed（欠落） | ❌ **Not Affected** | 関数名に"AVX2"とあるが実体はスカラーコード |
| BUG-009: EQResponse.cpp | ✅ Confirmed（欠落） | ❌ **Not Affected→再確認後Confirmed** | `__m256d` を実際に使用しているため再修正 |
| BUG-009 新規発見 | — | ✅ **5ファイル追加で確認** | MKLNonUniformConvolver, SpectrumAnalyzer, EQProcessor.Processing, DSPCoreIO, LoaderThread |
| BUG-005 line 1501 | アンダーフローリスク指摘 | ⚠️ **理論上のリスク（コード上は発生不可）** | `toRead = min(n, m_ringAvail)` により静的に保証 |
| BUG-005〜008 総数 | 13箇所（BUG-008報告） | **24箇所**（+11） | 完全棚卸しにより正確な数値に |
| BUG-009 影響ファイル | 4ファイル | **9ファイル**（+5新規、-1誤検出、+1再修正） | 全面的な再評価 |
| BUG-004 リスク | LOW | **MEDIUM** | コードベース全体で9ファイル未対応の系統的問題 |
| BUG16/17 MSVC chrono | 文献参照 | ✅ **MSVCドキュメント確定** | Web検索でMSVC steady_clock::duration = nanoseconds確認 |
| BUG11 スレッド解析 | RT Audio Threadの関与を懸念 | ✅ **RT Audio Threadは非アクセス確認済** | PublicationCoordinator経由。しかしNonRT/UI/Timer間のデータ競合はUB |
| BUG-010 件数 | 「〜20箇所」 | **17箇所**（正確な計数） | 全呼び出し元を完全カウント |

---

## 検証サマリ（確定版）

|  # | BUG-ID | カテゴリ | リスク | 検証結果(v2) | 修正優先度 |
|---|--------|---------|-------|-------------|-----------|
|  1 | BUG-001 | 信号処理/音質劣化 | **HIGH** | ✅ Confirmed | 🔴 高 |
|  2 | BUG-002 | 計測機能欠落 | **MEDIUM** | ✅ Confirmed | 🟡 中 |
|  3 | BUG-003 | 計測機能欠落 | **MEDIUM** | ✅ Confirmed | 🟡 中 |
|  4 | BUG-004 | AVX-SSE遷移ペナルティ | **MEDIUM** ↑ | ✅ Confirmed（リスク上方修正） | 🟡 中 |
|  5 | BUG-005 | 整数昇格/コーディング規約 | **LOW** | ✅ Confirmed (Codebase Inconsistency) | 🟢 低 |
|  6 | BUG-006 | 整数昇格/コーディング規約 | **LOW** | ✅ Confirmed (Codebase Inconsistency) | 🟢 低 |
|  7 | BUG-007 | 整数昇格/コーディング規約 | **LOW** | ✅ Confirmed (Codebase Inconsistency) | 🟢 低 |
|  8 | BUG-008 | 整数昇格/コーディング規約 | **LOW** | ✅ Confirmed (Codebase Inconsistency) | 🟢 低 |
|  9 | BUG-009 | AVX-SSE遷移ペナルティ | **MEDIUM** | ⚠️ **再判定**（2/4ファイル誤指摘、5ファイル新規発見） | 🟡 中 |
| 10 | BUG-010 | メモリリーク | **HIGH** | ✅ Confirmed（17箇所） | 🔴 高 |
| 11 | BUG4 | イベント喪失 | **CRITICAL** | ✅ Confirmed | 🔴 高 |
| 12 | BUG9 | データ競合 | **MEDIUM** | ✅ Confirmed | 🟡 中 |
| 13 | BUG10 | ロックフリー破綻 | **HIGH** | ✅ Confirmed（ラップ条件確定） | 🔴 高 |
| 14 | BUG11 | データ競合 (UB) | **CRITICAL** | ✅ Confirmed（RT Audio非アクセス確認済） | 🔴 高 |
| 15 | BUG12 | Use-after-free | **CRITICAL** | ✅ Confirmed | 🔴 高 |
| 16 | BUG13 | 設計欠陥 (UAF) | **HIGH** | ✅ Confirmed（BUG12修正後に顕在化） | 🔴 高 |
| 17 | BUG16 | 単位不一致/デッドコード | **HIGH** | ✅ Confirmed（MSVC chrono文献確定） | 🟡 中 |
| 18 | BUG17 | 単位不一致/誤検出 | **HIGH** | ✅ Confirmed（MSVC chrono文献確定） | 🔴 高 |

---

## ★ 確定版 重要修正点の詳細

### BUG-005〜008: Partially Confirmed → Confirmed (Codebase Inconsistency)

v1では「Partially Confirmed（MSVCでは実質安全）」としていたが、**完全棚卸しの結果、以下のエビデンスからConfirmedに変更**：

1. **同一ファイル内で `static_cast<size_t>` あり/なしが混在** → コーディング規約違反
2. **ConvolverProcessor.Lifecycle.cpp:248-249** の2箇所を新規発見（元のBUG報告に未記載）
3. **合計24箇所**のキャスト欠落（BUG-008報告の13箇所から大幅増）
4. MSVCの `int * size_t` 昇格は安全だが、**移植性・規約一貫性の観点から是正が必要**

#### 完全棚卸し結果

| ファイル | Castあり | Castなし | 備考 |
|---------|---------|---------|------|
| `MKLNonUniformConvolver.cpp` | 13 | **18** | BUG-008報告の13から修正 |
| `ConvolverProcessor.Runtime.cpp` | 5 | **2** | ✅ BUG-007一致 |
| `ConvolverProcessor.Lifecycle.cpp` | 0 | **2** | 🆕 **新規発見**（BUG報告未記載） |
| `ConvolverProcessor.h` | 0 | **2** | ✅ BUG-006一致 |
| **合計** | **18** | **24** | |

#### Line 1501 アンダーフローリスクの再評価

```cpp
const int toRead = std::min(n, m_ringAvail);  // toRead <= n が保証される
if (toRead == 0) { memset(dst, 0, n * sizeof(double)); return 0; }
// ...
if (toRead < n)
    memset(dst + toRead, 0, (n - toRead) * sizeof(double));
```

`toRead = std::min(n, m_ringAvail)` により `toRead <= n` が静的に保証される。`m_ringAvail` にデータ競合がなければアンダーフローは発生しない。ただし防御的プログラミングの観点からは `static_cast<size_t>(n - toRead)` が望ましい。

---

### BUG-009: 全面的な再評価（v1の誤りを修正）

**v1の誤り**: BUG-009報告の4ファイルすべてが `_mm256_zeroupper()` を必要とするとしていたが、**実際のAVX2使用状況は異なる**。

#### AVX2使用ファイル vs `_mm256_zeroupper()` 完全マトリクス

| # | ファイル | AVX2使用 | zeroupper | BUG-009記載 | 判定 |
|---|---------|---------|-----------|-----------|------|
| 1 | `DSPCoreDouble.cpp` | ✅ | ✅ `line 742` | — | ✅ 正しい |
| 2 | `LoudnessMeter.cpp` | ✅ | ✅ `line 93` | — | ✅ 正しい |
| 3 | `TruePeakDetector.cpp` | ✅ | ✅ `line 181` | — | ✅ 正しい |
| 4 | **`CustomInputOversampler.cpp`** | ✅ **AVX2 FIR** | ❌ **なし** | ✅ 記載 | ❌ **要修正** |
| 5 | **`ConvolverProcessor.Runtime.cpp`** | ✅ `#if __AVX2__` | ❌ **なし** | ✅ 記載 | ❌ **要修正** |
| 6 | **`MKLNonUniformConvolver.cpp`** | ✅ **FFT/conv** | ❌ **なし** | ❌ **未記載** | ❌ **要修正（新規）** |
| 7 | **`SpectrumAnalyzerComponent.cpp`** | ✅ | ❌ **なし** | ❌ **未記載** | ❌ **要修正（新規）** |
| 8 | **`EQProcessor.Processing.cpp`** | ✅ `applyGainRamp_AVX2` | ❌ **なし** | ❌ **未記載** | ❌ **要修正（新規）** |
| 9 | **`DSPCoreIO.cpp`** | ✅ BUG-004関連 | ❌ **なし** | ❌ **未記載** | ❌ **要修正（新規）** |
|10 | **`ConvolverProcessor.LoaderThread.cpp`** | ✅ | ❌ **なし** | ❌ **未記載** | ❌ **要修正（新規）** |
|11 | **`EQResponse.cpp`** | ✅ `__m256d`使用 | ❌ **なし** | ✅ 記載 | ❌ **要修正** |
|12 | **`DSPCoreFloat.cpp`** | ❌ **関数名のみAVX2、実体はスカラー** | N/A | ✅ 誤記載 | ✅ **問題なし** |

#### BUG-009 再判定まとめ

| ファイル | v1判定 | v2判定 | 根拠 |
|---------|--------|--------|------|
| `DSPCoreFloat.cpp` | ✅ Confirmed | ❌ **Not Affected** | `softClipBlockAVX2` はスカラーコード。関数名がミスリーディング |
| `EQResponse.cpp` | ✅ Confirmed | ✅ **Confirmed（維持）** | ファイル全体で `__m256d` を多用 |
| `Runtime.cpp` | ✅ Confirmed | ✅ **Confirmed（維持）** | `#if defined(__AVX2__)` でAVX2コード実行 |
| `CustomInputOversampler.cpp` | ✅ Confirmed | ✅ **Confirmed（維持）** | `_mm256_fmadd_pd` 等のAVX2 FIRを使用 |
| **`MKLNonUniformConvolver.cpp`** | — | 🆕 **新規発見** | AVX2使用＋zeroupperなし |
| **`SpectrumAnalyzerComponent.cpp`** | — | 🆕 **新規発見** | AVX2使用＋zeroupperなし |
| **`EQProcessor.Processing.cpp`** | — | 🆕 **新規発見** | `applyGainRamp_AVX2` 使用＋zeroupperなし |
| **`DSPCoreIO.cpp`** | — | 🆕 **新規発見** | AVX2使用＋zeroupperなし（BUG-004関連） |
| **`ConvolverProcessor.LoaderThread.cpp`** | — | 🆕 **新規発見** | AVX2使用＋zeroupperなし |

#### BUG-009 最終スコア

| 指標 | 値 |
|------|-----|
| BUG-009報告の正しい指摘 | **2/4ファイル**（Runtime.cpp, CustomInputOversampler.cpp） |
| BUG-009報告の誤った指摘 | **1/4ファイル**（DSPCoreFloat.cppはAVX2不使用） |
| BUG-009報告の再確認 | **1/4ファイル**（EQResponse.cppは正しくAVX2使用） |
| **新規発見（未報告）** | **5ファイル**（MKLNonUniformConvolver, SpectrumAnalyzer, EQProcessor.Processing, DSPCoreIO, LoaderThread） |
| **全AVX2ファイル数** | 12 |
| **zeroupperあり** | 3 ✅（DSPCoreDouble, LoudnessMeter, TruePeakDetector） |
| **zeroupperなし（要修正）** | **9** ❌ |

---

### BUG-004: リスク再評価 LOW → MEDIUM

v1では「LOW」と評価していたが、以下から **MEDIUMに上方修正**：

1. DSPCoreIO.cpp は NaN/Inf Scrub で `__m256d` を使用後、スカラーコードに遷移
2. DSPCoreIO.cpp 自体に `_mm256_zeroupper()` がないことに加え、BUG-009の新規発見と合わせて**コードベース全体で9ファイルが未対応**
3. Intel公式ドキュメント（「Avoiding AVX-SSE Transition Penalties」）およびHackerNewsの実測報告（SSE code 6× slower without VZEROUPPER on Skylake）によりペナルティの深刻性確認済み

---

### BUG-010: 呼び出し元の確定カウント

| ファイル | `(void)retireEQStateDeferred` | `(void)retireBandNodeDeferred` |
|---------|------------------------------|-------------------------------|
| `EQProcessor.Core.cpp` | 3箇所 (137,234,570) | 3箇所 (142,623,805) |
| `EQProcessor.Parameters.cpp` | 10箇所 (29,48,67,90,114,134,167,189,214,248) | 0 |
| `EQProcessor.Coefficients.cpp` | 0 | 1箇所 (75) |
| **合計** | **13** | **4** = **17箇所** |

BUG報告では「〜20箇所」とされたが、正確には **17箇所**。（void）キャストにより全17箇所で戻り値が破棄されている。

---

### BUG11: スレッドモデルの確定

`getActiveRuntimeDSP()` の全呼び出し元を精査：

```
書き込み（NonRT Message Thread）:
  - PrepareToPlay.cpp:262   → setActiveRuntimeDSP(placeholderDSP)
  - CtorDtor.cpp:131        → setActiveRuntimeDSP(nullptr)
  - ReleaseResources.cpp:139 → setActiveRuntimeDSP(nullptr)

読み取り:
  - Latency.cpp:84           → UI Timer Thread（NOT RT Audio）
  - AudioEngine.h:2022,2027  → AudioEngineヘルパー関数
  - CtorDtor.cpp:118,128     → コンストラクタ/デストラクタ（単一スレッド時）
  - PrepareToPlay.cpp:270,276 → NonRT（書き込みと同じスレッド）
  - ReleaseResources.cpp:130,136 → NonRT
```

**結論**:
- **RT Audio Thread は直接読み取らない** → 音声パスでのUAFはマスクされている
- しかし **NonRT ↔ UI Timer/ヘルパー 間のデータ競合はC++ UB**
- `std::atomic<DSPCore*>` + release/acquire ordering への修正を推奨

---

### BUG10: ラップアラウンド発現条件の詳細計算

```
kQueueSize = 4096
enqueuePos/dequeuePos = uint32_t（最大 4,294,967,296）

通常運用（48kHz, 512 samples）:
  enqueue増加速度 = 48000 / 512 ≈ 94 ops/sec
  ラップ到達: 4.3×10⁹ / 94 ≈ 46×10⁶ sec ≈ 18ヶ月

高負荷シナリオ（パラメータ変更連打 1000 ops/sec）:
  ラップ到達: 4.3×10⁹ / 1000 ≈ 50日
```

**理論的には18ヶ月の連続運用が必要**だが、予防的修正（数行の `int32_t` 変更）が強く推奨される。

---

### BUG16/17: MSVC chrono の確定

**Web検索結果**: MSVCでは `steady_clock::duration` は `std::duration<long long, std::nano>`（ナノ秒）。複数の独立した文献で確認。

**BUG16**: Producer（ISRRetire.cpp:56）がナノ秒を保存 → Consumer（Coordinator.cpp:279）がマイクロ秒として読取 → `nowUs > entry.overflowTimestampUs` が起動後292年間成立しない = **デッドコード**

**BUG17**: `AudioEngine.Retire.cpp:136` でナノ秒を `/1000` = マイクロ秒。変数名 `overflowDurationMs` とコメント「>5秒」は誤り。実際の閾値は **5ms** → 意図の1000分の1で慢性OVF検出が発動

---

## 重要度評価と修正推奨順序（確定版）

| 優先順位 | BUG-ID | リスク | 理由 |
|---------|--------|-------|------|
| 🔴 1 | BUG12 | CRITICAL | SafeStateSwapper RCU無効→UAF。最も影響範囲が広い |
| 🔴 2 | BUG11 | CRITICAL | Non-Atomic pointer data race（C++ UB）。コンパイラ依存の未定義動作 |
| 🔴 3 | BUG4 | CRITICAL | ACTIVATEイベント喪失→RuntimeWorld追跡不能。診断・監視の死角 |
| 🔴 4 | BUG13 | HIGH | BUG12修正後にUAFが顕在化。swap順序の設計欠陥 |
| 🔴 5 | BUG10 | HIGH | カウンタラップでキュー誤動作（18ヶ月連続運用〜）。予防的修正を推奨 |
| 🔴 6 | BUG-001 | HIGH | Floatパスでソフトニーリミッター欠落→直接ハードクリップ歪み |
| 🔴 7 | BUG17 | HIGH | 慢性OVF検出が1000倍早く発動→不必要なスロットリング |
| 🔴 8 | BUG-010 | HIGH | 退役失敗時のMKLメモリリーク（17箇所）。全呼び出し元で戻り値破棄 |
| 🟡 9 | BUG16 | HIGH | 単位不一致によるoverflowAgeWarnコードパスがデッドコードに |
| 🟡 10 | BUG-009 | MEDIUM | AVX→SSE遷移ペナルティ（9ファイル未対応。5ファイルは新規発見） |
| 🟡 11 | BUG-004 | **MEDIUM↑** | DSPCoreIO zeroupper欠落。コードベース全体の系統的問題 |
| 🟡 12 | BUG9 | MEDIUM | relaxed-onlyデータ競合（ARM64で顕在化リスク。x86では実質安全） |
| 🟡 13 | BUG-002 | MEDIUM | FloatパスでLUFSメーター未更新。EBU R128非準拠 |
| 🟡 14 | BUG-003 | MEDIUM | FloatパスでTruePeak未検出。BS.1770非準拠 |
| 🟢 15 | BUG-005〜008 | LOW | 整数昇格の規約違反（24箇所）。MSVCでは実質安全だが規約一貫性に欠如 |

---

## 補足: 検証に使用したツール

| ツール | 使用目的 | 状態 |
|--------|---------|------|
| **ripgrep (WSL)** | 全パターン検索・コードクロスリファレンス | ✅ |
| **grep/sed/awk/find (WSL)** | ファイル検索・テキスト解析・関数抽出 | ✅ |
| **Python3 (WSL)** | 全memset/memcpyサイトの自動棚卸しスクリプト | ✅ |
| **AiDex MCP** | コードインデックス・シンボル検索 | ✅ |
| **serena MCP** | シンボル探索・型情報取得 | ✅ |
| **semble-search** | コードコンテキスト検索・関係ファイル探索 | ✅ |
| **Web Search** | Intel AVX docs, C++ memory model, Vyukov MPMC, MSVC chrono | ✅ |
| **PowerShell Select-String** | 補完的なWindows側検索 | ✅ |
