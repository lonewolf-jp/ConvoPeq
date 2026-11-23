# ds_bug.md / qwen_bug.md の精査とバグ・非バグ分類

- 実施日時: 2026-11-15 15:00〜16:10 JST
- 対象: `C:\Users\user\Desktop\ds_bug.md`（10 件）と `C:\Users\user\Desktop\qwen_bug.md`（38 件）
- 参照ソース: `C:\VSC_Project\ConvoPeq` の作業ツリー（main `4d476d5c` ＋ 未コミット変更 189 件）
- 方法: 各主張の該当箇所をソースで再確認し、到達性・実害・設計意図で分類した

> 参照リビジョンについて: 作業ツリーと `origin/main`（`daf45501`）で差があるのは `src/MKLNonUniformConvolver.cpp`、`src/NoiseShaperLearner.h`、`src/audioengine/AudioEngine.h` の 3 ファイル。いずれも今回の判定に使ったコード（`l2Part` の計算、`CmaEsOptimizer optimizer;` のメンバ、`liveCount.fetch_sub`）は内容が同一で、行番号だけがずれる。判定は作業ツリーの行番号で示す。

> **【2026-11-15 18:20 訂正】このレポートの BUG-028 の判定は誤りだった。** ソース上に `_mm256_zeroupper()` が無いことだけを見て「バグ」と判定したが、コンパイラの生成コードを確認していなかった。MSVC 19.52 と icx 2026.2 の双方で、明示的な `vzeroupper` が 1 つも無いソースから AVX/SSE 境界に `vzeroupper` が自動生成されることを逆アセンブルで確認した。BUG-028 は**非バグ**である。詳細は `doc/work113/20261115-1800_4件修正/report.md` の第 6 章を参照。あわせて B-2 の推奨対応（下限到達で終了）も実測で設計品質を悪化させることが分かったため、下記の推奨対応順を訂正した。

---

## 1. 結論

| 区分 | 件数 | 内訳 |
| --- | ---: | --- |
| **バグ（実害あり）** | **3** | BUG-013、B-1、B-2（BUG-028 は 2026-11-15 18:20 に非バグへ訂正） |
| 非バグ（設計・到達不能・無害） | 44 | ds の 7 件、qwen の 35 件（BUG-028 を含む）、BUG-018、BUG-029 |
| レポート側の誤り | 3 | ds B-3、ds B-9、ds B-10 の一部 |

最重要は **BUG-013（qwen）**。非2冪の `tailL1L2Multiplier` を選ぶと L2 テールの FFT プランが実際より小さく作られ、音響出力が壊れる。次に **B-1（ds）** のデッドコードと OOM 時リーク、**B-2（ds）** の死んだ早期終了条件。**BUG-028（qwen）** は逆アセンブルで非バグと確認済み（訂正参照）。

---

## 2. ds_bug.md の判定（10 件）

| ID | 主張 | 判定 | 根拠 |
| --- | --- | --- | --- |
| B-1 | `AudioSegmentBuffer::create()` の 3 問題 | **バグ（OOM 時のみ）** | 下記 2.1 |
| B-2 | CMA-ES の早期打ち切り | **バグ（性能）** | 下記 2.2 |
| B-3 | `CmaEsOptimizer` が CSA でない | **非バグ（設計選択）。「デッドコード」は誤り** | 下記 2.3 |
| B-4 | `trySubmitImpl` の二度ビルド | **非バグ（性能）** | 下記 2.4 |
| B-5 | `std::abort()` | **非バグ（意図的 fail-fast）** | 下記 2.5 |
| B-6 | `liveCount.fetch_sub` | **非バグ（規約違反のみ）** | 下記 2.6 |
| B-7 | `resetForRuntime` の RT 呼出 | **非バグ（到達不能）** | 下記 2.7 |
| B-8 | `MKLAllocator` の typedef 欠落 | **非バグ（C++20）** | 下記 2.8 |
| B-9 | prefetch の OOB | **非バグ（無害）。「デッドコード」は誤り** | 下記 2.9 |
| B-10 | probe 系列の不一致 | **非バグ（実害なし）。ただし反証は不完全** | 下記 2.10 |

### 2.1 B-1: `AudioSegmentBuffer::create()`

`src/AudioSegmentBuffer.h:32-51` と `src/AlignedAllocation.h` を確認した。

- `convo::aligned_malloc` は失敗時に `nullptr` を返さず `std::bad_alloc` を送出する（`AlignedAllocation.h` の `if (ptr == nullptr) throw std::bad_alloc();`）。したがって `if (!left || !right)`（同 40 行）は**到達しないデッドコード**。指摘は正しい。
- `right` の確保で例外が出ると `left` が解放されない。`kCapacity = 5 × 768000 = 3,840,000`、`sizeof(double) = 8` なので 1 配列 **30,720,000 B（約 29.3 MiB）**。
- `new AudioSegmentBuffer()` で例外が出ると両方が漏れる（約 58.6 MiB）。

ただし到達するのは**メモリ枯渇時だけ**で、その状況ではプロセスが継続できない可能性が高い。重大度は「低（OOM 時の資源リーク＋誤解を招くデッドコード）」。ds の「当初より重い」は強めの表現だが、デッドコードと 2 つのリーク経路は事実。

### 2.2 B-2: CMA-ES の早期終了

`src/AllpassDesigner.cpp:327-332` と `:466-473`、`src/AllpassDesigner.h:74-82` を確認した。

- 既定は `cmaesParams.sigmaMin = 1e-6`、`cmaesInitialSigma = 0.3`。`p.sigmaMin = min(1e-6, 0.3) = 1e-6`、`p.sigmaMax = max(2.0, 0.3) = 2.0` で**設定は変わらない**。
- ループ内の閾値は `min(1e-4, 1e-6 × 0.1) = 1e-7`。`CmaEsOptimizerDynamic::setSigma` は `std::clamp(s, sigmaMin, sigmaMax)` するので sigma は 1e-6 未満にならない。**`sigma < 1e-7` は成立せず、sigma による早期終了は既定で無効**。
- コメント（468-469 行）の想定ケース `sigmaMin = 0.05` でも `min(1e-4, 0.005) = 1e-4` で、修正前と同じ。**コメントの意図は実装されていない**。

正しさへの影響はない（`bestFitness < 1.0`、停滞 6 回、最大世代で終了する）。無駄に最大 100 世代回る性能問題。ds の「当初の主張は誤り」は正しく、「より重大」は過大（性能のみ）。

### 2.3 B-3: `CmaEsOptimizer` の sigma 更新

`src/CmaEsOptimizer.h:132` の update はエリート集団の RMS 距離を sigma にする自己適応で、CSA ではない。ここまでは指摘どおり。

しかし「`CmaEsOptimizer` はデッドコード」は**誤り**。`src/NoiseShaperLearner.h:305` に `CmaEsOptimizer optimizer;` があり、`src/NoiseShaperLearner.cpp:895` と `:1059` から `optimizer.update(...)` が呼ばれている。ノイズシェイパー学習の本番経路で動いている。

sigma の更新式が CSA でないことは**設計選択**であり、誤動作の証拠はない。したがって「非バグ」だが、ds が付けた「デッドコードだから実害ゼロ」という根拠は成立しない。

### 2.4 B-4: `trySubmitImpl` の二度ビルド

`src/audioengine/RuntimePublicationOrchestrator.cpp:189`（1 回目）と `:264-266`（crossfade 必要時のみ 2 回目）を確認した。指摘どおり。

- 1 回目の `worldOwner` は `unique_ptr` の代入で破棄され、リークしない。
- 1 回目の world は publish されないので coordinator の状態に影響しない。
- `RuntimeBuilder::buildRuntimePublishWorld` は先頭で `engine.reserveRuntimePublicationIdentity()` を呼ぶ（`RuntimeBuilder.cpp:166`）ため、世代・worldId・公開シーケンスが 1 つ飛ぶ。monotonicity は `>` 比較なので飛びは許容される。

正しさへの影響なし。性能問題として妥当。

### 2.5 B-5: `std::abort()`

`src/audioengine/ISRLifecycle.cpp` に 12 箇所。`enterAudioCallback` は Releasing/Released/Shutdown と想定外 phase で、`leaveAudioCallback` は AudioRunning 以外で abort する。`hostChaosViolations_` に計数してから落とす。

JUCE の通常フロー（stop → join → releaseResources）では発火しない。**意図的な fail-fast 設計**で、バグではない。`leaveAudioCallback` 側が phase 遷移の競合で発火し得る点は設計上のトレードオフ。

### 2.6 B-6: `liveCount.fetch_sub`

`src/audioengine/AudioEngine.h:884`。`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内で `liveCount.fetch_sub(1, std::memory_order_relaxed)`。機能は `convo::fetchSubAtomic` と等価。規約（atomic のドット呼び出し禁止）違反のみで、バグではない。

### 2.7 B-7: `resetForRuntime` の RT 呼出

`resetForRuntime` の呼出元は `AudioEngine::DSPCore::reset()` だけ（`AudioEngine.Processing.DSPCoreLifecycle.cpp:409,410,430,431`）。その `DSPCore::reset()` の呼出元は**テストのみ**（`src/tests/AudioEngineHarness/V11EngineRetireProtectionTests.cpp:3348`）。本番経路からは到達しない。バグではない（テスト専用経路）。

### 2.8 B-8: `MKLAllocator`

`src/AlignedAllocation.h:182` に `value_type` と `rebind` があり、`size_type` 等は `std::allocator_traits` が補完する。C++20 で問題なく allocator として使える。非バグ。

### 2.9 B-9: prefetch の OOB

`src/eqprocessor/EQProcessor.Processing.cpp:295` の `_mm_prefetch(data + i + 64)` は、`numSamples` が小さいとバッファ外を指す。`_mm_prefetch` はヒントで fault しないため**実害なし**。

ただし「`applyGainRamp_AVX2` は呼び出されていない（デッドコード）」は**誤り**。同ファイル `:444`、`:991`、`:1290` の 3 箇所から呼ばれている。OOB は事実だが無害、という判定になる。

### 2.10 B-10: probe 系列の不一致

`src/audioengine/AudioEngine.Cache.cpp:63-79`（lock-free）と `:101-127`（locked）を確認した。どちらも `EQProcessor::probeCacheHash(hash, probe)` を使い、系列は同じ関数。

ただし lock-free ループで `hash` が進んだ場合、locked ループは**進んだ hash から**始まる（`:87` で `cache->paramsHash = hash`）。この場合、2 つの系列は 1 手目以降でずれる。`cacheKeyMatches` は「別キーを誤って返す」ことを防ぐが、「同じキーの重複挿入」は防げない。

影響は同一キーのキャッシュが 2 個できるだけで、係数は同一。**機能的な誤りはない**が、ds の「誤り・撤回」は言い過ぎで、「実害はメモリのみの理論的可能性」が正確。

---

## 3. qwen_bug.md の判定（38 件）

### 3.1 個別検証した 4 件（BUG-028 は 2026-11-15 に非バグへ訂正）

| ID | 主張 | 判定 | 根拠 |
| --- | --- | --- | --- |
| BUG-013 | `ProductionFft::createPlan` の 2 冪検証欠落 | **バグ（妥当・最重要）** | 下記 3.2 |
| BUG-028 | decimate 経路の zeroupper 欠落 | **非バグ（コンパイラが自動生成。2026-11-15 訂正）** | 下記 3.3 |
| BUG-018 | `TruePeakDetector::processBlock` の容量ガード欠落 | **非バグ（到達不能）** | 下記 3.4 |
| BUG-029 | `/fp:fast` の数値乖離 | **非バグ（意図的・未計測）** | 下記 3.5 |

### 3.2 BUG-013（バグ）

`src/FFTBackend.cpp:21-30`:

```cpp
plan.complexSize = fftSize / 2 + 1;
int order = 0; int tmp = fftSize;
while (tmp > 1) { tmp >>= 1; ++order; }   // 非2冪では floor(log2)
```

`ippsFFTGetSize_R_64f(order, ...)` は 2^order のプランを作る。`isValid()` は `spec != nullptr`（`src/FFTBackend.h:120`）だけなので、**非2冪でも true**。`complexSize` は `fftSize/2+1` のままなので、実際のプラン長と一致しない。

到達経路を確認した:

- `src/MKLNonUniformConvolver.cpp:682` で `tailL1L2Mult = jlimit(2, 16, filterSpec->tailL1L2Multiplier)`（整数 2〜16）。
- 同 `:705` の Air モードは `max(tailL1L2Mult, 6)` で、**6 がそのまま残る**。
- 同 `:841` `const int l2Part = l1Part * tailL1L2Mult;`。`l1Part` は 2 冪（`:824,836-840`）なので、mult が 2 冪でないと `l2Part` は非2冪。
- 同 `:905` `l.fftSize = l.partSize * 2;`、`:919` `ProductionFft::createPlan(l.fftSize)`。
- UI のスライダは 2〜16 を step 1 で選べる（`ConvolverProcessor.h:245-246`、`ConvolverControlPanel.cpp:159`）。

例: `l1Part = 1024`、mult = 6 → `l2Part = 6144`、`fftSize = 12288` → order = 13 → 実プランは 8192 点。`complexSize = 6145` に対し実プランは 4097 点で、L2 テールの FFT 結果が壊れる。既定値 8 は 2 冪なので既定構成では発火しない。

**バグとして確定**。修正は `createPlan` 冒頭の 2 冪検証（fail-closed）と、`tailL1L2Mult` を 2 冪に制約する（Air の下限 6 を 8 にする）の両方が必要。

### 3.3 BUG-028（非バグ・2026-11-15 18:20 に訂正）

`src/CustomInputOversampler.cpp`:

- `dotProductAvx2` は `:195` で `_mm256_zeroupper()` を実行。
- `dotProductDecimateAvx2`（`:211-276`）には明示的な zeroupper がない。
- `decimateStage` の convCount 4〜7 インライン AVX2 パス（`:672-694`）にもない。

精査時はこれを「規約から外れている」と判定したが、**コンパイラの生成コードを確認していなかった**。`_mm256_zeroupper()` をすべて削除したソースを MSVC 19.52（`/O2 /arch:AVX2`）と icx 2026.2（`/QxCORE-AVX2`）でコンパイルし、`dumpbin /disasm` で ymm を使う関数ごとに `vzeroupper` を数えたところ、両方とも AVX/SSE 境界に自動生成していた（`dotProductDecimateAvx2` は MSVC で 1 個、icx で 1 個、`decimateStage` は MSVC 3 個、icx 3 個）。したがって明示的な記述が無いことは欠陥ではない。**非バグ**。

### 3.4 BUG-018（非バグ・到達不能）

`src/TruePeakDetector.cpp:125-149` は `numSamples` に対して容量検査をしない。ただし呼出側で上限が保証されている:

- `AudioEngine.Processing.DSPCoreDouble.cpp:310` が `numSamples > maxSamplesPerBlock` で `buffer.clear(); return;`。
- `AudioEngine.Processing.AudioBlock.cpp:341` も同じ上限チェック。
- `prepare` は `maxInternalBlockSize` で確保し、`maxInternalBlockSize = inputMaxBlock * 8`（`DSPCoreLifecycle.cpp:161-164`）。`maxSamplesPerBlock = inputMaxBlock`。

つまり `numSamples ≤ maxSamplesPerBlock` で、作業バッファは **8 倍の余裕**がある。現行コードでは到達不能。ガード欠落は防御性の不足であり、バグではない。

### 3.5 BUG-029（非バグ・未計測）

`CMakeLists.txt` で MSVC Release は `/fp:fast` を外して `/fp:precise`（`:2101`）、icx Release は `/fp:fast` を残す（`:2196-2197`）。`#pragma float_control(precise, on)` は `ConvolverProcessor.Runtime.cpp` と `MKLNonUniformConvolver.cpp` の 2 ファイルのみ。

icx の `/fp:fast` は LLVM OOM 回避のための**意図的な選択**で、コメントにも明記されている。数値影響は未計測だが、誤動作の証拠はない。**非バグ（未検証項目として残す）**。

### 3.6 非バグ 34 件のスポットチェック

一覧の主張を代表箇所で確認した。いずれも記載どおりの対策が入っている。

| 確認した主張 | 確認結果 |
| --- | --- |
| CorruptSampleGuard が NaN/Inf を明示捕捉（000, 015, 016） | `src/CorruptSampleGuard.h:41-47` が `_CMP_UNORD_Q`（NaN）と `_CMP_GT_OQ`（abs(x) > 2^53）で判定。/fp:fast に依存しない |
| RCU の depth ゲートが acq_rel RMW（001〜009, 037） | `src/core/RCUReader.h:40-71`。所有権 CAS（acq_rel）を線形化点にし、その後に depth を加算。他スレッド所有時は状態を変更しない |
| `canonicalFloatBits` が符号保持・±0/NaN 正規化（017） | `src/eqprocessor/EQProcessor.h:387-388` に宣言、実装あり |
| デノーマル対策（020） | `juce::ScopedNoDenormals` を NonRT 経路で使用、RT はスレッド起動時に FTZ/DAZ を 1 回設定（`AudioEngine.Processing.AudioBlock.cpp:106` のコメント） |
| リングバッファの型制約（036） | `AudioEngine.h:475-485` に `is_trivially_copyable` / `is_standard_layout` / `is_trivial` / アラインメントの static_assert |
| EQ キャッシュのキー照合（021〜024） | `AudioEngine.Cache.cpp` の両経路で `cacheKeyMatches` を使用（B-10 の項を参照） |

残りの項目も同じファイル群の記載どおりの実装を確認した。ただし BUG-010〜012、014（NUPC 幾何）と BUG-025、026（オーバーサンプリング）は幾何・数値の網羅検証までは行っていない（コード上のガードの存在確認まで）。

---

## 4. レポート側の誤り

| 箇所 | 誤り | 正しい内容 |
| --- | --- | --- |
| ds B-3 | 「`CmaEsOptimizer` はデッドコード」 | `NoiseShaperLearner` がメンバとして保持し、`update` を 2 箇所から呼ぶ（本番経路） |
| ds B-9 | 「`applyGainRamp_AVX2` は呼び出されていない」 | `EQProcessor.Processing.cpp:444, 991, 1290` から呼ばれる |
| ds B-10 | 「誤り（撤回）」 | 系列は lock-free で hash が進んだ場合にずれ得る。重複挿入は理論上可能（実害はメモリのみ） |
| ds B-1 | 「当初より重い」 | リークは OOM 時のみ。重大度は低 |
| ds B-2 | 「より重大な別バグ」 | 性能問題（無駄な世代）。正しさへの影響なし |
| qwen BUG-018 | 「条件付きバグ（到達性是認）」 | 現行コードでは到達不能（呼出側の上限＋8 倍の余裕） |

---

## 5. 推奨対応順

1. **BUG-013**: `createPlan` に 2 冪検証を追加（fail-closed）＋ `tailL1L2Mult` を 2 冪に制約。音響出力に直結するため最優先。
2. **B-1**: `create()` を RAII 順（`unique_ptr` を作ってから `reset`）に変更し、デッドコードを削除。
3. **B-2**: 到達不能な sigma 条件を削除し、終了判定を停滞判定に一本化する（実測で「floor 到達で終了」は設計品質を悪化させたため。2026-11-15 訂正）。
4. **BUG-028**: 対応不要（コンパイラが `vzeroupper` を自動生成する。2026-11-15 訂正）。
5. 低優先度: B-6（規約統一）、BUG-018（防御的ガード追加）、B-10（重複挿入の可能性をコメント化）。

---

## 6. 未実施

- 修正の実装とテスト（本作業は分類のみ）
- BUG-013 の実機再現（非2冪の mult を設定して L2 テールの出力を測る）
- BUG-029 の icx/MSVC 実測比較
- NUPC 幾何（BUG-010〜012, 014）とオーバーサンプリング（025, 026）の数値網羅検証
