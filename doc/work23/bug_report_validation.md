# バグレポート妥当性検証 総合報告書

**作成日**: 2026-06-08
**検証範囲**: 5件の外部バグレポート（AI静的解析ツール・LLM生成レビュー）に含まれる全クレーム
**検証方法**: 実ソースコード (src/) 全ファイル直接読取り + grep/Select-String + serena MCP + ccc + semble + CodeGraph MCP + graphify MCP + 手動トレース
**集計**: 総クレーム数 40件、うち実バグ 6件、潜在的懸念 8件、誤報告 15件、重複/同類 11件

---

## 目次 {#toc}

1. [総合サマリー](#sec1)
2. [Critical 即時修正推奨](#sec2)
3. [Medium 修正推奨](#sec3)
4. [Minor 軽微](#sec4)
5. [潜在的懸念](#sec5)
6. [誤報告](#sec6)
7. [発信元別マトリクス](#sec7)
8. [修正着手時の注意事項](#sec8)

---

## 1. 総合サマリー {#sec1}

### 1.1 重要度分布

```text
🔴 Critical (即時修正) .. 2件
🟡 Medium (修正推奨) .. 2件
🟢 Minor (軽微) .. 2件
⚠️ 潜在的懸念 .. 8件
❌ 誤報告 .. 15件
```

### 1.2 実バグ一覧（優先度順）

| ID | 重要度 | ファイル | 問題概要 |
| --- | --- | --- | --- |
| BUG-01 | 🔴 Critical | AudioEngine.Processing.DSPCoreDouble.cpp | OutputFilter::process() が convIsLast=true 時に呼ばれず HC/LC フィルターが完全に無効 |
| BUG-02 | 🔴 Critical | MKLNonUniformConvolver.cpp:ringWrite() | overflow ブランチで m_ringWrite が二重更新、リングバッファ不変条件違反 |
| BUG-03 | 🟡 Medium | DeferredDeletionQueue.h:reclaim() | CAS成功後 scanPos = deqPos が旧値代入、1回の reclaim で最大1エントリしか解放されない |
| BUG-04 | 🟡 Medium | DSPCoreDouble.cpp:softClipBlockAVX2() | prevScalar がクリップ済み出力値を読み、インターサンプルピーク検出が過小評価 |
| BUG-05 | 🟢 Minor | SafeStateSwapper.h:kIdleEpoch | コメントは UINT64_MAX と記載するが実際の値は 0 |
| BUG-06 | 🟢 Minor | Fixed15TapNoiseShaper.h:ORDER | クラス名 15Tap だが ORDER=16 |

### 1.3 報告元別 実バグ発見数

| 報告元 | 実バグ | 潜在的懸念 | 誤報告 | 備考 |
| --- | --- | --- | --- | --- |
| Report-1 (ConvoPeq.md解析) | 0 | 3 | 0 | EBR/memory order/forceState |
| Report-2 (ソースコード解析A) | 0 | 1 | 4 | UAF/RCUReader/Allpass等 |
| Report-3 (静的解析A) | 2 | 1 | 2 | reclaim/softClip/UB/プリフェッチ |
| Report-4 (詳細静的解析) | 4 | 1 | 4 | ringWrite/fftWorkBuf/accumBuf/OutputFilter |
| Report-5 (徹底コードレビュー) | 0 | 2 | 4 | ScopedAlignedPtr/Hermitian/ISR |

---

## 2. 🔴 Critical — 即時修正推奨 {#sec2}

### BUG-01: OutputFilter HC/LC が convIsLast=true 時に未適用

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
**報告元**: Report-4 (H-2)

#### 問題の説明

`OutputFilter` クラスは以下の2モードを持つ:

- モード① (convIsLast=true): HCフィルター + LCフィルター（コンボルバー最終段用）
- モード② (convIsLast=false): HPフィルター（固定）+ LPフィルター（EQ最終段用）

しかし呼び出しコードは以下:

```cpp
const bool convIsLast = convActive &&
    (!eqActive || state.order == ProcessingOrder::EQThenConvolver);
if (!convIsLast)
{
    outputFilter.process(processBlock, false,
                         state.convHCMode, state.convLCMode, state.eqLPFMode);
}
```

`convIsLast=true` のとき `if (!convIsLast)` ブロックがスキップされ、モード①が一切適用されない。

#### 検証手順

1. `src/OutputFilter.h` (line 94-) で process() のシグネチャと mode①/②の仕様を確認
2. `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` (line 430-455) で呼び出し箇所を確認
3. `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` (line 246) で同一パターンの存在を確認

#### 修正案

```cpp
// Before
if (!convIsLast)
{
    outputFilter.process(processBlock, false, ...);
}

// After
outputFilter.process(processBlock, convIsLast,
                     state.convHCMode, state.convLCMode, state.eqLPFMode);
```

#### 影響範囲

HC/LC フィルターに依存するすべての出力。コンボルバー使用時（特に EQ bypass 時）に HC/LC 設定が無視される。Float版 (DSPCoreFloat.cpp) も同様のため両方修正が必要。

---

### BUG-02: ringWrite() overflow ブランチで m_ringWrite 二重更新

**ファイル**: `src/MKLNonUniformConvolver.cpp`
**報告元**: Report-4 (C-1)

#### 問題の説明

```cpp
void MKLNonUniformConvolver::ringWrite(const double* src, int n) noexcept
{
    // ... データ書き込み ...

    m_ringWrite = (m_ringWrite + n) & m_ringMask;          // [A] 正しい更新

    const int nextAvail = m_ringAvail + n;
    if (nextAvail > m_ringSize)
    {
        const int overflow = nextAvail - m_ringSize;
        m_ringRead = (m_ringRead + overflow) & m_ringMask;
        m_ringAvail = m_ringSize;
        m_ringWrite = (m_ringWrite + overflow) & m_ringMask; // [D] 二重更新!
    }
}
```

[A] で `m_ringWrite` は既に正しい次書き込み位置に更新されている。overflow 時に [D] でさらに進めると、未読データを次の書き込みで上書きする。

#### トレース検証例

```text
前提: m_ringSize=4, R=0, W=3, Avail=3, n=3 を書き込む

データ書き込み: buf[3]=new[0], buf[0]=new[1], buf[1]=new[2]
[A] W = (3+3)&3 = 2
nextAvail = 6 > 4 → overflow = 2
m_ringRead = (0+2)&3 = 2, Avail = 4
[D] W = (2+2)&3 = 0 ← BUG: 次回 buf[0] に書き込むと new[1] を上書き!
```

#### 発生条件

通常運用（n == l0Part, m_ringSize >= 2 * l0Part）ではオーバーフローは発生しないため latent bug。ブロックサイズ変動、システム過負荷、IR再ロード直後などの境界条件で発現。

#### 修正案

```cpp
    const int nextAvail = m_ringAvail + n;
    if (nextAvail > m_ringSize)
    {
        const int overflow = nextAvail - m_ringSize;
        m_ringRead = (m_ringRead + overflow) & m_ringMask;
        m_ringAvail = m_ringSize;
        // [A] で m_ringWrite は既に正しく更新済み。追加更新は不要かつ有害。
        // m_ringWrite = (m_ringWrite + overflow) & m_ringMask;  // ← 削除
        convo::fetchAddAtomic(m_ringOverflowCount, 1, std::memory_order_acq_rel);
    }
```

---

## 3. 🟡 Medium — 修正推奨 {#sec3}

### BUG-03: DeferredDeletionQueue::reclaim() が1エントリしか解放できない

**ファイル**: `src/DeferredDeletionQueue.h`
**報告元**: Report-3 (Claim 1)

#### 問題の説明

```cpp
if (convo::compareExchangeAtomic(dequeuePos,
                                 deqPos,
                                 static_cast<uint32_t>(deqPos + 1),
                                 std::memory_order_release,
                                 std::memory_order_acquire))
{
    // ... エントリ削除 ...
    convo::publishAtomic(seq_atom, scanPos + kQueueSize, std::memory_order_release);

    scanPos = deqPos;  // BUG: deqPos は CAS 前の旧値のまま
    scanned = 0;
}
```

CAS成功後、dequeuePos は deqPos+1 に進んだが、ローカル変数 deqPos は更新されない。scanPos = deqPos で旧値が代入され、次のループで解放済みスロットを検査 → diff != 0 → break。

#### 影響

1回の reclaim() 呼び出しにつき最大1エントリのみ解放。パラメータ連続変更時（EQバンドの一括設定など）にキューが溢れるリスク。

#### 修正案

```cpp
if (convo::compareExchangeAtomic(dequeuePos, deqPos, deqPos + 1, ...)) {
    // ... エントリ削除 ...
    convo::publishAtomic(seq_atom, scanPos + kQueueSize, ...);

    ++deqPos;           // deqPos を dequeuePos の新値に追従
    scanPos = deqPos;
    scanned = 0;
}
```

---

### BUG-04: softClipBlockAVX2 の prevScalar がクリップ済み出力値を読む

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
**報告元**: Report-3 (Claim 2)

#### 問題の説明

```cpp
result = _mm256_blendv_pd(x, result, needClip);
    _mm256_storeu_pd(data + i, result);

prevScalar = data[i + 3];  // ← クリップ済み出力値!
```

`prevScalar` は次イテレーションのインターサンプルピーク検出で使用される。クリップ済み値を元にピーク推定すると過小評価される。

#### 影響の定量評価

- prevScalar が元の入力値より小さい場合、mid-point 推定値が減少
- ゲインリダクション量が過小になり、クリッピングがやや強くかかる方向にバイアス
- 典型的な音楽信号ではこの影響は -0.5dB 未満と推定
- 報告にある「定期的な波形歪み（グリッチ）」は過大評価

#### 修正案

```cpp
const double nextPrev = data[i + 3];
_mm256_storeu_pd(data + i, result);
prevScalar = nextPrev;
```

---

## 4. 🟢 Minor — 軽微 / コード品質 {#sec4}

### BUG-05: SafeStateSwapper::kIdleEpoch のコメントと実装の矛盾

**ファイル**: `src/SafeStateSwapper.h`
**報告元**: Report-4 (M-1)

- コメント (line 13): `kIdleEpoch (UINT64_MAX)` と記載
- 実装 (line 63): `static constexpr uint64_t kIdleEpoch = 0;`

getMinReaderEpoch() では明示フィルタで除外しているため機能的に問題はないが、将来の変更で誤動作リスク。

### BUG-06: Fixed15TapNoiseShaper::ORDER とクラス名の乖離

**ファイル**: `src/Fixed15TapNoiseShaper.h`
**報告元**: Report-4 (L-1)

```cpp
class Fixed15TapNoiseShaper
{
public:
    static constexpr int ORDER = 16;  // 名称は "15Tap" だが実際は 16 次
};
```

係数配列は16要素（末尾は 0.0）。末尾ゼロ係数によりフィルタ次数は実質15次。機能影響なし。

---

## 5. ⚠️ 潜在的懸念 — 理論的リスクまたはエッジケース {#sec5}

### C-01: EBR epoch 選択 — snapshotRcuEpoch vs publishEpoch

**ファイル**: `src/ConvolverProcessor.Lifecycle.cpp`, `src/ConvolverProcessor.StateAndUI.cpp`
**報告元**: Report-1 (Claim 1)

snapshotRcuEpoch() (= currentEpoch()) も安全。reclaim() の minReaderEpoch 判定がガードするため UAF は発生しない。publishEpoch() への変更は性能改善にはなるが必須ではない。

### C-02: exchangeCurrentState memory order 不統一

**ファイル**: `src/eqprocessor/EQProcessor.Parameters.cpp`
**報告元**: Report-1 (Claim 2)

setBandFrequency のみ acq_rel、他は release。exchange の load 部が relaxed になるが、戻り値 prev は retireEQStateDeferred() に渡すのみで実害なし。統一が望ましい。

### C-03: forceSemanticTransactionState の上書き

**ファイル**: `src/audioengine/AudioEngine.Commit.cpp`
**報告元**: Report-1 (Claim 3)

forceSemanticTransactionState は transitionSemanticTransactionState の CAS 失敗時の hard override。Published → Rejected の上書きは論理的に可能だが、同一スレッド内の逐次処理であり競合しない。

### C-04: LoaderThread callAsync 失敗時

**報告元**: Report-2 (Claim 5)

callAsync 失敗は極めて稀。発生確率は限りなく低い。

### C-05: AudioSegmentBuffer データレース (UB)

**報告元**: Report-3 (Claim 5)

非 atomic 配列に対する RT/UI スレッド間の同時 read/write は C++ メモリモデル上 UB。UI 波形表示用途でありサンプル精度の厳密性は不要。

### C-06: CacheManager CRC64 + TOCTOU

**報告元**: Report-5 (Claim 3)

ファイル読み込み中の変更は理論的にあり得るが、IRファイルは通常不変。

### C-07: PGO + MKL static link の互換性

**報告元**: Report-5 (Claim 5)

MKL static + sequential と MSVC /GL の組み合わせで最適化が壊れる事例が報告されている。

### C-08: EpochDomain enterReader/exitReader memory order window

**ファイル**: `src/core/EpochDomain.h`
**報告元**: Report-2 (Claim 6)

exitReader() で depth 減算前に epoch が kInactiveEpoch でない短いウィンドウ。安全側（解放を遅らせる方向）に作用するため実害なし。

---

## 6. ❌ 誤報告 — 実コードと異なる／誤った解釈 {#sec6}

### F-01: EQCoeffCache 参照カウント不足によるUAF

**報告元**: Report-2 (Claim 1)

CacheMap の copy-on-write 設計ではコピー時に addRef、解放時に release が呼ばれる。get() で取得したポインタは少なくとも1つの CacheMap に保持されている限り有効。**確認**: `src/audioengine/AudioEngine.Cache.cpp`

### F-02: ConvolverProcessor::runtimeRcuReader の m_epochDomain 未初期化

**報告元**: Report-2 (Claim 2)

ConvolverProcessor.h line 1150 に `convo::EpochDomain m_epochDomain;` が存在する。報告は誤り。

### F-03: MKLNonUniformConvolver::freeAll() の fftWorkBuf 解放漏れ

**報告元**: Report-4 (C-2)

実際のソース (line 256-259) には `ippsFree(fftWorkBuf)` が存在する。ConvoPeq.md の古いスナップショットを参照した誤報告。

### F-04: NUC L1/L2 分散計算スキップ

**報告元**: Report-3 (Claim 3)

各入力パーティションは独立した accumBuf 初期化→FFT→FDL格納のサイクルを持つ。設計上正しい。[Bug2 fix] で既知の別問題は修正済み。

### F-05: accumBuf リセットによる前パーティション結果消失

**報告元**: Report-4 (H-1)

F-04 と同根。新しい入力パーティション到着時の accumBuf リセットは設計上正しい動作。

### F-06: ScopedAlignedPtr のデストラクタが非PODデストラクタを呼ばない

**報告元**: Report-5 (Claim 1)

`static_assert(std::is_trivially_destructible_v<T>)` がコンパイル時に非POD型の使用を禁止する。設計上問題なし。

### F-07: applyAllpassToIR Hermitian 対称性再構築不十分

**報告元**: Report-5 (Claim 2)

正側 half-spectrum に乗算後、負側を conj(spec[fftSize-k]) で厳密再構築。DC/Nyquist の虚部も強制ゼロ。Hermitian 条件は完全に満たされている。

### F-08: ISR publication 契約チェック不足

**報告元**: Report-5 (Claim 4)

具体的な契約違反の証拠が提示されていない。RuntimePublicationValidator が existence/authority/semantic completeness の各チェックを実装。

### F-09: r8brain + IPP 非互換

**報告元**: Report-5 (Claim 6)

CMakeLists.txt に「r8brain は IPP を使わず内蔵FFT」と明記。検証不能な意見。

### F-10: スカラーループ内の過剰プリフェッチ

**報告元**: Report-3 (Claim 4)

報告が参照する OutputFilter のスカラーループ内プリフェッチは実コードに存在しない。NUC 分散ループ内の prefetch は適切な粒度で発行。

### F-11: SetImpulse エラー時リソースリーク

**報告元**: Report-2 (Claim 4)

releaseAllLayers() は全レイヤーの全メンバを解放する。エラー時の部分リークは発生しない。

### F-12: AllpassDesigner::applyAllpassToIR NaN/Inf伝播（実害）

**報告元**: Report-2 (Claim 3)

NaN/Inf ガード不足はソース上の穴として存在するが、本関数は src/ 内から 0箇所で呼び出されている（デッドコード）。

### F-13: MixedPhaseOptimizationComponent タイマーコールバック競合

**報告元**: Report-2 (Claim 8)

atomic<float> の load() は安全。pendingOverrideLock 保護下の変数も Message Thread からの呼び出しでは問題ない。

### F-14: LockFreeRingBuffer::pop() のコピー問題

**報告元**: Report-2 (Claim 7)

DeletionEntry は static_assert(std::is_trivially_copyable_v) により静的に保証されている。

### F-15: 未使用変数 AllpassDesigner::clampOptimizationFrequency

**報告元**: Report-2 (Claim 9)

未使用変数は軽微な警告に過ぎない。

---

## 7. 発信元別 判定マトリクス {#sec7}

| 報告元 | 実バグ | 潜在的懸念 | 誤報告 | 評価 |
| --- | --- | --- | --- | --- |
| Report-1 "ConvoPeq.md ソースコード分析" | 0 | 3 | 0 | 妥当な指摘だが重大度低い |
| Report-2 "詳細ソースコード解析" | 0 | 1 | 4 | デッドコードや既修正問題を誤検出、精度低 |
| Report-3 "静的解析" | 2 | 1 | 2 | reclaim/softClip は価値あり、NUCは誤解 |
| Report-4 "詳細静的解析" | 4 | 1 | 4 | OutputFilter/ringWrite は価値高い、fftWorkBufは既修正 |
| Report-5 "徹底コードレビュー" | 0 | 2 | 4 | 最も精度低、POD制約やHermitian対称性を誤解 |

### 総合評価

- **Report-4** が最も価値: 2件のCriticalを発掘
- **Report-3** は reclaim/softClip の2件で貢献
- **Report-1** は妥当だが軽微
- **Report-2** および **Report-5** は誤検出多く精度低

---

## 8. 修正着手時の注意事項 {#sec8}

### 8.1 ConvoPeq.md と実ソースの乖離

ConvoPeq.md には実ソースより古いコードが含まれている。バグ修正は必ず src/ の実ソースを確認してから行うこと。

### 8.2 編集制限の遵守

- /JUCE および /r8brain-free-src 配下は編集禁止
- Audio Thread でのブロッキング禁止6カテゴリの遵守

### 8.3 各バグの独立した検証

各修正後:

1. Build_CMakeTools (Debug/Release) でビルド確認
2. Strict Atomic Dot-Call Scan で atomic 規約違反チェック
3. work21 EpochDomain CI Gate で epoch 関連ゲート通過確認

### 8.4 修正順序の推奨

```text
Step 1: BUG-01 (OutputFilter) → 最も影響大、リスク低
Step 2: BUG-02 (ringWrite) → 修正範囲限定、リスク低
Step 3: BUG-03 (DeferredDeletionQueue) → 影響範囲中
Step 4: BUG-04 (softClipBlockAVX2) → 修正範囲限定
Step 5: BUG-05, BUG-06 → コード品質
```

---

## 付録A: 検証に使用したツール {#appendix-a}

| ツール | 用途 | 使用法 |
| --- | --- | --- |
| grep/Select-String | シンボル検索 | Select-String -Pattern "symbol" -Path "src/**" |
| serena MCP | シンボル定義/参照検索 | find_symbol / get_symbols_overview |
| ccc | ASTベースコード検索 | ccc search "query" |
| semble | 参照解析/呼び出し元検出 | semble path --ref symbol --content code |
| CodeGraph MCP | 関数呼び出し関係/モジュール構造 | mcp_codegraph-mcp_find_callees |
| graphify MCP | 知識グラフ/アーキテクチャ探索 | mcp_graphify_get_node |
| 直接ファイル読取 | 実装詳細確認 | read_file (src/ 内該当ソース) |

## 付録B: バグ重要度判定基準 {#appendix-b}

| 重要度 | 定義 |
| --- | --- |
| 🔴 Critical | 機能不全・誤動作。ユーザー設定無視、データ破損の可能性 |
| 🟡 Medium | 特定条件下で性能低下や軽微な品質劣化 |
| 🟢 Minor | 機能的影響なし、コード品質・保守性の不整合 |
| ⚠️ 潜在的懸念 | 理論的リスクは存在するが現実的な影響未確認 |
| ❌ 誤報告 | 実コードと異なる、既修正、または誤った解釈 |

---

*報告書作成: 2026-06-08*
*検証者: GitHub Copilot (DeepSeek V4 Flash)*
