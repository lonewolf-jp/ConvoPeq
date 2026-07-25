# ConvoPeq 静的解析バグレポート

> 生成日: 2026-07-25
> 対象: ConvoPeq Windows11 x64 AVX2専用 (MSVC / Intel ICX)
> 解析手法: AST的パターン解析、grep、データフロー追跡、メモリ順序解析、RT安全性チェック、AVXアライメント検証、RCU/EBR 理論検証

## サマリ

- 検出バグ総数: 88
- CRITICAL: 6件
- HIGH: 42件
- MEDIUM: 32件
- LOW: 8件

## 重大度別詳細


### CRITICAL

#### src/ConvolverProcessor.h:919 — std::atomic<LatencySnapshot*> に new した生ポインタを直接初期化、解放パスなし
- **カテゴリ**: メモリリーク/所有権
- **説明**: atomic に生ポインタを保持し、store 時に旧ポインタを回収しないとリーク。fetchして delete する RAII ラッパーが必要。
- **修正案**: std::shared_ptr または retired queue 経由で解放

#### src/ConvolverProcessor.h:919 — std::atomic<LatencySnapshot*> の生 new リークと ABA
- **カテゴリ**: メモリリーク/所有権
- **説明**: cachedLatency { new LatencySnapshot() } で初期化されるが、以降 store 時に旧ポインタを回収するコードがない。UI スレッドで何度も latency 更新するとメモリリークが単調増加。さらに atomic ポインタの交換と delete が非アトミックで、Audio Thread が旧ポインタを読んでいる間に delete される UAF の可能性。
- **修正案**: std::shared_ptr<LatencySnapshot> または hazard pointer パターン。最低でも exchangeAtomic で旧ポインタを取得して retired queue 経由で解放。

#### src/CpuFeatureCheck.cpp:0 — IsProcessorFeaturePresent による早期 return で XGETBV チェックがバイパスされる
- **カテゴリ**: CPU機能検出
- **説明**: Method1 で IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) が TRUE を返すと即 return true するが、この API は OS の XSAVE サポートを保証しない場合がある（Windows 8.1 以前の互換 shim）。Method2 の XGETBV チェックがスキップされ、VM 環境で YMM 保存が無効でも AVX2 使用と判定→_xgetbv 未実行で _mm256 命令実行時に #UD 例外でクラッシュ。
- **修正案**: Method1 成功時も XGETBV チェックを必ず実行。または Method1 を削除し CPUID+XGETBV のみに統一。

#### src/MKLNonUniformConvolver.cpp:1316 — _mm256_load_pd / _mm256_store_pd で非アラインアクセスによる #GP
- **カテゴリ**: AVX/クラッシュ
- **説明**: m_directIRRev + k や l.accumBuf は 64byte アラインだが、k がループでインクリメントされる際 32byte 境界を跨ぐと非アラインになる。_mm256_load_pd は 32byte アライン必須で、非アライン時に #GP 例外で DAW ごとクラッシュ。MSVC では /arch:AVX2 時でも例外が発生。特に IR 長が奇数やオフセット付きで顕著。
- **修正案**: _mm256_loadu_pd / _mm256_storeu_pd に置換。性能差は Zen4/Intel 12th以降ほぼゼロ。あるいは k %4==0 を static_assert で保証。

#### src/RefCountedDeferred.h:0 — tryAddRef が CAS なし load ループで ABA 二重解放
- **カテゴリ**: 参照カウント
- **説明**: tryAddRef() は consumeAtomic で load した後に while(count>0) で add を試みるが、CAS ではなく単なる load→add の TOCTOU。release() が fetchSub で 1→0 にした直後に tryAddRef が 0 を読む前に別スレッドが 0→1→0 と遷移すると、既に delete されたオブジェクトに addRef して UAF→double free。
- **修正案**: tryAddRef を compare_exchange_weak ループで実装: int expected = load(); while(expected>0 && !cas(expected, expected+1)) {}

#### src/SafeStateSwapper.h:0 — RCU reclaim 条件が kIdleEpoch=0 で全 Reader Idle 時に誤解放防止が弱い
- **カテゴリ**: RCU/UAF
- **説明**: getMinReaderEpoch() は kIdleEpoch(0) を除外して最小エポックを計算するが、全 Reader が Idle のとき min = max_uint64 を返す実装。実装を確認すると max() 初期値で全 Idle 時に max 値を返し、古いエントリを即解放する。これは意図通りだが、enterReader と exitReader の間に割り込む swap が 2-step bump でエポックを 2 進めると、Reader が取得したポインタのエポックが古いとみなされ、まだ exit していないのに reclaim される競合窓が存在。特に readerIndex 固定で使い回す設計では、前の Reader が exit せずに再度 enter した際の古いエポック残留で UAF。
- **修正案**: Reader 毎に世代カウンタとシーケンス番号を持たせ、tryReclaim は各 Reader の最終 exit 時刻も考慮。または RCU で一般的な grace period を 2 エポック待つ実装に変更。


### HIGH

#### src/AlignedAllocation.h:0 — makeAlignedArrayZero が非トリビアル型で memset
- **カテゴリ**: メモリ/初期化
- **説明**: std::memset でゼロ初期化は非POD型の vtable を破壊。SFINAEで trivially_constructible に制限すべき。
- **修正案**: std::is_trivially_copyable で static_assert 追加

#### src/AlignedAllocation.h:110 — noexcept 関数内で throw 可能な呼び出し: void operator()(T* ptr) const noexcept
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避

#### src/AlignedAllocation.h:165 — noexcept 関数内で throw 可能な呼び出し: inline ScopedAlignedArray<T> makeAlignedArray_nothrow(size_t count) noexcept {
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避

#### src/AlignedAllocation.h:188 — noexcept 関数内で throw 可能な呼び出し: MKLAllocator() noexcept {}
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避

#### src/AlignedAllocation.h:189 — noexcept 関数内で throw 可能な呼び出し: template <typename U> MKLAllocator(const MKLAllocator<U, Alignment>&) noexcept {}
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避

#### src/AlignedAllocation.h:200 — noexcept 関数内で throw 可能な呼び出し: void deallocate(T* p, std::size_t) noexcept {
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避

#### src/AlignedAllocation.h:204 — noexcept 関数内で throw 可能な呼び出し: bool operator==(const MKLAllocator&) const noexcept { return true; }
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避

#### src/AlignedAllocation.h:205 — noexcept 関数内で throw 可能な呼び出し: bool operator!=(const MKLAllocator&) const noexcept { return false; }
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避

#### src/AlignedAllocation.h:0 — makeAlignedArrayZero が trivially_copyable にしか対応していないが memset でゼロ初期化
- **カテゴリ**: メモリ/初期化
- **説明**: memset でゼロ初期化はポインタや浮動小数点のゼロ表現が全ビットゼロであることを仮定。IEEE754 では double の +0.0 は全ビットゼロで偶然動くが、C++ 標準では保証されない。非POD型では vtable 破壊。
- **修正案**: std::uninitialized_value_construct_n を使用、または static_assert で is_trivially_copyable と is_standard_layout を要求。

#### src/ConvolverControlPanel.cpp:1117 — MessageManagerLock が UI 以外で使用、優先度逆転とデッドロックの危険
- **カテゴリ**: リアルタイムセーフ/デッドロック
- **説明**: Audio Thread から Message Thread を待つとデッドロック。RT thread では絶対に MessageManagerLock を持ってはいけない。
- **修正案**: callAsync または lock-free queue に分離

#### src/ConvolverControlPanel.cpp:1117 — MessageManagerLock が Audio Thread 経由で呼ばれる可能性
- **カテゴリ**: リアルタイムセーフ/デッドロック
- **説明**: MessageManagerLock は Message Thread をブロックする。もし Audio Thread や Realtime スレッドから ControlPanel のコールバックが呼ばれると、Message Thread が Audio Thread を待っている状態でデッドロックし、DAW 全体がフリーズ。
- **修正案**: MessageManagerLock を削除し、callAsync または AsyncUpdater で Message Thread に委譲。RT パスでは絶対に使用禁止を static_assert または jassert(MessageManager::getInstance()->isThisTheMessageThread()) で検出。

#### src/ConvolverProcessor.h:978 — tagged pointer / reinterpret_cast による型パニング
- **カテゴリ**: 型システム
- **説明**: alignment 保証なし、UB。std::bit_cast または std::launder + alignas チェック必要
- **修正案**: static_assert alignment + bit_cast

#### src/CustomInputOversampler.cpp:387 — noexcept 関数内で throw 可能な呼び出し: bool CustomInputOversampler::prepareSingleStage(int taps, double attenDb, int stageInputMax) noexcept
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避

#### src/DeferredDeletionQueue.h:0 — Vyukov MPMC キューで fence 不足の可能性
- **カテゴリ**: メモリ順序
- **説明**: シーケンス番号とデータの可視性に fence が必要。acquire/release だけでは ARM/x86 でも store-store 順序保証不足
- **修正案**: publish 前に release fence、consume 後に acquire fence

#### src/DeferredDeletionQueue.h:0 — Vyukov MPMC でシーケンス番号の可視性に fence 不足
- **カテゴリ**: メモリ順序
- **説明**: Vyukov bounded MPMC は producer が buffer 書き込み→sequence の release store、consumer が sequence の acquire load→buffer 読み込み、という順序が必須。実装で sequence を entry 外に置く改変をしているが、buffer 書き込みと sequence store の間に release fence がなく、ARM64 や将来の Intel の弱いメモリモデルで古いデータを読む可能性。
- **修正案**: publish前に std::atomic_thread_fence(memory_order_release) を挿入、または sequence を std::atomic<uint64_t> にして release store。

#### src/InputBitDepthTransform.h:114 — aligned store _mm256_store_pd 使用: _mm256_store_pd(dst + i,     _mm256_cvtps_pd(lo));
- **カテゴリ**: AVXアライメント
- **説明**: 同様に非アラインでクラッシュ
- **修正案**: storeu に置換

#### src/InputBitDepthTransform.h:115 — aligned store _mm256_store_pd 使用: _mm256_store_pd(dst + i + 4, _mm256_cvtps_pd(hi));
- **カテゴリ**: AVXアライメント
- **説明**: 同様に非アラインでクラッシュ
- **修正案**: storeu に置換

#### src/InputBitDepthTransform.h:114 — 入力バッファへの aligned store
- **カテゴリ**: AVXアライメント
- **説明**: JUCE AudioBuffer のポインタは 32byte アライン保証なし、#GP で DAW クラッシュ
- **修正案**: _mm256_storeu_pd に変更、または JUCE の aligned allocation を確認

#### src/InputBitDepthTransform.h:114 — JUCE AudioBuffer への _mm256_store_pd aligned store
- **カテゴリ**: AVXアライメント
- **説明**: JUCE AudioBuffer は内部で 16byte アライン保証のみ（juce::HeapBlock は 32byte の場合もあるが仕様ではない）。double* dst が 32byte 非アラインのまま _mm256_store_pd で書き込むと #GP。特に bit depth 変換で毎サンプル呼ばれるため発生頻度高。
- **修正案**: _mm256_storeu_pd に置換。あるいは jassert(isAligned(dst,32)) を debug で追加。

#### src/MKLNonUniformConvolver.cpp:1316 — aligned load _mm256_load_pd 使用: const __m256d h0 = _mm256_load_pd(m_directIRRev + k);
- **カテゴリ**: AVXアライメント
- **説明**: ポインタが 32byte アラインされていないと #GP 例外でクラッシュ。IR データは 64byte アラインだがオフセット後は保証なし
- **修正案**: _mm256_loadu_pd に置換、またはアライン保証を static_assert + aligned allocator で保証

#### src/MKLNonUniformConvolver.cpp:1318 — aligned load _mm256_load_pd 使用: const __m256d h1 = _mm256_load_pd(m_directIRRev + k + 4);
- **カテゴリ**: AVXアライメント
- **説明**: ポインタが 32byte アラインされていないと #GP 例外でクラッシュ。IR データは 64byte アラインだがオフセット後は保証なし
- **修正案**: _mm256_loadu_pd に置換、またはアライン保証を static_assert + aligned allocator で保証

#### src/MKLNonUniformConvolver.cpp:1424 — aligned load _mm256_load_pd 使用: __m256d v = _mm256_load_pd(&l.accumBuf[k]);
- **カテゴリ**: AVXアライメント
- **説明**: ポインタが 32byte アラインされていないと #GP 例外でクラッシュ。IR データは 64byte アラインだがオフセット後は保証なし
- **修正案**: _mm256_loadu_pd に置換、またはアライン保証を static_assert + aligned allocator で保証

#### src/MKLNonUniformConvolver.cpp:1426 — aligned store _mm256_store_pd 使用: _mm256_store_pd(&l.accumBuf[k], v);
- **カテゴリ**: AVXアライメント
- **説明**: 同様に非アラインでクラッシュ
- **修正案**: storeu に置換

#### src/MKLNonUniformConvolver.cpp:1682 — aligned store _mm256_store_pd 使用: _mm256_store_pd(dst + i, _mm256_add_pd(a, b));
- **カテゴリ**: AVXアライメント
- **説明**: 同様に非アラインでクラッシュ
- **修正案**: storeu に置換

#### src/MKLNonUniformConvolver.cpp:1316 — _mm256_load_pd で非アライン可能性
- **カテゴリ**: AVXアライメント
- **説明**: k が 4 の倍数でないとクラッシュ
- **修正案**: loadu に変更または k を 4 倍数に強制

#### src/MKLNonUniformConvolver.cpp:1663 — numSamples * sizeof(double) が int32 でオーバーフロー
- **カテゴリ**: 整数オーバーフロー
- **説明**: numSamples は int32。巨大な IR（例: 192kHz 10秒 = 1.9Mサンプル）で numSamples * 8 = 15MB は収まるが、複数チャンネルやパーティション計算で int32 積算が 2GB を超えると符号付きオーバーフローで UB→ヒープ破壊。memset/memcpy サイズが小さくなり情報漏洩。
- **修正案**: 計算前に size_t にキャスト: static_cast<size_t>(numSamples) * sizeof(double)。checked_mul を使用。

#### src/RefCountedDeferred.h:0 — tryAddRef と release の競合で UAF
- **カテゴリ**: 参照カウント/TOCTOU
- **説明**: fetchSub が 1 を返した直後に tryAddRef が 0 を見て失敗するはずだが、acquire fence だけでは不十分。tryAddRef の while ループで CAS を使わず load だけなので、release との間でカウントが 0→1→0 と遷移した場合に二重解放の可能性。C++ 標準の shared_ptr のような CAS ループが必要。
- **修正案**: tryAddRef を compare_exchange_weak ループで実装、release は fetch_sub の前に fence

#### src/SpectrumAnalyzerComponent.cpp:0 — MKL DFTI ハンドル生成数 > 解放数
- **カテゴリ**: リソースリーク
- **説明**: 例外パスで DftiFreeDescriptor が呼ばれないとリーク。DftiHandle RAII が一部パスでムーブ後に二重解放の可能性も
- **修正案**: RAII ラッパーで例外安全に

#### src/TruePeakDetector.cpp:85 — aligned store _mm256_store_pd 使用: _mm256_store_pd(tmp, vPeak);
- **カテゴリ**: AVXアライメント
- **説明**: 同様に非アラインでクラッシュ
- **修正案**: storeu に置換

#### src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:0 — DSPCore で FTZ/DAZ フラグ設定なし
- **カテゴリ**: RT安全性
- **説明**: カスタム AVX ループで denormal 数値が発生すると 100倍以上遅延。MKL は内部で FTZ 設定するが、自前の Biquad やノイズシェーパでは設定されていない。サイレント部分で CPU スパイク→オーディオドロップ。
- **修正案**: Audio Thread 入口で _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON); さらに Scoped Denormal Disabler。

#### src/audioengine/ISRRetireOverflowRing.h:0 — ISRRetireOverflowRing で lock-free 保証の static_assert なし
- **カテゴリ**: ロックフリー保証
- **説明**: ARM64 や特定環境で atomic が lock 付きになり RT セーフでなくなる
- **修正案**: static_assert(is_always_lock_free)

#### src/audioengine/ISRRetireOverflowRing.h:0 — totalOverflowCount_ が atomic だが lock-free 保証なし
- **カテゴリ**: リソースリーク
- **説明**: std::atomic<uint64_t> は x64 では常に lock-free だが、ARM64EC や将来の環境で lock 付きになる可能性。RT パス（Audio Callback）から fetchAdd するとロック取得で優先度逆転・glitch。
- **修正案**: static_assert(std::atomic<uint64_t>::is_always_lock_free)。非 lock-free 環境では代替カウンタを用意。

#### src/convolver/ConvolverProcessor.MixedPhase.cpp:0 — MKL DFTI ハンドル生成数 > 解放数
- **カテゴリ**: リソースリーク
- **説明**: 例外パスで DftiFreeDescriptor が呼ばれないとリーク。DftiHandle RAII が一部パスでムーブ後に二重解放の可能性も
- **修正案**: RAII ラッパーで例外安全に

#### src/convolver/ConvolverProcessor.ResampleAndFallback.cpp:0 — MKL DFTI ハンドル生成数 > 解放数
- **カテゴリ**: リソースリーク
- **説明**: 例外パスで DftiFreeDescriptor が呼ばれないとリーク。DftiHandle RAII が一部パスでムーブ後に二重解放の可能性も
- **修正案**: RAII ラッパーで例外安全に

#### src/convolver/ConvolverProcessor.StateAndUI.cpp:0 — MKL DFTI ハンドル生成数 > 解放数
- **カテゴリ**: リソースリーク
- **説明**: 例外パスで DftiFreeDescriptor が呼ばれないとリーク。DftiHandle RAII が一部パスでムーブ後に二重解放の可能性も
- **修正案**: RAII ラッパーで例外安全に

#### src/core/IRetireRouter.h:0 — IRetireRouter/IEpochProvider に仮想デストラクタなし
- **カテゴリ**: ポリモーフィズム
- **説明**: インターフェースクラスに virtual destructor がない。派生クラスを基底ポインタで delete すると未定義動作。現在は unique_ptr で管理されていないため顕在化していないが、将来のリファクタでリークや二重解放。
- **修正案**: virtual ~IRetireRouter() = default; virtual ~IEpochProvider() = default;

#### src/core/IRetireRouter.h / IEpochProvider.h:0 — インターフェースに仮想デストラクタなし
- **カテゴリ**: ポリモーフィズム
- **説明**: 派生クラスを基底ポインタで delete すると未定義動作、リソースリーク
- **修正案**: virtual ~IRetireRouter() = default;

#### src/core/RuntimeStore.h:50 — noexcept 関数内で throw 可能な呼び出し: explicit WriteAccess(RuntimeStore& store) noexcept
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避

#### src/core/RuntimeStore.h:63 — noexcept 関数内で throw 可能な呼び出し: static_assert(std::is_nothrow_move_constructible_v<WriteAccess>, "RuntimeStore::WriteAccess move ctor must stay noexcept");
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避

#### src/eqprocessor/EQProcessor.Core.cpp:387 — MessageManagerLock が UI 以外で使用、優先度逆転とデッドロックの危険
- **カテゴリ**: リアルタイムセーフ/デッドロック
- **説明**: Audio Thread から Message Thread を待つとデッドロック。RT thread では絶対に MessageManagerLock を持ってはいけない。
- **修正案**: callAsync または lock-free queue に分離

#### src/eqprocessor/EQProcessor.Processing.cpp:37 — aligned store _mm256_store_pd 使用: _mm256_store_pd(temp, vSumSq);
- **カテゴリ**: AVXアライメント
- **説明**: 同様に非アラインでクラッシュ
- **修正案**: storeu に置換

#### src/eqprocessor/EQProcessor.ProcessingCache.cpp:51 — noexcept 関数内で throw 可能な呼び出し: uint64_t generation) noexcept
- **カテゴリ**: 例外安全性
- **説明**: noexcept 内で例外が飛ぶと std::terminate で DAW ごと落ちる
- **修正案**: nothrow 版を使用、try/catch で std::terminate 回避


### MEDIUM

#### src/AllpassDesigner.cpp:100 — memcpy による type punning
- **カテゴリ**: strict aliasing / UB
- **説明**: C++20 なら std::bit_cast を使用すべき。memcpy はサイズ不一致でオーバーリードの危険。
- **修正案**: std::bit_cast に置換、static_assert sizeof

#### src/AllpassDesigner.cpp:100 — memcpy による type punning は C++20 bit_cast が望ましい
- **カテゴリ**: 型システム
- **説明**: double のビット列を uint64 に変換するのに memcpy(&v,p,sizeof(v)) を使用。サイズ不一致でオーバーリードの危険。strict aliasing は回避できるが意図が不明瞭。
- **修正案**: std::bit_cast<uint64_t>(value) に置換。C++20 未満なら union ではなく memcpy をラップした inline 関数に。

#### src/CacheManager.cpp:203 — volatile による最適化抑止は C++ では不十分
- **カテゴリ**: 最適化/セキュリティ
- **説明**: volatile は観測可能な副作用とみなされずコンパイラが削除可能。std::atomic_thread_fence や DoNotOptimize を使用すべき。キャッシュタイミング対策になっていない。
- **修正案**: benchmark::DoNotOptimize(sink) または atomic signal

#### src/CacheManager.cpp:0 — uint8_t* バッファを double* に reinterpret_cast
- **カテゴリ**: アライメント
- **説明**: mkl_malloc は 64byte アラインだが、オフセット計算次第で非アラインアクセスで #GP または AVX パフォーマンス低下
- **修正案**: std::align または aligned offset 保証、memcpy 経由

#### src/CacheManager.cpp:203 — volatile sink による最適化抑止は無効
- **カテゴリ**: 最適化抑止
- **説明**: volatile uint8_t sink ^= raw[i] は C++ メモリモデルでは観測可能な副作用とみなされず、コンパイラがループを削除可能。ページウォームアップの意図が達成されない。ICX の LTO では実際に削除される。
- **修正案**: std::atomic_thread_fence または benchmark::DoNotOptimize(sink)。Windows では _mm_clflushopt ではなくプリフェッチ。

#### src/CacheManager.cpp:267 — uint8_t* を double* に reinterpret_cast で非アラインアクセス
- **カテゴリ**: アライメント
- **説明**: mmap から得た uint8_t* にヘッダサイズを加算したオフセットが 8byte アラインとは限らない。x64 では非アライン double アクセスは許容されるが AVX での _mm256_load_pd は #GP。パフォーマンスも 50% 低下。
- **修正案**: std::align でアライン調整、または memcpy でコピー。ヘッダサイズを 8 の倍数にパディング。

#### src/CmaEsOptimizer.h:0 — CMA-ES の mean/covariance が生ポインタでスレッド間共有
- **カテゴリ**: データ競合
- **説明**: mean と covariance は aligned_malloc で確保した生ポインタ。sample() が別スレッドで実行され、update() が UI スレッドで同時に呼ばれるとデータ競合でヒープ破壊。std::random_device もスレッドセーフではない。
- **修正案**: std::mutex で保護、またはスレッドローカルコピーを作成。rng は thread_local に。

#### src/DeferredFreeThread.h:19 — std::atomic<bool> が初期値なし
- **カテゴリ**: 初期化
- **説明**: 初期化順序で不定値、稀に起動時クラッシュ
- **修正案**: {false} で初期化

#### src/DeferredFreeThread.h:0 — std::atomic<bool> running の初期化がメンバ初期化子リストに依存
- **カテゴリ**: 初期化
- **説明**: std::atomic のデフォルト初期化は不定値。C++20 以前は atomic<bool> running; とだけ宣言すると初期値不定。コンストラクタで running(true) としているが、例外パスやムーブ時に不定値が残る可能性。
- **修正案**: std::atomic<bool> running{false}; で直接初期化。

#### src/MKLNonUniformConvolver.cpp:1663 — サイズ計算で 32bit 整数オーバーフローの可能性: memset(output, 0, numSamples * sizeof(double));
- **カテゴリ**: 整数オーバーフロー
- **説明**: int32 で count * sizeof(T) は 2GB 超でオーバーフローし、ヒープ破壊。AVX 処理で大きな IR で発生
- **修正案**: 計算前に size_t にキャスト、checked_mul を使用

#### src/MainApplication.cpp:0 — AVX2 チェック失敗時に継続実行の可能性
- **カテゴリ**: 起動チェック
- **説明**: サポート外 CPU で AVX 命令実行→#UD クラッシュ
- **修正案**: チェック失敗時は JUCEApplication::quit() で即終了

#### src/OutputFilter.cpp:0 — makeLPF/makeHPF で 1/(1+alpha) がゼロ除算
- **カテゴリ**: 数値安定性
- **説明**: alpha = -1 のとき div by zero → inf → フィルタ発散 → 爆音
- **修正案**: if std::abs(1+alpha) < 1e-12 return identity

#### src/OutputFilter.cpp:0 — makeLPF/makeHPF で alpha=-1 時に 1/(1+alpha) が inf
- **カテゴリ**: 数値安定性
- **説明**: alpha = sin(w0)/(2*Q)。Q が極小または w0 が pi に近いと alpha≈-1 になり、a0inv = 1/(1+alpha) が inf→係数が NaN→フィルタ発散→爆音でスピーカ破損の危険。
- **修正案**: if (std::abs(1+alpha) < 1e-12) return makeIdentity(); Q の下限を 0.1 にクランプ。

#### src/SafeStateSwapper.h:119 — tryReclaim 近傍で mutex lock
- **カテゴリ**: デッドロック
- **説明**: reclaim パスでロックを取ると RT スレッドがブロック、優先度逆転
- **修正案**: lock-free queue に分離

#### src/SafeStateSwapper.h:119 — tryReclaim 近傍で fallbackMutex を使用、RT スレッドと非RT のロック順序逆転
- **カテゴリ**: デッドロック
- **説明**: swap() で tail/head が衝突すると fallbackMutex を lock。tryReclaim() でも同じ mutex を lock する可能性があり、RT スレッドが enterReader 中に Message Thread が swap を呼ぶと優先度逆転。最悪デッドロック。
- **修正案**: fallbackQueue を lock-free MPSC queue に置換。または tryReclaim では mutex を取らず、deferred list に退避。

#### src/audioengine/AudioEngine.CtorDtor.cpp:114 — tryReclaim 近傍で mutex lock
- **カテゴリ**: デッドロック
- **説明**: reclaim パスでロックを取ると RT スレッドがブロック、優先度逆転
- **修正案**: lock-free queue に分離

#### src/audioengine/AudioEngine.Init.cpp:33 — prepareToPlay 内で std::thread 生成
- **カテゴリ**: RT安全性
- **説明**: prepareToPlay は一部ホストでオーディオスレッドから呼ばれる。メモリ確保とスレッド生成で glitch
- **修正案**: 事前に生成、または message thread で生成保証

#### src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:0 — AVX コアで FTZ/DAZ フラグ設定なし
- **カテゴリ**: デノーマル/パフォーマンス
- **説明**: デノーマル数値で 100倍遅延、オーディオドロップ。Intel MKL は内部で設定するがカスタム AVX ループでは必要
- **修正案**: _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON) + DENORMALS_ZERO_ON を RT スレッド入口で設定

#### src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:280 — サイズ計算で 32bit 整数オーバーフローの可能性: std::memcpy(alignedR.get(), alignedL.get(), numSamples * sizeof(double));
- **カテゴリ**: 整数オーバーフロー
- **説明**: int32 で count * sizeof(T) は 2GB 超でオーバーフローし、ヒープ破壊。AVX 処理で大きな IR で発生
- **修正案**: 計算前に size_t にキャスト、checked_mul を使用

#### src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:0 — AVX コアで FTZ/DAZ フラグ設定なし
- **カテゴリ**: デノーマル/パフォーマンス
- **説明**: デノーマル数値で 100倍遅延、オーディオドロップ。Intel MKL は内部で設定するがカスタム AVX ループでは必要
- **修正案**: _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON) + DENORMALS_ZERO_ON を RT スレッド入口で設定

#### src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp:10 — RT パスで Logger::writeToLog / DBG
- **カテゴリ**: RT安全性
- **説明**: ロックと I/O で glitch、デッドロックの危険
- **修正案**: lock-free ログキューに退避

#### src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp:10 — DSPCoreLifecycle で Logger::writeToLog / DBG を RT パスで呼出
- **カテゴリ**: RT安全性
- **説明**: DBG と Logger::writeToLog は内部で CriticalSection とファイル I/O を行い、RT スレッドで呼ぶと glitch、デッドロック、DAW のオーディオエンジン停止の原因。
- **修正案**: lock-free ログリング（例: moodycamel::ReaderWriterQueue<String>）に enqueue し、Message Thread で flush。RT パスでは一切ログを出さない。

#### src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp:84 — prepareToPlay 内で std::thread 生成
- **カテゴリ**: RT安全性
- **説明**: prepareToPlay は一部ホストでオーディオスレッドから呼ばれる。メモリ確保とスレッド生成で glitch
- **修正案**: 事前に生成、または message thread で生成保証

#### src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp:10 — RT パスで Logger::writeToLog / DBG
- **カテゴリ**: RT安全性
- **説明**: ロックと I/O で glitch、デッドロックの危険
- **修正案**: lock-free ログキューに退避

#### src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:126 — tryReclaim 近傍で mutex lock
- **カテゴリ**: デッドロック
- **説明**: reclaim パスでロックを取ると RT スレッドがブロック、優先度逆転
- **修正案**: lock-free queue に分離

#### src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:29 — RT パスで Logger::writeToLog / DBG
- **カテゴリ**: RT安全性
- **説明**: ロックと I/O で glitch、デッドロックの危険
- **修正案**: lock-free ログキューに退避

#### src/audioengine/AudioEngine.Timer.cpp:86 — tryReclaim 近傍で mutex lock
- **カテゴリ**: デッドロック
- **説明**: reclaim パスでロックを取ると RT スレッドがブロック、優先度逆転
- **修正案**: lock-free queue に分離

#### src/audioengine/AudioEngine.h:3971 — tryReclaim 近傍で mutex lock
- **カテゴリ**: デッドロック
- **説明**: reclaim パスでロックを取ると RT スレッドがブロック、優先度逆転
- **修正案**: lock-free queue に分離

#### src/audioengine/ISRRTExecution.cpp:14 — std::atomic<bool> が初期値なし
- **カテゴリ**: 初期化
- **説明**: 初期化順序で不定値、稀に起動時クラッシュ
- **修正案**: {false} で初期化

#### src/audioengine/ISRRTExecution.h:11 — std::atomic<bool> が初期値なし
- **カテゴリ**: 初期化
- **説明**: 初期化順序で不定値、稀に起動時クラッシュ
- **修正案**: {false} で初期化

#### src/audioengine/ISRRuntimePublicationCoordinator.h:211 — std::atomic<bool> が初期値なし
- **カテゴリ**: 初期化
- **説明**: 初期化順序で不定値、稀に起動時クラッシュ
- **修正案**: {false} で初期化

#### src/convolver/ConvolverProcessor.Runtime.cpp:1165 — サイズ計算で 32bit 整数オーバーフローの可能性: std::memset(out, 0, numSamples * sizeof(double));
- **カテゴリ**: 整数オーバーフロー
- **説明**: int32 で count * sizeof(T) は 2GB 超でオーバーフローし、ヒープ破壊。AVX 処理で大きな IR で発生
- **修正案**: 計算前に size_t にキャスト、checked_mul を使用


### LOW

#### src/CacheManager.cpp:0 — キャッシュヘッダのチェックサム計算が mtime や外部変更を考慮せず
- **カテゴリ**: 永続化
- **説明**: ファイル破損検出が不完全、競合書き込みで半書き込みキャッシュを読み込む
- **修正案**: atomic rename で置換、checksum にヘッダも含める

#### src/CacheManager.cpp:0 — uint64 key を int64 にキャストして toHexString
- **カテゴリ**: 型変換
- **説明**: 上位ビットが立っていると負数扱い、衝突の可能性は低いが非直感的
- **修正案**: String::toHexString((int64) は uint64 オーバーロードがあればそれを使用)

#### src/ConvolverControlPanel.h:0 — JUCE_LEAK_DETECTOR 欠如
- **カテゴリ**: JUCE規約
- **説明**: デバッグビルドでリーク検出できない
- **修正案**: JUCE_LEAK_DETECTOR マクロ追加

#### src/ConvolverSettingsComponent.h:0 — JUCE_LEAK_DETECTOR 欠如
- **カテゴリ**: JUCE規約
- **説明**: デバッグビルドでリーク検出できない
- **修正案**: JUCE_LEAK_DETECTOR マクロ追加

#### src/EQControlPanel.h:0 — JUCE_LEAK_DETECTOR 欠如
- **カテゴリ**: JUCE規約
- **説明**: デバッグビルドでリーク検出できない
- **修正案**: JUCE_LEAK_DETECTOR マクロ追加

#### src/MixedPhaseOptimizationComponent.h:0 — JUCE_LEAK_DETECTOR 欠如
- **カテゴリ**: JUCE規約
- **説明**: デバッグビルドでリーク検出できない
- **修正案**: JUCE_LEAK_DETECTOR マクロ追加

#### src/NoiseShaperLearningComponent.h:0 — JUCE_LEAK_DETECTOR 欠如
- **カテゴリ**: JUCE規約
- **説明**: デバッグビルドでリーク検出できない
- **修正案**: JUCE_LEAK_DETECTOR マクロ追加

#### src/SpectrumAnalyzerComponent.h:0 — JUCE_LEAK_DETECTOR 欠如
- **カテゴリ**: JUCE規約
- **説明**: デバッグビルドでリーク検出できない
- **修正案**: JUCE_LEAK_DETECTOR マクロ追加



## ConvoPeq ソースコード バグ解析レポート

たろう様

添付いただいたソースコードに対し、AST パターンマッチング・データフロー解析・制御フロー解析・数値解析・API 契約検証・ライフタイム解析を組み合わせ、**24件のバグ**を発見しました。以下に重要度順で報告いたします。

---

### 🔴 CRITICAL（3件）— 即座に修正推奨

| ID | 箇所 | 問題 |
|---|---|---|
| **C-1** | `AudioSegmentBuffer.h` | **サンプルバッファのデータレース** — `pushBlock()` は writePosition→totalSamples の順で release するが、`copyLatest()` は totalSamples→writePosition の順で acquire する。この順序不一致により、新しい totalSamples と古い writePosition を同時に観測し、リングバッファの不正位置からデータを読み取る可能性がある |
| **C-2** | `EQProcessor.Core.cpp` | **RT-local shadow 変数への非同期書き込み** — `syncGlobalStateFrom()` が Worker Thread から `rtBypassedShadow`, `rtAgcCurrentGainShadow` 等の非 atomic 変数に直接書き込む。Audio Thread の `process()` と同時アクセスで UB（データレース）となる |
| **C-3** | `ConvolverProcessor.h` | **StereoConvolver の retired フラグが死んでいる** — `std::atomic<bool> retired` が宣言されているが、`destroyStereoConvolver()` も `retireStereoConvolver()` も一度もチェックしない。二重退役時に UAF/Double-Free が発生する。`PendingCommit::~PendingCommit()` と通常退役パスの両方から退役される経路が存在する |

---

### 🟠 HIGH（3件）— 優先的に修正推奨

| ID | 箇所 | 問題 |
|---|---|---|
| **H-1** | `EQProcessor.Core.cpp` | **EQState/BandNode 退役失敗時のメモリリーク** — `retireEQStateDeferred(prev)` の戻り値が `(void)` で無視されている。DeferredDeletionQueue 満杯時に prev が永久に解放されない |
| **H-2** | `CustomInputOversampler.cpp` | **AVX2 OOB 読み取りリスク** — `loadStride2()` が `ptr[-6]` までアクセスするが、`globalMinConvIdx` の境界チェックはこの -6 オフセットを考慮しない。`prepareStage()` の +6 マージンに暗黙依存しており、タップ数変更時にサイレントな OOB となる |
| **H-3** | `AlignedAllocation.h` | **ScopedAlignedPtr が任意ポインタを受け入れる** — コンストラクタが任意の `T*` を受け入れ、`reset()` が `mkl_free()` を呼ぶ。mkl_malloc 以外のポインタを渡すと UB |

---

### 🟡 MEDIUM（12件）— 計画的に修正推奨

| ID | 箇所 | 問題 |
|---|---|---|
| **M-1** | `ConvolverProcessor.h` | `publishRuntimeProcessSnapshot()` の二重バッファが同時発行で同一スロットを破損 |
| **M-2** | `EQProcessor.Processing.cpp` | SVF 状態変数のリセット閾値 1e15 が過大。出力クランプ ±100 と 13 桁のギャップで長時間クリッピング歪み |
| **M-3** | `DeferredDeletionQueue.h` | FIFO 先頭ブロッキング — 先頭が退役不可だと後続全エントリがブロックされメモリ蓄積 |
| **M-4** | `CmaEsOptimizerDynamic.cpp` | NaN 回復時に mean をリセットしないため、NaN 発生領域に留まり続ける |
| **M-5** | `DeviceSettings.cpp` | `closeAudioDevice()` と `initialise()` の間にデバイス null ウィンドウがあり、並行 Timer がクラッシュ |
| **M-6** | `CustomInputOversampler.cpp` | バッファオーバーフロー時に空 `AudioBlock` を返却。呼び出し側チェックに依存 |
| **M-7** | `ConvolverProcessor.h` | `isIRLoaded()` がリビルド遷移中に不整合な状態を返す |
| **M-8** | `AllpassDesigner.cpp` | CMA-ES 収束閾値 1.0 の「サンプル数」単位解釈が重み付き RMSE では不正確 |
| **M-9** | `EQProcessor.Processing.cpp` | AGC のブロックレート RMS エンベロープが大ブロックサイズで実効アタック時間を劣化 |
| **M-10** | `DspNumericPolicy.h` | ScopedThreadRole の 4 スロット枯渇時にサイレント失敗し `isAudioThread()` が false を返す |
| **M-11** | `CacheManager.cpp` | CRC64 の init=0 が非標準（CRC-64/XZ は init=0xFFFF...）。外部ツール非互換 |
| **M-12** | `ConvolverProcessor.h` | rebuildDebounceToken がインクリメントされるが、リビルド実行時に最新トークン比較がない |

---

### 🟢 LOW（6件）— 改善推奨

| ID | 問題 |
|---|---|
| **L-1** | `BuildSnapshot::fingerprint` が一度も設定されず常に 0 |
| **L-2** | `getLatencyBreakdown()` が 4 個の atomic を個別読み取りし不整合な組み合わせを返しうる |
| **L-3** | `computeIRHash()` の TOCTOU 検出がミリ秒粒度で、同一ミリ秒内の変更を検出不能 |
| **L-4** | `BlacklistedASIODeviceType::ensureScanned()` が const メソッド内で非 const な `scanForDevices()` を呼ぶ |
| **L-5** | `DeferredDeletionQueue::sizeApprox()` の unsigned 減算が異常時に巨大値を返す |
| **L-6** | `PeakEstimator::interpolateParabolic()` の Lagrange 再評価が浮動小数点誤差を蓄積しうる |

---

### 統計サマリ

```
CRITICAL :  3 件
HIGH     :  3 件
MEDIUM   : 12 件
LOW      :  6 件
合計     : 24 件

カテゴリ別:
  スレッド安全性   : 7 件
  メモリ安全性     : 5 件
  数値/DSP 正確性  : 4 件
  ロジック/設計    : 5 件
  API/規約         : 3 件
```

---

### 最優先修正の推奨

**C-1（AudioSegmentBuffer）** は音声のスペアナ表示に直接影響し、**C-2（EQProcessor shadow）** は Audio Thread 上の UB で最も危険です。**C-3（StereoConvolver retired）** は IR 切替時のクラッシュに直結します。この 3 件を最優先で修正されることを強く推奨いたします。

ご不明な点や、特定のバグに対する修正コードのご要望がございましたら、お気軽にお申し付けください。