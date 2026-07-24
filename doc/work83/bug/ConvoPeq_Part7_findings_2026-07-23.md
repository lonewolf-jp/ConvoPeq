# ConvoPeq バグ調査 詳細レポート Part 7（2026-07-23）

対象: `ConvoPeq.md`（279ファイル、81,346行、2026-07-23 00:33:53生成）
手法: `grep -n`によるパターン横断検索 + 該当ファイルの`view`による個別検証。書き込み側・読み取り側・呼び出し元スレッドの相互参照を実施したものを「確認済み」、参照が一部に留まるものを「要検証」と明記します。

---

## A. No.9: クロスフェードの線形/等電力不整合（確認済み）

### 現象

`AudioEngine.Processing.DSPCoreDouble.cpp`（double版）と`AudioEngine.Processing.DSPCoreFloat.cpp`（float版）内、`runLatencyAlignedCrossfadeMixLoop`に渡すラムダで、新旧コンボルバーエンジンのクロスフェード・ゲインが線形補完で計算されています。

**DSPCoreDouble.cpp:31960-31980（double版）**
```cpp
31957	        const double* oldL = (outChannels > 0) ? dspCrossfadeDoubleBuffer.getReadPointer(0, 0) : nullptr;
31958	        const double* oldR = (outChannels > 1) ? dspCrossfadeDoubleBuffer.getReadPointer(1, 0) : nullptr;
31959	
31960	        runLatencyAlignedCrossfadeMixLoop<double>(dstL,
31961	                                                  dstR,
31962	                                                  oldL,
31963	                                                  oldR,
31964	                                                  numSamples,
31965	                                                                  preparedCrossfade.latencyDelayOld,
31966	                                                                  preparedCrossfade.latencyDelayNew,
31967	                                                                  preparedCrossfade.latencyResetPending,
31968	                                                  [](double* outL,
31969	                                                     double* outR,
31970	                                                     int i,
31971	                                                     double gNew,
31972	                                                     double alignedOldL,
31973	                                                     double alignedOldR,
31974	                                                     double alignedNewL,
31975	                                                     double alignedNewR)
31976	                                                  {
31977	                                                      const double gOld = 1.0 - gNew;
31978	                                                      if (outL != nullptr) outL[i] = alignedNewL * gNew + alignedOldL * gOld;
31979	                                                      if (outR != nullptr) outR[i] = alignedNewR * gNew + alignedOldR * gOld;
31980	                                                  });
```

**DSPCoreFloat.cpp:31234-31250（float版、`dryScale`込み）**
```cpp
31234	                                                     [this, useDryAsOld](float* outL,
31235	                                                                         float* outR,
31236	                                                                         int i,
31237	                                                                         double gNew,
31238	                                                                         double alignedOldL,
31239	                                                                         double alignedOldR,
31240	                                                                         double alignedNewL,
31241	                                                                         double alignedNewR)
31242	                                                     {
31243	                                                         const double dryScale = useDryAsOld ? crossfadeRuntime_.getDryScaleGain().getNextValue() : 1.0;
31244	                                                         const double gOld = 1.0 - gNew;
31245	                                                         const double dryScaledL = alignedOldL * dryScale;
31246	                                                         const double dryScaledR = alignedOldR * dryScale;
31247	                                                         if (outL != nullptr)
31248	                                                             outL[i] = static_cast<float>(alignedNewL * gNew + dryScaledL * gOld);
31249	                                                             outL[i] = static_cast<float>(alignedNewL * gNew + dryScaledL * gOld);
31250	                                                         if (outR != nullptr)
31251	                                                             outR[i] = static_cast<float>(alignedNewR * gNew + dryScaledR * gOld);
```

### なぜ問題か

`gNew`は新エンジンのゲイン、`gOld = 1.0 - gNew`は旧エンジンのゲインです。`gNew + gOld = 1.0`（線形和一定）であり、`gNew=gOld=0.5`のクロスフェード中間点でのパワー和は `0.5² + 0.5² = 0.5`（-3dB相当）に低下します。新旧エンジンの出力は同一IRに由来するとはいえ、FDLパーティション構成・レイテンシ整列・内部状態が異なるため完全に同相加算されるとは限らず、実際には中間点で聴感上のレベル落ち（dip）が生じるリスクがあります。

### 本コードベース内に既に正しい実装が存在する（重要な補強証拠）

`ConvolverProcessor.Runtime.cpp:62106-62111`に、libm不使用（RT-safe）のsin近似による等電力クロスフェード用ヘルパーが**既に実装・使用されています**。

```cpp
62103	namespace
62104	{
62105	    // Audio thread path avoids libm calls for deterministic realtime behavior.
62106	    inline double equalPowerSin(double x) noexcept
62107	    {
62108	        const double t = x * (juce::MathConstants<double>::pi * 0.5);
62109	        const double t2 = t * t;
62110	        return t * (1.0 + t2 * (-1.0 / 6.0 + t2 * (1.0 / 120.0 + t2 * (-1.0 / 5040.0 + t2 * (1.0 / 362880.0)))));
62111	    }
```

同ファイル内での使用例（62680-62681、wet/dryではなくnew/oldの意味だが、両方を独立にsin近似で計算し積和が一定になる設計）:
```cpp
62680	            wg[i] = equalPowerSin(mix)         * headroom;
62681	            dg[i] = equalPowerSin(1.0 - mix);
```

同種のヘルパーは`EQProcessor.Processing.cpp:70374`にも重複定義され、そちらは`wNew`/`wOld`を**それぞれ独立に**`equalPowerSin()`で計算しています（71171-71172）:
```cpp
71171	            const double wNew = equalPowerSin(t);
71172	            const double wOld = equalPowerSin(1.0 - t);
```

つまり本プロジェクトは等電力クロスフェードの実装方法を複数箇所で正しく実践しており、`runLatencyAlignedCrossfadeMixLoop`のみが`gOld = 1.0 - gNew`という線形式に留まっている状態です。修正は上記と同じ関数を流用するだけで完結し、新規のDSP設計は不要です。

### 提案パッチ

`patch_09_equal_power_crossfade.diff` を参照（§Cの後に添付）。

---

## B. No.13: `RuntimePublicationCoordinator::getVersion()`の非atomicアクセス（確認済み・実害は現状未確認）

### 根拠

**ISRRuntimePublicationCoordinator.h:199-207**
```cpp
199	    // ★ 3個別 atomic に代わり、plain struct で3フィールドを論理一貫管理
200	    // IMPORTANT: persistentState_ is MessageThread-only.
201	    //   Any cross-thread access requires conversion to std::atomic<PersistentStateBlock>.
202	    PersistentStateBlock persistentState_{};
203	
204	    std::atomic<const void*> currentWorld_;
205	    std::atomic<RejectCode> lastRejectCode_;
206	    std::atomic<std::uint64_t> retireBacklogCount_;
```

**ISRRuntimePublicationCoordinator.cpp:169-172**
```cpp
169	std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
170	    // ★ 方式C: persistentState_ から直接導出（plain struct、atomic 不要）
171	    return persistentState_.mappedRuntimeGeneration;
172	}
```

`commit()`（同.cpp:75-111）は`persistentState_ = PersistentStateBlock{...}`という非atomicなプレーン代入で書き込みます。ヘッダのコメントは`persistentState_`を明確に「MessageThread-only」としていますが、`getVersion()`自体にはそれを強制する仕組み（アサート等）が一切ありません。このクラスの他の全メンバは`std::atomic`+明示的な`memory_order`で統一的に扱われており（本ファイルの大半を占める慎重なコメント付きatomic操作と対照的に）、`persistentState_`とその読み取り専用関数`getVersion()`だけがこの規律から外れています。

### 呼び出し元の確認

```
grep "\.getVersion()\|->getVersion()"
  77366, 77380, 77765, 77783 → いずれも src/tests/ 配下（単体テスト、単一スレッド実行）
```
本番コード（AudioEngine等）からの呼び出しは確認されませんでした。したがって**現状クロススレッドのデータレースは発生していない**と判断します。ただし将来的に何らかのスレッド（例：UIタイマーやHealthMonitor）から`getVersion()`が呼ばれるようになった場合、未定義動作（データレース）となります。

### 提案パッチ

`patch_13_getversion_thread_assert.diff`を参照。デバッグビルドでの呼び出しスレッド検証アサートを追加する軽量な提案です（本番挙動は変更しません）。

---

## C. No.16: RT-safeヘルパー関数の分散複製（確認済み）

### 発見の経緯

以下のシグネチャを持つ関数が、複数の`.cpp`ファイルの無名名前空間に**独立して**（`#include`による共有ではなく）定義されています。

```
inline bool isFiniteNoLibm(double x) noexcept        → DSPCoreDouble.cpp:32342, DSPCoreFloat.cpp:33046, DSPCoreIO.cpp:33520
inline double fastTanh(double x) noexcept            → DSPCoreFloat.cpp:33173, DSPCoreIO.cpp:33587
inline double musicalSoftClipScalar(...) noexcept    → DSPCoreDouble.cpp:32389, DSPCoreFloat.cpp:33192, DSPCoreIO.cpp:33606
inline double equalPowerSin(double x) noexcept       → ConvolverProcessor.Runtime.cpp:62106, EQProcessor.Processing.cpp:70374
inline bool isFiniteNoLibm(double value) noexcept    → EQProcessor.Processing.cpp:70320（引数名まで独自でコピペではなく独自実装の可能性）
```

一方、`src/dsp/math/FastTanhApprox.h`には**まさにこの重複を避ける目的**で設計されたPolicyベースの共有実装が存在します：

**FastTanhApprox.h:67933-67946**
```cpp
67933	//==============================================================================
67934	// FastTanhApprox — Tanh 近似の共通ユーティリティ
67935	//
67936	// ★ ISR Runtime 準拠（Single Semantic Source）:
67937	//   DSPCoreDouble（SoftClip）・EQProcessor（Saturation）のすべてが
67938	//   同一実装を参照する。係数と閾値は Policy テンプレートで注入し、
67939	//   将来の独立チューニングに備える。
67940	//
67941	// 使用方法:
67942	//   double y = convo::dsp::fastTanh<SoftClipPolicy>(x);
67943	//   __m128d yv = convo::dsp::fastTanhV128<EQSaturationPolicy>(xv);
67944	//
67945	//==============================================================================
```

`DSPCoreDouble.cpp`はAVX2一括処理部分でこのヘッダを正しく使っています（`#include`は32288行目）：
```cpp
32409	    const double clipped = threshold + knee * convo::dsp::fastTanh<convo::dsp::SoftClipPadéPolicy>((abs_x - threshold) / knee);
32473	        __m256d tanhVal = convo::dsp::fastTanhV256<convo::dsp::SoftClipPadéPolicy>(arg);
```

しかし**同じ`DSPCoreDouble.cpp`内のAVX2端数処理（スカラーフォールバック）**は、共有ヘッダのスカラー版`convo::dsp::fastTanh<SoftClipPadéPolicy>()`を呼ばず、独自に手書きした`fastTanh`/`musicalSoftClipScalar`（32342-32506行）を使っています：

```cpp
32494	    for (; i < numSamples; ++i)
32495	    {
32496	        const double inputVal = data[i]; // 元の入力を退避
32497	        double x = inputVal;
32498	        if (absNoLibm(x) > clip_start)
32499	            x = musicalSoftClipScalar(x, threshold, knee, asymmetry);
32500	
32501	        data[i] = x;
```

さらに`DSPCoreFloat.cpp`・`DSPCoreIO.cpp`は`FastTanhApprox.h`を`#include`すらしておらず（grep結果に該当行なし）、係数（10395/1260/21/4725/210/4.5）まで同一の実装を完全に独立して保持しています。

### リスク

将来`SoftClipPadéPolicy`の係数調整やバグ修正が`FastTanhApprox.h`に対して行われても、`DSPCoreFloat.cpp`・`DSPCoreIO.cpp`・`DSPCoreDouble.cpp`のスカラー経路には反映されず、**処理経路によってサチュレーション特性が異なる**という形で問題が顕在化します。これは複数セッションに渡って報告されている「繰り返し発見されるバグパターン」（本質は関数分散）の一種と考えられます。

### 提案パッチ（例示: DSPCoreIO.cpp）

`patch_16_dspcoreio_shared_tanh.diff`を参照。`DSPCoreFloat.cpp`への同様の適用、および`DSPCoreDouble.cpp`のスカラーフォールバックの置き換えは、呼び出し側のシグネチャ差異（`float*`か`double*`か、`threshold`等の型）を精査した上での追加作業が必要なため、今回は方針提示に留め、パッチはIO.cppのみ提示します。

---

## D. No.10 / No.11: 実オーディオコールバックスレッドのFTZ/DAZ + vmlSetMode欠落（確定・本セッション最重要所見）

### 結論

**実際のオーディオコールバックスレッド（`getNextAudioBlock()`が実行されるOSスレッド）は、アプリケーション実行中一度も自スレッド用のFTZ/DAZ・vmlSetModeを設定していません。** `getNextAudioBlock()`全体（730行）と`prepareToPlay()`全体（289行）を実際に走査し、コードベース自身が持つ根拠（後述）と合わせて確認したため、前回報告時点の「要検証」から「確定」に格上げします。

### 検証手順と根拠

**1. `getNextAudioBlock()`全文（30832〜31563行、730行）にFTZ/DAZ/vmlSetMode関連のキーワードが一切出現しない**

`_MM_SET_FLUSH_ZERO_MODE`、`_MM_SET_DENORMALS_ZERO_MODE`、`vmlSetMode`、`MKLRealTime::setup`、`ScopedMXCSR`のいずれも、関数の開始行から終了行までを走査して0件でした。

**2. `prepareToPlay()`は`ASSERT_NON_RT_THREAD()`で保護されており、`getNextAudioBlock()`とは別のOSスレッドで実行される**

```cpp
34596	void AudioEngine::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
34597	{
34598	    ASSERT_NON_RT_THREAD();
```

**3. コードベース自身が「Message Threadとオーディオコールバックスレッドは別スレッドであり、スレッド固有の初期化はコールバック側で行う必要がある」という設計原則を、MMCSSに関して既に正しく実践している**

`prepareToPlay()`内のコメント（34605-34608行）:
```cpp
34605	    // ★ [work70 v9.11] Unified MMCSS Layer: prepareToPlay は Message Thread のため
34606	    //    MMCSS 登録は行わない（Audio callback 初回で tryApplyMmcssForSelfManagedThread が実行）。
34607	    //    WASAPI: JUCE managed (JuceManaged) → nothing to do here.
34608	    //    ASIO/DS: Self-managed → callback で thread_local 登録。
```

そして`getNextAudioBlock()`冒頭（30853-30875行）で実際に「初回コールバックのみ」パターンが実装されています:
```cpp
30853	    // ★ [work70 v9.11] Unified MMCSS Layer:
30854	    //   - WASAPI(enum JuceManaged): JUCE manages → nothing to do
30855	    //   - ASIO(enum SelfManagedProAudio):  first-call registration (Pro Audio/CRITICAL)
30856	    //   - DirectSound(enum SelfManagedPlayback): first-call registration (Playback/HIGH)
30857	    //   Logging is guarded by #if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS (zero cost in Release).
30858	    //   thread_local ensures safety across driver-owned ASIO threads.
30859	    //   Shutdown: mmcssShutdownRequested flag from Message Thread → AvRevert here.
30860	    {
30861	        const auto mmcssPolicy = getCurrentMmcssPolicy();
30862	        if (mmcssPolicy == MmcssPolicy::SelfManagedProAudio
30863	            || mmcssPolicy == MmcssPolicy::SelfManagedPlayback)
30864	        {
30865	            // tryApplyMmcssForSelfManagedThread() uses W API, logs success/failure
30866	            // under CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS. RT impact: first call only (~50-200μs).
30867	            static_cast<void>(tryApplyMmcssForSelfManagedThread());
```

`tryApplyMmcssForSelfManagedThread()`自体は`thread_local bool t_mmcssTried`によるガードで実装されています（29885, 29934-29938行）。

**4. しかし、FTZ/DAZ・vmlSetModeには同じ「初回コールバック自己登録」パターンが適用されていない**

実際に確認できた`_MM_SET_FLUSH_ZERO_MODE`/`vmlSetMode`の全呼び出し元は以下の通りで、いずれも「ドライバ所有のオーディオコールバックスレッド」ではありません:

| 行 | ファイル | スレッド |
|---|---|---|
| 17173-17174 | `MKLRealTimeSetup.cpp`（`MKLRealTime::setup()`、`std::call_once`ガード） | Message Thread（唯一の呼び出し元17315行が`MainApplication.cpp`内） |
| 17342-17343, 17346 | `MainApplication.cpp` | Message Thread |
| 21205-21206, 21417-21418 | `NoiseShaperLearner.cpp` | 学習用バックグラウンドスレッド |
| 24004-24005 | `ProgressiveUpgradeThread.cpp` | ProgressiveUpgradeThread |
| 36352-36354 | `AudioEngine.RebuildDispatch.cpp` | rebuildThread |
| 59667-59670 | `ConvolverProcessor.LoaderThread.cpp` | LoaderThread |
| 60580-60581, 61163-61164 | `ConvolverProcessor.MixedPhase.cpp` | （未特定の非RTスレッド、要追加確認） |
| 67795-67796 | `core/WorkerThread.cpp` | WorkerThread |

加えて`core/ScopedMXCSR.h`は明示的に次のようにコメントされています:
```cpp
66365	/// RAII ラッパー: コンストラクタで FTZ/DAZ を設定し、デストラクタで復元する。
66366	/// ThreadPool ワーカー（std::async 等）で使用すること。
66367	/// 専用スレッドや Realtime Audio Thread では使用しない。
```
「専用スレッドやRealtime Audio Threadでは使用しない」という記述は、開発者が「RTオーディオスレッドには別の（RAIIでない、復元不要の）設定方法が必要」と認識していたことを示唆しますが、その専用の設定コードが実際には実装されていません。

### 影響

本アプリケーションはコンボルバー（IRテール減衰）・パラメトリックEQ・ノイズシェーパー等、信号が長時間かけてゼロへ収束する処理を多数含みます。実オーディオコールバックスレッドでFTZ/DAZが未設定の場合、これらの減衰・無音区間でCPUがデノーマル数演算に陥り、通常演算比で大幅な処理遅延（一般に10〜100倍）が生じる可能性があります。リアルタイムオーディオではコールバック内の処理が締め切り時間を超過するとドロップアウト・クラックルとして聴感上顕在化するため、静かな場面・フェードアウト・リバーブテール等で断続的な音切れが発生するリスクがあります。これは本セッション冒頭の規約「デノーマル対策をしっかり行ってください」「デノーマル対策を全ワーカースレッドで必須」に直接抵触する状態です。

### 提案パッチ

`patch_10_audio_thread_denormal_guard.diff`を参照。既存の`tryApplyMmcssForSelfManagedThread()`と全く同じ設計（`thread_local bool`ガード＋初回コールバックのみ実行、コストはMXCSR書き込み+TLS書き込みのみでロック・確保・例外を伴わない）を踏襲しているため、Audio Thread内での実行が許容されると判断しています。3ファイルにまたがる変更です:
1. `AudioEngine.h`: メンバ関数宣言追加
2. `AudioEngine.Mmcss.cpp`: `thread_local`ガード変数追加、実装追加
3. `AudioEngine.cpp`: `getNextAudioBlock()`冒頭に1行の呼び出し追加

### No.11: 誤解の発生源

`MKLRealTime::setup()`（`MKLRealTimeSetup.cpp:17159`）は`std::call_once`でガードされており、ヘッダコメントは「Safe to call multiple times (call_once ensures single execution)」と説明していますが、実際の呼び出し元は`MainApplication.cpp:17315`の1箇所のみです。内部で設定される`mkl_set_num_threads_local(1)`・FTZ/DAZ（コメントアウトされた`vmlSetMode`を含む）はすべてIntel MKL/CPU仕様上スレッドローカルであるため、`call_once`によって「最初に呼んだスレッド（＝Message Thread）にしか適用されない」設計になっています。ヘッダコメントの「複数回呼び出し安全」は「重複実行を避けられる」という意味では正しいですが、「どのスレッドから呼んでも全スレッド分の設定が行われる」という誤解を招きやすい表現です。`MKLNonUniformConvolver.cpp:15640`・`MKLRealTimeSetup.cpp:17175`の「MainApplicationで設定済みだからここでは呼ばない」というコメントは、この誤解の直接の産物と考えられます。

なお、`mkl_set_num_threads_local(1)`自体は、CMakeで`MKL_THREADING=sequential`（シングルスレッドリンク）が指定されているというコメント（17163行）が事実であれば、そもそもMKLがOSスレッドを内部生成しない構成のため、本設定が全スレッドに行き渡っていないこと自体の実害は限定的と推測されます。実害の中心はFTZ/DAZ・vmlSetModeの欠落（No.10）です。

---

## E. その他の確認事項（軽微・情報共有）

### E-1. `PublicationBuffer` / `MultiStagePublisher`（デッドコード、No.12）

`ISRRuntimePublicationCoordinator.h:246-256`で宣言、`.cpp:495-525`で定義される両クラスについて、コンストラクタ呼び出し・インスタンス化箇所を全文検索しましたが、定義箇所以外に**一切の使用箇所がありません**。

```cpp
506	void PublicationBuffer::enqueue(const void* world) {
507	    if (world == nullptr) {
508	        return;
509	    }
510	
511	    std::lock_guard<std::mutex> lock(guard_);
512	    queued_.push_back(world);
513	}
```

`PublicationBuffer`は`std::mutex`のロックと`std::vector::push_back`（動的確保を伴い得る）を内包しており、これらは規約上オーディオスレッドで使用禁止の操作です。現状デッドコードのため実害はありませんが、`ISRRuntimePublicationCoordinator.h`という「RT-safe設計が徹底された」ファイル内に紛れていることから、将来誤って結線されるリスクがあります。削除、または最低限「未使用・結線禁止」の明示コメントを推奨します。

### E-2. 命名の混乱（No.14）

- メンバ変数 `runtimePublicationBridge_`（`AudioEngine.h`）の型は `convo::isr::RuntimePublicationCoordinator`（`ISRRuntimePublicationCoordinator.h`で定義、`setRetireCoordinator(convo::isr::RuntimePublicationCoordinator*)`のシグネチャ一致で確認）。
- 一方 `AudioEngine`内には別途 `class RuntimePublicationBridge final { ... }`（`AudioEngine.h:42786`）という**別クラス**が存在。
- さらに同スコープ内で `using RuntimePublicationCoordinator = convo::RuntimePublicationCoordinator<RuntimePublishWorld, DSPCore*, RuntimePublicationBridge>;`（`AudioEngine.h:42849`）という、`src/core/RuntimePublicationCoordinator.h`で定義された**別の名前空間のテンプレートクラス**へのエイリアスも存在。

3つの異なる実体（`convo::isr::RuntimePublicationCoordinator`、`AudioEngine::RuntimePublicationBridge`、`convo::RuntimePublicationCoordinator<...>`のローカルエイリアス）が「Coordinator」「Bridge」という語を交差させて使っており、コードの追跡・保守時の誤認リスクが高い状態です。リネームは影響範囲が広いため今回はパッチを提示せず、情報共有のみとします。

### E-3. `RCUReader::enter()`のオーナートークン検証順序（No.15）

`core/RCUReader.h:592-616`：

```cpp
592	    void enter() noexcept
593	    {
594	        // acq_rel: acquire → 直前の exit() の nestingDepth release を観測してネストを安全化；
595	        //          release → depth > 0 を公開し、ネスト中の enter が早期 return できる。
596	        const uint32_t previousDepth = convo::fetchAddAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
597	        if (previousDepth > 0)
598	        {
599	            // ★ ネスト: 最外層の rootEnterSucceeded_ を維持
600	            return;
601	        }
602	
603	        const uint64_t threadToken = currentThreadToken();
604	        uint64_t expectedOwner = 0;
605	        // CAS acq_rel/acquire: 成功時 acq_rel → ownerThreadToken を取得し新オーナーを公開；
606	        //                     失敗時 acquire → 競合スレッドの最新 ownerThreadToken を観測。
607	        if (!convo::compareExchangeAtomic(ownerThreadToken,
608	                                          expectedOwner,
609	                                          threadToken,
610	                                          std::memory_order_acq_rel,
611	                                          std::memory_order_acquire)
612	            && expectedOwner != threadToken)
613	        {
614	            convo::fetchSubAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
615	            return;
616	        }
```

`nestingDepth`のインクリメントと`previousDepth>0`判定が、`ownerThreadToken`のCAS検証よりも**先に**行われます。このため、あるスレッドAが`enter()`保持中（`nestingDepth>0`）に、別スレッドBが同一`RCUReader`インスタンスへ`enter()`した場合、Bは`ownerThreadToken`の不一致チェックに到達する前に「ネスト」と誤認して早期returnし、Bの`enterReader()`登録（epoch保護への正式な参加）が行われないまま処理を継続する可能性があります。

ただし、本コードベースでの`RCUReader`インスタンス化箇所を全数確認したところ（`audioThreadRcuReader`, `messageThreadRcuReader`, `publicationReader`, EQProcessor内`rcuReader`等）、いずれも「1スレッド専用」であることがコメントで明記されており（例:「DSP_THREAD_STATE: AudioEngine process系で使うaudio-thread専用RCU reader。」）、現状の呼び出し規約が守られている限り本経路は顕在化しません。将来の拡張で複数スレッドから同一`RCUReader`インスタンスを共有するコードが追加された場合にのみ問題化する、**防御ロジックの構造的な穴**として報告します。

---

## G. No.18: `processToBuffer()`でモノラル入力時に右チャンネルが無音になる（要検証）

**AudioEngine.Processing.DSPCoreToBuffer.cpp:34401-34429（全文）**
```cpp
34401	void AudioEngine::DSPCore::processToBuffer(const juce::AudioSourceChannelInfo& source,
34402	                                           juce::AudioBuffer<float>& destination,
34403	                                           LockFreeAudioRingBuffer& analyzerFifo,
34404	                                           std::atomic<float>* inputLevelLinear,
34405	                                           std::atomic<float>* outputLevelLinear,
34406	                                           const ProcessingState& state)
34407	{
34408	    const int numSamples = source.numSamples;
34409	    const int numChannels = std::min(2, source.buffer != nullptr ? source.buffer->getNumChannels() : 0);
34410	
34411	    if (source.buffer == nullptr || numSamples <= 0 || destination.getNumSamples() < numSamples)
34412	    {
34413	        destination.clear();
34414	        return;
34415	    }
34416	
34417	    for (int ch = 0; ch < numChannels; ++ch)
34418	    {
34419	        const float* src = source.buffer->getReadPointer(ch, source.startSample);
34420	        float* dst = destination.getWritePointer(ch, 0);
34421	        juce::FloatVectorOperations::copy(dst, src, numSamples);
34422	    }
34423	
34424	    for (int ch = numChannels; ch < destination.getNumChannels(); ++ch)
34425	        destination.clear(ch, 0, numSamples);
34426	
34427	    juce::AudioSourceChannelInfo destinationInfo(&destination, 0, numSamples);
34428	    process(destinationInfo, analyzerFifo, inputLevelLinear, outputLevelLinear, state);
34429	}
```

### 現象

`numChannels = min(2, source.buffer->getNumChannels())`のため、入力デバイスが**モノラル（1チャンネル）**の場合、`numChannels == 1`となります。34417-34422のコピーループはチャンネル0のみを`destination`へコピーし、34424-34425の後続ループが`numChannels`（1）から`destination.getNumChannels()`（通常2＝ステレオ）までを**無音でクリア**します。結果として、モノラル入力デバイス使用時は**Lチャンネルに信号、Rチャンネルは常時無音**という状態でその後の`process()`（EQ・コンボルバー本処理）に渡ることになります。一般的なオーディオアプリケーションでは、モノラル入力はL/R両チャンネルへ複製する（センター定位のモノラル信号として扱う）のが期待される挙動であるため、本実装は聴感上「右チャンネルが壊れている」ように見える可能性があります。

### 検証状況

- 呼び出し元は`fading->processToBuffer(bufferToFill, ...)`（`AudioEngine.Processing.DSPCoreFloat.cpp:31211`）のみで、`bufferToFill`は`getNextAudioBlock()`が受け取った`juce::AudioSourceChannelInfo`をそのまま伝播していると推測されます（両者を突き合わせるコード上の断定的証拠までは本セッションで確認できていません）。
- 本アプリケーションが実際にモノラル入力オーディオデバイスの選択を許可しているか（`juce::AudioDeviceManager`の初期化パラメータや設定UI側の制約）は、本セッションのgrep範囲では確認できませんでした。もし常にステレオ入力に固定されているなら本経路は実質到達不能で影響はありません。
- 次回セッションでの確認を推奨します。

---

## F. 検証済みで問題なしと判定した領域（詳細根拠）

### F-1. `EQProcessor.Coefficients.cpp` 係数式（10関数すべて確認）

**Biquad（Audio EQ Cookbook, Robert Bristow-Johnson準拠）**: LowShelf・Peaking・HighShelf・LowPass・HighPassの5関数（`calcLowShelfBiquad`〜`calcHighPassBiquad`、68672-68832行）を、公開されているRBJ Cookbookの係数式と1項ずつ突合し、完全一致を確認しました。

**SVF/TPT（Cytomic技術資料準拠）**: `calcLowShelfSVF`/`calcPeakingSVF`/`calcHighShelfSVF`/`calcLowPassSVF`/`calcHighPassSVF`（68935-69122行）についても、Cytomicの公開Bell/Shelf/LP/HP式（`g`, `k`, `a1/a2/a3`, `m0/m1/m2`）と1項ずつ突合し完全一致を確認しました。特にBell型の`k = 1/(Q·A)`、Low/High Shelf型の`g`のsqrt(A)乗除（周波数シフト補正）は原典通りに実装されています。

本ファイルは前回セッションで「未調査・係数計算式検証」と指定されていた項目ですが、**数式レベルでの不具合は見つかりませんでした**。

### F-2. `MKLNonUniformConvolver::Layer::freeAll()`（15157-15218行）

診断ビルド経路（`freeTracked`、14バッファ）とリリースビルド経路（`if(ptr){mkl_free(ptr);ptr=nullptr;}`パターン、同じく14バッファ）を1対1で突合し、旧Bug#1として記録されていた`delayLineBuf`も含め、両経路で完全に一致するバッファ集合が解放されていることを確認しました。

### F-3. スレッドのjoin処理

- `DeferredFreeThread::~DeferredFreeThread()`（8713-8716行）→`shutdownAndDrain()`が停止フラグ・join・強制解放の順で実行。
- `WorkerThread::~WorkerThread()`（67765-67768行）→`stop()`が`if(thread.joinable()) thread.join();`を実行。
- `AudioEngine::~AudioEngine()`（28599-28733行）は`releaseResources()`を経由しない異常系も想定し、28616-28617行で無条件に`stopRebuildThread()`を呼び出しており（コメント：「releaseResources が未実行の異常系でも worker 終了を保証する」）、`stopRebuildThread()`内部（36328-36339行）で`if (rebuildThread.joinable()) rebuildThread.join();`を実行。

3クラスとも異常終了パスを含めて正しくjoinしていることを確認しました。

---

*本レポートはPart 1〜6（2026-07-20セッション）の内容を前提とし、それらのNo.1〜8のステータスには変更を加えていません。*
