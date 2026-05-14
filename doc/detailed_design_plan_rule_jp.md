# ConvoPeq 詳細設計書（plan.md準拠 / rule.md遵守）

## 0. 目的

本書は doc/plan.md の最終改修計画を、実装に直接着手できる設計粒度へ分解した詳細設計である。
最優先目的は機能追加ではなく、以下の不変条件（IR-1〜IR-7）の形式的回復と維持である。

- IR-1: レンダーフェーズ不変性
- IR-2: 単一公開世界
- IR-3: 所有権固定
- IR-4: RT隠蔽同期禁止
- IR-5: エポック一貫観測
- IR-6: 所有権移転
- IR-7: クロスフェード隔離

## 1. 適用範囲

### 1.1 対象ファイル（設計対象）

- src/audioengine/AudioEngine.h
- src/audioengine/AudioEngine.Commit.cpp
- src/audioengine/AudioEngine.Processing.BlockDouble.cpp
- src/audioengine/AudioEngine.UIEvents.cpp
- src/eqprocessor/EQProcessor.h
- src/eqprocessor/EQProcessor.Core.cpp
- src/eqprocessor/EQProcessor.Processing.cpp
- src/convolver/ConvolverProcessor.StateAndUI.cpp
- src/audioengine/DSPExecutionState.h

### 1.2 非対象（本設計で変更しない）

- JUCE/ 配下
- r8brain-free-src/ 配下
- Audio Thread での新規ロック/動的確保/例外/SEH導入

## 2. 実装前必須解析（rule.md 10章）

### 2.1 Call Graph（主要経路）

- Rebuild publish経路: requestRebuild -> RebuildDispatch -> commitNewDSP -> publishSmoothTransitionState/startImmediateSmoothTransition -> Audio Thread process
- Crossfade実行経路: AudioEngine.process* -> armCrossfadeIfPending -> runLatencyAlignedCrossfadeMixLoop -> finalizeCrossfadeMixPath/cleanupCrossfadeDirectPath
- UI同期経路（現在の違反経路）: convolverParamsChanged -> activeDSP->convolverRt().syncParametersFrom(uiConvolverProcessor)

### 2.2 Ownership Flow

- currentDSP/fadingOutDSP は publish後に Audio Thread が消費し、retireはEPR/RCU境界で処理する。
- ConvolverProcessor::syncParametersFrom は publish済みランタイムへの直接mutationを許しており、IR-6違反源。
- EQの実行時可変状態は DSPExecutionState::EQDSPState に寄せる設計が既に存在し、これを正規経路に一本化可能。

### 2.3 Publication Edge

- 現状は memory_order の直接記述が分散し、公開境界の意味論が局所化している。
- engineRuntimeState/runtimeGraphState/currentDSP/fadingOutDSP で公開ドメインが分裂しやすい。

### 2.4 Thread Affinity

- Message Thread責務: 構築、初期化、publish、UI整合、非RT計算。
- Audio Thread責務: immutable snapshot消費のみ。
- 現状の問題: armCrossfadeIfPending 内で ramp reset/setCurrentAndTargetValue を実施（初期化侵入）。

### 2.5 Crossfade Coexistence

- 旧/新DSPは混在期間に同時稼働するため、可変状態共有は禁止。
- dry-as-old 遷移や遅延補償の状態は準備済みスナップショットとして publish し、Audio Thread は activateのみ許可。

## 3. 詳細設計（Critical）

### 3.1 C1: Crossfade初期化のMessage Thread完全移管

#### 3.1.1 現状

- armCrossfadeIfPending が Audio Thread 上で以下を実行している。
- dspCrossfadeGain.reset(...)
- dspCrossfadeGain.setCurrentAndTargetValue(0.0)
- dspCrossfadeGain.setTargetValue(1.0)

#### 3.1.2 設計

- CrossfadePreparedState を導入し、初期化済み遷移を保持する。
- 例: sampleRate, fadeSec, initialGain, targetGain, latencyDelayOld/New, startDelayBlocks, useDryAsOld

- commitNewDSP（Message Thread）でのみ prepareCrossfadeTransition を呼ぶ。
- ramp初期値を確定
- 遅延補償値を確定
- dry-as-oldフラグを確定
- publishCrossfadePreparedState で公開

- Audio Thread の armCrossfadeIfPending を activateCrossfadeIfPrepared に置換。
- resetや初期化は禁止
- 既に準備済みの遷移を1回だけ有効化
- 成否は単一atomicフラグで判定

#### 3.1.3 受け入れ条件

- Audio Thread 側コードに reset/setCurrentAndTargetValue 呼び出しが存在しない。
- Crossfade開始時の状態差分は Message Thread でのみ作られる。

### 3.2 C2: EQProcessorのSmoothedValue排除

#### 3.2.1 現状

- EQProcessor.h に以下が残存。
- juce::SmoothedValue&lt;double&gt; smoothTotalGain
- juce::SmoothedValue&lt;double&gt; bypassFadeGain

#### 3.2.2 設計

1. メンバ型を convo::LinearRamp に置換。
1. prepareToPlay 相当（EQProcessor::prepare）で reset と初期値設定を実施。
1. Audio Thread（process）では setTargetValue/getNextValue/skip のみ使用。
1. バイパス遷移・トータルゲイン遷移の保持先を EQDSPState の ramp と統一し、二重状態を解消。

#### 3.2.3 非機能要件

- Audio Thread 内で libm（pow/exp/log/sin/cos）呼び出し禁止を維持。
- 知覚品質調整が必要な場合、Message Thread側の係数事前計算のみ許可。

#### 3.2.4 受け入れ条件

- EQProcessor.h から juce::SmoothedValue の参照が消える。
- process 経路で pow/exp 等の新規混入が無い。

### 3.3 C3: publish済みruntimeへのmutation禁止

#### 3.3.1 現状

- AudioEngine.UIEvents.cpp で activeDSP->convolverRt().syncParametersFrom(...) が呼ばれる。
- publish済み activeDSP への直接書換で IR-1/3/6 違反。

#### 3.3.2 設計

1. convolverParamsChanged は差分検出のみ行う。
1. 差分があれば RebuildReason を生成し requestRebuild のみ実行。
1. activeDSPへの直接同期（syncParametersFrom）は削除。
1. 反映は「新規DSP構築 -> publish原子差し替え」に一本化。

#### 3.3.3 受け入れ条件

- activeDSP->... の直接mutationコードが Message Thread から消える。
- syncParametersFrom はUIインスタンス間コピーのみに限定される。

### 3.4 C4: Publication Edgeの形式化

#### 3.4.1 設計方針

- atomic直接操作を helper API 経由に統一し、release/acquire契約を固定化する。

#### 3.4.2 導入API

- publishAtomicPtr(dst, value)
- consumeAtomicPtr(src)
- exchangeAtomicPtr(dst, value)
- publishSnapshot(dst, snapshot)

#### 3.4.3 適用対象

- currentDSP
- fadingOutDSP
- engineRuntimeState
- runtimeGraphState

#### 3.4.4 追加方針

- 複数atomicで同一世界を構成している箇所は RuntimeSnapshot へ集約し、単一公開へ移行。
- memory_order_relaxed はメータ/統計/デバッグ用途のみ許可。

#### 3.4.5 受け入れ条件

- publication対象ポインタで memory_order 直書きが原則消える。
- snapshot整合破壊パス（片側だけ更新）が無い。

## 4. 詳細設計（High）

### 4.1 H1: RCUReaderのコピー/ムーブ禁止

#### 4.1.1 設計

- RCUReader 型宣言へ以下を明示。
- copy ctor/assign delete
- move ctor/assign delete

#### 4.1.2 受け入れ条件

- RCUReaderを値コピーするコードがコンパイル不能になる。

### 4.2 H2: aligned_make_unique導入

#### 4.2.1 設計

- 64byteアライン確保の例外安全ヘルパを1箇所に集約。
- placement new の直接記述は禁止し、ヘルパ内に閉じ込める。
- Audio Thread での呼び出しは禁止。

#### 4.2.2 受け入れ条件

- oneMKL関連および大型バッファ確保で手書き配置newが増えない。

### 4.3 H3: commitNewDSP の通知経路非ブロッキング化

#### 4.3.1 設計

- rebuildMutex保持中の sendChangeMessage 呼び出しを禁止。
- pendingChangeNotification フラグで通知を合流。
- AsyncUpdaterを唯一の通知出口にする。

#### 4.3.2 受け入れ条件

- sendChangeMessage が rebuildMutexスコープ外でのみ呼ばれる。
- 連続rebuild時に通知洪水が発生しない。

## 5. 詳細設計（Medium）

### 5.1 M1: ライフサイクル状態機械

- shutdownInProgress/gShuttingDown を EngineLifecycleState(enum class) に統合。
- compare_exchange_strong で遷移関数を定義。
- boolフラグ追加は禁止。

### 5.2 M2: RebuildReasonビットマスク

- Structural/Latency/EQParams/Bypass/Learning 等をビット化。
- 依存グラフを明示。
- SR変更 -> トポロジ再構築必須
- ゲイン変更 -> パラメータ更新のみ

### 5.3 M3: NoiseShaperLearner状態機械

- Idle -> Starting -> Running -> Stopping
- Stopping中の再start拒否
- jthread + stop_token の協調停止

## 6. データ構造詳細

### 6.1 RuntimeSnapshot（単一公開世界）

- activeDSP pointer
- fadingDSP pointer
- crossfade prepared state
- latency align state
- generation/epoch

公開は単一atomicポインタ交換で行い、Audio Threadは consumeのみ。

### 6.2 CrossfadePreparedState

- fadeSec
- delayOld/delayNew
- startDelayBlocks
- useDryAsOld
- preparedGeneration

Message Threadで構築後 publish、Audio Threadで1回activate。

## 7. スレッド責務契約

### 7.1 Message Thread

- 変更差分判定
- ランタイム構築
- ramp初期化
- publication
- retire予約

### 7.2 Audio Thread

- snapshot取得
- preallocated buffer処理
- ramp進行（初期化禁止）
- 退役キュー投入（非ブロッキング）

### 7.3 Worker Thread

- 非RT計算のみ
- publishはMessage Threadに委譲

## 8. 実装シーケンス

1. C1実装: crossfade prepare/activate分離
1. C3実装: activeDSP直接mutation禁止
1. C4実装: publication helper導入
1. C2実装: SmoothedValue排除
1. H1/H2/H3実装
1. M1/M2/M3実装

依存上、C1とC3を先に適用しない限り、C4の検証精度が落ちる。

## 9. 検証設計

### 9.1 静的検証

- Audio Thread経路に以下が無いことを grep で確認
- reset/setCurrentAndTargetValue（Crossfade系）
- SmoothedValue
- std::pow/std::exp 等
- lock/mutex/new/delete/shared_ptr

### 9.2 実行時検証

- ASan Debugビルド
- 長時間再生でクロスフェード多発シナリオ
- IR有無切替、OS倍率変更、EQ連続操作
- publish/retireカウンタ単調性確認

### 9.3 受入判定

- 不変条件IR-1〜IR-7を破るコードパスが存在しない。
- 既存機能（IRロード、EQバイパス、クロスフェード、学習連携）が回帰しない。

## 10. 実装ルール再掲（rule.md反映）

- Audio Threadへ lock/libm/alloc/初期化を追加しない。
- publish後オブジェクトのmutation禁止。
- atomic直書きは helper化対象を優先的に削減。
- 最小差分で段階導入し、各段で検証を必須化する。
- 目的は「動作」ではなく「形式的不変条件の維持」。
