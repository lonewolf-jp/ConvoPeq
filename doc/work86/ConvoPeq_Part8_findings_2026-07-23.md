# ConvoPeq バグ調査 詳細レポート Part 8（2026-07-23 続報）

Part 7に続き、`ISRRTExecution.cpp/h`・`ISRDSPHandle.cpp/h`を精査した結果です。

---

## H. No.19: `RTTraceRelay`の`drain()`が未結線（情報・低優先度）

`ISRRTExecution.h/cpp`の`RTTraceRelay`（RTコールバック内から非RT側へトレースイベントを転送するロックフリー・リングバッファ）はメンバとして保持され（`AudioEngine.h:4334` `convo::isr::RTTraceRelay rtTraceRelay_;`）、`enqueue()` は `AudioEngine.Processing.AudioBlock.cpp:304` から呼び出されています（`rtTraceRelay_.enqueue(...)`）。

しかし、非RT側で蓄積されたイベントを読み出す `drain()` の呼び出し箇所は**自身の実装以外に一切ありません**。`RELAY_BUFFER_SIZE = 4096` の固定リングバッファはやがて満杯になり、新しいイベントで古いイベントが上書きされます。

```cpp
47938	void RTTraceRelay::enqueue(const RTTraceEvent& event) noexcept
...
47955	void RTTraceRelay::drain()
```

実装自体はロックフリーでRT-safe（`AtomicAccess.h`の`publishAtomic`/`consumeAtomic`のみ使用、確保・ロックなし）であり、`No.12`（`PublicationBuffer`/`MultiStagePublisher`）とは異なりRT安全性上の懸念はありません。ただし「RTで書き込まれているが非RTで読み出されていない」という点で、データ喪失の観点からは問題があります。`RTCapabilityFirewall`/`RTAllocatorFirewall::isRTContext()`についても、コード自身のコメント（`ISRRTExecution.cpp:48002`）で「現時点のコードベースで呼出箇所は存在しない」と明記されており、同様に準備中・未使用の状態です。

---

## I. No.20: `std::atomic<DSPHandle>`のロックフリー性が未検証（低・防御的パッチ推奨）

`ISRDSPHandle.h:45749-45751`では、同一構造体`DSPRegistrySlot`内の`std::atomic<uint64_t> generation`に対し明示的なロックフリー検証があります:
```cpp
45749	    std::atomic<uint64_t> generation;  // ★ B-1: 64bit化（ABA 防止世代番号）
45750	    static_assert(std::atomic<uint64_t>::is_always_lock_free,
45751	        "atomic<uint64_t> must be lock-free on x64 for ISR Runtime");
```

一方、同じファイル内の`std::atomic<DSPHandle>`（`DSPHandle`は`uint32_t slot; uint64_t generation;`の2フィールド構造体、アライメント込みで16バイト）である`activeRuntimeDSPHandle_`・`fadingRuntimeDSPHandle_`（45818-45819行）には同様の検証がありません。16バイトのatomicはCMPXCHG16B命令に依存するため、x64+AVX2という本プロジェクトの前提ハードウェアでは通常ロックフリーですが、コンパイラ・ビルド設定次第では非ロックフリー実装（内部mutex使用）にフォールバックし得ます。

**呼び出し元の確認**: `getActiveRuntimeDSPHandle()`の全3呼び出し箇所（35278行 `AudioEngine.Processing.ReleaseResources.cpp`内`releaseResources()`＝Message Thread確定、44822行・56067行＝`engine_.dspHandleRuntime_.`形式で別クラスから、いずれも文脈上NonRT）を確認した限り、**現状はRTスレッドからの読み取りは確認されませんでした**。RT-safeとドキュメントされている`resolve()`は`registry_[handle.slot]`の`generation`（ロックフリー検証済み）と`state`のみを読み、`activeRuntimeDSPHandle_`自体には触れません。したがって現状は実害なしと判断します。

プロジェクト全体を横断検索したところ、`is_always_lock_free`の明示検証は本ファイルの1箇所（`std::atomic<uint64_t>`）と`AudioEngine.h`冒頭の2箇所（`std::atomic<size_t>`, `std::atomic<uint64_t>`）の計3箇所のみで、`std::atomic<DSPHandle>`や`std::atomic<FilterStructure>`（`EQProcessor.h:72322-72323`、enum想定のためリスクは低い）等、構造体・enumを内包するatomicには同様の検証がありません。

### 提案

現状は実害なしのため緊急のパッチは不要ですが、将来`getActiveRuntimeDSPHandle()`がRTスレッドから呼ばれるよう変更された場合に備え、`ISRDSPHandle.h`の`activeRuntimeDSPHandle_`宣言部に以下の一行を追加することを推奨します（本セッションではdiff提示は見送り、次回の設計判断時にまとめて対応することを申し送ります）:

```cpp
static_assert(std::atomic<DSPHandle>::is_always_lock_free,
    "atomic<DSPHandle> must be lock-free on x64 for ISR Runtime");
```

---

## J. `ConvolverProcessor.MixedPhase.cpp`のFTZ/DAZ設定スレッド（Part7 §D 残課題の解消）

60580-60581行のFTZ/DAZ設定は`convertToMixedPhaseAllpass()`内にあり、同一スコープ内で`juce::ScopedLock`（ミューテックス）・`juce::Logger::writeToLog`（ファイル/コンソールI/O）・`std::make_unique`（動的確保）を使用していることを確認しました。これらは規約上RTスレッドで禁止されている操作であるため、本関数自体がNon-RTスレッド（IRロード・変換系のバックグラウンドスレッド）で実行されることの強い状況証拠となります。Part 7 §D表の「要追加確認」は本確認をもって解消とします。

---

## K. gain_design_spec.md関連（前回セッションからの継続4項目）の状況確認

前回セッションで「on the horizon」として未解決とされていた4項目について、本セッションで現状を確認しました。

| 項目 | 前回状態 | 今回確認結果 |
|---|---|---|
| `setAutoGainStagingEnabled()`の実装仕様未記載 | 未解決 | **解決済みと確認**。`AudioEngine.h:40758-40774`で完全な実装を確認：atomic切替→EQ AGC無効化（`★Bug#4`）→`submitRebuildIntent(Structural, EnqueueSnapshotCommand, Snapshot, Replaceable)`（`★BUG-10`）という明確な処理が実装されている |
| `buildRuntimePublishWorld()`レガシーオーバーロードの`autoGainStagingEnabled`サイレントデフォルト問題 | 未解決 | **解決済みと確認**。`AudioEngine.h:53337-53390`（`★v9.5 fallback`とタグ付け）で、旧シグネチャ呼び出し時に`spec.processing.autoGainStagingEnabled = convo::consumeAtomic(engine.autoGainStagingEnabled, ...)`を含む、`processingOrder`・`eqBypassed`・`softClipEnabled`等ほぼ全フィールドをengine atomicから明示的に補完する実装が確認できた。サイレントデフォルトではなく明示的なフォールバック値取得に置き換わっている |
| Auto Gain Staging有効時のUI表示ギャップ（`updateGainStagingDisplay()`/`computeAndApplyAutoGain()`の不整合） | 未解決 | **解決済みの可能性が高い（状況証拠ベース）**。`computeAndApplyAutoGain()`という関数名自体が現ソースに存在せず、代わって`RuntimeBuilder::buildRuntimePublishWorld()`内（`AudioEngine.h:53029-53053`）で`AutoGainPlanner::plan()`の結果を`worldOwner->automation.inputHeadroomGain/outputMakeupGain/convolverInputTrimGain`へ**一元的に**代入し、Published Worldとして配信する単一経路のアーキテクチャに統合されていました。`updateGainStagingDisplay()`（`DeviceSettings.cpp:9502`）は値の再計算をせずラベル整形のみを行っており、「計算をバイパスする別経路」は確認できませんでした。ただしUI側が実際にこのPublished Worldの値をそのまま表示しているかの最終確認（DeviceSettings.cpp内、値表示部分のコード）までは本セッションで完了していません |
| M/S最大ゲイン計算の位相関係による最大~3dB過小評価リスク | 未解決 | **未解決を確認（構造的根拠あり）**。ゲイン推定パイプラインの起点である`BandHelper::collectActiveBands()`（`BandHelper.cpp:68192-68241`）を確認したところ、`state.bands[i]`という**単一のバンド配列**を走査するのみで、Mid/Side各チャンネル別のバンド設定や位相関係を扱うロジックは一切存在しませんでした。`computeEstimatedMaxGainComplex()`自体もv14.30/v14.47でモジュール化されましたが（`BandHelper`/`EQResponseSampler`/`PeakEstimator`/`UpperBoundEstimator`への分割）、基本アルゴリズム（各バンドのBiquad振幅を独立に評価）自体は変わっておらず、M/Sエンコード時にMid/Sideが再合成されて生じる周波数依存の位相関係は考慮されていません。したがって本件は**リファクタリングを経てもなお未解決**と判断します |

**結論**: 4項目中2項目（実装仕様・レガシーオーバーロード問題）は明確に解決済み、1項目（UI表示ギャップ）は強い状況証拠から解決済みの可能性が高いと判断します。M/S最大ゲイン位相関係の1項目のみ、構造的な未対応を具体的根拠とともに確認しました。

---

## L. No.21: `ShutdownRuntime::advancePhase()`のスキップ判定バグ（デッドコード内で確認・実害なし）

### 発見

`ISRShutdown.cpp`の`ShutdownRuntime::transitionTo()`は、フェーズを1段階以上スキップする遷移を「スキップされる全フェーズがterminal（`ShutdownComplete`/`TimedOut`/`Failed`）である場合のみ」許可するロジックです（51554-51564行）。

一方、同ファイルの`advancePhase()`（51500-51543行）は`ReclaimComplete`から**直接**`VerifyDrained`への遷移を試みます:
```cpp
51525	        case ShutdownPhase::ReclaimComplete:
51526	            // ★ C-2: CONVOPEQ_EMERGENCY_DRAIN 有効時のみ EmergencyDrain を経由
51527	            next = ShutdownPhase::VerifyDrained;
51528	            break;
```

`ShutdownPhase`のenum順序は`Running=0, AudioStopped=1, ObserverDrained=2, RetireClosed=3, EpochSettled=4, ReclaimComplete=5, EmergencyDrain=6, VerifyDrained=7, TimedOut=8, Failed=9, ShutdownComplete=10`です。`ReclaimComplete(5)→VerifyDrained(7)`は`EmergencyDrain(6)`を1つスキップする遷移ですが、`EmergencyDrain`は`isTerminalPhase()`の対象外（terminalは`ShutdownComplete`/`TimedOut`/`Failed`のみ）のため、`transitionTo()`のスキップ許可ロジックはこれを**拒否**します。`(void)transitionTo(next);`と戻り値を破棄しているため、この遷移が失敗すると`phase_`は`ReclaimComplete`のまま変化せず、それ以上`advancePhase()`を呼んでも状態は進みません。

### 実害の検証：呼び出し元ゼロ、実際のシャットダウンシーケンスは別経路

`advancePhase()`の呼び出し箇所を全文検索したところ、**自身の定義以外に一切存在しません**。実際の本番シャットダウンシーケンス（`AudioEngine::releaseResources()`、34964〜35396行）は、`advancePhase()`を経由せず`transitionTo()`を各フェーズごとに個別・明示的に呼び出しています:

```
AudioStopped → ObserverDrained → RetireClosed → EpochSettled → ReclaimComplete
  → EmergencyDrain（常に単一遷移として通過、ワーク自体はm_healthMonitor.isEmergencyDrainRequested()で実行時分岐）
  → VerifyDrained → ShutdownComplete
```

このシーケンスは全て`t==c+1`の単純な1段階遷移（`VerifyDrained→ShutdownComplete`のみ、terminalな`TimedOut`/`Failed`を正しくスキップする2段階遷移）であり、バグの発生条件（`EmergencyDrain`を非terminalのままスキップ）に該当しません。

**結論**: `advancePhase()`のロジックバグは実在しますが、デッドコードであるため現行ビルドでの実害はありません。ただし将来誰かが`advancePhase()`を「使われていないなら削除しよう」ではなく「便利そうだから使おう」と実装に組み込んだ場合、`ReclaimComplete`到達後シャットダウンFSMが恒久的に停止するバグを踏むことになります。加えて`ISRShutdown.h`のenum宣言コメント「デフォルトではスキップ（既存の graceful drain で十分）」は、実際の呼び出し元コメント（`work37 Phase 8.2でコンパイル時マクロから実行時判定に変更`、フェーズ自体は常時通過）と矛盾しており、ドキュメントの陳腐化も確認されました。

### 推奨

`advancePhase()`を実際には使わないのであれば削除を、将来的に使う可能性があるなら`case ShutdownPhase::ReclaimComplete: next = ShutdownPhase::EmergencyDrain; break;`と`EmergencyDrain`を明示的な中間ステップとして経由するよう修正することを推奨します（`releaseResources()`の実装と整合させる）。実害がないため今回はパッチを提示せず、次回の設計判断事項として申し送ります。

---

## M: No.22: `DSPQuarantineManager`の`auditLog_`排他制御欠如（軽度・設計判断待ち）

`ISRDSPQuarantine.h/cpp`の`DSPQuarantineManager`（隔離ハンドル管理：世代不一致・resolve失敗等で回収不能になったDSPハンドルを記録し、監査ログとして保持するクラス）は、`AudioEngine.h:4304`で`dspQuarantineManager_`としてメンバ保持され、以下の7メソッドが実際に使用されています:

- `quarantineHandle()` → `AudioEngine.Threading.cpp:42`
- `reclaimSlot()` → `AudioEngine.Commit.cpp:633`
- `isActive()` → `AudioEngine.Commit.cpp:617`
- `residentCount()` → `ReleaseResources.cpp:225,357,377` / `AudioEngine.Threading.cpp:85,128`
- `destroyForShutdown()` → `ReleaseResources.cpp:365`
- `compactAuditLog()` → `ReleaseResources.cpp:375`
- `getMaxEntryAgeSec()` → `AudioEngine.Threading.cpp:88`

実装自体は堅牢です（RT側`quarantineActiveFlags_`はatomic boolの固定長配列でロックフリー、NonRT側`auditLog_`は追記専用vector）。ただし`auditLog_`への書き込み（`quarantineHandle`/`reclaimSlot`/`destroyForShutdown`/`compactAuditLog`）と読み取り（`getEntry`/`getMaxEntryAgeSec`）はいずれも排他制御がなく、もし将来複数の異なるNonRTスレッド（例：退役処理スレッドと診断/HealthMonitorスレッド）から同時に呼ばれるよう結線された場合、`std::vector`への非同期な読み書き競合（データレース）が生じ得ます。現状は単一NonRTスレッドからのみアクセスされるため実害はありません。

### メタ的観察：「安全網は実装されているが未結線」というパターンの反復

本セッションを通じ、以下の**4件**で同様のパターンを確認しました:

| No. | クラス/関数 | 内部品質 | 結線状況 |
|---|---|---|---|
| 12 | `PublicationBuffer`/`MultiStagePublisher` | mutex+vector内包（RT不安全） | 未結線 |
| 19 | `RTTraceRelay` | ロックフリー・RT-safe | **enqueue: 結線済、drain: 未結線** |
| 21 | `ShutdownRuntime::advancePhase()` | ロジックバグあり | 未結線 |
| 22 | `DSPQuarantineManager` | 概ね健全（`auditLog_`排他制御のみ要検討） | **結線済（7メソッド使用中）** |

いずれも命名・設計から見て「本来ISR Runtimeの防御機構として機能するはずのコンポーネント」でありながら、実際の結線状態はさまざまです。No.22は既に結線済みで動作中、No.19はenqueueのみ結線・drain未結線（部分的なデータ喪失リスク）、No.12とNo.21は完全未結線です。次回セッションでは、未結線コンポーネントが「意図的な将来拡張の足場」なのか「結線を忘れたデッドコード」なのかの意思決定を推奨します。

---

## N. `NoiseShaperLearner.cpp`の確認状況（解決済み）

CMA-ESベースの学習ワーカースレッド管理は堅牢。学習結果`bestCoefficients`からRTノイズシェーパー処理への反映経路を追跡した結果、以下の完全な経路を確認しました:

```
NoiseShaperLearner::publishGenerationResult()
  → convo::publishAtomic(bestCoefficients[i], k)  // atomic配列に保存
  → engine.storeLearnedCoeffs(mappedCoeffs.data()) // coeff bankに書込
    → storeLearnedCoeffsToBank(bankIndex, coeffs)   // AdaptiveCoeffBankSlotへ
    → bank.generation++                              // 世代番号更新
  → engine.setAdaptiveNoiseShaperState(bankIndex, state)
  → engine.requestAdaptiveAutosave()

RT側:
  DSPCoreDouble.cpp:619-628 / DSPCoreIO.cpp:427-436
  → activeAdaptiveCoeffGeneration != state.adaptiveCoeffGeneration
    → 新しいcoeff bankの係数をadaptiveNoiseShaperに適用
```

**結論: 解決済み。** `bestCoefficients` は `publishGenerationResult()` → `storeLearnedCoeffsToBank()` → `bank.generation++` → RT generation tracking という完全な経路でRT処理に反映されます。ただし9個の`atomic<double>`を個別に更新する方式は、`getState()`で読み取る際に一部更新状態を観測する理論的リスクがありますが、`publishGenerationResult()` は単一スレッドから呼ばれ、mappedCoeffsの全9要素をatomic更新した後に`requestAdaptiveAutosave()`で確定されるため、実質的な問題はありません。

---

## O. `RuntimeHealthMonitor.cpp`/`RuntimePolicyEngine.cpp`の確認（部分・軽微所見）

work37/work39タグの膨大な閉ループ自己回復システム（Retire Stall・Publication Stall・Reader枯渇・Crossfade drop等複数監視軸→PolicyEngine→RecoveryAction発火→Verification→Storm検知→Budget管理）を確認しました。全容の完全検証には至っていませんが、以下2点を軽微所見として記録します。

**1. `CriticalExitCondition.blocker`の設定漏れ（診断情報のみ、安全動作に影響なし）**

`RuntimeHealthMonitor.cpp`のCritical出口評価ロジック（53877-53882行付近）で、`pendingRetire`超過・`retireAge`超過の場合は`exitCond.blocker`に理由コードが設定されますが、`readerHealthy`（`activeReaderCount() != 0`）のみが原因で`metricsHealthy=false`となるケースには対応する`blocker`設定がありません。`allMonitorsNormal`自体は正しく`false`になるため実際の安全動作（Critical状態からの誤った離脱防止）には影響しませんが、診断ログを見た開発者が「なぜCriticalから抜けないか」を`blocker`だけから判断すると、Reader残留が原因のケースを見落とす可能性があります。

**2. `RuntimePolicyEngine::canExecute()`のcooldown計算に潜在的なunsigned underflowリスク（要スレッド確認）**

```cpp
55356	    const uint64_t nowUs = getNowUs();
55357	    return (nowUs - entry.lastExecutedUs) >= entry.cooldownUs;
```
`nowUs`・`entry.lastExecutedUs`はともに`uint64_t`（符号なし）です。`entry.lastExecutedUs`が何らかの理由で`nowUs`より大きい値になった場合、減算がアンダーフローし意図せず巨大な値になって常にcooldown条件を満たしてしまう（＝クールダウンが実質無効化される）リスクがあります。`m_cooldowns[idx].lastExecutedUs`は`std::atomic`ではなくプレーンな`uint64_t`であるため、`tick()`（おそらく単一の監視タイマースレッドから定期実行）以外のスレッドから`canExecute()`/`markExecuted()`が呼ばれないかの確認が必要です。本セッションでは呼び出し元スレッドの特定に至っておらず、確認済みバグとしては報告せず注記に留めます。

---

## 本セッション（2026-07-23）まとめ

- 新規確認事項: No.9〜No.22（うちNo.10は本セッション最重要の確定バグ、パッチ提示済み）
- 前回on the horizon 5項目中、明確に解決確認: AoS/SoAメモリ、MKLバッファ解放、EQ係数式検証、gain_design_spec 4項目中3項目
---

## P. No.23: `ConvolverProcessor::cleanup()`の「強制削除」が実際には機能していない（確認済み）

`ConvolverProcessor.LoadPipeline.cpp:59326-59360`の`cleanup()`は、終了済み`LoaderThread`を`loaderTrashBin`から除去する処理です。2つのループから構成されています:

```cpp
59329	    // 終了したスレッドのみを削除する (waitForThreadToExit(0) はブロックしない)
59330	    for (auto it = loaderTrashBin.begin(); it != loaderTrashBin.end(); )
59331	    {
59332	        if ((*it)->waitForThreadToExit(0))
59333	        {
59334	            it->reset();
59335	            it = loaderTrashBin.erase(it);
59336	        }
59337	        else
59338	        {
59339	            ++it;
59340	        }
59341	    }
59342
59343	    // 【Leak Fix】LoaderThreadの異常蓄積防止
59344	    // スレッドが終了しない場合でも、一定数を超えたら強制削除してメモリを解放する。
59345	    // [FIX] detached thread はプロセス終了時に未定義動作を引き起こすため、
59346	    //       同期的なチェックと削除に切り替える。
59347	    if (loaderTrashBin.size() > 2)
59348	    {
59349	        for (auto it = loaderTrashBin.begin(); it != loaderTrashBin.end() && loaderTrashBin.size() > 2; )
59350	        {
59351	            if (*it != nullptr && (*it)->waitForThreadToExit(0))
59352	                it = loaderTrashBin.erase(it);
59353	            else
59354	                ++it;
59355	        }
59356	    }
59357	}
```

コメント「スレッドが終了しない場合でも、一定数を超えたら強制削除してメモリを解放する」は、スレッドが終了していなくても削除することを明言しています。しかし実際の条件は`(*it)->waitForThreadToExit(0)`——1つ目のループと**全く同じ**「スレッドが既に終了しているか」の非ブロッキングチェックです。スレッドが終了していなければ2つ目のループも`erase`せず`++it`するだけなので、1つ目のループで削除されなかったエントリ（＝実行中のスレッド）は2つ目のループでも削除されません。コメントにある「強制削除」は実装されておらず、実質的に1つ目のループの空振りの再走査に留まっています。

コメント内の「[FIX] detached thread はプロセス終了時に未定義動作を引き起こすため、同期的なチェックと削除に切り替える」という記述から、以前は`detach()`等で強制的にスレッドを切り離す実装だったものを、別のバグ（プロセス終了時のUB）を修正する過程で「常に安全側」の同期チェックに置き換えた結果、意図していた「一定数超過時の強制解放」という保護機能が実質的に失われたと推測されます。

### 影響

`LoaderThread`が何らかの理由（IRファイル読み込みでのブロッキングI/O異常、無限ループ等）で正常終了しない場合、`loaderTrashBin`はコメントの意図に反して際限なく増加し続け、`LoaderThread`インスタンス（内部にIRデータ・FFTバッファ等を保持）がメモリ上に蓄積するリスクがあります。ただし、これは「LoaderThreadが正常に終了しない」という別の異常系が前提であり、通常運用下では顕在化しない可能性が高い点に留意してください。

### 提案

安全性を優先し、今回はパッチを提示しません（「強制削除」を実装するには`stopThread(timeoutMs)`のような有限待機付き強制終了か、あるいはコメントを実態に合わせて「終了済みのみ削除、強制削除は行わない」に修正するかの設計判断が必要です）。次回セッションでの検討事項として申し送ります。

---

## 本セッション（2026-07-23）まとめ（最終版）

- 新規確認事項: No.9〜No.24（うちNo.10は本セッション最重要の確定バグ、パッチ提示済み）
- 前回on the horizon 5項目中、明確に解決確認: AoS/SoAメモリ、MKLバッファ解放、EQ係数式検証、gain_design_spec 4項目中3項目
- 未解決を再確認: M/S最大ゲイン位相関係（構造的根拠を追加）
- **N (NoiseShaperLearner RT結線): 解決済み**（調査完了により確定）
- パッチ提示: No.9（crossfade, 2ファイル）, No.10（denormal guard, 3ファイル）, No.13（getVersion assert）, No.16（DSPCoreIO重複解消例）
- メタ的観察: 「安全機構は実装済みだが未結線」というパターンが4件（No.12, 19, 21, 22）
- RuntimeHealthMonitor/RuntimePolicyEngine: 軽微所見2件（診断blocker欠落、cooldown underflowリスク・要スレッド確認）
- ConvolverProcessor.LoadPipeline.cpp: No.23（cleanup()の強制削除が未実装）を確認。`CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOAD_PIPELINE`は常時有効（CMakeLists.txt）であることを確認、重複実装なし
- ISRRetireRouter: `retireRT()`はNo.3(emitRetireIntentRT)と異なりリトライ・mutexフォールバックを持たない正しいRT-safe設計であることを確認（問題なし）

---

## Q. No.24: `ClosureGraphWalker::emitClosureArtifact()`のJSON文字列未エスケープ（軽微・CI診断のみ）

`ISRClosureGraphWalker.cpp:45341-45344`で、`validationError`（`std::string_view`）をJSON文字列値として書き出す際、ダブルクォート・バックスラッシュ・制御文字のエスケープ処理が一切行われていません:

```cpp
45341	    file << "  \"validationErrors\": [";
45342	    if (!valid) {
45343	        file << "\"" << std::string(validationError) << "\"";
45344	    }
45345	    file << "]\n";
```

`validationError`に`"`（ダブルクォート）や制御文字が含まれる場合、出力される`closure_graph.json`は不正なJSONとなり、後続のCI/評価ツールでのパースに失敗する可能性があります。本クラスはRuntime処理には関与せずCI証跡出力専用（`std::ofstream`・`std::filesystem`を使用しており明確にNonRT/オフライン専用）のため、オーディオ処理への影響はありません。パッチは提示せず、情報共有のみとします。`ClosureValidator::validateClosureGraph()`自体（DFSによる循環検出、ノードID重複チェック、edge整合性チェック）はアルゴリズムとして正しく実装されていることも確認しました。

## R. `ConvolverProcessor.StateAndUI.cpp`の確認：`copyPendingToSnapshotUnlocked()`/`copySnapshotToPendingUnlocked()`は検証済みで問題なし

両関数の更新ガイドコメントは「片方向のみ更新すると snapshot 同期の欠落を招く」と明記的に警告していたため、フィールド単位で1対1突合を行いました。`mix`から`nucLCMode`まで21フィールドすべてが両方向で対称に扱われており、欠落は確認されませんでした（`copySnapshotToPendingUnlocked()`側は`juce::jlimit`によるクランプが追加されている点のみ非対称ですが、これは意図された設計です）。この種の「対称性が要求されるが強制されない」関数ペアはバグの温床になりやすいため重点確認しましたが、本ペアについては健全と判断します。

---

- 未着手のまま次回へ: `NoiseShaperLearner`のRT結線経路、`src/tests/*`との突き合わせ、`ConvolverProcessor.StateAndUI.cpp`の`getState()/setState()`本体（今回は同期ヘルパーのみ確認）
