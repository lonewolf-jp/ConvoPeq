# ConvoPeq バグ調査 詳細レポート Part 8（2026-07-23 続報）

Part 7に続き、`ISRRTExecution.cpp/h`・`ISRDSPHandle.cpp/h`を精査した結果です。

---

## H. No.19: `RTTraceRelay`がRT-safeに実装されているが未結線（情報・低優先度）

`ISRRTExecution.h/cpp`の`RTTraceRelay`（RTコールバック内から非RT側へトレースイベントを転送するロックフリー・リングバッファ）はメンバとして保持されています（`AudioEngine.h:43806` `convo::isr::RTTraceRelay rtTraceRelay_;`）が、その`enqueue()`・`drain()`の呼び出し箇所は**自身の実装以外に一切ありません**。

```cpp
47938	void RTTraceRelay::enqueue(const RTTraceEvent& event) noexcept
...
47955	void RTTraceRelay::drain()
```

実装自体はロックフリーでRT-safe（`AtomicAccess.h`の`publishAtomic`/`consumeAtomic`のみ使用、確保・ロックなし）であり、`No.12`（`PublicationBuffer`/`MultiStagePublisher`）とは異なりRT安全性上の懸念はありません。ただし「準備されたが結線されていない」インフラという意味で同種のパターンです。`RTCapabilityFirewall`/`RTAllocatorFirewall::isRTContext()`についても、コード自身のコメント（`ISRRTExecution.cpp:48002`）で「現時点のコードベースで呼出箇所は存在しない」と明記されており、同様に準備中・未使用の状態です。

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
| Auto Gain Staging有効時のUI表示ギャップ（`updateGainStagingDisplay()`/`computeAndApplyAutoGain()`の不整合） | 未解決 | **要再調査**。`computeAndApplyAutoGain()`という関数名自体が現在のソースに存在しないため、大規模リファクタリングにより問題自体の前提が変化している可能性があります。`updateGainStagingDisplay()`（`DeviceSettings.cpp:9502`）は現状、処理順序・バイパス状態に応じた表示ラベル（Headroom範囲テキスト）の更新のみを行っており、実ゲイン値の計算はしていません。実際の自動ゲイン値計算がどこで行われ、UI表示とどう同期しているかは`AutoGainPlanner.cpp`（未着手）の確認が必要です |
| M/S最大ゲイン計算の位相関係による最大~3dB過小評価リスク | 未解決 | **要再調査**。旧`computeEstimatedMaxGainDb()`は`computeEstimatedMaxGainComplex()`（`EQProcessor.Processing.cpp:68860`、`★v14.30`/`★v14.47`タグ）へ全面的に置き換えられており、`BandHelper::collectActiveBands`・`EQResponseSampler`・`PeakEstimator::estimate`・`UpperBoundEstimator::estimateMax`という新設ヘルパー群に処理が分散しています。M/S位相関係の扱いがこれらのどこに（あるいはまだ）実装されているかは、`BandHelper.cpp`・`PeakEstimator.cpp`・`UpperBoundEstimator.cpp`（いずれも未着手）を確認しないと判断できません |

**結論**: 4項目中2項目（実装仕様・レガシーオーバーロード問題）は明確に解決済みと確認できました。残り2項目は、前回指摘時点から基盤となるコード自体が大幅リファクタリング（v14.30〜v14.47等）されているため、問題の前提自体を`AutoGainPlanner.cpp`・`BandHelper.cpp`・`PeakEstimator.cpp`・`UpperBoundEstimator.cpp`で再確認する必要があります。次回セッションの優先事項とします。

---

*本レポートはPart 7の続報です。マスターサマリーは次回更新時にNo.19・No.20およびgain_design_spec項目の状況を反映します。*
