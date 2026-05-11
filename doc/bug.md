# ConvoPeq 最終バグ報告書（完全統合版 v4.0）

本報告書は、**DSP 数値計算・並行性制御・メモリ管理・状態遷移設計・ランタイムアーキテクチャ**にわたる全ソースコード監査の結果を統合した**決定版**です。
新たに発見された項目（A‑14〜A‑16, B‑10〜B‑12）を含め、実害リスクに基づく正確な優先度で整理しています。

---

## 🔍 アーキテクチャの根本課題（総論）

現在の ConvoPeq は、RCU（Read‑Copy‑Update）風のポインタ差し替えを行っていますが、**ランタイムオブジェクト（DSPCore）自体が「immutable な設定」ではなく「mutable な DSP 状態」を内包**しています。
このため、以下の問題が複合的に発生しています。

- クロスフェード時の状態汚染・所有権の多重化
- トランザクショナルな公開の不能（部分的 publish による hybrid state の観測）
- ウォームアップの不完全性
- シャットダウン順序の脆弱性
- リクレメーションのライブネス不透明

これらは単独では軽微に見えても、長時間運転・高負荷・DAW 環境下で**可聴ノイズ / クリック / 稀なクラッシュ**を引き起こします。
最終的な安定化には **「Immutable Runtime Graph + Per‑thread DSP State」** への移行が不可避ですが、本報告書では現実的な短中期改修を優先度付きで示します。

---

## ✅ 実装進捗（2026-05-11）

| ID | 状態 | 実装内容 | 検証 |
| --- | --- | --- | --- |
| **A-8** | **完了** | `ScopedDftiDescriptor` を導入し、DFTI ハンドル管理を RAII に統一。`DftiCreateDescriptor` 呼び出しは `put()` 経由へ統一し、早期 return 経路を含め手動 `DftiFreeDescriptor` 依存を解消。対象: `src/DftiHandle.h`, `src/AllpassDesigner.cpp`, `src/convolver/ConvolverProcessor.MixedPhase.cpp`, `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp`, `src/convolver/ConvolverProcessor.StateAndUI.cpp`, `src/SpectrumAnalyzerComponent.cpp`, `src/ConvolverState.h`。 | Debug ビルド成功（リンク完了）および `build.bat Release` 成功（`build/ConvoPeq_artefacts/Release/ConvoPeq.exe` 生成）。 |
| **A-5 / A-11** | **完了（現行設計 DoD 達成）** | `RuntimePublishState` を追加し、`resolveCurrentDSPFromRuntimePublish` / `resolveFadingDSPFromRuntimePublish` を導入。AudioBlock/BlockDouble/Learning/Latency/Snapshot/Timer の主要観測経路を publish 基準へ移行。crossfade 補助判定は `hasPendingCrossfade` と `pendingFadeDelayBlocks` を publish 由来で受け取る形へ整理し、float/double の arm 呼び出しを統一。さらに `commitNewDSP` の主要状態遷移分岐（queued fade / new fade / hard reset / dry-as-old）で `publishRuntimePublishState` 呼び出しを最終監査済み。加えて resolver 側で `runtimePublish` と atomic の整合を必須化し、stale publish pointer の参照を防止。audio thread 内で更新される crossfade 可変フラグ（`dspCrossfadePending` / `dspCrossfadeUseDryAsOld` / `dspCrossfadeStartDelayBlocks`）は atomic 優先評価へ統一し、`armCrossfadeIfPending` でも pending/fadeSec/latency を atomic 優先で取り扱うように修正。Timer の crossfade 観測判定（`hasPendingCrossfade`）も atomic 優先へ統一。 | クリーン後 Debug フルビルド成功（result code 0）。 |
| **A-1 / A-4** | **完了** | `unconstrainedToTheta()` を `0.99π` 上限へ変更し、`stableSigmoid01` へ集約。加えて `stableSigmoid01` 入力を `[-50, 50]` にクランプし、`std::exp` のオーバーフロー/アンダーフロー経路を除去。 | 実装確認: `src/AllpassDesigner.cpp` (`kThetaMax`, `std::clamp(x, -50.0, 50.0)`)。Debug ビルド成功。 |
| **A-3** | **完了** | `SecondOrderAllpass::response()` の分母処理を `safeDen = denUnit * max(denMag, kDenFloor)` へ変更し、分母極小時の不連続な固定値返却を回避。 | 実装確認: `src/AllpassDesigner.h` (`kDenFloor`, `safeDen`)。Debug ビルド成功。 |
| **A-2** | **完了** | `computeIRHash()` を xxHash64（全ファイル対象）+ TOCTOU 二段検証（サイズ/更新時刻の before/after 比較）へ改修。 | 実装確認: `src/AllpassDesigner.cpp` (`computeIRHash`)。Debug ビルド成功。 |
| **A-9** | **完了** | Nyquist 近傍ビンの重みを低減（`freq >= 0.499 * sampleRate` で `weight *= 0.1`）。 | 実装確認: `src/AllpassDesigner.cpp`。Debug ビルド成功。 |
| **A-6 (Phase 1-2)** | **完了（現行設計 DoD 達成）** | `commitNewDSP` の寿命遷移を helper 化し、queued fade / immediate fade / hard reset / dry-as-old 分岐を集約。加えて runtime UUID 診断ログ、self-retire ガード、`active/fading/queued` 重複検知を追加。さらに `commitNewDSP` 関数境界（entry/afterPublish/beforeSendChangeMessage）、Timer queued 昇格、`releaseResources`、デストラクタ、fade 完了 retire 連鎖に監視を展開。Audio Thread 経路は `validateDistinctRuntimeSlotsRT`（assert-only）を使用。加えて `EngineRuntime` の dual-write publish と resolver 境界整理（Phase 2 最小差分）を適用。 | クリーン後 Debug フルビルド成功（result code 0）。 |
| **A-12 (Lightweight + C2-C5)** | **完了** | `DSPCore` mutable state のうち `dc/ramp/history` に加え `convolver/eq` 参照境界も sidecar accessor 化。`ConvolverRuntimeState` / `EQRuntimeState` を導入し、`DSPCoreLifecycle`・`DSPCoreFloat/Double`・`Commit`・`RebuildDispatch`・`RuntimeBuilder`・`UIEvents`・`Timer`・`Latency`・`PrepareToPlay` の主要/補助経路を sidecar API 経由へ統一。 | 変更ファイルの `get_errors` はエラーなし、Debug ビルド成功（result code 0）。 |
| **A-14** | **完了** | `requestRebuild(sr,bs)` で `BuildParameterSnapshot` を導入し、rebuild 開始時点のパラメータ（dither/OS factor/OS type/noise shaper）を凍結。task 作成・重複判定・runtime command enqueue で同一 snapshot を使用するように統一。対象: `src/audioengine/AudioEngine.RebuildDispatch.cpp`。 | Debug ビルド成功（result code 0）。 |
| **A-15** | **完了（resolver hardening）** | `resolveCurrentDSPFromRuntimePublish` / `resolveFadingDSPFromRuntimePublish` の整合判定を pointer 一致依存から RuntimeUUID 一致優先へ変更し、ABA 的な同一アドレス再利用時の誤一致リスクを低減。対象: `src/audioengine/AudioEngine.h`。 | Debug ビルド成功（result code 0）。 |
| **B-1 〜 B-12** | **完了（短中期対策）** | B 系の短中期対策を一括実装。`AllpassDesigner.cpp/.h` では Hermitian 再構築の厳密化、サンプルレート依存の探索上限、相対有限差分、決定的 seed を導入。`DspNumericPolicy.h` と AudioEngine 処理系では RT/非RT アサーション、`EngineParameterSnapshot`、safe silent fallback、`prepareToPlay()` 失敗時ロールバックを追加。`NoiseShaperLearner.cpp` では worker/evaluation worker の例外捕捉を強化し、restart seed をセッション由来へ変更。`core/EBRQueue.h`、`SafeStateSwapper.h`、`DeferredFreeThread.h`、`AudioEngine.Threading.cpp`、`ConvolverProcessor.Lifecycle.cpp` では reclaim budget、backlog 診断、明示的 shutdown-and-drain を導入。 | Debug ビルド成功（CMake Tools / target `ConvoPeq`、result code 0）。 |
| **A-7 / A-10 / A-13 / A-16 / A-17** | **実装済み（維持）** | `EngineLifecycleState` と `ShutdownPhase` を導入済み。warmup は `RuntimeBuilder::executeWarmup` で `INIT -> ZERO_STATE -> DENORMAL_GUARD_ENABLE -> WARM_SIGNAL -> READY` フェーズを実装。クロスフェードは `runLatencyAlignedCrossfadeMixLoop` と `latencyDelayOld/New` でレイテンシ整合処理を実装。 | Debug ビルド成功（result code 0）。 |

備考:

- A-8 は「リーク防止（RAII 化）」としては完了。
- FFT 手順や数値ロジックそのものの改善は A-1/A-3/A-9 など別項目で継続管理。
- A-5/A-11 は、現行アーキテクチャ（`RuntimePublishState` + `EngineRuntime` dual-write + 既存 atomic 併用）での DoD 達成を意味する。
- `EngineRuntime` 単一 atomic への完全統合（報告書本文の最終推奨）は、引き続き将来改修項目として扱う。
- A-6/A-12 着手用の事前調査は `doc/a6_a12_preflight_jp.md` に集約。
- `AudioEngine.Commit.cpp` の `needsCrossfade` 分岐内クロスフェードフラグ観測（`hasPendingCrossfade` / `useDryAsOld`）も atomic 優先へ統一（2026-05-11）。
- A-4: `stableSigmoid01` に `std::clamp(x, -50.0, 50.0)` を追加し、`std::exp` オーバーフロー／アンダーフローを排除（2026-05-11）。
- A-6 Phase 1: `active/fading/queued` の重複監視は commit・timer・releaseResources・destructor・fade 完了経路まで展開済み（2026-05-11）。
- A-6 Phase 2: `EngineRuntime` 単一 publish への最小差分移行案（P2-1〜P2-5）を `doc/a6_a12_preflight_jp.md` に整理済み（2026-05-11）。
- A-12: C2-C5 を完了し、`convolver/eq` 分離足場・初期化責務集約・移行完了判定（prepare/reset/rebuild 経路）まで反映済み（2026-05-11）。
- A-14: `requestRebuild(sr,bs)` の BuildParameterSnapshot 凍結を実装し、同一 rebuild での入力パラメータ一貫性を保証（2026-05-11）。
- A-15: runtime publish resolver の一致判定を RuntimeUUID ベースへ強化し、ABA 的誤一致リスクを低減（2026-05-11）。
- A-10: シャットダウン順序は `ShutdownPhase` 実装に加え、`doc/nextgen_runtime_transition_design_jp.md` の lifecycle 表および TX-07/TX-08 に根拠を記載済み（2026-05-12 再監査）。
- B-1: C2C FFT のままでも負周波数側を更新済み正周波数ビンの共役から再構築するよう変更し、Hermitian 誤差の混入経路を遮断（2026-05-12）。
- B-2/B-3: `buildFrequencyCandidates()` と相対有限差分 `epsF0/epsGain` を導入し、高 SR での探索不足と固定 epsilon を解消（2026-05-12）。
- B-4/B-8/B-10: `ScopedThreadRole` と `ASSERT_AUDIO_THREAD` / `ASSERT_NON_RT_THREAD`、`EngineParameterSnapshot`、`applySafeSilentFallback()` を導入し、RT 入口の不変条件・ブロック境界スナップショット・安全無音フォールバックを整理（2026-05-12）。
- B-4: `processWithSnapshot()` にも `ASSERT_AUDIO_THREAD()` を追加し、AudioBlock/BlockDouble と合わせて RT 入口のアサーション網を補完（2026-05-12）。
- B-5/B-12: `NoiseShaperLearner` の worker/evaluation worker に `std::exception` 捕捉を追加し、restart seed を wall clock 依存からセッション依存の決定的導出へ変更（2026-05-12）。
- B-6: `AudioEngine::prepareToPlay()` に rollback helper を導入し、途中失敗時にレイテンシバッファと lifecycle state を巻き戻すよう変更（2026-05-12）。
- B-7/B-11: `EBRQueue` に reclaim budget と backlog 取得 API、`SafeStateSwapper` / `DeferredFreeThread` / `AudioEngine.Threading` に backlog 診断を追加し、非 RT 側の reclaim burst を抑制（2026-05-12）。
- B-9: `DeferredFreeThread::shutdownAndDrain()` を追加し、`ConvolverProcessor::releaseResources()` から明示的に停止・join・drain を実行する順序へ変更（2026-05-12）。

---

## 🔴 重要度 A：即時修正を強く推奨（可聴ノイズ・クラッシュ・リソース破損）

| ID | 項目 | ファイル | 問題の概要 | 推奨改修方針 |
| --- | --- | --- | --- | --- |
| **A-1** | `unconstrainedToTheta()` の Nyquist 極配置 | `src/AllpassDesigner.cpp` | θ→π を許容し、群遅延式の分母が極小化して数値的不安定性を引き起こす。CMA‑ES の収束を阻害。 | θ上限 `0.99π` を設定。FFT 分解能を考慮し `max(0.01π, 2π/N)` も検討。 |
| **A-2** | `computeIRHash()` の衝突・TOCTOU 競合 | `src/AllpassDesigner.cpp` | 先頭 1024 バイトのみでハッシュ計算。IR 後半の変更を検出できず、古いキャッシュを再利用。読み取り中のファイル変更で不正ハッシュ生成。 | ファイル全体の xxHash64 に変更。読み取り後にサイズ・更新時刻を再確認する二段検証を追加。 |
| **A-3** | `SecondOrderAllpass::response()` の位相不連続 | `src/AllpassDesigner.h` | 分母が 1e‑12 未満で (1,0) を返し、位相が突然ゼロに跳ぶ。CMA‑ES の目的関数に不連続な段差を生む。 | 分母の下限を設定し、連続的に振幅補正する安全な除算を実装。長期的には lattice/parcor 形式に移行。 |
| **A-4** | `std::exp()` のオーバーフロー／アンダーフロー | `src/AllpassDesigner.cpp` | CMA‑ES の Gaussian サンプリングが大きな値を生成し、`exp(-x)` がオーバーフローして FP アシストや数値的不安定性を引き起こす。 | 分岐とクランプ (max 50) による安定シグモイドを実装。 |
| **A-5** | 新ランタイムのトランザクショナルでない公開 | `src/audioengine/AudioEngine.Commit.cpp` | `currentDSP`, `fadingOutDSP`, クロスフェードフラグ, レイテンシが個別の atomic で更新され、オーディオスレッドが「新 DSP + 旧クロスフェード」など矛盾した状態を観測しうる。 | `EngineRuntime` 構造体を導入し、DSP グラフ・クロスフェード状態・レイテンシ設定を **単一の atomic で publish** する。 |
| **A-6** | `retireObject` とクロスフェードの所有権分離 | `AudioEngine.h` / `AudioEngine.Commit.cpp` | `currentDSP`, `fadingOutDSP`, `retireObject` の三重寿命管理。クロスフェード完了前に reclaim が走ると二重解放の危険。 | ランタイム全体を retire の単位に統合。`fadingOutDSP` は統合された `EngineRuntime` の一部として管理。 |
| **A-7** | `prepareToPlay`/`releaseResources` の状態機械不在 | `AudioEngine.Processing.PrepareToPlay.cpp` / `AudioEngine.Processing.ReleaseResources.cpp` | JUCE ホストが重複 `prepare` や `release` 抜きの `prepare` を発行しても、ワーカースレッドやビルダーが残存し、リソース破損やクラッシュを起こす。 | `EngineLifecycleState` を atomic 管理し、重複 `prepare` や不正遷移をガードする。 |
| **A-8** | MKL DFTI ハンドルの手動解放漏れ | `AllpassDesigner.cpp` 他 | 多数の早期リターンで `DftiFreeDescriptor` が漏れ、リークが蓄積する。 | `ScopedDftiDescriptor` + `put()/reset()/release()` パターンへの統一を実施済み（2026-05-11）。 |
| **A-9** | Nyquist bin の強制量子化による最適化不連続 | `AllpassDesigner.cpp` | Nyquist bin を +1/-1 に二値化し、CMA‑ES のコスト関数に不連続点を作る。Hermitian 制約上完全回避は不可能だが、軽減できる。 | 目的関数側で Nyquist bin の重みを低減する（例: `weight *= 0.1` at `k=half`）。 |
| **A-10** | シャットダウン順序の暗黙的依存 | `AudioEngine.CtorDtor.cpp` / `ConvolverProcessor.Lifecycle.cpp` | ワーカー停止→解放の順序が文書化されておらず、将来の変更で循環待機や終了時クラッシュを引き起こす。 | シャットダウン状態機械を `RUNNING→STOP_ACCEPTING_WORK→STOP_AUDIO→STOP_WORKERS→FORCE_EPOCH_ADVANCE→DRAIN_RETIRE→DESTROY` に正式化する。 |
| **A-11** | **Publication ordering の不十分性（hybrid state 観測）** | `AudioEngine.Commit.cpp` | 複数の atomic 更新が互いに異なるキャッシュラインに分散しており、メモリモデル上は `new currentDSP + old fade` など「論理的にありえない」組み合わせが合法で観測され得る。クリック／ポップ／オーバーラップバッファ不一致を引き起こす。 | A-5 と統合。単一 `std::atomic<EngineRuntime*>` で全状態を不可分に公開する。 |
| **A-12** | **ランタイムオブジェクトの mutable state 汚染** | `AudioEngine.Processing.DSPCoreDouble.cpp` 他 | `DSPCore` が Immutable なトポロジと mutable な処理状態（AGC, DC ブロッカー, FIR ヒストリ等）を混在させている。これが所有権を複雑化し、クロスフェード時の安全性を損ない、warmup を難しくしている。 | `ImmutableDSPGraph` + `PerAudioThreadDSPState` に分離する。短期的には、少なくともクロスフェード参加オブジェクトの mutable 状態を初期化する手順を明確化する。 |
| **A-13** | **Warmup 不完全によるトランジション過渡応答** | `AudioEngine.Commit.cpp` | 新 DSP 公開前の warmup が曖昧。FIR オーバーラップ・パーティション積算器・AGC エンベロープ等の初期化が不十分で、切り替え直後にクリックや HF バーストが発生する。 | ビルダーが `runtime->requiredWarmupBlocks()` を提供し、コミット前にその回数だけ無音で `process()` を実行する。 |
| **A-14** | **Runtime ビルド入力がリアルタイムパラメータと非同期** | `AudioEngine.RebuildDispatch.cpp` | ビルド依頼から実際のビルド完了までに UI パラメータが変化し、ビルド済みランタイムと現在のオーディオスレッドのパラメータが不一致になる（特にオーバーサンプリング比・FIR 分割サイズ等）。 | ビルド開始時に全パラメータを `BuildRequest::ParameterSnapshot` に凍結し、そのスナップショットに基づいてビルドする。 |
| **A-15** | **ABA 的ランタイム再利用** | ランタイム管理全般 | メモリ解放後に同じアドレスが別のランタイムに再利用され、ポインタ一致を前提とした誤った最適化（将来追加された場合）が働くと、クロスフェードをスキップしたり不正なキャッシュをヒットさせる危険。 | ランタイム識別に `RuntimeUUID` を導入し、ポインタ一致ではなく UUID で比較する。 |
| **A-16** | **クロスフェード時のレイテンシ不整合** | `AudioEngine.Commit.cpp`, `ConvolverProcessor.Runtime.cpp` | FIR 分割・オーバーサンプリング比・EQ 位相特性の変更時、新旧 DSP の内部レイテンシが異なり、単純なクロスフェードが時間軸で misalign する。コームフィルタ・過渡応答の二重化・クリックが発生。 | クロスフェード構造に `latencyDelta` を保持し、オーディオスレッド側で遅延補償しながらクロスフェードする。 |
| **A-17** | **Warmup と denormal suppression の順序問題** | `DSPCore` warmup パス | warmup が正しく行われても、denormal 抑制状態の初期化順序が逆だと、実信号入力後だけ denormal バーストが起こる（IIR テール・エンベロープフォロワ等で特に危険）。 | warmup フェーズを `INIT → ZERO_STATE → DENORMAL_GUARD_ENABLE → WARM_SIGNAL → READY` の状態機械化。 |

---

## 🟡 重要度 B：計画的な改善を推奨（品質・堅牢性・将来リスク低減）

| ID | 項目 | ファイル | 問題の概要 | 推奨改修方針 |
| --- | --- | --- | --- | --- |
| **B-1** | FFT Hermitian 処理の厳密化 | `AllpassDesigner.cpp` | C2C FFT の丸め誤差で Hermitian 対称性が微妙に崩れる。実害は小さいが、高精度最適化で影響しうる。 | 長期的に Real FFT (R2C/C2R) へ移行。 |
| **B-2** | `gridSearch2D()` 周波数範囲の固定 | `AllpassDesigner.cpp` | 候補周波数が 20kHz 固定で、高サンプルレート時に Nyquist までの最適化が行われない。 | 探索上限を `0.45 * sampleRate` に動的設定。 |
| **B-3** | `adaptiveGradientDescent()` の有限差分 ε | `AllpassDesigner.cpp` | ε=1e-6 固定がパラメータスケールに合わず、勾配推定精度が悪い。 | 周波数・ゲインごとに相対ステップを使用。 |
| **B-4** | Audio thread 不変性のアサーション強化 | 全ソース | RT 禁止操作の混入をレビューに依存しており、防御が弱い。 | `RT_AUDIO_THREAD_ONLY` / `NON_RT_ONLY` マクロと `ASSERT_NON_RT_THREAD` / `ASSERT_AUDIO_THREAD` を導入。 |
| **B-5** | ワーカースレッド例外の包括的捕捉 | `NoiseShaperLearner.cpp` | 未捕捉例外が `std::terminate` でプロセスを強制終了するリスク。 | 全ワーカーエントリポイントを try-catch で包囲。 |
| **B-6** | `prepareToPlay()` 部分確保失敗のロールバック | `AudioEngine.Processing.PrepareToPlay.cpp` | レイテンシバッファ確保の途中で失敗すると、既に確保したバッファがリークし、不整合状態が残る。 | 失敗時点で保持中の全バッファを解放する。 |
| **B-7** | Epoch reclamation の進捗保証 | `core/EBRQueue.h`, `SafeStateSwapper.h` | オーディオスレッドが停止（ブレークポイント、ホストサスペンド）すると epoch が進まず、解放待ちキューが際限なく蓄積する可能性。 | キューサイズ監視と診断ログ、強制排出機能を追加。 |
| **B-8** | Automation burst のパラメータ整合性 | パラメータ更新系全般 | DAW オートメーションで、ブロック内の複数パラメータが個別に更新され、一貫しない組み合わせを処理してしまう。 | パラメータ変更をブロック境界でまとめて commit する `ParameterSnapshot` を導入。 |
| **B-9** | Deferred deletion スレッドのシャットダウン競合 | `DeferredFreeThread.h`, `AudioEngine.CtorDtor.cpp` | ワーカースレッドが終了しても、解放スレッドがシングルトン破棄後に走り、プロセス終了時にクラッシュ。 | A-10 のシャットダウン状態機械に `DRAIN_RETIRE` 前にグローバルサービス停止を組み込む。 |
| **B-10** | **Audio thread のフォールバックパス不在** | オーディオエンジン全体 | ランタイムビルド失敗・IR 破損・MKL 初期化失敗時に、オーディオスレッドが安全に動作するフォールバック（無音またはバイパス）が定義されていない。 | RT‑safe な `BypassRuntime` または `SafeSilentRuntime` を定義し、致命的エラー時に自動的に切り替える。 |
| **B-11** | **Deferred reclaim のバーストによる非 RT スレッドの長時間停止** | `EBRQueue.h`, `DeferredFreeThread.h` | 大量のランタイム解放が短期間に集中すると、Message thread / Worker thread が解放処理で長時間ブロックされ、UI の応答性低下やビルド遅延を引き起こす。 | 1 サイクルあたりの reclaim 数制限（reclaim budget）を導入し、残りは次回以降に持ち越す。 |
| **B-12** | **Runtime builder の決定性不足** | `AllpassDesigner.cpp`, `NoiseShaperLearner.cpp` | CMA‑ES の RNG シードやスレッドスケジューリングによって、同一の設定・IR から異なる最適化結果が生成される可能性がある。キャッシュ無効化やデバッグ再現を困難にする。 | ビルドハッシュに最適化シード・バージョンを組み込み、再現可能なビルドを保証する。 |

---

## 🔵 重要度 C：長期的な保守性・性能改善（余裕がある時に対応）

| ID | 項目 | ファイル | 推奨改修方針 |
| --- | --- | --- | --- |
| C-1 | `ScopedAlignedPtr` ムーブ代入の防御的改善 | `AlignedAllocation.h` | `ptr = std::exchange(o.ptr, nullptr)` 形式に統一し、意図を明確化する。 |
| C-2 | ノイズシェイパー学習の False Sharing 対策 | `NoiseShaperLearner.h` | `candidatePopulation` と `candidateFitness` を `alignas(64)` で宣言。 |
| C-3 | アロケータドメインの統一 | `CmaEsOptimizer.h`, `ConvolverRuntime.h`, `PreparedIRState.h`, `MklFftEvaluator.h` | 64byte アラインの直接 `mkl_malloc` / `mkl_free` を `convo::aligned_malloc` / `convo::aligned_free` に集約し、解放 API の取り違えを防止。 |
| C-4 | `LinearRamp` の thread-compatible 性強化 | `DspNumericPolicy.h` | `reset` / `setCurrentAndTargetValue` を非RT、`setTargetValue` / `getNextValue` / `isSmoothing` をRT として明示し、単一ライター規約をアサーションで固定する。 |

---

## 📋 優先修正ロードマップ（完了履歴）

| 順位 | ID | 項目 | 分類 |
| --- | --- | --- | --- |
| 1 | A-5 / A-11 | トランザクショナルな EngineRuntime 公開 | 状態整合性・ノイズ防止（完了） |
| 2 | A-6 / A-12 | 所有権一元化と mutable state 分離 | メモリ安全・長期安定性（完了） |
| 3 | A-7 | EngineLifecycle 状態機械の導入 | 再初期化耐性（完了） |
| 4 | A-14 | ビルド入力のパラメータスナップショット化 | パラメータ整合性（完了） |
| 5 | A-1, A-4 | θクランプ・安定シグモイド | CMA-ES 安定性（完了） |
| 6 | A-3 | allpass 位相不連続の除去 | 最適化品質（完了） |
| 7 | A-2 | IR ハッシュ再設計 | キャッシュ完全性（完了） |
| 8 | A-8 | DFTI RAII 化（完了: 2026-05-11） | リソースリーク（完了） |
| 9 | A-10 | シャットダウン状態機械の正式化 | 終了時安定性（完了） |
| 10 | A-13 | トランジション warmup の必須化 | 可聴品質（完了） |
| 11 | A-15 | ランタイム UUID の導入 | 将来の誤最適化防止（完了） |
| 12 | A-16 | クロスフェード・レイテンシアラインメント | 可聴品質（完了） |
| 13 | A-17 | ウォームアップ／denormal 状態機械 | 数値的安定性（完了） |
| 14 | A-9 | Nyquist bin 重み緩和 | 最適化収束（完了） |

---

**本報告書は、全コードベースの徹底監査に基づく最終品質評価です。現状の ConvoPeq が抱える本質的課題は「mutable な DSP インスタンスのホットスワップ」であり、真の安定化には Immutable Runtime Graph への移行が不可欠です。しかし、上記の短期改修を優先して適用することで、現在のアーキテクチャ上でも大幅な堅牢性向上が見込めます。**
**優先度 A 項目は引き続き長期改善テーマとして扱い、B/C 項目は本監査時点で修正済みです。**
