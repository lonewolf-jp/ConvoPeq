# IRState 全読み取り点の監査と ConvoPeq.md サイズ差の解消

作業日時: 2026-11-12 23:00〜
基準 commit: `173e6995`（HEAD = origin/main）
性質: **読み取り専用**。production source / test / CMake / git を変更していない。build / test / debugger / replay も実行していない。

---

## 0. 結果

```
ConvoPeq.md のサイズ差  = 説明済み（別ブランチの抽出物。分割は無損失）
IRState の C3           = NOT MET（deref する読み手 9 箇所に EBR slot が無い）
IRState の即時破棄       = 2 箇所（releaseResources / デストラクタ）＋ provider 不在分岐
A の IRState への適用    = 不可（guard 追加または同一性証明が先）
システム全体の N1        = NOT PROVED（変更なし）
実装 GO                  = 未承認
```

Owner の懸念は 2 つとも当たっていた。**IRState は「既存 slot を確認すればよい」段階ではなく、確認すべき slot が存在しない**（`ScopedEpochReader` / `RCUReaderGuard` の出現は convolver の IR 経路に 0 件）。`releaseIRState()` は no-op なので、acquire/release の呼出しは寿命保護を意味しない。読み手は rebuild / message / loader の各スレッドに散っており、writer は message / upgrade / rebuild の 3 系統である。

---

## 1. ConvoPeq.md のサイズ差（7,035,321 B vs 7,011,745 B）

### 1.1 結論

**別の抽出物である。** 7,035,321 B は ci-fix ワークツリーで生成された `ConvoPeq-ci-fix` で、main ツリーの `ConvoPeq` ではない。分割は無損失（4 ファイルの合計が元ファイルと完全一致）。

| 項目 | main（本監査の基準） | ci-fix（Owner が取得した分割の元） |
|---|---|---|
| ファイル | `C:\VSC_Project\ConvoPeq\ConvoPeq.md` | `doc/work113/20261112-0922_CI修正/ConvoPeq.md` |
| タイトル行 | `# Project Extract & Source Code: ConvoPeq` | `# Project Extract & Source Code: ConvoPeq-ci-fix` |
| ヘッダ stamp | `2026-11-12 00:48:16` | `2026-11-12 10:01:45` |
| サイズ | 7,011,745 B | **7,035,321 B** |
| SHA-256 | `1801bbbacf4d65484418a1cf978acf0c09fc728028b3c8c8105ddce95f25e14f` | `9b58252970adc8a8a33b2b5e6208c5766c168f96e5dc977c36005bb737b7cd3d` |
| 収録ファイル | 406 | 395 |
| split 合計 | 7,067,903 B（`split_20261111`、別世代） | **7,035,321 B（元ファイルと完全一致）** |

差は 23,576 B。分割ファイル（part01 1,949,901 / part02 1,949,999 / part03 1,949,942 / part04 1,185,479）の合計は 7,035,321 で、**ci-fix の ConvoPeq.md と 1 バイトも違わない**。つまり分割にヘッダ等は付いておらず、差は抽出元の違いだけである。

### 1.2 ファイル単位の差

2 つの抽出物を `### 📄 <path>` で分割して比較した結果、**35 ファイルが異なり、ci-fix にしか無いファイルは 0、main にしか無いファイルが 11**（`src/InputInspection.*`、`src/convolver/IRTrailingSilence.h`、`src/learner/*`、テスト 5 本 — いずれも main の未コミット新規ファイル）。

監査の結論に関係するファイルの差は、**CR5 修正の「atomic dot-call を convo:: wrapper に置き換える」変更**である。

```
src\core\EpochDomain.h      : compare_exchange_weak → convo::compareExchangeWeakAtomic（1 箇所）
src\core\SnapshotCoordinator.h: s_rtViolationCount.load → convo::consumeAtomic（1 箇所）＋ include
src\core\ScopedEpochReader.h: .store / .compare_exchange_strong → convo:: wrapper（4 箇所）
```

いずれも同じ memory_order で同じ操作に委譲するもので、**順序・フェンス・公開の意味は変わらない**。このため A の証明（フェンス配置、W1/W3）に対する評価は変わらない。

### 1.3 IRState に関する限り差は無い

`src\convolver\ConvolverProcessor.Lifecycle.cpp` は 2 つの抽出物で**完全に同一**である（差異ファイル一覧に現れない）。Owner が引用した `acquireIRState()` / `releaseIRState()` / `updateIRState()` の記述は HEAD と一致する。したがって §2 以降の指摘は HEAD に対する指摘として成立する。

---

## 2. IRState の全読み取り点

### 2.1 deref する読み手（9 箇所）

`acquireIRState()` の定義は `consumeAtomic(currentIRState, acquire)` を返すだけで（`Lifecycle.cpp:21-24`）、`releaseIRState()` は空（`:26-29`）。以下は「ポインタ取得から最後の dereference まで」の範囲と、その間に EBR slot があるかを見たものである。

| # | 箇所 | 何を読むか | 実行スレッド | 範囲 | slot |
|---|---|---|---|---|---|
| 1 | `ConvolverProcessor.h:1267-1272` `getIrAdditionalAttenuationDb` | `state->additionalAttenuationDb`（1 フィールド） | 呼出し元は `RebuildDispatch.cpp:750` = rebuild | load → 1 フィールド読み | **無し** |
| 2 | `ConvolverProcessor.h:1274-1280` `getIrFreqPeakGainDb` | `state->irFreqPeakGainDb`（1 フィールド） | `RebuildDispatch.cpp:751` = rebuild | load → 1 フィールド読み | **無し** |
| 3 | `ConvolverProcessor.h:1288-1310` `getIRGeometry` | `state->ir->…`（IR 全体の走査） | `RuntimeBuilder.cpp:493` = rebuild | load → IR 走査 | **無し** |
| 4 | `ConvolverProcessor.h:1316-1336` `transferIRStateFrom` | `srcState->ir` を `updateIRState` へコピー | `RuntimeBuilder.cpp:473` = rebuild | load → IR コピー | **無し** |
| 5 | `ConvolverProcessor.LoadPipeline.cpp:30-38` | `state->ir->getNumSamples()`、`state->sampleRate` | message（`loadImpulseResponse`） | load → 判定 | **無し** |
| 6 | `ConvolverProcessor.LoadPipeline.cpp:73-86` | `state->ir` / `state->sampleRate` を `LoaderThread` の ctor へ参照渡し | message（`loadImpulseResponse`） | load → ctor 呼出し | **無し** |
| 7 | `ConvolverProcessor.LoadPipeline.cpp:539-567` | `irState->ir` の全サンプル走査（ピーク／RMS 検証） | `applyComputedIR`（message または upgrade） | load → 全サンプル走査 | 無し（writer と同一スレッド） |
| 8 | `ConvolverProcessor.LoaderThread.cpp:918-922` | `owner.acquireIRState()` → deref | loader | load → deref | **無し** |
| 9 | `ConvolverProcessor.Rebuild.cpp:64-109` | `state->ir` 等 | rebuild | load → deref | **無し** |

### 2.2 deref しない読み（対象外）

| 箇所 | 内容 |
|---|---|
| `ConvolverProcessor.h:444`（`isIRLoaded`） | `currentIRState` の null チェックのみ |
| `ConvolverProcessor.MixedPhase.cpp:445` | `currentIRState` の null チェックのみ |

### 2.3 guard の実測

convolver の IR 経路（`ConvolverProcessor.h`、`LoadPipeline.cpp`、`Rebuild.cpp`、`LoaderThread.cpp`、`RuntimeBuilder.cpp`、`RebuildDispatch.cpp`、`MixedPhase.cpp`）を `ScopedEpochReader` / `RCUReaderGuard` / `acquireReadGuard` / `getRetireRouter()` で走査した結果、**一致は 1 件だけ**で、それは `RebuildDispatch.cpp:706` の **EQ 用** guard である。IRState 用の guard は 0 件。

したがって **C3（reader が EBR slot を持ち、load の前に公開している）は満たさない**。Owner の条件「全呼出し点で slot 取得を確認する」を満たす対象が存在しないので、「slot の存在確認で完了」という道は現状存在しない。

### 2.4 writer（置換側）

| 箇所 | 呼出し元 | スレッド |
|---|---|---|
| `LoadPipeline.cpp:640` `updateIRState` | `applyComputedIR`（`loadIR` 経由 = `AudioEngine.Parameters.cpp:214`、`StateAndUI.cpp:422`） | **message** |
| 同上 | `ProgressiveUpgradeThread.cpp:208` | **upgrade（専用スレッド）** |
| `LoadPipeline.cpp:897` `updateIRState` | `executePendingCommit`（コメントに「Message Thread のみ」） | **message** |
| `ConvolverProcessor.h:1262` | `setImpulseResponse` 系 API | message（推定） |
| `ConvolverProcessor.h:1323` | `transferIRStateFrom` | **rebuild** |

読み手 #1〜#4・#9 は rebuild、#5・#6 は message、#8 は loader。**rebuild の読み手と message/upgrade の writer は並行し得る**。フラグ（`isRebuilding` / `isLoading`）や generation 検査は「古い結果を捨てる」ためのもので、ポインタの寿命を守らない。

### 2.5 即時破棄（epoch を使わない経路）

| 箇所 | 内容 | 前提 |
|---|---|---|
| `Lifecycle.cpp:506-511`（`releaseResources`） | `exchangeAtomic(currentIRState, nullptr)` → `~IRState()` → `mkl_free` | 471-473 で upgrade 停止・`forceCleanup`・`activeLoader.reset()` の後。rebuild の停止は engine 側 |
| `Lifecycle.cpp:126-131`（`~ConvolverProcessor`） | 同上（直接 free） | デストラクタ。`forceCleanup` / `activeLoader.reset()` の後 |
| `Lifecycle.cpp:58-61`（`updateIRState`、provider 不在） | `deleter(oldState)` をその場で実行 | provider 未設定（テスト等）。production では設定済み |

これらは EBR を使わないため、**quiescence（読み手が止まっていること）の証明が別途必要**（N7 と同じクラス）。`releaseResources` は upgrade と loader を止めるが、rebuild スレッドは engine が止める。順序の証明はエンジン側の停止シーケンスに依存する。

---

## 3. 判定と、設計判断に必要な材料

### 3.1 IRState の現在地

| 項目 | 判定 |
|---|---|
| deref する読み手の列挙 | 9 箇所（§2.1） |
| 各読み手の EBR slot | **0 箇所** |
| slot 公開が load より前か | 該当なし（slot が無い） |
| slot 退出が最後の deref より後か | 該当なし |
| 別スレッドへの引き渡し | #6 が `LoaderThread` ctor へ参照を渡す（コピーか保持かは ctor 次第。要確認） |
| releaseResources / デストラクタ | 即時 free。quiescence の証明が別途必要 |

**結論: C3 は満たさない。** A のフェンスは scan visibility を閉じるだけで、slot を持たない読み手を守らない。IRState に A を適用するには、先に guard を追加するか、読み手と writer の同一性を証明する必要がある。

### 3.2 設計選択肢（実装しない・比較のみ）

| 案 | 内容 | 変更規模 | 評価 |
|---|---|---|---|
| (a) 全 deref 読み手に `ScopedEpochReader`（engine ドメイン） | `getRcuProvider()->getRetireRouter()` から provider を取る。9 箇所（うち #7 は writer と同一スレッドなので省略可） | 中 | 確実。すべて非 RT なのでコストは小さい。ただし §2.5 の即時 free は別途 quiescence が必要 |
| (b) スカラーだけ先に抜く | #1・#2 は 1 フィールド読みなので、`updateIRState` 時に atomic<float> へ複製しておけば deref 自体が不要 | 小 | 2 箇所だけ確実に消せる。残りは (a) が要る |
| (c) 同一スレッド性で閉じる | rebuild / loader の読み手を message スレッドへ移す等 | 大 | 現状は並行なので不可。移す設計なら (a) より高コスト |
| (d) IRState 全体を別機構（RCU + guard 付き）へ置換 | SafeStateSwapper と同様の方式へ統合 | 大 | 今回の範囲では過剰。まず (a)+(b) で足りる見込み |

推奨は **(b) + (a) の併用**（1 フィールド読みは複製で消し、IR 本体を読む箇所に guard を付ける）。ただしこれは設計判断であり、本稿では確定しない。

### 3.3 次段階（Owner の 5 段階に対応）

| # | 段階 | 状態 |
|---|---|---|
| 1 | READ-ONLY: IRState の全読み取り点と guard 範囲 | **完了（本稿 §2）** |
| 2 | READ-ONLY: ConvoPeq.md のハッシュ・サイズ差の解消 | **完了（本稿 §1）** |
| 3 | 設計判断: 既存 slot か guard 追加か | 材料を提示（§3.2）。判断待ち |
| 4 | 実装 GO の確認 | 未 |
| 5 | 実装・検証 | 未 |

---

## 4. 残存義務（更新）

| # | 義務 | 内容 | 状態 |
|---|---|---|---|
| W1-O6 | snapshot 2 経路の順序修正 | 承認済み・未実装 | 未 |
| W1-O7 | IRState の guard | §3.2 の設計判断 → 実装 | 未（材料あり） |
| W1-O8 | `LoaderThread` ctor の参照の扱い | #6 が参照を保持するなら、ctor 後も IRState の寿命が必要 | 未確認 |
| W1-O9 | releaseResources / デストラクタの quiescence | 即時 free の前提（loader/upgrade/rebuild の停止順） | 未（N7 と同一クラス） |
| A-O1〜A-O5 | 前回のまま | フェンス配置・W1/W3・CAS 線形化点・解放経路・RT コスト | 未 |
| N6 / N7 / N9 | 前回のまま | ガード無し deref / force drain の quiescence / 別機構 | 未 |

---

## 5. 証跡

`doc/work113/20261112-2300_IRState監査/evidence/`

| ファイル | 内容 |
|---|---|
| `irstate_excerpts.md` | IRState の読み取り・書き込み・即時破棄の逐語抜粋（10 ファイル / 797 行 / 38,068 文字、SHA-256 付き、HEAD blob 基準） |

2 つの抽出物の比較は、`### 📄 <path>` 単位の分割比較で行った（作業用スクリプトは一時ディレクトリのみ。リポジトリには置いていない）。

---

## 6. 停止条件の確認

* production source / test / CMake を変更していない。
* build / test / debugger / replay を実行していない。
* git の変更操作を行っていない。HEAD は `173e6995`、`origin/main` と一致。
* AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。

**本稿の提出をもって停止する。** IRState は C3 未達で、A の適用対象ではない。snapshot の実装も開始していない。実装 GO・Gate 1 承認は付与しない。
