# ConvoPeq ソースコード調査報告書（Part 4）

Part 3で「要確認」とした`EQProcessor::reset()`の呼び出し元調査、および`ISRRetireRouter`のレビュー結果です。

---

## 0. 今回のサマリ

| # | 重大度 | ファイル | 概要 |
|---|--------|----------|------|
| 4 | **情報（バグではなくコメント不整合）** | `EQProcessor.Core.cpp` / `AudioEngine.h` | `DSPCore::reset()`（→`EQProcessor::reset()`）は現行コードベース中**呼び出し箇所が一つも見つからない**ことを確認。"(Audio Thread)"というコメントは実態不明・要更新。 |
| 5 | **Low**（防御的アサート推奨、現状悪用経路なし） | `ISRRetireRouter.cpp` | `enqueueRetire`/`retireRT`/`retire`の3関数とも `ptr==nullptr \|\| deleter==nullptr` を同一分岐で「成功」扱いしており、「解放不要（ptr無し）」と「呼び出し側の実装ミス（ptr有りだがdeleter無し＝サイレントリーク）」を区別していない。 |

---

## 1. 【情報】EQProcessor::reset() の呼び出し元調査結果

Part 3で「要確認」としていた件について、以下を横断的に検索しましたが `AudioEngine::DSPCore::reset()` の呼び出し箇所を**一つも発見できませんでした**:

- `AudioEngine.Processing.AudioBlock.cpp` / `AudioEngine.Processing.BlockDouble.cpp`（Audio Thread本体、全行精読済み）
- `RuntimeBuilder.cpp`（新規DSPCore構築処理。`aligned_make_unique<DSPCore>()`で構築後、`convolverRt().setVisualizationEnabled`→`applyBuildSnapshot`→`transferIRStateFrom`→`prepare()`という初期化列を確認したが、`reset()`は呼ばれていない）
- `ProgressiveUpgradeThread.cpp`
- `src/audioengine/ISR*.*` 全ファイル（約20ファイル、7,400行規模）
- `src/tests/` 配下全ファイル

`void reset();` はヘッダ上パブリックメソッドとして宣言されているため、コンパイルエラーにはなりません。しかし実行時に到達する経路が見当たらないことから、**現行の呼び出しグラフ上ではデッドコードである可能性が高い**と判断します（テストからも参照されていないため、意図的なpublic APIとしての保持なのか、リファクタリング時の消し忘れなのかは判別できません）。

これは「確定バグ」ではありませんが、以下の理由で記録に値します:

- コメント「(Audio Thread)」を信じた将来の開発者・AIが、この関数をAudio Threadから安全に呼べると誤解し、実際にRTパスへ組み込んでしまうと、`process()`で【Fix Bug #7】として既に修正済みの`decibelsToGain`(libm)呼び出しパターンが**再発**します（`reset()`は直接呼び出し+`storeTotalGainDb()`経由の間接呼び出しの計2回、libmを呼びます）。
- 呼び出し元を追加する際は、`process()`と同じ設計（dB→linear変換はMessage Thread側で完結させ、Audio Thread相当のコードは`totalGainTarget`アトミックを読むだけにする）を踏襲する必要があります。単純な`storeTotalGainDb()`の呼び出し順序変更だけでは、`storeTotalGainDb()`自体が内部で`decibelsToGain`を呼ぶため、根本解決にはなりません。

### 推奨対応

コード変更ではなく、コメントの実態確認を推奨します。具体的には:

1. 意図的に将来のRTパス用に予約されたAPIであれば、コメントを「未使用。Audio Threadから呼ぶ場合はprocess()と同様のlibm回避が必須」等に更新する。
2. 単なる旧設計の残骸であれば、削除するか非RT専用である旨を明記する。

いずれの場合も実際のコード修正（パッチ）は必要と判断しなかったため、今回はパッチを作成していません。

---

## 2. 【Low】ISRRetireRouter: null deleter のサイレント成功扱い

### 該当ファイル

`src/audioengine/ISRRetireRouter.cpp`（`enqueueRetire`, `retireRT`, `retire` の3関数、ローカル97-155行目）

### 現在のコード（3関数とも同型パターン）

```cpp
RetireEnqueueResult ISRRetireRouter::enqueueRetire(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch,
                                                    DeletionEntryType type) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return RetireEnqueueResult::Success;
    ...
```

```cpp
bool ISRRetireRouter::retireRT(void* ptr, void (*deleter)(void*)) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return true;
    ...
```

```cpp
void ISRRetireRouter::retire(void* ptr, void (*deleter)(void*)) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return;
    ...
```

### 問題点

`ptr == nullptr`（解放すべきものが無い、正当なno-op）と `ptr != nullptr && deleter == nullptr`（解放すべきオブジェクトはあるのに解放関数が指定されていない、呼び出し側の実装ミス）が同一の条件式でまとめられ、**どちらも「成功」として扱われます**。

後者が実際に発生した場合、そのポインタは:
- 遅延削除キューに一切登録されない
- 呼び出し元には「成功」が返るためエラーとして検知されない
- 結果としてそのメモリは二度と解放されない（サイレントリーク）

`RetireEnqueueResult`列挙型（`Success`/`QueuePressure`/`QueueFull`/`Shutdown`）に呼び出し側エラーを表す値が存在しないため、Release挙動を変えずにこの2ケースを区別する手段はDebugビルド用アサートの追加が妥当と判断しました。

### 実害の見積もり

3関数の呼び出し元を確認した限り、`deleter`は毎回`&具体的な関数名`という**コンパイル時定数のリテラル**として渡されており、現行コードでは`ptr != nullptr && deleter == nullptr`という組み合わせが実際に発生する呼び出し箇所は見当たりませんでした。そのため重大度はLow（防御的ハードニング）としています。

### 修正パッチ

```diff
--- a/src/audioengine/ISRRetireRouter.cpp
+++ b/src/audioengine/ISRRetireRouter.cpp
@@ -100,6 +100,12 @@
                                                     DeletionEntryType type) noexcept
 {
     assert(provider_ != nullptr);
+    // ★ FIX: ptr!=nullptr かつ deleter==nullptr は呼び出し側のバグ（本来解放すべき
+    //   オブジェクトを渡しているのにdeleterが無い＝サイレントリークになる）であり、
+    //   ptr==nullptr（何もすることがない、正当なno-op）とは区別すべきである。
+    //   RetireEnqueueResultに専用のエラー値がないため戻り値はSuccessのまま維持し
+    //   （Release挙動は不変）、Debugビルドでのみ検出できるようassertで区別する。
+    assert(!(ptr != nullptr && deleter == nullptr) && "enqueueRetire: valid ptr with null deleter will silently leak");
     if (ptr == nullptr || deleter == nullptr)
         return RetireEnqueueResult::Success;
 
@@ -140,6 +146,7 @@
 bool ISRRetireRouter::retireRT(void* ptr, void (*deleter)(void*)) noexcept
 {
     assert(provider_ != nullptr);
+    assert(!(ptr != nullptr && deleter == nullptr) && "retireRT: valid ptr with null deleter will silently leak");
     if (ptr == nullptr || deleter == nullptr)
         return true;
     return provider_->enqueueRetire(ptr, deleter, provider_->currentEpoch());
@@ -149,6 +156,7 @@
 void ISRRetireRouter::retire(void* ptr, void (*deleter)(void*)) noexcept
 {
     assert(provider_ != nullptr);
+    assert(!(ptr != nullptr && deleter == nullptr) && "retire: valid ptr with null deleter will silently leak");
     if (ptr == nullptr || deleter == nullptr)
         return;
     (void)enqueueWithRetry(ptr, deleter, provider_->currentEpoch(), DeletionEntryType::Generic);
```

`retireRT`はRT-safe（本パッチのassert()自体もRelease/NDEBUGビルドではコンパイル時に消去され、Debugビルドのみ評価されるため、Audio Threadから呼ばれる`retireRT`のRT-safety規約にも抵触しません）。

他の`ISRRetireRouter`内メソッド（Epoch API群、`tryReclaim`, `pendingRetireCount`, `drainAll`など）は単純な委譲（provider_への転送）のみで、ロジック上の問題は見つかりませんでした。`enqueueWithRetry`のリトライループ（最大2回、`QueuePressure`以外は即座に打ち切り）も正しく実装されています。

---

## 3. 調査範囲の更新

- `RuntimeBuilder.cpp`（455行）: 精読完了。
- `ISRRetireRouter.cpp`/`.h`（約400行）: 精読完了。
- ISR系ファイル全体（約20ファイル）: `reset()`呼び出しの有無のみ横断検索（内容の精読は`ISRRetireRouter`のみ）。残り約19ファイルは未精読。

---

## 4. 次のステップ（提案）

1. ISR系の残り約19ファイル（`ISRDSPHandle`, `ISRDSPQuarantine`, `ISRLifecycle`, `ISRRuntimePublicationCoordinator` 等 — RCU publish/admissionの本体）
2. `EQProcessor.Coefficients.cpp` の係数計算式の数式検証（まだ未実施）
3. `DSPCoreFloat.cpp` / `DSPCoreIO.cpp` / `DSPCoreToBuffer.cpp`
4. `ConvolverProcessor.*` 一式
