# ConvoPeq ソースコード調査報告書（Part 7）

`ISRRetire.cpp`（Vyukov MPSC queueベースのretire intent配送機構）のレビュー結果です。今回はこれまでで最も重大度の高い部類の指摘になります。

---

## 0. 今回のサマリ

| No | 重大度 | ファイル | 概要 |
|----|--------|----------|------|
| 9 | **High（潜在的）/ 現状は非該当を確認** | `ISRRetire.cpp` | `emitRetireIntentRT()`という名前にも関わらず、内部の`emitRetireIntent()`は輻輳時に`std::lock_guard<std::mutex>`を取得するフォールバック経路を持つ。現行の呼び出し元は全て非RTスレッドであることを確認したが、名前が実態と逆で、将来Audio Threadから誤って呼ばれるリスクが高い。 |

---

## 1. 【High（潜在的）】emitRetireIntentRT() の誤解を招く命名と内部のmutex

### 該当ファイル

`src/audioengine/ISRRetire.cpp`（`RetireRuntime::emitRetireIntent`: ローカル23-91行目、`emitRetireIntentRT`: ローカル93-96行目）

### 現在のコード

```cpp
void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    // ★ Step 1: MPSC Queue に slot を予約 (Vyukov protocol)
    const uint64_t ticket = convo::fetchAddAtomic(enqueueTicket_, 1, std::memory_order_acq_rel);
    const size_t idx = ticket % RETIRE_INTENT_QUEUE_SIZE;

    RetireIntent localIntent = intent;

    // ★ Step 2: bounded spin — Consumer が slot を解放するまで待機
    static constexpr int kMaxProducerSpin = 64;
    for (int spin = 0;; ++spin) {
        uint64_t slotSeq = convo::consumeAtomic(
            slots_[idx].sequence, std::memory_order_acquire);
        if (slotSeq == ticket) break;  // slot 獲得

        if (spin >= kMaxProducerSpin) {
            // ★ bounded spin 失敗 → tombstone + fallback
            slots_[idx].payload = RetireIntent{};
            slots_[idx].payload.dspSlot = UINT32_MAX;  // tombstone 識別子
            convo::publishAtomic(slots_[idx].sequence, ticket + 1, std::memory_order_release);

            std::lock_guard<std::mutex> lock(fallbackMutex_);   // ← ここ
            if (fallbackCount_ < FALLBACK_QUEUE_CAPACITY) {
                ...
            }
            ...
            return;
        }
        _mm_pause();
    }
    ...
}

void RetireRuntime::emitRetireIntentRT(const RetireIntent& intent) noexcept
{
    emitRetireIntent(intent);
}
```

`emitRetireIntent()`は、MPSCキューへの64回のbounded spinが失敗した場合（＝Consumer側の処理が追いつかず輻輳している場合）、`fallbackMutex_`という`std::mutex`をロックするフォールバック経路に入ります。

`emitRetireIntentRT()`は関数名から強く「Audio Thread（RT）から安全に呼べる版」であることを示唆していますが、実装は単に`emitRetireIntent()`を素通しで呼ぶだけで、**mutexロック経路を含めて完全に同一のコードパスを通ります**。つまりこの関数名は実態と逆で、"RT"を名乗りながら中身は全くRT-safeではありません。

### 現状の呼び出し元調査（結論: 現在は非RTのみ、ただし将来リスクは高い）

`emitRetireIntentRT`/`emitRetireIntent`の全ての本番コード（テスト除く）呼び出し元を洗い出し、それぞれのスレッドコンテキストを確認しました:

| 呼び出し元 | ファイル | スレッド確認方法 | 判定 |
|-----------|----------|-------------------|------|
| `emitRetireIntentRT()` | `AudioEngine.Commit.cpp`（`onRuntimeRetiredNonRt()`内） | 関数冒頭に`ASSERT_NON_RT_THREAD();`あり、関数名も`...NonRt` | 非RT |
| `emitRetireIntent()` ×2 | `AudioEngine.Processing.ReleaseResources.cpp` | JUCEの`releaseResources()`系コールバック（常に非RT） | 非RT |
| `emitRetireIntent()` ×1 | `AudioEngine.Timer.cpp` | 50ms周期timerCallback（コメントで確認） | 非RT |
| `emitRetireIntent()` ×3 | `ISRRuntimePublicationCoordinator.cpp`（`OverflowScheduler::drainOverflowRing`経由） | `drainOverflowRing`の呼び出し元コメント「50ms周期のtimerCallbackごとに呼出」で確認 | 非RT |

**結論**: 調査時点では、`emitRetireIntentRT`/`emitRetireIntent`のいずれも実際にAudio Threadから呼ばれている経路は見つかりませんでした。したがって**現状ではmutexロックがAudio Thread上で発生することはありません**。

### それでも報告する理由

これは「今動いているかどうか」よりも「今後も安全であり続けるか」の観点で重大なリスクだと判断しました:

- 関数名`emitRetireIntentRT`は、命名規約として最も自然な解釈が「RTスレッドから呼んでよい」であり、実装を読まずにこの名前だけを信じて将来Audio Threadから呼び出しコードを追加してしまうリスクが高いです。
- 万一Audio Threadから呼ばれた場合、通常時（spin成功時）は問題なく動作するため**開発中のテストでは発覚しにくく**、輻輳時（本番環境での高負荷時、Retire処理が溜まりやすいタイミング）にのみ`std::mutex`ロックが発生する、**再現困難な間欠的グリッチ/ドロップアウトの原因**になり得ます。
- Part 5・Part 6で報告した「安全網は用意したが接続されていない」パターン（RT Capability Firewall, LifecycleToken）とは逆に、こちらは「安全に見える名前だが実際には安全でない」という、より危険な逆方向のギャップです。もしPart 5のFirewallが接続されていれば、`emitRetireIntentRT`内の`std::mutex`確保をAudio Thread文脈で自動検知できたはずで、両者は関連する問題と言えます。

### 推奨対応

コード変更ではなく、まず命名・ドキュメントの是正を推奨します（具体的な実装変更は輻輳時のフォールバック戦略自体の再設計を伴うため、パッチとしては提示しません）:

1. `emitRetireIntentRT`という名前を、実態に即したもの（例: `emitRetireIntentFromNonRTCaller`等）に変更するか、最低限「このRTはRealTimeスレッド安全を意味しない」旨のコメントを関数直上に明記する。
2. 将来的に本当にAudio Threadから直接呼びたい場合は、mutexを使わない別実装（例: フォールバック失敗時は単純にdropしてカウンタだけ増やす、またはロックフリーのオーバーフロー専用リング（既存の`ISRRetireOverflowRing`）に直接pushする）を用意する。
3. Debugビルドで`emitRetireIntent()`内のmutex取得箇所に`ASSERT_NON_RT_THREAD()`相当のガードを追加し、万一将来RTスレッドから呼ばれた際に即座に検知できるようにする（Part 5で報告したFirewall接続の一環として実施するのが自然です）。

---

## 2. 追加の小さな観測: noexcept + 例外を投げ得る操作のパターンが再発

`RetireRuntime::dequeuePendingRetireIntents()`（ローカル137行目付近）は`noexcept`宣言ですが、内部で`std::vector<RetireIntent> result; result.reserve(128);`を実行しています。`vector::reserve`はメモリ確保失敗時に`std::bad_alloc`を送出しうるため、Finding #2（`IRAnalyzer.cpp`）と同型の「noexcept関数が例外を投げ得る操作を含む」パターンの3件目の再発です（1件目: `IRAnalyzer.cpp`、2件目は今回発見していませんが同種の指摘可能箇所が他にもある可能性を示唆）。

呼び出し元は`onRuntimeRetiredNonRt()`のみで、こちらも非RTスレッドであることを確認済みのため重大度は低いですが、OOM時に`std::terminate()`する経路である点は変わりません。正式なFindingとしては計上せず、パターンの記録として残します。

---

## 3. 調査範囲の更新

- `ISRRetire.cpp`（273行）: 精読完了。
- 併せて`emitRetireIntent`/`emitRetireIntentRT`/`dequeuePendingRetireIntents`の全呼び出し元（本番コード計7箇所）のスレッドコンテキストを横断確認。

---

## 4. 次のステップ（提案）

1. `ISRRetire.h`（構造体定義、キュー容量等の確認）
2. `ISRRetireRuntimeEx.cpp`（444行、Retire関連の拡張ロジック）
3. `ISRRuntimePublicationCoordinator.cpp`の残り（precheckPublish, PriorityScheduler, ShutdownScheduler等）
4. `EQProcessor.Coefficients.cpp`の係数計算式の数式検証
