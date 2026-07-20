# ConvoPeq ソースコード調査報告書（Part 5）

`ISRRTExecution.*` と `ISRDSPHandle.*` のレビュー結果です。

---

## 0. 今回のサマリ

| # | 重大度 | ファイル | 概要 |
|---|--------|----------|------|
| 6 | **Medium**（現状の動作に影響なし。安全網が機能していない） | `ISRRTExecution.cpp/.h` | RT-safety自動検知用の"Capability Firewall"/"Allocator Firewall"は、フラグの立て下ろし（`enter`/`leave`/`markRTContext`）自体はAudio Threadに正しく組み込まれているが、実際に違反を検知してassertする関数（`auditPublishAttempt`, `onAllocAttempt`）を呼び出す箇所がコードベース中に一つも無い。 |
| 7 | **要検証**（コンパイル未検証。ビルドが通らない可能性あり） | `ISRDSPHandle.h` | `std::atomic<DSPHandle>`（16byte構造体）に`is_always_lock_free`の静的検証が無い。兄弟フィールドの`std::atomic<uint64_t> generation`には同様の`static_assert`があるのに、こちらだけ無い非対称性を確認。 |

---

## 1. 【Medium】RT Capability/Allocator Firewall が未接続

### 該当ファイル

`src/audioengine/ISRRTExecution.cpp` / `ISRRTExecution.h`

### 発見の経緯

`ISRRTExecution.cpp`のコメント（`RTCapabilityFirewall::enter()`内、47行目付近）に開発者自身による以下の記述があります:

> 前提: isRTContext() 自体は実装済みだが、現時点のコードベースで呼出箇所は存在しない。

これを手掛かりに、関連する4つの関数の呼び出し状況を実際に横断検索しました。

### 検証結果

| 関数 | 役割 | 呼び出し元 |
|------|------|-----------|
| `RTCapabilityFirewall::enter()`/`leave()` | RTコンテキストフラグの立て下ろし | **`AudioEngine.Processing.AudioBlock.cpp`と`BlockDouble.cpp`のAudio Thread冒頭・末尾で正しく呼ばれている**（RAIIガード経由、2箇所） |
| `RTAllocatorFirewall::markRTContext(bool)` | 同上（別実装） | 同上、2箇所で正しく呼ばれている |
| `RTCapabilityFirewall::auditPublishAttempt()` | `publishAtomic`がRTコンテキスト中に呼ばれていないか検査してassert | **呼び出し箇所ゼロ**（定義・宣言のみ） |
| `RTAllocatorFirewall::onAllocAttempt()` | ヒープ確保がRTコンテキスト中に発生していないか検査してassert | **呼び出し箇所ゼロ**（定義・宣言のみ） |

さらに、`onAllocAttempt`のコメントには「Debug/CI build: operator new / malloc override で呼ばれる」とありますが、コードベース全体を検索しても `operator new` のグローバルオーバーライドは**存在しません**（`grep`で0件を確認）。同様に `AtomicAccess.h` の `publishAtomic()` の実装を確認しましたが、`auditPublishAttempt()` の呼び出しは含まれていません（単純な `std::atomic_store_explicit` のラッパーのみ）。

### 何を意味するか

- **フラグの追跡自体は正しく動作しています**（Audio Thread中は`sharedRtContextFlag`が正しく`true`になります）。
- しかし、そのフラグを実際に**チェックして違反を検知する側の仕組みが未接続**のため、Debug/CIビルドであっても「Audio Thread中にヒープ確保やpublishAtomicが起きたら即座にassertで検知する」という、コメントが示唆する自動防御は**現状機能していません**。
- これは今回のPart 3〜4で報告した Finding #4（`EQProcessor::reset()`が"(Audio Thread)"を名乗りながらlibm呼び出しを含む）のような問題を、本来なら自動検知できたはずが検知できない一因と考えられます。せっかく構築された安全網（`RTExecutionFrame`, `FirewallToken`, `RTTraceRelay`を含む一連のISR Layer 1インフラ）が、最後の一歩（実際の違反検知フック）で接続されていないため、"実装されているが機能していない"状態です。

### 推奨対応（パッチは提示しません — 設計判断が必要なため）

2つの方向性が考えられ、どちらもプロジェクト全体への影響範囲が大きいため、具体的な実装は貴殿の設計判断を仰ぎたく、あえてパッチ化していません:

1. **`onAllocAttempt`側**: Debug/CIビルド限定で `operator new`/`operator delete` のグローバルオーバーライドを実装し、`RTAllocatorFirewall::onAllocAttempt()` を呼ぶ。最も網羅的（見落としがない）ですが、プロジェクト全体のアロケーションに介入するため影響範囲が広いです。
2. **`auditPublishAttempt`側**: `AtomicAccess.h::publishAtomic()` から呼び出す案が最も自然ですが、`AtomicAccess.h`は汎用ユーティリティヘッダであり、そこに`ISRRTExecution.h`（audioengine固有ヘッダ）への依存を追加すると層構造上の逆依存になる可能性があります。依存関係を確認の上、Debug/CI限定のコンパイルスイッチ等で慎重に接続することを推奨します。

いずれの対応であっても、Release/NDEBUGビルドでは各関数の中身が空になる設計（`#if JUCE_DEBUG || CONVO_CI_BUILD`）のため、製品ビルドのパフォーマンスに影響はありません。

---

## 2. 【要検証】DSPHandle アトミックのロックフリー保証が未検証

### 該当ファイル・行

`src/audioengine/ISRDSPHandle.h`（`DSPHandleRuntime`クラス、ローカル160-165行目）

### 現在のコード

```cpp
private:
    std::array<DSPRegistrySlot, MAX_DSP_SLOTS> registry_{};
    std::atomic<DSPHandle> activeRuntimeDSPHandle_{ DSPHandle::null() };
    std::atomic<DSPHandle> fadingRuntimeDSPHandle_{ DSPHandle::null() };

    std::vector<CrossfadeRecord> crossfadeRecords_;
```

`DSPHandle`は`{uint32_t slot; uint64_t generation;}`という16byte程度の構造体です（アライメント都合で実質16byte）。同ファイル内の`DSPRegistrySlot::generation`（`std::atomic<uint64_t>`, 8byte）には

```cpp
static_assert(std::atomic<uint64_t>::is_always_lock_free,
    "atomic<uint64_t> must be lock-free on x64 for ISR Runtime");
```

という明示的な検証がありますが、同様の16byte版の検証が`activeRuntimeDSPHandle_`/`fadingRuntimeDSPHandle_`には存在しません。

### なぜ問題になり得るか

`activeRuntimeDSPHandle_`は（`getActiveRuntimeDSPHandle()`経由で）Audio Threadから毎コールバック参照される可能性が高い値です。C++標準では16byte（128bit）の`std::atomic<T>`が常にロックフリーであることは**保証されていません**。実装がロックフリー化に失敗した場合、内部的に`std::mutex`相当の機構にフォールバックし、**Audio Thread内でロックを取得する**という規約違反が、`std::atomic`という「一見安全に見える」型の裏で発生し得ます。

### 実際のリスクは低いと考えられる根拠

本プロジェクトはAVX2必須（コーディング規約）であり、AVX2をサポートするCPU（Haswell以降）は全て`CMPXCHG16B`命令をサポートします。MSVC(v143)のx64向け`std::atomic<T>`実装は、16byte型に対してこの命令ベースの真のロックフリー実装を提供するのが一般的です。そのため実運用上、実際にロックが使われる可能性は低いと考えられます。

### 検証根拠・パッチ提示にあたっての重要な注意

**このstatic_assertが実際にMSVC v143でコンパイルを通るかは、本調査環境（Windows/MSVC不在）では確認できていません。** 兄弟フィールドの`generation`（8byte、確実にロックフリー）とは異なり、16byteアトミックの`is_always_lock_free`がコンパイラの保守的な判定により`false`になる可能性はゼロではありません。

- もしコンパイルが通れば、それ自体が「16byteアトミックの安全性が型システムレベルで保証された」という有益な検証結果になります。
- もし**コンパイルが通らない場合**、それはこのstatic_assertの不備ではなく、**`activeRuntimeDSPHandle_`が実際にロックフリーでない**という、より重大な実装上の発見になります。その場合は本パッチを適用せず、`DSPHandle`をslot(uint32_t)とgeneration(uint64_t)の2つの独立した`std::atomic`に分割する等、より大きな設計変更が必要になります。

そのため、**必ずMSVC環境で単体コンパイル確認をしてから適用してください**。

```diff
--- a/src/audioengine/ISRDSPHandle.h
+++ b/src/audioengine/ISRDSPHandle.h
@@ -158,8 +158,13 @@
     void emitOwnershipTrace(const std::filesystem::path& outputPath) const;
 
 private:
     std::array<DSPRegistrySlot, MAX_DSP_SLOTS> registry_{};
     std::atomic<DSPHandle> activeRuntimeDSPHandle_{ DSPHandle::null() };
     std::atomic<DSPHandle> fadingRuntimeDSPHandle_{ DSPHandle::null() };
+    // ★ FIX: DSPRegistrySlot::generation の static_assert(is_always_lock_free) と
+    //   同様の検証が DSPHandle（16byte: uint32_t+padding+uint64_t）には無かった。
+    //   16byteアトミックはCMPXCHG16Bに依存するため、規約上のAVX2必須要件下では
+    //   実質問題ないはずだが、コンパイル時に明示検証し将来の環境変更を検知する。
+    static_assert(std::atomic<DSPHandle>::is_always_lock_free,
+        "atomic<DSPHandle> must be lock-free on x64/AVX2 target for ISR Runtime (RT thread reads activeRuntimeDSPHandle_)");
 
     std::vector<CrossfadeRecord> crossfadeRecords_;
```

---

## 3. 検証したが問題なしと判断した箇所（追加分）

### 3.1 `DSPHandleRuntime::create()`/`resolve()` の世代カウンタによる公開パターン

`create()`は非アトミックな`reg.instance = dspInstance`書き込みの**後に**`publishAtomic(reg.generation, gen, release)`を行い、`resolve()`は`consumeAtomic(reg.generation, acquire)`で世代を確認した**後に**`reg.instance`を読みます。Release-Acquireの正しいペアリングにより、非アトミックな`instance`フィールドへのクロススレッドアクセスは安全に同期されています。教科書的に正しいロックフリー実装です。

---

## 4. 調査範囲の更新

- `ISRRTExecution.cpp`/`.h`（354行）: 精読完了。
- `ISRDSPHandle.h`（199行）: 精読完了。`.cpp`（276行）は`create`/`resolve`/`beginCrossfade`/`activate`のみ確認、残り（`endCrossfade`, `retire`, `reclaim`, `quarantine`, `rollbackRegistration`等）は未確認。

---

## 5. 次のステップ（提案）

1. `ISRDSPHandle.cpp`の残り（`retire`/`reclaim`/`quarantine`周り）
2. `ISRDSPQuarantine.cpp/.h`
3. `ISRLifecycle.cpp/.h`
4. `ISRRuntimePublicationCoordinator.cpp/.h`（529行、ISR系最大ファイル）
5. `EQProcessor.Coefficients.cpp` の係数計算式の数式検証（まだ未実施のまま持ち越し）
