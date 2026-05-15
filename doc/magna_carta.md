# ConvoPeq 実装基準書（v2.0）

**― 決定論的並行 Immutable DSP Runtime 構築のための必須規約 ―**

本書は、ConvoPeq を「決定論的並行 Immutable DSP Runtime System」として完成させるための詳細実装基準を定義する。  
v1.0 からの主な改訂は、**Runtime Versioning の追加**、**所有権・退役境界の明確化**、**atomic / libm 規則の現実化**、および**クロスフェード中の時間的隔離規則の明文化**である。

---

## 1. 最重要原則：一方向データフロー

ConvoPeq は以下の**一方向データフロー**を破ってはならない。

```
UI/Input → Immutable Blueprint → Command Queue → Builder → Warmup → Validation → Atomic Publish → Audio Thread Consume → RCU Retire
```

**絶対禁止事項**
- Audio Thread → mutation / initialization / ownership transfer
- Published Runtime → mutation / parameter synchronization / lazy setup
- Audio Thread → logging / dynamic allocation / lock / libm / file I/O

---

## 2. Runtime Versioning（バージョン管理）

### 2.1 必須フィールド
すべての `RuntimeWorld` は以下の単調増加フィールドを持つこと。
```cpp
struct RuntimeWorld {
    uint64_t runtimeVersion;      // 単調増加、publish ごとにインクリメント
    uint64_t generation;          // 構築世代（再構築要求と対応）
    uint64_t transitionId;        // クロスフェード単位で一意
    // … runtime components …
};
```

### 2.2 バージョン順序の保証
- publish は `version` が単調増加することを保証しなければならない。
- 古い `version` の `RuntimeWorld` を publish しようとした場合、**無条件で拒否**すること。
- Audio Thread は観測した `version` を用いて stale な遷移を検出する。

---

## 3. RuntimeWorld の所有権と退役

### 3.1 単一公開単位
`std::atomic<RuntimeWorld*>` **のみ**を公開境界とする。部分公開は一切禁止。

### 3.2 所有権の明確化
- **Builder Thread** が `RuntimeWorld` を構築し、publish をもって所有権を放棄する。
- **退役システム（RCU retire）** のみが `RuntimeWorld` を破棄する権限を持つ。
- `delete runtime` の直接呼び出しは**禁止**。

### 3.3 退役キューの上限と診断
- 退役キューは**有限容量**とし、上限を超えた場合はエポックの進行が滞っているとみなし、警告ログを出力する。
- エポックが一定期間進行しない場合、**エポックストール診断**を発動し、全 Reader の状態をダンプすること。

---

## 4. Audio Thread 実装規約

### 4.1 Audio Thread は consume-only
**許可**：
- `const RuntimeWorld*` の読み取り
- 決定論的 DSP 処理
- クロスフェード進行（事前定義された遷移状態の更新のみ）
- 退役チェックポイントへの参加

**禁止**：
- 状態変更（mutation）
- 遅延初期化・キャッシュ構築
- 動的メモリ確保（`new`, `malloc`, `std::vector::resize` 他）
- `std::shared_ptr` の**所有権変更操作**（参照カウントの増減を伴うコピー／代入）  
  ※ `const` オブジェクトへの読み取り専用アクセスは、RCU 保護下で既に不要だが、完全禁止はしない。ただし Audio Thread 内で `shared_ptr` を新たに作成／破棄することは禁止。

### 4.2 Atomic 操作
- **許可**：`load(std::memory_order_acquire)`（公開境界の読み取り）、`fetch_add(relaxed)`（統計カウンタ、メーター等）
- **禁止**：`store`、`exchange`、`compare_exchange_strong`（同期セマンティクスを伴う書き込み全般）

### 4.3 libm 使用制限
**禁止対象**：実行時間の上限が保証されない関数
- `std::pow`、`std::exp`、`std::log`
- 三角関数 (`std::sin`, `std::cos`, `std::tan`)
- `std::sqrt` は一般に高速だが、分岐の可能性があるため**極力使用しない**（SSE/AVX 命令で代替）。

**許容**：`std::abs`、`std::floor`、`std::ceil` 等、定数時間で完了する単純関数。

---

## 5. Immutable Runtime 規約

### 5.1 Publish 後の不変性
`RuntimeWorld` が公開された後、いかなるスレッドからもその内容を変更してはならない。  
すべてのメンバは実質的に `const` であり、**遅延構築や内部キャッシュ更新は禁止**。

### 5.2 完全自己完結
`RuntimeWorld` はレンダリングに必要な全ての情報を内包する。外部のグローバル状態や動的ルックアップに依存してはならない。

### 5.3 不変トポロジと可変パラメータの階層分離（推奨）
トポロジ（IR 構成、フィルタ構造、OS 倍率など）を不変の `RuntimeWorld` に含める。  
ゲインやミックス比率などの可変パラメータは、軽量な `ParameterSnapshot` に分離し、`RuntimeWorld` 内の `atomic<const ParameterSnapshot*>` 経由で差し替え可能とすることで、不変性を保ちながら高頻度更新に対応する。

---

## 6. Publication 規約

### 6.1 Publish シーケンス
以下の順序を厳守する：
1. Build Complete（構築完了）
2. Warmup Complete（FFT/ブランチ/キャッシュの事前通電完了）
3. Validation Complete（動作確認）
4. Atomic Publish（`publishAtomic` ヘルパー使用）
5. Advance Epoch（エポック進行）
6. Retire Old（古い世界を退役キューに投入）

**この順序を逸脱した publish は禁止**。

### 6.2 Publication Helper の強制
生の `atomic<RuntimeWorld*>::store(...)` を直接使用してはならない。  
必ず次のようなテンプレート関数を通して公開すること：
```cpp
template<typename T>
void publishRuntime(std::atomic<const T*>& target, const T* newWorld) noexcept {
    target.store(newWorld, std::memory_order_release);
}
```

### 6.3 Publish 前ウォームアップの義務
新しい `RuntimeWorld` は、**最初の Audio Thread 処理が始まる前に**、FFT ワークバッファのウォームアップ、SIMD パスのキャッシュタッチ、デノーマルパスの初期化を完了させること。  
ウォームアップ未了の世界を公開してはならない。

---

## 7. Command Queue 規約

### 7.1 コマンドは不変の Value Object
すべてのコマンド（パラメータ変更、IR ロード要求、学習制御等）は、不変の `struct` として定義し、値をそのままキューに投入する。

### 7.2 キュー種別と実行スレッドの分離
単一の SPSC キューに全コマンドを詰め込むことは禁止。コマンドの**意味レイヤー**は統一するが、実行キューは性質に応じて分離する：

| コマンド種別 | 実行スレッド / キュー |
|--------------|----------------------|
| パラメータ更新 | Builder スレッド（パラメータキュー） |
| IR ロード要求 | IO ワーカー（IO キュー） |
| FFT プラン生成 | 前処理ワーカー |
| Publish 指示 | Builder スレッド（制御キュー） |

### 7.3 キュー溢れポリシー
各キューには上限を設け、溢れた場合の挙動（drop / coalesce / replace）を明示的に定義する。  
上限なしのキューは許可しない。

---

## 8. Builder 規約

### 8.1 Single Publication Authority
`RuntimeWorld` を公開する権限は**Builder スレッドのみ**が持つ。  
他のスレッドが直接 `publishRuntime` を呼んではならない。

### 8.2 Single Writer ≠ Single Worker（最重要）
重い前処理（IR 変換、FFT プラン生成、CMA-ES 学習）は別のワーカースレッドで行って構わない。  
しかし、**所有権の最終組み立てと公開は Builder スレッドの単独責務**である。  
これにより、部分構築状態の漏洩を防止する。

### 8.3 部分構築の可視化禁止
構築中の `RuntimeWorld` を一時バッファやグローバル変数として外部から参照できる状態にしてはならない。

### 8.4 再構築のキャンセル可能性
新しい構築要求が到着した際、進行中の古い構築タスクを安全に中断できる仕組み（`std::stop_token` 等）を備えること。

---

## 9. クロスフェード / 遷移の隔離（時間的隔離規則）

### 9.1 遷移状態の所有権
クロスフェードに必要な状態（ゲインランプ、レイテンシ整合情報）は、`RuntimeWorld` に属する**遷移専用オブジェクト**が所有する。  
`AudioEngine` のグローバル変数に散在させてはならない。

### 9.2 新旧世界の完全分離（Temporal Coexistence Rule）
クロスフェード中、`old RuntimeWorld` と `new RuntimeWorld` は**完全に独立したライフタイム**を持つ。  
- 共有の一時バッファ、遅延ライン、フィルター状態を使用しない。  
- 新旧の状態を混合する mutation は一切禁止。

### 9.3 遷移の決定論性
クロスフェードの進行は、ブロックサイズと経過サンプル数のみに依存し、ランタイムの状態に応じた動的分岐を伴ってはならない。

---

## 10. RCU / Epoch 規約

### 10.1 RCUReader コピー禁止
`RCUReader` はコピー／ムーブを禁止する：
```cpp
RCUReader(const RCUReader&) = delete;
RCUReader& operator=(const RCUReader&) = delete;
```

### 10.2 Reader のスレッド固定
各 `RCUReader` インスタンスは、単一のスレッドに固定して使用しなければならない。

### 10.3 退役タイミングの厳守
publish 完了後にエポックを進行させ、**古い世界はエポックに基づき安全に退役**させる。  
publish 直後に即時 `delete` するような危険な早期退役は禁止。

### 10.4 `reclaimAllIgnoringEpoch` の使用制限
最終シャットダウン時**のみ**使用を許可する。通常の運用パスでは絶対に使用しない。

---

## 11. メモリ管理規約

### 11.1 アロケータの統一
オーディオ処理に使用する全バッファは `convo::aligned_malloc` / `aligned_free` で管理し、`malloc`/`free` や `new`/`delete` との混在を禁止する。

### 11.2 `aligned_make_unique` の使用
生の `aligned_malloc` から placement new を呼び出す代わりに、例外安全なファクトリ関数 `aligned_make_unique<T>(...)` を常に使用する。

### 11.3 Audio Thread 内でのメモリ操作禁止
Audio Thread 内ではスタック変数のみ使用可能とし、動的確保やコンテナの拡張操作は一切禁止する。

---

## 12. Blueprint 規約

### 12.1 Immutable Blueprint
UI から Builder に渡されるパラメータの設計図（`EQBlueprint`, `ConvolverBlueprint` 等）は不変の値オブジェクトとする。

### 12.2 一方向伝達
Blueprint → Runtime の流れは一方通行であり、Runtime の状態を Blueprint にフィードバックしてはならない。

---

## 13. Shutdown 規約

### 13.1 状態機械による一元管理
複数の shutdown フラグを廃止し、単一の `EngineLifecycleState` 列挙型で状態を管理する。遷移は `compare_exchange_strong` で保護する。

### 13.2 固定された停止順序
1. オーディオコールバック停止
2. Builder スレッド停止
3. 全ワーカースレッド停止
4. 全退役キュー回収（この時点で全 Reader 退出済みのため `reclaimAllIgnoringEpoch` 可）
5. RuntimeWorld 破棄

---

## 14. SIMD / DSP 規約

### 14.1 デノーマル対策の必須化
Audio Thread の入口で `juce::ScopedNoDenormals`（または同等の MXCSR/FPCR 設定）を有効にする。

### 14.2 SIMD パスの一貫性
AVX/SSE の異なるパスで出力が完全に一致することを保証するか、決定論的な理由で許容する。

### 14.3 NaN 伝播の防御
フィルター出力、FFT 入出力、ボリューム計算の後段で、必ず `finite` チェックと安全な値へのクランプを行う。

---

## 15. AI 実装禁止事項

- **アーキテクチャ近道の禁止**：一時的な可変オブジェクトの共有で問題を回避しない。
- **RT 違反の隠蔽禁止**：`#ifdef DEBUG` で assert を外すことによりリリースビルドで RT 違反を通すことを禁止。
- **所有権迂回の禁止**：裸のポインタを外部に渡し、ライフタイム管理を曖昧にしない。
- **仮の互換レイヤーの乱用禁止**：旧 API を残すための shim は一時的とし、**削除期限と削除計画を必ず付与**すること。

---

## 16. 最終原則

ConvoPeq は **mutable DSP application ではない**。  
その本質は **「決定論的 immutable runtime orchestration system」** である。

改修のあらゆる場面で、次の問いを繰り返すこと：

> **「この変更は "runtime mutation" か？ "world replacement" か？」**

この規約に従うことで、ConvoPeq は x86／ARM を問わず、長期にわたって決定論的リアルタイム動作を保証するプロフェッショナル DSP エンジンとなる。