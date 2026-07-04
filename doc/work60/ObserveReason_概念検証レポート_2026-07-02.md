# ObserveReason 概念検証レポート

**作成日**: 2026-07-02
**対象**: `ObserveReason` enum と `observeReason` 文字列の概念整合性
**方法**: ソースコード全件調査 + 設計文書照合 + ISR Runtime アーキテクチャ分析
**使用ツール**: grep (WSL), AiDex MCP, Serena MCP, ConvoPeq.md 全文照合

---

## 1. 調査の背景

`doc/work60/診断ログRT違反_numeric-only設計レポート_2026-06-30.md` で定義された `ObserveReason` enum:

```cpp
enum class ObserveReason : uint8_t {
    None = 0,
    RuntimeWorldMismatch = 1,
    AuthorityMismatch = 2,
    PendingObservation = 3
};
```

と、実装コード（`AudioBlock.cpp`, `BlockDouble.cpp`）で使用されている文字列:

| 文字列 | 意味 |
|--------|------|
| `"Forward"` | `dspSeq > lastObservedSeq`（単調増加） |
| `"Rollback"` | `dspSeq < lastObservedSeq`（減少） |
| `"Replay"` | 上記以外（現実装では到達不能） |

が、実装上 `static_cast<ObserveReason>(n)` で 1:1 にマッピングされているが、**このマッピングに技術的根拠があるかどうか**を検証する。

---

## 2. 検証方法

1. `ObserveReason` enum の定義箇所と全使用箇所を抽出
2. `observeReason` 文字列の生成ロジックを解析
3. 設計文書における両者の定義を照合
4. ISR Runtime アーキテクチャ上の「原因」概念との整合性を確認
5. Forward/Rollback/Replay と RuntimeWorldMismatch/AuthorityMismatch/PendingObservation が同一概念か別概念かを判定

---

## 3. 調査結果

### 3.1 Enum 定義（正本）

| ファイル | 行 | 備考 |
|---------|----|------|
| `src/audioengine/AudioEngine.h` | 353-357 | **唯一のコンパイル単位** |
| `doc/work60/診断ログRT違反_numeric-only設計レポート_2026-06-30.md` | 216-220 | 設計書の定義（コードと一致） |

```cpp
enum class ObserveReason : uint8_t {
    None = 0,
    RuntimeWorldMismatch = 1,
    AuthorityMismatch = 2,
    PendingObservation = 3
};
```

**導入経緯**: `ObserveReason` enum は `doc/work60/診断ログRT違反_numeric-only設計レポート_2026-06-30.md` で初めて定義され、同日の実装コミットで `AudioEngine.h` / `AudioBlock.cpp` / `BlockDouble.cpp` / `Timer.cpp` に同時追加された。それ以前（`追加観測設計書_2026-06-28.md`）では `const char* observeReason` として Forward/Rollback/Replay の文字列が直接使用されており、enum 型は存在しなかった。このことから、Forward/Rollback/Replay（文字列）→ enum（数値）→ RuntimeWorldMismatch 等（型安全な名前）という変換は、**Numeric-Only 化の過程で後から付与された名前**である可能性が高い。

### 3.2 observeReason 文字列の生成ロジック

**全4箇所で同一実装:**

| ファイル | 行 | 種類 |
|---------|----|------|
| `ConvoPeq.md` | 28883-28896 | 設計記述（AudioBlock版） |
| `ConvoPeq.md` | 29502-29515 | 設計記述（BlockDouble版） |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 419-436 | **実装コード** |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 386-401 | **実装コード** |

```cpp
static uint64_t s_lastObservedSeq = 0;
const char* observeReason = "";
if (runtimeWorld != nullptr) {
    dspSeq = runtimeWorld->metadata.publicationSequence;
    if (dspSeq > 0 && dspSeq != s_lastObservedSeq) {
        if (dspSeq > s_lastObservedSeq)
            observeReason = "Forward";
        else if (dspSeq < s_lastObservedSeq)
            observeReason = "Rollback";
        else
            observeReason = "Replay";
        s_lastObservedSeq = dspSeq;
        observeUs = convo::getCurrentTimeUs();
    }
}
```

**判定に使用している情報**: `dspSeq` と `s_lastObservedSeq` の大小比較 **のみ**。

### 3.3 Forward/Rollback/Replay → ObserveReason へのマッピング

```cpp
// AudioBlock.cpp:586-598, BlockDouble.cpp:536-548
uint8_t reasonVal = 0;
if (observeReason[0] != '\0') {
    if (std::strcmp(observeReason, "Forward") == 0) reasonVal = 1;
    else if (std::strcmp(observeReason, "Rollback") == 0) reasonVal = 2;
    else if (std::strcmp(observeReason, "Replay") == 0) reasonVal = 3;
}
event.data.dspTiming.reason = static_cast<ObserveReason>(reasonVal);
```

**暗黙のマッピング:**

| `observeReason` | reasonVal | `ObserveReason` 列挙値 |
|----------------|-----------|----------------------|
| `"Forward"` | 1 | `RuntimeWorldMismatch` |
| `"Rollback"` | 2 | `AuthorityMismatch` |
| `"Replay"` | 3 | `PendingObservation` |

### 3.4 Timer 側の文字列出力

```cpp
// AudioEngine.Timer.cpp:1020-1027
auto reasonToString = [](ObserveReason r) noexcept -> const char* {
    switch (r) {
        case ObserveReason::None:                  return "None";
        case ObserveReason::RuntimeWorldMismatch:  return "WorldMismatch";
        case ObserveReason::AuthorityMismatch:     return "AuthorityMismatch";
        case ObserveReason::PendingObservation:    return "PendingObserve";
        default:                                   return "Unknown";
    }
};
```

---

## 4. 概念分析

### 4.1 二つの概念の定義

#### 概念A: publicationSequence の変化方向

| 値 | 条件 | 意味 |
|----|------|------|
| Forward | `dspSeq > lastSeq` | Publish が正常に前進した |
| Rollback | `dspSeq < lastSeq` | Publish が後退した（rollback/republish） |
| Replay | 上記以外 | 通常は発生しない異常系 |

**出自**: `doc/work60/追加観測設計書_2026-06-28.md:550-578`
**判定に必要な情報**: `dspSeq` と `s_lastObservedSeq` の **2変数のみ**
**命名の意図**: sequence の値そのものの変化傾向

#### 概念B: Observe 発生原因（ISR Runtime）

| 値 | 名称から推測される意味 |
|----|---------------|
| RuntimeWorldMismatch | RuntimeWorld の世界が切り替わった（epoch/世代不一致） |
| AuthorityMismatch | Publication 権限（Authority）の競合が発生した |
| PendingObservation | 観測が保留状態になっている |

**出自**: `doc/work60/診断ログRT違反_numeric-only設計レポート_2026-06-30.md:216-220`
**判定に必要な情報**: runtimeWorld の epoch, authority token, transition state, worldId など複数の ISR Runtime 状態
**命名の意図**: DA(Data Authority)/ISR Runtime の内部状態変化の要因

### 4.2 比較表

| 比較軸 | 概念A（状態遷移） | 概念B（観測原因） |
|--------|------------------|------------------|
| **本質** | **What happened?**（何が起きたか） | **Why did it happen?**（なぜ起きたか） |
| **判定材料** | `dspSeq` の増減のみ | world epoch, authority, transition 等の複合情報 |
| **情報源** | `runtimeWorld->metadata.publicationSequence` | `runtimeWorld` の複数フィールド + ISR Coordinator 状態 |
| **現在の実装状況** | ✅ 完全実装 | ❌ **判定ロジック未実装** |
| **命名の粒度** | 現象レベルの 3 分類 | 原因レベルの 3 分類 |
| **1:1 対応か** | — | **証明されていない** |

### 4.3 マッピングの技術的根拠の有無

Forward→RuntimeWorldMismatch, Rollback→AuthorityMismatch, Replay→PendingObservation の対応付けを正当化する記述は、**現時点で確認できる全ドキュメントのどこにも存在しない**:

- `ConvoPeq.md`（全文約70,000行）
- `doc/work60/診断ログRT違反_numeric-only設計レポート_2026-06-30.md`
- `doc/work60/追加観測設計書_2026-06-28.md`
- `doc/work60/implementation_checklist.md`
- `doc/work60/diag_event_design_verification_2026-07-01.md`
- `doc/work/ISR_World_Bridge_Runtime.md`

コードコメント（`AudioBlock.cpp:586`）でも:
```cpp
// DspTimingData.reason は uint8_t。observeReason(const char*) を数値にマップ。
//   0 = None(空文字), 1 = Forward, 2 = Rollback, 3 = Replay
```
と記述されており、**RuntimeWorldMismatch 等の名前には一切言及していない**。

> **注**: これは「根拠が存在しない」ことを証明したのではなく、「現在確認できる範囲では根拠を確認できなかった」ことを意味する。
> 設計者の意図が別の文書や口頭伝達に存在した可能性は否定できないが、少なくとも追跡可能な形では残っていない。

---

## 5. 結論

### 5.1 Forward/Rollback/Replay と RuntimeWorldMismatch/AuthorityMismatch/PendingObservation は別概念

- **概念A**: publicationSequence の変化方向（現象レベルの3分類）
- **概念B**: Observe 発生原因（ISR Runtime 内部状態に基づく原因レベルの3分類）

現在の実装は **概念A の情報しか保持できない** にもかかわらず、`static_cast<ObserveReason>(reasonVal)` で強制的に概念B の enum に詰め込んでいる。

### 5.2 現在の実装のリスク

1. **ログのミスリード**: `reason=WorldMismatch` と出力されるが、実際は単なる sequence 増加の観測であり、本当に world mismatch が発生したかは不明。
2. **拡張時の不整合**: 将来、本当の原因判定ロジックを追加した際に、enum 値の意味が衝突する。
3. **Replay/PendingObservation の存在意義**: 現在の条件式 `if (>) else if (<) else` では `Replay` 分岐は現実装では到達不能である。この到達不能な値に対応する列挙子が `PendingObservation` という名前で定義されているが、その存在理由を説明する設計資料は確認できなかった。防御的コードまたは将来の改変を見越した残骸の可能性がある。

### 5.2b 補足: 「Unknown になる」という指摘の修正

前回のレビューで「Timer 側で Unknown が出力される」と指摘したが、**これは誤りである。**

実際の動作:
```
reasonVal=1
  ↓
ObserveReason(1) = ObserveReason::RuntimeWorldMismatch
  ↓
reasonToString(RuntimeWorldMismatch) → "WorldMismatch"
```

となるため、**Unknown にはならない。**

問題の本質は「Unknown が出る」ことではなく、**「ログには WorldMismatch と出力されるが、実際に検出したものは Forward（sequence 増加）である」** という**名前と実態の不一致**にある。

この不一致により、ログ解析時に「本当に RuntimeWorld の不一致が起きたのか、単に正常な publish 前進を観測しただけなのか」の区別がつかなくなる。

---

### 5.2c 「根拠が確認できない」という分析の補足

上記 4.3 で述べた通り、この対応付けの根拠は現時点で確認できない。現在の実装は:
```
Forward という文字列
  → 1 という整数値
  → static_cast<ObserveReason>(1)
  → ObserveReason::RuntimeWorldMismatch
```
という数値的な 1:1 マッピングに過ぎない。

すなわち:
- **意味** (Forward = sequence 増加) → **文字列** (Forward) → **数値** (1) → **enum** (RuntimeWorldMismatch)
という意味連鎖ではなく:
- **1 という整数を流しているだけ**
である。この点が設計上の根本問題である。

### 5.3 選択肢A: enum 名の改名（Phase 1 実装修正案）

**ObserveReason enum を「PublicationDirection」に変更する。**

これは現在の実装と完全に一致する最も単純な修正。

#### 修正対象

| # | ファイル | 変更内容 |
|---|---------|---------|
| 1 | `src/audioengine/AudioEngine.h:353-358` | `ObserveReason` → `PublicationDirection`、enum 値を `Forward=1, Rollback=2, Replay=3` に変更 |
| 2 | `src/audioengine/AudioEngine.Timer.cpp:1020-1027` | `reasonToString()` → `directionToString()`、ケース名変更 |
| 3 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:598` | `static_cast<ObserveReason>(reasonVal)` → `static_cast<PublicationDirection>(reasonVal) + direction` |
| 4 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:548` | 同上 |
| 5 | `doc/work60/診断ログRT違反_numeric-only設計レポート_2026-06-30.md` | 設計書の enum 定義を修正 |
| 6 | `DspTimingData` 構造体内のフィールド名 | `ObserveReason reason` → `PublicationDirection direction` |

#### 修正後のコードイメージ（Phase 1）

```cpp
// AudioEngine.h
enum class PublicationDirection : uint8_t {
    None = 0,
    Forward = 1,    // publicationSequence が増加（正常前進）
    Rollback = 2,   // publicationSequence が減少（後退）
    Replay = 3      // 上記以外（通常は発生しない）
};

struct DspTimingData {
    uint64_t dspSeq;
    uint64_t generation;
    uint64_t worldId;
    PublicationDirection direction;  // ★ 型名をデータの意味に一致
    uint64_t observeLatencyUs;
    uint64_t pubToObserveUs;
    uint64_t callbacksUntilObserve;
    uint64_t publishCallbackIdx;
};
```

```cpp
// AudioEngine.Timer.cpp
auto directionToString = [](PublicationDirection d) noexcept -> const char* {
    switch (d) {
        case PublicationDirection::None:      return "None";
        case PublicationDirection::Forward:   return "Forward";
        case PublicationDirection::Rollback:  return "Rollback";
        case PublicationDirection::Replay:    return "Replay";
        default:                              return "Unknown";
    }
};
```

```cpp
// AudioBlock.cpp / BlockDouble.cpp — DSP_TIMING push 部
event.data.dspTiming.direction = static_cast<PublicationDirection>(reasonVal);
```

#### DspTimingData 構造体サイズ: 変更なし（64B）

### 5.3b 選択肢B（参考）: 概念分離（PublicationDirection + ObserveCause）

> **注**: この選択肢は Phase 1 では採用せず、Phase 2 以降に延期された。
> Phase 1 で `PublicationDirection` が確定したため、ここではその名称を使用する。

Forward/Rollback/Replay（現象）と RuntimeWorldMismatch/AuthorityMismatch/PendingObservation（原因）は別概念であるため、**最初から別の enum に分離する設計**も有力な選択肢。

```cpp
// ★ 選択肢B: 概念を分離した設計
enum class PublicationDirection : uint8_t {
    None = 0,
    Forward = 1,    // publicationSequence が増加（正常前進）
    Rollback = 2,   // publicationSequence が減少（後退）
    Replay = 3      // 上記以外（通常は発生しない）
};

enum class ObserveCause : uint8_t {
    None = 0,
    RuntimeWorldMismatch = 1,  // RuntimeWorld の世代不一致（将来用）
    AuthorityMismatch = 2,     // Authority 競合（将来用）
    PendingObservation = 3     // 観測保留（将来用）
};

struct DspTimingData {
    uint64_t dspSeq;
    uint64_t generation;
    uint64_t worldId;
    PublicationDirection direction;  // ★ Phase 1 で改名済み
    ObserveCause cause;              // ★ Phase 2 で追加（padding 再利用）
    uint64_t observeLatencyUs;
    uint64_t pubToObserveUs;
    uint64_t callbacksUntilObserve;
    uint64_t publishCallbackIdx;
};
```

#### ABI 影響評価（選択肢B）

`DspTimingData` の現在のレイアウト:

```
offset 0:  dspSeq           (uint64_t, 8B)
offset 8:  generation       (uint64_t, 8B)
offset 16: worldId          (uint64_t, 8B)
offset 24: reason           (ObserveReason = uint8_t, 1B)
offset 25: [padding 7B]
offset 32: observeLatencyUs (uint64_t, 8B)
...
sizeof = 64 bytes
```

選択肢B で `direction`(1B) + `cause`(1B) + `padding 6B` に変更しても:

```
offset 24: direction (ObserveDirection = uint8_t, 1B)
offset 25: cause     (ObserveCause = uint8_t, 1B)
offset 26: [padding 6B]
offset 32: observeLatencyUs (uint64_t, 8B)
...
sizeof = 64 bytes ✅ 変更なし
```

**構造体サイズは 64 バイトで不変。** `LockFreeRingBuffer` のバッファサイズ、`DiagEvent` の static_assert、シリアライズ形式に一切影響しない。

#### 現在と将来の実装イメージ（選択肢B）

**Phase 2 移行後（RT側、AudioBlock.cpp）:**
```cpp
// PublicationDirection は Phase 1 で改名済み。cause は Phase 2 で追加。
event.data.dspTiming.direction = static_cast<PublicationDirection>(reasonVal);
event.data.dspTiming.cause = computeObserveCause(runtimeWorld);  // ★ Phase 2
```

**Timer 側のフォーマット（Phase 3）:**
```cpp
auto directionToString = [](PublicationDirection d) noexcept -> const char* { ... };
auto causeToString = [](ObserveCause c) noexcept -> const char* { ... };

log = "[DSP_TIMING] seq=..." +
    " direction=" + juce::String(directionToString(direction)) +
    " cause=" + juce::String(causeToString(cause)) + ...;
```

#### 選択肢B のメリット

1. **概念の分離**: 現象と原因が明確に区別される
2. **将来拡張可能**: 原因判定ロジックを追加しても Backward Compatible
3. **ログの明確性**: `direction=Forward cause=WorldMismatch` のように両方出力可能
4. **ABI 互性**: 構造体サイズ不変、既存バイナリと互換

#### 選択肢B のデメリット

1. **オーバーエンジニアリング懸念**: 将来の原因判定の具体的計画が存在しない
2. **余剰フィールド**: `cause` が当面常に `None` になる
3. **コード変更量が増える**: enum 追加 + 構造体変更 + フォーマット変更

### 5.4 将来の拡張（本件とは別タスク）

> **重要**: Phase 2 以降の前提として、`ObserveCause` の「情報源」が存在することが必須である。
> 現状、Coordinator / RuntimeWorld / Runtime Publication Layer はいずれも「Observe の原因」を
> DSP 側に伝達する仕組みを持っていない。`ObserveCause` を導入するには、まず Coordinator が
> `RuntimeWorld` に原因情報（例: `world.observeCause` フィールド）を供給する設計が必要。

真に「RuntimeWorldMismatch / AuthorityMismatch / PendingObservation」を区別したい場合:

1. `runtimeWorld` または ISR Coordinator から「Observe 原因」を伝達するフィールドを追加（**これが最優先: 情報源の設計**）
2. 新たな enum（例: `enum class ObserveCause`）を定義
3. 概念A（sequence 変化方向）と概念B（Observe 原因）の両方を `DspTimingData` に保持

---

## 6. 追加調査: 将来計画・命名分析・概念分離の評価

### 6.1 命名パターンの調査

#### 6.1.1 ISR Runtime の既存 enum 命名パターン

`src/` 内の `enum class` (`uint8_t`) 定義を調査した結果:

| パターン | 例 | 命名規則 |
|---------|-----|--------|
| `{Domain}{Qualifier}` | `ShutdownPhase`, `ShutdownBlockingReason`, `TransitionPolicy`, `RestorePhase` | ドメインで分類 |
| `{Verb}{Domain}` | `PublishAuthority`, `RetireAuthority`, `ShutdownAuthority` | 操作＋対象 |
| `{Domain}{Type}` | `RecoveryAction`, `RecoveryOutcome`, `ValidationFailureReason`, `FailureReason`, `PublishStage`, `FailureStage` | **最も一般的** |
| 略称接頭辞 | `DiagCategory`, `HBReorderScenario` | 限定的に使用 |

**最も一般的なパターン**: `{Domain}{Type}` — 例: `TransitionPolicy`, `RecoveryAction`

#### 6.1.2 命名候補の比較

| 候補 | 長さ | 自己文書化 | ドメイン明確性 | 既存パターンとの整合性 |
|------|------|-----------|---------------|---------------------|
| `ObserveDirection` | 16 chars | △「Observe」の方向 | ❌ 方向の主体が不明 | ❌ 非典型的 |
| `PublicationDirection` | 21 chars | ✅ **publication の方向** | ✅ publication が主体 | ✅ `PublishAuthority` に類似 |
| `PublicationSequenceDirection` | 30 chars | ✅ publication sequence の方向 | ✅ 最も正確 | △ 冗長気味 |

**推奨: `PublicationDirection`**

理由:
1. `publication` は `publicationSequence` の短縮形として ISR Runtime 内で一意に解釈される
2. `PublishAuthority` 等の既存 enum との命名一貫性がある
3. 型名だけで「publication の方向（Forward/Rollback/Replay）」であることを読み取れる
4. `direction` フィールド名との組み合わせで `event.data.dspTiming.direction` が直感的に理解できる

`ObserveDirection` を推奨しなかった理由:
- `Observe` というドメイン名は存在しない（Observe は DSP callback 内の処理名であり、ISR Runtime の型ドメインではない）
- 型名だけでは「何の方向なのか」が分からない
- 既存命名パターン（`RecoveryAction`, `TransitionPolicy`）との整合性が低い

#### 6.1.3 Phase 2 以降の ObserveCause 命名

`ObserveCause` は Phase 1 では導入しないが、将来の命名候補として:

| 候補 | 評価 |
|------|------|
| `ObserveCause` | ⚠️ やや適切。Observe 発生時にしか意味を持たないためドメインは Observe で統一可能 |
| `PublicationCause` | ❌ publication は方向であって原因ではない |

`ObserveCause` は有力候補（Observe の理由であり、Publication の理由ではない）。

### 6.2 ISR Runtime 将来計画の調査

調査対象:

- `doc/work60/` 配下の全設計書
- `doc/work37/`（Recovery System Plan）
- `doc/work16/`（ISR Runtime 関連）
- `doc/work/`（ISR Bridge 関連）
- `src/tests/`（ISR Authority テスト）
- `ConvoPeq.md` 全文

**結果:**

| 調査項目 | 結果 |
|---------|------|
| `TODO.*Observe` / `FIXME.*Observe` | **該当なし** |
| `work61` / `work62` / `work63` 参照 | **該当なし** |
| ObserveReason 拡張に関する Phase 番号 | **該当なし** |
| RuntimeWorldMismatch 判定のための worldId 比較コード | **該当なし**（worldId は診断目的のみ） |
| `ObserveDirection` / `SequenceDirection` / `ObserveCause` の既存定義 | **該当なし** |
| ISR Runtime の「原因」を DSP observe に伝達する仕組み | **未実装** |

**結論: ObserveReason の拡張に関する具体的な将来計画は存在しない。**

唯一関連するのは `work37 recovery_system_plan.md` の `RetireRootCauseEvidence`（Phase 9.61、原因6分類）だが、これは **retire スタックのデッドロック原因分析** であり、DSP Observe reason とは無関係。

### 6.2 AuthorityMismatch という名前の二重使用

`AuthorityMismatch` という名前は **2つの異なる文脈で独立に使用** されている:

| 文脈 | 場所 | 意味 |
|------|------|------|
| **Publication Reject** | `AudioEngine.Commit.cpp:63-100` | `validateRuntimeGraphAuthorityContract()` による publish 拒否条件 |
| **ObserveReason enum** | `AudioEngine.h:356` | DSP observe 時の理由（現在は Forward/Rollback/Replay と 1:1 マッピング） |

これらは **全く別の概念** であり、ObserveReason の `AuthorityMismatch` を publication reject の文脈で解釈すると誤った分析につながる。

### 6.3 概念分離の ABI 影響評価

**選択肢B（ObserveDirection + ObserveCause 分離）の DspTimingData レイアウト:**

```cpp
// 現在のレイアウト
struct DspTimingData {
    uint64_t dspSeq;              // offset  0 (8B)
    uint64_t generation;          // offset  8 (8B)
    uint64_t worldId;             // offset 16 (8B)
    ObserveReason reason;         // offset 24 (1B) ← uint8_t
    // [padding 7B]
    uint64_t observeLatencyUs;    // offset 32 (8B)
    uint64_t pubToObserveUs;      // offset 40 (8B)
    uint64_t callbacksUntilObserve; // offset 48 (8B)
    uint64_t publishCallbackIdx;  // offset 56 (8B)
};
// sizeof = 64B

// 選択肢B（Phase 2）のレイアウト
struct DspTimingData {
    uint64_t dspSeq;              // offset  0 (8B)
    uint64_t generation;          // offset  8 (8B)
    uint64_t worldId;             // offset 16 (8B)
    PublicationDirection direction; // offset 24 (1B) ← Phase 1 で改名済み
    ObserveCause cause;           // offset 25 (1B) ← uint8_t
    // [padding 6B]              // padding が 7B → 6B に減少
    uint64_t observeLatencyUs;    // offset 32 (8B) ← 変更なし
    uint64_t pubToObserveUs;      // offset 40 (8B) ← 変更なし
    uint64_t callbacksUntilObserve; // offset 48 (8B) ← 変更なし
    uint64_t publishCallbackIdx;  // offset 56 (8B) ← 変更なし
};
// sizeof = 64B ✅ 変更なし
```

**結論: 構造体サイズ不変。LockFreeRingBuffer、DiagEvent の static_assert、メモリレイアウトに一切影響しない。**

`reason` フィールドは既に 1 バイトの enum であり、その後ろに 7 バイトのパディングがある。このパディングのうち 1 バイトを `cause` に割り当てるだけで、オフセット 32 以降に完全な互換性を維持できる。

### 6.4 二つの選択肢の比較

| 比較軸 | Phase 1（enum 改名） | Phase 2+（概念分離） |
|--------|-------------------|-------------------|
| **工数** | 小（6ファイル、軽微な修正） | 中（6ファイル、enum追加+構造体変更） |
| **現在の実装との一致** | ✅ 完全一致 | ✅ 完全一致（direction のみ使用） |
| **将来の拡張性** | ⚠️ Phase 2 で再度方向 enum の変更が必要 | ✅ cause フィールドを追加するだけ |
| **ログの明確性** | ✅ Forward/Rollback/Replay が直接出力 | ✅ direction= + cause= で両方表示可能 |
| **ABI 互換性** | ✅ サイズ不変 | ✅ サイズ不変 |
| **オーバーエンジニアリング** | ✅ なし | ❌ **cause は常に None**（未使用フィールド） |
| **未使用フィールドの方針** | ✅ **現在観測可能な状態のみを型で表現** | ❌ 未実装の概念を先に型に持つ |
| **設計の一貫性** | ✅ `PublicationDirection` で型名と実データが一致 | ✅ 現象と原因を明確に分離 |

### 6.5 設計方針の観点からの評価

**本設計における一貫した方針:**

> **現在観測可能な状態のみを型として保持する。将来必要になるかもしれない概念を先に型に組み込まない。**

ObserveReason の型名に関する分析:

- `ObserveReason` という型名は「原因（Reason）」を示唆しているが、実際に格納されている値は publicationSequence の変化方向（Direction）である → **型名が保持しているデータの意味と一致していない**
- この不一致を解消するには、**型名を実際のデータの意味に合わせる**のが第一優先
- 「将来 Cause が必要になるかもしれない」は現時点では仮定に過ぎない（Coordinator / Runtime Publication Layer はいずれも Observe に原因情報を渡していない）
- 未実装の概念を先に型として持つことは、上記の方針に反する

### 6.6 推奨: Phase-based Migration（段階的移行）

**選択肢A を Phase 1 とし、将来 Coordinator が原因情報を供給可能になった時点で選択肢B へ移行する。**

```
Phase 1（今回実施）: ObserveReason → PublicationDirection へ改名
    ↓
Phase 2（Coordinator が ObserveCause を生成可能になった時）:
    ObserveCause enum を追加し DspTimingData に cause field 追加
    ↓
Phase 3: Direction + Cause の両方をログ出力
```

#### Phase 1: PublicationDirection への改名（最小変更）

```cpp
// AudioEngine.h — 変更後
enum class PublicationDirection : uint8_t {
    None = 0,
    Forward = 1,    // publicationSequence が増加（正常前進）
    Rollback = 2,   // publicationSequence が減少（後退）
    Replay = 3      // 上記以外（通常は発生しない）
};

struct DspTimingData {
    uint64_t dspSeq;
    uint64_t generation;
    uint64_t worldId;
    PublicationDirection direction;  // ★ 型名をデータの意味に一致
    uint64_t observeLatencyUs;
    uint64_t pubToObserveUs;
    uint64_t callbacksUntilObserve;
    uint64_t publishCallbackIdx;
};
```

```cpp
// AudioEngine.Timer.cpp — 変更後
auto directionToString = [](PublicationDirection d) noexcept -> const char* {
    switch (d) {
        case PublicationDirection::None:      return "None";
        case PublicationDirection::Forward:   return "Forward";
        case PublicationDirection::Rollback:  return "Rollback";
        case PublicationDirection::Replay:    return "Replay";
        default:                              return "Unknown";
    }
};
```

**変更対象ファイル:**

| # | ファイル | 変更内容 |
|---|---------|---------|
| 1 | `src/audioengine/AudioEngine.h:353-358` | `ObserveReason` → `PublicationDirection`、enum 値を `Forward=1, Rollback=2, Replay=3` に変更 |
| 2 | `src/audioengine/AudioEngine.Timer.cpp:1020-1027` | `reasonToString()` → `directionToString()`、ケース名変更 |
| 3 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:598` | `static_cast<ObserveReason>(reasonVal)` → `static_cast<PublicationDirection>(reasonVal)` |
| 4 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:548` | 同上 |
| 5 | `doc/work60/診断ログRT違反_numeric-only設計レポート_2026-06-30.md` | 設計書の enum 定義を修正 |
| 6 | `DspTimingData` 構造体内のフィールド名 | `reason` → `direction` |

**DspTimingData 構造体サイズ: 変更なし（64B）**

#### Phase 2（拡張案）: ObserveCause 追加（Coordinator が原因情報を供給する設計が確立した場合）

> **前提条件**: この拡張案は以下の条件が全て満たされた場合にのみ検討すべきである:
>
> 1. Coordinator / RuntimeWorld が Observe 原因（RuntimeWorldMismatch / AuthorityMismatch / PendingObservation）を判定し、DSP 側に伝達できる**情報源**の設計が確立されていること
> 2. 実際に `world.observeCause` のようなフィールドが RuntimeWorld に追加され、RT 側で読み取り可能であること
> 3. 上記の判定ロジックが単なる sequence 比較ではなく、ISR Runtime の内部状態（epoch, authority, transition）に基づいていること
>
> **現時点では 1-3 のいずれも未実装である。したがってこの拡張案は保留。**

将来、Coordinator が `runtimeWorld` の epoch/authority/transition 状態に基づいて Observe 原因を判定できるようになった場合の拡張案として:

```cpp
// 新規追加
enum class ObserveCause : uint8_t {
    None = 0,
    RuntimeWorldMismatch = 1,  // RuntimeWorld の世代不一致
    AuthorityMismatch = 2,     // Authority 競合
    PendingObservation = 3     // 観測保留
};

// DspTimingData に cause フィールド追加
// ★ 既存の 7B padding のうち 1B を cause に割り当てる
//   sizeof は 64B のまま不変
struct DspTimingData {
    uint64_t dspSeq;
    uint64_t generation;
    uint64_t worldId;
    ObserveDirection direction;  // Phase 1 で改名済み
    ObserveCause cause;          // ★ Phase 2 で追加（padding の一部を再利用）
    // [padding 6B]（従来は 7B）
    uint64_t observeLatencyUs;
    // ...
};
```

#### Phase 3: Direction + Cause の両方出力

```cpp
// AudioEngine.Timer.cpp
auto directionToString = [](ObserveDirection d) noexcept -> const char* { ... };
auto causeToString = [](ObserveCause c) noexcept -> const char* { ... };

log = "[DSP_TIMING] seq=..."
    + " direction=" + juce::String(directionToString(direction))
    + " cause=" + juce::String(causeToString(cause)) + ...;
```

### 6.7 段階的移行のメリット

1. **Phase 1 で直ちに型名と実データの不一致を修正できる** — 数時間の作業で完了
2. **Phase 2 のタイミングを Coordinator 側の実装状況に委ねられる** — 先走った設計をしない
3. **Phase 2 の ABI 互換性が事前検証済み** — 64B 不変、静的アサート通過確認可能
4. **各 Phase が独立してリリース可能** — 大規模変更を伴わない

### 6.8 まとめ

| 観点 | 評価 |
|------|------|
| 選択肢A（enum改名のみ）は Phase 1 として適切 | ✅ 型名と実データの意味が一致する |
| 選択肢B（概念分離）は Phase 2 以降に延期 | ✅ Coordinator が原因情報を生成可能になってから |
| **Phase 1 の優先度** | **高（今回の改修に含める）** |
| **Phase 2 の優先度** | **低（Coordinator 側の実装が完了してから）** |

## 7. 本レポートの立場

本レポートは「現在の実装と型名の意味が一致しているか」を検証対象としており、`ObserveReason` の命名が設計意図として誤りであることを断定するものではない。現時点で追跡可能な実装および設計資料に基づく限り、型名よりも `PublicationDirection` の方が現在保持しているデータの意味をより正確に表現している、と評価したものである。

## 8. 改訂履歴
7 | 最終調整。ObserveCause「自然」→「有力候補」、概念B「推定される意味」→「名称から推測される意味」、Replay「制御フロー上」→「現実装では」。結論の立場を明記する段落を追加
| 日付 | 版 | 変更内容 |
|------|----|---------|
| 2026-07-02 | 1.0 | 初版作成 |
| 2026-07-02 | 1.6 | Replay「将来の保険コード」→「設計意図は確認できなかった」。Phase 2 を「拡張案（Coordinator が情報源を供給する設計が確立した場合）」に明確化。表現の客観性を最終調整。|
