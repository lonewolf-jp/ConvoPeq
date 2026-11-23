# Gate 1: Reader 側 memory-model 安全性の分析と修正設計（実装未承認）

Owner 指摘（2026-11-11）の「`EQProcessor.Processing.cpp:489` の acquire load が unlink 前の値を読み得る実行を、現在の happens-before 関係では排除できていない」に対する分析である。**製品コードは変更していない。**

## §0 結論

1. **保護された経路（reader slot 取得成功）は、現在のコードでも safety が成立する**ことを happens-before で示せる（§2）。
2. **保護されない経路が存在する。** `EQProcessor::process()` は `RCUReaderGuard` の入場成否を確認せずに `loadCurrentState()` を実行する（`Processing.cpp:488-489`）。slot 取得に失敗すると（slot 枯渇・登録クローズ・所有者不一致）、**epoch 保護なしで状態ポインタを読み、使用する**。これが Gate 1 の実体である（§3）。
3. 修正設計は「入場に失敗したら状態を読まず、所有権のあるフォールバック状態を使う」1 点（§4）。
4. reclaim 駆動頻度を上げる変更（v3 のドライバ）は、この穴の窓を広げる。**したがって段階①の前提として本修正が必要**という Owner 判断は妥当である（§5）。

## §1 参加者と順序（現行コード）

| 主体 | 操作 | 順序 |
|---|---|---|
| Writer（UI/control） | `exchangeCurrentState(new, acq_rel)`（`Core.cpp:266`）→ `retireEQStateDeferred(old)`（epoch = 現 global）→ `m_epochAdvancePending = true` → （後刻）`flushPendingEpochAdvance()` で `publishEpoch()`（release）→ `tryReclaim()` | この順 |
| Reader（audio） | `RCUReaderGuard` 構築 → `enterReader(slot)`（`slot.epoch = currentEpoch()` を release、`depth++` を acq_rel）→ `loadCurrentState(acquire)`（`Processing.cpp:489`） | この順 |
| Reclaimer | `minReaderEpoch()`（depth>0 の slot の最小 epoch）で `isOlder(entry.epoch, minReader)` を判定 | — |

## §2 保護された経路の safety（証明）

記号: `S_B` = `exchangeCurrentState` の store（`currentStateBits`）、`S_E` = `publishEpoch` の store（`globalEpoch`）、`L_E` = reader の `currentEpoch()`（acquire）、`L_B` = reader の `loadCurrentState()`（acquire）。

1. `S_B` は `S_E` より前に sequenced-before（`Core.cpp:266` → flush 内 `:87`）。
2. `S_E` は release store。`L_E` が `S_E` の値を読めば、`S_E` synchronizes-with `L_E`。
3. したがって `S_B` happens-before `L_E`（release の効果＋transitivity）。
4. reader の `L_E` は `enterReader` 内で `L_B` より前に sequenced-before。`L_B` は acquire なので、**後続の操作が `L_B` より前に移動することはない**。すなわち `L_E` → `L_B` の program order は保たれる。
5. `L_B` は `S_B` happens-before の後に位置するため、coherence により `S_B` 以降の値を読む（new ポインタ）。**stale（unlink 前）は読めない。**

reader が `L_E` で **E（publish 前）** を読んだ場合は、reader は E を announce するので `minReader = E`。retire された entry の epoch も E であり `isOlder(E, E) = false` → reclaim されない。**どちらの分岐も安全。**

> 補足: `L_B` の acquire が「後続を前に動かさない」ことを使っている。`L_B` を relaxed に変えると 4 が崩れるため、**acquire を維持する必要がある**。

## §3 保護されない経路（Gate 1 の実体）

```cpp
void EQProcessor::process(juce::dsp::AudioBlock<double>& block)
{
    convo::RCUReaderGuard guard(rcuReader);                        // ← 失敗し得る
    const auto* stateSnapshot = loadCurrentState(std::memory_order_acquire);  // ← 無条件
    ...
}
```

`RCUReader::enter()` は次の場合に **slot を取得せずに戻る**（`rootEnterSucceeded_ = false`）。

| 失敗条件 | 箇所 |
|---|---|
| 所有者 CAS の失敗（別スレッドが所有者） | `RCUReader.h:51-60` |
| `acquireThreadSlot()` が -1（slot 枯渇・登録クローズ） | `RCUReader.h:62-82` |

このとき `enterReader()` は呼ばれないため、**depth も epoch も公開されない**。Reclaimer から見ると「reader はいない」ので `minReader = currentEpoch`。そこへ writer が publish して reclaim が走ると、`process()` が読んだ旧ポインタは解放済みになり得る（UAF）。

`Processing.cpp` には `rootEnterSucceeded()` の確認が **1 か所も無い**（`grep` で 0 件）。したがって、この経路は「未証明」ではなく**反例が構成できる**。

### 3.1 反例の構成

1. Writer が `exchange(new)` して old を epoch E で retire。
2. flush が `publishEpoch()`（E+1）→ `tryReclaim()`。この時点で active reader は 0（対象 slot の audio スレッドはまだ入場していない、または入場に失敗している）。
3. `minReader = E+1` → old（E）を解放。
4. audio スレッドの `process()` が入場に失敗したまま `loadCurrentState()` を実行し、old を読む（release/acquire の同期が無いので、直前の store が見えていなくても矛盾しない）。
5. old を使用 → UAF。

## §4 修正設計（最小）

### 4.1 方針

**入場に失敗したら、共有状態を読まない。** 所有権が明確なフォールバック状態（メンバ）を使う。

```cpp
// 設計案（未実装）
convo::RCUReaderGuard guard(rcuReader);
const bool protectedRead = guard.rootEnterSucceeded();     // 追加
const auto* stateSnapshot = protectedRead
                          ? loadCurrentState(std::memory_order_acquire)
                          : &fallbackState_;               // 追加（メンバ・退役しない）
```

- `fallbackState_` は `EQProcessor` が所有し、**retire の対象にしない**（`EQState` のニュートラル設定）。したがって解放されない。
- `protectedRead == false` の間は EQ がニュートラル（素通し相当）になる。音は止まらず、UAF も起きない（fail-safe）。
- 既存の `active()` 相当の判定を再利用できる（`ScopedEpochReader::active()` と同型の API が `RCUReaderGuard` にあるかは実装時に確認）。

### 4.2 代替案（比較）

| 案 | 内容 | 評価 |
|---|---|---|
| **A（推奨）** | 失敗時はオーナー所有のフォールバック状態 | 追加コストは分岐 1 回。UAF を構造的に排除 |
| B | 失敗時は「前ブロックの状態ポインタ」を使う | 前ブロックのポインタも解放され得るため不可 |
| C | 失敗時は処理をスキップ（無音） | 音が途切れる。A の方が安全側で品質も保てる |
| D | slot 枯渇時に slot を強制解放して入場させる | 他 reader の安全性を壊す。不可 |

### 4.3 追加で確認すべき点（実装時）

- `RCUReaderGuard` に成否取得 API があるか（無ければ `rcuReader.rootEnterSucceeded()` を直接使う）。
- `fallbackState_` の初期化内容（既存の「初期 EQState」生成経路を再利用）。
- RT 制約: 追加は分岐のみ（確保・ログ・待機なし）。

## §5 reclaim 駆動頻度との関係（段階①の前提）

現状、EQ ドメインの reclaim は `prepareToPlay` / `releaseResources` / 破棄のときだけ走る。したがって §3 の窓は「その 3 回」にしか開かない。v3 のドライバは 100 ms ごとに reclaim するため、**窓の開く頻度が数百倍になる**。

よって「Reader safety 未解決のまま reclaim 駆動だけ増やす」ことはできない、という Owner 判断は技術的に正しい。段階①の前提として §4 の修正が必要である。

## §6 検証計画（Gate 1 修正）

| # | 検証 | 方法 |
|---|---|---|
| 1 | 入場失敗時に共有状態を読まない | 単体テストで slot を枯渇させ、`process()` がフォールバックを使うことを確認（`protectedRead` の分岐を観測） |
| 2 | 保護経路の safety | 既存の `STG10ReaderQuarantineTests` / `ScopedEpochReaderStructuralTests` と、`EpochDomain` の単体（enter→publish→reclaim の順序）で確認 |
| 3 | 反例の再現（可能なら） | slot 枯渇 + 高頻度 reclaim でのストレス（ASan ビルド）。x86 では窓が狭いため、`ThreadSanitizer` 相当は Windows で使えない点を明記 |
| 4 | 回帰 | 既存 60 テスト |

## §7 状態

- 本稿は分析と修正設計のみ。製品コードは未変更（HEAD `c94d3767`）。
- 段階①（reclaim ドライバ実装）は、§4 の修正承認後に着手する。
