# domain tag 実装の着手前チェック（変更範囲と既存未コミット変更の干渉）

実施: 2026-11-11 24:00 JST / 対象: HEAD `c94d3767` ＋ 未コミット変更のある作業ツリー

## 1. 実装が触るファイルと現在の状態

| ファイル | 現在の状態 | 実装での変更 |
|---|---|---|
| `src/core/RetireDomain.h` | 新設予定（未作成） | 新設: `enum RetireDomainId`（§2.3）。`<cstdint>` のみ include |
| `src/core/EpochDomain.h` | M（既存の未コミット変更あり） | 明示ドメイン ctor・`domainId_`・不正値カウンタ（§3.2） |
| `src/core/IRetireProvider.h` | clean（追跡済み・変更なし） | `retireDomainId()` の既定実装（§3.1） |
| `src/DeferredDeletionQueue.h` | M（既存の未コミット変更あり） | `DeletionEntry.domainId` と enqueue の刻印（§2.1・§3.2） |
| `src/audioengine/ISRRetireRouter.h` | clean（追跡済み・変更なし） | `retireDomainId()` の override と `TerminalReclaimAuthority`（:63）のシグネチャ変更（§3.1・§3.3） |
| `src/audioengine/ISRRetireRouter.cpp` | M（既存の未コミット変更あり） | sync / drain / drainAll へのドメイン引き渡し（§3.3） |
| `src/audioengine/ISRRetireLatency.h` | ??（未追跡の新規ファイル） | `retireLatencyClassified.domains[]` の集計（§5） |
| `src/audioengine/ISRHB.cpp` | M（既存の未コミット変更あり） | JSON 出力に `domains[]` を追加（`:354` の writer・§5.1） |
| `src/audioengine/AudioEngine.h` | M（既存の未コミット変更あり） | `EpochDomain` の生成値 `EngineISR`（`:5455`） |
| `src/eqprocessor/EQProcessor.h` | M（既存の未コミット変更あり） | `EpochDomain` の生成値 `EQ`（`:622`） |
| `src/ConvolverProcessor.h` | M（既存の未コミット変更あり） | `EpochDomain` の生成値 `Unknown`（`:1468`） |
| `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | M（既存の未コミット変更あり） | `EpochDomain` 生成 2 件 |
| `src/tests/AudioEngineHarness/V11ReaderReacquireTests.cpp` | clean（追跡済み・変更なし） | `EpochDomain` 生成 9 件 |
| `src/tests/D8_2_B_2_Tests.cpp` | M（既存の未コミット変更あり） | `EpochDomain` 生成 4 件 |
| `src/tests/invariant_INV3_INV5.cpp` | clean（追跡済み・変更なし） | `EpochDomain` 生成 3 件 |
| `src/tests/ScopedEpochReaderTests.cpp` | clean（追跡済み・変更なし） | `EpochDomain` 生成 4 件 |
| `src/tests/STG10ReaderQuarantineTests.cpp` | clean（追跡済み・変更なし） | `EpochDomain` 生成 4 件 |
| `src/tests/StuckReaderFallbackDrainTests.cpp` | clean（追跡済み・変更なし） | `EpochDomain` 生成 6 件 |
| `src/tests/STG11D2SnapshotRetireTests.cpp` | clean（追跡済み・変更なし） | `EpochDomain` 生成 3 件（`make_unique`） |
| `CMakeLists.txt` | M（既存の未コミット変更あり） | #15 の検証を新しいテスト実行ファイルにする場合のみ（既存ハーネスへ足すなら不要） |

## 2. 集計

| 区分 | 件数 |
|---|---|
| 対象ファイル | **20** |
| 既存の未コミット変更あり（M） | **10** |
| 未追跡の新規ファイル（??） | **1** |
| 追跡済み・変更なし（clean） | **8** |
| 新設予定（未作成） | **1** |

## 3. 干渉の評価

- 対象 20 件のうち **10 件がすでに未コミット変更を持つ**。今回の実装差分だけを切り出してレビュー・コミットすることは、現状のツリーではできない。
- とくに次の 7 ファイルは、Gate1／段階①／段階③の変更と同一ファイルで重なる:
  - `src/core/EpochDomain.h`
  - `src/DeferredDeletionQueue.h`
  - `src/audioengine/ISRRetireRouter.cpp`
  - `src/audioengine/ISRHB.cpp`
  - `src/audioengine/AudioEngine.h`
  - `src/eqprocessor/EQProcessor.h`
  - `src/ConvolverProcessor.h`
- `src/audioengine/ISRRetireLatency.h` は**未追跡**（段階③で新設）。ここへ `domains[]` を足すと、段階③の新規ファイルに変更が積み上がる。
- テストは 8 ファイル中 **2 件**（`STG11EQRetireTests.cpp`・`D8_2_B_2_Tests.cpp`）に既存変更がある。残り 6 件は clean。
- `CMakeLists.txt` も既に変更あり。**新しいテスト実行ファイルを増やす場合だけ**触る（既存ハーネスへ検証を足すなら不要）。
- したがって、ビルドと `ctest` の基準は「現状のツリーが green であること」を先に確認しないと、domain tag の影響と既存変更の影響を区別できない。

## 4. 実装前に決めておくこと（提案）

1. 着手直前に `git diff --stat` と `ctest` の結果を記録し、ベースラインを固定する。
2. domain tag の変更は「新設 1 ファイル＋既存 19 ファイル」に閉じる。整形・リネームを混ぜない。
3. 実装後は `git diff` を機能単位（①enum と ctor、②伝播、③刻印、④集計と JSON、⑤テスト）に分けて提示できる形にする。
4. 既存変更との切り分けのため、着手前の各ファイルの SHA-256 を記録しておく。

## 5. 着手前ハッシュ（切り分け用）

| ファイル | SHA-256 |
|---|---|
| `src/core/EpochDomain.h` | `94e1331b67c4206e951baa4de10b3d09eab3c03d04886f5539638a60fb5a1f40` |
| `src/core/IRetireProvider.h` | `47f5e492801dcdd374d460c037b21093b05ebd4cd49285c51791bdde02900d45` |
| `src/DeferredDeletionQueue.h` | `ab65b5e770f679145e8409abba777eede80499c0db44c8995acb62c19dcf083b` |
| `src/audioengine/ISRRetireRouter.h` | `67a3bb2ef62ad0bf05c845496bf1f1c54318e37047f41ed2eab8c45dc4af035e` |
| `src/audioengine/ISRRetireRouter.cpp` | `d11f6d87c0e6bbbecda7c38c102c14c9fa3677f6ae0002afb338610000af07d3` |
| `src/audioengine/ISRRetireLatency.h` | `6d25dc9f02f6d0e7bbb21c17129349b9726b03b6c49e2738e96da498131b7337` |
| `src/audioengine/ISRHB.cpp` | `61566df12fae0e935ba8f34460c0e5a41d1c6425dd452e416b61d7d324df63a5` |
| `src/audioengine/AudioEngine.h` | `6a5d3d1867ecc76cdb8fb8d142db73e3d7effbc2d4f5b4796339edf5190c872c` |
| `src/eqprocessor/EQProcessor.h` | `59f21513bd013c8a35fd675486145eb0242f6bc8d46178d3065078cab7818b37` |
| `src/ConvolverProcessor.h` | `16ba6f8ef85efa8c6672f302bc86c59658a566b64c6386c6025b40356955cb63` |
| `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | `6ee717d6f0f21d41ce7571d48ff4bda6a806840ed516f6c8c3dc656b03b9bf12` |
| `src/tests/AudioEngineHarness/V11ReaderReacquireTests.cpp` | `448931425e46963b26aca3f5a1cb1bf8ed5c408e0601d0bc9f8e4dca0a3992d5` |
| `src/tests/D8_2_B_2_Tests.cpp` | `4b78140b1b178f71fbdcb687494f5815ac577d104b29d6436282dc299c622671` |
| `src/tests/invariant_INV3_INV5.cpp` | `e8e44f5dd42366fe3dc0d9abbf7014a07635f6cf34dae570901d4e055b552b64` |
| `src/tests/ScopedEpochReaderTests.cpp` | `029fb00931b6f1d54bdcbe315a1f8ba7ca4f70d60eb9c3570a1f9c78d48c84c8` |
| `src/tests/STG10ReaderQuarantineTests.cpp` | `d921f01aa57d04855653e5c7570e19b98875ad78d1af63c3ee27ef084ba11b29` |
| `src/tests/StuckReaderFallbackDrainTests.cpp` | `36563d0eefe9838f742c5edc5178e4333c97c8f50891f5f0f5b5379053215819` |
| `src/tests/STG11D2SnapshotRetireTests.cpp` | `39cf8e8cb5f94655c4505a66454e7d5a98ca76f860b2161d60a3a20800a85d83` |
| `CMakeLists.txt` | `25f247d553bca62312e6cf1f7eab00951dd298796bc70670f5bb2087947c4bdf` |
