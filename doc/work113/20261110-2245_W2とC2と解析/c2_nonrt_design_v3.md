# C2 スナップショット退役リーク: Non-RT 専用 改訂設計 v3（実装未承認）

Owner 判断（2026-11-10）の指摘を反映した改訂版である。前版（v2 の Non-RT 案）からの変更点は **RT 違反時に状態を変更せず即座に返る**ことと、**検証計画の具体化**である。実装は行っていない。

## §1 RT 違反時の挙動（確定）

### 1.1 コード

```cpp
void SnapshotCoordinator::resetFadeStateAndRetireTarget() noexcept
{
    // ★ C2 (2026-11-10): 本 API は Non-RT 専用。RT からの呼出しは契約違反である。
    //   RT では「退役・移送・記録」を一切行わず、状態も変更しない。
    //   target はスロットに残るため、後続の Non-RT 呼出しが通常どおり退役させる。
    if (convo::numeric_policy::isAudioThread())
    {
        jassert (false && "resetFadeStateAndRetireTarget is Non-RT only");
        convo::fetchAddAtomic (m_rtViolationCount, std::uint64_t{1}, std::memory_order_release);
        return;
    }

    // ... 既存の処理（exchangeTarget → 記録 → enqueue/移送 → resetToIdle）...
}
```

### 1.2 何をしないか（明示）

| 操作 | RT 違反時 |
|---|---|
| `exchangeTarget` | **しない**（target の所有権はスロットに残る） |
| `recordRequest` | しない |
| `enqueueRetire` / Q / E / T への移送 | しない |
| `m_fade.resetToIdle()` | **しない**（状態を一切変更しない） |
| 例外・ブロック・確保 | しない（`isAudioThread()` と atomic 加算のみ） |

### 1.3 なぜフェード状態も変更しないか

- Owner 条件「RT 経路で退役・移送・記録を行わない」に加え、状態を部分的に変えると「target は残っているがフェード状態は idle」という中間状態が生まれる。
- 何も変えなければ、後続の Non-RT 呼出しが**同じ前提**で最初からやり直せる。中間状態の解釈が不要になる。
- `resetToIdle()` 自体は RT 安全だが、変えない方が契約が単純である。

### 1.4 検出の可視化

`m_rtViolationCount`（atomic）を追加し、既存の health / 診断出力に 1 行で出す。新しい判定式は作らない。Debug では `jassert` が先に止めるため、カウンタは主に Release の証跡になる。

## §2 呼出元の棚卸し（実作業ツリーで再確認）

Owner 指示に従い、2026-11-08 版の結合ソースではなく、**現在の作業ツリー**で再確認した（2026-11-10、`C:\VSC_Project\ConvoPeq`）。

```
$ grep -rn "resetFadeStateAndRetireTarget" src --include=*.cpp --include=*.h
src/core/SnapshotCoordinator.cpp:108:void SnapshotCoordinator::resetFadeStateAndRetireTarget() noexcept
src/core/SnapshotCoordinator.h:138:    void resetFadeStateAndRetireTarget() noexcept;
src/core/SnapshotCoordinator.h:194-199: (friend 宣言。no production caller の注記)
src/tests/D8_2_B_2_Tests.cpp:207-211 (friend 経由の呼出し、T10)
src/tests/AudioEngineHarness/V11EngineRetireProtectionTests.cpp:50-52 (friend 経由、T10)
```

| 呼出元 | 種別 | スレッド | 本番か | 備考 |
|---|---|---|---|---|
| なし | — | — | — | 本番呼出元は存在しない |
| `D8_2_B_2_Tests` T10 | テスト | テスト main | テスト | 直接呼出し |
| `V11EngineRetireProtectionTests` T10 | テスト | テスト main | テスト | 直接呼出し |
| `cpath_harness`（リポジトリ外） | ハーネス | 単一スレッド | 検証 | 要求・完了の 1 対 1 を確認済み |

`updateFade`（旧 RT 呼出し元）は 2026-07-28 に削除済みで、現在のソースに存在しない。

## §3 所有権の流れ（変更なし）

```
C2 開始（Non-RT）
  target = exchangeTarget(nullptr)          … 所有権はローカル変数へ
  ├ null → 何もしない
  └ 非 null
      ├ enqueueRetire 成功 → D が所有
      └ 失敗 → Q 受領 → （満杯）→ E 受領 → （満杯）→ T 受領（常に成功）
移送が確定するまで呼出元が所有権を保持する。
```

`exchangeTarget` の戻り値が同じポインタを 2 回返さないため、二重移送は構造的に起きない。

## §4 要求記録

| 項目 | 決定 |
|---|---|
| 位置 | 移送を実行する場所（C2 内）の 1 か所 |
| 回数 | **移送 1 回につき 1 回**（q→E→T へ段が進んでも再記録しない） |
| API | `recordRequest`（Non-RT 専用のため RT 版を使わない） |
| 取消し | 行わない（Q/E/T が受領するので完了が必ず来る） |
| 記録失敗 | `recordMisses` が増える（記録に失敗した**呼び出し回数**） |

`recordRequestRT` を使わない理由は、RT 経路が存在しなくなったためである。TLS 区画走査より全スロット走査の方が取りこぼしが少ない。

## §5 T の確保失敗（既存契約・Owner 判断の対象）

`TerminalReclaimAuthority::store()` は `noexcept` で `std::vector::push_back` を呼ぶ（`ISRRetireRouter.cpp:35`）。確保失敗時は `std::bad_alloc` が `noexcept` 境界を越えて `std::terminate()` になる。try/catch は無い。

| 利用者 | 同じ経路を使う |
|---|---|
| `switchImmediate` / `startFade` / `completeFade` / `retireCurrentAndTarget` | はい |
| `ISRRetireRouter::enqueueWithRetry` | はい |
| **C2（本設計）** | はい |

C2 で新しい失敗モードは増えない。契約は「確保に失敗したらプロセスを終了する（ポインタを失うリークにしない）」。**この挙動を許容するかは Owner 判断の対象として残す。** 許容しない場合の代替は、T を固定長にして満杯時を「プロセス終了」以外の扱いにする別設計になる（前案のリングは不採用済みなので、別の枠組みが必要）。

## §6 検証計画（実装時）

| # | ケース | 期待 |
|---|---|---|
| 1 | 成功経路 | D が受領。要求 1・完了 1・`unmatched 0`・`cancelled 0` |
| 2 | Q への移送 | `enqueueRetire` を false 固定 → Q 受領。要求 1・完了 1（Q drain 後）・`unmatched 0` |
| 3 | Q 満杯 → E | Q を 512 件埋める（noop deleter、epoch は drain されない値）→ C2 → E 受領。`quarantineResidentCount` と `emergencyQuarantineResidentCount` の変化で確認 |
| 4 | Q/E 満杯 → T | 両方 512 件埋める → C2 → T 受領。`terminalReclaimResidentCount` が 1 増える |
| 5 | 二重移送なし | 移送後に再度 C2 を呼ぶ → target が null のため何も起きない。residency が増えない |
| 6 | 記録は 1 回だけ | ケース 1〜4 のそれぞれで要求が +1（+2 にならない） |
| 7 | **RT 違反** | 別スレッドで `convo::numeric_policy::ScopedThreadRole(ThreadRole::AudioRealtime)` を構築し、そのスレッドから C2 を呼ぶ → target がスロットに残る（`getCurrentSnapshot` ではなく target の存在をテスト用アクセスで確認）。`m_rtViolationCount` が 1 増える。要求・完了・移送は 0 |
| 8 | **RT 違反後の正常退役** | ケース 7 の直後に Non-RT から C2 を呼ぶ → target が退役し、要求 1・完了 1・`unmatched 0` |
| 9 | 回帰 | 既存 60 テスト |

ケース 7・8 が Owner 指摘の「RT 違反時に target が保持され、後続 Non-RT 処理で正常に退役できること」の検証である。`ScopedThreadRole` は `DspNumericPolicy.h` の既存 API で、現在スレッドを Audio として登録する（`AudioEngine.Processing.AudioBlock.cpp:109` と同じ使い方）。

## §7 既存テストへの影響

| テスト | 影響 |
|---|---|
| `D8_2_B_2_Tests` T10 | なし（Non-RT から直接呼ぶ） |
| `D8_2_B_2_Tests` T11/T12 | なし（C2 を呼ばない） |
| `V11EngineRetireProtectionTests` T10 | なし |
| `cpath_harness`（リポジトリ外） | C2 の成功経路のみ検証。RT 判定の include が入ると JUCE 依存になるため、C2 の検証は D8 テストへ移す |

## §8 実装対象（承認後）

| ファイル | 変更 |
|---|---|
| `src/core/SnapshotCoordinator.cpp` | C2 に RT 判定・早期 return・違反カウンタ・失敗時の `quarantineRetireSink`・記録 1 回 |
| `src/core/SnapshotCoordinator.h` | `m_rtViolationCount` の追加、契約コメント更新、`:150` の RT 前提記述の削除 |
| `src/core/SnapshotCoordinator.cpp` | `#include "../DspNumericPolicy.h"`（`.cpp` 側に限定してヘッダ依存を増やさない） |
| `src/tests/D8_2_B_2_Tests.cpp` | ケース 1〜8 |

`DspNumericPolicy.h` は `JuceHeader.h` を include する。`SnapshotCoordinator.h` ではなく `.cpp` に入れることで、ヘッダ経由の波及を避ける。

## §9 未確定事項（Owner 判断）

1. T の `std::terminate` を既存契約として許容するか（§5）。
2. RT 違反カウンタの出力先（既存 health 出力のどこに 1 行足すか）。
3. RT 違反時にフェード状態も変更しない方針（§1.3）でよいか。変更して「フェードだけ解除」する案も成立する。
