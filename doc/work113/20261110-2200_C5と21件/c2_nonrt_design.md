# C2 スナップショット退役リーク: Non-RT 専用 確定設計（実装は未承認）

Owner 判断（2026-11-10）を受け、RT 対応案（フラグ＋延期）を廃止し、**Non-RT 専用案**の確定設計を示す。実装は行っていない。

## §1 呼出元の棚卸し

`SnapshotCoordinator::resetFadeStateAndRetireTarget()` は private である。

| 呼出元 | 種別 | スレッド | 本番か |
|---|---|---|---|
| なし | — | — | **本番呼出元は存在しない**（`SnapshotCoordinator.h:194-199` に明記。`updateFade` は 2026-07-28 に削除済み） |
| `D8_2_B_2_Tests` T10 | テスト（friend） | テスト main | テスト |
| `V11EngineRetireProtectionTests` T10 | テスト（friend） | テスト main | テスト |
| `cpath_harness`（検証用・リポジトリ外） | ハーネス | 単一スレッド | 検証 |

この棚卸しから、**C2 を Non-RT 専用として契約を確定しても、既存の呼出元はすべて条件を満たす**。

## §2 スレッド契約

### 2.1 契約文

- C2 は **Non-RT からのみ**呼べる。
- RT からの呼出しは契約違反であり、Debug ビルドでは `jassert` で検出する。Release ビルドではアサーションを無効にせず、**計測カウンタで検出可能**にする（§2.3）。

### 2.2 既存の RT 判定基盤

`convo::numeric_policy::isAudioThread()` が既に存在し、`ISRRetireRouter::enqueueWithRetry` などで同じ目的に使われている（`ISRRetireRouter.cpp:555`）。新しい仕組みは追加しない。

```cpp
void SnapshotCoordinator::resetFadeStateAndRetireTarget() noexcept
{
    jassert (! convo::numeric_policy::isAudioThread());   // ★ 契約の明文化
    ...
}
```

`jassert` は JUCE の Debug 専用マクロだが、`SnapshotCoordinator` は既に JUCE に依存している（`SnapshotFactory` 経由）。依存は増えない。

### 2.3 Release での検出

`jassert` は Release で消えるため、`snapshotRetireMisuseCount`（atomic）を 1 増やす行を併置する。

```cpp
if (convo::numeric_policy::isAudioThread())
    convo::fetchAddAtomic (m_rtMisuseCount, std::uint64_t{1}, std::memory_order_release);
```

読み出しは既存の診断面（`ISRDebugRuntime` または `AudioEngine` の health 出力）に 1 行追加する。**新規の判定式は作らない。**

## §3 所有権の流れ

```
C2 開始
  target = exchangeTarget(nullptr)        … 所有権はローカル変数へ
  │
  ├ target == null → 何もしない（所有権移転なし）
  │
  └ target != null
      ├ enqueueRetire(target) == true     → D キューが所有（成功）
      └ enqueueRetire(target) == false
          ├ Q（RetireQuarantineStore）受領 → Q が所有
          ├ Q 満杯 → E（EmergencyQuarantineStore）受領 → E が所有
          └ E 満杯 → T（TerminalReclaimAuthority）受領 → T が所有（常に成功）
```

- **移送が確定するまで呼出元（`exchangeTarget` の戻り値）が所有権を保持する。** 失敗して所有権を手放せない状態は発生しない。
- `quarantineRetireSink` は既存実装をそのまま使う（`ISRRetireRouter` の公開 API 経由。新しい移送先は作らない）。
- 所有権の二重化は、`exchangeTarget` が同じポインタを 2 回返さないことで構造的に排除される。

## §4 要求記録（D1）

| 項目 | 決定 |
|---|---|
| 記録の位置 | 移送を実行する場所（C2 内）の 1 か所 |
| 記録の回数 | 1 回（`recordRequestRT` を使わず、Non-RT なので `recordRequest` を使う） |
| 取消し | **行わない**。失敗時は Q/E/T が受領し、完了フックが必ず対応する |
| 記録失敗時 | レコーダの `recordMisses` が増える（本日追加した検出手段）。退役自体は変更しない |

`recordRequestRT` を `recordRequest` に置き換える理由は、RT 専用 API を使う必要がなくなったためである。TLS 区画の走査より全スロット走査の方が取りこぼしが少ない。

## §5 T の確保失敗時の挙動（既存契約の確認）

Owner 指摘のとおり、`TerminalReclaimAuthority::store()` は `noexcept` であり、内部で `std::vector::push_back` を呼ぶ（`ISRRetireRouter.cpp:35`）。確保に失敗すると `std::bad_alloc` が `noexcept` 境界を越えて **`std::terminate()`** になる。try/catch は無い。

これは C2 固有の性質ではなく、**Q→E→T 連鎖を使う既存の全経路が同じ契約**である。

| 既存の利用者 | 同じ経路を使う |
|---|---|
| `SnapshotCoordinator::switchImmediate` | はい（Q→E→T） |
| `SnapshotCoordinator::startFade` / `completeFade` | はい（Q→E→T） |
| `SnapshotCoordinator::retireCurrentAndTarget`（本日 C5 を追加） | はい（Q→E→T） |
| `ISRRetireRouter::enqueueWithRetry` | はい（Q→E→T） |

したがって C2 の変更で**新しい失敗モードは増えない**。契約として「メモリ確保に失敗した場合はプロセスを終了する（ポインタを失うリークにはしない）」を明記する。この挙動を許容できない場合は、T の格納を固定長リングへ戻す別設計（＝前案）が必要になるが、Owner 判断で前案は不採用である。

> 補足: `std::terminate` は「所有権を失う」より安全側である。プロセスが終了すれば UAF は起きない。

## §6 既存テストとの整合

| テスト | 現在の期待 | 変更後の影響 |
|---|---|---|
| `D8_2_B_2_Tests` T10 | `resetFadeStateAndRetireTarget` を直接呼び、Q/E/T に移送されないことを確認（`enqueueRetire` 成功） | 影響なし（成功経路は変わらない） |
| `D8_2_B_2_Tests` T11（本日追加） | C3 の要求・完了 1 対 1、quarantine で重複記録なし | 影響なし（C2 を呼ばない） |
| `V11EngineRetireProtectionTests` T10 | C2 の呼出しで D のみ使用 | 影響なし |
| `cpath_harness`（リポジトリ外） | C2 の要求・完了 1 対 1 | 影響なし（成功経路） |
| 失敗経路のテスト | **存在しない** | 実装時に 1 件追加する（Q 満杯を強制し、Q→E→T の移送と所有権を確認） |

`jassert` を追加すると、RT から呼ぶテストがあれば Debug で停止する。現状 RT から呼ぶテストは無い（棚卸しのとおり）。

## §7 変更対象（実装時の差分の概形）

| ファイル | 変更 |
|---|---|
| `SnapshotCoordinator.cpp` | `resetFadeStateAndRetireTarget` に (a) RT 検出、(b) 失敗時の `quarantineRetireSink` 呼出し、(c) 記録 1 回（取消しなし） |
| `SnapshotCoordinator.h` | `m_rtMisuseCount` の追加、契約コメントの更新、RT 呼出し前提の記述（`:150`）の削除 |
| テスト 1 ファイル | queue 満杯を強制する provider で、Q 受領・要求 1 件・完了 1 件を確認 |

`quarantineRetireSink` は private のまま（C2 から呼ぶので公開不要）。

## §8 検証計画（実装時）

1. 成功経路: `enqueueRetire` 成功 → D が所有、要求 1 件・完了 1 件、`unmatched=0`。
2. 失敗経路: `enqueueRetire` を false に固定 → Q が受領、要求 1 件・完了 1 件（Q drain 後）、`unmatched=0`、`cancelled=0`。
3. Q 満杯: 512 件を埋めてから C2 → E が受領、所有権が移る。
4. Q/E 満杯: E も埋める → T が受領、`terminalReclaimResidentCount` が 1 増える。
5. 二重移送なし: 移送後に再度 C2 を呼んでも対象が無い（target は null）ことを確認。
6. RT 検出: テストで `isAudioThread` 相当の状態を作れないため、カウンタのインクリメントを直接検証する（`numeric_policy` のテスト用フックが無い場合は、契約コメントと `jassert` のみで運用し、検証項目から外す）。
7. 回帰: 既存 60 テスト（ctest のターゲット数は変わらない。T11/T12 は `D8_2_B_2_Tests` 内のケース）。

## §9 RT が必要になった場合

RT からの退役が必要になった時点で、次のいずれかを別設計として起議する。

| 案 | 内容 | 前提 |
|---|---|---|
| RT-1 | RT では target を exchange せず、`m_slots` に残したまま 1 bit フラグを立てる。Non-RT の drain が移送する | フラグと `startFade` の競合を、フラグを世代番号（`uint64_t`）にして解決する必要がある（前案の不備） |
| RT-2 | RT 専用の固定長スロットを target ごとに 1 個だけ持つ（多重度 1 を構造で保証） | target は 1 個しか存在しないため、単一スロットで足りる。満杯＝既に 1 個保留中で、その場合は新たな退役をしない（＝exchange しない） |
| RT-3 | C2 を RT から呼ばない設計を維持し、RT 側は別 API（例: `requestRetireOnNextNonRtTick`）にする | 呼出し側の設計変更が必要 |

現時点では RT 呼出しが無いため、§1〜§8 の Non-RT 専用案を確定設計とする。
