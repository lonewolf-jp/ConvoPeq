# D141 — Intermittent AV Root-Cause Diagnostic Gate (read-only; GO-A)

**Date:** 2026-08-31 (+09:00)
**Type:** diagnostic micro-gate. **Production source changes: 0. Test source changes: 0.**（ビルド設定は cache 変数のみ一時変更し、監査後に原状復元済み）
**基準:** `ConvoPeq.md Generated: 2026-08-31 09:28:16`（P3 完了版）+ `git status` = P3 時と同一 5 ファイル（余計な変更なし）。
**Verdict: GO-A（原因特定完了）**

---

## 結論（先に）

断続 `0xC0000005` の原因は **既存テスト 4 本が `int world = 1;` のアドレスを `commit()` の `void const* newWorld` payload として渡している**ことによる **スタックバッファオーバーフロー**です。

- `RuntimeIntentCoordinator::commit` は payload を `RuntimeState*` にキャストし（cpp:116）、`publication` の double フィールドを **+0x198〜+0x1B0** に bake する（約 440 バイト先の書込）。
- 渡される先は **4 バイトの `int`**（テストのスタックローカル）→ 範囲外書込。
- クラッシュが断続的なのは、`int world` のフレーム位置からスタックガードページまでの距離がビルド/ASLR のフレームレイアウトで変わるため。レイアウトが有利な実行では他スタックメモリを黙って壊すだけで済む。
- **delivery race（D140）とは無関係**（別関数・別フィールド・別機構）。**H1/H2/H3 棄却、H4 確認（ただし production ではなく test 側の既存 UB）**。

## D141-1 診断ビルド（cache のみ）

- `CMAKE_EXE_LINKER_FLAGS_RELEASE` に `/DEBUG /MAP` を一時追加（コンパイラフラグ不変）。`build/Release/ISRSemanticValidationTests.pdb`（115MB）+ `.map`（37MB）生成。
- 注意: 再設定で objects が再コンパイルされ D140 バイナリとレイアウトが変化した（フォールト RVA が 0x17xxx→0x32xxx に移動）。そのため D140 の形状オフセットは使わず、**新ビルドで最初からやり直し**、dump 起点で確定させた。
- 監査終了後、cache を `/INCREMENTAL:NO` に復元し test ターゲットを再リンク（原状回復確認済み）。

## D141-2 統計再現

- 200 反復 → **13 回失敗（6.5%）**、全件 `exit=-1073741819 (0xC0000005)`。iter 156/157・186/188/189 の連続失敗も観測（レイアウト依存の偏り）。

## D141-3 dump 取得と解析

- HKCU LocalDumps は指定フォルダに書かず、既定の `%LOCALAPPDATA%\CrashDumps` に 10 個の .dmp（各 1.35MB、exception/thread/module ストリーム付き）が生成された。
- minidump をストリーム直解析（type 6/3/4）:

| dump | exaddr RVA | op | target | R9 | R9+0x1A0 |
|---|---|---|---|---|---|
| .10608 | 0x323a5 | **1=write** | 0xde23760038 | 0xde2375fe98相当 | =target ✓ |
| .1644 | 0x323ba | 1=write | 0xd2551b0000 | 0xd2551afe50 | 0xd2551b0000 ✓ |
| .23180 | 0x323ac | 1=write | 0xc861300000 | 0xc8612ffe60 | 0xc861300000 ✓ |

- **全件で `target = R9 + 0x1A0`**、かつ target がページ境界（スタック base 上限 0xc861300000 等）に一致。R9 はいずれもスタック領域上限の 0x1A0 手前。
- 教訓として明記: **r9 非 NULL ≠ 有効オブジェクト**（本件では「有効だが payload 型が int で、+0x1A0 がオブジェクト境界外」）。

## D141-4 faulting function のソース同定

- map の Publics by Value で RVA 0x322F0（=offset 0001:000312F0）を引くと **`?commit@RuntimeIntentCoordinator@isr@convo@@QEAAXW4PublishAuthority@23@W4RuntimeBoundary@23@PEBX_K333PEBURuntimeState@@@Z`** = `RuntimeIntentCoordinator::commit(PublishAuthority, RuntimeBoundary, const void*, u64, u64, u64, u64, const RuntimeState*)`。フォールトは commit+0xB5〜0xC3 の `mov [r9+1A0h], r10` 群。
- 逆アセンブル形状（[rax+0x198..0x1A8] 読取→NaN ガード比較→[r9+0x198..0x1B0] 書込、[rdx+0x64]=1/[rdx+0x65]=2|6 の状態バイト）は commit の bake 処理と一致。
- 呼び出し元（テスト側）の確定: `grep commit(` で payload に `&world` を渡す箇所を全数列挙 → **4 箇所すべて `int world = 1;`**:
  - `testCoordinatorDrainAndShutdownContract`（test:315-322）
  - `testShutdownCompleteFailsWhenNotDrained`（test:~352-358）
  - `testPressureStateNormalizationContract`（test:~383-389）
  - `testShutdownCompleteFailsWhenSwapPending`（test:~422-428）
  - いずれも**元々存在したテスト**（P2/P3 より前の main() 前半）→ bisect2（P2/P3 スキップ）で再現した事実と完全整合。

### lifetime chain（P4-7 相当）
```
allocation: int world（テスト関数のスタックフレーム、4 バイト）
init: =1
publication: &world を void* として commit に渡す（型消去）
reader/mutation: commit が RuntimeState* にキャストし +0x198..0x1B0 を bake → 境界外書込
retire/destruction: 該当なし（lifetime 問題ではなく SIZE/TYPE 誤用）
```
→ reader access と destruction の HB 問題ではなく、**payload の型/サイズ契約違反**。

## D141-6 H1-H4 再判定

| 仮説 | 判定 | 根拠 |
|---|---|---|
| H1 delivery race → 破損 → AV | **棄却** | フォールトは commit の payload bake 書込（delivery とは無関係なコード・フィールド）。原因は `int*` payload の直接の境界外書込で、race を介在しない |
| H2 recovery race → stale lifetime → AV | **棄却** | 同上。recovery 系テストスキップでも再現（当該 4 テストは recovery 非依存） |
| H3 DSP/object lifetime bug → 破損 | **棄却** | ヒープ lifetime でなくスタックローカルへの型誤用書込 |
| H4 既存 UB（P2/P3 独立） | **確認** | 原因は CW-3b 期からの既存テスト 4 本の payload 誤用。P2/P3 と独立に再現・独立に存在 |

「P3 と無関係に再現」だけから H4 を確定したのではなく、**フォールト関数・payload 実体・呼び出し元を特定したうえで** H4（test 側既存 UB）と確定した。

## 修復対象（次 Gate の入力 — 本 Gate では一切修正していない）

- **test 側**: 4 箇所の `int world = 1; ... &world` を、実 `RuntimeState`（`createForTest().get()`）または意図が「payload 不要」なら commit が書き込まない経路（authority/boundary の分岐）へ修正。**テストのみ・4 行規模**。
- **production 側の検討（任意・契約）**：commit の payload 型消去（`void const*`→`RuntimeState*` static_cast）は誤用を許す設計。型安全化（`RuntimeState*` 直受け）か jassert による sealed 検証の追加は P4 契約議論と分離して判断。
- **delivery race（D140）との関係**: ゼロ。P4 の修復契約（i/iii/ii）判断はこの AV 診断結果に依存しなくてよいことが確定した（D140 §8 の前提が解消）。

## 成果物チェックリスト（D141-10 相当）
1. faulting instruction: `mov [r9+1A0h], r10`（commit+0xBC 近傍、3 dump 一致）✅
2. function: `RuntimeIntentCoordinator::commit` ✅
3. source location: cpp:116 のキャスト + bake ✅
4. pointer origin: テストの `int world` アドレス ✅
5. object lifetime: スタックローカル int（型/サイズ契約違反）✅
6. H1-H4 再判定: H4 確認・他棄却 ✅
7. 再現統計: 13/200・全 write・target=R9+0x1A0 ✅
8. ビルド設定原状復元 ✅（cache=/INCREMENTAL:NO、再リンク済み）

## 残存後片付けの記録
- `%LOCALAPPDATA%\CrashDumps\ISRSemanticValidationTests.exe.*.dmp`（10 個）はリポジトリ外に残置（必要なら削除可）。
- `tools/symtool.cs` のみ bash 削除がフックに拒否されたため残置（非追跡の診断用 C#、プロジェクトソースではない）。それ以外の一時スクリプト（d141_*.bat/ps1、symtool.exe 等）は除去済み。HKCU LocalDumps キーは削除済み。

**STOP — 修正 0。GO-A として原因を報告し、修復（test 4 箇所）は次 Gate の指示を待つ。P4 実装契約はこの結果に依存しないことが確定。**
