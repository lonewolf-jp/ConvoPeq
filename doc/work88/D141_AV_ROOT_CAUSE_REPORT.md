# D141 — Intermittent AV Root-Cause Diagnostic Gate（Work Report）

**Status: GO-A（原因特定完了）。Production/Test source changes: 0。**
**詳細:** `evidence/D141_AV_ROOT_CAUSE_GO-A.md`

## 原因（確定）
断続 `0xC0000005` は **既存テスト 4 本の payload 型誤用によるスタックバッファオーバーフロー**:
```cpp
int world = 1;
coordinator.commit(Granted, NonRTWorld, &world, 1,1,1,1, nullptr);
```
`commit` は payload を `RuntimeState*` にキャスト（cpp:116）し `publication` の double を **+0x198〜+0x1B0** に bake → 4 バイトの int へ約 440 バイト先まで書込。該当テスト: `testCoordinatorDrainAndShutdownContract` / `testShutdownCompleteFailsWhenNotDrained` / `testPressureStateNormalizationContract` / `testShutdownCompleteFailsWhenSwapPending`（いずれも P2/P3 以前の既存テスト）。断続性はフレーム位置とガードページ距離のレイアウト依存で説明。

## 特定手順（D140 の未達点の解消）
1. cache のみ `/DEBUG /MAP` 化（コンパイラフラグ不変）→ PDB+map 生成。再設定で objects が再コンパイルされレイアウトが動いたため、新ビルドで再追跡。
2. 200 反復 → 13 回失敗（6.5%）、全 0xC0000005。
3. dump は既定 `%LOCALAPPDATA%\CrashDumps` に 10 個取得。minidump ストリーム直解析で **全件 `target = R9 + 0x1A0`（write）**、R9 はスタック上限直下、RIP RVA=0x323a5/323ba/0x323ac。
4. map の offset 0001:000312F0 → **`RuntimeIntentCoordinator::commit`**（fault=commit+0xBC の `mov [r9+1A0h],r10`）。
5. 呼び出し元 grep で `&world`（int）4 箇所を確定。lifetime chain は「スタック int への型/サイズ契約違反書込」で、heap/retire 問題でないことを確認。

## H1-H4 再判定
- H1（delivery race→AV）**棄却** — 別関数・別フィールド・race 非介在。
- H2（recovery race→stale lifetime）**棄却**。
- H3（DSP/object lifetime）**棄却**。
- H4（既存 UB、P2/P3 独立）**確認** — ただし **production ではなく test 側**。再現統計だけでなくフォールト関数・payload 実体・呼び出し元を特定したうえでの確定。

## 帰結
- **delivery race（D140 の DATA RACE/UB）は実運用クラッシュの原因ではないことが確定**。P4 の修復契約（i/iii/ii）の判断は AV 診断結果に依存しなくてよい（D140 §8 の前提が解消）。
- 修復対象は **テスト 4 箇所の payload 修正**（実 RuntimeState か安全な経路へ）— 次 Gate の指示待ち。本 Gate では修正しない。
- 副次推奨（任意・別判断）: commit の `void const*` 型消去の型安全化は P4 契約議論と分離して検討。

## 後片付け
- build cache を `/INCREMENTAL:NO` に復元 + test ターゲット再リンク済み（`git status` は P3 時と同一 5 ファイル、変更 0 を確認）。
- HKCU LocalDumps キー削除済み。一時スクリプト除去済み（`tools/symtool.cs` のみ bash 削除がフックに拒否され残置 — 非追跡の診断用）。dump 群はリポジトリ外（既定 CrashDumps）に残置。

**STOP — GO-A 報告まで。修復（test 4 箇所）・P4 実装・P5/P6 には進まない。**
