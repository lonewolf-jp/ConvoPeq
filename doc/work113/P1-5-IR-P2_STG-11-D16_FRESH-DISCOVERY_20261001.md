# STG-11-D16 Fresh Discovery — Parameter-to-Snapshot-to-Runtime Consistency Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D16_FRESH-DISCOVERY_20261001.md`
- Work item: **STG-11-D16** — parameter 伝播境界の一貫性監査
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20261001-054723).md` と同一。
  D11 マーカー 8 件・FRESH を作業開始時に確認。過去資料・記憶・旧行番号は未使用。
  D12〜D15 の結論の流用なし）
- Commit / push: **禁止**

---

## 0. 判定

```
NO-GO — no concrete defect
```

parameter writer inventory / snapshot completeness / dirty-flag /
snapshot atomicity / multiple-writer / dependency / publish correspondence /
cache invalidation / semantic parity を実コードで監査した結果、
D16 GO 条件 7 項目をすべて満たす concrete defect は存在しない。
Implementation / Gate は作成しない。

D1〜D11 の未 commit 変更・D12〜D16 監査成果物はすべて保持する。

---

## 1. Parameter writer inventory（§2）

主要 parameter の writer→storage→dirty→snapshot を追跡。setter は正しく、
かつ RuntimeBuilder が読む snapshot への反映を確認した：

- EQ（freq/gain/Q/enabled/type/channel/totalGain/agc/saturation/structure）：
  setter→`EQState` exchange→`updateBandNode`→snapshot→builder。欠落なし。
- Convolver（mix/phase/IR 長/tail/mixed/cache/FFT/nuc）：setter→pendingOverride→
  snapshot→builder。欠落なし。
- Engine（order/bypass/gain/dither/shaper/oversampling/analyzer/filter）：
  setter→atomic triple→snapshot→builder。欠落なし。

## 2. Writer → snapshot completeness（§3）

- D16-1（snapshot 漏れ）：なし。全 setter 変更は snapshot 経路に載る。
- D16-2（未読 field）：なし。`computeParamsHash` は band 全 field＋totalGain＋
  agc＋saturation＋structure＋sr＋block を包含（BUG-047 対応済み）。
- D16-3（古い field 読込）：なし。snapshot は単一 source から構築。
- D16-4（二重表現の片更新）：gain triple（dB／linear／m_current）は
  3 箇所の writer（setter／defaults／revalidate）すべてが triple 同時 publish。
  乖離なし。
- D16-5（default 上書き）：`applyDefaultsForCurrentMode` は restoring 中抑止
  （`m_isRestoringState`）＋AutoGain 時抑止。意図せぬ上書きなし。
- D16-6（dependency 欠落）：bypass／order 変更は `revalidateInputHeadroomForCurrentMode`
  を呼ぶ。SR 変更は oversampling／FFT／partition／latency の再導出経路あり。
  欠落なし。

## 3. Dirty / pending flag map（§4）

- `m_pendingIRChange`：writer 2 箇所（Timer／UIEvents、先に Structural intent 発行）、
  acknowledge は snapshot 側の単一 exchange。消失・二重 consume なし（v2 監査確定）。
- `m_pendingNSChange`／`m_pendingAGCChange`：同 snapshot で consume。持ち越しなし。
- `rebuildPendingAfterLoad`：exchange once。未実行再送あり。
- 「値変更済み・flag false」の concrete 経路なし。

## 4. Snapshot atomicity / mixed-state（§5）

- EQ の freq／gain／Q は setter ごとに state 全体を exchange するため、
  reader が観測するのは常に valid な中間状態（無効な組み合わせは生じない）。
  論理 update の多 field 原子性は設計上不要（UI は単一操作）。
- IR length＋IR data 等の組み合わせは generation／currency guard で保護（D14 確定）。

## 5. Multiple writer / duplicate representation（§6）

- authoritative storage は各 parameter ごとに単一（atomic triple は同時更新で一体）。
- UI／session／atomic／pending／runtime／DSP の各表現の writer／reader／同期点を
  確認。stale representation の concrete 経路なし
  （`A 更新→B 旧値→builder が B 読込` 型の欠陥なし）。

## 6. Parameter dependency map（§7）

- bypass／order→headroom ceiling 再検証あり。
- SR→oversampling／FFT／partition／IR hardMax／latency の再導出あり。
- `A 更新・B 旧値` 型の欠陥なし。cache invalidation 不足なし
  （hash が全依存を含むため stale-cache 出力なし）。

## 7. Publish correspondence（§8）

- snapshot→builder→validation→publish の identity は generation／fingerprint／
  currency guard で対応付け（D14 確定）。`S1→R1／S2→R2` の混在の証拠なし。

## 8. Semantic parity map（§10）

| 変換 | 契約 | verdict |
| --- | --- | --- |
| dB→linear | setter 内で同時 publish（有限入力のみ到達） | 一致 |
| sec→samples | 消費側で変換（`computeTargetIRLength` 等） | 一致 |
| enum→index | D7/D10 ガード済み＋等価 chain | 一致 |
| bool→feature | 直接対応 | 一致 |

明示の変換契約と異なる実装なし。

## 9. なぜ NO-GO か

D16 GO 条件 7 項目に照らし、「semantic information の喪失」
「stale／mixed／lost update」を満たす候補が存在しない。
parameter 伝播系は triple-write 規律・包括 hash・単一 source・文書化 flag 契約・
dependency 再検証により一貫している。

## 10. 推奨

- D6 NO-GO・R8 flake・R4（P3）・dead code 一括整理は引き続き未解決として残る。

## 11. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**
- D1〜D11 の未 commit 変更・D12〜D16 監査成果物はすべて保持
- Implementation / Gate は作成しない（NO-GO のため）
