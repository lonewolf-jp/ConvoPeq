ConvoPeq の最新ソースコード（2026-06-11版）を確認したうえで評価すると、v7.2 はこれまでの v5～v7.1 の中では最も Practical Stable ISR Bridge Runtime に近い構成です。特に P1-A と P3-B は v7.1 より改善されています。

ただし、**まだ 2 点だけ設計上の懸念があります。**

---

# 総評

| 項目                         | 評価 |
| -------------------------- | -- |
| P0-A IEpochProvider統一      | ◎  |
| P0-B Fallback削除            | ◎  |
| P1-A RCUReader Fail-Closed | ○  |
| P1-B Admission閉ループ         | ◎  |
| P1-C Crossfade監視           | ○  |
| P2群                        | ◎  |
| P3-A RetirePressure統合      | ○  |
| P3-B Diagnostic Audit      | ◎  |

全体評価:

**98.5〜99/100**

---

# P1-A の評価

これは v7.1 より良いです。

現行 RCUReader は

```cpp
enter()
↓
acquireThreadSlot()
↓
registerReaderThread()
```

で slot を取得しています。

また EpochDomain 側でも

```cpp
registerReaderThread()
```

が唯一の Authority です。

そのため

```cpp
activeReaderCount() >= kMaxReaders
```

による事前拒否は確かに TOCTOU です。

v7 の

```cpp
rootEnterSucceeded_
```

のみへ戻した判断は妥当です。

---

# ただし P1-A に 1 点だけ注意

提案されている

```cpp
exit()
{
    rootEnterSucceeded_ = false;

    fetchSub(depth);
    ...
}
```

は、

```cpp
enter()
enter()
exit()
```

というネストケースで意味論が少し崩れます。

例

```cpp
enter(); // root=true

enter(); // nested

exit();  // nested exit
```

ここで

```cpp
rootEnterSucceeded_=false
```

になるため、

まだ outer scope が生存しているのに

```cpp
rootEnterSucceeded()==false
```

になります。

---

## 改善案

最外層のみリセット

```cpp
const uint32_t previousDepth = fetchSub(...);

if (previousDepth == 0)
{
    rootEnterSucceeded_ = false;
    ...
}

if (previousDepth > 1)
{
    return;
}

rootEnterSucceeded_ = false;
```

の方が RCU セマンティクスに一致します。

現状案でも致命傷ではありませんが、

厳密には

> 「reader がまだ active なのに false」

が発生します。

---

# P1-C の評価

これは v6 より明確に良いです。

即 Critical は強すぎました。

ConvoPeq のクロスフェードは

10～80ms 程度です。

そのため

```text
1回 drop
↓
即 Publication 停止
```

は過剰反応です。

---

## delta監視は正しい

v7.1 の

```cpp
delta = current - lastObservedDropCount
```

は良い改善です。

累積監視だと

```text
半年で10回
↓
Critical
```

になるためです。

---

## ただし閾値10は根拠が弱い

現在の案

```cpp
Critical = delta >= 10
```

は少し恣意的です。

本当に見たいのは

```text
単位時間あたりの drop rate
```

です。

例えば

```text
1 tickで10回
```

と

```text
10分で10回
```

は意味が違います。

Practical Stable 的には

```cpp
dropsPerMinute
```

または

```cpp
consecutiveTicksWithDrop
```

の方が監視品質は高いです。

ただし現段階では許容範囲です。

---

# P3-B の評価

v7.2 が正しいです。

v7.1 の

```cpp
load
if
fetchSub
```

には TOCTOU がありました。

例えば

```text
count=1

ThreadA load -> 1
ThreadB load -> 1

A fetchSub -> 0
B fetchSub -> UINT64_MAX
```

が起きます。

v7.2 の

```cpp
prev = fetchSub()

if(prev==0)
```

はこの問題を解消しています。

これは正しい修正です。

---

# RuntimeDrainAudit との整合性

ここも問題ありません。

現行コードベースでは

RuntimeDrainAudit が shutdown 監査側であり、ISRRetireRouter は依然として EpochDomain 固有 API を dynamic_cast に依存しています。

したがって

```text
WorldLifecycleAudit
↓
Diagnostic

RuntimeDrainAudit
↓
Authority
```

という v7.2 の整理は妥当です。

---

# 最終判定

v7.2 はこれまで提示された版の中で最も妥当です。

ただし **完全確定前に修正を推奨する点が1つだけあります。**

### 修正推奨

P1-A

```cpp
exit() 先頭で rootEnterSucceeded_ = false;
```

↓

```cpp
最外層 exit 時のみ false にする
```

理由:

```text
enter
 enter
 exit
```

のネストケースで

まだ reader active なのに

```cpp
rootEnterSucceeded()==false
```

となるため。

---

この1点を修正した場合、Practical Stable ISR Bridge Runtime 観点では **99.5/100 以上、実装着手可能レベル** と評価できます。




現時点で未確定事項、要調査事項、棚卸し事項、保留事項については、可能な限り詳細に調査を行い、確定させ、設計に反映してください。
なお、ソースコードの調査・検証・検索・棚卸し・解析には、原則として以下のツールをすべて使用すること。
・grep/Select-String
・oraios/serena mcp server（使用法を https://oraios.github.io/serena/01-about/000_intro.html で学習しておくこと。）
・codegraph mcp server(使用前にインデックス整備すること。tasks.jsonから起動すること。使用法を https://github.com/nahisaho/CodeGraphMCPServer で学習しておくこと。）
・cocoindex code(CLIコマンド。使用法を https://github.com/cocoindex-io/cocoindex-code で学習しておくこと。）
・graphify mcp server（解析AIにはgeminiではなくDeepseekを使用すること。使用法を https://github.com/safishamsi/graphify で学習しておくこと。）
・semble(CLIコマンド。使用法を https://github.com/MinishLab/semble で学習しておくこと。）