# BUG-037: loaderTrashBin 内スレッドが ConvolverProcessor 破棄後に dangling reference を保持

## 発見日
2026-07-26

## ファイル
`src/ConvolverProcessor.LoadPipeline.cpp:51-55, 551-579`

## 問題
`loadImpulseResponse()` が既存の LoaderThread を `loaderTrashBin` に移してから
新しいローダースレッドを開始する:

```cpp
if (activeLoader != nullptr)
{
    activeLoader->signalThreadShouldExit();    // 終了シグナル
    loaderTrashBin.push_back(std::move(activeLoader));  // ゴミ箱行き
}
```

ゴミ箱内のスレッドは引き続き `owner`（ConvolverProcessor への参照）を保持したまま
バックグラウンドで動作しうる。

`cleanup()`（line 551-579）は終了したスレッドのみ削除する:
```cpp
for (auto it = loaderTrashBin.begin(); it != loaderTrashBin.end(); )
{
    if ((*it)->waitForThreadToExit(0))   // 即時ノンブロッキング確認
        it = loaderTrashBin.erase(it);   // 終了済みのみ削除
    else
        ++it;  // 動作中のスレッドは残す
}
```

### Use-After-Free シナリオ
1. ConvolverProcessor が破棄される
2. `loaderTrashBin` のデストラクタが Thread オブジェクトを `delete` する
3. Thread のデストラクタが `stopThread(-1)` を呼び、スレッドの終了を待つ（ブロッキング）
4. スレッドが持つ `ConvolverProcessor& owner` は既に dangling
5. スレッドが owner のメンバにアクセス → Use-After-Free

`WeakReference`（LoaderThreadInline.h:77）は存在するが、スレッドの直接実行パスでは
使用されておらず、`callAsync` パスでのみ利用される。

またスレッドが無限ループや長時間 I/O でスタックした場合、`stopThread(-1)` は
永久にブロックする。

### リスク評価
- **重大度**: HIGH — Use-After-Free、プロセッサ破棄時のクラッシュ
- **発生頻度**: 低（スレッド終了とプロセッサ破棄のタイミングに依存）
- **影響範囲**: プラグインアンロード/ホスト終了時の不安定性

### 修正方針
以下のいずれか:
1. Thread に `ownerAlive` フラグを追加し、ConvolverProcessor 破棄時に設定する
2. `WeakReference` をスレッドの全実行パスでチェックする
3. プロセッサ破棄時に全ローダースレッドを `stopThread(1000)` で強制終了する
   （タイムアウト付き）
