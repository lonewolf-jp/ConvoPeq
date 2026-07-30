# BUG-014: juce::String currentDeviceTypeName_ のデータ競合（CoW 非スレッドセーフ）

- **発見日**: 2026-07-26
- **カテゴリ**: データ競合 / 未定義動作
- **関連**: AudioEngine.h, AudioEngine.Mmcss.cpp
- **リスク**: HIGH
- **修正**: 未

## 概要

`AudioEngine::currentDeviceTypeName_` は `juce::String` 型であり、
Message Thread からのみ書き込み（`setAudioDeviceTypeName`）、
Audio Thread から読み取り（`getCurrentMmcssPolicy`）される設計になっている。

しかし `juce::String` は **コピーオンライト（CoW）** を使用しており、
同時読み書きに対してスレッドセーフではない。
コメントでは「セッション中は不変」と主張しているが、
初回設定タイミングとオーディオコールバック開始タイミングの競合により
データ競合が発生する可能性がある。

## 該当箇所

### 1. メンバ定義

**`AudioEngine.h:2278-2279`**

```cpp
// デバイス種類名キャッシュ（Message Thread からのみ書き込み、Audio Thread から読み取り）
juce::String currentDeviceTypeName_;
```

### 2. 書き込み（Message Thread）

**`AudioEngine.h:2271`**

```cpp
void setAudioDeviceTypeName(const juce::String& type) noexcept { currentDeviceTypeName_ = type; }
```

`juce::String` の代入演算子は以下の操作を行う（非アトミック）：
1. 現在の文字列の参照カウントを読み取り
2. 参照カウント > 1 の場合、新しいメモリを確保してコピー
3. 新しい文字列データを書き込み

### 3. 読み取り（Audio Thread）

**`AudioEngine.Mmcss.cpp:54-64`**

```cpp
[[nodiscard]] AudioEngine::MmcssPolicy AudioEngine::getCurrentMmcssPolicy() const noexcept
{
    const auto& type = currentDeviceTypeName_;  // ← CoW トリガー
    if (type.containsIgnoreCase("WASAPI") || type.containsIgnoreCase("Windows Audio"))
        return MmcssPolicy::JuceManaged;
    if (type.containsIgnoreCase("ASIO"))
        return MmcssPolicy::SelfManagedProAudio;
    if (type.containsIgnoreCase("DirectSound"))
        return MmcssPolicy::SelfManagedPlayback;
    return MmcssPolicy::None;
}
```

`const auto& type = currentDeviceTypeName_;` は内部バッファへの参照を取得する。
この参照が保持されている間に Message Thread が `currentDeviceTypeName_ = type;`
を実行すると、CoW により内部バッファが再割り当て・解放され、
Audio Thread は **dangling reference** にアクセスする。

## 問題の詳細

### タイミングシーケンス

```
Message Thread                    Audio Thread
1. setAudioDeviceTypeName("ASIO")
   currentDeviceTypeName_ = "ASIO"
   ↓ (CoW: refcount > 1 → allocate new buffer, copy)
2. (new buffer allocated)
                                    3. const auto& type = currentDeviceTypeName_
                                       (holds reference to OLD buffer)
4. (old buffer freed)
                                    5. type.containsIgnoreCase("ASIO")
                                       → USE-AFTER-FREE!
```

### コメントの矛盾

```cpp
// AudioEngine.h:2268-2269
// ★ [work70 v9.11] 現在のオーディオデバイスの種類名（例: "WASAPI", "ASIO", "DirectSound"）。
//    setAudioDeviceTypeName() 経由で Message Thread からのみ書き込む（通常セッション開始時に1度だけ）。
```

「通常セッション開始時に1度だけ」という主張は：
- セッション開始時に Audio Thread が既にコールバックを受信している可能性
- デバイス切替時に `setAudioDeviceTypeName` が呼ばれる可能性
- コメントが「通常」と言っているが、例外的なタイミングでの呼び出しを防げない

## 影響

- **USE-AFTER-FREE**: Audio Thread が解放済みバッファにアクセス
- **クラッシュ**: オーディオコールバック中のクラッシュ → オーディオデバイスのロック
- **メモリ破壊**: 解放後のメモリが再利用された場合、不正な文字列比較結果

## 修正案

### オプション 1: atomic<const char*> + 手動メモリ管理

```cpp
std::atomic<const char*> currentDeviceTypeName_{nullptr};

void setAudioDeviceTypeName(const juce::String& type) noexcept {
    const char* old = currentDeviceTypeName_.load(std::memory_order_acquire);
    const char* newStr = strdup(type.toRawUTF8());
    currentDeviceTypeName_.store(newStr, std::memory_order_release);
    if (old) free(const_cast<char*>(old));
}

const char* getAudioDeviceTypeNameRaw() const noexcept {
    return currentDeviceTypeName_.load(std::memory_order_acquire);
}
```

### オプション 2: セッション開始前に設定完了を保証

Audio Thread コールバック開始前に `setAudioDeviceTypeName` が完了していることを
保証するフェンスを追加する。

### オプション 3: juce::String ではなく std::string_view + atomic

C++20 の `std::atomic<std::string_view>` を使用（ただし実装依存）。

## 推奨

**オプション 1** — `atomic<const char*>` は最も安全で移植性が高い。
juce::String の CoW は設計上スレッドセーフではないため、
このような RT/NonRT 間の共有データには使用すべきではない。
