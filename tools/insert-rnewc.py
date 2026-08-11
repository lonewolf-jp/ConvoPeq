# -*- coding: utf-8 -*-
"""R-新規C を §2.1 表の R-新規D 行の直前に挿入する（範囲限定を解除）。"""
import sys

PATH = r'c:\VSC_Project\ConvoPeq\doc\work88\big_bug\INTEGRATED_BUG_FIX.md'


def load_lines():
    with open(PATH, 'rb') as f:
        data = f.read()
    return data.decode('utf-8').split('\n')


def save_lines(lines):
    out = '\n'.join(lines)
    with open(PATH, 'wb') as f:
        f.write(out.encode('utf-8'))


def main():
    lines = load_lines()
    anchor = '| R-新規D |'
    new_row = '| R-新規C | bad_alloc 時リーク（OOM） | LoaderThread.cpp:367 | P3 | runSynchronously を try/catch + retire 化（§3-34）|'

    idx = -1
    for i, ln in enumerate(lines):
        if ln.rstrip('\r').startswith(anchor):
            idx = i
            break
    if idx < 0:
        print('[FAIL] R-新規D not found')
        sys.exit(1)

    # 重複確認（§2.1 範囲内）
    dup = False
    for ln in lines:
        if ln.rstrip('\r').startswith('| R-新規C |'):
            dup = True
            break
    if dup:
        print('[SKIP] R-新規C already present')
    else:
        lines.insert(idx, new_row + '\r')
        print('[OK] inserted R-新規C before R-新規D')

    save_lines(lines)
    print('DONE: written')


if __name__ == '__main__':
    main()
