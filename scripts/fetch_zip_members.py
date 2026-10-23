#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按 Range 请求从远程 ZIP 中按需取文件，不下载整个压缩包。

场景：Kaggle 数据集是**单个 28 GB zip**，但默认代理带宽只有 ~0.5 MB/s，
全量下载要十几小时。ZIP 支持随机访问，只要读末尾的中心目录（EOCD），
就能定位任意成员并只取那一段字节。

用法：
    # 列出成员（名称 + 压缩/原始大小）
    python scripts/fetch_zip_members.py list --url <URL> [--proxy http://127.0.0.1:7897]

    # 提取指定成员（支持通配）
    python scripts/fetch_zip_members.py extract --url <URL> --match "sample_preview.jsonl" -o out/

    # 提取前 N 个匹配项（按原始大小升序）
    python scripts/fetch_zip_members.py extract --url <URL> --match "*.jsonl.zst" --limit 20 -o out/

支持 Zip64（>4 GB 的单文件/总大小），依赖标准库 + urllib。
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import struct
import sys
import urllib.request
import zlib

EOCD_SIG = b"PK\x05\x06"
EOCD64_SIG = b"PK\x06\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
CDH_SIG = b"PK\x01\x02"


def _opener(proxy: str | None) -> urllib.request.OpenerDirector:
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def _range(op, url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with op.open(req, timeout=120) as resp:
        return resp.read()


def _size(op, url: str) -> int:
    """取远端大小。Kaggle 的下载端点不接受 HEAD，退回 Range: bytes=0-0 + Content-Range。"""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with op.open(req, timeout=60) as resp:
            return int(resp.headers["Content-Length"])
    except Exception:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with op.open(req, timeout=60) as resp:
            cr = resp.headers.get("Content-Range") or ""
            if "/" in cr:
                return int(cr.rsplit("/", 1)[1])
            raise RuntimeError(f"无法确定远端大小: headers={dict(resp.headers)}")


class RemoteZip:
    def __init__(self, url: str, proxy: str | None = None):
        self.url = url
        self.op = _opener(proxy)
        self.size = _size(self.op, url)
        self.entries: list[dict] = []
        self._read_central_directory()

    def _read_central_directory(self) -> None:
        tail_len = min(1 << 20, self.size)
        tail_start = self.size - tail_len
        tail = _range(self.op, self.url, tail_start, self.size - 1)

        pos = tail.rfind(EOCD_SIG)
        if pos < 0:
            raise RuntimeError("未找到 EOCD，可能不是 ZIP 或需要更大的尾块")
        cd_size, cd_off = struct.unpack_from("<II", tail, pos + 12)
        total = struct.unpack_from("<H", tail, pos + 10)[0]

        # Zip64：EOCD 里的值会被置为 0xFFFFFFFF，真实值在 Zip64 EOCD 中
        loc = tail.rfind(EOCD64_LOC_SIG, 0, pos)
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or total == 0xFFFF or loc >= 0:
            if loc >= 0:
                z64_off = struct.unpack_from("<Q", tail, loc + 8)[0]
                head = _range(self.op, self.url, z64_off, z64_off + 55)
                if head[:4] == EOCD64_SIG:
                    total = struct.unpack_from("<Q", head, 32)[0]
                    cd_size = struct.unpack_from("<Q", head, 40)[0]
                    cd_off = struct.unpack_from("<Q", head, 48)[0]

        cd = _range(self.op, self.url, cd_off, cd_off + cd_size - 1)
        p = 0
        while p + 46 <= len(cd) and cd[p:p + 4] == CDH_SIG:
            (method,) = struct.unpack_from("<H", cd, p + 10)
            csize, usize = struct.unpack_from("<II", cd, p + 20)
            nlen, elen, clen = struct.unpack_from("<HHH", cd, p + 28)
            (lho,) = struct.unpack_from("<I", cd, p + 42)
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
            # Zip64 扩展字段：把 0xFFFFFFFF 的字段逐个替换
            if 0xFFFFFFFF in (csize, usize, lho):
                q = 0
                while q + 4 <= len(extra):
                    tag, sz = struct.unpack_from("<HH", extra, q)
                    if tag == 0x0001:
                        data = extra[q + 4:q + 4 + sz]
                        vals = list(struct.unpack_from("<" + "Q" * (sz // 8), data, 0))
                        it = iter(vals)
                        if usize == 0xFFFFFFFF:
                            usize = next(it)
                        if csize == 0xFFFFFFFF:
                            csize = next(it)
                        if lho == 0xFFFFFFFF:
                            lho = next(it)
                        break
                    q += 4 + sz
            self.entries.append({"name": name, "method": method, "csize": csize,
                                 "usize": usize, "lho": lho})
            p += 46 + nlen + elen + clen

    def _data_offset(self, e: dict) -> int:
        head = _range(self.op, self.url, e["lho"], e["lho"] + 29)
        nlen, elen = struct.unpack_from("<HH", head, 26)
        return e["lho"] + 30 + nlen + elen

    def read(self, e: dict) -> bytes:
        off = self._data_offset(e)
        raw = _range(self.op, self.url, off, off + e["csize"] - 1)
        if e["method"] == 0:
            return raw
        if e["method"] == 8:
            return zlib.decompress(raw, -15)
        raise RuntimeError(f"不支持的压缩方式 {e['method']}（{e['name']}）")

    def select(self, pattern: str, limit: int | None) -> list[dict]:
        hits = [e for e in self.entries if fnmatch.fnmatch(e["name"], pattern)]
        hits.sort(key=lambda x: x["usize"])
        return hits[:limit] if limit else hits


def main() -> int:
    ap = argparse.ArgumentParser(description="按 Range 从远程 ZIP 取单个成员")
    ap.add_argument("action", choices=["list", "extract"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--proxy", default=os.environ.get("https_proxy"))
    ap.add_argument("--match", default="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("-o", "--out", default=".")
    args = ap.parse_args()

    z = RemoteZip(args.url, args.proxy)
    print(f"远端 {z.size / 1073741824:.2f} GB，{len(z.entries)} 个成员", file=sys.stderr)
    hits = z.select(args.match, args.limit)

    if args.action == "list":
        for e in hits:
            print(f"{e['usize']:>12}  {e['csize']:>12}  m={e['method']}  {e['name']}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    for e in hits:
        dst = os.path.join(args.out, os.path.basename(e["name"]))
        data = z.read(e)
        with open(dst, "wb") as fh:
            fh.write(data)
        print(f"取出 {e['name']}  {len(data)} B -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
