"""
游戏内资源压缩与解压系统 - 核心模块

功能：
- compress()/decompress(): 基于 zlib 的高效压缩，支持字典、多线程、
  全覆盖 HMAC 校验和、进度回调
- dictionary_encoding(): 基于频率统计的字典编码，字典持久化到磁盘
- checksum_verification(): 带密钥的 HMAC-SHA256，覆盖全部数据
- compress_stream()/decompress_stream(): 流式处理，内存可控，支持断点续传

仅使用 Python 标准库。
"""
import hashlib
import hmac
import io
import json
import os
import re
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

MAGIC = b"RCP1"
VERSION = 1
HEADER_CAPACITY = 4096          # 固定头部容量，便于流式写完后回填
DEFAULT_CHUNK_SIZE = 1 << 20    # 1 MiB，流式内存上限
DEFAULT_KEY = b"resource-compressor-default-key"
ESCAPE = 0xFF                   # 字典编码转义字节
MAX_DICT_WORDS = 4096           # 字典词条上限（2字节索引）


class CompressionError(Exception):
    """压缩/解压过程中的通用错误。"""


class ChecksumError(CompressionError):
    """校验和验证失败：数据损坏或被篡改。"""


class DictionaryError(CompressionError):
    """字典相关错误。"""


@dataclass
class Config:
    level: int = 6
    num_threads: int = 0          # 0 表示按 CPU 核数
    chunk_size: int = DEFAULT_CHUNK_SIZE
    checksum_key: bytes = DEFAULT_KEY
    dict_path: Optional[str] = None


ProgressCallback = Callable[[int, Optional[int]], None]


class ResourceCompressor:
    """游戏资源压缩器。

    config 支持的键：
        level:        zlib 压缩级别 1-9，默认 6
        num_threads:  压缩/解压线程数，默认 CPU 核数
        chunk_size:   流式分块大小（内存上限），默认 1 MiB
        checksum_key: HMAC 密钥（bytes），校验和防篡改
        dict_path:    字典持久化路径，设置后自动加载/保存
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.level = int(self.config.get("level", 6))
        cpu = os.cpu_count() or 4
        self.num_threads = max(1, int(self.config.get("num_threads") or cpu))
        self.chunk_size = max(1024, int(self.config.get("chunk_size", DEFAULT_CHUNK_SIZE)))
        key = self.config.get("checksum_key", DEFAULT_KEY)
        self.checksum_key = key if isinstance(key, bytes) else str(key).encode()
        self.dict_path = self.config.get("dict_path")
        self._state = {}
        self._history = []
        self._dict_words: List[bytes] = []
        self._dictionary = b""
        if self.dict_path and os.path.exists(self.dict_path):
            self.load_dictionary(self.dict_path)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------ #
    # 校验和：HMAC-SHA256，带密钥，覆盖全部数据
    # ------------------------------------------------------------------ #
    def _chunk_hmac(self, chunk: bytes) -> bytes:
        return hmac.new(self.checksum_key, chunk, hashlib.sha256).digest()

    def _checksum(self, chunk_hmacs: Iterable[bytes]) -> str:
        """对全部数据块的 HMAC 再做一次 HMAC，覆盖所有数据且可增量计算。"""
        final = hmac.new(self.checksum_key, digestmod=hashlib.sha256)
        for digest in chunk_hmacs:
            final.update(digest)
        return final.hexdigest()

    def checksum_verification(self, data: bytes, expected_hex: str) -> bool:
        """验证整段数据的校验和（带密钥、全覆盖）。"""
        actual = self._checksum([self._chunk_hmac(data)])
        return hmac.compare_digest(actual, expected_hex)

    # ------------------------------------------------------------------ #
    # 字典：构建、优化、持久化
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_tokens(sample: bytes) -> List[bytes]:
        return re.findall(rb"[A-Za-z0-9_./\-]{4,}", sample)

    def build_dictionary(self, samples: Iterable[bytes],
                         max_words: int = MAX_DICT_WORDS) -> List[bytes]:
        """按词频统计构建字典：出现次数越多的常用词越优先入典。"""
        freq: Dict[bytes, int] = {}
        for sample in samples:
            for token in self._extract_tokens(sample):
                freq[token] = freq.get(token, 0) + 1
        words = sorted(freq, key=lambda w: (-freq[w], w))[:max_words]
        self._dict_words = words
        self._dictionary = b"".join(words)
        if self.dict_path:
            self.save_dictionary(self.dict_path)
        return list(words)

    def save_dictionary(self, path: str):
        import base64
        payload = {
            "version": VERSION,
            "words": [base64.b64encode(w).decode("ascii") for w in self._dict_words],
        }
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp)

    def load_dictionary(self, path: str):
        import base64
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        self._dict_words = [base64.b64decode(w) for w in payload.get("words", [])]
        self._dictionary = b"".join(self._dict_words)

    @property
    def dictionary_words(self) -> List[bytes]:
        return list(self._dict_words)

    # ------------------------------------------------------------------ #
    # 字典编码：用 2 字节索引替换字典词条，实际生效
    # ------------------------------------------------------------------ #
    def dictionary_encoding(self, data: bytes) -> bytes:
        if not self._dict_words:
            raise DictionaryError("字典为空，请先 build_dictionary() 或加载字典")
        indexed = sorted(enumerate(self._dict_words),
                         key=lambda kv: -len(kv[1]))
        out = bytearray()
        pos = 0
        size = len(data)
        while pos < size:
            byte = data[pos]
            if byte == ESCAPE:
                out += b"\xff\xff"
                pos += 1
                continue
            hit = -1
            hit_len = 0
            for idx, word in indexed:
                if data.startswith(word, pos):
                    hit = idx
                    hit_len = len(word)
                    break
            if hit >= 0:
                out.append(ESCAPE)
                out += struct.pack(">H", hit)
                pos += hit_len
            else:
                out.append(byte)
                pos += 1
        return bytes(out)

    def dictionary_decoding(self, encoded: bytes) -> bytes:
        if not self._dict_words:
            raise DictionaryError("字典为空，请先 build_dictionary() 或加载字典")
        out = bytearray()
        pos = 0
        size = len(encoded)
        while pos < size:
            byte = encoded[pos]
            if byte == ESCAPE:
                if pos + 1 >= size:
                    raise DictionaryError("字典编码数据损坏：转义字节后缺少内容")
                marker = encoded[pos + 1]
                if marker == ESCAPE:
                    out.append(ESCAPE)
                    pos += 2
                else:
                    if pos + 3 > size:
                        raise DictionaryError("字典编码数据损坏：索引不完整")
                    (idx,) = struct.unpack(">H", encoded[pos + 1:pos + 3])
                    if idx >= len(self._dict_words):
                        raise DictionaryError("字典编码数据损坏：索引越界")
                    out += self._dict_words[idx]
                    pos += 3
            else:
                out.append(byte)
                pos += 1
        return bytes(out)

    # ------------------------------------------------------------------ #
    # 压缩 / 解压（zlib + 字典 + 多线程 + 校验和 + 进度）
    # ------------------------------------------------------------------ #
    def _compress_chunk(self, chunk: bytes) -> bytes:
        kwargs = {}
        if self._dictionary:
            kwargs["zdict"] = self._dictionary
        obj = zlib.compressobj(self.level, zlib.DEFLATED, zlib.MAX_WBITS,
                               8, zlib.Z_DEFAULT_STRATEGY, **kwargs)
        return obj.compress(chunk) + obj.flush()

    def _decompress_chunk(self, payload: bytes) -> bytes:
        kwargs = {}
        if self._dictionary:
            kwargs["zdict"] = self._dictionary
        obj = zlib.decompressobj(zlib.MAX_WBITS, **kwargs)
        try:
            return obj.decompress(payload) + obj.flush()
        except zlib.error as exc:
            raise ChecksumError(f"数据损坏：无法解压 ({exc})") from exc

    def _make_header(self, original_size: int, num_chunks: int,
                     checksum: str) -> bytes:
        header = {
            "version": VERSION,
            "algorithm": "zlib",
            "level": self.level,
            "chunk_size": self.chunk_size,
            "original_size": original_size,
            "num_chunks": num_chunks,
            "dict_size": len(self._dictionary),
            "checksum": checksum,
        }
        raw = json.dumps(header, sort_keys=True).encode("utf-8")
        if len(raw) > HEADER_CAPACITY:
            raise CompressionError("头部元数据超出容量")
        return raw.ljust(HEADER_CAPACITY, b"\x00")

    @staticmethod
    def _parse_header(raw: bytes) -> Dict:
        return json.loads(raw.rstrip(b"\x00").decode("utf-8"))

    def _write_header(self, writer, header: bytes):
        writer.write(MAGIC)
        writer.write(struct.pack(">I", HEADER_CAPACITY))
        writer.write(header)

    def _read_header(self, reader) -> Dict:
        magic = self._read_exact(reader, len(MAGIC))
        if magic != MAGIC:
            raise CompressionError("非法的压缩包：魔数不匹配")
        (capacity,) = struct.unpack(">I", self._read_exact(reader, 4))
        return self._parse_header(self._read_exact(reader, capacity))

    @staticmethod
    def _read_exact(reader, size: int) -> bytes:
        buf = bytearray()
        while len(buf) < size:
            piece = reader.read(size - len(buf))
            if not piece:
                raise CompressionError("数据不完整：流提前结束")
            buf += piece
        return bytes(buf)

    @staticmethod
    def _frame(payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + payload

    def compress(self, data: bytes,
                 progress_callback: Optional[ProgressCallback] = None) -> bytes:
        """压缩：zlib + 字典 + 多线程分块，头部带全覆盖 HMAC 校验和。"""
        chunks = [data[i:i + self.chunk_size]
                  for i in range(0, len(data), self.chunk_size)]
        compressed: List[Optional[bytes]] = [None] * len(chunks)
        total = len(data)
        done = 0
        if chunks:
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                futures = {pool.submit(self._compress_chunk, c): i
                           for i, c in enumerate(chunks)}
                for future in as_completed(futures):
                    idx = futures[future]
                    compressed[idx] = future.result()
                    done += len(chunks[idx])
                    if progress_callback:
                        progress_callback(done, total)
        checksum = self._checksum(self._chunk_hmac(c) for c in chunks)
        out = io.BytesIO()
        self._write_header(out, self._make_header(total, len(chunks), checksum))
        for payload in compressed:
            out.write(self._frame(payload))
        return out.getvalue()

    def decompress(self, blob: bytes,
                   progress_callback: Optional[ProgressCallback] = None,
                   verify: bool = True) -> bytes:
        """解压：多线程分块解压，校验和验证失败抛 ChecksumError。"""
        reader = io.BytesIO(blob)
        header = self._read_header(reader)
        num_chunks = header["num_chunks"]
        payloads = []
        for _ in range(num_chunks):
            (length,) = struct.unpack(">I", self._read_exact(reader, 4))
            payloads.append(self._read_exact(reader, length))
        chunks: List[Optional[bytes]] = [None] * num_chunks
        total = header["original_size"]
        done = 0
        if num_chunks:
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                futures = {pool.submit(self._decompress_chunk, p): i
                           for i, p in enumerate(payloads)}
                for future in as_completed(futures):
                    idx = futures[future]
                    chunk = future.result()
                    chunks[idx] = chunk
                    done += len(chunk)
                    if progress_callback:
                        progress_callback(done, total)
        if verify:
            checksum = self._checksum(self._chunk_hmac(c) for c in chunks)
            if not hmac.compare_digest(checksum, header["checksum"]):
                raise ChecksumError("校验和不匹配：数据已损坏或被篡改")
        return b"".join(chunks)

    # ------------------------------------------------------------------ #
    # 流式处理：内存可控（分块），支持断点续传
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stream_size(stream) -> Optional[int]:
        try:
            pos = stream.tell()
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(pos)
            return size
        except (OSError, AttributeError):
            return None

    def _save_checkpoint(self, path: str, offset: int,
                         chunk_hmacs: List[str]):
        state = {"offset": offset, "chunk_hmacs": chunk_hmacs}
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(state, fp)

    def compress_stream(self, reader, writer,
                        progress_callback: Optional[ProgressCallback] = None,
                        checkpoint_path: Optional[str] = None,
                        resume: bool = False) -> int:
        """流式压缩：每次只在内存中保留一个分块。

        resume=True 时从 checkpoint_path 记录的位置继续，实现断点续传。
        返回写入的总字节数。
        """
        total_size = self._stream_size(reader)
        offset = 0
        chunk_hmacs: List[str] = []
        if resume and checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as fp:
                state = json.load(fp)
            offset = int(state["offset"])
            chunk_hmacs = list(state["chunk_hmacs"])
            reader.seek(offset)
            writer.seek(0, 2)
        else:
            self._write_header(writer, b"\x00" * HEADER_CAPACITY)

        start_pos = writer.tell()
        done = offset
        while True:
            chunk = reader.read(self.chunk_size)
            if not chunk:
                break
            payload = self._compress_chunk(chunk)
            writer.write(self._frame(payload))
            chunk_hmacs.append(self._chunk_hmac(chunk).hex())
            done += len(chunk)
            if checkpoint_path:
                self._save_checkpoint(checkpoint_path, done, chunk_hmacs)
            if progress_callback:
                progress_callback(done, total_size)

        checksum = self._checksum(bytes.fromhex(h) for h in chunk_hmacs)
        header = self._make_header(done, len(chunk_hmacs), checksum)
        end_pos = writer.tell()
        writer.seek(0)
        self._write_header(writer, header)
        writer.seek(end_pos)
        return end_pos - start_pos + (8 + HEADER_CAPACITY)

    def decompress_stream(self, reader, writer,
                          progress_callback: Optional[ProgressCallback] = None,
                          verify: bool = True) -> int:
        """流式解压：边下载边解压，内存占用不超过一个分块。

        每个分块解压后立即写出并增量验证 HMAC，全部数据都被校验覆盖。
        返回写出的原始字节数。
        """
        header = self._read_header(reader)
        total = header["original_size"]
        done = 0
        chunk_hmacs = []
        for _ in range(header["num_chunks"]):
            (length,) = struct.unpack(">I", self._read_exact(reader, 4))
            payload = self._read_exact(reader, length)
            chunk = self._decompress_chunk(payload)
            chunk_hmacs.append(self._chunk_hmac(chunk))
            writer.write(chunk)
            done += len(chunk)
            if progress_callback:
                progress_callback(done, total)
        if verify:
            checksum = self._checksum(chunk_hmacs)
            if not hmac.compare_digest(checksum, header["checksum"]):
                raise ChecksumError("校验和不匹配：数据已损坏或被篡改")
        return done
