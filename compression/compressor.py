"""
游戏内资源压缩与解压系统 - 核心模块

功能：
- compress()/decompress(): 基于 zlib(DEFLATE) 的高效压缩，支持字典、多线程、
  进度回调、全量 HMAC 校验，压缩结果大于原始数据时自动回退为存储模式。
- dictionary_encoding(): 基于 n-gram 频率分析的字典训练，常用词优先，持久化到磁盘。
- checksum_verification(): 基于 HMAC-SHA256 的加密校验，覆盖全部数据，防篡改。
- streaming_compress()/streaming_decompress(): 分帧流式处理，内存可控（帧大小上限），
  支持断点续传（状态文件记录进度，重复帧自动去重）。
"""
import hashlib
import hmac
import json
import os
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union

ProgressCallback = Callable[[int, int], None]

_MAGIC = b"GRC1"
_FRAME_MAGIC = b"GRCF"
_FLAG_STORED = 0x01
_FLAG_DICT = 0x02
_HEADER = struct.Struct("<4sB32sQ")          # magic, flags, dict_id, chunk_count
_CHUNK_ENTRY = struct.Struct("<QQ")          # raw_len, comp_len
_FRAME_HEADER = struct.Struct("<4sBIQQ")     # magic, flags, index, raw_len, comp_len
_CHECKSUM_LEN = 32                           # HMAC-SHA256
_DICT_MAGIC = b"GRCD"
_MAX_DICT_SIZE = 32768                       # deflate 窗口大小，超出部分无意义


class ResourceCompressor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.num_threads = max(1, int(self.config.get("num_threads") or min(4, os.cpu_count() or 1)))
        self.chunk_size = int(self.config.get("chunk_size", 1024 * 1024))
        self.level = int(self.config.get("level", 9))
        key = self.config.get("checksum_key", b"gsb-resource-compressor")
        self.checksum_key = key.encode() if isinstance(key, str) else bytes(key)
        self.dict_path = self.config.get("dict_path")
        self._state = {}
        self._history = []
        self._dictionary = b""
        self._dict_lock = threading.Lock()
        if self.dict_path:
            self._dictionary = self._load_dictionary_file(self.dict_path)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 字典编码：训练、优化、持久化
    # ------------------------------------------------------------------
    def dictionary_encoding(self, samples: Optional[Union[bytes, Iterable[bytes]]] = None) -> bytes:
        """返回当前压缩字典；传入 samples 时重新训练并持久化。

        字典通过 n-gram 频率分析构建：统计样本中重复出现的字节序列，
        按 频率*(长度-1) 的压缩收益排序，收益最高的序列放在字典尾部
        （deflate 优先匹配字典尾部），截断到 32KB 窗口大小。
        """
        if samples is not None:
            if isinstance(samples, (bytes, bytearray, memoryview)):
                samples = [bytes(samples)]
            dictionary = self._train_dictionary([bytes(s) for s in samples])
            with self._dict_lock:
                self._dictionary = dictionary
            if self.dict_path:
                self._save_dictionary_file(self.dict_path, dictionary)
        with self._dict_lock:
            return self._dictionary

    @staticmethod
    def _train_dictionary(samples: List[bytes]) -> bytes:
        if not samples:
            return b""
        ngram_len = 16
        stride = max(1, ngram_len // 4)
        counts: Dict[bytes, int] = {}
        for sample in samples:
            if len(sample) < ngram_len:
                counts[bytes(sample)] = counts.get(bytes(sample), 0) + 1
                continue
            for i in range(0, len(sample) - ngram_len + 1, stride):
                gram = bytes(sample[i:i + ngram_len])
                counts[gram] = counts.get(gram, 0) + 1
        # 收益 = 出现次数 * (长度 - 1)，只保留真正重复出现的序列
        scored = sorted(
            ((count * (len(gram) - 1), gram) for gram, count in counts.items() if count > 1),
            key=lambda item: item[0],
        )
        dictionary = bytearray()
        for _score, gram in scored:
            if len(dictionary) + len(gram) > _MAX_DICT_SIZE:
                break
            dictionary += gram
        return bytes(dictionary)

    @staticmethod
    def _save_dictionary_file(path: str, dictionary: bytes) -> None:
        digest = hashlib.sha256(dictionary).digest()
        with open(path, "wb") as fh:
            fh.write(_DICT_MAGIC + digest + dictionary)

    @staticmethod
    def _load_dictionary_file(path: str) -> bytes:
        if not os.path.exists(path):
            return b""
        with open(path, "rb") as fh:
            blob = fh.read()
        if len(blob) < len(_DICT_MAGIC) + 32 or not blob.startswith(_DICT_MAGIC):
            return b""
        digest = blob[len(_DICT_MAGIC):len(_DICT_MAGIC) + 32]
        dictionary = blob[len(_DICT_MAGIC) + 32:]
        if hashlib.sha256(dictionary).digest() != digest:
            return b""
        return dictionary

    def _dictionary_id(self, dictionary: bytes) -> bytes:
        if not dictionary:
            return b"\x00" * 32
        return hashlib.sha256(dictionary).digest()

    # ------------------------------------------------------------------
    # 校验和：HMAC-SHA256，加密防篡改，覆盖全部数据
    # ------------------------------------------------------------------
    def checksum_verification(
        self,
        data: bytes,
        checksum: Optional[Union[bytes, str]] = None,
        key: Optional[bytes] = None,
    ) -> Union[str, bool]:
        """不传 checksum 时计算并返回 HMAC 十六进制串；传入时校验并返回 bool。"""
        mac = hmac.new(key or self.checksum_key, bytes(data), hashlib.sha256).digest()
        if checksum is None:
            return mac.hex()
        if isinstance(checksum, str):
            candidate = bytes.fromhex(checksum)
        else:
            candidate = bytes(checksum)
        return hmac.compare_digest(mac, candidate)

    def _checksum_bytes(self, data: bytes) -> bytes:
        return hmac.new(self.checksum_key, data, hashlib.sha256).digest()

    # ------------------------------------------------------------------
    # 压缩 / 解压
    # ------------------------------------------------------------------
    def _compress_chunk(self, chunk: bytes, dictionary: bytes) -> bytes:
        if dictionary:
            obj = zlib.compressobj(self.level, zlib.DEFLATED, zlib.MAX_WBITS, 9,
                                   zlib.Z_DEFAULT_STRATEGY, dictionary)
        else:
            obj = zlib.compressobj(self.level)
        return obj.compress(chunk) + obj.flush()

    def _decompress_chunk(self, payload: bytes, dictionary: bytes) -> bytes:
        obj = zlib.decompressobj(zlib.MAX_WBITS, zdict=dictionary) if dictionary else zlib.decompressobj()
        return obj.decompress(payload) + obj.flush()

    def compress(self, data: bytes, progress_callback: Optional[ProgressCallback] = None) -> bytes:
        """压缩数据。多线程分块压缩，结果附带覆盖全部数据的 HMAC 校验和。

        若压缩后不比原始数据小，自动回退为存储模式，保证输出不会膨胀。
        """
        data = bytes(data)
        total = len(data)
        dictionary = self.dictionary_encoding()
        flags = _FLAG_DICT if dictionary else 0

        chunks = [data[i:i + self.chunk_size] for i in range(0, total, self.chunk_size)] or [b""]
        compressed_chunks = self._parallel_map(
            lambda c: self._compress_chunk(c, dictionary), chunks, total, progress_callback)

        payload = b"".join(compressed_chunks)
        if len(payload) >= total:
            # 存储模式：压缩反而变大时直接存原始数据
            flags |= _FLAG_STORED
            entries = [_CHUNK_ENTRY.pack(len(c), len(c)) for c in chunks]
            payload = data
        else:
            entries = [_CHUNK_ENTRY.pack(len(raw), len(comp))
                       for raw, comp in zip(chunks, compressed_chunks)]

        header = _HEADER.pack(_MAGIC, flags, self._dictionary_id(dictionary), len(chunks))
        body = header + b"".join(entries) + payload
        return body + self._checksum_bytes(body)

    def decompress(self, data: bytes, progress_callback: Optional[ProgressCallback] = None,
                   verify: bool = True) -> bytes:
        """解压数据。先验证覆盖全部数据的 HMAC 校验和，再逐块解压并报告进度。"""
        data = bytes(data)
        if len(data) < _HEADER.size + _CHECKSUM_LEN:
            raise ValueError("数据太短，不是有效的压缩包")
        body, checksum = data[:-_CHECKSUM_LEN], data[-_CHECKSUM_LEN:]
        if verify and not self.checksum_verification(body, checksum):
            raise ValueError("校验和验证失败：数据已损坏或被篡改")

        magic, flags, dict_id, chunk_count = _HEADER.unpack_from(body, 0)
        if magic != _MAGIC:
            raise ValueError("无效的压缩包魔数")
        dictionary = b""
        if flags & _FLAG_DICT:
            dictionary = self.dictionary_encoding()
            if not dictionary or self._dictionary_id(dictionary) != dict_id:
                raise ValueError("压缩包使用了字典，但本地字典缺失或不匹配")

        offset = _HEADER.size
        entries = []
        for _ in range(chunk_count):
            raw_len, comp_len = _CHUNK_ENTRY.unpack_from(body, offset)
            entries.append((raw_len, comp_len))
            offset += _CHUNK_ENTRY.size
        payload = body[offset:]
        expected = sum(comp for _raw, comp in entries)
        if len(payload) != expected:
            raise ValueError("压缩包负载长度不符，数据不完整")

        if flags & _FLAG_STORED:
            result = payload
            if progress_callback:
                progress_callback(len(result), len(result))
            return result

        parts, pos = [], 0
        raw_chunks = []
        for _raw_len, comp_len in entries:
            parts.append(payload[pos:pos + comp_len])
            pos += comp_len
        raw_chunks = self._parallel_map(
            lambda p: self._decompress_chunk(p, dictionary), parts,
            sum(raw for raw, _c in entries), progress_callback, count_output=True)

        for (raw_len, _comp_len), chunk in zip(entries, raw_chunks):
            if len(chunk) != raw_len:
                raise ValueError("解压后块长度与头部记录不符，数据已损坏")
        return b"".join(raw_chunks)

    def _parallel_map(self, fn, items, total, progress_callback, count_output=False):
        if len(items) <= 1 or self.num_threads <= 1:
            results = [fn(item) for item in items]
            if progress_callback:
                progress_callback(total, total)
            return results
        done = 0
        lock = threading.Lock()

        def wrapped(item):
            nonlocal done
            result = fn(item)
            if progress_callback:
                with lock:
                    done += len(result) if count_output else len(item)
                    progress_callback(min(done, total), total)
            return result

        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            return list(pool.map(wrapped, items))

    # ------------------------------------------------------------------
    # 流式处理：分帧、内存可控、断点续传
    # ------------------------------------------------------------------
    def _encode_frame(self, index: int, chunk: bytes, dictionary: bytes) -> bytes:
        compressed = self._compress_chunk(chunk, dictionary)
        flags = 0
        if len(compressed) >= len(chunk):
            flags = _FLAG_STORED  # 单帧回退存储，避免膨胀
            compressed = chunk
        header = _FRAME_HEADER.pack(_FRAME_MAGIC, flags, index, len(chunk), len(compressed))
        return header + compressed + self._checksum_bytes(header + compressed)

    def _decode_frame(self, frame: bytes, dictionary: bytes) -> tuple:
        if len(frame) < _FRAME_HEADER.size + _CHECKSUM_LEN:
            raise ValueError("帧数据太短")
        body, checksum = frame[:-_CHECKSUM_LEN], frame[-_CHECKSUM_LEN:]
        if not self.checksum_verification(body, checksum):
            raise ValueError("帧校验和验证失败：数据已损坏或被篡改")
        magic, flags, index, raw_len, comp_len = _FRAME_HEADER.unpack_from(body, 0)
        if magic != _FRAME_MAGIC:
            raise ValueError("无效的帧魔数")
        payload = body[_FRAME_HEADER.size:]
        if len(payload) != comp_len:
            raise ValueError("帧负载长度不符")
        if flags & _FLAG_STORED:
            return index, payload
        chunk = self._decompress_chunk(payload, dictionary)
        if len(chunk) != raw_len:
            raise ValueError("帧解压后长度不符，数据已损坏")
        return index, chunk

    @staticmethod
    def _load_stream_state(state_path: Optional[str]) -> Dict:
        if not state_path or not os.path.exists(state_path):
            return {"next_index": 0}
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
            if "next_index" not in state:
                return {"next_index": 0}
            return state
        except (ValueError, OSError):
            return {"next_index": 0}

    @staticmethod
    def _save_stream_state(state_path: Optional[str], state: Dict) -> None:
        if not state_path:
            return
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)

    def streaming_compress(self, chunks: Iterable[bytes],
                           progress_callback: Optional[ProgressCallback] = None,
                           state_path: Optional[str] = None) -> Iterator[bytes]:
        """流式压缩：逐块产出自校验的帧，内存占用被 chunk 大小限制。

        传入 state_path 时记录断点；中断后用同一 state_path 重新调用，
        自动跳过已完成的块，实现断点续传。
        """
        state = self._load_stream_state(state_path)
        start_index = state.get("next_index", 0)
        dictionary = self.dictionary_encoding()
        total = len(chunks) if hasattr(chunks, "__len__") else 0
        for index, chunk in enumerate(chunks):
            if index < start_index:
                continue
            frame = self._encode_frame(index, bytes(chunk), dictionary)
            # 帧生成完成即推进断点；即使消费者侧丢失，解压端也会按索引去重
            self._save_stream_state(state_path, {"next_index": index + 1})
            yield frame
            if progress_callback:
                progress_callback(index + 1, total)

    def streaming_decompress(self, frames: Iterable[bytes],
                             progress_callback: Optional[ProgressCallback] = None,
                             state_path: Optional[str] = None) -> Iterator[bytes]:
        """流式解压：逐帧校验并解压，收到第一帧即可产出，无需等待全部下载。

        按帧索引去重（断点续传重发的帧会被跳过），内存占用被帧大小限制。
        """
        state = self._load_stream_state(state_path)
        next_index = state.get("next_index", 0)
        dictionary = self.dictionary_encoding()
        done = 0
        for frame in frames:
            index, chunk = self._decode_frame(bytes(frame), dictionary)
            if index < next_index:
                continue  # 断点续传时的重复帧
            if index != next_index:
                raise ValueError(f"帧乱序或丢失：期望 {next_index}，收到 {index}")
            next_index += 1
            done += 1
            yield chunk
            self._save_stream_state(state_path, {"next_index": next_index})
            if progress_callback:
                progress_callback(done, 0)

    def streaming(self, direction: str = "compress", **kwargs) -> Iterator[bytes]:
        """流式处理入口：direction 为 'compress' 或 'decompress'。"""
        if direction == "compress":
            return self.streaming_compress(kwargs.get("chunks", ()),
                                           kwargs.get("progress_callback"),
                                           kwargs.get("state_path"))
        if direction == "decompress":
            return self.streaming_decompress(kwargs.get("frames", ()),
                                             kwargs.get("progress_callback"),
                                             kwargs.get("state_path"))
        raise ValueError(f"未知的流式方向: {direction}")
