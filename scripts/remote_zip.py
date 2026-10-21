"""Read single files out of a remote zip over HTTP range requests, without downloading the whole archive."""
import io
import struct
import urllib.request
import zipfile
import zlib


def _get(url, start, n):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + n - 1}"})
    return urllib.request.urlopen(req, timeout=120).read()


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(head, timeout=60).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = _get(self.url, self.pos, n)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url):
    """Parses the whole central directory: open it once in the parent process, not in every worker (83k entries is ~100 MB)."""
    return zipfile.ZipFile(io.BufferedReader(RangeFile(url), 1 << 20))


def read_member(url, header_offset, compress_size, compress_type):
    """One member's bytes, given the three ZipInfo fields. Needs no central directory, so workers stay small."""
    name_len, extra_len = struct.unpack("<HH", _get(url, header_offset, 30)[26:30])
    raw = _get(url, header_offset + 30 + name_len + extra_len, compress_size)
    return raw if compress_type == zipfile.ZIP_STORED else zlib.decompress(raw, -15)
