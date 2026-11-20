"""Local dashboard for the saved Chapter 3 features and KTH videos."""
import argparse
from base64 import b64encode
import csv
from contextlib import closing
from functools import lru_cache
from hashlib import sha256
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from ipaddress import IPv4Address, IPv4Network
import json
from pathlib import Path
import re
from secrets import token_urlsafe
import sqlite3
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "video_dashboard"
DATASET = (ROOT / "Dataset").resolve()
SOURCE = ROOT / "results" / "full"
DEFAULT_QUERY = "boxing/person01_boxing_d1.mp4"
RADMIN_NETWORK = IPv4Network("26.0.0.0/8")
GROUPS = {"A": slice(0,16), "B": slice(16,136), "C": slice(136,160),
          "D": slice(160,244), "E": slice(244,292), "F": slice(292,324)}
METRICS = ("manhattan", "sqeuclidean", "huber", "cosine")
RATINGS = ("similar", "different", "unsure")


def prepare_database(path):
    """Import an extraction once, retaining ratings for the same video IDs."""
    feature_file = SOURCE / "features.npz"
    manifest_file = SOURCE / "manifest.csv"
    fingerprint = sha256(feature_file.read_bytes() + manifest_file.read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS dashboard_source (digest TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY, action TEXT NOT NULL, person TEXT NOT NULL,
                setting TEXT NOT NULL, frames INTEGER NOT NULL, fps REAL NOT NULL,
                valid_pairs INTEGER NOT NULL, dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL CHECK(length(vector) = 1296),
                valid BLOB NOT NULL CHECK(length(valid) = 324)
            );
            CREATE TABLE IF NOT EXISTS ratings (
                query TEXT NOT NULL, candidate TEXT NOT NULL,
                verdict TEXT NOT NULL CHECK(verdict IN ('similar','different','unsure')),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (query, candidate)
            );
        """)
        if db.execute("SELECT digest FROM dashboard_source").fetchone() == (fingerprint,):
            return
        with manifest_file.open(encoding="utf-8", newline="") as stream:
            manifest = list(csv.DictReader(stream))
        with np.load(feature_file, allow_pickle=False) as archive:
            ids = archive["video_ids"].tolist()
            names = archive["feature_names"].tolist()
            vectors = archive["vectors"].astype("<f4", copy=False)
            valid = archive["valid"].astype(np.bool_, copy=False)
        if (ids != [item["video_id"] for item in manifest] or not ids
                or len(set(ids)) != len(ids)
                or vectors.shape != valid.shape or vectors.shape != (len(ids), 324)
                or not np.isfinite(vectors).all()
                or any(any(not name.startswith(group + "_") for name in names[span])
                       for group, span in GROUPS.items())):
            raise ValueError("Extraction manifest and vector schema are inconsistent")
        rows = []
        for item, vector, mask in zip(manifest, vectors, valid):
            key = item["video_id"]
            action, filename = key.split("/")
            person, _, setting = Path(filename).stem.split("_")
            rows.append((key, action, person, setting, int(item["frames"]), float(item["fps"]),
                         int(item["valid_pairs"]), int(mask.sum()), vector.tobytes(), mask.tobytes()))
        db.execute("DELETE FROM videos")
        db.executemany("INSERT INTO videos VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        db.execute("DELETE FROM ratings WHERE query NOT IN (SELECT id FROM videos) OR candidate NOT IN (SELECT id FROM videos)")
        db.execute("DELETE FROM dashboard_source")
        db.execute("INSERT INTO dashboard_source VALUES (?)", (fingerprint,))


def measure(metric, query, candidates, common):
    delta = np.abs(candidates - query)
    size = np.maximum(common.sum(axis=1), 1)
    if metric == "cosine":
        left = np.where(common, query, 0)
        right = np.where(common, candidates, 0)
        norm_l = np.linalg.norm(left, axis=1)
        norm_r = np.linalg.norm(right, axis=1)
        norms = norm_l * norm_r
        similarity = np.divide((left * right).sum(axis=1), norms,
                               out=np.zeros(len(right)), where=norms > 1e-12)
        distance = 1 - np.clip(similarity, -1, 1)
        return np.where((norm_l <= 1e-12) & (norm_r <= 1e-12), 0, distance)
    if metric == "sqeuclidean":
        delta = delta ** 2
    elif metric == "huber":
        delta = np.where(delta <= 1, .5 * delta ** 2, delta - .5)
    return np.where(common, delta, 0).sum(axis=1) / size


class Library:
    def __init__(self, path):
        self.path = path
        prepare_database(path)
        with closing(self.connect()) as db:
            records = db.execute("SELECT * FROM videos ORDER BY id").fetchall()
        self.items = [
            dict(zip(("id", "action", "person", "setting", "frames", "fps", "valid_pairs", "dimensions"), row[:8]))
            for row in records
        ]
        self.index = {item["id"]: i for i, item in enumerate(self.items)}
        self.vectors = np.stack([np.frombuffer(row[8], dtype="<f4") for row in records]).astype(float)
        self.valid = np.stack([np.frombuffer(row[9], dtype=np.bool_) for row in records])
        self.people = np.array([item["person"] for item in self.items])
        self.default = DEFAULT_QUERY if DEFAULT_QUERY in self.index else self.items[0]["id"]

    def connect(self):
        return sqlite3.connect(self.path)

    def rate(self, query, candidate, verdict):
        with closing(self.connect()) as db, db:
            if verdict is None:
                db.execute("DELETE FROM ratings WHERE query=? AND candidate=?", (query, candidate))
            else:
                db.execute("""
                    INSERT INTO ratings (query, candidate, verdict) VALUES (?, ?, ?)
                    ON CONFLICT(query,candidate) DO UPDATE SET
                      verdict=excluded.verdict, updated_at=CURRENT_TIMESTAMP
                """, (query, candidate, verdict))
            return db.execute("SELECT count(*) FROM ratings").fetchone()[0]

    def rating_csv(self):
        with closing(self.connect()) as db:
            rows = db.execute("SELECT query,candidate,verdict,updated_at FROM ratings ORDER BY query,candidate").fetchall()
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(("video_truy_van", "video_ket_qua", "nhan_xet", "cap_nhat_utc"))
        labels = {"similar": "Tương tự", "different": "Khác", "unsure": "Chưa chắc"}
        writer.writerows((query, candidate, labels[verdict], stamp) for query, candidate, verdict, stamp in rows)
        return stream.getvalue().encode("utf-8-sig")

    @lru_cache(maxsize=512)
    def rank(self, key, metric, with_d, other_people):
        q = self.index[key]
        keep = np.arange(len(self.items)) != q
        if other_people:
            keep &= self.people != self.people[q]
        pool = np.flatnonzero(keep)
        if not len(pool):
            return [], 0
        vectors = self.vectors[pool]
        masks = self.valid[pool]
        n = masks.sum(axis=0)
        mean = np.where(masks, vectors, 0).sum(axis=0) / np.maximum(n, 1)
        variance = np.where(masks, (vectors - mean) ** 2, 0).sum(axis=0) / np.maximum(n, 1)
        std = np.sqrt(variance)
        usable = (n > 0) & (std > 1e-12)
        scale = np.where(usable, std, 1)
        query = np.where(usable, (self.vectors[q] - mean) / scale, 0)
        vectors = np.where(usable, (vectors - mean) / scale, 0)
        distances = np.zeros(len(pool))
        weights = np.zeros(len(pool))
        details = {}
        for group, span in GROUPS.items():
            common = masks[:, span] & self.valid[q, span]
            if group == "D":
                common &= usable[span]
            counts = common.sum(axis=1)
            size = span.stop - span.start
            weight = counts / size if with_d or group != "D" else np.zeros(len(pool))
            cost = measure(metric, query[span], vectors[:, span], common)
            distances += cost * weight
            weights += weight
            details[group] = (counts, size, weight, cost)
        distance = np.divide(distances, weights, out=np.full(len(pool), np.inf), where=weights > 0)
        order = np.lexsort((pool, distance))
        results = []
        for j in order[np.isfinite(distance[order])][:5]:
            groups = [
                {"key": g, "shared": int(n_shared[j]), "size": size, "weight": float(weight[j]),
                 "distance": float(cost[j]) if n_shared[j] else None}
                for g, (n_shared, size, weight, cost) in details.items()
            ]
            results.append({"video": self.items[pool[j]], "distance": float(distance[j]),
                            "score": float(1 / (1 + distance[j])), "groups": groups})
        return results, len(pool)

    def search(self, key, metric, with_d, other_people):
        results, total = self.rank(key, metric, with_d, other_people)
        with closing(self.connect()) as db:
            ratings = dict(db.execute("SELECT candidate,verdict FROM ratings WHERE query=?", (key,)))
            count = db.execute("SELECT count(*) FROM ratings").fetchone()[0]
        return {"query": self.items[self.index[key]], "candidates": total,
                "results": [dict(result, verdict=ratings.get(result["video"]["id"])) for result in results],
                "rating_count": count}


@lru_cache(maxsize=128)
def poster(path):
    capture = cv2.VideoCapture(str(path))
    try:
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(count * .49)))
        ok, frame = capture.read()
        if not ok:
            raise OSError(f"Cannot read {path.name}")
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError(f"Cannot encode {path.name}")
        return data.tobytes()
    finally:
        capture.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    static = {"/": ("index.html", "text/html; charset=utf-8"),
              "/app.css": ("app.css", "text/css; charset=utf-8"),
              "/app.js": ("app.js", "text/javascript; charset=utf-8")}

    @property
    def library(self):
        return self.server.library

    def respond(self, code, content, mime, **headers):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(content)

    def send_json(self, data, code=200):
        self.respond(code, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                     "application/json; charset=utf-8", Cache_Control="no-store")

    def has_access(self):
        token = self.server.viewer_token
        if token is None:
            return True
        if IPv4Address(self.client_address[0]) not in RADMIN_NETWORK:
            self.respond(403, b"", "text/plain", Cache_Control="no-store", Connection="close")
            self.close_connection = True
            return False
        offered = self.headers.get("Authorization", "").encode("utf-8")
        if compare_digest(offered, token):
            return True
        self.respond(401, b"", "text/plain", Cache_Control="no-store", Connection="close",
                     WWW_Authenticate='Basic realm="Kho video", charset="UTF-8"')
        self.close_connection = True
        return False

    def video_path(self, key):
        path = (DATASET / key).resolve()
        if key not in self.library.index or not path.is_relative_to(DATASET) or not path.is_file():
            raise FileNotFoundError
        return path

    def send_video(self, path):
        size = path.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get("Range")
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
            if match:
                low, high = match.groups()
                if low:
                    start = int(low)
                    end = min(end, int(high)) if high else end
                elif high:
                    start = max(0, size - int(high))
            if not match or not (match[1] or match[2]) or start > end or start >= size:
                self.respond(416, b"", "video/mp4", Content_Range=f"bytes */{size}")
                return
        self.send_response(206 if header else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command != "HEAD":
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    data = stream.read(min(remaining, 64 * 1024))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)

    def do_GET(self):
        if not self.has_access():
            return
        url = urlsplit(self.path)
        args = parse_qs(url.query)
        try:
            if url.path in self.static:
                name, mime = self.static[url.path]
                self.respond(200, (WEB / name).read_bytes(), mime, Cache_Control="no-cache")
            elif url.path == "/api/videos":
                self.send_json({"videos": self.library.items, "default": self.library.default,
                                "viewer": self.server.viewer_token is not None})
            elif url.path == "/api/search":
                key = args.get("query", [self.library.default])[0]
                metric = args.get("metric", ["manhattan"])[0]
                if key not in self.library.index or metric not in METRICS:
                    self.send_json({"error": "Video hoặc thước đo không hợp lệ."}, 400)
                else:
                    options = (key, metric, args.get("d", ["1"])[0] == "1", args.get("other", ["0"])[0] == "1")
                    if self.server.viewer_token is not None:
                        results, total = self.library.rank(*options)
                        self.send_json({"query": self.library.items[self.library.index[key]],
                                        "candidates": total, "results": results})
                    else:
                        self.send_json(self.library.search(*options))
            elif url.path == "/api/ratings.csv":
                if self.server.viewer_token is not None:
                    self.send_error(HTTPStatus.FORBIDDEN)
                else:
                    self.respond(200, self.library.rating_csv(), "text/csv; charset=utf-8",
                                 Content_Disposition='attachment; filename="nhan_xet_video.csv"')
            elif url.path.startswith("/media/"):
                self.send_video(self.video_path(unquote(url.path[7:])))
            elif url.path.startswith("/poster/"):
                content = poster(self.video_path(unquote(url.path[8:])))
                self.respond(200, content, "image/jpeg", Cache_Control="private, max-age=3600")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    do_HEAD = do_GET

    def do_POST(self):
        if not self.has_access():
            return
        if self.server.viewer_token is not None:
            self.close_connection = True
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if urlsplit(self.path).path != "/api/rating":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 8192 or self.headers.get_content_type() != "application/json":
                raise ValueError
            body = json.loads(self.rfile.read(length))
            query, candidate, verdict = body["query"], body["candidate"], body["verdict"]
            if (not isinstance(query, str) or not isinstance(candidate, str) or query == candidate
                    or query not in self.library.index or candidate not in self.library.index
                    or (verdict is not None and verdict not in RATINGS)):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            self.send_json({"error": "Nhận xét không hợp lệ."}, 400)
            return
        self.send_json({"rating_count": self.library.rate(query, candidate, verdict)})

    def log_message(self, fmt, *args):
        if fmt.startswith("code"):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8061)
    parser.add_argument("--db", type=Path, default=ROOT / "results" / "video_dashboard.sqlite3")
    parser.add_argument("--vpn-address", type=IPv4Address,
                        help="Bind to this machine's Radmin VPN address with a read-only account")
    args = parser.parse_args()
    if args.vpn_address is not None and args.vpn_address not in RADMIN_NETWORK:
        parser.error("--vpn-address must be the IPv4 address assigned by Radmin VPN")
    library = Library(args.db.resolve())
    cv2.setNumThreads(1)
    address = str(args.vpn_address) if args.vpn_address is not None else "127.0.0.1"
    with ThreadingHTTPServer((address, args.port), Handler) as server:
        server.library = library
        server.viewer_token = None
        if args.vpn_address is not None:
            password = token_urlsafe(18)
            server.viewer_token = b"Basic " + b64encode(f"viewer:{password}".encode("ascii"))
            print(f"Viewer: http://{address}:{server.server_port}/\nUsername: viewer\nPassword: {password}", flush=True)
        else:
            print(f"Local: http://{address}:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
