#!/usr/bin/env python3
"""Parallel, resumable downloader for the dm-meshgraphnets GCS datasets.

WHY THIS EXISTS
---------------
The official `meshgraphnets/download_dataset.sh` uses `wget`, and on this host
that sustains roughly 67 KB/s against the dm-meshgraphnets bucket. A single
`curl` stream starts near 9.5 MB/s but GCS throttles it down to ~150-350 KB/s
after a sustained burst. Running several ranged requests in parallel recovers
the throughput (measured ~14 MB/s aggregate), so that is what this does.

`flag_simple` is 12.2 GB. At 67 KB/s that is ~2 days; here it is ~15 minutes.

Design:
  * Each file is split into fixed-size chunks fetched by parallel `curl -r`
    range requests into `.parts/<file>.part<NN>`.
  * Chunks that already exist at the right size are skipped, so re-running
    resumes instead of restarting.
  * Chunks are concatenated in order and the final size is verified against
    the server's Content-Length.
  * Safe to interrupt and re-run.

Usage:
    python3 fetch_dataset.py <dataset_name> <output_dir> [--workers 12]
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import urllib.request

BASE = "https://storage.googleapis.com/dm-meshgraphnets"
CHUNK = 64 * 1024 * 1024  # 64 MiB chunks


def remote_size(url, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return int(resp.headers["Content-Length"])
        except Exception as exc:  # noqa: BLE001 - transient GCS errors
            if attempt == retries - 1:
                raise
            print("  HEAD %s failed (%s), retrying" % (url, exc), file=sys.stderr)
    return None


def fetch_chunk(url, dest, start, end, retries=8):
    """Download bytes [start, end] inclusive into `dest` via curl.

    Retries RESUME from however many bytes already landed rather than starting
    over. GCS drops long-lived connections on this host, so a naive
    delete-and-restart loop can livelock on a 64 MiB chunk: the connection dies
    at ~10 MiB, the partial is discarded, and every retry dies at the same
    place. Appending the remaining range makes each retry strictly forward
    progress.
    """
    want = end - start + 1
    if os.path.exists(dest) and os.path.getsize(dest) == want:
        return dest  # already done

    for attempt in range(retries):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have > want:  # corrupt/oversized from an earlier run
            os.remove(dest)
            have = 0
        if have == want:
            return dest

        # Append the still-missing tail of the range.
        with open(dest, "ab") as out:
            proc = subprocess.run(
                [
                    "curl", "-sSL", "--fail",
                    "--connect-timeout", "30",
                    "--max-time", "3600",
                    "-r", "%d-%d" % (start + have, end),
                    url,
                ],
                stdout=out,
                stderr=subprocess.PIPE,
            )
        if os.path.exists(dest) and os.path.getsize(dest) == want:
            return dest
        if attempt == retries - 1:
            raise RuntimeError(
                "chunk %d-%d failed after %d tries (got %d/%d bytes): %s"
                % (start, end, retries,
                   os.path.getsize(dest) if os.path.exists(dest) else 0, want,
                   proc.stderr.decode()[:200])
            )
    return dest


def fetch_file(url, dest, workers):
    total = remote_size(url)
    if os.path.exists(dest) and os.path.getsize(dest) == total:
        print("[skip] %s already complete (%d bytes)" % (os.path.basename(dest), total))
        return

    print("[get ] %s (%d bytes, %d chunks, %d workers)"
          % (os.path.basename(dest), total, -(-total // CHUNK), workers))

    parts_dir = dest + ".parts"
    os.makedirs(parts_dir, exist_ok=True)

    jobs = []
    for i, start in enumerate(range(0, total, CHUNK)):
        end = min(start + CHUNK - 1, total - 1)
        jobs.append((i, start, end, os.path.join(parts_dir, "part%04d" % i)))

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_chunk, url, p, s, e): (i, s, e)
            for i, s, e, p in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            i, s, e = futures[fut]
            fut.result()  # re-raises
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print("       %s: %d/%d chunks" % (os.path.basename(dest), done, len(jobs)))

    # Concatenate in order.
    tmp = dest + ".assembling"
    with open(tmp, "wb") as out:
        for _, _, _, p in jobs:
            with open(p, "rb") as f:
                while True:
                    buf = f.read(8 * 1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
    got = os.path.getsize(tmp)
    if got != total:
        os.remove(tmp)
        raise RuntimeError("%s: assembled %d bytes, expected %d" % (dest, got, total))
    os.replace(tmp, dest)

    # Cleanup is best-effort: if a previous instance is still winding down, a
    # part may already be gone. The assembled file is verified above, so a
    # missing scratch file is not an error.
    for _, _, _, p in jobs:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    try:
        os.rmdir(parts_dir)
    except OSError:
        pass
    print("[ ok ] %s (%d bytes)" % (os.path.basename(dest), got))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("output_dir")
    ap.add_argument("--workers", type=int, default=12,
                    help="parallel chunk downloads per file (default 12)")
    ap.add_argument("--files", default=None,
                    help="comma-separated subset, e.g. valid.tfrecord,test.tfrecord")
    ap.add_argument("--serial-files", action="store_true",
                    help="fetch files one at a time instead of concurrently")
    args = ap.parse_args()

    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    names = args.files.split(",") if args.files else [
        "meta.json", "train.tfrecord", "valid.tfrecord", "test.tfrecord"
    ]

    # Files are fetched concurrently, not one after another: the big
    # train.tfrecord otherwise leaves the two 1 GB splits waiting behind it and
    # adds their full transfer time to the tail.
    if args.serial_files:
        for name in names:
            fetch_file("%s/%s/%s" % (BASE, args.dataset, name),
                       os.path.join(out_dir, name), args.workers)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = [
                pool.submit(fetch_file, "%s/%s/%s" % (BASE, args.dataset, name),
                            os.path.join(out_dir, name), args.workers)
                for name in names
            ]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

    print("\n--- %s ---" % out_dir)
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        print("  %12d  %s" % (os.path.getsize(path), name))


if __name__ == "__main__":
    main()
