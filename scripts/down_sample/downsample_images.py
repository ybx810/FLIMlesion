"""Reproducible RGB downsampling. See README_downsample.md for data policy."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import warnings
from collections import Counter

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('lanczos', 'area', 'gaussian_pyramid')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
FIELDS = ('run_id source_path relative_path method target_size original_width original_height '
          'oriented_width oriented_height output_width output_height original_file_size_mb '
          'output_file_size_mb processing_time_seconds decode_time_seconds output_path status '
          'error_message input_mode input_bit_depth input_channels icc_sha256 exif_orientation '
          'color_policy output_mode estimated_peak_gib library interpolation pyramid_levels '
          'pyramid_sizes final_interpolation').split()


def target_dimensions(width, height, target):
    """Integer round-half-up; clamp a subpixel short edge to one pixel."""
    longest = max(width, height)
    if target > longest:
        raise ValueError('Upsampling is forbidden')
    return tuple(max(1, (v * target * 2 + longest) // (2 * longest)) for v in (width, height))


def resample(rgb, method, target):
    """Input is immutable HxWx3 uint8 RGB; OpenCV filtering is channel agnostic."""
    height, width = rgb.shape[:2]
    dims = target_dimensions(width, height, target)
    meta = dict(library=f'OpenCV {cv2.__version__}', pyramid_levels=0,
                pyramid_sizes='[]', final_interpolation='none')
    if method in ('lanczos', 'area'):
        flag = cv2.INTER_LANCZOS4 if method == 'lanczos' else cv2.INTER_AREA
        meta['interpolation'] = 'INTER_LANCZOS4 (8x8)' if method == 'lanczos' else 'INTER_AREA'
        return cv2.resize(rgb, dims, interpolation=flag), meta
    if method != 'gaussian_pyramid':
        raise ValueError(method)
    current = rgb
    levels = []
    while True:
        h, w = current.shape[:2]
        next_dims = ((w + 1) // 2, (h + 1) // 2)
        if max(next_dims) < target or next_dims == (w, h):
            break
        current = cv2.pyrDown(current, dstsize=next_dims, borderType=cv2.BORDER_REFLECT_101)
        levels.append(next_dims)
    meta.update(interpolation='pyrDown: 5x5 [1,4,6,4,1] outer product /256; BORDER_REFLECT_101',
                pyramid_levels=len(levels), pyramid_sizes=json.dumps(levels))
    if (current.shape[1], current.shape[0]) != dims:
        if current.shape[1] < dims[0] or current.shape[0] < dims[1]:
            raise ValueError('Final interpolation would upsample an axis')
        current = cv2.resize(current, dims, interpolation=cv2.INTER_LINEAR)
        meta['final_interpolation'] = 'INTER_LINEAR'
    return current, meta


def inspect_image(path, info):
    """Header inspection before decoding, including RGB16 PNG/TIFF hidden by Pillow."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', Image.DecompressionBombWarning)
        # Our explicit pixel-based memory guard replaces Pillow's generic pixel limit.
        previous = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            im = Image.open(path)
        finally:
            Image.MAX_IMAGE_PIXELS = previous
    try:
        bits = 8
        if im.format == 'PNG':
            with path.open('rb') as f:
                header = f.read(26)
            bits = header[24]
        elif im.format == 'TIFF':
            bits = im.tag_v2.get(258, (1,))
        orientation = int(im.getexif().get(274, 1))
        icc = im.info.get('icc_profile', b'')
        info.update(original_width=im.width, original_height=im.height,
                    input_mode=im.mode, input_bit_depth=str(bits), input_channels=len(im.getbands()),
                    exif_orientation=orientation, icc_sha256=hashlib.sha256(icc).hexdigest() if icc else '',
                    color_policy='RGB samples unchanged; ICC preserved, no color conversion; EXIF transpose',
                    output_mode='RGB/8-bit')
        bit_values = bits if isinstance(bits, (tuple, list)) else (bits,)
        if im.mode != 'RGB' or any(b != 8 for b in bit_values) or 'transparency' in im.info:
            raise ValueError(f'Unsupported input: mode={im.mode}, bits={bits}, transparency={"transparency" in im.info}; no implicit conversion')
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError('Multi-frame images are rejected; select a frame explicitly upstream')
        if orientation not in range(1, 9):
            raise ValueError(f'Invalid EXIF Orientation: {orientation}')
        w, h = im.size if orientation < 5 else (im.height, im.width)
        info.update(oriented_width=w, oriented_height=h)
        return im, icc
    except Exception:
        im.close()
        raise


def output_path(root, relative, method, target):
    # Retain extension AND a full relative-path hash: stable across scan/limit changes.
    digest = hashlib.sha256(relative.as_posix().encode('utf-8')).hexdigest()
    name = f'{relative.stem[:72]}__{relative.suffix[1:].lower()}__{digest}.png'
    path = root / method / str(target) / relative.parent / name
    if root.resolve() not in path.resolve().parents:
        raise ValueError(f'Output path escapes output root through a link: {path}')
    return path


def save_png(im, path, icc, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.downsample-', suffix='.png', dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        im.save(temp, format='PNG', icc_profile=icc or None, compress_level=6)
        if overwrite:
            os.replace(temp, path)
        else:
            # Exclusive creation prevents a second run from overwriting an existing output.
            os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def make_comparison(rgb, target, path, icc, overwrite):
    w, h = target_dimensions(rgb.shape[1], rgb.shape[0], target)
    # Exact output pixels, no second resize. White label strip is outside the panels.
    with Image.new('RGB', (w * 3, h + 48), 'white') as canvas:
        draw = ImageDraw.Draw(canvas)
        for index, method in enumerate(METHODS):
            arr, _ = resample(rgb, method, target)
            with Image.fromarray(arr) as panel:
                canvas.paste(panel, (index * w, 48))
            draw.text((index * w + 4, 4), f'{method}\n{w} x {h} | VISUAL ONLY', fill='black')
            del arr
        save_png(canvas, path, icc, overwrite)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/downsampled')
    p.add_argument('--methods', nargs='+', choices=(*METHODS, 'all'), default=['all'])
    p.add_argument('--sizes', nargs='+', type=int, default=[1024, 2048, 4096])
    p.add_argument('--limit', type=int)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--memory-budget-gb', type=float, default=18.0, help='Estimated per-image peak GiB budget')
    p.add_argument('--threads', type=int, default=1, help='OpenCV CPU threads; images remain sequential')
    p.add_argument('--comparisons', type=int, default=0, help='Number of first eligible sources to compare')
    p.add_argument('--comparison-size', type=int, default=1024)
    return p


def run(args):
    start = time.perf_counter()
    source_root, out = args.input_dir.resolve(), args.output_dir.resolve()
    if not source_root.is_dir():
        raise ValueError(f'Input directory does not exist: {source_root}')
    if source_root == out or source_root in out.parents or out in source_root.parents:
        raise ValueError('Input/output trees must be disjoint (protect source images and avoid rescanning outputs)')
    if any(s < 1 for s in args.sizes) or args.memory_budget_gb <= 0 or args.threads < 1 or args.comparison_size < 1 or args.comparisons < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError('Sizes, limit, memory budget and threads must be positive; comparisons nonnegative')
    methods = METHODS if 'all' in args.methods else tuple(dict.fromkeys(args.methods))
    sizes = sorted(set(args.sizes))
    files = sorted((p for p in source_root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.relative_to(source_root).as_posix())
    scanned = len(files)
    files = files[:args.limit]
    cv2.setNumThreads(args.threads)
    counts = Counter()
    run_id = time.strftime('%Y%m%dT%H%M%S') + f'-{time.time_ns()}'
    log = None
    writer = None
    comparisons = 0
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        for method in methods:
            for size in sizes:
                (out / method / str(size)).mkdir(parents=True, exist_ok=True)
        manifest = dict(run_id=run_id, python=platform.python_version(), numpy=np.__version__,
                        pillow=PIL.__version__, opencv=cv2.__version__, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        rounding='integer half-up, minimum short edge 1', decode='Pillow; EXIF transpose; RGB uint8 only',
                        memory_estimate='16 * source_pixels + 32 * max_requested_target^2 + 256 MiB')
        (out / f'run_{run_id}.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        log_path = out / 'processing_log.csv'
        needs_header = not log_path.exists() or log_path.stat().st_size == 0
        log = log_path.open('a', newline='', encoding='utf-8-sig')
        writer = csv.DictWriter(log, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
    try:
        with tqdm(total=len(files) * len(methods) * len(sizes), desc='Output tasks', disable=args.dry_run) as progress:
            for source in files:
                relative = source.relative_to(source_root)
                info = dict(run_id=run_id, source_path=str(source), relative_path=str(relative))
                rgb = im = None
                icc = b''
                error = ''
                decode_seconds = 0
                try:
                    info['original_file_size_mb'] = source.stat().st_size / 1024**2
                    im, icc = inspect_image(source, info)
                    longest = max(info['oriented_width'], info['oriented_height'])
                    active = [s for s in sizes if s <= longest and any(args.overwrite or not output_path(out, relative, m, s).exists() for m in methods)]
                    compare = comparisons < args.comparisons and longest >= args.comparison_size
                    requested = active + ([args.comparison_size] if compare else [])
                    peak = (16 * im.width * im.height + 32 * max(requested, default=1)**2 + 256 * 1024**2) / 1024**3
                    info['estimated_peak_gib'] = peak
                    if requested and not args.dry_run:
                        if peak > args.memory_budget_gb:
                            raise MemoryError(f'Estimated {peak:.2f} GiB exceeds budget {args.memory_budget_gb:.2f} GiB')
                        tick = time.perf_counter()
                        # TIFF plugins can apply orientation while loading; ImageOps handles this consistently.
                        oriented = ImageOps.exif_transpose(im)
                        try:
                            rgb = np.array(oriented, dtype=np.uint8, copy=True)
                            if rgb.shape != (info['oriented_height'], info['oriented_width'], 3):
                                raise ValueError(f'Decoded dimensions/channels disagree with inspected metadata: {rgb.shape}')
                            rgb.flags.writeable = False
                        finally:
                            oriented.close()
                        decode_seconds = time.perf_counter() - tick
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                finally:
                    if im is not None:
                        im.close()
                for method in methods:
                    for size in sizes:
                        tick = time.perf_counter()
                        path = output_path(out, relative, method, size)
                        row = dict(info, method=method, target_size=size, output_path=str(path), decode_time_seconds=decode_seconds)
                        result = None
                        try:
                            if path.exists() and not args.overwrite:
                                row.update(status='skipped_existing', error_message='Output exists; not revalidated')
                            elif info.get('oriented_width') and max(info['oriented_width'], info['oriented_height']) < size:
                                row.update(status='skipped_small_image', error_message='Original longest edge is smaller than target; upsampling forbidden')
                            elif error:
                                raise RuntimeError(error)
                            else:
                                dims = target_dimensions(info['oriented_width'], info['oriented_height'], size)
                                row.update(output_width=dims[0], output_height=dims[1])
                                if args.dry_run:
                                    row.update(status='planned', error_message='')
                                else:
                                    result, metadata = resample(rgb, method, size)
                                    row.update(metadata)
                                    with Image.fromarray(result) as output:
                                        save_png(output, path, icc, args.overwrite)
                                    row.update(status='success', output_file_size_mb=path.stat().st_size / 1024**2)
                        except FileExistsError:
                            row.update(status='skipped_existing', error_message='Output created by another process')
                        except Exception as exc:
                            row.update(status='failed', error_message=f'{type(exc).__name__}: {exc}')
                        finally:
                            del result
                        row['processing_time_seconds'] = time.perf_counter() - tick
                        counts[row['status']] += 1
                        if writer:
                            writer.writerow(row)
                            log.flush()
                        else:
                            print(json.dumps(row, ensure_ascii=False))
                        progress.update(1)
                if rgb is not None and comparisons < args.comparisons and max(rgb.shape[:2]) >= args.comparison_size:
                    comparison = output_path(out, relative, 'comparisons', args.comparison_size)
                    try:
                        if args.overwrite or not comparison.exists():
                            make_comparison(rgb, args.comparison_size, comparison, icc, args.overwrite)
                        comparisons += 1
                    except Exception as exc:
                        with (out / 'comparison_errors.jsonl').open('a', encoding='utf-8') as f:
                            f.write(json.dumps(dict(source=str(source), error=str(exc)), ensure_ascii=False) + '\n')
                        tqdm.write(f'Comparison failed: {source}: {exc}')
                del rgb
                gc.collect()
    finally:
        if log:
            log.close()
    summary = dict(scanned_images=scanned, selected_images=len(files), success=counts['success'],
                   skipped=counts['skipped_existing'] + counts['skipped_small_image'], failed=counts['failed'],
                   planned=counts['planned'], comparisons=comparisons, total_seconds=round(time.perf_counter() - start, 3))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    p = parser()
    args = p.parse_args()
    try:
        summary = run(args)
    except ValueError as exc:
        p.error(str(exc))
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
