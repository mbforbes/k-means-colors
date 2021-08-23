"""K Means Colors.

Usage:
python k_means_colors.py
"""

import argparse
import code
import colorsys
import os
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator, Union

from imgcat import imgcat
import numpy as np
from PIL import Image, ImageDraw
from rich import traceback, pretty, inspect  # , print. REPL: inspect(obj, methods=True)
from rich.console import Console
from rich.progress import track
from sklearn.cluster import KMeans

# pretty.install()  # Does make lists annoying (1 per line). Alt: (max_length=20)
traceback.install()  # (show_locals=True)
C = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image", type=str, default="example.png", help="Input image")
    parser.add_argument(
        "--out-name", type=str, default="example", help="Prefix for output palette .png"
    )
    parser.add_argument("--min-k", type=int, default=1, help="Low k to try")
    parser.add_argument("--max-k", type=int, default=12, help="High k to try")
    parser.add_argument(
        "--space", choices=["rgb", "hsv"], default="rgb", help="Pixel color space"
    )
    parser.add_argument(
        "--size", type=int, default=75, help="Square size for each output palette color"
    )
    parser.add_argument(
        "--bg",
        type=int,
        default=0,
        help="Output palette bg greyscale color, in 0 ... 255",
    )
    args = parser.parse_args()

    in_path = os.path.expanduser(args.image)
    space, min_k, max_k, size = args.space.upper(), args.min_k, args.max_k, args.size
    assert min_k <= max_k and min_k > 0 and max_k < 10000  # Untested big upper bound
    assert size > 0 and size < 10000  # Untested big upper bound
    assert args.bg >= 0 and args.bg <= 255

    bg = (args.bg,) * 3
    ks = range(min_k, max_k + 1)

    C.log(f"Using {space}, k = {list(ks)}")
    with C.status("Loading image") as status:
        with Image.open(in_path) as im:
            if space == "RGB":
                x = np.asarray(im)[:, :, :3]  # ignore alpha channel (could include)
            elif space == "HSV":
                x = np.asarray(im.convert("HSV"))
            else:
                assert False, f"Unsupported color space '{space}'"
            x = x.reshape(x.shape[0] * x.shape[1], 3)  # flatten

    res = {}
    for k in track(ks):
        # n_init=1, max_iter=2  (pass for quickly testing later steps)
        kmeans = KMeans(n_clusters=k).fit(x)
        res[k] = kmeans.cluster_centers_

    # Render
    n_ks = max_k - min_k + 1
    im_palette = Image.new("RGB", (max_k * size, n_ks * size), bg)
    d = ImageDraw.Draw(im_palette)
    for k in ks:
        # Convert centers to RGB for display.
        if space == "RGB":
            centers = res[k].tolist()
        elif space == "HSV":
            centers = (
                np.asarray([colorsys.hsv_to_rgb(*c) for c in (res[k] / 255).tolist()])
                * 255
            ).tolist()
        else:
            assert False, f"Unsupported color space '{space}'"

        # Sorting colors by HSV. https://www.alanzucconi.com/2015/09/30/colour-sorting/
        centers.sort(key=lambda rgb: colorsys.rgb_to_hsv(*rgb))
        for i, color in enumerate(centers):
            d.rectangle(
                [
                    (i * size, (k - min_k) * size),
                    ((i + 1) * size, (k - min_k + 1) * size),
                ],
                fill=tuple(int(c) for c in color),
            )

    # Preview
    imgcat(im_palette)

    # Save
    k_str = f"{min_k}-{max_k}" if min_k != max_k else f"{min_k}"
    out_path = f"{args.out_name}.{space.lower()}.palette-{k_str}.png"
    im_palette.save(out_path)
    C.log(f"Wrote palette to {out_path}")


if __name__ == "__main__":
    main()
