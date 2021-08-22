"""K Means Colors.

Usage:
python k_means_colors.py
"""

import argparse
import code
import logging
from time import sleep
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator, Union

import numpy as np
from PIL import Image, ImageDraw
from rich import traceback, pretty, inspect  # , print. REPL: inspect(obj, methods=True)
from rich.console import Console
from sklearn.cluster import KMeans

pretty.install()  # Does make lists annoying (1 per line). Alternative: (max_length=20)
traceback.install()  # (show_locals=True)
console = Console()


def main() -> None:
    # settings
    k = 10
    size = 75  # for rendering after

    with console.status("Loading image") as status:
        with Image.open("example.png") as im:
            x = np.asarray(im)[:, :, :3]  # ignore alpha channel (could include)
            x = x.reshape(x.shape[0] * x.shape[1], 3)  # flatten

    with console.status("Running K-means"):
        kmeans = KMeans(n_clusters=k).fit(x)

    # Render.
    im_palette = Image.new("RGB", (k * size, size))
    d = ImageDraw.Draw(im_palette)
    for i, color in enumerate(kmeans.cluster_centers_):
        d.rectangle(
            [(i * size, 0), ((i + 1) * size), size], fill=tuple(int(c) for c in color)
        )
    im_palette.show()


if __name__ == "__main__":
    main()
