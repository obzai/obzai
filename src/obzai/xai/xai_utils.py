# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.colors as clr
from typing import List
import cmasher as cmr
import numpy as np
import torch


COLOR_MAP = clr.LinearSegmentedColormap.from_list(
     "Random gradient 1030",
     (
          (0.000, (0.000, 0.890, 1.000)),
          (0.370, (0.263, 0.443, 0.671)),
          (0.500, (0.000, 0.000, 0.000)),
          (0.630, (0.545, 0.353, 0.267)),
          (1.000, (1.000, 0.651, 0.000)),
          ),
          )


def get_cmap(xai_map: np.ndarray):
    """Return a diverging colormap, ensuring 0 is at the center (black)."""
    abs_max = abs(xai_map).max()
    if abs_max == 0:
        return COLOR_MAP  # If all values are zero, return the full colormap
    bottom = max(0.0, 0.5 - abs(xai_map.min() / abs_max) / 2)
    top = min(1.0, 0.5 + abs(xai_map.max() / abs_max) / 2)
    return cmr.get_sub_cmap(COLOR_MAP, bottom, top)
    

def normalize_xai_maps(xai_maps: torch.Tensor) -> List[np.ndarray]:
        final_maps = []
        for xai_map in xai_maps:
            cmap = get_cmap(xai_map)
            normalized_map = clr.Normalize(vmin=xai_map.min(), vmax=xai_map.max())(xai_map.squeeze().numpy())
            normalized_map = clr.Normalize(vmin=xai_map.min(), vmax=xai_map.max())(xai_map.squeeze().numpy())
            new_xai_map = cmap(normalized_map)[..., :3]
            final_xai_map = (new_xai_map*255).astype('uint8')
            final_maps.append(final_xai_map)
        return final_maps