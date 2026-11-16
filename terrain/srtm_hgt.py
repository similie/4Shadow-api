# terrain/srtm_hgt.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from affine import Affine


@dataclass
class DemMosaic:
    data: np.ndarray  # 2D array (rows, cols)
    transform: Affine  # Affine transform, EPSG:4326
    crs: CRS  # Expected: EPSG:4326

    @property
    def cellsize_deg(self) -> float:
        # Angular cellsize in degrees (assuming square pixels)
        return float(abs(self.transform.a))


def mosaic_hgt_tiles(paths: List[str]) -> DemMosaic:
    """
    Mosaic a list of SRTM .hgt tiles into a single DEM.

    Relies on GDAL's SRTMHGT driver, which infers georeferencing
    from the filename (N/S, E/W + degrees).
    """
    if not paths:
        raise ValueError("No .hgt paths provided to mosaic_hgt_tiles")

    srcs = [rasterio.open(p) for p in paths]
    try:
        # merge returns (bands, rows, cols)
        mosaic, out_transform = merge(srcs)
        crs = srcs[0].crs or CRS.from_epsg(4326)
    finally:
        for s in srcs:
            s.close()

    data = mosaic[0].astype("float32")
    return DemMosaic(
        data=data, transform=out_transform, crs=crs
    )  # # terrain/srtm_hgt.py


# from __future__ import annotations

# import math
# import os
# import re
# from dataclasses import dataclass
# from typing import Iterable, List, Tuple

# import numpy as np

# HGT_NODATA = -32768

# _HGT_RE = re.compile(
#     r"^(?P<ns>[NS])(?P<lat>\d{2})(?P<ew>[EW])(?P<lon>\d{3})\.hgt$",
#     re.IGNORECASE,
# )


# @dataclass
# class HgtTileMeta:
#     path: str
#     lat_deg: int  # integer latitude of SW corner (e.g. -9 for S09)
#     lon_deg: int  # integer longitude of SW corner (e.g. 124 for E124)
#     size: int  # number of rows/cols (e.g. 3601, 1201)
#     cellsize_deg: float  # angular spacing in degrees


# @dataclass
# class DemMosaic:
#     data: np.ndarray  # 2D float32 array (np.nan for nodata)
#     lat0: float  # southern edge (deg)
#     lon0: float  # western edge (deg)
#     cellsize_deg: float  # angular spacing (deg)

#     @property
#     def shape(self) -> Tuple[int, int]:
#         return self.data.shape

#     def latlon_to_rowcol(self, lat: float, lon: float) -> Tuple[int, int]:
#         """
#         Convert lat/lon (EPSG:4326) to raster indices (row, col).
#         Row 0 = north, col 0 = west.
#         """
#         rows, cols = self.data.shape

#         # northern edge of mosaic
#         lat_north = self.lat0 + self.cellsize_deg * (rows - 1)
#         lon_west = self.lon0

#         # row increases southward
#         row_f = (lat_north - lat) / self.cellsize_deg
#         col_f = (lon - lon_west) / self.cellsize_deg

#         row = int(math.floor(row_f))
#         col = int(math.floor(col_f))

#         if not (0 <= row < rows and 0 <= col < cols):
#             raise ValueError(f"Lat/Lon ({lat}, {lon}) outside DEM extent")

#         return row, col


# def parse_hgt_filename(filename: str) -> Tuple[int, int]:
#     """
#     Parse HGT tile name like 'S09E124.hgt' → (lat_deg, lon_deg) of SW corner.
#     """
#     name = os.path.basename(filename)
#     m = _HGT_RE.match(name)
#     if not m:
#         raise ValueError(f"Invalid HGT filename: {name}")

#     lat = int(m.group("lat"))
#     if m.group("ns").upper() == "S":
#         lat = -lat

#     lon = int(m.group("lon"))
#     if m.group("ew").upper() == "W":
#         lon = -lon

#     return lat, lon


# def read_hgt_tile(
#     path: str, nodata: int = HGT_NODATA
# ) -> Tuple[np.ndarray, HgtTileMeta]:
#     """
#     Read a single .hgt file into a 2D float32 DEM with np.nan for nodata.
#     """
#     lat_deg, lon_deg = parse_hgt_filename(path)

#     raw = np.fromfile(path, dtype=">i2")  # big-endian int16
#     size = int(math.sqrt(raw.size))
#     if size * size != raw.size:
#         raise ValueError(f"HGT file {path} is not square (n={raw.size})")

#     dem = raw.reshape(size, size).astype("float32")
#     dem[dem == nodata] = np.nan

#     cellsize_deg = 1.0 / (size - 1)  # 1 degree per tile

#     meta = HgtTileMeta(
#         path=path,
#         lat_deg=lat_deg,
#         lon_deg=lon_deg,
#         size=size,
#         cellsize_deg=cellsize_deg,
#     )
#     return dem, meta


# def mosaic_hgt_tiles(paths: Iterable[str]) -> DemMosaic:
#     """
#     Mosaic multiple SRTM HGT tiles into a single DEM.

#     Assumes tiles are north-up, pixel-registered, and share a common resolution.
#     """
#     tiles: List[Tuple[np.ndarray, HgtTileMeta]] = [read_hgt_tile(p) for p in paths]
#     if not tiles:
#         raise ValueError("No HGT tiles provided")

#     # Enforce common resolution
#     size0 = tiles[0][1].size
#     cellsize_deg = tiles[0][1].cellsize_deg
#     for _, meta in tiles:
#         if meta.size != size0:
#             raise ValueError(f"Resolution mismatch: {meta.path} has size {meta.size}")
#         if abs(meta.cellsize_deg - cellsize_deg) > 1e-12:
#             raise ValueError(f"Angular resolution mismatch: {meta.path}")

#     size = size0
#     inc_per_deg = size - 1  # grid increments per degree

#     # Determine mosaic bounds in degrees (tile edges)
#     south_edge = min(meta.lat_deg for _, meta in tiles)
#     north_edge = max(meta.lat_deg + 1 for _, meta in tiles)
#     west_edge = min(meta.lon_deg for _, meta in tiles)
#     east_edge = max(meta.lon_deg + 1 for _, meta in tiles)

#     # Number of rows/cols = increments + 1
#     nrows = int((north_edge - south_edge) * inc_per_deg) + 1
#     ncols = int((east_edge - west_edge) * inc_per_deg) + 1

#     mosaic = np.full((nrows, ncols), np.nan, dtype="float32")

#     for dem, meta in tiles:
#         # North edge of this tile
#         tile_lat_north = meta.lat_deg + 1.0

#         # Row index of this tile's row 0 (north)
#         row0 = int(round((north_edge - tile_lat_north) * inc_per_deg))

#         # Column index of this tile's col 0 (west)
#         col0 = int(round((meta.lon_deg - west_edge) * inc_per_deg))

#         r1 = row0 + size
#         c1 = col0 + size

#         if row0 < 0 or col0 < 0 or r1 > nrows or c1 > ncols:
#             raise RuntimeError(
#                 f"Tile {meta.path} ({meta.lat_deg},{meta.lon_deg}) "
#                 f"does not fit into mosaic bounds"
#             )

#         mosaic[row0:r1, col0:c1] = dem

#     return DemMosaic(
#         data=mosaic,
#         lat0=south_edge,
#         lon0=west_edge,
#         cellsize_deg=cellsize_deg,
#     )


# # from __future__ import annotations

# # import math
# # import os
# # import re
# # from dataclasses import dataclass
# # from typing import Iterable, Tuple

# # import numpy as np

# # HGT_NODATA = -32768


# # @dataclass
# # class HgtTileMeta:
# #     path: str
# #     lat_deg: int  # integer lat at SW corner (e.g. -9 for S09)
# #     lon_deg: int  # integer lon at SW corner (e.g. 124 for E124)
# #     size: int  # number of rows/cols (e.g. 1201, 3601)
# #     cellsize_deg: float  # angular resolution in degrees


# # @dataclass
# # class DemMosaic:
# #     data: np.ndarray  # 2D float32 DEM with np.nan for nodata
# #     lat0: float  # latitude of southern edge (degrees)
# #     lon0: float  # longitude of western edge (degrees)
# #     cellsize_deg: float

# #     @property
# #     def shape(self) -> Tuple[int, int]:
# #         return self.data.shape

# #     def latlon_to_rowcol(self, lat: float, lon: float) -> Tuple[int, int]:
# #         """
# #         Convert lat/lon (EPSG:4326) to DEM indices (row, col).
# #         Assumes north-up, row 0 = north, col 0 = west.
# #         """
# #         nrows, ncols = self.data.shape

# #         # northern edge = lat0 + height
# #         lat_north = self.lat0 + self.cellsize_deg * (nrows - 1)
# #         lon_west = self.lon0

# #         # row increases southward
# #         row_f = (lat_north - lat) / self.cellsize_deg
# #         col_f = (lon - lon_west) / self.cellsize_deg

# #         row = int(math.floor(row_f))
# #         col = int(math.floor(col_f))

# #         if not (0 <= row < nrows and 0 <= col < ncols):
# #             raise ValueError(f"Lat/Lon ({lat}, {lon}) outside DEM extent")

# #         return row, col


# # _HGT_RE = re.compile(
# #     r"^(?P<ns>[NS])(?P<lat>\d{2})(?P<ew>[EW])(?P<lon>\d{3})\.hgt$", re.IGNORECASE
# # )


# # def parse_hgt_filename(filename: str) -> Tuple[int, int]:
# #     """
# #     Parse SRTM tile name like S09E124.hgt into (lat_deg, lon_deg) of SW corner.
# #     """
# #     name = os.path.basename(filename)
# #     m = _HGT_RE.match(name)
# #     if not m:
# #         raise ValueError(f"Not a valid HGT filename: {name}")

# #     lat = int(m.group("lat"))
# #     if m.group("ns").upper() == "S":
# #         lat = -lat

# #     lon = int(m.group("lon"))
# #     if m.group("ew").upper() == "W":
# #         lon = -lon

# #     return lat, lon


# # def read_hgt_tile(
# #     path: str, nodata: int = HGT_NODATA
# # ) -> Tuple[np.ndarray, HgtTileMeta]:
# #     """
# #     Read a single .hgt file into a 2D float32 array (np.nan for nodata)
# #     plus metadata about its geolocation.
# #     """
# #     lat_deg, lon_deg = parse_hgt_filename(path)

# #     # HGT is big-endian int16
# #     raw = np.fromfile(path, dtype=">i2")
# #     size = int(math.sqrt(raw.size))
# #     if size * size != raw.size:
# #         raise ValueError(f"HGT file {path} is not square (found {raw.size} samples)")

# #     dem = raw.reshape(size, size).astype("float32")
# #     dem[dem == nodata] = np.nan

# #     cellsize_deg = 1.0 / (size - 1)  # one degree per tile

# #     meta = HgtTileMeta(
# #         path=path,
# #         lat_deg=lat_deg,
# #         lon_deg=lon_deg,
# #         size=size,
# #         cellsize_deg=cellsize_deg,
# #     )
# #     return dem, meta


# # def mosaic_hgt_tiles(paths: Iterable[str]) -> DemMosaic:
# #     tiles = [read_hgt_tile(p) for p in paths]
# #     if not tiles:
# #         raise ValueError("No HGT tiles provided")

# #     # enforce uniform tile size
# #     size = tiles[0][1].size
# #     for dem, meta in tiles:
# #         if dem.shape[0] != size:
# #             raise ValueError(f"Mixed resolutions: {meta.path} has {dem.shape}")

# #     # cell size (deg)
# #     cellsize = 1.0 / (size - 1)

# #     # compute mosaic bounds using tile edges
# #     south_edge = min(meta.lat_deg for _, meta in tiles)
# #     north_edge = max(meta.lat_deg + 1 for _, meta in tiles)
# #     west_edge = min(meta.lon_deg for _, meta in tiles)
# #     east_edge = max(meta.lon_deg + 1 for _, meta in tiles)

# #     # rows/cols: number of increments + 1 to include all points
# #     nrows = int((north_edge - south_edge) * (size - 1)) + 1
# #     ncols = int((east_edge - west_edge) * (size - 1)) + 1

# #     # allocate mosaic and fill with NaN
# #     mosaic = np.full((nrows, ncols), np.nan, dtype=np.float32)

# #     for dem, meta in tiles:
# #         # offsets in the global mosaic
# #         row0 = int((north_edge - (meta.lat_deg + 1)) * (size - 1))
# #         col0 = int((meta.lon_deg - west_edge) * (size - 1))
# #         # place tile
# #         mosaic[row0 : row0 + size, col0 : col0 + size] = dem

# #     dem_mosaic = DemMosaic(
# #         data=mosaic, lat0=south_edge, lon0=west_edge, cellsize_deg=cellsize
# #     )
# #     return dem_mosaic


# # # def mosaic_hgt_tiles(paths: Iterable[str]) -> DemMosaic:
# # #     """
# # #     Mosaic multiple HGT tiles into a single DEM covering their extent.
# # #     Assumes all tiles have same resolution.
# # #     """
# # #     tiles: List[Tuple[np.ndarray, HgtTileMeta]] = [read_hgt_tile(p) for p in paths]
# # #     if not tiles:
# # #         raise ValueError("No HGT tiles provided")

# # #     # Assume common resolution
# # #     cellsize_deg = tiles[0][1].cellsize_deg
# # #     size = tiles[0][1].size

# # #     # Determine global extent in integer degrees
# # #     min_lat = min(t[1].lat_deg for t in tiles)
# # #     max_lat = max(t[1].lat_deg for t in tiles) + 1  # +1 degree for northern edge
# # #     min_lon = min(t[1].lon_deg for t in tiles)
# # #     max_lon = max(t[1].lon_deg for t in tiles) + 1  # +1 degree for eastern edge

# # #     # Convert to row/col counts
# # #     total_rows = int(round((max_lat - min_lat) / cellsize_deg))  # south→north
# # #     total_cols = int(round((max_lon - min_lon) / cellsize_deg))  # west→east

# # #     # Global DEM, north-up (row 0 north)
# # #     mosaic = np.full((total_rows, total_cols), np.nan, dtype="float32")

# # #     for dem, meta in tiles:
# # #         print(meta.path, dem.shape, meta.lat_deg, meta.lon_deg)
# # #         if meta.cellsize_deg != cellsize_deg or meta.size != size:
# # #             raise ValueError("All HGT tiles must have same resolution")

# # #         # For this tile:
# # #         # row 0 = north edge at (meta.lat_deg + 1)
# # #         tile_rows, tile_cols = dem.shape

# # #         # Global row index of this tile's row 0
# # #         tile_lat_north = meta.lat_deg + 1.0
# # #         global_lat_north = min_lat + cellsize_deg * (total_rows - 1)
# # #         row0 = int(round((global_lat_north - tile_lat_north) / cellsize_deg))

# # #         # Global col index of this tile's col 0 (west edge)
# # #         col0 = int(round((meta.lon_deg - min_lon) / cellsize_deg))

# # #         mosaic[row0 : row0 + tile_rows, col0 : col0 + tile_cols] = dem

# # #     # southern + western edges in degrees
# # #     lat0 = min_lat
# # #     lon0 = min_lon

# # #     return DemMosaic(data=mosaic, lat0=lat0, lon0=lon0, cellsize_deg=cellsize_deg)
