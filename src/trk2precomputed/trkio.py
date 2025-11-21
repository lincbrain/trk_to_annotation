import itertools
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from nibabel.orientations import (
    aff2axcodes, axcodes2ornt, ornt_transform, inv_ornt_aff
)

from trk2precomputed.io import NATIVE_BYTEORDER, Whence, smart_open, smart_read

# https://trackvis.org/docs/?subsect=fileformat
TRKHEADER_DTYPE = np.dtype([
    ('id_string', 'S6'),            # ID string for track file. The first 5 characters must be "TRACK".
    ('dim', 'i2', 3),               # Dimension of the image volume.
    ('voxel_size', 'f4', 3),        # Voxel size of the image volume.
    ('origin', 'f4', 3),            # Origin of the image volume. This field is not yet being used by TrackVis. That means the origin is always (0, 0, 0).
    ('n_scalars', 'i2'),            # Number of scalars saved at each track point (besides x, y and z coordinates).
    ('scalar_names', 'S20', 10),    # Name of each scalar. Can not be longer than 20 characters each. Can only store up to 10 names.
    ('n_properties', 'i2'),         # Number of properties saved at each track.
    ('property_names', 'S20', 10),  # Name of each property. Can not be longer than 20 characters each. Can only store up to 10 names.
    ('vox2ras', 'f4', (4, 4)),      # 4x4 matrix for voxel to RAS (crs to xyz) transformation. If vox_to_ras[3][3] is 0, it means the matrix is not recorded. This field is added from version 2.
    ('reserved', 'S444'),           # Reserved space for future version.
    ('voxel_order', 'S4'),          # Storing order of the original image data.
    ('pad2', 'S4'),                 # Paddings.
    ('image_orientation_patient', 'f4', 6),  # Image orientation of the original image. As defined in the DICOM header.
    ('pad1', 'S2'),                 # Paddings.
    ('invert_x', 'u1'),             # Inversion/rotation flags used to generate this track file. For internal use only.
    ('invert_y', 'u1'),             # As above.
    ('invert_z', 'u1'),             # As above.
    ('swap_xy', 'u1'),              # As above.
    ('swap_yz', 'u1'),              # As above.
    ('swap_zx', 'u1'),              # As above.
    ('n_count', 'i4'),              # Number of tracks stored in this track file. 0 means the number was NOT stored.
    ('version', 'i4'),              # Version number. Current version is 2.
    ('hdr_size', 'i4')              # Size of the header. Used to determine byte swap. Should be 1000.
])


def point3d_dtype(order: str = '') -> np.dtype:
    return np.dtype([
        ('x', f'{order}f4'),
        ('y', f'{order}f4'),
        ('z', f'{order}f4'),
    ])


def point_binary_dtype(
    scalar_names: list[str],
    order: str = ''
) -> np.dtype:
    dtype = [
        ('point', point3d_dtype(order))
    ]
    dtype += [
        ('scalar_' + name, f'{order}f4')
        for name in scalar_names
    ]
    return np.dtype(dtype)


def streamline_binary_dtype(
    nb_points: int,
    scalar_names: list[str],
    property_names: list[str],
    order: str = ''
) -> np.dtype:
    dtype = [
        ('per_point', point_binary_dtype(scalar_names, order), nb_points),
    ]
    dtype += [
        ('prop_' + name, f'{order}f4')
        for name in property_names
    ]
    return np.dtype(dtype)


POINT3D_DTYPE = point3d_dtype()


def per_streamline_dtype(property_names: list[str]) -> np.dtype:
    dtype = [
        ('id', 'i4'),
        ('length', 'i2'),
        ('offset', 'i8'),
    ]
    dtype += [
        ('prop_' + name, 'f4')
        for name in property_names
    ]
    return np.dtype(dtype)


def per_point_dtype(scalar_names: list[str]) -> np.dtype:
    dtype = [
        ('streamline', 'i8'),
        ('point', POINT3D_DTYPE)
    ]
    dtype += [
        ('scalar_' + name, 'f4')
        for name in scalar_names
    ]
    return np.dtype(dtype)


class Tractogram:
    """
    Holder for a set of streamlines that have been loaded in memory.

    Properties
    ----------
    per_streamline : shape (N,) np.ndarray
        id: int,             # Streamline ID
        length: int,         # Number of points in the streamline
        offset: int,         # Offset to the first point in per_point array
        prop_<name>: float,  # Per-streamline properties
    per_point : shape (M,) np.ndarray
        streamline: int,                        # Streamline ID
        point: (x: float, y: float, z: float),  # 3D coordinates of the point
        scalar_<name>: float,                   # Per-point scalars
    """

    def __init__(self, per_streamline, per_point) -> None:
        self.per_streamline = per_streamline
        self.per_point = per_point


class TrkHeader(Mapping):
    """
    TRK file header.
    """

    def __init__(
        self,
        header: np.ndarray,
        order: str = NATIVE_BYTEORDER
    ) -> None:
        self.header = header
        self.order = order
        self._trk2ras = None

    @classmethod
    def from_file(
        cls,
        file: str | Path | BinaryIO,
        **kwargs,
    ) -> 'TrkHeader':
        """Load a TRK header from a file.

        Parameters
        ----------
        file : str | Path | BinaryIO
            Path to the TRK file or a binary file object.
        kwargs : dict
            Additional arguments to pass to `smart_open`.

        Returns
        -------
        header : TrkHeader
            The loaded TRK header.
        """
        with smart_open(file, "rb", **kwargs) as file_obj:
            header = smart_read(file_obj, TRKHEADER_DTYPE)
            order = NATIVE_BYTEORDER
            if header['hdr_size'] != 1000:
                order = {'<': '>', '>': '<'}[NATIVE_BYTEORDER]
                header = header.byteswap(inplace=True)
            if header['hdr_size'] != 1000:
                raise ValueError('Not a TRK header.')
        return cls(header, order)

    # --- Coordinate space transformations -----------------------------

    @property
    def vox2ras(self) -> np.ndarray:
        """Get affine mapping voxel space to RAS+ mm space.

        Returns
        -------
        vox2ras : shape (4, 4) array
            Affine array mapping coordinates in voxel space to RAS+ mm space.
        """
        return self.header['vox2ras'].astype(np.float32)

    @property
    def ras2vox(self) -> np.ndarray:
        """Get affine mapping RAS+ mm space to voxel space.

        Returns
        -------
        ras2vox : shape (4, 4) array
            Affine array mapping coordinates in RAS+ mm space to voxel space.
        """
        return np.linalg.inv(self.vox2ras)

    @property
    def trk2ras(self) -> np.ndarray:
        """Get affine mapping trackvis voxelmm space to RAS+ mm space

        The streamlines in a trackvis file are in 'voxelmm' space, where the
        coordinates refer to the corner of the voxel.

        Compute the affine matrix that will bring them back to RAS+ mm space,
        where the coordinates refer to the center of the voxel.

        Returns
        -------
        trk2ras : shape (4, 4) array
            Affine array mapping coordinates in 'voxelmm' space to
            RAS+ mm space.
        """
        if self._trk2ras is not None:
            return self._trk2ras

        # Copied from nibabel

        # TRK's streamlines are in 'voxelmm' space, we will compute the
        # affine matrix that will bring them back to RAS+ and mm space.
        affine = np.eye(4)

        # The affine matrix found in the TRK header requires the points to
        # be in the voxel space.
        # voxelmm -> voxel
        scale = np.eye(4)
        scale[range(3), range(3)] /= self.voxel_size
        affine = np.dot(scale, affine)

        # TrackVis considers coordinate (0,0,0) to be the corner of the
        # voxel whereas streamlines returned assumes (0,0,0) to be the
        # center of the voxel. Thus, streamlines are shifted by half a voxel.
        offset = np.eye(4)
        offset[:-1, -1] -= 0.5
        affine = np.dot(offset, affine)

        # If the voxel order implied by the affine does not match the voxel
        # order in the TRK header, change the orientation.
        # voxel (header) -> voxel (affine)
        vox_order = self.voxel_order
        affine_ornt = ''.join(aff2axcodes(self.vox2ras))
        header_ornt = axcodes2ornt(vox_order)
        affine_ornt = axcodes2ornt(affine_ornt)
        ornt = ornt_transform(header_ornt, affine_ornt)
        M = inv_ornt_aff(ornt, self.dim)
        affine = np.dot(M, affine)

        # Applied the affine found in the TRK header.
        # voxel -> rasmm
        voxel_to_rasmm = self.vox2ras
        affine_voxmm_to_rasmm = np.dot(voxel_to_rasmm, affine)
        self._trk2ras = affine_voxmm_to_rasmm.astype(np.float32)

        return self._trk2ras

    @property
    def ras2trk(self) -> np.ndarray:
        """Get affine mapping RAS+ mm space to trackvis voxelmm space.

        Returns
        -------
        ras2trk : shape (4, 4) array
            Affine array mapping coordinates in RAS+ mm space to
            trackvis voxelmm space.
        """
        return np.linalg.inv(self.trk2ras)

    @property
    def trk2vox(self) -> np.ndarray:
        """Get affine mapping trackvis voxelmm space to voxel space.

        Returns
        -------
        trk2vox : shape (4, 4) array
            Affine array mapping coordinates in trackvis voxelmm space to
            voxel space.
        """
        return self.ras2trk @ self.vox2ras

    @property
    def vox2trk(self) -> np.ndarray:
        """Get affine mapping voxel space to trackvis voxelmm space.

        Returns
        -------
        vox2trk : shape (4, 4) array
            Affine array mapping coordinates in voxel space to
            trackvis voxelmm space.
        """
        return self.ras2trk @ self.vox2ras

    def _bbox(self, vox2space: np.ndarray, mode: str = 'edges') -> np.ndarray:
        """Get bounding box of the MRI volume in given space.

        Parameters
        ----------
        vox2space : shape (4, 4) array
            Affine mapping voxel space to target space.
        mode : {'centers', 'edges'}, default='edges'
            Whether to return the bounding box defined by the centers of
            the corner voxels ('centers') or by the edges of the volume
            ('edges').

        Returns
        -------
        bbox : shape (2, 3) array
            Bounding box of the MRI volume in target space.
            First row is the minimum corner, second row is the maximum corner.
        """
        if mode[0].lower() == 'c':
            corners = np.array(list(itertools.product(
                [0, self.dim[0] - 1],
                [0, self.dim[1] - 1],
                [0, self.dim[2] - 1],
            )), dtype=np.float32)
        else:
            corners = np.array(list(itertools.product(
                [-0.5, self.dim[0] - 0.5],
                [-0.5, self.dim[1] - 0.5],
                [-0.5, self.dim[2] - 0.5],
            )), dtype=np.float32)

        corners = vox2space[:3, :3] @ corners.T
        corners += vox2space[:3, -1:]
        min_corner = corners.min(axis=1)
        max_corner = corners.max(axis=1)

        return np.vstack([min_corner, max_corner])

    def bbox_trk(self, mode: str = 'edges') -> np.ndarray:
        """Get bounding box of the MRI volume in trackvis voxelmm space.

        Parameters
        ----------
        mode : {'centers', 'edges'}, default='edges'
            Whether to return the bounding box defined by the centers of
            the corner voxels ('centers') or by the edges of the volume
            ('edges').

        Returns
        -------
        bbox_trk : shape (2, 3) array
            Bounding box of the MRI volume in trackvis voxelmm space.
            First row is the minimum corner, second row is the maximum corner.
        """
        return self._bbox(self.vox2trk, mode)

    def bbox_ras(self, mode: str = 'edges') -> np.ndarray:
        """Get bounding box of the MRI volume in RAS+ mm space.

        Parameters
        ----------
        mode : {'centers', 'edges'}, default='edges'
            Whether to return the bounding box defined by the centers of
            the corner voxels ('centers') or by the edges of the volume
            ('edges').

        Returns
        -------
        bbox_ras : shape (2, 3) array
            Bounding box of the MRI volume in RAS+ mm space.
            First row is the minimum corner, second row is the maximum corner.
        """
        return self._bbox(self.vox2ras, mode)

    # --- Dot access and properties ------------------------------------

    @property
    def id_string(self) -> str:
        return self.header['id_string'].decode('utf-8').strip('\x00')

    @property
    def scalar_names(self) -> list[str]:
        nb_scalars = self.header['n_scalars']
        return [
            name.decode('utf-8').strip('\x00')
            for name in self.header['scalar_names'][:nb_scalars]
        ]

    @property
    def property_names(self) -> list[str]:
        nb_properties = self.header['n_properties']
        return [
            name.decode('utf-8').strip('\x00')
            for name in self.header['property_names'][:nb_properties]
        ]

    @property
    def voxel_order(self) -> str:
        return self.header['voxel_order'].decode('utf-8').strip('\x00')

    def __getattr__(self, name: str) -> any:
        if name not in self.header.dtype.names:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        value = self.header[name]
        if np.ndim(value) == 0:
            return value.item()
        if np.ndim(value) == 1:
            return value.tolist()
        return value

    # --- Mapping abstract methods -------------------------------------

    def __getitem__(self, name: str) -> any:
        if name not in self.header.dtype.names:
            raise KeyError(f"Header has no field '{name}'")
        return getattr(self, name)

    def __len__(self) -> int:
        return len(self.header.dtype.names)

    def __iter__(self):
        for name in self.header.dtype.names:
            yield name

    def __str__(self) -> str:
        nchar = max(map(len, self.keys()))
        s = ['TrkHeader']
        for name, value in self.items():
            prefix = f'| {name:{nchar}} : '
            if isinstance(value, np.ndarray):
                row = prefix + np.array2string(
                    value, prefix=prefix, max_line_width=1024
                )
                row = row.replace('\n ', '\n|')
            else:
                row = f'{prefix}{value}'
            s += [row]
        return '\n'.join(s)

    __repr__ = __str__


class TrkReader(Sequence):
    """
    TRK reader that allows partial loading of streamlines.

    Example
    -------
    >>> reader = TrkReader('tracts.trk')
    >>> print(reader.header)
    >>> print(f'The file contains {len(reader)} streamlines.')
    >>> first_streamline = reader[0]
    >>> some_streamlines = reader[10:20]

    Note the streamline coordinates are returned in trackvis 'voxelmm' space.
    To convert them to RAS+ mm space, use the header's `trk2ras` affine:
    >>> points = some_streamlines.per_point['point']
    >>> trk2ras = reader.header.trk2ras
    >>> points_ras = trk2ras[:3, :3] @ points.T + trk2ras[:3, -1:]
    """

    def __init__(
        self,
        file: str | Path | BinaryIO,
        buffering: int = 0,
        keep_open: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        file : str | Path | BinaryIO
            Path to the TRK file or a binary file object.
        buffering : int, optional
            Buffering size for reading the file.
        keep_open : bool, default=False
            Whether to keep the file open after initialization.
        """
        # --- file i/o setup ---
        self.file: str | Path | BinaryIO = file
        self._fileobj: str | Path | BinaryIO = file
        self._mine: bool = not hasattr(self.file, 'read')
        self._buffering: int = buffering

        if hasattr(self.file, 'tell'):
            self._file_offset: int = self.file.tell()
        else:
            self._file_offset = 0

        if keep_open:
            self.open()

        # --- lazy properties ---
        self._header: TrkHeader | None = None
        self._offsets: np.ndarray | None = None

    # --- Lazy properties ----------------------------------------------

    @property
    def header(self) -> TrkHeader:
        """Return the TRK header."""
        if not getattr(self, '_header', None):
            self._load_header()
        return self._header

    @property
    def order(self) -> str:
        """Return the byte order of the TRK file {'<', '>'}."""
        return self.header.order

    @property
    def offsets(self) -> np.ndarray:
        """Return the offset to each streamline in file."""
        if getattr(self, '_offsets', None) is None:
            self._load_offsets()
        return self._offsets

    # --- Bounding boxes -----------------------------------------------

    def _bbox(
        self,
        trk2space: np.ndarray | None = None,
        chunk: int | None = 1024**2
    ) -> np.ndarray:
        chunk = chunk or len(self)
        nb_chunks = (len(self) + chunk - 1) // chunk

        min_corner = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        max_corner = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

        for i in range(nb_chunks):
            tracts = self[i * chunk:(i + 1) * chunk]
            points = tracts.per_point['point']

            if trk2space is not None:
                points = trk2space[:3, :3] @ points.T
                points += trk2space[:3, -1:]

            min_corner = np.minimum(min_corner, points.min(axis=0))
            max_corner = np.maximum(max_corner, points.max(axis=0))

        return np.array([min_corner, max_corner], dtype=np.float32)

    def bbox_trk(
        self,
        chunk: int | None = 1024**2
    ) -> np.ndarray:
        """Get bounding box of all streamlines in trackvis voxelmm space.

        Parameters
        ----------
        chunk : int | None, default=1024**2
            Number of streamlines to load at once to compute the bounding box.
            If None, load all streamlines at once.

        Returns
        -------
        bbox_trk : shape (2, 3) array
            Bounding box of all streamlines in trackvis voxelmm space.
            First row is the minimum corner, second row is the maximum corner.
        """
        return self._bbox(None, chunk)

    def bbox_ras(
        self,
        chunk: int | None = 1024**2
    ) -> np.ndarray:
        """Get bounding box of all streamlines in RAS+ mm space.

        Parameters
        ----------
        chunk : int | None, default=1024**2
            Number of streamlines to load at once to compute the bounding box.
            If None, load all streamlines at once.

        Returns
        -------
        bbox_ras : shape (2, 3) array
            Bounding box of all streamlines in RAS+ mm space.
            First row is the minimum corner, second row is the maximum corner.
        """
        return self._bbox(self.header.trk2ras, chunk)

    # --- File I/O methods ---------------------------------------------

    def open(self) -> BinaryIO:
        """Open the underlying file if it is not already open."""
        if not hasattr(self._fileobj, 'read'):
            self._fileobj = open(
                self._fileobj, 'rb',
                buffering=self._buffering
            )
        return self._fileobj

    def close(self) -> None:
        """Close the underlying file if it was opened by this reader."""
        if self._mine:
            self._fileobj.close()
            self._fileobj = self.file

    def smart_open(self) -> smart_open:
        """
        Return a smart file context manager for the TRK file.
        The file will be opened at the correct offset, and will only
        be closed if it was not already opened.
        """
        return smart_open(
            self._fileobj, "rb",
            buffering=self._buffering,
            offset=self._file_offset,
        )

    # --- Lazy loading methods -----------------------------------------

    def _load_header(self) -> None:
        """Load the header of a TRK file.

        Set Properties
        --------------
        header : np.ndarray[TRKHEADER_DTYPE]
            The loaded TRK header (byte-swapped if necessary).
        order : {'<', '>'}
            Whether the bytes need to be swapped.
        """
        with self.smart_open() as file:
            self._header = TrkHeader.from_file(file)

    def _load_offsets(self) -> None:
        """
        Load the offsets to each streamline in the TRK file.

        Set Properties
        --------------
        offsets : (N,) np.ndarray[np.int64]
            The offsets to each streamline in the TRK file.
        file : BinaryIO
            The binary file object.
        """
        header = self.header
        order = self.order

        nb_properties = header.n_properties
        nb_scalars = header.n_scalars
        nb_streamlines = header.n_count

        if nb_streamlines:
            offsets = np.zeros([nb_streamlines + 1], dtype=np.int64)
        else:
            offsets = []

        with self.smart_open() as file:
            # Move to the start of the streamlines
            file.seek(self.header.hdr_size, Whence.CURRENT)

            i = 0
            offset = 1000
            if nb_streamlines:
                offsets[0] = offset
            else:
                offsets.append(offset)

            while file and (nb_streamlines == 0 or i < nb_streamlines):
                nb_points = smart_read(file, f'{order}i4')
                streamline_size = (3 + nb_scalars) * nb_points * 4
                streamline_size += nb_properties * 4
                file.seek(streamline_size, Whence.CURRENT)
                streamline_size += 4
                offset += streamline_size
                i += 1

                if isinstance(offsets, list):
                    offsets.append(offset)
                else:
                    offsets[i] = offset

        if isinstance(offsets, list):
            offsets = np.array(offsets, dtype=np.int64)

        self._offsets = offsets

    def __len__(self) -> int:
        if self.header.n_count:
            return self.header.n_count
        return len(self.offsets) - 1

    def load_per_streamline(
        self, index: int | slice = slice(None)
    ) -> np.ndarray:
        """
        Load per-streamline data for a streamline or a set of streamlines
        by index or slice.

        Parameters
        ----------
        index : int | slice
            The index or slice of the streamline(s) to load.

        Returns
        -------
        np.ndarray
            The per-streamline data for the requested streamline(s).
        """
        header = self.header
        order = self.order
        offsets = self.offsets

        nb_streamlines = len(offsets) - 1
        nb_props = header.n_properties
        property_names = header.property_names
        prop_keys = ['prop_' + name for name in property_names]
        perline_dtype = per_streamline_dtype(property_names)

        # --- preproc streamline indices -------------------------------
        if isinstance(index, slice):
            if index.step not in (None, 1):
                raise ValueError('Slice step other than 1 is not supported.')
            index_start = 0 if index.start is None else index.start
            index_stop = nb_streamlines if index.stop is None else index.stop
        else:
            index_start = index
            index_stop = index + 1

        if index_start < 0:
            index_start += nb_streamlines
        if index_stop < 0:
            index_stop += nb_streamlines
        index_stop = min(index_stop, nb_streamlines)
        # --------------------------------------------------------------

        # --- allocate arrays ------------------------------------------
        nb_streamlines = index_stop - index_start
        per_streamline = np.zeros([nb_streamlines], dtype=perline_dtype)
        # --------------------------------------------------------------

        # --- load data ------------------------------------------------
        with self.smart_open() as file:
            file.seek(offsets[index_start], Whence.CURRENT)

            point_offset = 0
            for i in range(index_start, index_stop):
                j = i - index_start

                # read number of points in this streamline
                nb_points = smart_read(file, f'{order}i4')

                per_streamline[j]['id'] = i
                per_streamline[j]['length'] = nb_points
                per_streamline[j]['offset'] = point_offset

                # skip per-point data
                point_size = (3 + len(header.scalar_names)) * nb_points * 4
                file.seek(point_size, Whence.CURRENT)

                # read per-streamline properties
                if nb_props:
                    prop_values = smart_read(file, f'{order}f4', nb_props)
                    per_streamline[j][prop_keys] = prop_values

                point_offset += nb_points
        # --------------------------------------------------------------

        return per_streamline

    def load(
        self,
        index: int | slice = slice(None),
        load_points: bool = True
    ) -> Tractogram:
        """
        Load a streamline or a set of streamlines by index or slice.

        Parameters
        ----------
        index : int | slice
            The index or slice of the streamline(s) to load.
        load_points : bool, default=True
            Whether to load the per-point data.
            If False, only per-streamline data will be loaded.

        Returns
        -------
        Tractogram
            The requested streamline(s) as a Tractogram object.
        """
        if not load_points:
            per_streamline = self.load_per_streamline(index)
            return Tractogram(per_streamline, None)

        header = self.header
        order = self.order
        offsets = self.offsets

        nb_streamlines = len(offsets) - 1
        scalar_names = header.scalar_names
        property_names = header.property_names

        prop_keys = ['prop_' + name for name in property_names]
        scalar_keys = ['scalar_' + name for name in scalar_names]

        perline_dtype = per_streamline_dtype(property_names)
        perpoint_dtype = per_point_dtype(scalar_names)

        # --- preproc streamline indices -------------------------------
        if isinstance(index, slice):
            if index.step not in (None, 1):
                raise ValueError('Slice step other than 1 is not supported.')
            index_start = 0 if index.start is None else index.start
            index_stop = nb_streamlines if index.stop is None else index.stop
        else:
            index_start = index
            index_stop = index + 1

        if index_start < 0:
            index_start += nb_streamlines
        if index_stop < 0:
            index_stop += nb_streamlines
        index_stop = min(index_stop, nb_streamlines)
        # --------------------------------------------------------------

        offset_start = offsets[index_start]
        offset_stop = offsets[index_stop]

        # --- allocate arrays ------------------------------------------
        nb_streamlines = index_stop - index_start
        nb_points = (offset_stop - offset_start)
        nb_points -= nb_streamlines * 4 * (1 + len(property_names))
        nb_points //= (3 + len(scalar_names)) * 4

        per_streamline = np.zeros([nb_streamlines], dtype=perline_dtype)
        per_point = np.zeros([nb_points], dtype=perpoint_dtype)
        # --------------------------------------------------------------

        # --- load binary data -----------------------------------------
        with self.smart_open() as file:
            file.seek(offsets[index_start], Whence.CURRENT)
            buf = file.read(offset_stop - offset_start)
        buf = BytesIO(buf)
        # --------------------------------------------------------------

        point_start = 0
        for i in range(index_start, index_stop):
            j = i - index_start

            # read number of points in this streamline
            nb_points = smart_read(buf, f'{order}i4')
            point_stop = point_start + nb_points

            # read binary data for this streamline
            streamline_dtype = streamline_binary_dtype(
                nb_points,
                scalar_names,
                property_names,
                order
            )
            streamline_data = smart_read(buf, streamline_dtype)

            # save per-streamline data
            if prop_keys:
                per_streamline[j][prop_keys] = streamline_data[prop_keys]
            per_streamline[j]['id'] = i
            per_streamline[j]['length'] = nb_points
            per_streamline[j]['offset'] = point_start

            # save per-point data
            per_point[point_start:point_stop][
                ['point'] + scalar_keys
            ] = streamline_data['per_point']
            per_point[point_start:point_stop]['streamline'] = i

            point_start = point_stop

        return Tractogram(per_streamline, per_point)

    def __getitem__(self, index: int | slice) -> Tractogram:
        """
        Get a streamline or a set of streamlines by index or slice.

        Parameters
        ----------
        index : int | slice
            The index or slice of the streamline(s) to retrieve.

        Returns
        -------
        Tractogram
            The requested streamline(s) as a Tractogram object.
        """
        return self.load(index)
