import MDAnalysis as mda
import numpy as np
import time

import struct
import io
import os


class Timer(list):
    def __call__(self, msg=None, nopar=False):

        # With a message, run as context
        if msg is not None:
            self.append([msg, time.time()])
            return self

        # Without a message, run as decorator
        def inner(func):
            run = [0]
            def wrapper(*args, **kwargs):
                run.append(run[-1] + 1)
                msg = f'{func.__name__}[{run[-1]}]'
                msg += ' ...' if nopar else f'{args} {kwargs}'
                with self(msg=msg):
                    func(*args, **kwargs)
            return wrapper
        return inner

    def __repr__(self):
        out = ''.join(f'{lbl}: {e-s:.2f}s\n' for lbl, s, e in self)
        self.clear()
        return out

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_tb):
        self[-1].append(time.time())


# XTC:
#    What                 Bytes Format
#  0. Magic number (4)    0 -  4 l
#  1. Atoms (4)           4 -  8 l
#  2. Step  (4)           8 - 12 l 
#  3. Time (4)           12 - 16 f
#  4. Box (9*4)          16 - 52 fffffffff
# 13. Atoms (4)          52 - 56 l           (checked to be equal to b.)
# 14. Precision (4)      56 - 60 f
# 15. Extent (6*4)       60 - 84 llllll      (MIN: x,y,z, MAX: x,y,z)
# 21. smallidx (4)       84 - 88 l
# 22. Size in bytes (4)  88 - 92 l 
# 23. Coordinates (compressed)
class XTCIndexer(io.FileIO):
    def __init__(self, filename):
        super().__init__(filename, 'rb')
        self.headers = []
        self.tag = self.read(8)
        self.size = self.seek(0, 2)
        self.n_atoms = struct.unpack('>l', self.tag[4:])[0]
        self.positions = [ self.seek(88) - 88 ]
        self.pos = 0
        while self.positions[-1] < self.size:
            self.nextframe()

    def __len__(self):
        return len(self.positions) - 1

    def nextframe(self):
        fsize = struct.unpack('>l', self.read(4))[0]
        fsize += -fsize % 4
        self.positions.append(self.seek(fsize + 88, 1) - 88)

    def write(self, filename, start=0, stop=None, step=None):
        if isinstance(start, int):
            start = range(len(self.positions))[start:stop:step]
        with open(filename, 'wb') as out:
            for f in start:
                self.seek(self.positions[f])
                out.write(self.read(self.positions[f+1] - self.positions[f]))
            size = out.tell()

        np.savez(
            f'.{filename}_offsets.npz',
            offsets=self.positions[:-1],
            size=size,
            ctime=os.path.getctime(filename),
            n_atoms=self.n_atoms
        )


def dim2pbc(arr: np.ndarray) -> np.ndarray:
    '''
    Convert unit cell definition from PDB CRYST1 format to lattice definition.
    '''

    lengths = arr[:, :3]
    angles = arr[:, 3:] * (np.pi / 180)

    cosa = np.cos(angles)
    sing = np.sin(angles[:, 2])

    pbc = np.zeros((len(arr), 9))
    pbc[:, 0] = lengths[:, 0]
    pbc[:, 3] = lengths[:, 1] * cosa[:, 2]
    pbc[:, 4] = lengths[:, 1] * sing
    pbc[:, 6] = lengths[:, 2] * cosa[:, 1]
    pbc[:, 7] = lengths[:, 2] * (cosa[:, 0] - cosa[:, 1] * cosa[:, 2]) / sing
    pbc[:, 8] = (lengths[:, 2] ** 2 - (pbc[:, 6:8] ** 2).sum(axis=1)) ** 0.5

    return pbc.reshape((-1, 3, 3))


class TrajectoryWithPBC(np.ndarray):
    '''
    A simple class to contain a trajectory with the coordinates and 
    unit cell definitions for all frames read in as numpy arrays.
    '''
    attributes = (
        'universe', # The mda.Universe from the selected atoms
        'times', # The times of the frames
        'pbc', # The PBC lattice matrices for all frames
        'centers', # The centers
        'orientations', # The orientations
        'rgyr_' # Radii of gyration (set by align)
        'rmsd_' # Root mean square deviation (set by align)
    )

    def __new__(cls, tpr: str, trj: str, selection=None, start=None, stop=None, step=None):

        # Bookkeeping: MDA stuff
        if selection is None:
            selection = 'all'
        try:
            atomgroup = mda.Universe(tpr, trj).select_atoms(selection)
        except OSError:
            raise OSError(f'XDR read error in {tpr}:{trj}')

        # Content: times, pbc, coordinates
        pbc, times, coords = [], [], []
        for frame in atomgroup.universe.trajectory[start:stop:step]:
            pbc.append(atomgroup.dimensions.copy())
            coords.append(atomgroup.positions.copy())
            times.append(frame.time)

        # The object
        obj = np.array(coords).view(cls)
        obj.pbc = dim2pbc(np.array(pbc))
        obj.times = np.array(times)
        obj.universe = mda.Merge(atomgroup)
        obj.centers = np.zeros((len(coords), 3))
        obj.orientations = np.outer(np.ones(len(coords)), np.eye(3)).reshape((-1, 3, 3))

        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        for attr in self.attributes:
            setattr(self, attr, getattr(obj, attr, None))

    def __matmul__(self, other):
        trj = (self.view(np.ndarray) @ other).view(self.__class__)
        self._transfer_attributes(trj)
        trj.pbc = self.pbc @ other
        return trj

    def __getitem__(self, item):
        if item is None:
            # Override the behaviour of adding an axis
            # to allow NoneType selections
            return self
        if isinstance(item, (int, slice)):
            result = super().__getitem__(item)
            self._transfer_attributes(result)
            result.pbc = self.pbc[item]
            return result
        if isinstance(item, str):
            item = self.universe.select_atoms(item)
        if isinstance(item, mda.AtomGroup):
            result = self[:, item.ix]
            result.universe = mda.Merge(result.universe.atoms[item.ix])
            return result
        return super().__getitem__(item)

    def __and__(self, selection):
        '''Return the trajectory for the intersection of the atomgroup and selection'''
        return self[:, self.universe.select_atoms('selection').ix]

    def _transfer_attributes(self, other):
        for attr in self.attributes:
            setattr(other, attr, getattr(self, attr, None))

    @property
    def coords(self):
        return self.view(np.ndarray)

    @property
    def bcoords(self):
        # Integers indicate cells, fractions positions in cells
        result = self @ np.linalg.inv(self.pbc)
        result.pbc = self.pbc
        return result

    @property
    def angles(self):
        result = self @ (2 * np.pi * np.linalg.inv(self.pbc))
        result.pbc = self.pbc
        return result

    @property
    def skewed(self):
        result = self @ self.pbc
        result.pbc = self.pbc
        return result

    @property
    def angskewed(self):
        result = self @ (0.5 / np.pi * self.pbc)
        result.pbc = self.pbc
        return result

    def align(self, selection=None, reference=None, plane=None):
        '''Align the trajectory with respect to reference and selection'''
        # PBC safe centering
        self.origin(selection)

        fit = self[selection].coords
        natoms = len(fit[0])
        if reference is None:
            reference = fit[0]

        # Radii of gyration (for RMSD)
        rgyr2 = (fit ** 2).sum(axis=(1, 2)) / natoms
        refrg2 = (reference ** 2).sum() / natoms

        # Actual fitting
        U, L, V = np.linalg.svd(reference.T @ fit)
        R = U @ V
        result = self @ R.transpose((0, 2, 1))

        # Bookkeeping
        result.orientations = R
        rmsd2 = rgyr2 + refrg2 - 2 * L.sum(axis=1) / natoms
        rmsd2[rmsd2 < 0] = 0
        result.rmsd_ = rmsd2 ** 0.5
        result.rgyr_ = rgyr2 ** 0.5

        return result

    def alignxy(self, selection=None, reference=None):
        '''Align the trajectory with respect to reference and selection'''
        # PBC safe centering
        self.origin(selection)

        fit = self[selection].coords[:, :, :2]
        natoms = len(fit[0])
        if reference is None:
            reference = fit[0]

        # Actual fitting
        U, L, V = np.linalg.svd(reference[:, :2].T @ fit)
        R = np.zeros_like(self.pbc)
        R[:, :2, :2] = U @ V
        R[:, 2, 2] = 1
        result = self @ R.transpose((0, 2, 1))

        # Bookkeeping
        result.orientations = R
        return result

    def origin(self, selection=None):
        '''Center the selection at the origin for all frames (PBC safe)'''
        if selection is None:
            selection = 'all'
        # Use box coordinate angles to unambiguously define center of mass
        angles = self[selection].angles
        cosa, sina = np.cos(angles), np.sin(angles)
        centers = np.arctan2(sina.mean(axis=1), cosa.mean(axis=1))[:, None] @ (0.5 / np.pi * self.pbc)
        self.centers = centers
        self -= centers
        return self

    def center(self, selection=None):
        '''Center the selection in the center of the triclinic cell (PBC safe'''
        return self.origin(selection) + 0.5 * self.pbc.sum(axis=1)[:, None]

    def inbox(self):
        '''Put all particles in triclinic unit cell'''
        boxed = self @ np.linalg.inv(self.pbc)
        boxed = (boxed - np.floor(boxed)) @ self.pbc
        boxed.pbc = self.pbc
        return boxed

    def originbox(self):
        '''Put all particles in triclinic unit cell around origin'''
        boxed = self @ np.linalg.inv(self.pbc)
        boxed = (boxed - np.floor(boxed + 0.5)) @ self.pbc
        boxed.pbc = self.pbc
        return boxed

    def compact(self, around=None):
        if around is not None:
            self.origin(around)
        B = self.bcoords.reshape((-1, 3))
        # Shift origin to middle of box
        B += 0.5
        B -= np.floor(B)
        x, y = B[:, :2].coords.T
        check = (y < 0.5 - x) | (y > 1.5 - x)
        up = y > x
        halfx = 0.5 * x
        twox = 2 * x
        B[check &  up & (y > 1.25 - halfx), 1] -= 1
        B[check & ~up & (y < 0.25 - halfx), 1] += 1
        B[check &  up & (y < 0.5 - twox), 0] += 1
        B[check & ~up & (y > 2.5 - twox), 0] -= 1
        # Shift middle of hexagon to origin and transform
        B = ((B.reshape(self.shape) - 0.5) @ self.pbc).view(self.__class__)
        self._transfer_attributes(B)
        return B

    def molbox(self, selection):
        # This may be expensive...
        # Voxelized will be faster
        ...

    def split(self, what):
        '''Split the trajectory according to mda.atomgroup.split'''
        return [ self[ag] for ag in self.universe.atoms.split(what) ]


class PBCHexagonalGrid(np.ndarray):
    '''
    A near hexagonal grid with PBC in two dimensions
    '''
    attributes = (
        'pbc', # The PBC lattice matrices for all frames
        'avg', # The mean lattice
        'unit', # The unit cells for all frames
        'nbox', # The lattice in units (equal for all frames)
        'z', # The height
    )

    def __new__(cls, pbc, resolution=1):
        # Determine the hixel dimensions and number
        unit = resolution * np.array(((1, 0), (0.5, 0.75 ** 0.5)))
        mean = pbc.mean(axis=0)[:2, :2]
        nbox = (np.linalg.inv(unit) @ mean).astype(int)
        # This cell is the closest to hexagonal that fits the lattice
        # for the specified resolution
        unit = np.linalg.inv(nbox) @ pbc[:, :2, :2]
        # This is the grid that fits the lattice
        grid = np.mgrid[:nbox[0, 0], :nbox[1, 1]].T.reshape((-1, 2)).view(cls)
        grid.pbc = pbc
        grid.avg = mean
        grid.nbox = nbox
        grid.unit = unit
        grid.z = np.zeros_like(grid[0])
        return grid

    def __array_finalize__(self, obj):
        if obj is None:
            return None
        for attr in self.attributes:
            setattr(self, attr, getattr(obj, attr, None))

    @property
    def points(self):
        return self @ (np.linalg.inv(self.nbox) @ self.avg)

    def bin(self, trj):
        '''Set z values to ...'''
        ...

    def kde(self, trj, prop=None, bw=0.01):
        '''Set z values to distance weighted sum of property'''
        # The grid in box coordinates
        G = self @ np.linalg.inv(self.nbox)
        out = np.zeros((len(trj), len(G)))
        # Distance vectors in box coordinates
        for idx, frame in enumerate(trj.bcoords.coords):
            frame = frame[:, None, :2] - G[None, :]
            frame = (frame - np.floor(frame + 0.5)) @ G.avg
            # Weights per atom (frameatoms, gridpoints)
            W = np.exp(-(frame ** 2).sum(axis=2) / bw)
            if prop is None:
                out[idx] = W.sum(axis=0)
            elif isinstance(prop, np.ndarray) and len(prop.shape) == 2:
                out[idx] = (W * prop[idx]).sum(axis=0) / W.sum(axis=0)
            else:
                out[idx] = (W * prop).sum(axis=0) / W.sum(axis=0)
        return out
