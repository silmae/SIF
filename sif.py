"""Tooling for computing SIF and applying corrections"""

import pandas as pd
import numpy as np
import os.path as osp


FOREST = {
    'latitude_deg': 39.126650,
    'longitude_deg': -77.221930,
    'elevation': 115,
}
"""Data for the FOREST test site for sun angle calculation as guesstimated
from Google Earth.
"""


def SIF_O2A(df):
    """Compute SIF in the O2-A absorption band.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with radiances in columns L_757 and L_760
        and corresponding irradiances in E_757 and E_760.
    """

    cols = ['L_757', 'L_760', 'E_757', 'E_760']
    return _SIF_O2A(**df[cols])


def SIF_O2B(df):
    """Compute SIF in the O2-B absorption band.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with radiances in columns L_685 and L_687
        and corresponding irradiances in E_685 and E_687.
    
    """
    cols = ['L_685', 'L_687', 'E_685', 'E_687']
    return _SIF_O2B(**df[cols])


def _SIF_O2A(*, L_757, L_760, E_757, E_760):
    """Compute SIF in the O2-A absorption band.
    
    Parameters
    ----------
    L_757 : array-like
        Radiances at 757nm
    L_760 : array-like
        Radiances at 760nm
    E_757 : array-like
        Solar irradiances at 757nm
    E_760 : array-like
        Solar irradiances at 760nm
    """
    return FLD(
        L_in=L_760,
        L_out=L_757,
        E_in=E_760,
        E_out=E_757,
        )


def _SIF_O2B(*, L_685, L_687, E_685, E_687):
    """Compute SIF in the O2-B absorption band.
    
    Parameters
    ----------
    L_685 : array-like
        Radiances at 685nm
    L_687 : array-like
        Radiances at 687nm
    E_685 : array-like
        Solar irradiances at 685nm
    E_687 : array-like
        Solar irradiances at 687nm
    
    """
    return FLD(
        L_in=L_685,
        L_out=L_687,
        E_in=E_685,
        E_out=E_687
        )


def FLD(*, L_in, L_out, E_in, E_out):
    """Fraunhofer line depth method of SIF estimation.
    
    Parameters
    ----------
    L_in : array-like
        Radiance at the bottom of the well.
    L_out : array-like
        Radiance at the shoulder of the well.
    E_in : array-like
        Irradiance at the bottom of the well.
    E_out : array-like
        Irradiance at the shoulder of the well.
    
    """
    return (E_out * L_in - L_out * E_in) / (E_out - E_in)


def cosine(n, sun):
    """Calculate the cosine correction given the surface and sun angles.

    Parameters
    ----------
    n : array-like
        N x 2 array of surface azimuths and altitudes.
    sun : array-like
        M x 2 array of sun azimuths and altitudes.
    
    Returns
    -------
    result : array-like
        N x M Array of cosines between the given vectors
    """

    nv = pos_to_vec(n)
    sunv = pos_to_vec(sun)

    return cosine_vec(nv, sunv)


def cosine_vec(n, sun):
    """Calculate the cosine correction given a normal and sun direction
    as vectors.

    Parameters
    ----------
    n : array-like
        N x 3 array of (unit) vectors giving the surface normals.
    sun : array-like
        M x 3 array of (unit) vectors giving the position of the sun.
    
    Returns
    -------
    result : array-like
        N x M Array of cosines between the given vectors
    """
    return (np.dot(n, sun.T) / (
        np.linalg.norm(n, 2, axis=1) * np.linalg.norm(sun, 2, axis=1)
        )).T


def read_SIF_data(path, tz='US/Eastern'):
    """Read data from a CSV file and compute the UTC timestamps
    
    Parameters
    ----------
    path : str
        Full path to the CSV file.
    tz : str, optional
        Timezone of the timestamps in the file. 
        Default is US/Eastern.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the times converted to UTC from US/Eastern
    """

    df = pd.read_csv(path)
    date = _extract_date(osp.basename(path))

    df['Time'] = pd.to_datetime(date) + pd.to_timedelta(df['Time'], 'h')
    df = df.set_index('Time')
    df = df.tz_localize(tz)
    df = df.tz_convert('UTC')
    return df


def _extract_date(filename):
    """Parse date from filenames

    Given a filename matching pattern

        '*_MMDDYY_*.*',

    returns the string 'MMDDYY'.

    """
    s = osp.splitext(filename)[0]
    return s.split('_')[1]


def sun_positions(location_info, utc_times):
    """Get sun positions (azimuth, altitude) given a place and set of times
    
    Parameters
    ----------
    location_info : dict-like
        Dictionary with the keys
            'latitude_deg',
            'longitude_deg',
            and 'elevation' (optional, in meters)
    utc_times : list-like
        List of UTC datetime objects to get sun positions for.
    
    Returns
    -------
    numpy.ndarray
        N x 2 array of sun azimuths and altitudes in degrees.
    """

    from pysolar.solar import get_position

    return np.array(
        [get_position(**location_info, when=t) for t in utc_times]
        )


def pos_to_vec(pos):
    """Compute a direction vector given the sun position in degrees.
    
    Input should be azimuth (degrees clockwise from North) and altitude
    (degrees above horizon).

    The resulting vectors will be [X, Y, Z], with
        X increasing East,
        Y increasing North,
    and Z increasing up (from the horizon level).

    Parameters
    ----------
    pos : array-like
        N x 2 array of pairs of azimuth, altitude values in degrees.
    
    Returns
    -------
    numpy.ndarray
        N x 3 array of unit vectors pointing towards the sun.

    """
    rad = np.deg2rad(pos)
    res = np.stack([
        np.sin(rad[:, 0]),
        np.cos(rad[:, 0]),
        np.sin(rad[:, 1])
        ], axis=0)
    return (res / np.linalg.norm(res, 2, axis=0)).T


def plot_sun(posvecs):
    """Plot a 3D quiver plot of the sun position.
    
    Parameters
    ----------
    posvecs : array-like
        Sun position vectors as given by pos_to_vec
    
    Returns
    -------
    fig, ax
        Figure and axis handles to the created plot
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X, Y, Z = zip(*np.zeros_like(posvecs))
    U, V, W = zip(*posvecs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver3D(X, Y, Z, U, V, W)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return fig, ax


def apply_cosine(df, location, leaf):
    df = df.copy()
    df['cosine'] = cosine(leaf, sun_positions(location, df.index))
    irs = [c for c in df.columns if 'L_' in c]

    for c in irs:
        tmp = df['cosine'] * df[c]
        df[c] = tmp

    return df


def compute_SIFs(df):
    df = df.copy()
    df['SIF_O2A'] = SIF_O2A(df)
    df['SIF_O2B'] = SIF_O2B(df)
    return df