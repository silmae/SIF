"""Tooling for computing SIF and applying corrections"""

import pandas as pd
import numpy as np
import os.path as osp


def SIF_O2A(*, L_757, L_760, E_757, E_760):
    """Compute SIF in the O2-A absorption band.
    
    Parameters
    ----------
    L_757 : array-like
        Radiances at 685nm
    L_760 : array-like
        Radiances at 687nm
    E_757 : array-like
        Solar irradiances at 685nm
    E_760 : array-like
        Solar irradiances at 687nm
    """
    return FLD(
        L_in=L_760,
        L_out=L_757,
        E_in=E_760,
        E_out=E_757,
        )


def SIF_O2B(*, L_685, L_687, E_685, E_687):
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
    """Calculate the cosine correction given a normal and sun direction.

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
    return np.dot(n, sun.T) / (np.sum(n, axis=1) * np.sum(sun, axis=1))


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

