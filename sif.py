"""Tooling for computing SIF and applying corrections"""


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