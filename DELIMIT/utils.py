import numpy as np


# helper functions
def cart2sph(x, y, z):
    """
    cart2sph(x, y, z) -> theta, phi, r

    Computes the corresponding spherical coordinate of the given input parameters :attr:`x`, :attr:`y` and :attr:`x`.

    Args:
        x (Number): x position
        y (Number): y position
        z (Number): z position

    Example::

        >>> cart2sph(1, 1, 1)
        (0.78539816339744828, 0.95531661812450919, 1.7320508075688772)
    """
    azimuthal_angle = np.arctan2(y, x)
    radial_distance = np.sqrt(x**2 + y**2 + z**2)
    polar_angle = np.arccos(z/radial_distance)
    return azimuthal_angle, polar_angle, radial_distance


def sph2cart(azimuthal_angle, polar_angle, radial_distance):
    """
    sph2cart(polar_angle, azimuthal_angle, radial_distance) -> x, y, z

    Computes the corresponding cartesian coordinate of the given input parameters :attr:`polar_angle`,
    :attr:`azimuthal_angle` and :attr:`radial_distance`.

    Args:
        polar_angle (Number): polar_angle
        azimuthal_angle (Number): azimuthal_angle
        radial_distance (Number): radial_distance

    Example::

        >>> cart2sph(0.78539816339744828, 0.95531661812450919, 1.7320508075688772)
        (0.99999999999999978, 0.99999999999999967, 1.0)
    """
    x = radial_distance * np.sin(polar_angle) * np.cos(azimuthal_angle)
    y = radial_distance * np.sin(polar_angle) * np.sin(azimuthal_angle)
    z = radial_distance * np.cos(polar_angle)
    return x, y, z