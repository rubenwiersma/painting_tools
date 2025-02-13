from pathlib import Path
import numpy as np
import spectral.io.envi as envi


def read_capture(capture_folder):
    """Read the calibrated reflectance,
    raw image, dark reference and white reference for a given capture folder.

    Args:
        capture_folder (pathlib.Path): Path to the capture folder.

    Returns:
        tuple: Tuple containing:
            - reflectance (numpy.ndarray): Calibrated reflectance.
            - img (numpy.ndarray): Raw image.
            - darkref (numpy.ndarray): Dark reference.
            - whiteref (numpy.ndarray): White reference.
            - wavelengths (numpy.ndarray): Wavelengths.
    """
    capture_id = capture_folder.name
    
    reflectance_path = capture_folder / 'results' / f'REFLECTANCE_{capture_id}'
    reflectance = envi.open(str(reflectance_path) + '.hdr', str(reflectance_path) + '.dat')

    img_path = capture_folder / 'capture' / capture_id
    img = envi.open(str(img_path) + '.hdr', str(img_path) + '.raw')

    darkref_path = capture_folder / 'capture' / f'DARKREF_{capture_id}'
    darkref = envi.open(str(darkref_path) + '.hdr', str(darkref_path) + '.raw')

    whiteref_path = capture_folder / 'capture' / f'WHITEREF_{capture_id}'
    whiteref = envi.open(str(whiteref_path) + '.hdr', str(whiteref_path) + '.raw')
    return reflectance.asarray(), img.asarray(), darkref.asarray(), whiteref.asarray(), reflectance.metadata


def read_and_calibrate(raw, reference, root_folder, rot90=0):
    """Read and calibrate the reflectance for a given capture from
    the raw data and calibrated reference.

    Args:
        raw (str): Name of the raw capture folder.
            This is typically a number. If None, only the calibrated
            reflectance is returned.
        reference (str): Name of the reference capture folder.
        root_folder (pathlib.Path): Path to the folder where
            the capture folders are located.
        rot90 (int, optional): Number of times to rotate the
            reflectance image 90 degrees counter-clockwise.
            Defaults to 0.
    
    Returns:
        tuple: Tuple containing:
            - reflectance (numpy.ndarray): Calibrated reflectance.
            - wavelengths (numpy.ndarray): Wavelengths.
    """
    if raw and isinstance(raw, int):
        raw = str(raw)
    if isinstance(reference, int):
        reference = str(reference)
    if isinstance(root_folder, str):
        root_folder = Path(root_folder)
    
    reflectance, _, darkref_reference, whiteref_reference, metadata = read_capture(root_folder / reference)
    if raw is not None:
        _, img, darkref, _, metadata = read_capture(root_folder / raw)
        reflectance = ((img - darkref) / (whiteref_reference - darkref_reference)).astype(np.float32)

    reflectance = np.rot90(reflectance, -1 + rot90)
    return reflectance, metadata