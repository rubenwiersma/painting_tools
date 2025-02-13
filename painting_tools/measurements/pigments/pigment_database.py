from distutils.log import error
from os import listdir
import os.path as osp
import csv
from ...rendering.kubelka_munk import reflectance_to_ks

import numpy as np

class PigmentDatabase(object):

    def __init__(self, path):
        """
        Loads a database of pigment coefficients from a given directory.
        The databases are formatted as a tab-delimitted spreadsheets for a given set of pigments,
        where each pigment is given by a row starting with its name, followed by either:
        - W absorption / scatter ratios
        - W absoroption coefficients and W scatter coefficients,
        where W is the number of wavelengths in the database.
        """
        self.path = osp.join(osp.dirname(osp.realpath(__file__)), path)
        self.pigment_sets = [f.split('.')[0] for f in listdir(self.path) if osp.isfile(osp.join(self.path, f)) and '.rs' in f]
        spectra_file = osp.join(self.path, 'spectra.txt')
        try:
            self.spectra = np.loadtxt(spectra_file)
        except:
            raise Exception(f'Wavelengths could not be found. Make sure you have stored the wavelengths in {spectra_file}.')

    def __getitem__(self, key):
        return self.get_pigments(key)

    def get_pigments(self, set):
        if type(set) is int:
            set = self.pigment_sets[set]

        pigment_names = []
        pigments = []
        n_spectra = self.spectra.shape[0]
        with open(osp.join(self.path, set + '.rs'), newline='') as csvfile:
            rs_file = csv.reader(csvfile, delimiter='\t')
            # Each row of the file represents one pigment
            for pigment in rs_file:
                pigment_names.append(pigment[0])
                vals = np.array([float(val) for val in pigment[1:] if not val == ''])

                # Error-case: the number of given values is less than the number of wavelengths.
                # Pad with zeros.
                if vals.shape[0] < n_spectra:
                    vals = np.concatenate([vals, np.zeros(n_spectra - vals.shape[0])])

                # Expected case: the number of given values is larger than the number of wavelengths.
                # In this case, we assume that both the absorption and scattering coefficients are given.
                if vals.shape[0] > n_spectra:
                    a = vals[:n_spectra]
                    s = vals[n_spectra:]
                else:
                    a = reflectance_to_ks(vals)
                    s = np.ones_like(vals)
                pigments.append(np.stack([self.spectra, a, s], axis=1))

        return pigments, pigment_names