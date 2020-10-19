#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.scf import khf,kuhf

L = 2.
He = pbcgto.Cell()
He.verbose = 0
He.a = np.eye(3)*L
He.atom =[['He' , ( L/2+0., L/2+0., L/2+0.)],]
He.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
He.build()

def make_primitive_cell(mesh):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'Si 0.,  0.,  0.; Si 1.3467560987,  1.3467560987,  1.3467560987'
    cell.a = '''0.            2.6935121974    2.6935121974
                2.6935121974  0.              2.6935121974
                2.6935121974  2.6935121974    0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = mesh
    cell.spin = 0
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    return cell

cell = make_primitive_cell([17]*3)
nk = [1,2,2]

def tearDownModule():
    global cell, He, nk
    del cell, He, nk

class KnownValues(unittest.TestCase):
    def test_krhf_gamma_center(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=True)
        kmf0 = khf.KRHF(cell, kpts=kpts0).run()

        kpts = cell.make_kpts(nk, with_gamma_point=True,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = khf.KRHF(cell, kpts=kpts).run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_krhf_monkhorst(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=False)
        kmf0 = khf.KRHF(cell, kpts=kpts0).run()

        kpts = cell.make_kpts(nk, with_gamma_point=False,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = khf.KRHF(cell, kpts=kpts).run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_kuhf_gamma_center(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=True)
        kmf0 = kuhf.KUHF(cell, kpts=kpts0)
        kmf0 = pscf.addons.smearing_(kmf0, sigma=0.001, method='fermi',fix_spin=True)
        kmf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=True,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = kuhf.KUHF(cell, kpts=kpts)
        kumf = pscf.addons.smearing_(kumf, sigma=0.001, method='fermi',fix_spin=True)
        kumf.kernel()
        self.assertAlmostEqual(kumf.e_tot, kmf0.e_tot, 7)

    def test_kuhf_monkhorst(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=False)
        kmf0 = kuhf.KUHF(cell, kpts=kpts0)
        kmf0 = pscf.addons.smearing_(kmf0, sigma=0.001, method='fermi',fix_spin=True)
        kmf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=False,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = kuhf.KUHF(cell, kpts=kpts)
        kumf = pscf.addons.smearing_(kumf, sigma=0.001, method='fermi',fix_spin=True)
        kumf.kernel()
        self.assertAlmostEqual(kumf.e_tot, kmf0.e_tot, 7)

    def test_krhf_df(self):
        kpts0 = He.make_kpts(nk)
        kmf0 = khf.KRHF(He, kpts=kpts0).density_fit().run()
        
        kpts = He.make_kpts(nk, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = khf.KRHF(He, kpts=kpts).density_fit().run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_krhf_mdf(self):
        kpts0 = He.make_kpts(nk)
        kmf0 = khf.KRHF(He, kpts=kpts0).mix_density_fit().run()

        kpts = He.make_kpts(nk, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = khf.KRHF(He, kpts=kpts).mix_density_fit().run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

if __name__ == '__main__':
    print("Full Tests for HF with k-point symmetry")
    unittest.main()