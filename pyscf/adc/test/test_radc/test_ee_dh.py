# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Author: Terrence Stahl <terrencestahl1@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
DH-ADC(2) (double-hybrid ADC(2), Mester & Kallay, JCTC 2019, 15, 4440):
the CIS part of the ADC(2) matrix is replaced by the TDA singles block of a
double-hybrid functional and the second-order terms are scaled by alpha_C.
'''
import unittest
import numpy as np
import math
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import adc
from pyscf.tdscf import rhf as tdscf_rhf

def setUpModule():
    global mol, mf_hf, mf_pbe02, mf_rhf
    mol = gto.Mole()
    r = 0.957492
    x = r * math.sin(104.468205 * math.pi/(2 * 180.0))
    y = r * math.cos(104.468205* math.pi/(2 * 180.0))
    mol.atom = [
        ['O', (0., 0.    , 0)],
        ['H', (0., -x, y)],
        ['H', (0., x , y)],]
    mol.basis = {'H': 'cc-pVDZ',
                 'O': 'cc-pVDZ',}
    mol.verbose = 0
    mol.build()

    # Pure-HF Kohn-Sham reference (A^DH reduces to the CIS block)
    mf_hf = dft.RKS(mol, xc='HF')
    mf_hf.conv_tol = 1e-11
    mf_hf.kernel()

    # PBE0-2 double-hybrid SCF part: 79.3701% HF exchange, 20.6299% PBE
    # exchange, 50% PBE correlation (alpha_X = 0.793701, alpha_C = 1/2)
    mf_pbe02 = dft.RKS(mol, xc='0.793701*HF + 0.206299*PBE, 0.5*PBE')
    mf_pbe02.conv_tol = 1e-11
    mf_pbe02.kernel()

    mf_rhf = scf.RHF(mol)
    mf_rhf.conv_tol = 1e-11
    mf_rhf.kernel()

def tearDownModule():
    global mol, mf_hf, mf_pbe02, mf_rhf
    del mol, mf_hf, mf_pbe02, mf_rhf

class KnownValues(unittest.TestCase):

    def test_limit_cis(self):
        # DH-ADC(2) with a pure-HF functional and alpha_C = 0 reduces to CIS
        # (the A^[2] block and the 2p2h coupling vanish).
        myadc = adc.ADC(mf_hf)
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.dh = True
        myadc.alpha_c = 0.0
        e, v, p, x = myadc.kernel(nroots=3)
        self.assertTrue(p is None and x is None)

        tda = tdscf_rhf.TDA(mf_hf)
        tda.verbose = 0
        e_tda = tda.kernel(nstates=3)[0]
        np.testing.assert_allclose(e, e_tda, atol=1e-9)

    def test_limit_tda(self):
        # DH-ADC(2) with alpha_C = 0 equals the TDA of the double-hybrid
        # functional (the singles block is exactly the TDDFT A matrix).
        myadc = adc.ADC(mf_pbe02)
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.dh = True
        myadc.alpha_c = 0.0
        e, v, p, x = myadc.kernel(nroots=3)

        tda = tdscf_rhf.TDA(mf_pbe02)
        tda.verbose = 0
        e_tda = tda.kernel(nstates=3)[0]
        np.testing.assert_allclose(e, e_tda, atol=1e-9)

    def test_limit_adc2(self):
        # DH-ADC(2) with a pure-HF functional and alpha_C = 1 reduces to the
        # standard ADC(2) (the PBE0-2/HF reference equals the RHF one).
        myadc = adc.ADC(mf_hf)
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.dh = True
        myadc.alpha_c = 1.0
        e, v, p, x = myadc.kernel(nroots=3)

        myadc2 = adc.ADC(mf_rhf)
        myadc2.method = "adc(2)"
        myadc2.method_type = "ee"
        e_std, v_std, p_std, x_std = myadc2.kernel(nroots=3)

        np.testing.assert_allclose(e, e_std, atol=1e-8)

    def test_pbe02_h2o(self):
        # PBE0-2 DH-ADC(2) on water (alpha_X = alpha_C = 1/2)
        myadc = adc.ADC(mf_pbe02)
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.dh = True
        e_corr, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(myadc.alpha_c, 0.5, 10)
        self.assertAlmostEqual(e_corr, -0.10919060780, 7)

        e, v, p, x = myadc.kernel(nroots=4)
        self.assertTrue(p is None and x is None)
        np.testing.assert_allclose(
            e, [0.30755923, 0.38002560, 0.40193849, 0.47676229], atol=1e-8)

    def test_frozen(self):
        # Freezing the core orbital keeps the method well-defined
        myadc = adc.ADC(mf_pbe02, frozen=1)
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.dh = True
        e_corr, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e_corr, -0.10798592347, 7)

        e, v, p, x = myadc.kernel(nroots=3)
        np.testing.assert_allclose(
            e, [0.30755492, 0.38001558, 0.40196913], atol=1e-8)

    def test_reference_gating(self):
        # Standard ADC(2) with a Kohn-Sham reference is not supported
        myadc = adc.ADC(mf_pbe02)
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        with self.assertRaises(NotImplementedError):
            myadc.kernel(nroots=1)

        # DH-ADC(2) requires a Kohn-Sham reference
        myadc = adc.ADC(mf_rhf)
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.dh = True
        with self.assertRaises(NotImplementedError):
            myadc.kernel(nroots=1)

        # DH-ADC(2) is only defined for EE excitations
        myadc = adc.ADC(mf_pbe02)
        myadc.method = "adc(2)"
        myadc.method_type = "ip"
        myadc.dh = True
        with self.assertRaises(NotImplementedError):
            myadc.kernel(nroots=1)

    def test_singles_operator_equivalence(self):
        # The A^DH operator (TDDFT response) plus alpha_C*A^[2] must equal the
        # explicit matrix form A^DH + alpha_C*A^[2] for arbitrary vectors.
        from pyscf.adc import radc_ee
        myadc = adc.ADC(mf_pbe02)
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.dh = True
        myadc.alpha_c = 0.3
        myadc.compute_properties = False
        myadc.kernel_gs()

        n_singles = myadc._nocc * myadc._nvir
        a2 = radc_ee.get_imds(myadc)
        a_dh = radc_ee.get_a_dh(myadc)
        vind_dh, hdiag = radc_ee.get_vind_dh(myadc)
        np.testing.assert_allclose(
            hdiag,
            (myadc.mo_energy[myadc._nocc:]
             - myadc.mo_energy[:myadc._nocc, None]).ravel(),
            atol=1e-12)

        rng = np.random.default_rng(7)
        for _ in range(3):
            x = rng.standard_normal(n_singles)
            op = vind_dh(x.reshape(myadc._nocc, myadc._nvir)).ravel() \
                + myadc.alpha_c * (a2 @ x)
            mat = (a_dh + myadc.alpha_c * a2) @ x
            np.testing.assert_allclose(op, mat, atol=1e-9)

    def test_ethene_paper(self):
        # Real-world validation against the benchmark state of Mester &
        # Kallay (JCTC 2019, 15, 4440): ethene 1 1B2u (pi-pi*) at the Thiel
        # set geometry.  The paper uses the TZVP basis of the Thiel set; the
        # cc-pVTZ run here reproduces the published 8.38 eV to 0.009 eV
        # (8.389 eV, the first singlet root).
        mol_et = gto.M(
            atom='H 0.000000 0.923274 1.238289; H 0.000000 -0.923274 1.238289;'
                 'H 0.000000 0.923274 -1.238289; H 0.000000 -0.923274 -1.238289;'
                 'C 0.000000 0.000000 0.668188; C 0.000000 0.000000 -0.668188',
            basis='cc-pVTZ', verbose=0)
        mf = dft.RKS(mol_et, xc='0.793701*HF + 0.206299*PBE, 0.5*PBE')
        mf.conv_tol = 1e-11
        mf.kernel()
        myadc = adc.ADC(mf)
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.dh = True
        # Request more roots than asserted: the shared-ADC Davidson guess
        # heuristic occasionally skips low-lying roots for small nroots
        # (affects stock ADC(2) identically).
        e, v, p, x = myadc.kernel(nroots=4)
        e_ev = np.asarray(e) * 27.211386245988
        self.assertAlmostEqual(e_ev[0], 8.38896, delta=2e-3)

if __name__ == "__main__":
    print("EE DH-ADC(2) calculations for water molecule")
    unittest.main()
