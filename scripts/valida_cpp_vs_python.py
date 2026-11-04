"""
valida_cpp_vs_python.py
-----------------------
Verifica que la compensacion integrada en SMCP.exe (C++) da la misma nube que Python.

PASOS
    1) En SMCP.exe: Calibration > Load GP compensation model...  (modelo_gp_smcp.yml)
    2) Calibration > Compensate existing .xyz file...  y elegir XYZ_ORIGINAL.
       SMCP escribe <nombre>_gp.xyz en la misma carpeta.
    3) python valida_cpp_vs_python.py   [original.xyz  original_gp.xyz]

Criterio: diferencia maxima < 0.01 um (ambos calculan en doble precision; lo que queda es
redondeo de la escritura a 12 cifras).

REQUIERE: numpy
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exporta_modelo_yml import aplica, MODELO_NPZ   # noqa: E402

XYZ_ORIGINAL = r'C:\Users\oicasio\Desktop\MedicionEsferas\2026-sep.-15_09.57.19.561.xyz'
XYZ_CPP      = XYZ_ORIGINAL.replace('.xyz', '_gp.xyz')
TOLERANCIA_UM = 0.01


def main():
    original, cpp = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (XYZ_ORIGINAL, XYZ_CPP)
    P = np.loadtxt(original)[:, :3]
    C = np.loadtxt(cpp)[:, :3]
    if len(P) != len(C):
        raise SystemExit(f'Numero de puntos distinto: original {len(P)}, C++ {len(C)}')
    X = aplica(P, dict(np.load(MODELO_NPZ)))
    d = np.linalg.norm(C - X, axis=1) * 1000
    corr = np.linalg.norm(X - P, axis=1) * 1000
    print(f'{len(P):,} puntos; correccion aplicada: media {corr.mean():.0f} um, max {corr.max():.0f} um')
    print(f'C++ vs Python: diferencia max {d.max():.2e} um, RMS {np.sqrt((d ** 2).mean()):.2e} um')
    print('OK: C++ reproduce a Python' if d.max() < TOLERANCIA_UM else
          f'FALLA: diferencia mayor a {TOLERANCIA_UM} um')


if __name__ == '__main__':
    main()
