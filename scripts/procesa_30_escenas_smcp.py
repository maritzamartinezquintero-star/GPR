"""
procesa_30_escenas_smcp.py
--------------------------
Segmentacion automatica (RANSAC) + ajuste geometrico de esferas con rechazo 2.5 sigma
para la campana de 30 escaneos del arreglo de 3 esferas con el sistema SMCP.

Entrada : carpeta con archivos .xyz  (X Y Z R G B, separados por espacios)
Salida  : resultados_30_escenas.csv  y  resultados_30_escenas.npz

Requiere: numpy, scipy
Uso:
    python procesa_30_escenas_smcp.py
"""

import os
import glob
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations, permutations

# ---------------------------------------------------------------- CONFIGURACION
REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raiz del repo
CARPETA = r'C:\Users\oicasio\Desktop\MedicionEsferas'   # <-- EDITAR (escaneos .xyz)
SALIDA  = os.path.join(REPO, 'resultados', 'esferas')

R_NOM        = 50.0    # radio nominal (mm)
TOL_RANSAC   = 1.0     # tolerancia de inlier en RANSAC (mm)
TOL_REFINA   = 1.5     # tolerancia de refinamiento (mm)
N_ITER       = 4000    # iteraciones RANSAC por esfera
VOXEL        = 2.0     # submuestreo para RANSAC (mm); el ajuste final usa full-res
RADIO_CORTE  = 58.0    # radio de extraccion alrededor del centro semilla (mm)
SIGMA_REJ    = 2.5     # rechazo de atipicos, en desviaciones estandar
MIN_INLIERS  = 200     # minimo de inliers para aceptar una esfera

# Valores de referencia MMC
REF_DIAM = {'A': 100.0071, 'B': 100.0166, 'C': 100.0033}
REF_DIST = {('A', 'B'): 217.4531, ('A', 'C'): 196.0364, ('B', 'C'): 308.1691}


# ---------------------------------------------------------------- AJUSTE
def fit_alg(X):
    """Ajuste algebraico de esfera (semilla lineal)."""
    A = np.c_[2 * X, np.ones(len(X))]
    b = (X ** 2).sum(1)
    s, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = s[:3]
    return c, np.sqrt(s[3] + c @ c)


def fit_geo(X, c0, r0):
    """Ajuste geometrico (ortogonal) por minimos cuadrados no lineales."""
    def resid(p):
        return np.linalg.norm(X - p[:3], axis=1) - p[3]
    sol = least_squares(resid, np.r_[c0, r0], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return sol.x[:3], sol.x[3], resid(sol.x)


def ajusta_con_rechazo(X, max_iter=8):
    """Ajuste geometrico iterativo con rechazo 2.5 sigma (desviacion estandar)."""
    c, r = fit_alg(X)
    for _ in range(max_iter):
        c, r, res = fit_geo(X, c, r)
        keep = np.abs(res) < SIGMA_REJ * res.std()
        if keep.all():
            break
        X = X[keep]
    c, r, res = fit_geo(X, c, r)
    return c, r, res, len(X)


# ---------------------------------------------------------------- SEGMENTACION
def detecta_esferas(P, rng):
    """RANSAC secuencial: localiza las 3 esferas en la escena completa."""
    key = np.floor(P / VOXEL).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    rem = P[idx]

    centros = []
    for _ in range(3):
        mejor = (0, None, None)
        for _ in range(N_ITER):
            s = rem[rng.choice(len(rem), 4, replace=False)]
            if np.ptp(s, axis=0).max() > 2.8 * R_NOM:
                continue
            try:
                c, r = fit_alg(s)
            except np.linalg.LinAlgError:
                continue
            if not (0.9 * R_NOM < r < 1.12 * R_NOM):
                continue
            n = (np.abs(np.linalg.norm(rem - c, axis=1) - r) < TOL_RANSAC).sum()
            if n > mejor[0]:
                mejor = (n, c, r)

        n, c, r = mejor
        if n < MIN_INLIERS:
            break
        for _ in range(3):
            m = np.abs(np.linalg.norm(rem - c, axis=1) - r) < TOL_REFINA
            c, r = fit_alg(rem[m])
        centros.append(c)
        rem = rem[np.abs(np.linalg.norm(rem - c, axis=1) - r) >= TOL_REFINA]

    return centros


# ---------------------------------------------------------------- ETIQUETADO
def etiqueta(centros):
    """Asigna A, B, C por firma de distancias contra la referencia MMC."""
    ref = np.array([REF_DIST[('A', 'B')], REF_DIST[('A', 'C')], REF_DIST[('B', 'C')]])
    mejor, etq = np.inf, None
    for perm in permutations(range(3)):
        C = [centros[i] for i in perm]           # perm -> (A, B, C)
        med = np.array([np.linalg.norm(C[0] - C[1]),
                        np.linalg.norm(C[0] - C[2]),
                        np.linalg.norm(C[1] - C[2])])
        err = np.abs(med - ref).sum()
        if err < mejor:
            mejor, etq = err, perm
    # etq[k] = indice del centro que corresponde a A, B, C
    return {'A': etq[0], 'B': etq[1], 'C': etq[2]}, mejor


# ---------------------------------------------------------------- PRINCIPAL
def main():
    rng = np.random.default_rng(0)
    archivos = sorted(glob.glob(os.path.join(CARPETA, '*.xyz')))
    print(f'{len(archivos)} archivos encontrados\n')

    filas = []
    for k, f in enumerate(archivos, 1):
        nombre = os.path.basename(f)
        try:
            P = np.loadtxt(f)[:, :3]
        except Exception as e:
            print(f'[{k:2d}] {nombre}: ERROR de lectura -> {e}')
            continue

        semillas = detecta_esferas(P, rng)
        if len(semillas) < 3:
            print(f'[{k:2d}] {nombre}: solo {len(semillas)} esferas detectadas -- SE OMITE')
            continue

        # ajuste final a resolucion completa
        cen, rad, sig, npts = [], [], [], []
        for c0 in semillas:
            X = P[np.linalg.norm(P - c0, axis=1) < RADIO_CORTE]
            c, r, res, n = ajusta_con_rechazo(X)
            cen.append(c); rad.append(r); sig.append(res.std()); npts.append(n)

        mapa, err_firma = etiqueta(cen)
        cen = np.array(cen); rad = np.array(rad)
        sig = np.array(sig); npts = np.array(npts)

        fila = {'archivo': nombre,
                'mtime': os.path.getmtime(f),
                'err_firma': err_firma}
        for L in 'ABC':
            i = mapa[L]
            fila[f'D_{L}']     = 2 * rad[i]
            fila[f'eD_{L}']    = 2 * rad[i] - REF_DIAM[L]
            fila[f'sigma_{L}'] = sig[i]
            fila[f'n_{L}']     = npts[i]
            fila[f'X_{L}'], fila[f'Y_{L}'], fila[f'Z_{L}'] = cen[i]
        for L1, L2 in combinations('ABC', 2):
            m = np.linalg.norm(cen[mapa[L1]] - cen[mapa[L2]])
            fila[f'd_{L1}{L2}']  = m
            fila[f'ed_{L1}{L2}'] = m - REF_DIST[(L1, L2)]
        fila['Z_centroide'] = cen.mean(axis=0)[2]

        filas.append(fila)
        print(f"[{k:2d}] {nombre}  Z={fila['Z_centroide']:7.1f}  "
              f"eD=({fila['eD_A']*1000:+6.0f},{fila['eD_B']*1000:+6.0f},{fila['eD_C']*1000:+6.0f}) um  "
              f"ed=({fila['ed_AB']*1000:+7.0f},{fila['ed_AC']*1000:+7.0f},{fila['ed_BC']*1000:+7.0f}) um")

    if not filas:
        print('\nNo se proceso ningun archivo.')
        return

    # ---- guardar
    cols = list(filas[0].keys())
    csv_path = os.path.join(SALIDA, 'resultados_30_escenas.csv')
    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write(','.join(cols) + '\n')
        for fi in filas:
            fh.write(','.join(str(fi[c]) for c in cols) + '\n')

    npz_path = os.path.join(SALIDA, 'resultados_30_escenas.npz')
    np.savez(npz_path, **{c: np.array([fi[c] for fi in filas]) for c in cols})

    # ---- resumen
    print('\n' + '=' * 62)
    print('RESUMEN DE LA CAMPANA')
    print('=' * 62)
    for L in 'ABC':
        e = np.array([fi[f'eD_{L}'] for fi in filas]) * 1000
        print(f'  Diametro {L}:  media {e.mean():+8.1f} um   sigma {e.std():6.1f} um')
    for L1, L2 in combinations('ABC', 2):
        e = np.array([fi[f'ed_{L1}{L2}'] for fi in filas]) * 1000
        ref = REF_DIST[(L1, L2)]
        print(f'  Dist {L1}-{L2}:   media {e.mean():+8.1f} um   sigma {e.std():6.1f} um'
              f'   ({e.mean()/ref:+7.0f} ppm)')

    Z = np.array([fi['Z_centroide'] for fi in filas])
    print(f'\n  Rango de Z del centroide: {Z.min():.1f} a {Z.max():.1f} mm  '
          f'(amplitud {Z.ptp():.1f} mm)')
    print(f'\nGuardado:\n  {csv_path}\n  {npz_path}')


if __name__ == '__main__':
    main()
