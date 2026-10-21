"""
gp_smcp.py
----------
Compensacion del sistema SMCP (luz estructurada camara-proyector) con regresion por
procesos gaussianos (GPR; en geoestadistica, kriging universal). Es la alternativa al
MLP del PINN con la misma fisica, los mismos datos, los mismos folds y la misma linea
base que pinn_smcp.py (version 2), para que la comparacion sea directa.

MODELO
    p_f         = p * (1 + s(z, rho))        bloque fisico (identico al PINN)
    p_corregido = p_f - e(p_f)               e = campo de error residual (GP)

    s(z,rho) = a0 + a1*z + a2*z^2 + a3*rho^2 + a4*rho^4 + a5*z*rho^2
        Se resuelve por minimos cuadrados: es el minimo exacto de la misma perdida
        L_R + W_DIST*L_D del PINN, sin descenso de gradiente.

    e(p) ~ GP(0, k(p,p') I3),   k = s_lin^2 (p-p0).(p'-p0) + s_f^2 Matern52(p,p'; lx,ly,lz)
        termino lineal   deformacion homogenea: escala distinta por eje y cortante
                         (la "tendencia" del kriging universal)
        Matern 5/2 ARD   lo no lineal, con una longitud de correlacion por eje

OBSERVACIONES INDIRECTAS (la parte clave)
    En el SMCP no se conoce el error punto por punto, solo los 6 mensurandos de cada
    escaneo. Para desplazamientos pequenos, ajustar una esfera es LINEAL en los puntos:
        [dc; dR] = (B'B)^-1 B' (n_i . dp_i),        B = [n_i  1]
    asi que cada diametro y cada distancia es un funcional lineal del campo e, y un GP
    condicionado en funcionales lineales sigue siendo un GP con solucion cerrada
    (Rasmussen & Williams 2006, sec. 9.4, misma idea que las observaciones de
    derivadas). No hay red ni epocas: solo algebra lineal.
    Para no evaluar el kernel en ~15 000 puntos por esfera, los pesos de cada esfera se
    agrupan en celdas de H_CELDA mm y se corrigen para que reproduzcan exactamente el
    funcional de todos los puntos en cualquier campo cuadratico (error de pocos um,
    verificado contra el calculo con todos los puntos).

SEGUNDA PASADA (opcional, PASADAS_GP = 2)
    Al corregir, las esferas quedan mas redondas y el rechazo 2.5 sigma del reajuste
    descarta otros puntos: eso deja un sesgo sistematico (en estos datos, ~-25 um en
    D_A y D_C). Con PASADAS_GP = 2 se corrigen los escaneos de entrenamiento con la
    primera pasada, se reajustan con el mismo pipeline y un segundo GP (mismos
    hiperparametros) aprende lo que quedo. En los 37 escaneos quita ese sesgo
    (D_A 43 -> 32 um, D_C 46 -> 39 um) pero empeora D_B y el RMS global no mejora
    (55.4 -> 56.1 um); por eso viene apagada.

HIPERPARAMETROS
    s_lin, s_f, lx, ly, lz y el ruido de diametros (s_D) y de distancias (s_L) se
    estiman maximizando la verosimilitud marginal SOLO con los escaneos de entrenamiento
    de cada fold. s_D y s_L estiman el ruido aleatorio por escaneo: el piso al que
    ningun modelo de compensacion deberia poder bajar.

INCERTIDUMBRE
    Para cada escaneo de prueba el GP predice, antes de corregir, la sigma del error
    que quedara en cada mensurando. Se valida con la cobertura: ~68 % de los errores
    dentro de +-1 sigma, ~95 % dentro de +-2 sigma y RMS(z) ~ 1. Es incertidumbre del
    modelo dado el kernel, no un presupuesto GUM completo (no incluye la incertidumbre
    de los 6 coeficientes fisicos ni la de la referencia MMC). En los 37 escaneos la
    sigma salio ~1.4 veces corta (RMS(z) = 1.4); con ese factor la cobertura llega a
    68 % / 95 %. Hay que confirmarlo con escaneos nuevos antes de usarlo.

VALIDACION
    k-fold POR ESCANEO con los mismos folds que pinn_smcp.py (SEMILLA = 0). Los
    escaneos de prueba se corrigen punto por punto y sus esferas se reajustan desde
    cero con el mismo pipeline (rechazo 2.5 sigma). Linea base: escala fija estimada
    con los escaneos de entrenamiento.

CONFIGURACIONES (todas en una sola corrida)
    fisica      solo los 6 coeficientes
    gp_lineal   fisica + solo el termino lineal del GP (deformacion homogenea)
    gp          fisica + GP completo (lineal + Matern)

USO
    1) Revisar CARPETA_XYZ, SALIDA y PINN_NPZ
    2) python gp_smcp.py          (reutiliza nubes_segmentadas.npz; ~30 min)
    Si encuentra PINN_NPZ (resultados_pinn_smcp_pinn.npz de la version 2 del PINN),
    agrega el PINN a la comparacion escaneo por escaneo.
    Con ENTRENA_FINAL = True ajusta el modelo con todos los escaneos y lo exporta a
    modelo_gp_smcp.npz / .mat. Para corregir una nube nueva:
        from gp_smcp import carga_modelo, aplica_modelo
        P_corr = aplica_modelo(P, carga_modelo(ruta_npz))

REQUIERE: numpy, scipy, pandas, openpyxl   (no usa torch)
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.io import savemat
from itertools import combinations, permutations

# ---------------------------------------------------------------- CONFIGURACION
CARPETA_XYZ = r'C:\Users\oicasio\Desktop\MedicionEsferas'                   # <-- EDITAR
SALIDA      = r'C:\Users\oicasio\Desktop\GPR\Resultado GPR'                # <-- EDITAR
CACHE       = r'C:\Users\oicasio\Desktop\MedicionEsferas\Esferas limpias\nubes_segmentadas.npz'
PINN_NPZ    = r'C:\Users\oicasio\Desktop\GPR\resultados_pinn_smcp_pinn 1\resultados_pinn_smcp_pinn.npz'   # opcional (PINN v2)

# referencia MMC (la misma con la que se reportaron los 97.9 um del PINN)
REF_DIAM = {'A': 100.0071, 'B': 100.0166, 'C': 100.0033}
REF_DIST = {('A', 'B'): 217.4531, ('A', 'C'): 196.0364, ('B', 'C'): 308.1691}
# octubre 2026 (promedio de 3 mediciones); cambian < 1.3 um:
# REF_DIAM = {'A': 100.0067, 'B': 100.01613, 'C': 100.00313}
# REF_DIST = {('A', 'B'): 217.45227, ('A', 'C'): 196.0361, ('B', 'C'): 308.16863}
ETIQ = ['A', 'B', 'C']

# segmentacion / ajuste (identicos a pinn_smcp.py; NO cambiar o el cache deja de valer)
R_NOM, TOL_RANSAC, TOL_REFINA = 50.0, 1.0, 1.5
N_ITER, VOXEL, RADIO_CORTE, SIGMA_REJ = 4000, 2.0, 58.0, 2.5

# validacion y bloque fisico (identicos a pinn_smcp.py version 2)
K_FOLDS     = 5
SEMILLA     = 0
PTS_POR_ESF = 2000      # submuestreo para ajustar la fisica (la evaluacion usa todo)
W_DIST      = 1.0       # peso de L_D

# proceso gaussiano
H_CELDA     = 20.0      # mm; celda para agrupar los pesos de cada esfera
L_INICIALES = (100.0, 300.0, 1000.0)  # mm; arranques de l (se queda el de mayor verosimilitud)
LIMITES = {
    's_lin': (1e-6, 1e-2),    # deformacion lineal (adimensional; 1e-4 = 100 ppm)
    's_f':   (1e-4, 2.0),     # mm; amplitud del Matern
    'l':     (50.0, 3000.0),  # mm; longitudes de correlacion
    's_D':   (1e-3, 1.0),     # mm; ruido de diametros
    's_L':   (1e-3, 1.0),     # mm; ruido de distancias
}
INICIALES = {'s_lin': 1e-4, 's_f': 0.1, 's_D': 0.05, 's_L': 0.05}
GP_CONFIGS = {'gp_lineal': True, 'gp': False}   # nombre: solo_lineal
PASADAS_GP  = 1         # 2 = agrega la segunda pasada (ver SEGUNDA PASADA arriba)
ENTRENA_FINAL = True

PARES = list(combinations(range(3), 2))
S5 = np.sqrt(5.0)


# ================================================================ AJUSTE DE ESFERAS
def fit_alg(X):
    A = np.c_[2 * X, np.ones(len(X))]
    s, *_ = np.linalg.lstsq(A, (X ** 2).sum(1), rcond=None)
    c = s[:3]
    return c, np.sqrt(s[3] + c @ c)


def fit_geo(X, c0, r0):
    f = lambda p: np.linalg.norm(X - p[:3], axis=1) - p[3]
    s = least_squares(f, np.r_[c0, r0], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return s.x[:3], s.x[3], f(s.x)


def ajusta(X, max_iter=8):
    """Identico a pinn_smcp.py. Devuelve tambien los indices de los puntos aceptados."""
    idx = np.arange(len(X))
    c, r = fit_alg(X)
    for _ in range(max_iter):
        c, r, res = fit_geo(X, c, r)
        keep = np.abs(res) < SIGMA_REJ * res.std()
        if keep.all():
            break
        X, idx = X[keep], idx[keep]
    c, r, res = fit_geo(X, c, r)
    return c, r, res, idx


# ================================================================ SEGMENTACION
def detecta(P, rng):
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
        if n < 200:
            break
        for _ in range(3):
            m = np.abs(np.linalg.norm(rem - c, axis=1) - r) < TOL_REFINA
            c, r = fit_alg(rem[m])
        centros.append(c)
        rem = rem[np.abs(np.linalg.norm(rem - c, axis=1) - r) >= TOL_REFINA]
    return centros


def etiqueta(centros):
    ref = np.array([REF_DIST[('A', 'B')], REF_DIST[('A', 'C')], REF_DIST[('B', 'C')]])
    mejor, etq = np.inf, None
    for perm in permutations(range(3)):
        C = [centros[i] for i in perm]
        med = np.array([np.linalg.norm(C[0] - C[1]),
                        np.linalg.norm(C[0] - C[2]),
                        np.linalg.norm(C[1] - C[2])])
        e = np.abs(med - ref).sum()
        if e < mejor:
            mejor, etq = e, perm
    return {L: etq[k] for k, L in enumerate(ETIQ)}


def preprocesa(carpeta, cache):
    """Segmenta todos los .xyz y guarda un cache compacto (identico a pinn_smcp.py)."""
    rng = np.random.default_rng(SEMILLA)
    archivos = sorted(glob.glob(os.path.join(carpeta, '*.xyz')))
    print(f'Segmentando {len(archivos)} escaneos (solo la primera vez)...')
    pts, id_esc, id_esf, nombres = [], [], [], []
    for f in archivos:
        P = np.loadtxt(f)[:, :3]
        sem = detecta(P, rng)
        if len(sem) < 3:
            print(f'  {os.path.basename(f)}: {len(sem)} esferas -- OMITIDO')
            continue
        mapa = etiqueta(sem)
        ie = len(nombres)
        for s, L in enumerate(ETIQ):
            X = P[np.linalg.norm(P - sem[mapa[L]], axis=1) < RADIO_CORTE]
            c, r, res, _ = ajusta(X)
            X = X[np.abs(np.linalg.norm(X - c, axis=1) - r) < SIGMA_REJ * res.std()]
            pts.append(X)
            id_esc.append(np.full(len(X), ie))
            id_esf.append(np.full(len(X), s))
        nombres.append(os.path.basename(f))
        print(f'  [{len(nombres):2d}] {os.path.basename(f)}')
    np.savez_compressed(cache,
                        pts=np.vstack(pts).astype(np.float32),
                        id_esc=np.concatenate(id_esc).astype(np.int32),
                        id_esf=np.concatenate(id_esf).astype(np.int32),
                        nombres=np.array(nombres))
    print(f'Cache guardado: {cache}')


# ================================================================ EVALUACION BASICA
def mensurandos(pts_por_esf):
    """Ajusta las 3 esferas y devuelve los 6 errores en um (identico a pinn_smcp.py)."""
    cen, dia = [], []
    for s in range(3):
        c, r, _, _ = ajusta(pts_por_esf[s])
        cen.append(c)
        dia.append(2 * r)
    e = [(dia[s] - REF_DIAM[ETIQ[s]]) * 1000 for s in range(3)]
    for i, j in PARES:
        e.append((np.linalg.norm(cen[i] - cen[j]) - REF_DIST[(ETIQ[i], ETIQ[j])]) * 1000)
    return np.array(e)


def escala_fija(errs_tr):
    """Factor de escala unico estimado sobre los escaneos de entrenamiento."""
    ref = np.array([REF_DIAM[L] for L in ETIQ] +
                   [REF_DIST[(ETIQ[i], ETIQ[j])] for i, j in PARES])
    return 1.0 + (errs_tr / 1000 / ref).mean()


# ================================================================ BLOQUE FISICO
def base_fisica(P, mu_z, sd_z):
    zn = (P[:, 2] - mu_z) / sd_z
    rho2 = (P[:, 0] ** 2 + P[:, 1] ** 2) / P[:, 2] ** 2
    return np.c_[np.ones(len(P)), zn, zn ** 2, rho2, rho2 ** 2, zn * rho2]


def aplica_fisica(P, a, mu_z, sd_z):
    return P * (1.0 + base_fisica(P, mu_z, sd_z) @ a)[:, None]


def ajusta_fisica(P, fila, nE, mu_z, sd_z, a0, C0):
    """Minimiza L_R + W_DIST*L_D (la perdida del PINN sin MLP) sobre a0..a5 y los centros."""
    N, nS = len(P), 3 * nE
    H = base_fisica(P, mu_z, sd_z)
    Rref = np.array([REF_DIAM[ETIQ[s]] / 2 for _ in range(nE) for s in range(3)])[fila]
    ii = np.array([3 * e + i for i, j in PARES for e in range(nE)])
    jj = np.array([3 * e + j for i, j in PARES for e in range(nE)])
    Lref = np.repeat([REF_DIST[(ETIQ[i], ETIQ[j])] for i, j in PARES], nE)
    nD = len(ii)
    wR, wD = 1 / np.sqrt(N), np.sqrt(W_DIST / nD)

    def geom(v):
        a, C = v[:6], v[6:].reshape(nS, 3)
        D = P * (1 + H @ a)[:, None] - C[fila]
        U = C[ii] - C[jj]
        return D, np.linalg.norm(D, axis=1), U, np.linalg.norm(U, axis=1)

    def residuos(v):
        D, nr, U, nu = geom(v)
        return np.r_[wR * (nr - Rref), wD * (nu - Lref)]

    def jacobiano(v):
        D, nr, U, nu = geom(v)
        n, u = D / nr[:, None], U / nu[:, None]
        ja = wR * (n * P).sum(1)[:, None] * H
        f1 = np.repeat(np.arange(N), 6)
        k1 = np.tile(np.arange(6), N)
        f2 = np.repeat(np.arange(N), 3)
        k2 = 6 + 3 * np.repeat(fila, 3) + np.tile(np.arange(3), N)
        f3 = N + np.repeat(np.arange(nD), 3)
        k3 = 6 + 3 * np.repeat(ii, 3) + np.tile(np.arange(3), nD)
        k4 = 6 + 3 * np.repeat(jj, 3) + np.tile(np.arange(3), nD)
        return sp.csr_matrix((np.r_[ja.ravel(), -wR * n.ravel(), wD * u.ravel(), -wD * u.ravel()],
                              (np.r_[f1, f2, f3, f3], np.r_[k1, k2, k3, k4])),
                             shape=(N + nD, 6 + 3 * nS))

    sol = least_squares(residuos, np.r_[a0, C0.ravel()], jac=jacobiano, method='trf',
                        x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=300)
    return sol.x[:6]


def entrena_fisica(pts, id_esc, id_esf, escaneos_tr, centros_ini, s_fija):
    """Mismo submuestreo, normalizacion y arranque que pinn_smcp.py version 2."""
    rng = np.random.default_rng(SEMILLA)
    sel = []
    for e in escaneos_tr:
        for s in range(3):
            idx = np.where((id_esc == e) & (id_esf == s))[0]
            if len(idx) > PTS_POR_ESF:
                idx = rng.choice(idx, PTS_POR_ESF, replace=False)
            sel.append(idx)
    sel = np.concatenate(sel)
    P = pts[sel]
    remap = {e: i for i, e in enumerate(escaneos_tr)}
    fila = np.array([remap[e] * 3 + s for e, s in zip(id_esc[sel], id_esf[sel])])
    mu_z, sd_z = P[:, 2].mean(), P[:, 2].std()
    a0 = np.zeros(6)
    a0[0] = 1.0 / s_fija - 1.0                         # arranque = escala fija
    C0 = np.array([centros_ini[(e, s)] / s_fija for e in escaneos_tr for s in range(3)])
    return ajusta_fisica(P, fila, len(escaneos_tr), mu_z, sd_z, a0, C0), mu_z, sd_z


# ================================================================ OBSERVACIONES INDIRECTAS
def _monomios(U):
    """1, x, y, z y los 6 terminos cuadraticos."""
    x, y, z = U.T
    return np.column_stack([np.ones(len(U)), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z])


def operador_escaneo(esferas):
    """
    esferas: las 3 nubes (A, B, C) ya corregidas por la fisica.
    Devuelve
        y     (6,)       errores de los mensurandos en mm
        reps  (N, 3)     centroides de las celdas (puntos donde se evalua el campo)
        O     (6, N, 3)  pesos: d_mensurando = sum_c O[m, c, :] . desplazamiento(reps[c])
    """
    cen, rad, reps, Vs = [], [], [], []
    for X0 in esferas:
        c, r, _, idx = ajusta(X0)
        X = X0[idx]
        n = X - c
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        B = np.c_[n, np.ones(len(n))]
        M = np.linalg.solve(B.T @ B, B.T)              # (4, Npts): dc_x, dc_y, dc_z, dR
        w = np.concatenate([M[q][:, None] * n for q in range(4)], 1)   # (Npts, 12)
        _, inv = np.unique(np.floor(X / H_CELDA).astype(np.int64), axis=0,
                           return_inverse=True)
        inv = inv.ravel()
        nc = inv.max() + 1
        cnt = np.bincount(inv, minlength=nc)
        P = np.stack([np.bincount(inv, X[:, k], nc) for k in range(3)], 1) / cnt[:, None]
        V = np.stack([np.bincount(inv, w[:, j], nc) for j in range(12)], 1)  # (nc, 12)
        # correccion de momentos: los pesos de las celdas reproducen EXACTAMENTE el
        # funcional de todos los puntos para cualquier campo cuadratico en la esfera
        A = _monomios((P - c) / r).T
        T = _monomios((X - c) / r).T @ w
        V += A.T @ np.linalg.lstsq(A @ A.T, T - A @ V, rcond=1e-12)[0]
        reps.append(P)
        Vs.append(V.reshape(nc, 4, 3).transpose(1, 0, 2))   # (4, nc, 3)
        cen.append(c)
        rad.append(r)
    y = [2 * rad[s] - REF_DIAM[ETIQ[s]] for s in range(3)]
    y += [np.linalg.norm(cen[i] - cen[j]) - REF_DIST[(ETIQ[i], ETIQ[j])] for i, j in PARES]
    off = np.cumsum([0] + [len(p) for p in reps])
    O = np.zeros((6, off[-1], 3))
    for s in range(3):
        O[s, off[s]:off[s + 1]] = 2 * Vs[s][3]                       # dD = 2 dR
    for m, (i, j) in enumerate(PARES):
        u = (cen[i] - cen[j]) / np.linalg.norm(cen[i] - cen[j])      # dL = u.(dc_i - dc_j)
        O[3 + m, off[i]:off[i + 1]] += np.einsum('q,qck->ck', u, Vs[i][:3])
        O[3 + m, off[j]:off[j + 1]] -= np.einsum('q,qck->ck', u, Vs[j][:3])
    return np.array(y), np.vstack(reps), O


def matern52(A, B, l):
    r = cdist(A / l, B / l)
    return (1 + S5 * r + 5.0 / 3.0 * r * r) * np.exp(-S5 * r)


class GPIndirecto:
    """GP vectorial observado a traves de funcionales lineales (diametros y distancias)."""

    def __init__(self, ys, reps, Os, solo_lineal=False):
        self.y = np.concatenate(ys)
        self.P = np.vstack(reps)
        self.solo_lineal = solo_lineal
        n_obs, N = len(self.y), len(self.P)
        O = np.zeros((n_obs, N, 3))
        fo = co = 0
        for o, p in zip(Os, reps):
            O[fo:fo + 6, co:co + len(p)] = o
            fo, co = fo + 6, co + len(p)
        self.O = [sp.csr_matrix(O[:, :, k]) for k in range(3)]
        self.p0 = self.P.mean(0)
        Pc = self.P - self.p0
        self.Phi = np.column_stack([self.O[k] @ Pc[:, b] for k in range(3) for b in range(3)])
        self.esD = np.tile(np.r_[np.ones(3), np.zeros(3)], n_obs // 6).astype(bool)
        # diferencias al cuadrado por eje, una sola vez (float32 para ahorrar memoria)
        self.D2 = None if solo_lineal else [
            np.square(self.P[:, k, None] - self.P[None, :, k]).astype(np.float32) for k in range(3)]

    def kernel(self, l, grad=False):
        """Matern 5/2 entre todos los puntos de soporte; con grad, tambien dk/dr."""
        # aritmetica en float64: en float32 la verosimilitud sale "ruidosa" y el
        # optimizador se detiene antes de tiempo
        s = np.multiply(self.D2[0], 5.0 / l[0] ** 2, dtype=np.float64)
        for k in (1, 2):
            s += np.multiply(self.D2[k], 5.0 / l[k] ** 2, dtype=np.float64)
        np.sqrt(s, out=s)                                 # s = sqrt(5) r
        e = np.exp(-s)
        K = s / 3
        K += 1
        K *= s
        K += 1
        K *= e                                            # (1 + s + s^2/3) e
        if not grad:
            return K, None
        s += 1
        s *= e
        s *= 5.0 / 3.0                                    # (5/3)(1 + s) e
        return K, s

    def proyecta(self, K):
        """sum_k O_k K O_k'  (covarianza entre observaciones)"""
        return sum(self.O[k] @ (self.O[k] @ K).T for k in range(3))

    def nlml(self, th, grad=True):
        """-log verosimilitud marginal y su gradiente respecto a log(hiperparametros)."""
        sl, sf, lx, ly, lz, sD, sL = np.exp(th)
        l = np.array([lx, ly, lz])
        n = len(self.y)
        Clin = self.Phi @ self.Phi.T
        ruido = np.where(self.esD, sD ** 2, sL ** 2)
        C = sl ** 2 * Clin + np.diag(ruido)
        if not self.solo_lineal:
            K, gk = self.kernel(l, grad)
            CM = self.proyecta(K)
            del K
            C = C + sf ** 2 * CM
        C[np.diag_indices(n)] += 1e-12
        cf = cho_factor(C, lower=True)
        a = cho_solve(cf, self.y)
        val = 0.5 * self.y @ a + np.log(np.diag(cf[0])).sum() + 0.5 * n * np.log(2 * np.pi)
        if not grad:
            return val
        W = np.outer(a, a) - cho_solve(cf, np.eye(n))
        g = np.zeros(7)
        g[0] = np.sum(W * Clin) * sl ** 2
        if not self.solo_lineal:
            g[1] = np.sum(W * CM) * sf ** 2
            for k in range(3):
                dK = self.D2[k] * gk                      # dK/dlog(l_k) = gk * dx_k^2 / l_k^2
                dK /= l[k] ** 2
                g[2 + k] = 0.5 * sf ** 2 * np.sum(W * self.proyecta(dK))
                del dK
        g[5] = np.sum(np.diag(W) * self.esD) * sD ** 2
        g[6] = np.sum(np.diag(W) * ~self.esD) * sL ** 2
        return val, -g

    def ajusta_hiper(self, th_previo=None):
        """Maximiza la verosimilitud marginal desde varios arranques; se queda el mejor."""
        lim = [LIMITES['s_lin'], LIMITES['s_f']] + [LIMITES['l']] * 3 + [LIMITES['s_D'], LIMITES['s_L']]
        lim = [(np.log(a), np.log(b)) for a, b in lim]
        arranques = [np.log([INICIALES['s_lin'], INICIALES['s_f'], l0, l0, l0,
                             INICIALES['s_D'], INICIALES['s_L']])
                     for l0 in (L_INICIALES[:1] if self.solo_lineal else L_INICIALES)]
        if th_previo is not None:
            arranques.append(np.clip(th_previo, [a for a, b in lim], [b for a, b in lim]))
        mejor = None
        for th0 in arranques:
            r = minimize(self.nlml, th0, jac=True, method='L-BFGS-B', bounds=lim)
            if mejor is None or r.fun < mejor.fun:
                mejor = r
        self.th = mejor.x
        self.prepara()
        return mejor

    def prepara(self):
        self.sl, self.sf, lx, ly, lz, self.sD, self.sL = np.exp(self.th)
        self.l = np.array([lx, ly, lz])
        n = len(self.y)
        C = self.sl ** 2 * self.Phi @ self.Phi.T + np.diag(np.where(self.esD, self.sD ** 2,
                                                                     self.sL ** 2))
        if not self.solo_lineal:
            C = C + self.sf ** 2 * self.proyecta(self.kernel(self.l)[0])
        C[np.diag_indices(n)] += 1e-12
        self.cf = cho_factor(C, lower=True)
        self.alpha = cho_solve(self.cf, self.y)
        # media posterior del campo:  e(p) = B (p - p0) + sum_j k(p, P_j) W_j
        self.B = (self.sl ** 2 * self.Phi.T @ self.alpha).reshape(3, 3)
        self.W = self.sf ** 2 * np.column_stack([self.O[k].T @ self.alpha for k in range(3)])
        if self.solo_lineal:
            self.W[:] = 0.0

    def campo(self, X, bloque=4000):
        """Media posterior del campo de error en los puntos X (mm)."""
        e = (X - self.p0) @ self.B.T
        if not self.solo_lineal:
            for i in range(0, len(X), bloque):
                e[i:i + bloque] += matern52(X[i:i + bloque], self.P, self.l) @ self.W
        return e

    def predice(self, reps_t, O_t):
        """Media y covarianza posterior de los 6 errores de un escaneo nuevo (mm)."""
        Pc = reps_t - self.p0
        Phi_t = np.column_stack([O_t[:, :, k] @ Pc[:, b] for k in range(3) for b in range(3)])
        Kty = self.sl ** 2 * Phi_t @ self.Phi.T
        Ktt = self.sl ** 2 * Phi_t @ Phi_t.T
        if not self.solo_lineal:
            Kx = matern52(reps_t, self.P, self.l)
            Kxx = matern52(reps_t, reps_t, self.l)
            for k in range(3):
                Kty += self.sf ** 2 * (self.O[k] @ (O_t[:, :, k] @ Kx).T).T
                Ktt += self.sf ** 2 * O_t[:, :, k] @ Kxx @ O_t[:, :, k].T
        m = Kty @ self.alpha
        S = Ktt - Kty @ cho_solve(self.cf, Kty.T)
        ruido = np.r_[np.full(3, self.sD ** 2), np.full(3, self.sL ** 2)]
        return m, S, ruido

    def hiper(self):
        sf, l = (np.nan, [np.nan] * 3) if self.solo_lineal else (self.sf * 1000, self.l)
        return {'s_lin (ppm)': self.sl * 1e6, 's_f (um)': sf,
                'l_x (mm)': l[0], 'l_y (mm)': l[1], 'l_z (mm)': l[2],
                's_D (um)': self.sD * 1000, 's_L (um)': self.sL * 1000,
                'celdas': len(self.P), 'observaciones': len(self.y)}


def ajusta_pasadas(ys, reps, Os, esferas_tr, solo_lineal, th_ini=None):
    """
    Pasada 1: GP con hiperparametros por maxima verosimilitud.
    Pasadas 2..PASADAS_GP: mismos hiperparametros, linealizado en la nube ya corregida
    (esferas_tr: submuestra de los escaneos de entrenamiento, ya con la fisica).
    """
    gp = GPIndirecto(ys, reps, Os, solo_lineal)
    opt = gp.ajusta_hiper(th_ini)
    etapas, resid = [gp], None
    actual = esferas_tr
    for _ in range(PASADAS_GP - 1):
        actual = [[X - etapas[-1].campo(X) for X in esc] for esc in actual]
        ys2, reps2, Os2 = zip(*[operador_escaneo(esc) for esc in actual])
        if resid is None:
            resid = np.array(ys2) * 1000           # residuo dentro de muestra tras la pasada 1
        g2 = GPIndirecto(ys2, reps2, Os2, solo_lineal)
        g2.th = gp.th
        g2.prepara()
        etapas.append(g2)
    return etapas, opt, resid


def corrige(X, etapas):
    for st in etapas:
        X = X - st.campo(X)
    return X


# ================================================================ MODELO EXPORTADO
def carga_modelo(ruta):
    z = np.load(ruta)
    return {k: z[k] for k in z.files}


def aplica_modelo(P, m, bloque=4000):
    """
    Corrige una nube (n, 3) en mm, en el marco de la camara, con el modelo exportado:
        p_f = p (1 + s(z, rho));   luego, por cada pasada k:
        p  <- p - [ B_k (p - p0_k) + sum_j k52(p, P_rep_k[j]) W_k[j] ]
    """
    X = aplica_fisica(np.asarray(P, dtype=np.float64), m['a'], float(m['mu_z']),
                      float(m['sd_z']))
    for k in range(1, int(m['pasadas']) + 1):
        e = (X - m[f'p0_{k}']) @ m[f'B_{k}'].T
        for i in range(0, len(X), bloque):
            e[i:i + bloque] += matern52(X[i:i + bloque], m[f'P_rep_{k}'], m['l']) @ m[f'W_{k}']
        X = X - e
    return X


def exporta_modelo(ruta, a, mu_z, sd_z, etapas):
    gp = etapas[0]
    d = {'a': a, 'mu_z': mu_z, 'sd_z': sd_z, 'l': gp.l, 's_lin': gp.sl, 's_f': gp.sf,
         's_D': gp.sD, 's_L': gp.sL, 'h_celda': H_CELDA, 'pasadas': len(etapas)}
    for k, st in enumerate(etapas, 1):
        d.update({f'p0_{k}': st.p0, f'B_{k}': st.B, f'P_rep_{k}': st.P, f'W_{k}': st.W})
    np.savez(ruta + '.npz', **d)
    savemat(ruta + '.mat', d)


# ================================================================ REPORTE
def rms(M, eje=0):
    return np.sqrt((M ** 2).mean(eje))


def _escribe(destino, hojas):
    with pd.ExcelWriter(destino, engine='openpyxl') as w:
        for nombre, df in hojas.items():
            df.to_excel(w, sheet_name=nombre, index=False)
            ws = w.sheets[nombre]
            ws.freeze_panes = 'A2'
            fmt = {'Coeficientes': None, 'Z_GP': '0.00'}.get(nombre, '0.0')
            for j, col in enumerate(df.columns, 1):
                letra = ws.cell(1, j).column_letter
                ws.column_dimensions[letra].width = min(max(len(str(col)) + 2, 10), 34)
                if fmt:
                    for celda in ws[letra][1:]:
                        if isinstance(celda.value, float):
                            celda.number_format = fmt


def guarda_excel(ruta, hojas):
    """Escribe el Excel; si el archivo esta abierto, guarda con otro nombre."""
    destino = ruta
    try:
        _escribe(destino, hojas)
    except PermissionError:
        destino = ruta.replace('.xlsx', time.strftime('_%H%M%S.xlsx'))
        _escribe(destino, hojas)
        print(f'  (el archivo estaba abierto; se guardo como {destino})')
    return destino


def cobertura(E, S):
    """E: errores tras corregir (um); S: sigma predicha (um)."""
    filas = []
    for nombre, cols in (('Diametros', slice(0, 3)), ('Distancias', slice(3, 6)),
                         ('Global', slice(0, 6))):
        e, s = E[:, cols].ravel(), S[:, cols].ravel()
        zz = e / s
        filas.append({'Grupo': nombre, 'RMS error (um)': rms(e),
                      'RMS sigma predicha (um)': rms(s), 'RMS(z) [ideal 1]': rms(zz),
                      '% dentro de 1 sigma [ideal 68]': 100 * np.mean(np.abs(zz) <= 1),
                      '% dentro de 2 sigma [ideal 95]': 100 * np.mean(np.abs(zz) <= 2)})
    return pd.DataFrame(filas)


# ================================================================ PRINCIPAL
def main():
    t_ini = time.time()
    if not os.path.exists(CACHE):
        preprocesa(CARPETA_XYZ, CACHE)

    z = np.load(CACHE, allow_pickle=True)
    pts, id_esc, id_esf = z['pts'].astype(np.float64), z['id_esc'], z['id_esf']
    nombres = z['nombres']
    nE = len(nombres)
    print(f'\n{nE} escaneos, {len(pts):,} puntos ({len(pts) // (nE * 3):,} por esfera en promedio)')

    print('Ajustando esferas crudas...')
    grupos = {(e, s): pts[(id_esc == e) & (id_esf == s)] for e in range(nE) for s in range(3)}
    centros_ini = {k: ajusta(v)[0] for k, v in grupos.items()}
    rs = np.random.default_rng(SEMILLA)
    sub = {k: np.sort(rs.choice(len(v), min(PTS_POR_ESF, len(v)), replace=False))
           for k, v in grupos.items()}          # submuestra para la segunda pasada
    E_crudo = np.array([mensurandos([grupos[(e, s)] for s in range(3)]) for e in range(nE)])

    # k-fold por escaneo (mismos folds que pinn_smcp.py)
    rng = np.random.default_rng(SEMILLA)
    orden = rng.permutation(nE)
    folds = np.array_split(orden, K_FOLDS)
    fold_de = np.empty(nE, dtype=int)
    for k, test in enumerate(folds):
        fold_de[test] = k + 1

    nom = [f'D_{L}' for L in ETIQ] + [f'd_{ETIQ[i]}-{ETIQ[j]}' for i, j in PARES]
    E = {c: np.full_like(E_crudo, np.nan) for c in ['escala fija', 'fisica', *GP_CONFIGS]}
    SIG = {c: np.full_like(E_crudo, np.nan) for c in GP_CONFIGS}
    LIN = {c: np.full_like(E_crudo, np.nan) for c in GP_CONFIGS}
    hip, coefs, th_prev = [], [], {}

    for k, test in enumerate(folds):
        t0 = time.time()
        train = np.setdiff1d(orden, test)
        print(f'\n--- Fold {k + 1}/{K_FOLDS}: {len(train)} entrenamiento, {len(test)} prueba')

        # linea base: escala fija de entrenamiento
        s_f = escala_fija(E_crudo[train].mean(0))
        for e in test:
            E['escala fija'][e] = mensurandos([grupos[(e, s)] / s_f for s in range(3)])

        # etapa 1: bloque fisico
        a, mu_z, sd_z = entrena_fisica(pts, id_esc, id_esf, train, centros_ini, s_f)
        coefs.append(np.r_[k + 1, a, s_f])
        print('    fisica a0..a5:', np.array2string(a, precision=6, max_line_width=200))
        fis = {(e, s): aplica_fisica(grupos[(e, s)], a, mu_z, sd_z)
               for e in range(nE) for s in range(3)}

        # observaciones indirectas (sobre la nube ya corregida por la fisica)
        ops_tr = [operador_escaneo([fis[(e, s)] for s in range(3)]) for e in train]
        ops_te = {e: operador_escaneo([fis[(e, s)] for s in range(3)]) for e in test}
        for e in test:
            E['fisica'][e] = ops_te[e][0] * 1000
        ys, reps, Os = zip(*ops_tr)

        # etapa 2: GP (hiperparametros solo con este fold; sin arranque en caliente)
        esf_tr = [[fis[(e, s)][sub[(e, s)]] for s in range(3)] for e in train]
        for cfg, solo_lineal in GP_CONFIGS.items():
            t1 = time.time()
            etapas, opt, resid = ajusta_pasadas(ys, reps, Os, esf_tr, solo_lineal)
            gp = etapas[0]
            th_prev[cfg] = gp.th             # se usa solo como arranque del modelo final
            hip.append({'fold': k + 1, 'config': cfg, **gp.hiper(), '-log ML': opt.fun})
            for e in test:
                y_t, reps_t, O_t = ops_te[e]
                m, S, ruido = gp.predice(reps_t, O_t)
                E[cfg][e] = mensurandos([corrige(fis[(e, s)], etapas) for s in range(3)])
                LIN[cfg][e] = (y_t - m) * 1000
                SIG[cfg][e] = np.sqrt(np.clip(np.diag(S), 0, None) + ruido) * 1000
            h = gp.hiper()
            txt = (f'l = ({h["l_x (mm)"]:.0f}, {h["l_y (mm)"]:.0f}, {h["l_z (mm)"]:.0f}) mm  '
                   f's_f {h["s_f (um)"]:.0f} um  ' if not solo_lineal else '')
            print(f'    {cfg:9s}: s_lin {h["s_lin (ppm)"]:.0f} ppm  {txt}'
                  f's_D {h["s_D (um)"]:.0f} um  s_L {h["s_L (um)"]:.0f} um  '
                  f'({time.time() - t1:.0f} s)')
            if resid is not None:
                print(f'               sesgo dentro de muestra tras la pasada 1 (um): '
                      f'{np.array2string(resid.mean(0), precision=1)}')

        linea = '   '.join(f'{c} {rms(E[c][test], None):6.1f}' for c in E)
        print(f'    prueba (RMS um): {linea}   [{(time.time() - t0) / 60:.1f} min]')

    # ------------------------------------------------------------ PINN (opcional)
    cols = {'crudo': E_crudo, **E}
    if os.path.exists(PINN_NPZ):
        zp = np.load(PINN_NPZ, allow_pickle=True)
        if len(zp['nombres']) == nE and all(zp['nombres'] == nombres):
            dif = np.abs(zp['E_crudo'] - E_crudo).max()
            cols['pinn'] = zp['E_pinn']
            print(f'\nPINN cargado de {PINN_NPZ}' +
                  (f'  (OJO: crudo difiere hasta {dif:.1f} um; revisa REF_DIAM/REF_DIST)'
                   if dif > 0.5 else ''))
        else:
            print(f'\n{PINN_NPZ}: los escaneos no coinciden; no se incluye el PINN')

    # ------------------------------------------------------------ REPORTE
    comp = pd.DataFrame({'Mensurando': nom + ['GLOBAL', 'Mediana |error|', 'P95 |error|']})
    for c, M in cols.items():
        comp[c] = list(rms(M)) + [rms(M, None), np.median(np.abs(M)),
                                  np.percentile(np.abs(M), 95)]
    g_esc = rms(E['escala fija'], None)

    print('\n' + '=' * 78)
    print('COMPARACION  (RMS en um; k-fold por escaneo; prueba nunca vista al entrenar)')
    print('=' * 78)
    print(comp.to_string(index=False, float_format=lambda v: f'{v:9.1f}'))
    for c in [c for c in cols if c not in ('crudo', 'escala fija')]:
        mej = (1 - rms(cols[c]) / rms(E['escala fija'])) * 100
        print(f'  {c:10s}: mejora sobre la escala fija {(1 - rms(cols[c], None) / g_esc) * 100:5.1f}%'
              f'   (por mensurando: {", ".join(f"{v:.0f}" for v in mej)} %)')

    cob = {c: cobertura(E[c], SIG[c]) for c in GP_CONFIGS}
    for c in GP_CONFIGS:
        print(f'\nINCERTIDUMBRE PREDICHA ({c})')
        print(cob[c].to_string(index=False, float_format=lambda v: f'{v:8.1f}'))
        print(f'  error real (reajuste con rechazo 2.5 sigma) vs error predicho por el GP: '
              f'diferencia max {np.abs(E[c] - LIN[c]).max():.1f} um, RMS {rms(E[c] - LIN[c], None):.1f} um')

    pf = pd.DataFrame({'fold': np.arange(1, K_FOLDS + 1),
                       'escaneos de prueba': [len(t) for t in folds]})
    for c, M in cols.items():
        if c != 'crudo':
            pf[f'RMS {c} (um)'] = [rms(M[t], None) for t in folds]
    pf['Mejora gp vs escala fija (%)'] = (1 - pf['RMS gp (um)'] / pf['RMS escala fija (um)']) * 100
    print('\nPOR FOLD')
    print(pf.to_string(index=False, float_format=lambda v: f'{v:8.1f}'))

    hdf = pd.DataFrame(hip)
    print('\nHIPERPARAMETROS POR FOLD')
    print(hdf.to_string(index=False, float_format=lambda v: f'{v:8.1f}'))

    por_escaneo = pd.DataFrame({'archivo': nombres, 'fold': fold_de})
    for c, M in cols.items():
        por_escaneo[f'RMS {c} (um)'] = rms(M, 1)
    if 'pinn' in cols:
        gana = (rms(E['gp'], 1) < rms(cols['pinn'], 1)).sum()
        print(f'\nEscaneos donde el GP queda por debajo del PINN: {gana} de {nE}')

    hoja = lambda M: pd.DataFrame(M, columns=nom).assign(archivo=nombres, fold=fold_de)
    dfc = pd.DataFrame(coefs, columns=['fold'] + [f'a{i}' for i in range(6)] + ['escala fija'])
    hojas = {'Comparacion': comp, 'PorFold': pf, 'Cobertura_GP': cob['gp'],
             'Cobertura_GP_lineal': cob['gp_lineal'], 'Hiperparametros': hdf,
             'Coeficientes': dfc, 'PorEscaneo': por_escaneo, 'Crudo': hoja(E_crudo),
             'EscalaFija': hoja(E['escala fija']), 'Fisica': hoja(E['fisica']),
             'GP_lineal': hoja(E['gp_lineal']), 'GP': hoja(E['gp']),
             'Sigma_GP': hoja(SIG['gp']), 'Z_GP': hoja(E['gp'] / SIG['gp'])}
    destino = guarda_excel(os.path.join(SALIDA, 'resultados_gp_smcp.xlsx'), hojas)
    np.savez(os.path.join(SALIDA, 'resultados_gp_smcp.npz'), nombres=nombres, folds=fold_de,
             E_crudo=E_crudo, E_esc=E['escala fija'], E_fisica=E['fisica'],
             E_gp_lineal=E['gp_lineal'], E_gp=E['gp'], sigma_gp=SIG['gp'],
             sigma_gp_lineal=SIG['gp_lineal'], pred_gp=LIN['gp'], coefs=dfc.values)
    print(f'\nGuardado: {destino}')

    # ------------------------------------------------------------ MODELO FINAL
    if ENTRENA_FINAL:
        print('\nModelo final con todos los escaneos...')
        todos = np.arange(nE)
        s_f = escala_fija(E_crudo.mean(0))
        a, mu_z, sd_z = entrena_fisica(pts, id_esc, id_esf, todos, centros_ini, s_f)
        fis = {k: aplica_fisica(v, a, mu_z, sd_z) for k, v in grupos.items()}
        ys, reps, Os = zip(*[operador_escaneo([fis[(e, s)] for s in range(3)]) for e in todos])
        esf_tr = [[fis[(e, s)][sub[(e, s)]] for s in range(3)] for e in todos]
        etapas, _, _ = ajusta_pasadas(ys, reps, Os, esf_tr, False, th_prev.get('gp'))
        gp = etapas[0]
        ruta = os.path.join(SALIDA, 'modelo_gp_smcp')
        exporta_modelo(ruta, a, mu_z, sd_z, etapas)
        h = gp.hiper()
        print(f'  a0..a5: {np.array2string(a, precision=6, max_line_width=200)}')
        print(f'  s_lin {h["s_lin (ppm)"]:.0f} ppm   s_f {h["s_f (um)"]:.0f} um   '
              f'l = ({h["l_x (mm)"]:.0f}, {h["l_y (mm)"]:.0f}, {h["l_z (mm)"]:.0f}) mm   '
              f's_D {h["s_D (um)"]:.0f} um   s_L {h["s_L (um)"]:.0f} um')
        print(f'  guardado: {ruta}.npz / .mat  ({len(etapas)} pasadas, '
              f'{sum(len(st.P) for st in etapas)} puntos de soporte)')

    print(f'\nTiempo total: {(time.time() - t_ini) / 60:.1f} min')


if __name__ == '__main__':
    main()
