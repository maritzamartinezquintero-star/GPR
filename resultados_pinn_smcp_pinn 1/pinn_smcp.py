"""
pinn_smcp.py  (version 2)
-------------------------
Compensacion del sistema SMCP (luz estructurada camara-proyector) mediante una red
neuronal informada por fisica (PINN) que corrige la nube de puntos punto por punto.

MODELO
    p_corregido = p + p * s(z, rho) + d_mlp(p)

    s(z,rho) = a0 + a1*z + a2*z^2 + a3*rho^2 + a4*rho^4 + a5*z*rho^2
        p*s mueve cada punto a lo largo de su rayo de vista (desde la camara).
        a0        error de escala global (base |T|). ARRANCA en la escala fija.
        a1,a2     deriva de escala con la profundidad (z normalizada)
        a3,a4     distorsion radial residual (rho = tangente del angulo al eje optico)
        a5        acoplamiento profundidad-radio
    d_mlp(p) = ESCALA_MLP * tanh(MLP(p_norm))
        Absorbe lo que la fisica no explica. Salida acotada a +-ESCALA_MLP por eje.
        Su entrada se recorta a la caja de trabajo del entrenamiento: fuera de ella
        el MLP no extrapola.

PERDIDAS
    L_R      (||p_c - c_sk|| - R_ref)^2    esfericidad + diametro (R_ref de la MMC)
    L_D      (||c_i - c_j|| - D_ref)^2     distancias entre centros (ancla de escala)
    L_mag    ||d_mlp||^2                   el MLP solo corrige lo necesario
    L_suave  ||grad d_mlp||^2              en puntos de colocacion de toda la caja:
                                           el campo de error de un sistema optico es
                                           suave; evita contracciones locales falsas

CAMBIOS RESPECTO A LA VERSION 1
    1) La regularizacion ya NO penaliza la parte fisica. En v1, L_reg = ||p*s||^2:
       como los puntos estan a ~1 m de la camara, corregir una escala de ~0.8%
       costaba milimetros de desplazamiento. La red evitaba usar a0 y "encogia" las
       esferas con el MLP (contracciones locales), que no generalizan.
    2) a0 arranca en la escala fija de entrenamiento: el modelo parte del baseline.
    3) MLP acotado (tanh), con entrada recortada y suavidad en toda la caja.
    4) Normalizacion y caja calculadas SOLO con escaneos de entrenamiento.
    5) Varias configuraciones en una sola corrida + diagnostico por escaneo.

VALIDACION
    k-fold POR ESCANEO (mismos folds que v1 con SEMILLA=0). Los escaneos de prueba
    nunca ven sus centros aprendidos: se ajustan despues de corregir con el mismo
    pipeline geometrico + rechazo 2.5 sigma. Baseline honesto: escala fija estimada
    con los escaneos de entrenamiento.

USO
    1) Revisar CARPETA_XYZ y SALIDA
    2) python pinn_smcp.py   (reutiliza nubes_segmentadas.npz si ya existe)

REQUIERE: numpy, scipy, pandas, openpyxl, torch
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from itertools import combinations, permutations

# ---------------------------------------------------------------- CONFIGURACION
CARPETA_XYZ = r'C:\Users\Maritza\Desktop\CENAM\PINNPINN'
SALIDA      = r'C:\Users\Maritza\Desktop\CENAM\PINNPINN'
CACHE       = os.path.join(SALIDA, 'nubes_segmentadas.npz')

# referencia MMC
REF_DIAM = {'A': 100.0071, 'B': 100.0166, 'C': 100.0033}
REF_DIST = {('A', 'B'): 217.4531, ('A', 'C'): 196.0364, ('B', 'C'): 308.1691}
ETIQ = ['A', 'B', 'C']

# segmentacion / ajuste (identicos a v1; NO cambiar o el cache deja de ser valido)
R_NOM, TOL_RANSAC, TOL_REFINA = 50.0, 1.0, 1.5
N_ITER, VOXEL, RADIO_CORTE, SIGMA_REJ = 4000, 2.0, 58.0, 2.5

# entrenamiento
K_FOLDS     = 5
PTS_POR_ESF = 2000      # submuestreo para entrenar (la evaluacion usa todos los puntos)
EPOCAS      = 2000
LR_FISICA   = 2e-4      # coeficientes a0..a5
LR_RED      = 1e-3      # pesos del MLP
LR_CENTROS  = 1e-2      # centros de las esferas (mm)
W_DIST      = 1.0       # peso de L_D
W_MAG       = 3e-3      # peso de L_mag   (solo MLP)
W_SUAVE     = 100.0     # peso de L_suave (solo MLP)
N_COLOC     = 4096      # puntos de colocacion por epoca para L_suave
H_DIF       = 2.0       # mm; paso de diferencias finitas para el gradiente del MLP
MARGEN_CAJA = 0.10      # la caja de trabajo se amplia 10% por lado
ESCALA_MLP  = 0.5       # mm; cota de la salida del MLP por eje
OCULTAS     = 64
SEMILLA     = 0

# configuraciones que se comparan en una sola corrida
CONFIGS = [
    ('fisica', dict(usa_mlp=False)),     # solo los 6 coeficientes (rapido)
    ('pinn',   dict(usa_mlp=True)),      # fisica + MLP
    # ('pinn_sin_suave', dict(usa_mlp=True, w_suave=0.0)),   # ablacion opcional
]


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
    c, r = fit_alg(X)
    for _ in range(max_iter):
        c, r, res = fit_geo(X, c, r)
        keep = np.abs(res) < SIGMA_REJ * res.std()
        if keep.all():
            break
        X = X[keep]
    c, r, res = fit_geo(X, c, r)
    return c, r, res


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
    """Segmenta todos los .xyz y guarda un cache compacto."""
    rng = np.random.default_rng(SEMILLA)
    archivos = sorted(glob.glob(os.path.join(carpeta, '*.xyz')))
    print(f'Segmentando {len(archivos)} escaneos (solo la primera vez)...')

    pts, id_esc, id_esf, nombres = [], [], [], []
    for k, f in enumerate(archivos):
        P = np.loadtxt(f)[:, :3]
        sem = detecta(P, rng)
        if len(sem) < 3:
            print(f'  {os.path.basename(f)}: {len(sem)} esferas -- OMITIDO')
            continue
        mapa = etiqueta(sem)
        ie = len(nombres)
        for s, L in enumerate(ETIQ):
            X = P[np.linalg.norm(P - sem[mapa[L]], axis=1) < RADIO_CORTE]
            # limpieza previa con el ajuste robusto, para no meter atipicos al PINN
            c, r, res = ajusta(X)
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


# ================================================================ MODELO
class CampoCorreccion(nn.Module):
    """p_c = p + p * s(z,rho) + ESCALA_MLP * tanh(MLP(p_norm recortada))"""

    def __init__(self, mu, sd, lo, hi, a0, usa_mlp):
        super().__init__()
        f32 = dict(dtype=torch.float32)
        self.register_buffer('mu', torch.tensor(mu, **f32))
        self.register_buffer('sd', torch.tensor(sd, **f32))
        self.register_buffer('lo', torch.tensor((lo - mu) / sd, **f32))
        self.register_buffer('hi', torch.tensor((hi - mu) / sd, **f32))
        a = torch.zeros(6)
        a[0] = float(a0)                                   # arranque = escala fija
        self.a = nn.Parameter(a)
        self.usa_mlp = usa_mlp
        self.mlp = nn.Sequential(
            nn.Linear(3, OCULTAS), nn.SiLU(),
            nn.Linear(OCULTAS, OCULTAS), nn.SiLU(),
            nn.Linear(OCULTAS, OCULTAS), nn.SiLU(),
            nn.Linear(OCULTAS, 3))
        # arranque suave: el MLP empieza en cero
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def s_fisica(self, p):
        z = (p[:, 2] - self.mu[2]) / self.sd[2]
        rho2 = (p[:, 0] ** 2 + p[:, 1] ** 2) / p[:, 2] ** 2
        a = self.a
        return (a[0] + a[1] * z + a[2] * z ** 2
                + a[3] * rho2 + a[4] * rho2 ** 2 + a[5] * z * rho2)

    def d_mlp(self, p):
        if not self.usa_mlp:
            return torch.zeros_like(p)
        pn = (p - self.mu) / self.sd
        pn = torch.maximum(torch.minimum(pn, self.hi), self.lo)   # no extrapola
        return ESCALA_MLP * torch.tanh(self.mlp(pn))

    def forward(self, p):
        dm = self.d_mlp(p)
        return p + p * self.s_fisica(p).unsqueeze(1) + dm, dm


def gradiente_mlp(modelo, Q):
    """Derivada direccional del MLP (adimensional) por diferencias centrales."""
    u = torch.randn_like(Q)
    u = u / u.norm(dim=1, keepdim=True)
    g = (modelo.d_mlp(Q + H_DIF * u) - modelo.d_mlp(Q - H_DIF * u)) / (2 * H_DIF)
    return (g ** 2).sum(1).mean()


# ================================================================ ENTRENAMIENTO
def entrena(pts, id_esc, id_esf, escaneos_tr, centros_ini, s_fija,
            usa_mlp=True, w_suave=None, verbose=True):
    if w_suave is None:
        w_suave = W_SUAVE
    torch.manual_seed(SEMILLA)
    rng = np.random.default_rng(SEMILLA)

    # submuestreo balanceado por (escaneo, esfera)
    sel = []
    for e in escaneos_tr:
        for s in range(3):
            idx = np.where((id_esc == e) & (id_esf == s))[0]
            if len(idx) > PTS_POR_ESF:
                idx = rng.choice(idx, PTS_POR_ESF, replace=False)
            sel.append(idx)
    sel = np.concatenate(sel)
    Ptr = pts[sel]
    P = torch.tensor(Ptr, dtype=torch.float32)

    remap = {e: i for i, e in enumerate(escaneos_tr)}
    fila = torch.tensor([remap[e] * 3 + s for e, s in
                         zip(id_esc[sel], id_esf[sel])], dtype=torch.long)

    # todo lo que define al modelo sale SOLO de los escaneos de entrenamiento
    mu, sd = Ptr.mean(0), Ptr.std(0)
    ext = Ptr.max(0) - Ptr.min(0)
    lo = Ptr.min(0) - MARGEN_CAJA * ext
    hi = Ptr.max(0) + MARGEN_CAJA * ext
    a0 = 1.0 / s_fija - 1.0

    # centros iniciales coherentes con el arranque (escala respecto a la camara)
    C0 = np.array([centros_ini[(e, s)] / s_fija for e in escaneos_tr for s in range(3)])
    C = nn.Parameter(torch.tensor(C0, dtype=torch.float32))
    Rref = torch.tensor([REF_DIAM[ETIQ[s]] / 2 for _ in escaneos_tr for s in range(3)],
                        dtype=torch.float32)

    modelo = CampoCorreccion(mu, sd, lo, hi, a0, usa_mlp)
    opt = torch.optim.Adam([{'params': [modelo.a], 'lr': LR_FISICA},
                            {'params': modelo.mlp.parameters(), 'lr': LR_RED},
                            {'params': [C], 'lr': LR_CENTROS}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCAS)

    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)
    pares = [(i, j, REF_DIST[(ETIQ[i], ETIQ[j])]) for i, j in combinations(range(3), 2)]
    nE = len(escaneos_tr)
    cero = torch.zeros(())

    for ep in range(EPOCAS):
        opt.zero_grad()
        Pc, dm = modelo(P)

        # L_R : esfericidad + diametro
        d = torch.linalg.norm(Pc - C[fila], dim=1) - Rref[fila]
        L_R = (d ** 2).mean()

        # L_D : distancias entre centros
        Cr = C.view(nE, 3, 3)
        L_D = torch.stack([(torch.linalg.norm(Cr[:, i] - Cr[:, j], dim=1) - ref) ** 2
                           for i, j, ref in pares]).mean()

        # L_mag y L_suave : solo sobre el MLP (la fisica no se penaliza)
        if usa_mlp:
            L_mag = (dm ** 2).sum(1).mean()
            Q = lo_t + (hi_t - lo_t) * torch.rand(N_COLOC, 3)
            L_suave = gradiente_mlp(modelo, Q)
        else:
            L_mag = L_suave = cero

        loss = L_R + W_DIST * L_D + W_MAG * L_mag + w_suave * L_suave
        loss.backward()
        opt.step()
        sched.step()

        if verbose and (ep % 500 == 0 or ep == EPOCAS - 1):
            print(f'    ep {ep:5d}  esfera {L_R.item() ** 0.5 * 1000:7.1f} um  '
                  f'dist {L_D.item() ** 0.5 * 1000:7.1f} um  '
                  f'|mlp| {dm.norm(dim=1).mean().item() * 1000:6.1f} um  '
                  f'grad {L_suave.item() ** 0.5 * 1e6:7.1f} ppm')

    a = modelo.a.detach().numpy().copy()
    if verbose:
        print('    coef a0..a5:', np.array2string(a, precision=6, max_line_width=200))
    return modelo, a


# ================================================================ EVALUACION
def mensurandos(pts_por_esf):
    """Ajusta las 3 esferas y devuelve los 6 errores en um."""
    cen, dia = [], []
    for s in range(3):
        c, r, _ = ajusta(pts_por_esf[s])
        cen.append(c); dia.append(2 * r)
    e = [(dia[s] - REF_DIAM[ETIQ[s]]) * 1000 for s in range(3)]
    for i, j in combinations(range(3), 2):
        e.append((np.linalg.norm(cen[i] - cen[j]) - REF_DIST[(ETIQ[i], ETIQ[j])]) * 1000)
    return np.array(e)


def escala_fija(errs_tr):
    """Factor de escala unico estimado sobre los escaneos de entrenamiento."""
    ref = np.array([REF_DIAM[L] for L in ETIQ] +
                   [REF_DIST[(ETIQ[i], ETIQ[j])] for i, j in combinations(range(3), 2)])
    return 1.0 + (errs_tr / 1000 / ref).mean()


def casquete(X, c):
    """Semiangulo (grados) del casquete visible de la esfera."""
    u = X - c
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    m = np.linalg.norm(u.mean(0))
    return np.degrees(np.arccos(np.clip(2 * m - 1, -1, 1)))


def rms(M, eje=0):
    return np.sqrt((M ** 2).mean(eje))


def guarda_excel(ruta, hojas):
    """Escribe el Excel; si el archivo esta abierto, guarda con otro nombre."""
    try:
        destino = ruta
        with pd.ExcelWriter(destino, engine='openpyxl') as w:
            for nombre, df in hojas.items():
                df.to_excel(w, sheet_name=nombre, index=False)
    except PermissionError:
        destino = ruta.replace('.xlsx', time.strftime('_%H%M%S.xlsx'))
        with pd.ExcelWriter(destino, engine='openpyxl') as w:
            for nombre, df in hojas.items():
                df.to_excel(w, sheet_name=nombre, index=False)
        print(f'  (el archivo original estaba abierto; se guardo como {destino})')
    return destino


def main():
    t_ini = time.time()
    if not os.path.exists(CACHE):
        preprocesa(CARPETA_XYZ, CACHE)

    z = np.load(CACHE, allow_pickle=True)
    pts, id_esc, id_esf = z['pts'].astype(np.float64), z['id_esc'], z['id_esf']
    nombres = z['nombres']
    nE = len(nombres)
    print(f'\n{nE} escaneos, {len(pts):,} puntos en total '
          f'({len(pts) // (nE * 3):,} por esfera en promedio)')

    # subconjuntos, centros iniciales y errores crudos (una sola vez)
    print('Ajustando esferas crudas...')
    grupos = {(e, s): pts[(id_esc == e) & (id_esf == s)] for e in range(nE) for s in range(3)}
    centros_ini = {k: ajusta(v)[0] for k, v in grupos.items()}
    E_crudo = np.array([mensurandos([grupos[(e, s)] for s in range(3)]) for e in range(nE)])

    # k-fold por escaneo (mismos folds que v1)
    rng = np.random.default_rng(SEMILLA)
    orden = rng.permutation(nE)
    folds = np.array_split(orden, K_FOLDS)
    fold_de = np.empty(nE, dtype=int)
    for k, test in enumerate(folds):
        fold_de[test] = k + 1

    # baseline: escala fija estimada en entrenamiento
    E_esc = np.full_like(E_crudo, np.nan)
    s_fold = []
    for test in folds:
        train = np.setdiff1d(orden, test)
        s = escala_fija(E_crudo[train].mean(0))
        s_fold.append(s)
        for e in test:
            E_esc[e] = mensurandos([grupos[(e, ss)] / s for ss in range(3)])

    # diagnostico: posicion, cobertura y cercania a esferas de entrenamiento
    filas = []
    for k, test in enumerate(folds):
        train = np.setdiff1d(orden, test)
        cen_tr = np.array([centros_ini[(e, s)] for e in train for s in range(3)])
        for e in test:
            for s in range(3):
                c = centros_ini[(e, s)]
                filas.append({'archivo': nombres[e], 'fold': k + 1, 'esfera': ETIQ[s],
                              'x (mm)': c[0], 'y (mm)': c[1], 'z (mm)': c[2],
                              'rho': np.hypot(c[0], c[1]) / c[2],
                              'n puntos': len(grupos[(e, s)]),
                              'casquete (grados)': casquete(grupos[(e, s)], c),
                              'dist. a esfera de entrenamiento (mm)':
                                  np.linalg.norm(cen_tr - c, axis=1).min()})
    diag = pd.DataFrame(filas)

    nom = [f'D_{L}' for L in ETIQ] + [f'd_{ETIQ[i]}-{ETIQ[j]}'
                                      for i, j in combinations(range(3), 2)]
    resultados = {}

    for nombre_cfg, cfg in CONFIGS:
        t0 = time.time()
        print('\n' + '#' * 78 + f'\n### CONFIGURACION: {nombre_cfg}  {cfg}\n' + '#' * 78)
        E_mod = np.full_like(E_crudo, np.nan)
        coefs = []

        for k, test in enumerate(folds):
            train = np.setdiff1d(orden, test)
            print(f'--- Fold {k + 1}/{K_FOLDS}: {len(train)} entrenamiento, {len(test)} prueba')
            modelo, a = entrena(pts, id_esc, id_esf, train, centros_ini, s_fold[k], **cfg)
            coefs.append(a)

            modelo = modelo.double().eval()
            with torch.no_grad():
                for e in test:
                    corr = [modelo(torch.tensor(grupos[(e, ss)], dtype=torch.float64))[0]
                            .numpy() for ss in range(3)]
                    E_mod[e] = mensurandos(corr)
            print(f'    prueba de este fold: escala fija {rms(E_esc[test], None):6.1f} um'
                  f'   {nombre_cfg} {rms(E_mod[test], None):6.1f} um')

        resultados[nombre_cfg] = E_mod

        # ---- reporte de la configuracion
        tabla = pd.DataFrame({'Mensurando': nom,
                              'RMS crudo (um)': rms(E_crudo),
                              'RMS escala fija (um)': rms(E_esc),
                              f'RMS {nombre_cfg} (um)': rms(E_mod)})
        tabla['Mejora vs escala fija (%)'] = (1 - tabla[f'RMS {nombre_cfg} (um)'] /
                                              tabla['RMS escala fija (um)']) * 100
        g = [rms(M, None) for M in (E_crudo, E_esc, E_mod)]
        print('\n' + '=' * 78)
        print(f'RESULTADO {nombre_cfg}  (k-fold por escaneo; prueba nunca vista al entrenar)')
        print('=' * 78)
        print(tabla.to_string(index=False, float_format=lambda v: f'{v:9.1f}'))
        print(f'\nGLOBAL   crudo {g[0]:7.1f}   escala fija {g[1]:7.1f}   '
              f'{nombre_cfg} {g[2]:7.1f}  um')
        print(f'  mejora sobre la escala fija : {(1 - g[2] / g[1]) * 100:5.1f}%'
              f'   <-- numero a reportar')
        print(f'  mediana |error|: escala fija {np.median(np.abs(E_esc)):6.1f} um   '
              f'{nombre_cfg} {np.median(np.abs(E_mod)):6.1f} um')
        print(f'  tiempo: {(time.time() - t0) / 60:.1f} min')

        hoja = lambda M: pd.DataFrame(M, columns=nom).assign(archivo=nombres, fold=fold_de)
        dfc = pd.DataFrame(coefs, columns=[f'a{i}' for i in range(6)])
        dfc.insert(0, 'fold', np.arange(1, K_FOLDS + 1))
        dfc['escala fija'] = s_fold
        destino = guarda_excel(os.path.join(SALIDA, f'resultados_pinn_smcp_{nombre_cfg}.xlsx'),
                               {'Resumen': tabla, 'Crudo': hoja(E_crudo),
                                'EscalaFija': hoja(E_esc), 'Modelo': hoja(E_mod),
                                'Coeficientes': dfc, 'Diagnostico': diag})
        np.savez(os.path.join(SALIDA, f'resultados_pinn_smcp_{nombre_cfg}.npz'),
                 E_crudo=E_crudo, E_esc=E_esc, E_pinn=E_mod, nombres=nombres,
                 coefs=np.array(coefs), folds=fold_de)
        print(f'  guardado: {destino}')

    # ---- comparacion final
    cols = {'crudo': E_crudo, 'escala fija': E_esc, **resultados}
    comp = pd.DataFrame({'Mensurando': nom + ['GLOBAL', 'Mediana |error|']})
    for c, M in cols.items():
        comp[c] = list(rms(M)) + [rms(M, None), np.median(np.abs(M))]
    por_escaneo = pd.DataFrame({'archivo': nombres, 'fold': fold_de})
    for c, M in cols.items():
        por_escaneo[f'RMS {c} (um)'] = rms(M, 1)

    print('\n' + '=' * 78)
    print('COMPARACION FINAL  (RMS en um; k-fold por escaneo)')
    print('=' * 78)
    print(comp.to_string(index=False, float_format=lambda v: f'{v:9.1f}'))
    g_esc = rms(E_esc, None)
    for c, M in resultados.items():
        print(f'  mejora de {c:>8s} sobre la escala fija: {(1 - rms(M, None) / g_esc) * 100:5.1f}%')

    ultimo = list(resultados)[-1]
    peor = por_escaneo.sort_values(f'RMS {ultimo} (um)', ascending=False).head(5)
    print(f'\nEscaneos con mayor error ({ultimo}); revisa su hoja Diagnostico:')
    print(peor.to_string(index=False, float_format=lambda v: f'{v:8.1f}'))

    destino = guarda_excel(os.path.join(SALIDA, 'comparacion_smcp.xlsx'),
                           {'Comparacion': comp, 'PorEscaneo': por_escaneo,
                            'Diagnostico': diag})
    print(f'\nGuardado: {destino}')
    print(f'Tiempo total: {(time.time() - t_ini) / 60:.1f} min')


if __name__ == '__main__':
    main()
