"""
pinn_smcp.py
------------
Compensacion del sistema SMCP (luz estructurada camara-proyector) mediante una red
neuronal informada por fisica (PINN) que corrige la nube de puntos punto por punto.

MODELO
    p_corregido = p + p * s(z, rho) + MLP(p) * escala_mlp

    s(z,rho) = a0 + a1*z + a2*z^2 + a3*rho^2 + a4*rho^4 + a5*z*rho^2
        a0        error de escala global (baseline |T| de la calibracion)
        a1,a2     deriva de escala de triangulacion dependiente de la profundidad
        a3,a4     distorsion radial residual no capturada por Brown-Conrady
        a5        acoplamiento profundidad-radio
        rho       = sqrt(x^2+y^2)/z  (tangente del angulo respecto al eje optico)
    El MLP absorbe lo que la parte fisica no explica (anisotropia, rizo de gamma).

PERDIDAS
    L_R    ( ||p_c - c_sk|| - R_ref_s )^2   esfericidad Y diametro a la vez,
                                            porque R_ref viene de la MMC
    L_D    ( ||c_i - c_j|| - D_ref_ij )^2   ancla la escala sobre la longitud del arreglo
    L_reg  ||delta||^2                      mantiene la correccion minima

    Los centros c_sk son parametros libres que se aprenden junto con la red, uno por
    cada (escaneo, esfera). La pose rigida del artefacto es inobservable y no importa:
    no afecta a los mensurandos.

VALIDACION
    k-fold POR ESCANEO, nunca por punto. Los escaneos de prueba nunca ven sus centros
    aprendidos: se ajustan despues de corregir, con el mismo pipeline geometrico +
    rechazo 2.5 sigma de siempre. El baseline honesto no es la nube cruda sino la
    correccion de escala fija estimada sobre los escaneos de entrenamiento.

USO
    1) Editar CARPETA_XYZ y SALIDA
    2) python pinn_smcp.py
       La primera corrida segmenta las nubes y cachea el resultado (tarda).
       Las siguientes reutilizan el cache.

REQUIERE: numpy, scipy, pandas, torch
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from itertools import combinations, permutations

# ---------------------------------------------------------------- CONFIGURACION
CARPETA_XYZ = r'C:\Users\oicasio\Desktop\MedicionEsferas'   # <-- EDITAR
SALIDA  = r'C:\Users\oicasio\Desktop\MedicionEsferas\Esferas limpias'   # <-- EDITAR
CACHE       = os.path.join(SALIDA, 'nubes_segmentadas.npz')

# referencia MMC
REF_DIAM = {'A': 100.0071, 'B': 100.0166, 'C': 100.0033}
REF_DIST = {('A', 'B'): 217.4531, ('A', 'C'): 196.0364, ('B', 'C'): 308.1691}
ETIQ = ['A', 'B', 'C']

# segmentacion / ajuste (identicos a procesa_30_escenas_smcp.py)
R_NOM, TOL_RANSAC, TOL_REFINA = 50.0, 1.0, 1.5
N_ITER, VOXEL, RADIO_CORTE, SIGMA_REJ = 4000, 2.0, 58.0, 2.5

# entrenamiento
K_FOLDS       = 5
PTS_POR_ESF   = 4000      # submuestreo para entrenar (la evaluacion usa todos)
EPOCAS        = 3000
LR_RED        = 1e-3
LR_CENTROS    = 1e-2
W_DIST        = 1.0       # peso de L_D
W_ESF         = 1.0       # peso de L_R (esfericidad + diametro)
NORMALIZA     = False
W_REG         = 3e-3      # peso de L_reg
ESCALA_MLP    = 0.5       # mm; acota la salida del MLP
OCULTAS       = 64
SEMILLA       = 0


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
    """delta(p) = p * s(z,rho)  +  MLP(p_norm) * ESCALA_MLP"""

    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer('mu', torch.tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.tensor(sd, dtype=torch.float32))
        self.a = nn.Parameter(torch.zeros(6))          # terminos fisicos
        self.mlp = nn.Sequential(
            nn.Linear(3, OCULTAS), nn.SiLU(),
            nn.Linear(OCULTAS, OCULTAS), nn.SiLU(),
            nn.Linear(OCULTAS, OCULTAS), nn.SiLU(),
            nn.Linear(OCULTAS, 3))
        # arranque suave: el MLP empieza en cero
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, p):
        pn = (p - self.mu) / self.sd
        z = pn[:, 2]
        rho = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) / p[:, 2]
        a = self.a
        s = (a[0] + a[1] * z + a[2] * z ** 2
             + a[3] * rho ** 2 + a[4] * rho ** 4 + a[5] * z * rho ** 2)
        delta = p * s.unsqueeze(1) + self.mlp(pn) * ESCALA_MLP
        return p + delta, delta


# ================================================================ ENTRENAMIENTO
def entrena(pts, id_esc, id_esf, escaneos_tr, centros_ini, verbose=True):
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

    P = torch.tensor(pts[sel], dtype=torch.float32)
    # indice compacto (escaneo local, esfera) -> fila en la tabla de centros
    remap = {e: i for i, e in enumerate(escaneos_tr)}
    fila = torch.tensor([remap[e] * 3 + s for e, s in
                         zip(id_esc[sel], id_esf[sel])], dtype=torch.long)

    C0 = np.array([centros_ini[(e, s)] for e in escaneos_tr for s in range(3)])
    C = nn.Parameter(torch.tensor(C0, dtype=torch.float32))
    Rref = torch.tensor([REF_DIAM[ETIQ[s]] / 2 for _ in escaneos_tr for s in range(3)],
                        dtype=torch.float32)

    modelo = CampoCorreccion(pts.mean(0), pts.std(0))
    opt = torch.optim.Adam([{'params': modelo.parameters(), 'lr': LR_RED},
                            {'params': [C], 'lr': LR_CENTROS}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCAS)

    # pares para L_D
    pares = [(i, j, REF_DIST[(ETIQ[i], ETIQ[j])]) for i, j in combinations(range(3), 2)]
    nE = len(escaneos_tr)

    for ep in range(EPOCAS):
        opt.zero_grad()
        Pc, delta = modelo(P)

        # L_R : esfericidad + diametro
        d = torch.linalg.norm(Pc - C[fila], dim=1) - Rref[fila]
        L_R = (d ** 2).mean()

        # L_D : distancias entre centros
        Cr = C.view(nE, 3, 3)
        L_D = torch.stack([(torch.linalg.norm(Cr[:, i] - Cr[:, j], dim=1) - ref) ** 2
                           for i, j, ref in pares]).mean()

        # L_reg : correccion minima
        L_reg = (delta ** 2).sum(1).mean()

        if ep == 0:
            nR = L_R.detach().clamp(min=1e-12) if NORMALIZA else torch.tensor(1.0)
            nD = L_D.detach().clamp(min=1e-12) if NORMALIZA else torch.tensor(1.0)

        loss = W_ESF * L_R / nR + W_DIST * L_D / nD + W_REG * L_reg / nR
        loss.backward()
        opt.step()
        sched.step()

        if verbose and (ep % 500 == 0 or ep == EPOCAS - 1):
            print(f'    ep {ep:5d}  L_R {L_R.item()*1e6:8.1f} um^2  '
                  f'L_D {L_D.item()*1e6:9.1f} um^2  '
                  f'|delta| {delta.norm(dim=1).mean().item()*1000:7.1f} um')
    if verbose:
        a = modelo.a.detach().numpy()
        print('    coef a0..a5:', np.array2string(a, precision=5, suppress_small=False))

    return modelo


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


def main(tag=''):
    if not os.path.exists(CACHE):
        preprocesa(CARPETA_XYZ, CACHE)

    z = np.load(CACHE, allow_pickle=True)
    pts, id_esc, id_esf = z['pts'].astype(np.float64), z['id_esc'], z['id_esf']
    nombres = z['nombres']
    nE = len(nombres)
    print(f'\n{nE} escaneos, {len(pts):,} puntos en total '
          f'({len(pts)//(nE*3):,} por esfera en promedio)\n')

    # subconjuntos y centros iniciales
    grupos = {(e, s): pts[(id_esc == e) & (id_esf == s)] for e in range(nE) for s in range(3)}
    centros_ini = {k: ajusta(v)[0] for k, v in grupos.items()}

    # errores crudos por escaneo
    E_crudo = np.array([mensurandos([grupos[(e, s)] for s in range(3)]) for e in range(nE)])

    # k-fold por escaneo
    rng = np.random.default_rng(SEMILLA)
    orden = rng.permutation(nE)
    folds = np.array_split(orden, K_FOLDS)

    E_esc = np.full_like(E_crudo, np.nan)
    E_pinn = np.full_like(E_crudo, np.nan)

    for k, test in enumerate(folds):
        train = np.setdiff1d(orden, test)
        print(f'--- Fold {k+1}/{K_FOLDS}: {len(train)} entrenamiento, {len(test)} prueba')

        # baseline: escala fija estimada en entrenamiento
        s = escala_fija(E_crudo[train].mean(0))
        for e in test:
            E_esc[e] = mensurandos([grupos[(e, ss)] / s for ss in range(3)])

        # PINN
        modelo = entrena(pts, id_esc, id_esf, train, centros_ini)
        modelo.eval()
        with torch.no_grad():
            for e in test:
                corr = []
                for ss in range(3):
                    X = torch.tensor(grupos[(e, ss)], dtype=torch.float32)
                    corr.append(modelo(X)[0].numpy().astype(np.float64))
                E_pinn[e] = mensurandos(corr)

    # ---- reporte
    nom = [f'D_{L}' for L in ETIQ] + [f'd_{ETIQ[i]}-{ETIQ[j]}'
                                      for i, j in combinations(range(3), 2)]
    rms = lambda M: np.sqrt((M ** 2).mean(0))
    tabla = pd.DataFrame({'Mensurando': nom,
                          'RMS crudo (um)': rms(E_crudo),
                          'RMS escala fija (um)': rms(E_esc),
                          'RMS PINN (um)': rms(E_pinn)})
    tabla['Mejora vs escala fija (%)'] = (1 - tabla['RMS PINN (um)'] /
                                          tabla['RMS escala fija (um)']) * 100

    g = [np.sqrt((M ** 2).mean()) for M in (E_crudo, E_esc, E_pinn)]
    print('\n' + '=' * 78)
    print('RESULTADO  (k-fold por escaneo; ningun escaneo de prueba se uso al entrenar)')
    print('=' * 78)
    print(tabla.to_string(index=False, float_format=lambda v: f'{v:9.1f}'))
    print(f'\nGLOBAL   crudo {g[0]:7.1f}   escala fija {g[1]:7.1f}   PINN {g[2]:7.1f}  um')
    print(f'  mejora del PINN sobre la escala fija : {(1-g[2]/g[1])*100:5.1f}%   <-- numero a reportar')
    print(f'  mejora del PINN sobre la nube cruda  : {(1-g[2]/g[0])*100:5.1f}%   (incluye la escala trivial)')

    xls = os.path.join(SALIDA, f'resultados_pinn_smcp{tag}.xlsx')
    with pd.ExcelWriter(xls, engine='openpyxl') as w:
        tabla.to_excel(w, sheet_name='Resumen', index=False)
        for M, h in ((E_crudo, 'Crudo'), (E_esc, 'EscalaFija'), (E_pinn, 'PINN')):
            (pd.DataFrame(M, columns=nom)
               .assign(archivo=nombres)
               .to_excel(w, sheet_name=h, index=False))
    np.savez(os.path.join(SALIDA, f'resultados_pinn_smcp{tag}.npz'),
             E_crudo=E_crudo, E_esc=E_esc, E_pinn=E_pinn, nombres=nombres)
    print(f'\nGuardado: {xls}')

if __name__ == '__main__':
    for w in [0.1, 0.25, 0.5]:
        W_DIST = w
        print(f'\n{"#"*78}\n### W_DIST = {w}\n{"#"*78}')
        main(tag=f'_wd{w}')
