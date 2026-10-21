"""
exporta_modelo_yml.py
---------------------
Convierte el modelo final del GP (modelo_gp_smcp.npz, generado por gp_smcp.py) a un
.yml de cv::FileStorage (OpenCV 2.4.13) para SMCP.exe, y lo liga a la calibracion con
la que se reconstruyeron los 37 escaneos de entrenamiento.

El .yml contiene:
    a (1x6), mu_z, sd_z          bloque fisico s(z, rho)
    l (1x3), pasadas             longitudes de correlacion del Matern 5/2
    p0_k (1x3), B_k (3x3)        tendencia lineal de la pasada k
    P_rep_k (Nx3), W_k (Nx3)     puntos de soporte y pesos de la pasada k
    caja_min / caja_max (1x3)    volumen cubierto por los puntos de soporte (mm)
    calib_*                      copia de cam_K, cam_kc, proj_K, proj_kc, R, T
                                 (SMCP.exe no compensa si la calibracion cargada difiere)

Al final relee el .yml y verifica que la correccion evaluada con lo escrito es identica
a la del .npz (diferencia ~0) en una nube de prueba.

USO
    python exporta_modelo_yml.py

REQUIERE: numpy
"""

import os
import re
import numpy as np

# ---------------------------------------------------------------- CONFIGURACION
CARPETA     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resultados', 'gp')
MODELO_NPZ  = os.path.join(CARPETA, 'modelo_gp_smcp.npz')
MODELO_YML  = os.path.join(CARPETA, 'modelo_gp_smcp.yml')
CALIBRACION = r'C:\Users\oicasio\Desktop\CMCP\YA2\calibration.yml'      # la de los 37 escaneos
XYZ_PRUEBA  = r'C:\Users\oicasio\Desktop\MedicionEsferas\2026-sep.-15_09.57.19.561.xyz'

CLAVES_CALIB = ['cam_K', 'cam_kc', 'proj_K', 'proj_kc', 'R', 'T']
S5 = np.sqrt(5.0)


# ================================================================ YAML DE OPENCV
def lee_matrices_opencv(ruta):
    """Lee las matrices !!opencv-matrix de un .yml de cv::FileStorage."""
    txt = open(ruta, encoding='utf-8').read()
    patron = re.compile(r'^(\w+): !!opencv-matrix\s+rows: (\d+)\s+cols: (\d+)\s+dt: \w+\s+'
                        r'data: \[([^\]]*)\]', re.M)
    mats = {}
    for nombre, filas, cols, datos in patron.findall(txt):
        v = np.array([float(x) for x in datos.replace('\n', ' ').split(',') if x.strip()])
        mats[nombre] = v.reshape(int(filas), int(cols))
    return mats


def _matriz(nombre, M):
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    datos = [f'{v:.17e}' for v in M.ravel()]
    lineas = [', '.join(datos[i:i + 3]) for i in range(0, len(datos), 3)]
    return (f'{nombre}: !!opencv-matrix\n   rows: {M.shape[0]}\n   cols: {M.shape[1]}\n'
            f'   dt: d\n   data: [ ' + ',\n       '.join(lineas) + ' ]\n')


def escribe_yml(ruta, m, calib, ruta_calib):
    pasadas = int(m['pasadas'])
    P_todos = np.vstack([m[f'P_rep_{k}'] for k in range(1, pasadas + 1)])
    partes = ['%YAML:1.0\n',
              'descripcion: "Compensacion SMCP: fisica s(z,rho) + GP (lineal + Matern 5/2 ARD). '
              'Generado por exporta_modelo_yml.py desde modelo_gp_smcp.npz"\n',
              'version: 1\n',
              f'calibracion_archivo: "{ruta_calib.replace(os.sep, "/")}"\n',
              f'mu_z: {float(m["mu_z"]):.17e}\n',
              f'sd_z: {float(m["sd_z"]):.17e}\n',
              f'pasadas: {pasadas}\n',
              _matriz('a', m['a']),
              _matriz('l', m['l']),
              _matriz('caja_min', P_todos.min(0)),
              _matriz('caja_max', P_todos.max(0))]
    for k in range(1, pasadas + 1):
        partes += [_matriz(f'p0_{k}', m[f'p0_{k}']), _matriz(f'B_{k}', m[f'B_{k}']),
                   _matriz(f'P_rep_{k}', m[f'P_rep_{k}']), _matriz(f'W_{k}', m[f'W_{k}'])]
    for c in CLAVES_CALIB:
        partes.append(_matriz(f'calib_{c}', calib[c]))
    with open(ruta, 'w', encoding='ascii', newline='\n') as f:
        f.write(''.join(partes))


# ================================================================ CORRECCION
def aplica(P, m, bloque=4000):
    """Misma formula que gp_smcp.aplica_modelo (sin depender de pandas)."""
    P = np.asarray(P, dtype=np.float64)
    a = np.ravel(m['a'])
    zn = (P[:, 2] - float(m['mu_z'])) / float(m['sd_z'])
    rho2 = (P[:, 0] ** 2 + P[:, 1] ** 2) / P[:, 2] ** 2
    s = a[0] + a[1] * zn + a[2] * zn ** 2 + a[3] * rho2 + a[4] * rho2 ** 2 + a[5] * zn * rho2
    X = P * (1.0 + s)[:, None]
    l = np.ravel(m['l'])
    for k in range(1, int(m['pasadas']) + 1):
        e = (X - np.ravel(m[f'p0_{k}'])) @ np.asarray(m[f'B_{k}']).T
        Ps = m[f'P_rep_{k}'] / l
        for i in range(0, len(X), bloque):
            Xs = X[i:i + bloque] / l
            r = np.sqrt(np.maximum((Xs ** 2).sum(1)[:, None] + (Ps ** 2).sum(1)[None, :]
                                   - 2 * Xs @ Ps.T, 0.0))
            e[i:i + bloque] += ((1 + S5 * r + 5.0 / 3.0 * r * r) * np.exp(-S5 * r)) @ m[f'W_{k}']
        X = X - e
    return X


def main():
    m = dict(np.load(MODELO_NPZ))
    calib = lee_matrices_opencv(CALIBRACION)
    faltan = [c for c in CLAVES_CALIB if c not in calib]
    if faltan:
        raise SystemExit(f'{CALIBRACION}: faltan {faltan}')
    fy_fx = calib['proj_K'][1, 1] / calib['proj_K'][0, 0]
    print(f'Calibracion: {CALIBRACION}  (proyector fy/fx = {fy_fx:.4f})')

    escribe_yml(MODELO_YML, m, calib, CALIBRACION)
    print(f'Modelo escrito: {MODELO_YML}  ({os.path.getsize(MODELO_YML) / 1e6:.2f} MB, '
          f'{int(m["pasadas"])} pasada(s), {len(m["P_rep_1"])} puntos de soporte)')

    # verificacion: releer el .yml y comparar la correccion contra el .npz
    y = lee_matrices_opencv(MODELO_YML)
    txt = open(MODELO_YML).read()
    for clave in ('mu_z', 'sd_z', 'pasadas'):
        y[clave] = float(re.search(rf'^{clave}: (\S+)', txt, re.M).group(1))
    for c in CLAVES_CALIB:
        assert np.array_equal(y[f'calib_{c}'], calib[c]), c
    P = np.loadtxt(XYZ_PRUEBA)[:, :3]
    d = np.abs(aplica(P, y) - aplica(P, m)).max() * 1000
    corr = np.linalg.norm(aplica(P, m) - P, axis=1) * 1000
    print(f'Verificacion en {os.path.basename(XYZ_PRUEBA)} ({len(P):,} puntos): '
          f'correccion media {corr.mean():.0f} um, max {corr.max():.0f} um; '
          f'diferencia .yml vs .npz = {d:.2e} um')


if __name__ == '__main__':
    main()
