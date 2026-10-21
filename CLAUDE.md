# Proyecto CENAM — Compensación de sistemas de medición por coordenadas con redes neuronales

Estancia profesional de Maritza Martínez Quintero (UNAQ, IDMA) en CENAM, Querétaro.
Asesor: **Octavio Icasio Hernández** · Co-asesor: **Bala (Muralikrishnan)** · Colaborador SMCP: **José Luis Aguilera Luzanía**.
Idioma de trabajo: **español**. Respuestas cortas, tablas/listas, nada de texto de relleno.

---

## Objetivo general
Compensar errores de sistemas de medición por coordenadas con modelos físicos + redes neuronales (PINN). Tres frentes:

| Frente | Estado |
|---|---|
| **SMCP** (cámara–proyector, luz estructurada) | **Activo — foco principal** |
| LT–TLS (Laser Tracker + escáner láser terrestre) | Terminado y publicado (pausado) |
| Meas5 (MMC/CMM, dataset de Bala) | Pausado |

---

## SMCP — sistema
- Cámara FLIR Grasshopper3 (2448×2048, 3.45 µm), lente 12 mm; proyector DLP AAXA 4K1 (960×540 activo); base Newport.
- Calibración estéreo Moreno–Taubin; código Gray + phase-shifting.
- Software propio **SMCP.exe**: C++17 / Qt / OpenCV **2.4.13** / Spinnaker SDK, Visual Studio 2022, Release x64. Instalaciones en `D:\` (el .vcxproj original apuntaba a `C:\`; ya se corrigieron rutas y Qt VS Tools).
- **Anomalía conocida de calibración:** fy/fx del proyector ≈ 0.935 (−6.47 %), físicamente imposible (DMD de píxel cuadrado); causa: poca variación de poses. **Decisión: se conserva esta calibración** y el error sistemático grande se usa como caso para demostrar la compensación. Recalibrar (varias profundidades, tablero inclinado 20–45°, esquinas) queda diferido.

## Artefacto de referencia (3 esferas, valores MMC)
| Mensurando | Valor (mm) |
|---|---|
| D_A / D_B / D_C | 100.0071 / 100.0166 / 100.0033 |
| A–B / A–C / B–C | 217.4531 / 196.0364 / 308.1691 |

Remedición MMC de octubre 2026: diferencias 0–1.3 µm → despreciables, no requieren reentrenar.

## Pipeline Python (hecho)
Estructura del repo: `scripts/` (código), `resultados/{esferas,pinn_v2,gp}/` (salidas, rutas relativas al repo), `archivo/pinn_v1/` (histórico). Solo `CARPETA_XYZ` (escaneos .xyz, fuera del repo) se edita a mano. Ver `README.md`.

1. `procesa_30_escenas_smcp.py` — segmentación RANSAC, ajuste de esferas RANSAC + mínimos cuadrados con rechazo 2.5σ (desv. estándar, no MAD), etiquetado A/B/C, salida `resultados_30_escenas.xlsx`. Validado contra PolyWorks (±1 µm).
2. `pinn_smcp.py` **v2** — PINN:
   - `p_corr = p + p·s(z,ρ) + MLP(p)·ESCALA_MLP`
   - `s(z,ρ)`: 6 coeficientes físicos interpretables (escala a0, profundidad, radial).
   - MLP 3→64→64→64→3 (~8.8k parámetros) absorbe el residual.
   - Centros por (escaneo, esfera) optimizados como parámetros libres.
   - Pérdidas: L_R (esfericidad + diámetro), L_D (distancias entre centros), L_reg (mínima corrección del MLP — **no** penalizar el término físico; ese fue el bug de v1).
   - Validación **k-fold por escaneo (k=5)**, nunca por punto.
   - Guarda `.npz` + `.xlsx` (hojas Crudo, EscalaFija, Modelo, Coeficientes, Diagnostico); comparación en `comparacion_smcp.xlsx` hoja `Comparacion`.
3. Datos: 37 escaneos (29 originales + 8 multi-profundidad). Los 8 nuevos bajaron la correlación Y–Z de 0.997 a 0.266 → modelado 3D válido.

## Resultados SMCP (v2, 37 escaneos, k-fold por escaneo)
| Corrección | RMS global | Mejora vs escala fija |
|---|---|---|
| Crudo | ~1708 µm | — |
| **Escala fija (baseline honesto)** | **378 µm** | — |
| Solo física s(z,ρ) | 287 µm | 24 % |
| **PINN** | **97.9 µm** | **74.1 %** |

Mejoran los 6 mensurandos y los 37 escaneos.

## Reglas de validación y reporte (no negociables con Octavio)
- Reportar mejora **contra escala fija**, nunca contra la nube cruda (el “88 %” es engañoso).
- Split por escaneo/posición, nunca por punto (fuga de datos).
- Diámetros y distancias se comportan distinto: diámetros robustos; distancias sensibles a errores globales de calibración (sesgo anisótropo = calibración, no ruido).
- El 66 % es de **LT–TLS**, no de SMCP.
- Validar contra referencia (PolyWorks, certificado MMC) antes de avanzar de etapa.

---

## GPR (oct 2026) — reemplaza al PINN como modelo de compensación
- `scripts/gp_smcp.py`: física s(z,ρ) + GP (lineal + Matérn 5/2 ARD) observado vía diámetros/distancias. Mismos 37 escaneos, folds (SEMILLA=0) y escala fija que PINN v2.
- **55.4 µm RMS, 85.3 % vs escala fija** (PINN v2: 97.9 µm, 74.1 %). Gana en 31/37 escaneos; ventaja en distancias (3–4×). D_B ≈ 106 µm en ambos (cobertura de datos, no del modelo). 5 de 6 escaneos donde gana el PINN son de la 2.ª sesión (11:4x–11:5x).
- σ predicha ~1.4× corta (cobertura 55 % / 87 %).
- PINN v2 de referencia: `scripts/pinn_smcp.py` + `resultados/pinn_v2/`. Lo de `archivo/pinn_v1/` es la v1 (~222 µm), no usarlo.
- Escaneos de octubre: **descartados**, se trabaja solo con los 37 de septiembre.

## Integración en SMCP.exe (hecha, oct 2026)
- Proyecto: `C:\Users\oicasio\Desktop\CMCP\SMCP-CPP` (copia en OneDrive sin cambios). Calibración de los 37 escaneos: `CMCP\YA2\calibration.yml`.
- `exporta_modelo_yml.py`: `modelo_gp_smcp.npz` → `modelo_gp_smcp.yml` (cv::FileStorage, incluye copia de la calibración).
- `src_v2/GPCompensation.h/.cpp`: carga el `.yml`, verifica la calibración, corrige con `cv::parallel_for_` (sin dependencias nuevas). Llamado en `Application::reconstruct_model(_dump)` justo después de la triangulación.
- Menú Calibration: *Load GP compensation model…*, *Apply GP compensation on reconstruction*, *Compensate existing .xyz file…*. Config: `compensation/gp_file`, `compensation/gp_enabled`.
- Validado: C++ vs Python diferencia máx 5e-6 µm (`valida_cpp_vs_python.py`); ~1 s por 108k puntos.
- Puntos fuera del volumen de entrenamiento (z 687–1021 mm) se corrigen igual (extrapolación) y se cuentan en el mensaje.

---

## Otros frentes (referencia)
- **LT–TLS:** PINN PyTorch, 182.9 → 69.4 µm RMS (62 %) en blancos independientes. Publicado: Icasio-Hernández et al. 2026, *Meas. Sci. Technol.*, DOI 10.1088/1361-6501/ae957c. Scripts: `PINN_TLS.py`, `Comparacion_final.m`, `aplicar_compensacion.m`.
- **Meas5:** `Meas5.mat` → `P` (175×6; col 4 = diámetro medido ~19 mm, cols 5–6 probablemente error de forma y nº de puntos, por confirmar), `d`/`D` (15 225 distancias medidas/nominales), `C.txt` = coordenadas nominales. RF ~47 %; PINN 94.5 % en calibración vs ~55 % en puntos retenidos (techo práctico ~57 %). Errores dominantes: +928 ppm escala X, ~335 ppm perpendicularidad Y–Z. Script: `pinn_meas5.py`.
- **Transductor Philips L9-5:** Maritza reconstruyó la pieza en SolidWorks a partir de escaneos SMCP. El registro multivista (`rototraslacion_multiview_smcp.py`) no lo hizo ella; tiene TODOs (carga/segmentación de 14 nubes).

## Entorno
- Windows; venv Python en `C:\venvs\pinn` (3.14; numpy, scipy; **falta pandas/openpyxl**). `C:\venvs\pann` tiene torch. Usar raw strings en rutas con espacios.
- Visual Studio 18 Community, Qt 5.14.2 (`C:\Qt`), OpenCV 2.4.13 en `CMCP\opencv\build`.
- MATLAB, PolyWorks, SolidWorks (en español).
- Resultados en `.npz` + `.mat` para compatibilidad con MATLAB.
- Reportes mensuales: `MMQ - Reporte - [Mes] OIH.docx`, formato del reporte de julio; placeholders de imagen en cursiva/gris.

## Cómo trabajar con Maritza
- Itera en tiempo real; pega errores/capturas con poco contexto — diagnosticar desde eso.
- Cuando se acumulan cambios, prefiere el **código completo** a parches sueltos.
- Ser explícito en ediciones: “agrega esta línea” vs “cambia este valor”.
- Corrige errores directamente; no inventar lo que ella hizo (verificar autoría antes de redactar reportes).
