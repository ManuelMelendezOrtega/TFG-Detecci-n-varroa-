import os
import glob

# ============================================================================
# Validación de Integridad del Dataset (Sanity Check)
# Objetivo: Comprobar que todas las imágenes generadas tienen su 
# correspondiente archivo de anotación XML (y viceversa) para evitar 
# errores de lectura durante el entrenamiento de la red neuronal.
# ============================================================================

DIR = "./augmented"

# 1. Extracción de nombres base mediante comprensión de conjuntos (Sets)
# Se extrae el nombre del archivo sin la extensión (ej. "foto1" en lugar de "foto1.jpg")
# Se usan Sets {} en lugar de Listas [] por su alta eficiencia matemática.
imgs = {
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(DIR, "*.jpg"))
    + glob.glob(os.path.join(DIR, "*.png"))
}

xmls = {
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(DIR, "*.xml"))
}

# 2. Operaciones de diferencia de conjuntos
# Detecta archivos huérfanos restando los conjuntos entre sí
solo_imgs = sorted(imgs - xmls)  # Imágenes que no tienen archivo de etiquetas
solo_xmls = sorted(xmls - imgs)  # Archivos de etiquetas que no tienen imagen asociada

# 3. Reporte de resultados por consola
print(f"Total archivos esperados: {len(imgs) * 2} (si el pareado es perfecto)")
print(f"Imágenes detectadas: {len(imgs)} | XML detectados: {len(xmls)}")
print(f"Archivos huérfanos (solo imagen): {len(solo_imgs)}")
print(f"Archivos huérfanos (solo XML): {len(solo_xmls)}")

# Muestra una pequeña muestra de los errores para facilitar su corrección manual
if solo_imgs[:10]:
    print("Ejemplos de error (solo imagen):", solo_imgs[:10])
if solo_xmls[:10]:
    print("Ejemplos de error (solo xml):", solo_xmls[:10])