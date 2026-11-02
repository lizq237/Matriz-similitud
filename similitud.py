

from collections import Counter, defaultdict
from math import isclose, log, sqrt
from pathlib import Path
import re



STOPWORDS = {"i", "to", "in", "about", "me"}


def limpiar(texto):
    """Minúsculas, separación en palabras y eliminación de stopwords."""
    palabras = re.findall(r"[^\W\d_]+", texto.lower(), flags=re.UNICODE)
    return [palabra for palabra in palabras if palabra not in STOPWORDS]


def map1(k, v):
    """k = nombre del archivo; v = lista de palabras limpias.

    Emite ((archivo, palabra), 1) por CADA aparición, como en la foto.
    """
    lista = []
    for palabra in v:
        lista.append(((k, palabra), 1))
    return lista


def agrupar(pares):
    """Group: reúne los valores que tienen la misma clave."""
    grupos = defaultdict(list)
    for clave, valor in pares:
        grupos[clave].append(valor)
    return grupos


def reduce1(k, v):
    """Devuelve ((archivo, palabra), conteo de apariciones)."""
    return k, sum(v)


def map2(k, v):
    """Recibe una clave (archivo, palabra) YA reducida y emite (palabra, 1).

    Cada palabra aporta una sola unidad por documento, aunque se repita.
    El valor v es el conteo anterior; aquí no se necesita.
    """
    return k[1], 1


def reduce2(k, v):
    """Devuelve (palabra, número de documentos que contienen esa palabra)."""
    return k, sum(v)


def norma(vector):
    """Longitud euclidiana: raíz de la suma de los cuadrados."""
    return sqrt(sum(valor ** 2 for valor in vector))


def coseno(vector_a, vector_b):
    """Producto punto / producto de normas; ambos vectores usan igual orden."""
    if len(vector_a) != len(vector_b):
        raise ValueError("Los vectores deben tener la misma cantidad de elementos.")
    denominador = norma(vector_a) * norma(vector_b)
    # El coseno de un vector nulo no está definido. La aplicación usa 0
    # como convención para documentos vacíos y consultas sin coincidencias.
    if denominador == 0:
        return 0.0
    producto_punto = sum(a * b for a, b in zip(vector_a, vector_b))
    return max(0.0, min(1.0, producto_punto / denominador))


def crear_indice(filenames):
    """Lee los archivos y calcula TF, DF, pesos, vectores y matriz N × N.

    filenames: lista de strings con rutas de archivos .txt en UTF-8.
    Devuelve todos los datos para reutilizarlos en el HTML y las consultas.
    """
    filenames = list(filenames)
    if not filenames:
        raise ValueError("Se necesita al menos un archivo.")
    archivos = [str(Path(nombre).resolve()) for nombre in filenames]
    if len(set(archivos)) != len(archivos):
        raise ValueError("No repitas la misma ruta de archivo en la lista.")
    textos = [Path(nombre).read_text(encoding="utf-8-sig") for nombre in archivos]
    tokens = [limpiar(texto) for texto in textos]
    n = len(archivos)

    # Primera etapa: MAP -> GROUP -> REDUCE para contar por documento.
    pares1 = []
    for archivo, palabras in zip(archivos, tokens):
        pares1.extend(map1(archivo, palabras))
    conteos = dict(reduce1(k, v) for k, v in agrupar(pares1).items())

    # Segunda etapa: una emisión por cada pareja (documento, palabra).
    pares2 = [map2(k, v) for k, v in conteos.items()]
    df_conteos = dict(reduce2(k, v) for k, v in agrupar(pares2).items())

    # Orden de primera aparición: igual que nuestra tabla hecha a mano.
    vocabulario = list(dict.fromkeys(palabra for doc in tokens for palabra in doc))
    df = {palabra: df_conteos[palabra] / n for palabra in vocabulario}
    # Fórmula del maestro: el +1 está DENTRO del logaritmo natural.
    idf = {palabra: log(1 / df[palabra] + 1) for palabra in vocabulario}
    tf = []
    vectores = []
    for archivo, palabras in zip(archivos, tokens):
        total = len(palabras)
        frecuencias = [
            conteos.get((archivo, palabra), 0) / total if total else 0.0
            for palabra in vocabulario
        ]
        tf.append(frecuencias)
        vectores.append([
            frecuencia * idf[palabra]
            for palabra, frecuencia in zip(vocabulario, frecuencias)
        ])

    matriz = [[coseno(a, b) for b in vectores] for a in vectores]
    return {
        "archivos": archivos,
        "textos": textos,
        "tokens": tokens,
        "vocabulario": vocabulario,
        "conteos": conteos,
        "df_conteos": df_conteos,
        "df": df,
        "idf": idf,
        "tf": tf,
        "vectores": vectores,
        "normas": [norma(vector) for vector in vectores],
        "matriz": matriz,
    }


def similitudes(filenames):
    """Interfaz pedida en el pizarrón: lista de archivos -> matriz N × N."""
    return crear_indice(filenames)["matriz"]


def buscar(consulta, indice):
    """Compara una consulta con los documentos, sin recalcular su DF.

    La consulta NO se añade como un cuarto documento. Las palabras que no
    existen en el vocabulario no tienen componente en el vector de búsqueda.
    """
    palabras = limpiar(consulta)
    conteos = Counter(palabras)
    total = len(palabras)
    vector = [
        (conteos[palabra] / total) * indice["idf"][palabra] if total else 0.0
        for palabra in indice["vocabulario"]
    ]
    resultados = []
    for posicion, vector_doc in enumerate(indice["vectores"]):
        compartidas = [
            palabra for palabra in dict.fromkeys(palabras)
            if palabra in indice["tokens"][posicion]
        ]
        resultados.append({
            "documento": posicion + 1,
            "archivo": Path(indice["archivos"][posicion]).name,
            "similitud": coseno(vector, vector_doc),
            "coincidencias": compartidas,
        })
    resultados.sort(key=lambda item: (-item["similitud"], item["documento"]))
    mejor = resultados[0]["similitud"] if resultados else 0.0
    recomendados = [
        item for item in resultados
        if mejor > 0 and isclose(item["similitud"], mejor, rel_tol=1e-9, abs_tol=1e-12)
    ]
    return {
        "consulta": consulta,
        "tokens": palabras,
        "vector": vector,
        "norma": norma(vector),
        "desconocidas": [p for p in dict.fromkeys(palabras) if p not in indice["idf"]],
        "resultados": resultados,
        "recomendados": recomendados,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genera una matriz de similitud TF-IDF.")
    parser.add_argument("archivos", nargs="*", help="Rutas de archivos UTF-8.")
    parser.add_argument("--query", help="Consulta opcional para recomendar documentos.")
    argumentos = parser.parse_args()
    carpeta = Path(__file__).resolve().parent / "documentos"
    rutas = argumentos.archivos or [str(carpeta / f"documento{i}.txt") for i in range(1, 4)]
    try:
        indice = crear_indice(rutas)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"No se pudo crear el índice: {error}\n")
    print("Matriz de similitudes:")
    for fila in indice["matriz"]:
        print("  ".join(f"{valor:.4f}" for valor in fila))
    if argumentos.query is not None:
        respuesta = buscar(argumentos.query, indice)
        print(f"\nConsulta: {argumentos.query}")
        for resultado in respuesta["resultados"]:
            print(f"D{resultado['documento']} ({resultado['archivo']}): {resultado['similitud']:.4f}")
        if not respuesta["recomendados"]:
            print("Sin coincidencias: no se recomienda ningún documento.")
        else:
            print("Recomendación: " + ", ".join(f"D{r['documento']}" for r in respuesta["recomendados"]))
