


import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from urllib.parse import parse_qs, urlsplit

from similitud import STOPWORDS, buscar, crear_indice


BASE = Path(__file__).resolve().parent
FILENAMES = [str(BASE / "documentos" / f"documento{i}.txt") for i in range(1, 4)]


def resaltar(texto, palabras):
    """Escapa el texto antes de mostrarlo; destaca coincidencias completas."""
    import re
    partes = re.split(r"([^\W\d_]+)", texto, flags=re.UNICODE)
    return "".join(
        "<mark>" + escape(parte) + "</mark>" if parte.lower() in palabras else escape(parte)
        for parte in partes
    )


def mostrar_resultados(respuesta):
    if respuesta is None:
        return "<p>Escribe una consulta para buscar un documento.</p>"

    if not respuesta["tokens"]:
        return "<p>Escribe una consulta con palabras válidas.</p>"

    recomendados = respuesta["recomendados"]

    if not recomendados:
        return "<p>No se encontraron documentos relacionados con la consulta.</p>"

    if len(recomendados) == 1:
        numero = recomendados[0]["documento"]
        mensaje = f"Documento recomendado: Documento {numero}"
    else:
        numeros = ", ".join(
            str(documento["documento"])
            for documento in recomendados
        )
        mensaje = f"Hay un empate entre los documentos: {numeros}"

    html = f"<p><strong>{mensaje}</strong></p>"
    html += "<p>Similitud de cada documento con la consulta:</p>"
    html += "<ol>"

    for resultado in respuesta["resultados"]:
        numero = resultado["documento"]
        similitud = resultado["similitud"]

        html += (
            f"<li>Documento {numero}: "
            f"{similitud:.4f}</li>"
        )

    html += "</ol>"

    return html


def renderizar(indice, consulta=None):
    respuesta = buscar(consulta, indice) if consulta is not None else None
    palabras_consulta = set(respuesta["tokens"]) if respuesta else set()
    n = len(indice["textos"])
    documentos = []

    for i, texto in enumerate(indice["textos"]):
        documentos.append(f"""
            <div class="documento">
                <h3>Documento {i + 1}</h3>
                <p>{escape(texto.strip())}</p>
            </div>
        """)

    cabecera = ''.join(f'<th scope="col">D{i + 1}</th>' for i in range(n))
    matriz = ['<table class="matrix"><caption class="sr-only">Matriz de similitud entre documentos</caption><thead><tr><th scope="col">Documento</th>' + cabecera + '</tr></thead><tbody>']
    for i, fila in enumerate(indice["matriz"]):
        matriz.append(f'<tr><th scope="row">D{i + 1}</th>')
        for j, valor in enumerate(fila):
            clase = "diagonal" if i == j and valor > 0 else ""
            matriz.append(f'<td class="{clase}">{valor:.4f}</td>')
        matriz.append('</tr>')
    matriz.append('</tbody></table>')

    tabla = ['<table class="weights"><caption class="sr-only">Frecuencias y pesos por palabra</caption><thead><tr><th scope="col">Palabra</th>']
    tabla.extend(f'<th scope="col">TF D{i + 1}</th>' for i in range(n))
    tabla.append('<th scope="col">DF</th>')
    tabla.extend(f'<th scope="col">W D{i + 1}</th>' for i in range(n))
    tabla.append('</tr></thead><tbody>')
    for p, palabra in enumerate(indice["vocabulario"]):
        tabla.append(f'<tr><th scope="row">{escape(palabra)}</th>')
        for i, archivo in enumerate(indice["archivos"]):
            conteo = indice["conteos"].get((archivo, palabra), 0)
            tf_texto = f'{conteo}/{len(indice["tokens"][i])}' if conteo else '0'
            tabla.append(f'<td>{tf_texto}</td>')
        tabla.append(f'<td>{indice["df_conteos"][palabra]}/{n}</td>')
        tabla.extend(f'<td>{indice["vectores"][i][p]:.4f}</td>' for i in range(n))
        tabla.append('</tr>')
    tabla.append('</tbody></table>')

    aviso = ''
    if any(valor == 0 for valor in indice["normas"]):
        aviso = '<p class="notice-inline">Hay documentos sin palabras útiles. Su coseno no está definido; se muestra 0 por convención, también en su diagonal.</p>'

    plantilla = Template((BASE / "index.html").read_text(encoding="utf-8"))
    return plantilla.substitute(
        consulta=escape(consulta or "", quote=True),
        resultados=mostrar_resultados(respuesta),
        documentos="".join(documentos),
        matriz="".join(matriz),
        tabla_pesos="".join(tabla),
        aviso=aviso,
        total_documentos=n,
        total_palabras=len(indice["vocabulario"]),
        stopwords=escape(", ".join(sorted(STOPWORDS))),
    )


def crear_handler(indice):
    class Handler(BaseHTTPRequestHandler):
        def enviar(self, contenido, tipo="text/html; charset=utf-8", estado=200):
            datos = contenido.encode("utf-8") if isinstance(contenido, str) else contenido
            self.send_response(estado)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(datos)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(datos)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/style.css":
                self.enviar((BASE / "style.css").read_bytes(), "text/css; charset=utf-8")
            elif url.path == "/favicon.ico":
                self.enviar(b"", "image/x-icon", 204)
            elif url.path == "/":
                parametros = parse_qs(url.query, keep_blank_values=True)
                consulta = parametros.get("q", [None])[0]
                if consulta is not None and len(consulta) > 1000:
                    self.enviar("La consulta admite como máximo 1000 caracteres.", "text/plain; charset=utf-8", 400)
                    return
                self.enviar(renderizar(crear_indice(FILENAMES), consulta))
            else:
                self.enviar("Página no encontrada.", "text/plain; charset=utf-8", 404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Buscador local de documentos TF-IDF.")
    parser.add_argument("--port", type=int, default=8000, help="Puerto local (por defecto: 8000).")
    argumentos = parser.parse_args()
    if not 1 <= argumentos.port <= 65535:
        parser.exit(1, "El puerto debe estar entre 1 y 65535.\n")
    try:
        indice = crear_indice(FILENAMES)
        servidor = ThreadingHTTPServer(("127.0.0.1", argumentos.port), crear_handler(indice))
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"No se pudo iniciar: {error}\nRevisa los .txt o prueba otro puerto con --port 8001.\n")
    print(f"Abre http://127.0.0.1:{argumentos.port}", flush=True)
    print("Para detener: Ctrl+C. Si editas los .txt, reinicia el servidor.", flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
