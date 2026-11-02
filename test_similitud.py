
from math import log, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
import unittest
from http.server import ThreadingHTTPServer

from app import FILENAMES, crear_handler, renderizar
from similitud import buscar, coseno, crear_indice, limpiar, map1, map2, norma, similitudes


class PruebasCalculo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indice = crear_indice(FILENAMES)

    def test_limpieza_del_ejercicio(self):
        self.assertEqual(limpiar("I want to start learning to change something in life."),
                         ["want", "start", "learning", "change", "something", "life"])
        self.assertEqual(limpiar("Learning something about me no one else knows."),
                         ["learning", "something", "no", "one", "else", "knows"])
        self.assertEqual(limpiar("Never stop learning."), ["never", "stop", "learning"])

    def test_map_de_las_fotos(self):
        self.assertEqual(map1("d1", ["life", "life"]), [(("d1", "life"), 1), (("d1", "life"), 1)])
        self.assertEqual(map2(("d1", "life"), 2), ("life", 1))

    def test_frecuencias_y_pesos(self):
        i = self.indice
        p = i["vocabulario"].index("learning")
        self.assertEqual(i["df_conteos"]["learning"], 3)
        self.assertEqual(i["df"]["something"], 2 / 3)
        self.assertEqual(i["tf"][2][p], 1 / 3)
        self.assertAlmostEqual(i["vectores"][0][p], log(2) / 6, places=14)
        self.assertAlmostEqual(i["vectores"][2][p], log(2) / 3, places=14)

    def test_matriz_con_formula_independiente(self):
        # Pesos sin el factor TF común, que se cancela en el coseno.
        a, b, c = log(4), log(2), log(2.5)
        n1, n3 = sqrt(4 * a*a + b*b + c*c), sqrt(2 * a*a + b*b)
        esperado12 = (b*b + c*c) / (n1*n1)
        esperado13 = b*b / (n1*n3)
        matriz = similitudes(FILENAMES)
        self.assertAlmostEqual(matriz[0][1], esperado12, places=14)
        self.assertAlmostEqual(matriz[0][2], esperado13, places=14)
        for x in range(3):
            self.assertAlmostEqual(matriz[x][x], 1, places=14)
            for y in range(3):
                self.assertAlmostEqual(matriz[x][y], matriz[y][x], places=14)
                self.assertGreaterEqual(matriz[x][y], 0)
                self.assertLessEqual(matriz[x][y], 1)

    def test_consulta_life_learning(self):
        resultado = buscar("Life Learning", self.indice)
        self.assertEqual([r["documento"] for r in resultado["resultados"]], [1, 3, 2])
        self.assertEqual([r["documento"] for r in resultado["recomendados"]], [1])
        self.assertEqual([f'{r["similitud"]:.4f}' for r in resultado["resultados"]],
                         ["0.5164", "0.1491", "0.1033"])
        p = self.indice["vocabulario"].index("life")
        self.assertAlmostEqual(resultado["vector"][p], log(4) / 2)

    def test_consulta_identica_tiene_similitud_uno(self):
        respuesta = buscar(self.indice["textos"][2], self.indice)
        self.assertEqual(respuesta["recomendados"][0]["documento"], 3)
        self.assertAlmostEqual(respuesta["resultados"][0]["similitud"], 1)

    def test_sin_coincidencias_no_inventa_recomendacion(self):
        for consulta in ["", "!!!", "i to about me", "galaxies pizza"]:
            with self.subTest(consulta=consulta):
                respuesta = buscar(consulta, self.indice)
                self.assertEqual(respuesta["recomendados"], [])
                self.assertTrue(all(r["similitud"] == 0 for r in respuesta["resultados"]))

    def test_palabras_desconocidas_no_alteran_df(self):
        anterior = dict(self.indice["df"])
        respuesta = buscar("life learning galaxy", self.indice)
        self.assertEqual(respuesta["desconocidas"], ["galaxy"])
        self.assertEqual(self.indice["df"], anterior)
        # El factor del TF de consulta se cancela en el coseno.
        base = buscar("life learning", self.indice)
        self.assertAlmostEqual(respuesta["resultados"][0]["similitud"], base["resultados"][0]["similitud"])

    def test_empate(self):
        respuesta = buscar("want no", self.indice)
        self.assertEqual([r["documento"] for r in respuesta["recomendados"]], [1, 2])

    def test_repeticiones_no_incrementan_df(self):
        with TemporaryDirectory() as carpeta:
            rutas = [Path(carpeta) / "a.txt", Path(carpeta) / "b.txt"]
            rutas[0].write_text("life life learning", encoding="utf-8")
            rutas[1].write_text("learning", encoding="utf-8")
            indice = crear_indice([str(ruta) for ruta in rutas])
            self.assertEqual(indice["df_conteos"]["life"], 1)
            self.assertEqual(indice["df_conteos"]["learning"], 2)
            self.assertEqual(indice["tf"][0][0], 2 / 3)
            self.assertEqual(len(indice["matriz"]), 2)

    def test_documentos_vacios(self):
        with TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "vacio.txt"
            ruta.write_text("i to in about me", encoding="utf-8")
            indice = crear_indice([str(ruta)])
            self.assertEqual(indice["matriz"], [[0.0]])
            self.assertEqual(buscar("life", indice)["recomendados"], [])

    def test_errores_de_entrada(self):
        with self.assertRaises(ValueError):
            crear_indice([])
        with self.assertRaises(ValueError):
            crear_indice([FILENAMES[0], FILENAMES[0]])
        with self.assertRaises(FileNotFoundError):
            crear_indice([str(Path(__file__).parent / "no_existe.txt")])
        with self.assertRaises(ValueError):
            coseno([1], [1, 2])
        self.assertEqual(norma([3, 4]), 5)

    def test_html_escapa_la_consulta(self):
        html = renderizar(self.indice, '<script>alert("x")</script>')
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)


class PruebasServidor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), crear_handler(crear_indice(FILENAMES)))
        cls.hilo = Thread(target=cls.server.serve_forever, daemon=True)
        cls.hilo.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.hilo.join()

    def test_formulario_y_matriz(self):
        with urlopen(self.url + "/") as respuesta:
            html = respuesta.read().decode("utf-8")
        self.assertIn('name="q"', html)
        self.assertIn('0.1466', html)
        self.assertIn('0.0770', html)

    def test_busqueda_desde_html(self):
        with urlopen(self.url + "/?" + urlencode({"q": "life learning"})) as respuesta:
            html = respuesta.read().decode("utf-8")
        self.assertIn('Documento 1 es la mejor coincidencia', html)
        self.assertIn('0.5164', html)
        self.assertIn('0.1491', html)

    def test_css_y_ruta_inexistente(self):
        with urlopen(self.url + "/style.css") as respuesta:
            self.assertIn("text/css", respuesta.headers["Content-Type"])
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url + "/similitud.py")
        self.assertEqual(error.exception.code, 404)

    def test_consulta_demasiado_larga(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url + "/?q=" + "x" * 1001)
        self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
