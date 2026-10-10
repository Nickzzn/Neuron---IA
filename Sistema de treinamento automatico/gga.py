import json
import os

class GGA:
    def __init__(self, arquivo_pesos="pesos.json"):
        pasta_atual = os.path.dirname(os.path.abspath(__file__))
        self.arquivo_pesos = os.path.join(pasta_atual, "pesos.json")

    def calcular_media_e_aplicar(self, lista_gradientes):
        qtd = len(lista_gradientes)

        if qtd == 0:
            return False

        if not os.path.exists(self.arquivo_pesos):
            return False

        with open(self.arquivo_pesos, "r", encoding="utf-8") as f:
            pesos_mestre = json.load(f)

        for idx in range(8):
            soma_pesos = [0.0] * 8
            soma_bias = 0.0

            for g in lista_gradientes:
                for i in range(8):
                    soma_pesos[i] += g["saida"][idx]["pesos"][i]

                soma_bias += g["saida"][idx]["bias"]

            for i in range(8):
                pesos_mestre["saida"][idx]["pesos"][i] -= (soma_pesos[i] / qtd)

            pesos_mestre["saida"][idx]["bias"] -= (soma_bias / qtd)

        for m_idx in range(8):
            soma_pesos = [0.0] * 16
            soma_bias = 0.0

            for g in lista_gradientes:
                for i in range(16):
                    soma_pesos[i] += g["memoria"][idx]["pesos"][i]

                soma_bias += g["memoria"][m_idx]["bias"]

            for i in range(16):
                pesos_mestre["memoria"][m_idx]["pesos"][i] -= (soma_pesos[i] / qtd)

            pesos_mestre["memoria"][m_idx]["bias"] -= (soma_bias / qtd)

        todas_letras = set()

        for g in lista_gradientes:
            if "letras" in g:
                todas_letras.update(g["letras"].keys())

        for letra in todas_letras:
            if letra not in pesos_mestre["letras"]:
                continue

            for idx in range(8):
                soma_pesos = [0.0] * 8
                soma_pesos = 0.0
                contagem_letra = 0

                for g in  lista_gradientes:
                    if "letra" in g and letra in g["letras"]:
                        contagem_letra += 1

                        for i in range(8):
                            soma_pesos[i] += g["letras"][letra][idx]["pesos"][i]

                        soma_bias += g["letras"][letra][idx]["bias"]

                if contagem_letra > 0:
                    for i in range(8):
                        pesos_mestre["letras"][letra][idx]["pesos"][i] -= (soma_pesos[i] /contagem_letra)

                    pesos_mestre["letras"][letra][idx]["bias"] -= (soma_bias / contagem_letra)

        todos_arquivos = set()

        for g in lista_gradientes:
            if "arquivo_memoria" in g:
                todos_arquivos.update(g["arquivo_memoria"].keys())

        for identificador in todos_arquivos:
            if identificador not in pesos_mestre["arquivo_memoria"]:
                continue

            soma_matriz = [[0.0] * 8 for _ in range(8)]
            contagem_arq = 0

            for g in lista_gradientes:
                if "arquivo_memoria" in g and identificador in g["arquivo_memoria"]:
                    contagem_arq += 1

                    for i in range(8):
                        for j in range(8):
                            soma_matriz[i][j] += g["arquivo_memoria"][identificador][i][j]

            if contagem_arq > 0:
                for i in range(8):
                    for j in range(8):
                        pesos_mestre["arquivo_memoria"][identificador][i][j] -= (soma_matriz[i][j] / contagem_arq)

        try:
            with open(self.arquivo_pesos, "w", encoding="utf-8") as f:
                json.dump(pesos_mestre, f, indent=4)

            return True

        except Exception:
            return False
        