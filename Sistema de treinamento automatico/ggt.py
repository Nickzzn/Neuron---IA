import subprocess
import sys
import json
import time
import os
from gga import GGA         

class GGT:
    def __init__(self):
        self.gga = GGA()
        pasta_atual = os.path.dirname(os.path.abspath(__file__))
        self.arquivo_banco = os.path.join(pasta_atual, "banco treino", "banco_dados.json")
        self.historico_lotes = []
        self.banco_questoes = []
        self.carregar_banco()

    def carregar_banco(self):
        if os.path.exists(self.arquivo_banco):
            try:
                with open(self.arquivo_banco, "r", encoding="utf-8") as f:
                    self.banco_questoes = json.load(f)

            except Exception:
                print('[GGT] Erro ao ler "banco_dados.json" (Arquivo corrompido ou vazio).')
                self.banco_questoes = []

        if not self.banco_questoes:
            print('[GGT] Banco de dados não encontrado ou vazio. Injetando dados padrão de segurança.')

            self.banco_questoes = [{
                "pergunta": "Oi",
                "resposta_esperada": "Olá",
                "arquivo_contexto": "memoria/saudacao.txt"
            }]

    def iniciar_treinamento(self):
        print("   Gerenciamento geral de treinamento (GGT)")

        qtd_modulos = int(input("Quantos módulos paralelos deseja executar? "))
        total_ciclos = int(input("Digite a quantidade total de ciclos de treino: "))

        tempo_inicio_geral = time.time()
        caminho_modulo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modulo_treino.py")

        print(f"\n[GGT] Inicializando {qtd_modulos} instâncias paralelas...")
        processos_ativos = []
        
        for _ in range(qtd_modulos):
            try:
                proc = subprocess.Popen(
                    [sys.executable, caminho_modulo],
                    stdin = subprocess.PIPE,
                    stdout = subprocess.PIPE,
                    stderr = sys.stderr,
                    text = True,
                    encoding = "utf-8",
                    bufsize = 1
                )

                processos_ativos.append(proc)

            except Exception as e:
                print(f"[GGT Erro Crítico] Falha ao instanciar modulo_treino: {e}")
                sys.exit(1)

        print("[GGT] Instâncias prontas. Iniciando processamento...")

        ciclos_concluidos = 0
        lote_atual_dados = {
            "acertos": 0,
            "erros": 0,
            "gradientes": []
        }

        while ciclos_concluidos < total_ciclos:
            trabalhadores_neste_passo = []

            for proc in processos_ativos:
                if ciclos_concluidos < total_ciclos:
                    tarefa = self.banco_questoes[ciclos_concluidos % len(self.banco_questoes)]

                    try:
                        proc.stdin.write(json.dumps(tarefa) + "\n")
                        proc.stdin.flush()
                        trabalhadores_neste_passo.append(proc)
                        ciclos_concluidos += 1

                    except Exception as e:
                        print(f"[GGT] Erro ao enviar tarefa: {e}")

            for proc in trabalhadores_neste_passo:
                try:
                    linha_retorno = proc.stdout.readline()

                    if linha_retorno:
                        resultado = json.loads(linha_retorno.strip())

                        if resultado.get("status") == "sucesso":
                            lote_atual_dados["gradientes"].append(resultado["gradientes"])

                            if resultado["acertou"] == 1:
                                lote_atual_dados["acertos"] += 1

                            else:
                                lote_atual_dados["erros"] += 1

                        elif resultado.get("status") == "erro":
                            print(f"[GGT] Erro interno retornado pela IA: {resultado.get('erro')}")

                except Exception as e:
                    print(f"[GGT] Erro de comunicação/leitura: {e}")

            if ciclos_concluidos % 10 == 0 and lote_atual_dados["gradientes"]:
                print(f"[GGT] Processando fim de lote. Enviando gradientes para o GGA...")

                pesos_modificados = self.gga.calcular_media_e_aplicar(lote_atual_dados["gradientes"])

                if pesos_modificados:
                    print("[GGA] Confirmação de modificação")

                else:
                    print("[GGA] Pesos não modificados!")

                total_lote = lote_atual_dados["acertos"] + lote_atual_dados["erros"]
                taxa_acerto = (lote_atual_dados["acertos"] / total_lote) * 100 if total_lote > 0 else 0.0

                self.historico_lotes.append({
                    "lote": ciclos_concluidos // 10,
                    "taxa_acerto": taxa_acerto,
                    "taxa_erro": 100.0 - taxa_acerto
                })

                print(f" -> [GGT] Lote {len(self.historico_lotes)} Concluído | Precisão: {taxa_acerto:.1f}%")
                lote_atual_dados = {"acertos": 0, "erros": 0, "gradientes": []}

        if lote_atual_dados["gradientes"]:
            print(f"[GGT] Processando gradientes residuais finais...")
            self.gga.calcular_media_e_aplicar(lote_atual_dados["gradientes"])
            total_lote = lote_atual_dados["acertos"] + lote_atual_dados["erros"]
            taxa_acerto = (lote_atual_dados["acertos"] / total_lote) * 100 if total_lote > 0 else 0.0

            self.historico_lotes.append({
                "lote": (ciclos_concluidos + 9) // 10,
                "taxa_acerto": taxa_acerto,
                "taxa_erro": 100.0 - taxa_acerto
            })

            print(f" -> [GGT] Lote Final Concluído | Precisão: {taxa_acerto:.1f}%")

        print("\n[GGT] Finalizando trabalhadores paralelos...")
        for proc in processos_ativos:
            try:
                if proc.poll() is None:
                    proc.stdin.write("SAIR\n")
                    proc.stdin.flush()

            except OSError:
                pass

            finally:
                proc.kill()
                proc.wait()

        tempo_total = time.time() - tempo_inicio_geral
        self.gerar_relatorio_txt(tempo_total, ciclos_concluidos)

    def gerar_relatorio_txt(self, tempo_total, ciclos):
        if not self.historico_lotes:
            print("[GGT] Histórico de lotes vazio. Relatório não pôde ser gerado.")
            return

        primeiro_lote = self.historico_lotes[0]
        ultimo_lote = self.historico_lotes[-1]
        evolucao_liquida = ultimo_lote["taxa_acerto"] - primeiro_lote["taxa_acerto"]

        pasta_ggt = os.path.dirname(os.path.abspath(__file__))
        caminho_relatorio = os.path.join(pasta_ggt, "relatorio_treinamento.txt")

        with open(caminho_relatorio, "w", encoding="utf-8") as f:
            f.write("Relaótrio de historico do treinamneto (GGT)")
            f.write(f"Duração total do treino: {tempo_total:.2f} segundos\n")
            f.write(f"Total de ciclos efetuados: {ciclos}\n")
            f.write("Desempenho do lote: \n")

            for lote in self.historico_lotes:
                f.write(f"Lote {lote['lote']:02d} -> Taxa de acertos: {lote['taxa_acerto']:.2f}% | Erros: {lote['taxa_erro']:.2f}%\n")
            
            f.write("\nAnálise de progressão\n")
            f.write(f"Aproveitamento inicial (Lote 01) : {primeiro_lote['taxa_acerto']:.2f}%\n")
            f.write(f"Aproveitamento Final   (Lote {ultimo_lote['lote']:02d}) : {ultimo_lote['taxa_acerto']:.2f}%\n")
            f.write(f"Evolução Real da Rede Neuron     : {evolucao_liquida:+.2f}% de precisão global.")

        print(f"\n[GGT] Treinamento completo! Relatório 'relatorio_treinamento.txt' gerado em: {os.path.abspath('relatorio_treinamento.txt')}")

if __name__ == "__main__":
    gerenciador = GGT()
    gerenciador.iniciar_treinamento()
