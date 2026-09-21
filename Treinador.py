import subprocess
import sys
import time
import os

pasta_programa = os.path.dirname(os.path.abspath(__file__))
caminho_neuron = os.path.join(pasta_programa, "Neuron.py")

alfabeto = "abcdefghijklmnopqrstuvwxyz"

processo = subprocess.Popen(
    [sys.executable, caminho_neuron, "--treinamento"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

def enviar(mensagem):
    processo.stdin.write(mensagem + "\n")
    processo.stdin.flush()

def receber(): 
    return processo.stdout.readline().strip()

treino = 0
for rodada in range(100):
    if treino >= 25:
        treino = 0

    letra_enviar = alfabeto[treino] + alfabeto[treino+1]
    letra_esperada = alfabeto[treino+2]

    enviar(letra_enviar)
    resposta = receber()
    print(f"\nEscolhida: {letra_enviar}")
    print("Resposta: ", resposta)

    if resposta != letra_esperada:
        enviar("n")
        enviar(letra_esperada)

    else:
        enviar("s")
        treino += 1

    print("Neuron respondeu: ", receber())
    time.sleep(0.1)
enviar("encerrar")
