import random
import math
import json
import os
import sys

#Verifica se o treinamento automatico deve ou não ser ativado
treinamento_automatico = False

if "--treinamento" in sys.argv:
    treinamento_automatico = True

else:
    treinamento_automatico = False

#procura o arquivo de pesos na mesma pasta que o arquivo 
pasta_programa = os.path.dirname(os.path.abspath(__file__))
arquivo_pesos = os.path.join(pasta_programa, "pesos.json")

#função estetica do terminal
def quebrar_texto():
    os.system('cls' if os.name == 'nt' else 'clear')

#taxa de aprendizagem
taxa_aprendizagem = 0.01

memoria = [0, 0, 0, 0, 0]

#dicionario de codificação letra -> binario
alfabeto = {
    "a": "00000",
    "b": "00001",
    "c": "00010",
    "d": "00011",
    "e": "00100",
    "f": "00101",
    "g": "00110",
    "h": "00111",
    "i": "01000",
    "j": "01001",
    "k": "01010",
    "l": "01011",
    "m": "01100",
    "n": "01101",
    "o": "01110",
    "p": "01111",
    "q": "10000",
    "r": "10001",
    "s": "10010",
    "t": "10011",
    "u": "10100",
    "v": "10101",
    "w": "10110",
    "x": "10111",
    "y": "11000",
    "z": "11001",
}

#Cria a classe de neuronio
class Neuronio:
    def __init__(self, pesos, bias):
        self.pesos = pesos
        self.bias = bias

    #faz o somatorio total dos pesos multiplicados pelos valores binarios
    def somatorio(self, binario):
        result = []
        for i, j in zip(self.pesos, binario):
            multiplicacao = i * j
            result.append(multiplicacao)

        result.append(self.bias)

        return sum(result)

    #função de ativação do euronio, decidindo se retorna 1 ou 0 para o binario
    def ativacao(self, soma):
        sigmoide = 1 / (1 + math.exp(-soma))

        if sigmoide >= 0.5:
            return 1

        elif sigmoide < 0.5:
            return 0


class NeuronioLetra:
    def __init__(self, quantidade_neuronios):
        self.neuronios = []

        for i in range(quantidade_neuronios):
            pesos, bias = gerar_pesos()
            neuronio = Neuronio(pesos, bias)
            self.neuronios.append(neuronio)

    def calcular(self, entrada):
        valores = []

        for neuronio in self.neuronios:
            soma = neuronio.somatorio(entrada)

            valor = 1 / (1 + math.exp(-soma))

            valores.append(valor)

        return valores

class NeuronioMemoria:
    def __init__(self, pesos, bias):
        self.pesos = pesos
        self.bias = bias

    def calcular(self, entrada):
        soma = 0

        for peso, valor in zip(self.pesos, entrada):
            soma += peso * valor
        
        soma += self.bias

        return 1 / (1 + math.exp(-soma))
    
#salva pesos em um json
def salvar_pesos(neuronios):
    dados = []

    for neuronio in neuronios:
        dados.append({
            "pesos": neuronio.pesos,
            "bias": neuronio.bias
        })

    with open(arquivo_pesos, "w") as arquivos:
        json.dump(dados, arquivos, indent=4)

#abre o arquivo em json e cria objetos atribuindo a eles esses valores
def carregar_pesos():
    neuronios = []

    with open(arquivo_pesos, "r") as arquivos:
        dados = json.load(arquivos)

    for dados_neuronio in dados:
        pesos = dados_neuronio["pesos"]
        bias = dados_neuronio["bias"]

        neuronio = Neuronio(pesos, bias)
        neuronios.append(neuronio)

    return neuronios

#pega a letra e retorna o binario do alfabeto
def converter_letra(letra):
    return alfabeto[letra]

#faz o caminho inverso, pegando o binario e retornando uma letra
def converter_bin(dicio, binario):
    valor_encontrado = None

    for chave, valor in dicio.items():
        if valor == binario:
            valor_encontrado = chave
            break

    return valor_encontrado


def refinar_binario(binario):
    binario_convertido = []
    for i in binario:
        numero_separado = int(i)
        binario_convertido.append(numero_separado)

    return binario_convertido


def gerar_pesos():
    pesos = []
    for i in range(5):
        peso_gerado = random.uniform(-1, 1)
        pesos.append(peso_gerado)

    bias = random.uniform(-1, 1)

    return pesos, bias


def gerar_pesos_memoria():
    pesos = []

    for i in range(10):
        peso_gerado = random.uniform(-1, 1)
        pesos.append(peso_gerado)

    bias = random.uniform(-1, 1)

    return pesos, bias

def gerar_saida(memoria):
    valores_saida = []

    for neuronio in neuronios:
        soma = neuronio.somatorio(memoria)
        valor = 1 / (1 + math.exp(-soma))

        valores_saida.append(valor)

    return valores_saida

def memoria_para_binario(memoria):
    valores_saida = gerar_saida(memoria)

    bits_saida = []

    for valor in valores_saida:

        if valor >= 0.5:
            bits_saida.append(1)

        else:
            bits_saida.append(0)

    return bits_saida


def calcular_gradiente(neuronio, entrada, correto):
    soma = neuronio.somatorio(entrada)

    y = 1 / (1 + math.exp(-soma))

    erro = y - correto

    derivada_sigmoid = y * (1 - y)

    fator = erro * derivada_sigmoid

    gradientes = []

    for bit in entrada:
        gradiente = fator * bit
        gradientes.append(gradiente)

    gradiente_bias = fator

    return gradientes, gradiente_bias


def aprender(entrada, binario_correto):
    for posicao, correto in enumerate(binario_correto):

        neuronio = neuronios[posicao]

        gradientes, gradiente_bias = calcular_gradiente(
            neuronio,
            entrada,
            int(correto)
        )

        for indice in range(len(neuronio.pesos)):
            neuronio.pesos[indice] -= taxa_aprendizagem * gradientes[indice]

        neuronio.bias -= taxa_aprendizagem * gradiente_bias

neuronios = []
bits = []

if os.path.exists(arquivo_pesos):
    neuronios = carregar_pesos()

else:
    neuronios = []
    for i in range(5):
        pesos, bias = gerar_pesos()
        neuronio = Neuronio(pesos, bias)

        neuronios.append(neuronio)

    salvar_pesos(neuronios)

#neuronios feitos para cada letra
neuronio_letras = {}

for letra in alfabeto:
    neuronio_letras[letra] = NeuronioLetra(5)

neuronios_memoria = []

for i in range(5):
    pesos, bias = gerar_pesos_memoria()

    neuronio = NeuronioMemoria(pesos, bias)

    neuronios_memoria.append(neuronio)

def atualizar_memoria(memoria, valores_letra):
    entrada_memoria = memoria + valores_letra

    nova_memoria = []

    for neuronio in neuronios_memoria:
        valor = neuronio.calcular(entrada_memoria)
        nova_memoria.append(valor)

    return nova_memoria

def testar_rede(texto):

    memoria = [0, 0, 0, 0, 0]

    for letra in texto:

        if letra not in alfabeto:
            continue

        resultado_binario = converter_letra(letra)
        bits_refinados = refinar_binario(resultado_binario)

        valores_letra = neuronio_letras[letra].calcular(bits_refinados)

        memoria = atualizar_memoria(memoria, valores_letra)

        print(f"\nLetra atual: {letra}")
        print(f"Representação: {valores_letra}")
        print(f"Memória: {memoria}")

    bits_saida = memoria_para_binario(memoria)

    binario_saida = "".join(str(bit) for bit in bits_saida)

    letra_saida = converter_bin(alfabeto, binario_saida)

    print("\nRESULTADO FINAL ")
    print(f"Memória: {memoria}")
    print(f"Binário: {binario_saida}")
    print(f"Saída: {letra_saida}")

#modo adm
adm = False

if treinamento_automatico:
    resposta_menu = 1

else:
    resposta_menu = 1

while True:
    valores_calculados = []
    bits = []

    while True:
        valores_calculados = []
        bits = []

        if resposta_menu == 1:
            texto = input("Escreva um texto: ").lower()

            if texto == "adm.mode on":
                adm = True

                while adm:
                    quebrar_texto()
                    modo_adm = input("> ")

                    if modo_adm == "adm.mode off":
                        adm = False

                    elif modo_adm == "/menu":
                        break

            else:

                testar_rede(texto)
                salvar_pesos(neuronios)

        elif resposta_menu == 2:
            salvar_pesos(neuronios)
            print("pesos salvos")
            exit()


        elif resposta_menu == 3:

            texto = input("escreva um texto: ")
            
            if texto == "adm.mode on":
                adm = True

                while adm:
                    quebrar_texto()
                    modo_adm = input("> ")

                    if modo_adm == "adm.mode off":
                        adm = False

                    elif modo_adm == "/menu":
                        break

                else:
                    print("opção invaldia")

                break

            else:
                print("")
                #mesma coisa do 1 mas com interface voltada ao usuario
        else:
            exit()

    quebrar_texto()
    print("Sistema Neuron")
    print("\n Deseja:")
    print("1 - Iniciar treinamento")
    print("2 - Salvar e terminar treinamento")
    print("3 - Testar")
    print("4 - Enserrar")
    resposta_menu = int(input("Escolha um item do menu: "))
    quebrar_texto()