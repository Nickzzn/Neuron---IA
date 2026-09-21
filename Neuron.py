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

adm = False

if treinamento_automatico:
    resposta_menu = 1

else:
    resposta_menu = 3

while True:
    valores_calculados = []
    bits = []

    while True:
        valores_calculados = []
        bits = []

        if resposta_menu == 1:

            if treinamento_automatico ==  False:
                letra_converter = input("Diga uma letra: ")

            else: 
                letra_converter = input()

    
            if adm:
                if letra_converter == "/menu":
                    break

            if letra_converter == "adm.mode off":
                adm = False
                resposta_menu = 3

            else:

                for i in letra_converter:
                    resultado_binario = converter_letra(i)
                    bits_refinados = refinar_binario(resultado_binario)

                if treinamento_automatico == False:
                    print(" ")

                for i in range(5):
                    valores = neuronios[i].somatorio(bits_refinados)
                    valores_calculados.append(valores)

                    if treinamento_automatico == False:
                        print(f"N{i+1}: {valores}")

                for i in range(5):
                    bit = neuronios[i].ativacao(valores_calculados[i])
                    bits.append(bit)

                binario_gerado = "".join(str(bit) for bit in bits)
                letra_prevista = converter_bin(alfabeto, binario_gerado)

                if treinamento_automatico == False:
                    print(f"\nBinario gerado: {binario_gerado}")
                    print(letra_prevista)

                else:
                    print(letra_prevista, flush = True)

                if treinamento_automatico == False:
                    acerto = input("Acertei a letra? (s/n) ").lower()

                else:
                    acerto = input()

                if acerto == "n":
                    if treinamento_automatico == False:
                        letra_correta = input("qual letra deveria ser? ").lower()

                    else: 
                        letra_correta = input()
                        
                    letra_c_binario = converter_letra(letra_correta)
                    aprender(bits_refinados, letra_c_binario)

                    if treinamento_automatico:
                        print("OK", flush = True)

                elif acerto == "s":
                    if treinamento_automatico:
                        print("OK", flush = True)

                    else:
                        print("Tudo certo, continuaremos.")    

                else:
                    print("valor invalido")
                    continue

                salvar_pesos(neuronios)
                quebrar_texto()


        elif resposta_menu == 2:
            salvar_pesos(neuronios)
            print("pesos salvos")
            exit()

        elif resposta_menu == 3:

            letra_converter = input("Diga uma letra: ")
            
            if letra_converter == "adm.mode on":
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
                valores_calculados = []
                bits = []           
                resultado_binario = converter_letra(letra_converter)
                bits_refinados = refinar_binario(resultado_binario)
    
                for i in range(5):
                    valores = neuronios[i].somatorio(bits_refinados)
                    valores_calculados.append(valores)
    
                for i in range(5):
                    bit = neuronios[i].ativacao(valores_calculados[i])
                    bits.append(bit)
    
                binario_gerado = "".join(str(bit) for bit in bits)
    
                print(f"Resposta Neuron: {converter_bin(alfabeto, binario_gerado)}")

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