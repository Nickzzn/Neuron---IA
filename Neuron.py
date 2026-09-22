import random
import math
import json
import os
import sys

#procura o arquivo de pesos na mesma pasta que o arquivo
pasta_programa = os.path.dirname(os.path.abspath(__file__))
arquivo_pesos = os.path.join(pasta_programa, "pesos.json")

#função estetica do terminal
def quebrar_texto():
    os.system('cls' if os.name == 'nt' else 'clear')

#taxa de aprendizagem
taxa_aprendizagem = 0.1

#dicionario de codificação letra -> binario
alfabeto = {
    #miusculas
    "A": "00000000",
    "Á": "00000001",
    "À": "00000010",
    "Â": "00000011",
    "Ã": "00000100",
    "B": "00000101",
    "C": "00000110",
    "D": "00000111",
    "E": "00001000",
    "É": "00001001",
    "Ê": "00001010",
    "F": "00001011",
    "G": "00001100",
    "H": "00001101",
    "I": "00001110",
    "Í": "00001111",
    "J": "00010000",
    "K": "00010001",
    "L": "00010010",
    "M": "00010011",
    "N": "00010100",
    "O": "00010101",
    "Ó": "00010110",
    "Ô": "00010111",
    "Õ": "00011000",
    "P": "00011001",
    "Q": "00011010",
    "R": "00011011",
    "S": "00011100",
    "T": "00011101",
    "U": "00011110",
    "Ú": "00011111",
    "V": "00100000",
    "W": "00100001",
    "X": "00100010",
    "Y": "00100011",
    "Z": "00100100",
    #minusculas
    "a": "00100101",
    "á": "00100110",
    "à": "00100111",
    "â": "00101000",
    "ã": "00101001",
    "b": "00101010",
    "c": "00101011",
    "d": "00101100",
    "e": "00101101",
    "é": "00101110",
    "ê": "00101111",
    "f": "00110000",
    "g": "00110001",
    "h": "00110010",
    "i": "00110011",
    "í": "00110100",
    "j": "00110101",
    "k": "00110110",
    "l": "00110111",
    "m": "00111000",
    "n": "00111001",
    "o": "00111010",
    "ó": "00111011",
    "õ": "00111100",
    "p": "00111101",
    "q": "00111110",
    "r": "00111111",
    "s": "01000000",
    "t": "01000001",
    "u": "01000010",
    "ú": "01000011",
    "v": "01000100",
    "w": "01000101",
    "x": "01000110",
    "y": "01000111",
    "z": "01001000",
    #numeros
    "0": "01001001",
    "1": "01001010",
    "2": "01001011",
    "3": "01001100",
    "4": "01001101",
    "5": "01001110",
    "6": "01001111",
    "7": "01010000",
    "8": "01010001",
    "9": "01010010",
    #caracteres especiais
    " ": "01010011",
    ",": "01010100",
    ".": "01010101",
    ":": "01010110",
    "!": "01010111",
    "?": "01011000"

}

#Classe do neuronio de saida
class Neuronio:
    def __init__(self, pesos, bias):
        self.pesos = pesos
        self.bias = bias

    #faz o somatorio total dos pesos multiplicados pelos valores binarios
    def somatorio(self, binario):
        self.ultima_entrada = binario
        result = []

        for i, j in zip(self.pesos, binario):
            multiplicacao = i * j
            result.append(multiplicacao)

        result.append(self.bias)
        return sum(result)

    def calcular(self, entrada):
        soma = self.somatorio(entrada)
        self.ultima_saida = 1 / (1 + math.exp(-soma))

        return self.ultima_saida

    def retropropar_e_atualizar(self, erro_gradiente):
        erros_entrada = [erro_gradiente * peso for peso in self.pesos]

        for i in range(len(self.pesos)):
            self.pesos[i] -= taxa_aprendizagem * erro_gradiente * self.ultima_entrada[i]

        self.bias -= taxa_aprendizagem * erro_gradiente

        return erros_entrada


#cria um neuronio para cada letra
class NeuronioLetra:
    def __init__(self, quantidade_neuronios):
        self.neuronios = []

        for i in range(quantidade_neuronios):
            pesos, bias = gerar_pesos_saida()
            self.neuronios.append(Neuronio(pesos, bias))

    def calcular(self, entrada):
        return [n.calcular(entrada) for n in self.neuronios]

    def retropropar_e_atualizar(self, erros_gradiente_saida):
        for n, erro in zip(self.neuronios, erros_gradiente_saida):
            n.retropropar_e_atualizar(erro)

    
class NeuronioMemoria:
    def __init__(self, pesos, bias):
        self.pesos = pesos
        self.bias = bias
        self.ultima_entrada = None
        self.ultima_saida = None

    def calcular(self, entrada):
        self.ultima_entrada = entrada
        soma = sum(p * v for p, v in zip(self.pesos, entrada)) + self.bias
        self.ultima_saida = 1 / (1 + math.exp(-soma))

        return self.ultima_saida

    def retropropar_e_atualizar(self, erro_camada_seguinte):
        derivada_sigmoid = self.ultima_saida * (1 - self.ultima_saida)
        erro_gradiente = erro_camada_seguinte * derivada_sigmoid
        erro_entrada = [erro_gradiente * peso for peso in self.pesos]

        for i in range(len(self.pesos)):
            self.pesos[i] -= taxa_aprendizagem * erro_gradiente * self.ultima_entrada[i]

        self.bias -= taxa_aprendizagem * erro_gradiente

        return erro_entrada

#salva pesos em um json
def salvar_pesos(neuronios, neuronios_letras, neuronios_memoria):
    dados = {
        "saida": [{"pesos": n.pesos, "bias": n.bias} for n in neuronios],
        "letras": {letra: [{"pesos": n.pesos, "bias": n.bias} for n in nl.neuronios] for letra, nl in neuronios_letras.items()},
        "memoria": [{"pesos": n.pesos, "bias": n.bias} for n in neuronios_memoria]
    }

    with open(arquivo_pesos, "w") as arquivos:
        json.dump(dados, arquivos, indent=4)

def carregar_pesos():
    with open(arquivo_pesos, "r") as arquivos:
        dados = json.load(arquivos)

    neuronios = [Neuronio(d["pesos"], d["bias"]) for d in dados["saida"]]

    neuronios_letras = {}

    for letra, letra_n in dados["letras"]. items():
        nl = NeuronioLetra(0)
        nl.neuronios = [Neuronio(d["pesos"], d["bias"]) for d in letra_n]
        neuronios_letras[letra] = nl

    neuronios_memoria = [NeuronioMemoria(d["pesos"], d["bias"]) for d in dados["memoria"]]
    return neuronios, neuronios_letras, neuronios_memoria


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

#transforma o binario de texto em uma lista de numeros
def refinar_binario(binario):
    binario_convertido = []

    for i in binario:
        numero_separado = int(i)
        binario_convertido.append(numero_separado)

    return binario_convertido


def gerar_pesos_saida():
    pesos = []

    for i in range(8):
        peso_gerado = random.uniform(-1, 1)
        pesos.append(peso_gerado)

    bias = random.uniform(-1, 1)

    return pesos, bias

#cria os pesos do neuronio de memoria
def gerar_pesos_memoria():
    pesos = []

    for i in range(16):
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

#transforma a memoria em binario para os neuroios de saida
def memoria_para_binario(memoria):
    valores_saida = gerar_saida(memoria)
    bits_saida = []

    for valor in valores_saida:
        if valor >= 0.5:
            bits_saida.append(1)

        else:
            bits_saida.append(0)

    return bits_saida


neuronios = []
bits = []


neuronios = []
neuronios_letras = {}
neuronios_memoria = []

if os.path.exists(arquivo_pesos):
    try:
        neuronios, neuronios_letras, neuronios_memoria = carregar_pesos()

    except (TypeError, KeyError, json.JSONDecodeError):
        neuronios = [Neuronio(*gerar_pesos_saida()) for _ in range(8)]

        for letra in alfabeto:
            neuronios_letras[letra] = NeuronioLetra(8)

        neuronios_memoria = [
            NeuronioMemoria(*gerar_pesos_memoria())
            for _ in range(8)
        ]

        salvar_pesos(
            neuronios,
            neuronios_letras,
            neuronios_memoria
        )

else:
    neuronios = [
        Neuronio(*gerar_pesos_saida())
        for _ in range(8)
    ]

    for letra in alfabeto:
        neuronios_letras[letra] = NeuronioLetra(8)

    neuronios_memoria = [
        NeuronioMemoria(*gerar_pesos_memoria())
        for _ in range(8)
    ]

    salvar_pesos(
        neuronios,
        neuronios_letras,
        neuronios_memoria
    )

#atualiza a memoria
def atualizar_memoria(memoria, valores_letra):
    entrada_memoria = memoria + valores_letra
    nova_memoria = []

    for neuronio in neuronios_memoria:
        valor = neuronio.calcular(entrada_memoria)
        nova_memoria.append(valor)

    return nova_memoria

def executar_backpropagation(texto, letra_correta):
    texto_filtrado = [letra for letra in texto if letra in alfabeto]

    if len(texto_filtrado) == 0:
        print("O texto não possui letras válidas para o alfabeto.")
        return

    memoria = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    historico_letras_ativadas = []
    historico_memorias = [memoria.copy()]

    for letra in texto:
        if letra not in alfabeto:
            continue
        
        bits_refinados = refinar_binario(converter_letra(letra))
        
        valores_letra = neuronios_letras[letra].calcular(bits_refinados)
        historico_letras_ativadas.append((letra, valores_letra))
        
        entrada_memoria = memoria + valores_letra
        memoria = [n.calcular(entrada_memoria) for n in neuronios_memoria]
        historico_memorias.append(memoria.copy())


    saidas_finais = [n.calcular(memoria) for n in neuronios]
    binario_correto = refinar_binario(converter_letra(letra_correta))

    erros_propagar_memoria = [0.0] * 8

    for idx, correto in enumerate(binario_correto):
        y = saidas_finais[idx]
        erro = y - correto
        derivada_sig = y * (1 - y)
        erro_gradiente = erro * derivada_sig
        
        erros_voltam = neuronios[idx].retropropar_e_atualizar(erro_gradiente)

        for i in range(8):
            erros_propagar_memoria[i] += erros_voltam[i]

    historico_letras_ativadas.reverse()
    historico_memorias.pop()
    historico_memorias.reverse()

    for idx, (letra, valores_letra) in enumerate(historico_letras_ativadas):
        proximos_erros_memoria = [0.0] * 8
        erros_para_neuronio_letra = [0.0] * 8
        
        for m_idx, n_mem in enumerate(neuronios_memoria):
            erros_voltam = n_mem.retropropar_e_atualizar(erros_propagar_memoria[m_idx])
            
            for i in range(8):
                proximos_erros_memoria[i] += erros_voltam[i]


            for i in range(8):
                erros_para_neuronio_letra[i] += erros_voltam[8 + i]
                
        neuronios_letras[letra].retropropar_e_atualizar(erros_para_neuronio_letra)
        
        erros_propagar_memoria = proximos_erros_memoria

def testar_rede(texto):
    memoria = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for letra in texto:
        if letra not in alfabeto: 
            continue
        
        bits_refinados = refinar_binario(converter_letra(letra))
        valores_letra = neuronios_letras[letra].calcular(bits_refinados)
        memoria = [n.calcular(memoria + valores_letra) for n in neuronios_memoria]

    bits_saida = [1 if n.calcular(memoria) >= 0.5 else 0 for n in neuronios]
    binario_saida = "".join(str(bit) for bit in bits_saida)
    letra_saida = converter_bin(alfabeto, binario_saida)

    print("\nRESULTADO")
    print(f"Binário gerado: {binario_saida}")
    print(f"Letra: {letra_saida}")
    return letra_saida


#modo adm
adm = False

resposta_menu = 3

while True:
    valores_calculados = []
    bits = []

    while True:
        valores_calculados = []
        bits = []

        if resposta_menu == 1:
            quebrar_texto()
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
                break
            else:
                palpite = testar_rede(texto)

                acerto = input("\nAcertei? (s/n) ").lower()

                if acerto == "n":
                    letra_certa = input("digite a resposta correta: ")

                    executar_backpropagation(texto, letra_certa)
                    input("\nPressione ENTER pra continuar")
                else:
                    print("Ótimo, vamos continuar")

                salvar_pesos(neuronios, neuronios_letras, neuronios_memoria)
                quebrar_texto()

        elif resposta_menu == 2:
            salvar_pesos(neuronios, neuronios_letras, neuronios_memoria)
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