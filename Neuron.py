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
caracteres = {
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
    "ç": "00101100",
    "d": "00101101",
    "e": "00101110",
    "é": "00101111",
    "ê": "00110000",
    "f": "00110001",
    "g": "00110010",
    "h": "00110011",
    "i": "00110100",
    "í": "00110101",
    "j": "00110110",
    "k": "00110111",
    "l": "00111000",
    "m": "00111001",
    "n": "00111010",
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
    "z": "01000100",
    #numeros
    "0": "01000101",
    "1": "01000110",
    "2": "01000111",
    "3": "01001000",
    "4": "01001001",
    "5": "01001010",
    "6": "01001011",
    "7": "01001100",
    "8": "01001101",
    "9": "01001110",
    #caracteres especiais
    " ": "01001111",
    ",": "01001001",
    ".": "01001010",
    ":": "01001011",
    "!": "01001100",
    "?": "01001101",
    #comandos
    "/parar": "01001101"

}

class Let:
    def __init__(self):
        self.caracteres = []
        self.estado_memoria = None
        self.proximo = None

class Letter:
    def __init__(self):
        self.lets = [Let() for _ in range(5)]

        for i in range(4):
            self.lets[i].proximo = self.lets[i+1]

def gerar_let_letters(texto, neuronio_letra_dict, neuronio_memoria_lista):
    letras_validas = [letra for letra in texto if letra in caracteres]
    lista_letters = []

    if not letras_validas:
        return lista_letters, [0.0] * 8

    memoria = [0.0] * 8
    contador_atualizacoes = 0

    letter_atual = Letter()
    indice_let_atual = 0

    for letra in letras_validas:
        bits_refinados = [int(b) for b in caracteres[letra]]
        valores_letra = neuronio_letra_dict[letra].calcular(bits_refinados)
        entrada_memoria = memoria + valores_letra
        memoria = [n.calcular(entrada_memoria) for n in neuronio_memoria_lista]
        
        contador_atualizacoes += 1

        if contador_atualizacoes % 5 == 0:
            letter_atual.lets[indice_let_atual].estado_memoria = list(memoria)
            indice_let_atual += 1

            if indice_let_atual == 5:
                lista_letters.append(letter_atual)
                letter_atual = Letter()
                indice_let_atual = 0
    
    if contador_atualizacoes % 5 != 0:
        letter_atual.lets[indice_let_atual].estado_memoria = list(memoria)

        lista_letters.append(letter_atual)

    return lista_letters, memoria


        
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

    def gerar_saida(memoria):
        return [neuronio.calcular(memoria) for neuronio in neuronios]

    def memoria_para_binario(memoria):
        valores_saida = gerar_saida(memoria)
        bits_saida = []
        
        for valor in valores_saida:
            if valor >= 0.5:
                bits_saida.append(1)

            else:
                bits_saida.append(0)

        return bits_saida

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
def converter_caractere(caractere):
    return caracteres[caractere]


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

        for letra in caracteres:
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

    for letra in caracteres:
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

def executar_backpropagation(texto_entrada, texto_correto):
    letters_pergunta, memoria = gerar_let_letters(texto_entrada, neuronios_letras, neuronios_memoria)
    letters_correcao, _ = gerar_let_letters(texto_correto, neuronios_letras, neuronios_memoria)

    for letra_alvo in texto_correto:

        memoria_anterior = memoria.copy()

        saidas_finais = [n.calcular(memoria) for n in neuronios]
        binario_correto = refinar_binario(converter_caractere(letra_alvo))

        erros_propagar_memoria = [0.0] * 8

        for idx, correto in enumerate(binario_correto):
            y = saidas_finais[idx]
            erro = y - correto
            derivada_sig = y * (1 - y)
            erro_gradiente = erro * derivada_sig

            erros_voltam = neuronios[idx].retropropar_e_atualizar(erro_gradiente)
            for i in range(8):
                erros_propagar_memoria[i] += erros_voltam[i]

        vaslores_letra_alvo = neuronios_letras[letra_alvo].calcular(binario_correto)

        entrada_memoria_atual = memoria_anterior + vaslores_letra_alvo

        erros_para_neuronio_letra = [0.0] * 8
        for m_idx, n_mem in enumerate(neuronios_memoria):
            
            erros_voltam = n_mem.retropropar_e_atualizar(erros_propagar_memoria[m_idx])
            for i in range(8):
                erros_para_neuronio_letra[i] += erros_voltam[8 + i]

        neuronios_letras[letra_alvo].retropropar_e_atualizar(erros_para_neuronio_letra)

        memoria = [n.calcular(entrada_memoria_atual) for n in neuronios_memoria]
    

def testar_rede(texto):
    letters_teste, memoria = gerar_let_letters(texto, neuronios_letras, neuronios_memoria)

    comando_parar = "/parar"
    letra = ""
    limite_caracteres = 20
    resposta_gerada = []

    while True:
        bits_saida = [1 if n.calcular(memoria) >= 0.5 else 0 for n in neuronios]
        binario_saida = "".join(str(bit) for bit in bits_saida)
        letra_saida = converter_bin(caracteres, binario_saida)

        if letra_saida is not None:
            resposta_gerada.append(letra_saida)
            teste_limite = "".join(resposta_gerada)

            if len(teste_limite) >= limite_caracteres:
                break

            elif letra_saida == comando_parar:
                break

            else:
                bits_refinados = refinar_binario(binario_saida)
                valores_letra = neuronios_letras[letra_saida].calcular(bits_refinados)
                memoria = [n.calcular(memoria + valores_letra) for n in neuronios_memoria]
        else:
            print("\nRESPOSTA")
            print("Resposta: None")
            return "Erro"
        
    texto_final = "".join(resposta_gerada)


    print("\nRESULTADO")
    print(f"Resposta: {texto_final}")
    return texto_final


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
                    while True:
                        texto_certo = input("digite a resposta correta: ")

                        valido = any(letra in caracteres for letra in texto_certo)

                        if valido:
                            executar_backpropagation(texto, texto_certo)
                            print("Pesos balanceados")
                            break

                        else:
                            print("Caractere invalido")
                        
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