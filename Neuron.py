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
    "o": "00111011",
    "ó": "00111100",
    "õ": "00111101",
    "p": "00111110",
    "q": "00111111",
    "r": "01000000",
    "s": "01000001",
    "t": "01000010",
    "u": "01000011",
    "ú": "01000100",
    "v": "01000101",
    "w": "01000110",
    "x": "01000111",
    "y": "01001000",
    "z": "01001001",
    #numeros
    "0": "01001010",
    "1": "01001011",
    "2": "01001100",
    "3": "01001101",
    "4": "01001110",
    "5": "01001111",
    "6": "01010000",
    "7": "01010001",
    "8": "01010010",
    "9": "01010011",
    #caracteres especiais
    " ": "01010100",
    ",": "01010101",
    ".": "01010110",
    ":": "01010111",
    "!": "01011000",
    "?": "01011001",
    "%": "01011010",
    #comandos
    "/parar": "01011011"

}

class Let:
    def __init__(self):
        self.estado_memoria = None
        self.historico_memoria = []
        self.posicao = None
        self.memoria_inicial = None
        self.caracteres = []

class Letter:
    def __init__(self):
        self.lets = [Let() for _ in range(5)]


def gerar_let_letters(texto, neuronio_letra_dict, neuronio_memoria_lista):
    letras_validas = [letra for letra in texto if letra in caracteres]
    lista_letters = []

    if not letras_validas:
        return lista_letters

    letter_atual = Letter()
    indice_let = 0
    posicao_let = 1

    contador_caracteres = 0
    memoria = [0.0] * 8

    for letra in letras_validas:

        if contador_caracteres == 0:
            letter_atual.lets[indice_let].memoria_inicial = list(memoria)
        bits_refinados = [int(b) for b in caracteres[letra]]

        valores_letra = neuronio_letra_dict[letra].calcular(bits_refinados)

        memoria_anterior = list(memoria)

        entrada_memoria = memoria_anterior + valores_letra

        memoria = atualizar_memoria(memoria_anterior, valores_letra)

        letter_atual.lets[indice_let].historico_memoria.append({
            "memoria_anterior": memoria_anterior,
            "valores_letra": list(valores_letra),
            "entrada": list(entrada_memoria),
            "memoria_resultante": list(memoria)
        })

        letter_atual.lets[indice_let].caracteres.append(letra)

        contador_caracteres += 1

        if contador_caracteres == 5:
            letter_atual.lets[indice_let].estado_memoria = list(memoria)
            letter_atual.lets[indice_let].posicao = posicao_let

            indice_let += 1
            posicao_let += 1

            contador_caracteres = 0
            memoria = [0.0] * 8

            if indice_let == 5:

                lista_letters.append(letter_atual)

                letter_atual = Letter()
                indice_let = 0

    if contador_caracteres > 0:

        letter_atual.lets[indice_let].estado_memoria = list(memoria)
        letter_atual.lets[indice_let].posicao = posicao_let

        indice_let += 1

    if indice_let > 0:
        lista_letters.append(letter_atual)

    return lista_letters

        
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
        valores_ativados = []

        for n in self.neuronios:
            soma = n.somatorio(entrada)
            valores_ativados.append(1 / (1 + math.exp(-soma)))

        return valores_ativados

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


def gerar_saida(let_objeto):
    if let_objeto.estado_memoria is None:
        return [0.0] * 8
        
    memoria = let_objeto.estado_memoria
    valores_saida = []

    for neuronio in neuronios:
        soma = neuronio.somatorio(memoria)
        valor = 1 / (1 + math.exp(-soma)) 
        valores_saida.append(valor)

    return valores_saida

# Transforma a memória do Let em binário para os neurônios de saída
def memoria_para_binario(let_objeto):
    valores_saida = gerar_saida(let_objeto)
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
    neuronios = [Neuronio(*gerar_pesos_saida()) for _ in range(8)]

    for letra in caracteres:
        neuronios_letras[letra] = NeuronioLetra(8)

    neuronios_memoria = [NeuronioMemoria(*gerar_pesos_memoria()) for _ in range(8)]

    salvar_pesos(neuronios, neuronios_letras, neuronios_memoria)

#atualiza a memoria
def atualizar_memoria(memoria, valores_letra):
    entrada_memoria = memoria + valores_letra
    nova_memoria = [neuronio.calcular(entrada_memoria) for neuronio in neuronios_memoria]

    return nova_memoria

def executar_backpropagation(texto_entrada, texto_correto):
    memoria_global_pergunta = gerar_let_letters(texto_entrada, neuronios_letras, neuronios_memoria)
    memoria_global_correcao = gerar_let_letters(texto_correto, neuronios_letras, neuronios_memoria)

    for indice_letter, letter in enumerate(memoria_global_pergunta):
        if indice_letter >= len(memoria_global_correcao):
            break

        letter_correto = memoria_global_correcao[indice_letter]

        for indice_let, let in enumerate(letter.lets):
            if indice_let >= len(letter_correto.lets):
                break

            if let.estado_memoria is None:
                continue

            let_correto = letter_correto.lets[indice_let]
            if let_correto.estado_memoria is None:
                continue

            memoria = list(let.estado_memoria)
            memoria_correta = list(let_correto.estado_memoria)

            erro_memoria_atual = [memoria[i] - memoria_correta[i] for i in range(8)]
            historico = let.historico_memoria

            for indice_reverso, passo in enumerate(reversed(historico)):
                entrada_memoria = passo["entrada"]
                erros_entrada = [0.0] * 16

                for m_idx, n_mem in enumerate(neuronios_memoria):
                    soma = sum(peso * valor for peso, valor in zip(n_mem.pesos, entrada_memoria)) + n_mem.bias
                    saida = 1 / (1 + math.exp(-soma))
                    derivada_sigmoid = saida * (1 - saida)
                    
                    erro_gradiente = erro_memoria_atual[m_idx] * derivada_sigmoid

                    for i in range(16):
                        erros_entrada[i] += erro_gradiente * n_mem.pesos[i]

                    for i in range(16):
                        n_mem.pesos[i] -= taxa_aprendizagem * erro_gradiente * entrada_memoria[i]

                    n_mem.bias -= taxa_aprendizagem * erro_gradiente

                erro_memoria_atual = erros_entrada[:8]
                erros_valores_letra = erros_entrada[8:16]

                indice_caractere = len(historico) - 1 - indice_reverso
                letra_entrada = let.caracteres[indice_caractere]

                if letra_entrada not in neuronios_letras:
                    continue

                neuronios_letras[letra_entrada].retropropar_e_atualizar(erros_valores_letra)
    if memoria_global_correcao:
        ultimo_letter = memoria_global_correcao[-1]
        ultimo_let = None
        for let in reversed(ultimo_letter.lets):
            if let.estado_memoria is not None:
                ultimo_let = let
                break
        
        if ultimo_let is not None:
            memoria_final = ultimo_let.estado_memoria
            
            bits_parar = [int(b) for b in caracteres["/parar"]]
            
            for idx, neuronio_saida in enumerate(neuronios):
                soma = neuronio_saida.somatorio(memoria_final)
                saida_atual = 1 / (1 + math.exp(-soma))
                
                erro_bit = saida_atual - bits_parar[idx]
                derivada = saida_atual * (1 - saida_atual)
                gradiente_saida = erro_bit * derivada
                
                neuronio_saida.retropropar_e_atualizar(gradiente_saida)


def testar_rede(texto):
    memoria_global = gerar_let_letters(texto, neuronios_letras, neuronios_memoria)

    comando_parar = "/parar"
    limite_caracteres = 20
    resposta_gerada = []

    if not memoria_global:
        print("\nRESPOSTA")
        print("Neuron: None")
        return "Erro"

    letter_atual = memoria_global[-1]
    let_atual = None

    for let in reversed(letter_atual.lets):
        if let.estado_memoria is not None:
            let_atual = let
            break

    if let_atual is None:
        print("\nRESPOSTA")
        print("Neuron: None")
        return "Erro"

    memoria = list(let_atual.estado_memoria)

    while True:
        bits_saida = []

        for n in neuronios:
            soma = n.somatorio(memoria)
            ativacao = 1 / (1 + math.exp(-soma)) # Mantém o valor estritamente entre 0 e 1
            bits_saida.append(1 if ativacao >= 0.5 else 0)
            
        binario_saida = "".join(str(bit) for bit in bits_saida)
        letra_saida = converter_bin(caracteres, binario_saida)

        if letra_saida is None:
            print("\nRESPOSTA")
            print(f"Neuron: Erro de decodificação (Binário gerado inválido: {binario_saida})")
            return "Erro"

        if letra_saida == comando_parar:
            break

        resposta_gerada.append(letra_saida)
        texto_atual = "".join(resposta_gerada)

        if len(texto_atual) >= limite_caracteres:
            break

        bits_refinados = refinar_binario(binario_saida)
        valores_letra = neuronios_letras[letra_saida].calcular(bits_refinados)

        memoria = atualizar_memoria(memoria, valores_letra)

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
            texto = input("Escreva um texto: ")

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