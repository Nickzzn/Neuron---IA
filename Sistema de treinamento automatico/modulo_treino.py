import sys
import subprocess
import json
import os

def iniciar_trabalhador():
    caminho_ia = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Mani.py")
    
    processo_ia = subprocess.Popen(
        [sys.executable, caminho_ia],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, 
        text=True,
        bufsize=1
    )
    
    while True:
        linha_ggt = sys.stdin.readline()

        if not linha_ggt:
            break

        if linha_ggt.strip() == "SAIR":
            break
            
        processo_ia.stdin.write(linha_ggt.strip() + "\n")
        processo_ia.stdin.flush()
        
        if processo_ia.poll() is not None:
            break
            
        retorno_ia = processo_ia.stdout.readline()
        print(retorno_ia.strip(), flush=True)

    if processo_ia.poll() is None:
        processo_ia.terminate()
        processo_ia.wait()

if __name__ == "__main__":
    iniciar_trabalhador()
