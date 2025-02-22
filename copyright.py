#!/usr/bin/env python3
import os
from datetime import datetime

# Texto do cabeçalho de copyright
COPYRIGHT_HEADER = '''"""
Copyright (c) {} Patricia Jaques
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License. To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
"""

'''


def add_copyright_header(file_path):
    """Adiciona cabeçalho de copyright se já não existir."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Verifica se já existe um cabeçalho de copyright
    if "Copyright" not in content:
        # Adiciona o cabeçalho com o ano atual
        new_content = COPYRIGHT_HEADER.format(datetime.now().year) + content

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(new_content)

        print(f"Adicionado copyright em: {file_path}")
    else:
        print(f"Copyright já existe em: {file_path}")


def process_directory(directory):
    """Processa recursivamente todos os arquivos .py no diretório."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                add_copyright_header(file_path)


if __name__ == "__main__":
    # Diretório do projeto (ajuste conforme necessário)
    project_dir = "/Users/patricia/Documents/code/python-code/behavior-detection/src"

    # Confirma com o usuário
    print(
        f"Isso adicionará cabeçalhos de copyright em todos os arquivos .py em: {project_dir}"
    )
    response = input("Deseja continuar? (s/n): ")

    if response.lower() == "s":
        process_directory(project_dir)
        print("Concluído!")
    else:
        print("Operação cancelada.")
