import os

def merge_files(root_dir, output_file):
    """
    Percorre todas as subpastas procurando arquivos .py, Dockerfile e requirements.txt
    e os concatena em um único arquivo.
    
    Args:
        root_dir (str): Diretório raiz para iniciar a busca
        output_file (str): Nome do arquivo de saída
    """
    # Verifica se o arquivo já existe
    # if os.path.exists(output_file):
    #     response = input(f'O arquivo {output_file} já existe. Deseja sobrescrevê-lo? (s/n): ')
    #     if response.lower() != 's':
    #         print('Operação cancelada.')
    #         return

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py') or file == 'Dockerfile' or file == 'requirements.txt':
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, root_dir)
                    
                    outfile.write('\n' + '#' * 80 + '\n')
                    outfile.write(f'# Arquivo: {relative_path}\n')
                    outfile.write('#' * 80 + '\n\n')
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                            outfile.write('\n\n')
                    except Exception as e:
                        outfile.write(f'# Erro ao ler o arquivo: {str(e)}\n\n')

if __name__ == '__main__':
    # Substitua '.' pelo caminho da sua pasta, se necessário
    root_directory = '.'
    output_filename = 'todos_arquivos.txt'
    
    merge_files(root_directory, output_filename)
    print(f'Arquivos mesclados com sucesso em {output_filename}')