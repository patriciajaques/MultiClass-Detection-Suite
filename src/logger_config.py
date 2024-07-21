import logging
import os
from datetime import datetime

class LoggerConfig:
    @staticmethod
    def configure_log_file(file_main_name='bayesian_optimization', log_term=".log"):
        # Obter o caminho do diretório atual do script
        current_dir = os.path.dirname(__file__)
        
        # Construir o caminho até a pasta `output` no mesmo nível do diretório do script
        output_dir = os.path.join(current_dir, '..', 'output')
        
        # Certificar-se de que a pasta `output` existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Gerar um nome de arquivo com data e hora
        log_filename = datetime.now().strftime(file_main_name+'_%Y%m%d_%H%M'+log_term)

        # Caminho completo para o arquivo de log
        log_file_path = os.path.join(output_dir, log_filename)

        # Configurar o logger para usar o arquivo de log especificado
        logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='w',
                            format='%(asctime)s:%(levelname)s:%(message)s')

    @staticmethod
    def log_results(result):
        """
        Registra os parâmetros testados e a pontuação para cada iteração.
        Inverte a pontuação se ela for negativa, apenas para exibição.
        """
        if len(result.x_iters) > 0:  # Verificar se há iterações para logar
            # Inverter o sinal da pontuação para exibição se ela for negativa
            score = -result.func_vals[-1] if result.func_vals[-1] < 0 else result.func_vals[-1]
            logging.info(f"Iteration {len(result.x_iters)}: tested parameters: {result.x_iters[-1]}, score: {score}")

def main():
    # Exemplo de uso
    LoggerConfig.configure_log_file('meu_arquivo.log')
    logging.info('Registro de log configurado com sucesso.')

if __name__ == "__main__":
    main()