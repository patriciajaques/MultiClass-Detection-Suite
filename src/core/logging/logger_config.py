import logging
import os
import warnings
from functools import wraps
from datetime import datetime
from pathlib import Path

class LoggerConfig:
    _loggers = {}  # Cache para armazenar loggers já criados

    @staticmethod
    def get_logger(logger_name):
        """
        Obtém ou cria um logger com o nome especificado.
        
        Args:
            logger_name (str): Nome do logger a ser obtido/criado
            
        Returns:
            logging.Logger: Logger configurado
        """
        # Se o logger já existe no cache, retorna ele
        if logger_name in LoggerConfig._loggers:
            return LoggerConfig._loggers[logger_name]
        
        # Configura o arquivo de log para o logger
        LoggerConfig.configure_log_file(
            file_main_name=logger_name,
            log_extension=".log",
            logger_name=logger_name
        )
        
        # Obtém o logger configurado
        logger = logging.getLogger(logger_name)
        
        # Armazena no cache
        LoggerConfig._loggers[logger_name] = logger
        
        return logger

    @staticmethod
    def configure_log_file(file_main_name='bayesian_optimization', log_extension=".log", logger_name=None):
        """
        Configura um arquivo de log. Pode configurar o logger raiz ou um logger nomeado.

        Args:
            file_main_name (str): Nome base para o arquivo de log.
            log_extension (str): Extensão do arquivo de log.
            logger_name (str, optional): Nome do logger a ser configurado. Se None, configura o logger raiz.
        """
        output_dir = LoggerConfig._get_output_directory()
        log_filename = LoggerConfig._generate_log_filename(file_main_name, log_extension)
        log_file_path = os.path.join(output_dir, log_filename)
        
        if logger_name:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            
            # Evita adicionar múltiplos handlers se já existirem
            if not logger.handlers:
                # Handler para arquivo com nível DEBUG
                fh = logging.FileHandler(log_file_path)
                fh.setLevel(logging.DEBUG)
                
                # Handler para console com nível INFO
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                
                # Formatação dos logs
                formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
                fh.setFormatter(formatter)
                ch.setFormatter(formatter)
                
                # Adiciona os handlers ao logger
                logger.addHandler(fh)
                logger.addHandler(ch)
        else:
            # Configura o logger raiz se ainda não estiver configurado
            root_logger = logging.getLogger()
            if not root_logger.handlers:
                logging.basicConfig(
                    filename=log_file_path,
                    level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s:%(levelname)s:%(message)s'
                )

    @staticmethod
    def _get_output_directory():
        """
        Obtém o diretório de saída para armazenar os arquivos de log.

        Returns:
            str: Caminho do diretório de saída.
        """
        # Obtém o diretório atual onde o arquivo Python está localizado
        current_dir = Path(__file__).resolve()
        
        # Encontra a raiz do projeto subindo até encontrar a pasta 'src'
        src_dir = current_dir
        while src_dir.name != 'behavior-detection' and src_dir.parent != src_dir:
            src_dir = src_dir.parent

        # Verifica se a pasta 'src' foi encontrada
        if src_dir.name != 'behavior-detection':
            raise FileNotFoundError("Diretório 'src' não encontrado na estrutura de diretórios.")
        
        # Define o diretório de saída dentro da pasta src
        output_dir = src_dir / 'output'
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def _generate_log_filename(file_main_name, log_extension):
        """
        Gera um nome de arquivo para o log baseado no timestamp atual.

        Args:
            file_main_name (str): Nome base para o arquivo de log.
            log_extension (str): Extensão do arquivo de log.

        Returns:
            str: Nome completo do arquivo de log.
        """
        return datetime.now().strftime(f'{file_main_name}_%Y%m%d_%H%M{log_extension}')

    @staticmethod
    def log_results(result):
        """
        Registra os resultados de uma iteração de otimização.

        Args:
            result: Resultado da iteração (deve ter atributos x_iters e func_vals).
        """
        if hasattr(result, 'x_iters') and hasattr(result, 'func_vals') and result.x_iters:
            score = abs(result.func_vals[-1])
            logging.info(f"Iteration {len(result.x_iters)}: tested parameters: {result.x_iters[-1]}, score: {score}")

def with_logging(logger_name: str):
    """
    Decorator que utiliza a LoggerConfig existente para configurar logging.
    
    Args:
        logger_name (str): Nome para identificar o logger
        
    Returns:
        function: Decorador que configura o logger na classe
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @wraps(cls.__init__)
        def new_init(self, *args, **kwargs):
            # Configura o logger usando o método get_logger
            self.logger = LoggerConfig.get_logger(logger_name)
            
            # Configura o tratamento de warnings para usar o logger
            def warning_to_logger(message, category, filename, lineno, file=None, line=None):
                msg = f"{category.__name__}: {str(message)}"
                self.logger.warning(msg)
            
            # Guarda o handler original de warnings
            original_showwarning = warnings.showwarning
            warnings.showwarning = warning_to_logger
            
            try:
                # Executa o __init__ original
                original_init(self, *args, **kwargs)
            finally:
                # Restaura o handler original de warnings
                warnings.showwarning = original_showwarning
        
        cls.__init__ = new_init
        return cls

    return decorator