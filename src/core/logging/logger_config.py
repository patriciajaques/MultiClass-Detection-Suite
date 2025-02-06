import logging
import os
import warnings
from functools import wraps
from datetime import datetime

from core.utils.path_manager import PathManager

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
        Configura um arquivo de log com handlers separados para logs gerais e erros.
        """
        output_dir = LoggerConfig._get_output_directory()

        # Arquivo para logs gerais
        log_filename = LoggerConfig._generate_log_filename(
            file_main_name, log_extension)
        log_file_path = os.path.join(output_dir, log_filename)

        # Arquivo específico para erros e warnings
        error_filename = LoggerConfig._generate_log_filename(
            f"{file_main_name}_errors", log_extension)
        error_file_path = os.path.join(output_dir, error_filename)

        if logger_name:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)

            # Limpa handlers existentes
            if logger.handlers:
                logger.handlers.clear()

            # Handler para arquivo geral (DEBUG e acima)
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(logging.DEBUG)

            # Handler específico para erros e warnings
            error_fh = logging.FileHandler(error_file_path)
            error_fh.setLevel(logging.WARNING)

            # Handler para console (INFO e acima)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Formatação dos logs
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s')
            fh.setFormatter(formatter)
            error_fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            # Adiciona os handlers ao logger
            logger.addHandler(fh)
            logger.addHandler(error_fh)
            logger.addHandler(ch)

            # Configura o capturador de warnings
            def warning_to_logger(message, category, filename, lineno, file=None, line=None):
                warning_message = f"{filename}:{lineno}: {category.__name__}: {message}"
                logger.warning(warning_message)

            # Substitui o handler padrão de warnings pelo nosso
            warnings.showwarning = warning_to_logger

            # Força todos os warnings a serem mostrados
            warnings.filterwarnings('always')

    @staticmethod
    def _get_output_directory():
        return PathManager.get_path('output')

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