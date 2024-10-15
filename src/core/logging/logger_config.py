import logging
import os
from datetime import datetime

class LoggerConfig:
    @staticmethod
    def configure_log_file(file_main_name='bayesian_optimization', log_extension=".log"):
        output_dir = LoggerConfig._get_output_directory()
        log_filename = LoggerConfig._generate_log_filename(file_main_name, log_extension)
        log_file_path = os.path.join(output_dir, log_filename)

        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            filemode='w',
            format='%(asctime)s:%(levelname)s:%(message)s'
        )

    @staticmethod
    def _get_output_directory():
        current_dir = os.path.dirname(__file__)
        output_dir = os.path.join(current_dir, '..', 'output')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def _generate_log_filename(file_main_name, log_extension):
        return datetime.now().strftime(f'{file_main_name}_%Y%m%d_%H%M{log_extension}')

    @staticmethod
    def log_results(result):
        if result.x_iters:
            score = abs(result.func_vals[-1])  # Use absolute value for simplicity
            logging.info(f"Iteration {len(result.x_iters)}: tested parameters: {result.x_iters[-1]}, score: {score}")

def main():
    LoggerConfig.configure_log_file('example_log')
    logging.info('Log configuration successful.')

if __name__ == "__main__":
    main()