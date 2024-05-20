import pandas as pd

def create_output_file (output_file, results):
    # Create a DataFrame from the results list
    df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)