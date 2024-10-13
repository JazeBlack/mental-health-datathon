# Define Beta
Beta = 0.95

def calculate_final_output(output1, output2):
    
    if output2 is None:
        print("Error: output2 is None, please ensure the questionnaire is completed.")
        return None

    # Compute the final result using the given formula
    x = Beta * output2 + (1 - Beta) * output1
    # Round x to the nearest integer
    final_output = round(x)

    return final_output
