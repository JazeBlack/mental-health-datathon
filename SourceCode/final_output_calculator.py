# Define the Beta constant
Beta = 0.95

def calculate_final_output(output1, output2):
    """
    Calculate the final prediction output by combining two intermediate outputs (output1 and output2)
    using a weighted formula based on the Beta value.

    Args:
        output1 (int): The first output, typically derived from the abstract analysis.
        output2 (int): The second output, typically derived from the questionnaire analysis.

    Returns:
        int: The final output after applying the weighted combination formula and rounding to the nearest integer.
             Returns None if output2 is not available.
    """
    if output2 is None:
        print("Error: output2 is None, please ensure the questionnaire is completed.")
        return None

    # Calculate the weighted combination of output1 and output2
    x = Beta * output2 + (1 - Beta) * output1

    # Round the result to the nearest integer
    final_output = round(x)

    return final_output
