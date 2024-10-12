# Importing necessary functions from abstract.py and mhd.py
from abstract import get_output1
from mhd import get_output2, run_questionnaire

# Define Beta
Beta = 0.95

def calculate_final_output():
    # Ensure the questionnaire is run and output2 is set
    run_questionnaire()  # Trigger the questionnaire in mhd.py
    
    # Get output1 and output2 from the imported functions
    output1 = get_output1()  # Output from abstract.py
    output2 = get_output2()  # Output from mhd.py
    
    if output2 is None:
        print("Error: output2 is None, please ensure the questionnaire is completed.")
        return None

    # Compute the final result using the given formula
    x = Beta * output2 + (1 - Beta) * output1
    print(f"Output: {output1}; Type: {type(output1)}")
    print(f"Output: {output2}; Type: {type(output2)}")
    # Round x to the nearest integer
    final_output = round(x)

    return final_output

if __name__ == "__main__":
    final_value = calculate_final_output()
    if final_value is not None:
        print(f"The final output is: {final_value}")
