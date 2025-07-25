import os
import time
from dotenv import load_dotenv
import json
import google.generativeai as genai
import tempfile 
import subprocess

# --- Configuration & API Key Loading ---
try:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("FATAL ERROR: 'GOOGLE_API_KEY' environment variable not set.")
    print("Please set it in a .env file before running the script.")
    exit()

# Initialize the Gemini Pro model
# This model will act as both Stage 1 and Stage 2 LLMs in this orchestration script
model = genai.GenerativeModel('gemini-2.5-pro')

# --- PROMPTS FOR THE LLM STAGES ---

# LLM Stage 1 Prompt: For generating the Python ODE model function
LLM_PROMPT_GENERATE_ODE_FUNCTION = """
You are an expert Python programmer specializing in epidemiological Ordinary Differential Equation (ODE) modeling.
Your task is to convert raw differential equations into a precise and runnable Python function suitable for numerical integration using `scipy.integrate.odeint`.

The equations will be provided as plain text, in the format:
"d[Compartment Name]/dt = [mathematical expression for rate of change]"

You must generate a single Python function named `model_ode` with the signature `def model_ode(y, t, compartments_map):`.

Here are the strict rules for the `model_ode` function:
1.  **Imports:** Include `import numpy as np` at the top of the script.
2.  **Function Signature:** `def model_ode(y, t, compartments_map):`
    * `y`: A NumPy array where `y[index]` holds the current population of the compartment.
    * `t`: The current time point (required by `odeint`, but may not be used directly if equations are time-invariant).
    * `compartments_map`: A dictionary mapping compartment names (strings) to their corresponding 0-based index in the `y` array.
3.  **Compartment Variable Unpacking:** At the beginning of `model_ode`, unpack each compartment's population from `y` using `compartments_map` for readability. Convert compartment names with spaces to valid Python variable names by replacing spaces with underscores (e.g., `Death due to AIDS` becomes `Death_due_to_AIDS`).
4.  **Equation Conversion (CRITICAL - BDMAS/PEMDAS):**
    * For each equation `d[CompartmentName]/dt = [expression]`, you must define the corresponding derivative as `d_[Python_Compartment_Name]_dt = [python_expression]`.
    * **Strictly adhere to the mathematical order of operations (BDMAS/PEMDAS):**
        * **B**rackets (Parentheses)
        * **D**ivision / **M**ultiplication (from left to right)
        * **A**ddition / **S**ubtraction (from left to right)
    * **Be meticulous with parentheses** to preserve the intended grouping of terms in complex expressions. Assume all provided numerical values are constants.
    * Any terms like `X * Y` (e.g., `0.3333 * People living with AIDS`) should be directly translated to `X * Y_variable`.
    * Division (e.g., `A / B`) should be `A / B_variable`.
    * Terms like `λh` or `μ` that appear in the equations will be treated as pre-computed numerical rates, so they should be represented as their numerical value. (The input equations already have numerical rates; treat them as such.)
5.  **Return Value:** The function must return a `list` of derivatives (`dydt`) in the **exact order** that the compartment names would appear in a sorted list (e.g., alphabetical by compartment name, or in the order specified by the `compartments_map` you will assume is available to the external script). For simplicity, you can assume the `compartments_map` correctly reflects the order of the compartment names as they appear in the *input equations*.
6.  **Output Format:** Return ONLY the Python function definition, starting with `import numpy as np` and ending with the `return` statement. Do NOT include any extra comments outside the function, no markdown code blocks, no explanations, no `if __name__ == "__main__":` blocks. 

**Equations to convert (provided by user):**
"""

# LLM Stage 2 Prompt: For generating the full simulation runner script
LLM_PROMPT_GENERATE_SIMULATION_RUNNER = """
**CRITICAL:** Your output must be a plain Python script. DO NOT include markdown code block delimiters (```python or ```) at the beginning or end of your output.
You are an expert Python script generator for epidemiological simulations.
Your task is to create a complete, runnable Python script that takes initial population data, runs a simulation using a provided ODE function, and saves the results to a CSV file.

You will be given:
1.  The `model_ode` Python function (as a string).
2.  Initial population data (as a markdown table string).

Here are the strict rules for the simulation runner script:
1.  **Imports:** Include `import numpy as np`, `from scipy.integrate import odeint`, and `import pandas as pd` at the top.
2.  **Include `model_ode`:** The provided `model_ode` function string must be directly included in the script.
3.  **Compartment Order:** Define a list `compartment_names_order` that lists all compartment names exactly as they appear as variables in the `model_ode` function's body (e.g., `Death_due_to_AIDS`, `Recruitment_Homosexual_Men`). This list must define the order of the `y` array and `dydt` list within `model_ode`.
4.  **Compartments Map:** Create a dictionary `compartments_map = {name: i for i, name in enumerate(compartment_names_order)}`.
5.  **Parse Initial Population Data:**
    * The `initial_population_data_str` will be a markdown table.
    * Parse this table to create an `initial_populations_parsed` dictionary mapping compartment names to their float values.
    * Handle cases where a compartment name in the table might not be in `compartment_names_order` or vice-versa (e.g., use `.get(name, 0.0)` when creating `y0`).
6.  **Initial Conditions (y0):** Create a NumPy array `y0` of initial conditions based on `initial_populations_parsed` and `compartment_names_order`. Ensure `y0` is in the correct order.
7.  **Simulation Parameters:**
    * Define `num_years = 10` (for 10 years simulation).
    * Define `num_time_points = 1001`.
    * Define `t = np.linspace(0, num_years, num_time_points)`.
8.  **Run Simulation:** Use `solution = odeint(model_ode, y0, t, args=(compartments_map,))`. Include basic `try-except` for simulation errors.
9.  **Post-Simulation Data Processing & Sampling:**
    * After obtaining the `solution` from `odeint`, create a full DataFrame `df_full_solution` from the solution, with 'Time' and compartment columns.
    * Then, create a *new* DataFrame, `df_sampled_solution`, by selecting only the rows from `df_full_solution` where the 'Time' column is an **integer** (e.g., 0.0, 1.0, 2.0, ..., 10.0). This will ensure you get annual data points.
10. **Save Results:** Save `df_sampled_solution` (the *sampled* DataFrame) to a CSV file using `pandas.DataFrame.to_csv()`.
    * The CSV filename should be `simulation_results.csv`.
    * Ensure `index=False` for `to_csv`.
11. **Output Format:** Return ONLY the complete, runnable Python script. No extra comments, markdown formatting, or explanations outside the script itself.

**model_ode function to include:**
"""

# --- Orchestration Function ---
def generate_and_run_simulation_pipeline(equations_file_path: str, initial_pop_data_markdown: str, output_csv_name: str = "simulation_results.csv") -> str:
    """
    Orchestrates the LLM-driven generation and execution of epidemiological simulations.

    Args:
        equations_file_path (str): Path to the .txt file containing the differential equations.
        initial_pop_data_markdown (str): Markdown table string of initial population values.
        output_csv_name (str): Desired filename for the final CSV output.

    Returns:
        str: A message indicating success or failure.
    """
    separator = "\n" + "=" * 80 + "\n" # Use a different separator for orchestration logs

    # --- Step 1: Read Equations ---
    try:
        with open(equations_file_path, 'r', encoding='utf-8') as f:
            equations_text = f.read().strip()
        print(f"Equations loaded successfully from '{equations_file_path}'.")
    except FileNotFoundError:
        return f"Error: Equations file '{equations_file_path}' not found."
    except Exception as e:
        return f"Error reading equations file: {e}"

    # --- Step 2: LLM Stage 1 - Generate ODE function ---
    print("\n--- LLM Stage 1: Generating ODE Model Function (model_ode) ---")
    llm1_input_prompt = LLM_PROMPT_GENERATE_ODE_FUNCTION + "\n" + equations_text
    
    ode_function_code = ""
    try:
        response = model.generate_content(llm1_input_prompt)
        ode_function_code = response.text.strip()
        print("Stage 1 LLM response (model_ode function) received.")
        
        # Save generated ODE function to a temporary file for debugging/inspection
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as tmp_file:
            tmp_file.write(ode_function_code)
            ode_function_temp_path = tmp_file.name
        print(f"Generated model_ode function saved temporarily to: {ode_function_temp_path}")

    except Exception as e:
        print(f"Error during Stage 1 LLM generation: {e}")
        return f"Error: Could not generate model_ode function. {e}"

    # --- Step 3: LLM Stage 2 - Generate Simulation Runner Script ---
    print("\n--- LLM Stage 2: Generating Simulation Runner Script ---")
    llm2_input_prompt = (
        LLM_PROMPT_GENERATE_SIMULATION_RUNNER +
        "\n\nmodel_ode function to include:\n```python\n" +
        ode_function_code +
        "\n```\n\nInitial population data:\n```markdown\n" +
        initial_pop_data_markdown +
        "\n```"
    )

    simulation_runner_code = ""
    try:
        response = model.generate_content(llm2_input_prompt)
        simulation_runner_code = response.text.strip()
        print("Stage 2 LLM response (simulation runner script) received.")

        # Save generated simulation runner script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as tmp_file:
            tmp_file.write(simulation_runner_code)
            runner_script_temp_path = tmp_file.name
        print(f"Generated simulation runner script saved temporarily to: {runner_script_temp_path}")

    except Exception as e:
        print(f"Error during Stage 2 LLM generation: {e}")
        return f"Error: Could not generate simulation runner script. {e}"

    # --- Step 4: Execute the Generated Simulation Runner Script ---
    print(f"\n--- Executing Generated Simulation Script: {runner_script_temp_path} ---")
    try:
        # Run the generated script as a subprocess
        result = subprocess.run(
            ["python", runner_script_temp_path],
            capture_output=True, text=True, check=True
        )
        print("Simulation script output:\n", result.stdout)
        if result.stderr:
            print("Simulation script errors (if any):\n", result.stderr)

        # Assuming the generated script will save to 'simulation_results.csv'
        # We need to ensure the final output CSV is named as requested
        # Check if the generated script created 'simulation_results.csv'
        default_output_name_from_runner = "simulation_results.csv"
        if os.path.exists(default_output_name_from_runner):
            if default_output_name_from_runner != output_csv_name:
                os.rename(default_output_name_from_runner, output_csv_name)
                print(f"Renamed output CSV from '{default_output_name_from_runner}' to '{output_csv_name}'.")
        else:
            print(f"Warning: Generated simulation script did not create '{default_output_name_from_runner}'. Check script logic.")
            return f"Error: Simulation script ran but did not produce expected CSV output."


    except subprocess.CalledProcessError as e:
        print(f"Error executing generated simulation script. Return code: {e.returncode}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        return f"Error: Generated simulation script failed to execute. {e}"
    except FileNotFoundError:
        return f"Error: Python interpreter not found. Ensure Python is in your PATH."
    except Exception as e:
        print(f"An unexpected error occurred during script execution: {e}")
        return f"Error: Failed to execute generated simulation script. {e}"
    finally:
        # Clean up temporary files
        if os.path.exists(ode_function_temp_path):
            os.remove(ode_function_temp_path)
        if os.path.exists(runner_script_temp_path):
            os.remove(runner_script_temp_path)
        print("Temporary files cleaned up.")

    return f"Full simulation pipeline completed successfully. Results saved to '{output_csv_name}'."

# --- Example Usage (User will add their specific equation and population data here) ---
if __name__ == "__main__":
    # --- PLACEHOLDER FOR USER'S EQUATION DATA ---
    # You will replace this with content from your equation files (e.g., for HIV, COVID, Simple SIR)
    # Example: Load from a file, or define a multiline string
    # For now, using the HIV equations you provided earlier for demonstration
    example_equations_text = """
Death due to AIDS: dDeath due to AIDS/dt = + 0.3333 * People living with AIDS 
Recruitment (Homosexual Men): dRecruitment (Homosexual Men)/dt = - 12.7872 * Recruitment (Homosexual Men) 
People living with AIDS: dPeople living with AIDS/dt = + 0.03333 * Untreated Infected (Homosexual Men) + 0.03333 * Untreated Infected (Women) + 0.03333 * Untreated Infected (Heterosexual Men) + 0.018 * Treated with ART - 0.3333 * People living with AIDS - 0.0129 * People living with AIDS 
Recruitment (Heterosexual Men): dRecruitment (Heterosexual Men)/dt = - 147.0528 * Recruitment (Heterosexual Men) 
Untreated Infected (Women): dUntreated Infected (Women)/dt = + 1.637E-5 * Susceptible (Women) + 1.355E-5 * Susceptible (Women) - 0.0129 * Untreated Infected (Women) - 0.29997 * Untreated Infected (Women) - 0.03333 * Untreated Infected (Women) 
Susceptible (Heterosexual Men): dSusceptible (Heterosexual Men)/dt = + 147.0528 * Recruitment (Heterosexual Men) - 2.5E-6 * Susceptible (Heterosexual Men) - 1.1368E-4 * Susceptible (Heterosexual Men) - 0.0129 * Susceptible (Heterosexual Men) 
Untreated Infected (Heterosexual Men): dUntreated Infected (Heterosexual Men)/dt = + 2.5E-6 * Susceptible (Heterosexual Men) + 1.1368E-4 * Susceptible (Heterosexual Men) - 0.0129 * Untreated Infected (Heterosexual Men) - 0.29997 * Untreated Infected (Heterosexual Men) - 0.03333 * Untreated Infected (Heterosexual Men) 
Untreated Infected (Homosexual Men): dUntreated Infected (Homosexual Men)/dt = + 0.09636 * Susceptible (Homosexual Men) - 0.0129 * Untreated Infected (Homosexual Men) - 0.29997 * Untreated Infected (Homosexual Men) - 0.03333 * Untreated Infected (Homosexual Men) 
Treated with ART: dTreated with ART/dt = + 0.29997 * Untreated Infected (Homosexual Men) + 0.29997 * Untreated Infected (Women) + 0.29997 * Untreated Infected (Heterosexual Men) - 0.018 * Treated with ART - 0.0129 * Treated with ART 
Recruitment (Women): dRecruitment (Women)/dt = - 173.16 * Recruitment (Women) 
Susceptible (Homosexual Men): dSusceptible (Homosexual Men)/dt = + 12.7872 * Recruitment (Homosexual Men) - 0.09636 * Susceptible (Homosexual Men) - 0.0129 * Susceptible (Homosexual Men) 
Susceptible (Women): dSusceptible (Women)/dt = + 173.16 * Recruitment (Women) - 1.637E-5 * Susceptible (Women) - 1.355E-5 * Susceptible (Women) - 0.0129 * Susceptible (Women) 
Natural Death: dNatural Death/dt = + 0.0129 * Susceptible (Homosexual Men) + 0.0129 * Susceptible (Women) + 0.0129 * Susceptible (Heterosexual Men) + 0.0129 * Untreated Infected (Homosexual Men) + 0.0129 * Untreated Infected (Women) + 0.0129 * Untreated Infected (Heterosexual Men) + 0.0129 * Treated with ART + 0.0129 * People living with AIDS 
    """

    # --- PLACEHOLDER FOR USER'S INITIAL POPULATION DATA ---
    # You will replace this with content from your initial population data (e.g., from hivModel, covidModel, simpleModel data)
    # This should be a markdown table string as expected by LLM_PROMPT_GENERATE_SIMULATION_RUNNER
    example_initial_population_markdown = """
| Compartment Name                     | Initial Population |
|--------------------------------------|--------------------|
| Susceptible (Homosexual Men)         | 2446               |
| Susceptible (Women)                  | 189994             |
| Susceptible (Heterosexual Men)       | 171173             |
| Untreated Infected (Homosexual Men)  | 79                 |
| Untreated Infected (Women)           | 6                  |
| Untreated Infected (Heterosexual Men)| 29                 |
| Treated with ART                     | 107                |
| People living with AIDS              | 47                 |
| Death due to AIDS                    | 0                  |
| Recruitment (Homosexual Men)         | 0                  |
| Recruitment (Heterosexual Men)       | 0                  |
| Recruitment (Women)                  | 0                  |
| Natural Death                        | 0                  |
    """

    # --- Call the orchestration pipeline ---
    print("Initiating full LLM simulation generation and execution pipeline...")
    result_message = generate_and_run_simulation_pipeline(
        equations_file_path="hiv_equations.txt", # You would provide a path to a file with your equations
        initial_pop_data_markdown=example_initial_population_markdown,
        output_csv_name="final_hiv_simulation_output.csv"
    )
    print(result_message)

    # Note: For actual use, you'd load your equations from files like:
    # with open("path/to/your/hiv_equations.txt", "r") as f:
    #     hiv_eqs = f.read()
    # Then call:
    # generate_and_run_simulation_pipeline(hiv_eqs, example_initial_population_markdown, "hiv_sim_output.csv")