# 🔢 Root-Finding Methods Comparison in Python

This repository contains a **Python program** that implements and compares **seven classical numerical methods** for finding roots of nonlinear equations. The purpose of this project is to analyze and visualize the convergence behavior, error reduction, and execution performance of each method.

---

## 📁 Project Structure

```
numerical-method/
│
├── .gitignore
├── LICENSE
├── numeric.py           # Main numerical methods implementation
├── README.md            # This documentation
├── testing.ipynb        # Jupyter notebook for testing
│
└── data/                # Data and output directories
    ├── graph/           # Visualization outputs
    │   ├── exponential/
    │   │   ├── convergence_error_EXPONENTIAL.png
    │   │   ├── convergence_path_EXPONENTIAL.png
    │   │   └── execution_time_EXPONENTIAL.png
    │   ├── mixed/
    │   │   ├── convergence_error_MIXED.png
    │   │   ├── convergence_path_MIXED.png
    │   │   └── execution_time_MIXED.png
    │   ├── polynomial/
    │   │   ├── convergence_error_POLYNOMIAL.png
    │   │   ├── convergence_path_POLYNOMIAL.png
    │   │   └── execution_time_POLYNOMIAL.png
    │   └── trigonometric/
    │       ├── convergence_error_TRIGONOMETRIC.png
    │       ├── convergence_path_TRIGONOMETRIC.png
    │       └── execution_time_TRIGONOMETRIC.png
    │
    └── table/           # Tabular results (CSV format)
        ├── exponential/
        │   ├── comparison_summary_EXPONENTIAL.csv
        │   ├── iterations_Aitken_EXPONENTIAL.csv
        │   ├── iterations_Bisection_EXPONENTIAL.csv
        │   ├── iterations_Fixed_Point_EXPONENTIAL.csv
        │   ├── iterations_Newton_EXPONENTIAL.csv
        │   ├── iterations_Regula_Falsi_EXPONENTIAL.csv
        │   ├── iterations_Secant_EXPONENTIAL.csv
        │   └── iterations_Steffensen_EXPONENTIAL.csv
        ├── mixed/
        │   ├── comparison_summary_MIXED.csv
        │   ├── iterations_Aitken_MIXED.csv
        │   ├── iterations_Bisection_MIXED.csv
        │   ├── iterations_Fixed_Point_MIXED.csv
        │   ├── iterations_Newton_MIXED.csv
        │   ├── iterations_Regula_Falsi_MIXED.csv
        │   ├── iterations_Secant_MIXED.csv
        │   └── iterations_Steffensen_MIXED.csv
        ├── polynomial/
        │   ├── comparison_summary_POLYNOMIAL.csv
        │   ├── iterations_Aitken_POLYNOMIAL.csv
        │   ├── iterations_Bisection_POLYNOMIAL.csv
        │   ├── iterations_Fixed_Point_POLYNOMIAL.csv
        │   ├── iterations_Newton_POLYNOMIAL.csv
        │   ├── iterations_Regula_Falsi_POLYNOMIAL.csv
        │   ├── iterations_Secant_POLYNOMIAL.csv
        │   └── iterations_Steffensen_POLYNOMIAL.csv
        └── trigonometric/
            ├── comparison_summary_TRIGONOMETRIC.csv
            ├── iterations_Aitken_TRIGONOMETRIC.csv
            ├── iterations_Bisection_TRIGONOMETRIC.csv
            ├── iterations_Fixed_Point_TRIGONOMETRIC.csv
            ├── iterations_Newton_TRIGONOMETRIC.csv
            ├── iterations_Regula_Falsi_TRIGONOMETRIC.csv
            ├── iterations_Secant_TRIGONOMETRIC.csv
            └── iterations_Steffensen_TRIGONOMETRIC.csv
```

## 📘 Overview

Root-finding is a fundamental problem in numerical analysis where the goal is to find the value(s) of \( x \) such that \( f(x) = 0 \).  
This project provides an implementation of **seven root-finding algorithms**:

1. **Bisection Method**
2. **Regula Falsi (False Position) Method**
3. **Fixed-Point Iteration Method**
4. **Newton-Raphson Method**
5. **Secant Method**
6. **Aitken's Δ² Acceleration Method**
7. **Steffensen's Method**

Each method is implemented with iteration tracking, convergence checking, and visual comparison.

---

## ⚙️ Features

### 1. **Computation of Roots**

Each of the seven methods can compute the root of a given nonlinear function \( f(x) \) with specified initial guesses and tolerance.

### 2. **Error Convergence Graph**

Displays a **comparison plot of error decay per iteration** for all methods.  
This allows users to visually assess **how fast each algorithm converges** toward the true root.

### 3. **Method Behavior Visualization**

Shows how each method approaches the **exact root** over successive iterations.  
The plot illustrates **the iterative movement of approximations** toward the actual solution.

### 4. **Execution Time Comparison**

A bar chart comparing **execution times** for all methods under identical function and tolerance settings.  
This helps identify which method is the most **computationally efficient**.

### 5. **Comprehensive Output Organization**

Results are systematically organized into:

- **Graphical outputs** (PNG images) in `data/graph/`
- **Tabular data** (CSV files) in `data/table/`
- Organized by function type: exponential, mixed, polynomial, and trigonometric

---

## 📊 Example Visualizations

Below are example plots generated by the program:

1. **Error Reduction per Iteration**

   - Compares the convergence speed among methods.

2. **Root Approximation Progress**

   - Displays how each method approaches the exact root step-by-step.

3. **Execution Time Comparison**
   - Summarizes runtime differences between methods.

All visualizations are automatically saved to the `data/graph/` directory, organized by function type.

---

## 🧮 How It Works

The workflow of the program is as follows:

1. **Define the nonlinear function** \( f(x) \).
2. **Set the initial conditions** (initial guesses, tolerance, maximum iterations).
3. **Run all seven root-finding methods** sequentially.
4. **Store iteration results** (root approximation, error, iteration count, runtime).
5. **Generate plots and CSV files** to visualize and compare results.
6. **Organize outputs** into appropriate directories for easy analysis.

---

## 🧠 Key Concepts Demonstrated

- Convergence analysis
- Numerical stability
- Error reduction rate
- Computational efficiency
- Visualization of iterative methods
- Systematic result organization
- Comparative analysis across multiple function types

---

## 📈 Output Files Description

### CSV Files (in `data/table/`):

- `comparison_summary_*.csv`: Summary statistics for all methods
- `iterations_[Method]_*.csv`: Detailed iteration data for each method

### PNG Files (in `data/graph/`):

- `convergence_error_*.png`: Error reduction over iterations
- `convergence_path_*.png`: Approximation path toward root
- `execution_time_*.png`: Runtime comparison across methods

---

## 🚀 Usage

### Basic Import

```python
import os
from numeric import *
```

### Setting Up Output Directories

```python
main_folder = "data"
table = os.path.join(main_folder, "table")
graph = os.path.join(main_folder, "graph")
```

### Main Analysis Function

```python
def main():
    # Define your function and parameters
    function = "x**7-2*x**6+4*x**5-1"
    g_func = "real_root(2*x**6-4*x**5+1,7)"  # For fixed-point methods
    g_func_1 = "real_root((-x**7+2*x**6+1)/4,5)"  # Alternative fixed-point function
    max_iter = 100
    tol = 1e-10
    low, up = 0, 1
    lower_bound, upper_bound = 0, up

    print("=== NUMERICAL METHODS COMPARISON ===")
    print(f"Function: f(x) = {function}")
    print(f"g(x) for fixed-point: {g_func}")
    print(f"Tolerance: {tol}")
    print(f"Root search interval: [{lower_bound}, {upper_bound}]")
    print()

    # Run all methods with timing
    results = {}
    exec_times = {}

    methods_config = {
        'Bisection': {'method': 'bisection', 'low': low, 'up': up},
        'Regula Falsi': {'method': 'regula_falsi', 'low': low, 'up': up},
        'Secant': {'method': 'secant', 'low': low, 'up': up},
        'Fixed-Point': {'method': 'fixed_point', 'initial_guess': up, 'g_function': g_func_1},
        'Aitken': {'method': 'aitken', 'initial_guess': up, 'g_function': g_func},
        'Steffensen': {'method': 'steffensen', 'initial_guess': up, 'g_function': g_func_1},
        'Newton': {'method': 'newton', 'initial_guess': (low + up) / 2}
    }

    import time
    for name, config in methods_config.items():
        start = time.perf_counter()
        res = numeric(function, tol=tol, max_iter=max_iter, **config)
        end = time.perf_counter()
        results[name] = res
        exec_times[name] = (end - start) * 1000
```

### Viewing Results

```python
    # Display final results
    print("="*80)
    print("FINAL RESULTS AFTER CONVERGENCE OF EACH METHOD")
    print("="*80)
    for index, (name, res) in enumerate(results.items(), 1):
        print(f"\n {index}. Method {name}")
        if res['error']:
            print(f"   Error: {res['error']}")
        elif res['root'] is not None:
            iterations = res['converged_at'] if res['converged_at'] is not None else max_iter
            status = "✓ Converged" if res['converged_at'] is not None else "⚠ Max iterations"
            print(f"   Root: {res['root']:.15f}")
            print(f"   Status: {status} at iteration {iterations}")
        else:
            print("   Failed")
```

### Finding Exact Roots

```python
    # Find all exact roots
    try:
        all_exact_roots = find_all_roots(function, x_range=(-3, 3), steps=2000)
        print(f"\nFound {len(all_exact_roots)} real roots globally:")
        for i, r in enumerate(all_exact_roots, 1):
            print(f"  Root {i}: {r:.15f}")

        exact_roots_in_interval = [r for r in all_exact_roots if lower_bound <= r <= upper_bound]
        print(f"\nRoots in interval [{lower_bound}, {upper_bound}]: {len(exact_roots_in_interval)} found")
        for i, r in enumerate(exact_roots_in_interval, 1):
            print(f"  → Root {i}: {r:.15f}")

    except Exception as e:
        print(f"\nFailed to compute exact solutions: {e}")
        exact_roots_in_interval = []
```

### Saving Results to CSV

```python
    # Save iteration data to CSV
    output_dir = os.path.join(table, "polynomial")
    os.makedirs(output_dir, exist_ok=True)

    for name, res in results.items():
        if not res['error'] and not res['iterations'].empty:
            safe_name = "".join(c if c.isalnum() else "_" for c in name)
            filename = os.path.join(output_dir, f"iterations_{safe_name}_POLYNOMIAL.csv")
            res['iterations'].to_csv(filename, index=False, float_format="%.12f")

    summary_file = os.path.join(output_dir, "comparison_summary_POLYNOMIAL.csv")
    df_summary.to_csv(summary_file, index=False)
```

### Generating Visualizations

```python
    # Generate convergence plots
    graph_poly = os.path.join(graph, "polynomial")
    os.makedirs(graph_poly, exist_ok=True)

    # Convergence error plot (log scale)
    plt.figure(figsize=(12, 8))
    # ... plotting code ...
    plt.savefig(os.path.join(graph_poly, "convergence_error_POLYNOMIAL.png"), dpi=600, bbox_inches='tight')
    plt.show()

    # Convergence path plot
    plt.figure(figsize=(14, 8))
    # ... plotting code ...
    plt.savefig(os.path.join(graph_poly, "convergence_path_POLYNOMIAL.png"), dpi=600, bbox_inches='tight')
    plt.show()

    # Execution time comparison
    plt.figure(figsize=(10, 6))
    # ... plotting code ...
    plt.savefig(os.path.join(graph_poly, "execution_time_POLYNOMIAL.png"), dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
```

### Complete Working Example

For a complete working example, see the provided `testing.ipynb` notebook or create a new script with the following structure:

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numeric import *

def run_comparison(function, func_type="polynomial",
                   low=0, up=1, lower_bound=0, upper_bound=1,
                   g_func=None, g_func_1=None,
                   tol=1e-10, max_iter=100):
    """
    Run complete comparison for a given function.

    Parameters:
    -----------
    function : str
        The function expression as a string
    func_type : str
        Type of function (polynomial, exponential, trigonometric, mixed)
    low, up : float
        Initial interval for bracketing methods
    lower_bound, upper_bound : float
        Interval for root search
    g_func, g_func_1 : str
        Fixed-point iteration functions
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    """

    # Setup directories
    main_folder = "data"
    table_dir = os.path.join(main_folder, "table", func_type)
    graph_dir = os.path.join(main_folder, "graph", func_type)
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    # Default fixed-point functions if not provided
    if g_func is None:
        g_func = f"real_root(2*x**6-4*x**5+1,7)"
    if g_func_1 is None:
        g_func_1 = f"real_root((-x**7+2*x**6+1)/4,5)"

    # Run methods (similar to the example above)
    # ... implementation ...

    return results, exec_times

# Example usage for different function types
if __name__ == "__main__":
    # Polynomial function
    results_poly, times_poly = run_comparison(
        function="x**7-2*x**6+4*x**5-1",
        func_type="polynomial",
        low=0, up=1
    )

    # Exponential function example
    results_exp, times_exp = run_comparison(
        function="exp(x) - 3*x",
        func_type="exponential",
        low=0, up=2,
        g_func="log(3*x)",
        g_func_1="exp(x)/3"
    )
```

### Function Types Supported

The system supports four function types:

1. **polynomial**: Polynomial equations
2. **exponential**: Equations involving exponential functions
3. **trigonometric**: Equations involving trigonometric functions
4. **mixed**: Equations with mixed function types

### Available Methods in `numeric()` Function

| Method       | Required Parameters                                   | Description                              |
| ------------ | ----------------------------------------------------- | ---------------------------------------- |
| Bisection    | `method='bisection'`, `low`, `up`                     | Bracketing method using interval halving |
| Regula Falsi | `method='regula_falsi'`, `low`, `up`                  | False position method                    |
| Secant       | `method='secant'`, `low`, `up`                        | Secant method using two initial points   |
| Fixed-Point  | `method='fixed_point'`, `initial_guess`, `g_function` | Fixed-point iteration                    |
| Newton       | `method='newton'`, `initial_guess`                    | Newton-Raphson method                    |
| Aitken       | `method='aitken'`, `initial_guess`, `g_function`      | Aitken's acceleration                    |
| Steffensen   | `method='steffensen'`, `initial_guess`, `g_function`  | Steffensen's method                      |

### Running Predefined Tests

To run the predefined test cases:

```python
# Execute all test cases from numeric.py
exec(open('numeric.py').read())
```

Or use the Jupyter notebook:

```bash
jupyter notebook testing.ipynb
```

---

## 📝 License

This project is licensed under the terms contained in the `LICENSE` file in the root directory.

---

## 🔧 Dependencies

- Python 3.x
- NumPy
- Matplotlib
- pandas
- Jupyter Notebook (for interactive use)

Install dependencies with:

```bash
pip install numpy matplotlib pandas jupyter
```
