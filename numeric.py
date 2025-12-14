import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = sp.symbols('x')

# ==============================================================================
# NUMERIC FUNCTIONS
# ==============================================================================
def numeric(function: str, tol: float, max_iter: int, method: str = 'bisection',
            low: float = None, up: float = None,
            initial_guess: float = None, g_function: str = None):
    result = {
        'root': None,
        'iterations': pd.DataFrame(),
        'error': None,
        'converged_at': None
    }

    """
    Numerically solves for a root of a given mathematical function using various iterative methods.

    Parameters:
    -----------
    function : str
        A string representation of the target function f(x) whose root is to be found.
        Must be a valid SymPy-compatible expression (e.g., "x**2 - 4").

    tol : float
        Tolerance threshold for convergence. The algorithm stops when |f(x)| < tol or
        the change between successive approximations is less than tol (depending on method).

    max_iter : int
        Maximum number of iterations allowed before termination, regardless of convergence.

    method : str, optional (default: 'bisection')
        The root-finding algorithm to use. Supported methods:
        - 'bisection'       : Bisection method (requires interval [low, up])
        - 'regula_falsi'    : Regula Falsi (false position) method
        - 'secant'          : Secant method (requires two initial guesses: low, up)
        - 'fixed_point'     : Fixed-point iteration (requires g_function and initial_guess)
        - 'newton'          : Newton-Raphson method (requires initial_guess)
        - 'aitken'          : Aitken's delta-squared acceleration applied to fixed-point iteration
        - 'steffensen'      : Steffensen's method (accelerated fixed-point, requires g_function)

    low : float, optional
        Lower bound of the interval for bracketing methods (bisection, regula_falsi)
        or the first initial guess for the secant method.

    up : float, optional
        Upper bound of the interval for bracketing methods or the second initial guess
        for the secant method.

    initial_guess : float, optional
        Starting point for open methods (Newton, fixed-point, Aitken, Steffensen).

    g_function : str, optional
        String representation of the iteration function g(x) used in fixed-point-based methods,
        where the root of f(x) = 0 is reformulated as x = g(x). Required for:
        'fixed_point', 'aitken', and 'steffensen'.

    Returns:
    --------
    result : dict
        A dictionary containing the following keys:

        - 'root' : float or None
            The computed root approximation. None if an error occurred before computation.

        - 'iterations' : pandas.DataFrame
            A table logging iteration details (columns vary by method). Empty if error occurred early.

        - 'error' : str or None
            Error message if any validation or runtime error occurred; None if no error.

        - 'converged_at' : int or None
            The iteration number at which convergence was achieved (based on tolerance).
            None if the method did not converge within max_iter.

    Notes:
    ------
    - The global SymPy symbol `x` is used internally for parsing expressions.
    - All function evaluations are performed with floating-point arithmetic.
    - Complex results (e.g., from invalid g(x)) are treated as errors in fixed-point variants.
    - Bracketing methods (bisection, regula_falsi) require f(low) * f(up) < 0.
    - Newton’s method requires a non-zero derivative at each step.
    - Aitken and Steffensen methods accelerate fixed-point iteration and require a valid g(x).
    """

    try:
        f = sp.sympify(function)
    except Exception as e:
        result['error'] = f"Invalid function: {e}"
        return result

    method_key = method.lower()
    valid_methods = ['bisection', 'regula_falsi', 'secant', 'fixed_point', 'newton', 'aitken', 'steffensen']
    if method_key not in valid_methods:
        result['error'] = f"Method '{method}' is not recognized"
        return result

    # General validation
    if method_key in ['bisection', 'regula_falsi']:
        if low is None or up is None or low >= up:
            result['error'] = "This method requires low < up"
            return result

    if method_key == 'secant':
        if low is None or up is None:
            result['error'] = "Secant method requires two initial guesses (low, up)"
            return result

    # g(x)-based methods require g_function and initial_guess
    if method_key in ['fixed_point', 'aitken', 'steffensen']:
        if g_function is None:
            result['error'] = f"Method {method_key} requires g_function"
            return result
        if initial_guess is None:
            result['error'] = f"Method {method_key} requires initial_guess"
            return result
        try:
            g = sp.sympify(g_function)
        except Exception as e:
            result['error'] = f"Invalid g_function: {e}"
            return result

    if method_key == 'newton':
        if initial_guess is None:
            result['error'] = "Newton's method requires initial_guess"
            return result
        try:
            df_expr = sp.diff(f, x)
        except Exception as e:
            result['error'] = f"Failed to compute derivative: {e}"
            return result

    # Sign check for interval-based methods
    if method_key in ['bisection', 'regula_falsi']:
        try:
            f_low = float(f.subs(x, low).evalf())
            f_up = float(f.subs(x, up).evalf())
            if f_low * f_up > 0:
                result['error'] = f"No root in [{low}, {up}]: f(low)={f_low:.6f}, f(up)={f_up:.6f}"
                return result
        except Exception as e:
            result['error'] = f"Error evaluating bounds: {e}"
            return result

    iteration_data = []

    try:
        converged = False

        # ========== BISECTION ==========
        if method_key == 'bisection':
            a, b = float(low), float(up)
            for i in range(1, max_iter + 1):
                c = (a + b) / 2
                f_c = float(f.subs(x, c).evalf())
                iteration_data.append({'iterasi': i, 'a': a, 'b': b, 'c': c, 'f(c)': f_c})
                if abs(f_c) < tol:
                    result['root'] = c
                    result['converged_at'] = i
                    converged = True
                    break
                f_a = float(f.subs(x, a).evalf())
                if f_a * f_c < 0:
                    b = c
                else:
                    a = c
            if not converged:
                result['root'] = c

        # ========== REGULA FALSI ==========
        elif method_key == 'regula_falsi':
            a, b = float(low), float(up)
            for i in range(1, max_iter + 1):
                f_a = float(f.subs(x, a).evalf())
                f_b = float(f.subs(x, b).evalf())
                if abs(f_b - f_a) < 1e-15:
                    result['error'] = "f(b) - f(a) is too small"
                    return result
                c = (a * f_b - b * f_a) / (f_b - f_a)
                f_c = float(f.subs(x, c).evalf())
                iteration_data.append({'iterasi': i, 'a': a, 'b': b, 'c': c, 'f(c)': f_c})
                if abs(f_c) < tol:
                    result['root'] = c
                    result['converged_at'] = i
                    converged = True
                    break
                if f_a * f_c < 0:
                    b = c
                else:
                    a = c
            if not converged:
                result['root'] = c

        # ========== SECANT ==========
        elif method_key == 'secant':
            x0, x1 = float(low), float(up)
            f_x0 = float(f.subs(x, x0).evalf())
            f_x1 = float(f.subs(x, x1).evalf())
            for i in range(1, max_iter + 1):
                if abs(f_x1 - f_x0) < 1e-15:
                    result['error'] = "f(x1) - f(x0) is too small"
                    return result
                x2 = x1 - (f_x1 * (x1 - x0)) / (f_x1 - f_x0)
                f_x2 = float(f.subs(x, x2).evalf())
                iteration_data.append({'iterasi': i, 'x0': x0, 'x1': x1, 'x2': x2, 'f(x2)': f_x2})
                if abs(f_x2) < tol:
                    result['root'] = x2
                    result['converged_at'] = i
                    converged = True
                    break
                x0, x1 = x1, x2
                f_x0, f_x1 = f_x1, f_x2
            if not converged:
                result['root'] = x2

        # ========== FIXED-POINT ==========
        elif method_key == 'fixed_point':
            x_prev = float(initial_guess)
            for i in range(1, max_iter + 1):
                x_next_expr = g.subs(x, x_prev).evalf()
                x_next = complex(x_next_expr)
                if abs(x_next.imag) > 1e-10:
                    result['error'] = f"Complex result at iteration {i}"
                    return result
                x_next = x_next.real
                f_val = float(f.subs(x, x_prev).evalf())
                error = abs(x_next - x_prev)
                iteration_data.append({
                    'iterasi': i,
                    'x_n': x_prev,
                    'x_next': x_next,
                    'error': error,
                    'f(x_n)': f_val
                })
                if error < tol or abs(f_val) < tol:
                    result['root'] = x_next
                    result['converged_at'] = i
                    converged = True
                    break
                x_prev = x_next
            if not converged:
                result['root'] = x_prev

        # ========== NEWTON ==========
        elif method_key == 'newton':
            x_curr = float(initial_guess)
            for i in range(1, max_iter + 1):
                f_val = float(f.subs(x, x_curr).evalf())
                df_val = float(df_expr.subs(x, x_curr).evalf())
                if abs(df_val) < 1e-15:
                    result['error'] = f"Zero derivative at iteration {i}"
                    return result
                x_next = x_curr - f_val / df_val
                error = abs(x_next - x_curr)
                iteration_data.append({
                    'iterasi': i,
                    'x_n': x_curr,
                    'f(x_n)': f_val,
                    "f'(x_n)": df_val,
                    'error': error
                })
                if error < tol or abs(f_val) < tol:
                    result['root'] = x_next
                    result['converged_at'] = i
                    converged = True
                    break
                x_curr = x_next
            if not converged:
                result['root'] = x_curr

        # ========== AITKEN'S DELTA-SQUARED ==========
        elif method_key == 'aitken':
            x0 = float(initial_guess)
            for i in range(1, max_iter + 1):
                # Compute x1 = g(x0), x2 = g(x1)
                x1_expr = g.subs(x, x0).evalf()
                x1 = complex(x1_expr).real
                x2_expr = g.subs(x, x1).evalf()
                x2 = complex(x2_expr).real

                denom = x2 - 2*x1 + x0
                if abs(denom) < 1e-15:
                    # If denominator is zero, use x0 as fallback
                    x_hat = x0
                    f_val = float(f.subs(x, x_hat).evalf())
                    error = 0.0  # or NaN
                    iteration_data.append({
                        'iterasi': i,
                        'x0': x0,
                        'x1': x1,
                        'x2': x2,
                        'x_hat': x_hat,
                        'error': error,
                        'f(x_hat)': f_val
                    })
                    # Continue to next iteration
                    # Record error but do not stop the process
                    if result['error'] is None:
                        result['error'] = f"Aitken denominator zero at iteration {i} — fallback to x0 used"
                    # Still update x0 for the next iteration
                    x0 = x_hat
                else:
                    # Aitken's formula: x_hat = x0 - (x1 - x0)^2 / (x2 - 2*x1 + x0)
                    x_hat = x0 - (x1 - x0)**2 / denom
                    f_val = float(f.subs(x, x_hat).evalf())
                    error = abs(x_hat - x0)

                    iteration_data.append({
                        'iterasi': i,
                        'x0': x0,
                        'x1': x1,
                        'x2': x2,
                        'x_hat': x_hat,
                        'error': error,
                        'f(x_hat)': f_val
                    })

                    if abs(f_val) < tol or error < tol:
                        result['root'] = x_hat
                        result['converged_at'] = i
                        converged = True
                        break

                    x0 = x_hat  # Use accelerated result as next starting point

            if not converged:
                result['root'] = x_hat

        # ========== STEFFENSEN'S METHOD ==========
        elif method_key == 'steffensen':
            x_curr = float(initial_guess)
            for i in range(1, max_iter + 1):
                gx = float(g.subs(x, x_curr).evalf())
                ggx = float(g.subs(x, gx).evalf())
                denom = ggx - 2*gx + x_curr
                if abs(denom) < 1e-15:
                    result['error'] = f"Steffensen denominator zero at iteration {i}"
                    return result
                x_next = x_curr - (gx - x_curr)**2 / denom
                f_val = float(f.subs(x, x_next).evalf())
                error = abs(x_next - x_curr)

                iteration_data.append({
                    'iterasi': i,
                    'x': x_curr,
                    'g(x)': gx,
                    'g(g(x))': ggx,
                    'x_next': x_next,
                    'error': error,
                    'f(x_next)': f_val
                })

                if abs(f_val) < tol or error < tol:
                    result['root'] = x_next
                    result['converged_at'] = i
                    converged = True
                    break

                x_curr = x_next  # update for next iteration

            if not converged:
                result['root'] = x_next

        result['iterations'] = pd.DataFrame(iteration_data)
        return result

    except Exception as e:
        result['error'] = f"Internal error: {e}"
        return result

def find_all_roots(function_str: str, x_range: tuple = (-3, 3), steps: int = 1000, tol: float = 1e-12):
    """
    Find all real roots of the function within the interval [x_min, x_max]
    by scanning for sign changes and using nsolve.
    """
    x = sp.symbols('x')
    f = sp.sympify(function_str)
    f_lambdified = sp.lambdify(x, f, 'numpy')

    x_min, x_max = x_range
    xs = np.linspace(x_min, x_max, steps)
    fs = f_lambdified(xs)

    # Avoid NaN/inf
    valid = np.isfinite(fs)
    xs = xs[valid]
    fs = fs[valid]

    roots = []
    for i in range(len(xs) - 1):
        if fs[i] == 0:
            root = float(xs[i])
            roots.append(root)
        elif fs[i] * fs[i+1] < 0:  # Sign change → root exists
            try:
                guess = (xs[i] + xs[i+1]) / 2
                root = float(sp.nsolve(f, guess, tol=tol, maxsteps=100))
                # Avoid duplicates (very close roots)
                if not any(abs(root - r) < 1e-8 for r in roots):
                    roots.append(root)
            except Exception:
                continue  # nsolve failed, skip

    roots.sort()
    return roots