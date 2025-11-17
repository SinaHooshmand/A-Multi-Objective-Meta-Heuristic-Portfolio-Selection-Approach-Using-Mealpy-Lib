import numpy as np
import pandas as pd
from Parameters import parameters, config
from mealpy.utils.problem import FloatVar, IntegerVar
from mealpy.swarm_based import PSO
import optuna
from optuna_dashboard import run_server

class Portfolio:
    def __init__(self):
        self.x_ij = np.zeros((parameters.num_i, parameters.num_j))
        self.n_ij = np.zeros((parameters.num_i, parameters.num_j))

    def random_initializer(self):
        # Only used if needed, for PSO it will generate agents
        mask = np.array([
            [1]*11,
            [1]*10 + [np.nan],
            [1]*9 + [np.nan]*2,
            [1]*4 + [np.nan]*7
        ])
        self.x_ij = np.random.randint(0, 10, self.x_ij.shape) * mask
        self.n_ij = np.random.randint(0, 2, self.n_ij.shape) * mask

    def calculate_objectives(self):
        # Ensure all are scalars
        R = float(np.nansum(np.multiply(self.x_ij, config.r_ij)))
        npv = float(np.nansum(np.multiply(config.npv_ij, self.n_ij)))
        v = float(np.nansum(np.multiply(config.v_ij, self.n_ij)))
        pir = float(np.nansum(np.multiply(np.multiply(config.p_ijk, config.It + config.Ic + config.Iq), np.expand_dims(self.n_ij, axis=2))))
        z = float(parameters.W1 / R + parameters.W2 / npv + parameters.W3 / v + parameters.W4 * pir)
        return R, npv, v, pir, z

    def is_feasible(self):
        valid_mask_x = ~np.isnan(self.x_ij)
        valid_mask_n = ~np.isnan(self.n_ij)

        eq6 = np.nansum(np.multiply(config.p_ij, valid_mask_x)) >= parameters.r_min
        eq7 = np.nansum(valid_mask_x, axis=1) == 1  # each row only one non-NaN
        eq8 = parameters.n_min <= np.nansum(valid_mask_n) <= parameters.n_max
        eq9 = np.all(((valid_mask_x > 0) & (valid_mask_n == 1)) | ((valid_mask_x == 0) & (valid_mask_n == 0)))
        eq10 = np.all(valid_mask_x >= 0)
        eq12 = np.all(np.multiply(self.x_ij, parameters.B) <= config.c_ij)
        eq13 = np.nansum(np.multiply(config.c_ij, valid_mask_n)) <= parameters.B

        return np.all(eq6) and np.all(eq7) and eq8 and eq9 and eq10 and eq12 and eq13

# --- Objective function for MOPSO ---
def multi_objective_function(x):
    p = Portfolio()
    num_i, num_j = parameters.num_i, parameters.num_j
    p.x_ij = x[:num_i*num_j].reshape((num_i, num_j))
    p.n_ij = x[num_i*num_j:].reshape((num_i, num_j))

    #if not p.is_feasible():
        # Penalize infeasible solutions
        #return [1e6, 1e6, 1e6, 1e6, 1e6]

    R, npv, v, pir, z = p.calculate_objectives()
    # Maximize R, npv, v → minimize negative; Minimize pir, z as usual
    return [-R, -npv, -v, pir, z]

# --- Constraints function (optional, handled by penalty above) ---
def constraints_function(x):
    p = Portfolio()
    num_i, num_j = parameters.num_i, parameters.num_j
    p.x_ij = x[:num_i*num_j].reshape((num_i, num_j))
    p.n_ij = x[num_i*num_j:].reshape((num_i, num_j))
    return [0] if p.is_feasible() else [1]

# --- Problem setup ---
num_i, num_j = parameters.num_i, parameters.num_j
size_x = num_i * num_j
size_n = num_i * num_j

bounds_x = FloatVar(lb=[0]*size_x, ub=[10]*size_x, name="x")
bounds_n = FloatVar(lb=[0]*size_n, ub=[.99]*size_n, name="n")

problem_dict = {
    "obj_func": multi_objective_function,
    "minmax": "min",  # multi-objective handled internally
    "bounds": [bounds_x, bounds_n],
    "constraints": constraints_function
}

# --- Optuna objective ---
def optuna_objective(trial):
    epoch = trial.suggest_int("epoch", 1000, 5000)
    pop_size = trial.suggest_int("pop_size", 20, 80)
    c1 = trial.suggest_float("c1", 1.0, 2.5)
    c2 = trial.suggest_float("c2", 1.0, 2.5)
    alpha = trial.suggest_float("alpha", 0.4, 0.95)

    model = PSO.AIW_PSO(epoch=epoch, pop_size=pop_size, c1=c1, c2=c2, alpha=alpha)
    g_best = model.solve(problem_dict)
    # Optuna minimizes a single scalar; we can take sum of first 3 objectives for tuning
    return sum(g_best.target.objectives)

# --- Run Optuna study ---
storage = "sqlite:///optuna_study.db"
study = optuna.create_study(direction="minimize", storage=storage, study_name="mystudy", load_if_exists=True)
study.optimize(optuna_objective, n_trials=1, n_jobs=1)

best_params = study.best_params
print("Best hyperparameters found by Optuna:")
print(best_params)

# --- Solve final MOPSO with best hyperparameters ---
best_model = PSO.AIW_PSO(**best_params)
g_best = best_model.solve(problem_dict)

# Extract solution
best_flat = np.array(g_best.solution)
best_x_ij = best_flat[:size_x].reshape((num_i, num_j))
best_n_ij = best_flat[size_x:].reshape((num_i, num_j))

mask = np.array([
    [1]*11 + [0]*0,
    [1]*10 + [0]*1,
    [1]*9  + [0]*2,
    [1]*4  + [0]*7
])

print(f"\nBest fitness vector: {g_best.target.objectives}")

#print(f"Whole Table{pd.DataFrame(best_x_ij * mask)}\n", f"Each Row Summation{np.sum(best_x_ij * mask, axis=1)}\n", f"Whole Table Sumation {np.sum(best_x_ij * mask)}")

print("\nBest n_ij matrix:")
thresholded_n_ij = best_n_ij.copy()  # make a copy so original is not modified
thresholded_n_ij[thresholded_n_ij > 0.8] = 0
thresholded_n_ij[thresholded_n_ij < 0.4]  = 0
thresholded_n_ij = thresholded_n_ij * mask
print("\nBest n_ij matrix (thresholded):")
print(f"Whole Table:\n{pd.DataFrame(np.round(thresholded_n_ij, 3))}")
print(f"Each Row Summation: {np.round(np.sum(thresholded_n_ij, axis=1), 3)}")
print(f"Whole Table Summation: {np.round(np.sum(thresholded_n_ij), 3)}")

print("\nBest x_ij matrix:")
thresholded_n_ij[(thresholded_n_ij >= 0.4) & (thresholded_n_ij <= 0.8)] = 1
print(f"Whole Table\n{pd.DataFrame(np.round(best_x_ij * thresholded_n_ij, 3))}\n", f"Each Row Summation{np.round(np.sum(best_x_ij * thresholded_n_ij, axis=1), 3)}\n", f"Whole Table Sumation {np.round(np.sum(best_x_ij * thresholded_n_ij), 3)}")

# --- Run Optuna dashboard ---
run_server(storage="sqlite:///optuna_study.db", host="127.0.0.1", port=8080)