This repository implements a meta-heuristic portfolio selection model using the MEALPy optimization library, combined with Optuna for hyperparameter tuning.
The model optimizes a portfolio matrix using PSO (Particle Swarm Optimization) while respecting structural constraints and minimizing a multi-objective fitness vector.

The approach is tailored for portfolio structures with triangular/irregular allocation masks, and supports complex feasibility rules on both decision matrices.

🚀 Main Features
✔ Multi-Objective Portfolio Optimization

The portfolio is evaluated using five separate criteria:

R – returns

NPV – net present value

V – value

PIR – portfolio interaction risk

Z – combined weighted score

The solver minimizes a vector of:
[-R, -NPV, -V, PIR, Z]

✔ Hybrid Decision Variables

The model works with two decision matrices:

xᵢⱼ → continuous values (0–10)

nᵢⱼ → binary-like continuous values (0–1)

Both follow shape constraints defined by a mask matrix.

✔ Advanced Constraint System

Feasibility checks include:

Minimum return requirement

One chosen item per row

Portfolio cardinality bounds

Logical relationship between xᵢⱼ and nᵢⱼ

Bounding rules

Budget constraints

Capacity requirements

✔ Optuna-Driven PSO Hyperparameter Tuning

Optuna searches for the best:

number of epochs

population size

PSO coefficients: c1, c2

inertia strategy parameter alpha

The best hyperparameters are then used to run a final optimized PSO simulation.
