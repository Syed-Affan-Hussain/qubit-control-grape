# Numerical Implementation of GRAPE Algorithm for Optimal Quantum State Transfer in a Single Qubit System

This repository contains a Python implementation of the **GRAPE (Gradient Ascent Pulse Engineering)** algorithm to optimize quantum control pulses for a single qubit system. The goal is to find time-dependent control fields that drive a qubit from a known initial state to a desired target state with high fidelity.

## Project Overview

The GRAPE algorithm is used to optimize control pulses that steer a quantum system to a target state by maximizing state fidelity. This implementation includes:
- Definition of a single qubit system with Pauli operators.
- Forward and backward propagation of quantum states.
- Gradient-based optimization using the adjoint method.
- Visualization of the optimization process, including fidelity history, control pulses, and the Bloch sphere trajectory.

The code uses **NumPy**, **Matplotlib**, and **SciPy** for numerical computations and plotting. The system evolves a qubit from the $|0\rangle$ state to the $|+\rangle$ state (or other target states) using piecewise constant control pulses.

## Mathematical Formulation

The objective is to find time-dependent control fields that maximize the fidelity between the final evolved state and a target state. Below is the mathematical framework implemented in the code.

### 1. Quantum System Definition

#### 1.1 Basis States
The computational basis states for a single qubit are:
\[
|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}
\]
These are complex vectors in the Hilbert space $\mathcal{H} \cong \mathbb{C}^2$.

#### 1.2 Pauli Matrices
The Pauli matrices (with $\hbar=1$) are:
\[
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
\]
The identity operator is:
\[
I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
\]

#### 1.3 Qubit State Representation (Bloch Sphere)
A general pure state is:
\[
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle
\]
where $\alpha, \beta \in \mathbb{C}$ and $|\alpha|^2 + |\beta|^2 = 1$. Using Bloch sphere coordinates $(\theta, \phi_{\text{azimuthal}})$:
\[
|\psi(\theta, \phi_{\text{azimuthal}})\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi_{\text{azimuthal}}}\sin\left(\frac{\theta}{2}\right)|1\rangle
\]
The Cartesian Bloch vector $\vec{r} = (r_x, r_y, r_z)$ is:
\[
r_x = 2 \text{Re}(\alpha^* \beta), \quad r_y = 2 \text{Im}(\alpha^* \beta), \quad r_z = |\alpha|^2 - |\beta|^2
\]
For pure states, $r_x^2 + r_y^2 + r_z^2 = 1$.

### 2. Quantum Dynamics

#### 2.1 Hamiltonian
The total Hamiltonian is:
\[
H(t) = H_0 + \sum_{k=1}^{N_c} c_k(t) H_k
\]
where $H_0 = \frac{1}{2} \omega_0 \sigma_z$ is the drift Hamiltonian, $H_k \in \{\sigma_x, \sigma_y\}$ are control Hamiltonians, and $c_k(t)$ are control amplitudes.

#### 2.2 Time Evolution (Propagator)
The state evolves via the Schrödinger equation ($\hbar=1$):
\[
i \frac{d}{dt}|\psi(t)\rangle = H(t)|\psi(t)\rangle
\]
The unitary propagator $U(t_f, t_i)$ is:
\[
|\psi(t_f)\rangle = U(t_f, t_i)|\psi(t_i)\rangle
\]
For a time-independent $H$ over $\Delta t$:
\[
U(\Delta t) = \exp(-i H \Delta t)
\]

#### 2.3 Piecewise Constant Controls
The evolution time $T$ is divided into $N_s$ segments of duration $\Delta t = T/N_s$. In segment $j$, the Hamiltonian is:
\[
H_j = H_0 + \sum_{k=1}^{N_c} c_{k,j} H_k
\]
The segment propagator is:
\[
U_j = \exp(-i H_j \Delta t)
\]
The total propagator is:
\[
U_{\text{total}} = U_{N_s} U_{N_s-1} \cdots U_1
\]

#### 2.4 State Evolution
The state after segment $j$ is:
\[
|\psi(t_j)\rangle = U_j |\psi(t_{j-1})\rangle
\]
The final state is:
\[
|\psi(T)\rangle = U_{\text{total}} |\psi_0\rangle
\]

### 3. Objective Function: State Fidelity
The fidelity between the final state $|\psi(T)\rangle$ and target state $|\psi_{\text{target}}\rangle$ is:
\[
\mathcal{F} = |\langle \psi_{\text{target}} | \psi(T) \rangle|^2 = |\langle \psi_{\text{target}} | U_{\text{total}} | \psi_0 \rangle|^2
\]

### 4. Gradient Calculation (Adjoint Method)
The gradient of fidelity with respect to control $c_{k,j}$ is:
\[
\frac{\partial \mathcal{F}}{\partial c_{k,j}} = -2 \Delta t \cdot \text{Im} \left( \langle \lambda(t_j) | H_k | \psi(t_{j-1}) \rangle \right)
\]
where $|\psi(t_{j-1})\rangle$ is the forward-propagated state, and $|\lambda(t_j)\rangle = U_{j+1}^\dagger \cdots U_{N_s}^\dagger |\psi_{\text{target}}\rangle$ is the backward-propagated adjoint state.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/grape-qubit-optimization.git
   cd grape-qubit-optimization
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy
   ```
3. Run the script:
   ```bash
   python grape_qubit.py
   ```

## Usage

The main script (`grape_qubit.py`) runs the GRAPE optimization and generates three plots:
- **Fidelity History**: Convergence of the fidelity over iterations.
- **Control Pulses**: Optimized control amplitudes over time.
- **Bloch Sphere Trajectory**: Path of the qubit state on the Bloch sphere.

To modify the target state or parameters:
- Edit `initial_state_qubit` and `target_state_qubit` in the `if __name__ == "__main__":` block.
- Adjust `total_time_qubit`, `num_segments_qubit`, `learning_rate_qubit`, or `num_iterations_qubit` for different optimization settings.

Example output:
![Example Plots](example_output.png) *(Add your own screenshot to the repository)*

## Code

Below is the complete Python implementation (`grape_qubit.py`):

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.array([[1, 0], [0, 1]], dtype=complex)

def bloch_coords_to_state(theta: float, phi_azimuthal: float) -> np.ndarray:
    return np.cos(theta / 2) * ket0 + np.exp(1j * phi_azimuthal) * np.sin(theta / 2) * ket1

def state_to_cartesian_bloch_vector(state_vector: np.ndarray) -> tuple[float, float, float]:
    state_vector = state_vector / np.linalg.norm(state_vector)
    alpha = state_vector[0]
    beta = state_vector[1]
    rx = 2 * np.real(alpha * np.conj(beta))
    ry = 2 * np.imag(alpha * np.conj(beta))
    rz = np.abs(alpha)**2 - np.abs(beta)**2
    return rx, ry, rz

def get_hamiltonian(H0, Hc_ops, controls):
    H = H0.copy()
    for i, control in enumerate(controls):
        H += control * Hc_ops[i]
    return H

def get_propagator(H, dt):
    return expm(-1j * H * dt)

def get_propagation(H0, Hc_ops, control_pulses, total_time, num_segments):
    dt = total_time / num_segments
    propagators = []
    U_total = identity.copy()
    for i in range(num_segments):
        current_controls = control_pulses[i]
        H = get_hamiltonian(H0, Hc_ops, current_controls)
        U_segment = get_propagator(H, dt)
        propagators.append(U_segment)
        U_total = U_segment @ U_total
    return propagators, U_total

def evolve_state(initial_state, propagators):
    states = [initial_state.copy()]
    current_state = initial_state.copy()
    for U_segment in propagators:
        current_state = U_segment @ current_state
        states.append(current_state)
    return states

def state_fidelity(final_state, target_state):
    final_state = final_state / np.linalg.norm(final_state)
    target_state = target_state / np.linalg.norm(target_state)
    return np.abs(np.vdot(target_state, final_state))**2

def calculate_gradient(H0, Hc_ops, control_pulses, total_time, num_segments, initial_state, target_state):
    num_controls = len(Hc_ops)
    dt = total_time / num_segments
    gradient = np.zeros_like(control_pulses, dtype=float)
    propagators, U_total = get_propagation(H0, Hc_ops, control_pulses, total_time, num_segments)
    states_forward = evolve_state(initial_state, propagators)
    lambda_backward = target_state.copy()
    dt_hbar = dt
    for i in range(num_segments - 1, -1, -1):
        current_controls = control_pulses[i]
        H_i = get_hamiltonian(H0, Hc_ops, current_controls)
        U_i = propagators[i]
        psi_i = states_forward[i]
        for j in range(num_controls):
            Hc_j = Hc_ops[j]
            gradient[i, j] = -2 * np.imag(np.vdot(lambda_backward, Hc_j @ psi_i)) * dt_hbar
        lambda_backward = U_i.T.conj() @ lambda_backward
    return gradient

def grape_optimize(H0, Hc_ops, initial_state, target_state, total_time, num_segments, num_controls, learning_rate, num_iterations, initial_control_guess=None):
    if initial_control_guess is not None:
        control_pulses = initial_control_guess
    else:
        control_pulses = (np.random.rand(num_segments, num_controls) - 0.5) * 2 * np.pi
    objective_history = []
    final_state_bloch_history = []
    print("Starting GRAPE optimization...")
    for iteration in range(num_iterations):
        gradient = calculate_gradient(H0, Hc_ops, control_pulses, total_time, num_segments, initial_state, target_state)
        control_pulses += learning_rate * gradient
        _, U_total = get_propagation(H0, Hc_ops, control_pulses, total_time, num_segments)
        final_state = U_total @ initial_state
        objective = state_fidelity(final_state, target_state)
        objective_history.append(objective)
        final_state_bloch_history.append(state_to_cartesian_bloch_vector(final_state))
        if (iteration + 1) % (num_iterations // 10) == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Fidelity: {objective:.6f}")
    print("Optimization finished.")
    return control_pulses, objective_history, final_state_bloch_history

def plot_bloch_trajectory(ax, x_coords: list, y_coords: list, z_coords: list, title: str, show_start_end=True, color='b'):
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    xs_sphere = np.cos(u)*np.sin(v)
    ys_sphere = np.sin(u)*np.sin(v)
    zs_sphere = np.cos(v)
    ax.plot_wireframe(xs_sphere, ys_sphere, zs_sphere, color="gray", alpha=0.2, linewidth=0.5)
    ax.plot(x_coords, y_coords, z_coords, color=color, lw=2.0, label='Trajectory')
    if show_start_end and len(x_coords) > 0:
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='g', s=60, label='Start', depthshade=True, zorder=10)
        ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='r', s=60, label='End', depthshade=True, zorder=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(title, pad=15)
    ax.legend()

def plot_bloch_points(ax, states, labels, colors):
    for state, label, color in zip(states, labels, colors):
        x, y, z = state_to_cartesian_bloch_vector(state)
        ax.scatter(x, y, z, color=color, s=100, label=label, depthshade=True, zorder=10)

if __name__ == "__main__":
    omega0 = 1.0
    H0_qubit = 0.5 * omega0 * sigma_z
    Hc_ops_qubit = [sigma_x, sigma_y]
    num_controls_qubit = len(Hc_ops_qubit)
    initial_state_qubit = ket0
    target_state_qubit = (ket0 + ket1) / np.sqrt(2)
    total_time_qubit = 2*np.pi
    num_segments_qubit = 200
    learning_rate_qubit = 0.1
    num_iterations_qubit = 500
    initial_control_guess = np.zeros((num_segments_qubit, num_controls_qubit))
    initial_control_guess[:, 1] = np.pi / total_time_qubit
    optimized_controls, fidelity_history, final_bloch_history = grape_optimize(
        H0_qubit,
        Hc_ops_qubit,
        initial_state_qubit,
        target_state_qubit,
        total_time_qubit,
        num_segments_qubit,
        num_controls_qubit,
        learning_rate_qubit,
        num_iterations_qubit,
        initial_control_guess=initial_control_guess
    )
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(fidelity_history)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fidelity")
    ax1.set_title("Fidelity Optimization History")
    ax1.grid(True)
    ax1.set_ylim([0, 1.05])
    ax2 = fig.add_subplot(1, 3, 2)
    time_points_step = np.linspace(0, total_time_qubit, num_segments_qubit + 1)
    ax2.step(time_points_step, np.append(optimized_controls[:, 0], optimized_controls[-1:, 0]), where='post', label='Control 1 (Hc_ops[0] - sigma_x)')
    if num_controls_qubit > 1:
        ax2.step(time_points_step, np.append(optimized_controls[:, 1], optimized_controls[-1:, 1]), where='post', label='Control 2 (Hc_ops[1] - sigma_y)')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Control Amplitude")
    ax2.set_title("Optimized Control Pulses (Piecewise Constant)")
    ax2.legend()
    ax2.grid(True)
    ax_bloch = fig.add_subplot(1, 3, 3, projection='3d')
    dt = total_time_qubit / num_segments_qubit
    current_state = initial_state_qubit.copy()
    x_traj, y_traj, z_traj = [], []
    rx, ry, rz = state_to_cartesian_bloch_vector(initial_state_qubit)
    x_traj.append(rx)
    y_traj.append(ry)
    z_traj.append(rz)
    for i in range(num_segments_qubit):
        current_controls = optimized_controls[i]
        H = get_hamiltonian(H0_qubit, Hc_ops_qubit, current_controls)
        U_segment = get_propagator(H, dt)
        current_state = U_segment @ current_state
        rx, ry, rz = state_to_cartesian_bloch_vector(current_state)
        x_traj.append(rx)
        y_traj.append(ry)
        z_traj.append(rz)
    plot_bloch_trajectory(ax_bloch, x_traj, y_traj, z_traj, 
                          f"Optimized Bloch Trajectory ({num_segments_qubit} segments)")
    plot_bloch_points(ax_bloch,
                      [initial_state_qubit, target_state_qubit],
                      ['Initial State', 'Target State'],
                      ['g', 'r'])
    plt.tight_layout()
    plt.show()
```

## Repository Structure

```
grape-qubit-optimization/
├── README.md              # Project description and documentation
├── grape_qubit.py         # Main Python script
├── example_output.png       # Screenshot of output plots (add your own)
└── LICENSE                # License file (e.g., MIT)
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is based on the GRAPE algorithm as described in quantum control literature. Thanks to the open-source community for NumPy, SciPy, and Matplotlib.
