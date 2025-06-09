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
    x_traj, y_traj, z_traj = [], [], []
    rx_start, ry_start, rz_start = state_to_cartesian_bloch_vector(initial_state_qubit)
    x_traj.append(rx_start)
    y_traj.append(ry_start)
    z_traj.append(rz_start)
    for i in range(num_segments_qubit):
        current_controls = optimized_controls[i]
        H = get_hamiltonian(H0_qubit, Hc_ops_qubit, current_controls)
        U_segment = get_propagator(H, dt)
        current_state = U_segment @ current_state
        rx, ry, rz = state_to_cartesian_bloch_vector(current_state)
        x_traj.append(rx)
        y_traj.append(ry)
        z_traj.append(rz)
    plot_bloch_trajectory(ax_bloch, x_traj, y_traj, z_traj, f"Optimized Bloch Trajectory ({num_segments_qubit} segments)")
    plot_bloch_points(ax_bloch,
                      [initial_state_qubit, target_state_qubit],
                      ['Initial State', 'Target State'],
                      ['g', 'r'])
    plt.tight_layout()
    plt.show()
