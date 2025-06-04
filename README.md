# Numerical Implementation of GRAPE Algorithm for Optimal Quantum State Transfer in a Single Qubit System
In this project, I implemented the GRAPE (Gradient Ascent Pulse Engineering) algorithm to optimize quantum control pulses for a single qubit system. My objective was to find time-dependent control fields that drive a qubit from a known initial state to a desired target state with high fidelity.

## Mathematical Formulation of the GRAPE Implementation

We aim to find time-dependent control fields that steer a quantum system from a given initial state to a desired target state with high fidelity. The GRAPE algorithm achieves this by iteratively updating the control amplitudes based on the gradient of the fidelity with respect to these amplitudes.

### 1. Quantum System Definition

**1.1 Basis States:**  
The computational basis states for a single qubit are denoted as:  
$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$  
These are complex vectors in the Hilbert space $\mathcal{H} \cong \mathbb{C}^2$.

**1.2 Pauli Matrices:**  
The Pauli matrices are fundamental operators for a qubit system (setting $\hbar=1$):  
$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$  
The identity operator is:  
$$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

**1.3 Qubit State Representation (Bloch Sphere):**  
A general pure state of a qubit can be written as:  
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$  
where $\alpha, \beta \in \mathbb{C}$ and $|\alpha|^2 + |\beta|^2 = 1$.  
This state can also be parameterized using Bloch sphere coordinates $(\theta, \phi_{azimuthal})$:  
$$|\psi(\theta, \phi_{azimuthal})\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi_{azimuthal}}\sin\left(\frac{\theta}{2}\right)|1\rangle$$  
where $\theta \in [0, \pi]$ is the polar angle and $\phi_{azimuthal} \in [0, 2\pi)$ is the azimuthal angle.

The state vector can be mapped to a Cartesian Bloch vector $\vec{r} = (r_x, r_y, r_z)$:  
$$r_x = \langle\psi|\sigma_x|\psi\rangle = 2 \text{Re}(\alpha^* \beta)$$  
$$r_y = \langle\psi|\sigma_y|\psi\rangle = 2 \text{Im}(\alpha^* \beta)$$  
$$r_z = \langle\psi|\sigma_z|\psi\rangle = |\alpha|^2 - |\beta|^2$$  
For pure states, $\|\vec{r}\| = r_x^2 + r_y^2 + r_z^2 = 1$.

### 2. Quantum Dynamics

**2.1 Hamiltonian:**  
The total Hamiltonian of the system, $H(t)$, is composed of a time-independent drift Hamiltonian $H_0$ and a sum of time-dependent control Hamiltonians $H_k$, each modulated by a control function $c_k(t)$:  
$$H(t) = H_0 + \sum_{k=1}^{N_c} c_k(t) H_k$$  
where $N_c$ is the number of control fields.  
In the provided code, $H_0 = \frac{1}{2} \omega_0 \sigma_z$ and $H_k \in \{\sigma_x, \sigma_y\}$.

**2.2 Time Evolution (Propagator):**  
The evolution of the quantum state $|\psi(t)\rangle$ is governed by the Schrödinger equation (with $\hbar=1$):  
$$i \frac{d}{dt}|\psi(t)\rangle = H(t)|\psi(t)\rangle$$  
The unitary time evolution operator (propagator) $U(t_f, t_i)$ evolves the state from time $t_i$ to $t_f$:  
$$|\psi(t_f)\rangle = U(t_f, t_i)|\psi(t_i)\rangle$$  
The propagator satisfies:  
$$i \frac{d}{dt}U(t, t_0) = H(t)U(t, t_0), \quad U(t_0, t_0) = I$$  
For a time-independent Hamiltonian $H$ over a small time interval $\Delta t$, the propagator is:  
$$U(\Delta t) = \exp(-i H \Delta t)$$

**2.3 Piecewise Constant Controls:**  
The total evolution time $T$ is divided into $N_s$ small segments, each of duration $\Delta t = T/N_s$. Within each segment $j$ (from time $t_j = (j-1)\Delta t$ to $t_{j+1} = j\Delta t$), the control amplitudes $c_{k,j}$ are assumed to be constant.  
The Hamiltonian for the $j$-th segment is:  
$$H_j = H_0 + \sum_{k=1}^{N_c} c_{k,j} H_k$$  
The propagator for the $j$-th segment is:  
$$U_j = \exp(-i H_j \Delta t)$$  
The total propagator $U_{total}$ from $t=0$ to $t=T$ is the product of the segment propagators:  
$$U_{total} = U_{N_s} U_{N_s-1} \cdots U_2 U_1$$

**2.4 State Evolution through Segments:**  
Let $|\psi_0\rangle$ be the initial state. The state after the $j$-th segment is:  
$$|\psi(t_j)\rangle = U_j |\psi(t_{j-1})\rangle$$  
The final state at time $T$ is:  
$$|\psi(T)\rangle = U_{total} |\psi_0\rangle$$

### 3. Objective Function: State Fidelity

The goal is to maximize the fidelity between the final evolved state $|\psi(T)\rangle$ and a target state $|\psi_{target}\rangle$. For pure states:  
$$\mathcal{F} = |\langle \psi_{target} | \psi(T) \rangle|^2$$  
Which becomes:  
$$\mathcal{F} = |\langle \psi_{target} | U_{total} | \psi_0 \rangle|^2$$

### 4. Gradient Calculation (Adjoint Method)

Let $|\psi_j\rangle$ be the state at the end of segment $j$:  
$$|\psi_j\rangle = U_j U_{j-1} \cdots U_1 |\psi_0\rangle$$  
Then:  
$$\mathcal{F} = \langle \psi(T) | \psi_{target} \rangle \langle \psi_{target} | \psi(T) \rangle$$

Gradient with respect to control $c_{m,l}$ is:  
$$\frac{\partial \mathcal{F}}{\partial c_{m,l}} = 2 \text{Re} \left( \langle \psi_{target} | \psi(T) \rangle \langle \psi_{target} | \frac{\partial \psi(T)}{\partial c_{m,l}} \rangle \right)$$  
Using:  
$$\frac{\partial U_l}{\partial c_{m,l}} \approx -i H_m \Delta t U_l$$

Gradient becomes:  
$$\frac{\partial \mathcal{F}}{\partial c_{k,j}} = 2 \text{Re} \left[ \langle \psi_{target} | U_{N_s} \cdots U_{j+1} \left( \frac{\partial U_j}{\partial c_{k,j}} \right) U_{j-1} \cdots U_1 | \psi_0 \rangle \langle \psi(T) | \psi_{target} \rangle \right]$$

Define:  
- $|\chi_j\rangle = U_{j-1} \cdots U_1 |\psi_0\rangle$  
- $|\lambda_{j+1}\rangle = U_{j+1}^\dagger \cdots U_{N_s}^\dagger |\psi_{target}\rangle$  

Then:  
$$\tau_{k,j} = \langle \psi_{target} | U_{total} \rangle \langle \lambda_{j+1} | \left( \frac{\partial U_j}{\partial c_{k,j}} \right) | \chi_j \rangle$$  
And using:  
$$\frac{\partial U_j}{\partial c_{k,j}} = -i H_k \Delta t U_j$$

The gradient becomes:  
$$\frac{\partial \mathcal{F}}{\partial c_{k,j}} = -2 \Delta t \cdot \text{Im} \left[ \langle \psi(T) | \psi_{target} \rangle \langle \psi_{target} | U_{N_s} \cdots U_{j+1} H_k U_j \cdots U_1 | \psi_0 \rangle \right]$$

Or equivalently, with forward and backward propagated states:  
$$\frac{\partial \mathcal{F}}{\partial c_{k,j}} \propto -2 \Delta t \cdot \text{Im} \left( \langle \lambda(t_j) | H_k | \psi(t_{j-1}) \rangle \right)$$

Where:  
- $|\psi(t_{j-1})\rangle$ is the forward-propagated state  
- $|\lambda(t_j)\rangle$ is the adjoint backward-propagated state  
