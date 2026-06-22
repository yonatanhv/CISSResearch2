import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


# 2D Quantum Functions

def hamiltonianOperator2D(psi, dx, dy, V):
    laplacian_x = (np.roll(psi, -1, axis=1) - 2 * psi + np.roll(psi, 1, axis=0)) / dx ** 2
    laplacian_y = (np.roll(psi, -1, axis=0) - 2 * psi + np.roll(psi, 1, axis=1)) / dy ** 2
    laplacian_z = (np.roll(psi, -1, axis=0) - 2 * psi + np.roll(psi, 1, axis=2)) / dz ** 2
    laplacian = laplacian_x + laplacian_y + laplacian_z
    return -0.5 * laplacian + V * psi


def createWavePacket2D(X, Y, x0, y0, sigma_x, sigma_y, ky, m, dx, dy):
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    phi = np.arctan2(Y - y0, X - x0)

    gaussian_envelope = np.exp(-((X - x0) ** 2) / (2 * sigma_x ** 2)) * np.exp(-((Y - y0) ** 2) / (2 * sigma_y ** 2))
    angular_momentum_term = (r ** np.abs(m)) * np.exp(1j * m * phi)
    plane_wave = np.exp(1j * ky * Y)

    psi = angular_momentum_term * gaussian_envelope * plane_wave
    probability_sum = np.sum(np.abs(psi) ** 2) * dx * dy
    psi /= np.sqrt(probability_sum)
    return psi


def timeDerivative2D(t, psi, dx, dy, V):
    return -1j * hamiltonianOperator2D(psi, dx, dy, V)


def RK4_2D(t, psi, dx, dy, dt, V):
    k1 = timeDerivative2D(t, psi, dx, dy, V) * dt
    k2 = timeDerivative2D(t + 0.5 * dt, psi + 0.5 * k1, dx, dy, V) * dt
    k3 = timeDerivative2D(t + 0.5 * dt, psi + 0.5 * k2, dx, dy, V) * dt
    k4 = timeDerivative2D(t + dt, psi + k3, dx, dy, V) * dt

    psi_next = psi + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return psi_next


def LJ_wall(X, Y, y_wall_pos, epsilon, sigma, Vmax):
    V = np.zeros_like(X)
    mask = Y < y_wall_pos
    dy_wall = y_wall_pos - Y[mask]
    V[mask] = 4 * epsilon * ((sigma / dy_wall) ** 12 - (sigma / dy_wall) ** 6)
    V[~mask] = Vmax
    V = np.clip(V, -epsilon, Vmax)
    return V


def scatterer_potential(X, Y, y_c, A_au, b_au):
    r = np.sqrt(X ** 2 + (Y - y_c) ** 2)
    return A_au * np.exp(-r / b_au)



# Function 1: The Animation & Physics Engine

def animate_system(psi_init, dx, dy, dt, V_wall, X, Y):
    time_array, norm_array = [], []
    E_quant_array, E_class_array, E_tot_array = [], [], []

    fig, ax = plt.subplots(figsize=(8, 8))
    m_c, k_spring, y_eq = 0.2, 1.0, 1.0
    A_au, b_au = 200.0 / 27.2114, 0.3 / 0.52918
    X_sq = X ** 2

    state = {'psi': psi_init, 't': 0.0, 'y_c_curr': y_eq, 'y_c_prev': y_eq}
    initial_norm = np.sum(np.abs(psi_init) ** 2) * dx * dy

    img = ax.imshow(np.abs(state['psi']) ** 2, extent=[X.min(), X.max(), Y.min(), Y.max()],
                    origin='lower', cmap='magma', animated=True, vmax=np.max(np.abs(state['psi']) ** 2))

    ax.contour(X, Y, V_wall, levels=[1.0, 10.0, 50.0, 100.0], colors='white', alpha=0.3, linestyles='dashed')
    scatter_marker, = ax.plot([0], [y_eq], 'o', markerfacecolor='none', markeredgecolor='green', markersize=14,
                              markeredgewidth=2)

    ax.set_title("Wavepacket driving a Classical Oscillator")
    fig.colorbar(img, ax=ax, label="Probability Density |ψ|²")

    telemetry_text = ax.text(0.03, 0.97, '', transform=ax.transAxes, color='white', family='monospace',
                             verticalalignment='top')

    def update(frame):
        for _ in range(250):
            dy_c = Y - state['y_c_curr']
            r_safe = np.sqrt(X_sq + dy_c ** 2) + 1e-10

            V_ws = A_au * np.exp(-(r_safe - 1e-10) / b_au)
            V_current = V_wall + V_ws

            quantum_force = np.sum((np.abs(state['psi']) ** 2) * (-V_ws * (dy_c / (b_au * r_safe)))) * dx * dy
            acceleration = (-k_spring * (state['y_c_curr'] - y_eq) + quantum_force) / m_c

            y_c_next = 2 * state['y_c_curr'] - state['y_c_prev'] + acceleration * dt ** 2
            state['y_c_prev'], state['y_c_curr'] = state['y_c_curr'], y_c_next

            state['psi'] = RK4_2D(state['t'], state['psi'], dx, dy, dt, V_current)
            state['t'] += dt

        # Data Collection
        current_norm = np.sum(np.abs(state['psi']) ** 2) * dx * dy
        dy_c_final = Y - state['y_c_curr']
        V_current_final = V_wall + A_au * np.exp(-np.sqrt(X_sq + dy_c_final ** 2) / b_au)

        quant_E = np.real(
            np.sum(np.conj(state['psi']) * hamiltonianOperator2D(state['psi'], dx, dy, V_current_final)) * dx * dy)
        class_E = 0.5 * m_c * (((state['y_c_curr'] - state['y_c_prev']) / dt) ** 2) + 0.5 * k_spring * (
                    state['y_c_curr'] - y_eq) ** 2

        time_array.append(state['t'])
        norm_array.append(current_norm)
        E_quant_array.append(quant_E)
        E_class_array.append(class_E)
        E_tot_array.append(quant_E + class_E)

        img.set_data(np.abs(state['psi']) ** 2)
        scatter_marker.set_data([0], [state['y_c_curr']])
        telemetry_text.set_text(f"t   : {state['t']:.3f}\nΔN  : {current_norm - initial_norm:+.2e} ")
        return [img, scatter_marker, telemetry_text]

    ani = FuncAnimation(fig, update, frames=400, interval=30, blit=True)
    plt.show()

    return time_array, norm_array, E_quant_array, E_class_array, E_tot_array


def animate_dashboard(psi_init, dx, dy, dt, V_wall, X, Y):
    # Setup Figure and Grid Layout
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1])

    # Create the 3 Subplots
    ax_sim = fig.add_subplot(gs[:, 0])  # Left: Simulation
    ax_energy = fig.add_subplot(gs[0, 1])  # Top Right: Energy
    ax_norm = fig.add_subplot(gs[1, 1])  # Bottom Right: Normalization

    # --- 1. Setup Simulation Axis ---
    m_c, k_spring, y_eq = 0.2, 1.0, 1.0
    A_au, b_au = 200.0 / 27.2114, 0.3 / 0.52918
    X_sq = X ** 2

    state = {'psi': psi_init, 't': 0.0, 'y_c_curr': y_eq, 'y_c_prev': y_eq}

    # Calculate absolute baselines to extract Deltas
    initial_norm = np.sum(np.abs(psi_init) ** 2) * dx * dy
    H_psi_init = hamiltonianOperator2D(psi_init, dx, dy,
                                       V_wall + A_au * np.exp(-np.sqrt(X_sq + (Y - y_eq) ** 2) / b_au))
    initial_e_quant = np.real(np.sum(np.conj(psi_init) * H_psi_init) * dx * dy)
    initial_e_class = 0.0

    img = ax_sim.imshow(np.abs(state['psi']) ** 2, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower',
                        cmap='magma', animated=True, vmax=np.max(np.abs(state['psi']) ** 2))
    ax_sim.contour(X, Y, V_wall, levels=[1.0, 10.0, 50.0, 100.0], colors='white', alpha=0.3, linestyles='dashed')
    scatter_marker, = ax_sim.plot([0], [y_eq], 'o', markerfacecolor='none', markeredgecolor='green', markersize=14,
                                  markeredgewidth=2)
    ax_sim.set_title("Wavepacket driving a Classical Oscillator")
    fig.colorbar(img, ax=ax_sim, label="Probability Density |ψ|²", fraction=0.046, pad=0.04)
    telemetry_text = ax_sim.text(0.03, 0.97, '', transform=ax_sim.transAxes, color='white', family='monospace',
                                 verticalalignment='top')

    # --- 2. Setup Energy Axis ---
    line_eq, = ax_energy.plot([], [], label='Δ Quantum', color='blue')
    line_ec, = ax_energy.plot([], [], label='Δ Classical', color='green')
    line_etot, = ax_energy.plot([], [], label='Δ Total (Error)', color='red', linestyle='dashed')
    ax_energy.set_xlim(0, 2.2)  # Max simulation time
    ax_energy.set_ylim(-2.5, 2.5)
    ax_energy.set_title('Live Energy Exchange (ΔE)')
    ax_energy.grid(True, linestyle='--', alpha=0.6)
    ax_energy.legend(loc='upper left')

    # --- 3. Setup Normalization Axis ---
    line_norm, = ax_norm.plot([], [], label='Δ Norm', color='blue')
    ax_norm.set_xlim(0, 2.2)
    ax_norm.set_ylim(-5e-15, 5e-15)
    ax_norm.set_title('Live Normalization Drift (ΔN)')
    ax_norm.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_norm.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Data arrays for live plotting
    time_arr, norm_arr, eq_arr, ec_arr, etot_arr = [], [], [], [], []

    def update(frame):
        for _ in range(250):
            dy_c = Y - state['y_c_curr']
            r_safe = np.sqrt(X_sq + dy_c ** 2) + 1e-10

            V_ws = A_au * np.exp(-(r_safe - 1e-10) / b_au)
            V_current = V_wall + V_ws

            quantum_force = np.sum((np.abs(state['psi']) ** 2) * (-V_ws * (dy_c / (b_au * r_safe)))) * dx * dy
            acceleration = (-k_spring * (state['y_c_curr'] - y_eq) + quantum_force) / m_c

            y_c_next = 2 * state['y_c_curr'] - state['y_c_prev'] + acceleration * dt ** 2
            state['y_c_prev'], state['y_c_curr'] = state['y_c_curr'], y_c_next
            state['psi'] = RK4_2D(state['t'], state['psi'], dx, dy, dt, V_current)
            state['t'] += dt

        # Calculate current state
        dy_c_final = Y - state['y_c_curr']
        V_current_final = V_wall + A_au * np.exp(-np.sqrt(X_sq + dy_c_final ** 2) / b_au)

        current_norm = np.sum(np.abs(state['psi']) ** 2) * dx * dy
        quant_E = np.real(
            np.sum(np.conj(state['psi']) * hamiltonianOperator2D(state['psi'], dx, dy, V_current_final)) * dx * dy)
        class_E = 0.5 * m_c * (((state['y_c_curr'] - state['y_c_prev']) / dt) ** 2) + 0.5 * k_spring * (
                    state['y_c_curr'] - y_eq) ** 2

        # Append Deltas
        time_arr.append(state['t'])
        norm_arr.append(current_norm - initial_norm)
        eq_arr.append(quant_E - initial_e_quant)
        ec_arr.append(class_E - initial_e_class)
        etot_arr.append((quant_E + class_E) - (initial_e_quant + initial_e_class))

        # Update visual elements
        img.set_data(np.abs(state['psi']) ** 2)
        scatter_marker.set_data([0], [state['y_c_curr']])
        telemetry_text.set_text(f"t : {state['t']:.3f}")

        # Update line graphs
        line_eq.set_data(time_arr, eq_arr)
        line_ec.set_data(time_arr, ec_arr)
        line_etot.set_data(time_arr, etot_arr)
        line_norm.set_data(time_arr, norm_arr)

        return [img, scatter_marker, telemetry_text, line_eq, line_ec, line_etot, line_norm]

    ani = FuncAnimation(fig, update, frames=400, interval=30, blit=True)
    plt.show()


# Function 2: Normalization Delta Graph

def plot_normalization_delta(t_arr, n_arr):
    if not t_arr: return
    delta_n = np.array(n_arr) - n_arr[0]

    plt.figure(figsize=(8, 5))
    plt.plot(t_arr, delta_n, color='blue', linewidth=2, label='Δ Norm')
    plt.title('Wavepacket Normalization Drift (ΔN)')
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Δ ∫|ψ|² dxdy (Numerical Error)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()



# Function 3: Energy Delta Graph

def plot_energy_deltas(t_arr, e_quant, e_class, e_tot):
    if not t_arr: return
    delta_q = np.array(e_quant) - e_quant[0]
    delta_c = np.array(e_class) - e_class[0]
    delta_tot = np.array(e_tot) - e_tot[0]

    plt.figure(figsize=(8, 5))
    plt.plot(t_arr, delta_q, label='Δ Quantum Energy', color='blue', alpha=0.7)
    plt.plot(t_arr, delta_c, label='Δ Classical Energy', color='green', alpha=0.7)
    plt.plot(t_arr, delta_tot, label='Δ Total Energy (Error)', color='red', linestyle='dashed', linewidth=2)
    plt.title('System Energy Exchange & Conservation')
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Δ Energy (a.u.)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    Resolution = 10
    L_limit = 4.0
    numPoints = int(2 * L_limit * Resolution)

    x_arr = np.linspace(-L_limit, L_limit, numPoints)
    y_arr = np.linspace(-L_limit, L_limit, numPoints)
    z_arr = np.linspace(-L_limit, L_limit, numPoints)
    X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]
    dz = z_arr[1] - z_arr[0]

    dt = 0.00002
    psi = createWavePacket2D(X, Y, 0.0, -2.0, 0.75, 0.75, 5.0, -1, dx, dy)
    V_wall = LJ_wall(X, Y, 3.5, 2.0, 0.5, 150.0)

    # 1. Run the simulation
    #t_arr, n_arr, eq_arr, ec_arr, etot_arr = animate_system(psi, dx, dy, dt, V_wall, X, Y)

    # 2. Plot diagnostics (Executes after you close the animation window)
    #print("Generating pure delta plots...")
    #plot_normalization_delta(t_arr, n_arr)
    #plot_energy_deltas(t_arr, eq_arr, ec_arr, etot_arr)

    animate_dashboard(psi, dx, dy, dt, V_wall, X, Y)