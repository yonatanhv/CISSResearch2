import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from numba import njit
import pyvista as pv

# 2D Quantum Functions

@njit
def hamiltonianOperator3D(psi, dx, dy, dz, V):
    nx, ny, nz = psi.shape
    result = np.empty_like(psi)

    # Pre-calculate squared denominators for speed
    dx2 = dx ** 2
    dy2 = dy ** 2
    dz2 = dz ** 2

    for i in range(nx):
        # The modulo (%) enforces Periodic Boundary Conditions!
        i_next = (i + 1) % nx
        i_prev = (i - 1) % nx
        for j in range(ny):
            j_next = (j + 1) % ny
            j_prev = (j - 1) % ny
            for k in range(nz):
                k_next = (k + 1) % nz
                k_prev = (k - 1) % nz

                # Calculate Laplacian directly point-by-point
                lap_x = (psi[i_next, j, k] - 2.0 * psi[i, j, k] + psi[i_prev, j, k]) / dx2
                lap_y = (psi[i, j_next, k] - 2.0 * psi[i, j, k] + psi[i, j_prev, k]) / dy2
                lap_z = (psi[i, j, k_next] - 2.0 * psi[i, j, k] + psi[i, j, k_prev]) / dz2

                laplacian = lap_x + lap_y + lap_z

                # Apply Hamiltonian
                result[i, j, k] = -0.5 * laplacian + V[i, j, k] * psi[i, j, k]

    return result

def createWavePacket3D(X, Y, Z, x0, y0, z0, sigma_x, sigma_y, sigma_z, ky, m, dx, dy, dz):
    #distance X to Z
    r_xz = np.sqrt((X - x0) ** 2 + (Z - z0) ** 2)
    #angle
    phi = np.arctan2(Z - z0, X - x0)

    #gaussian formation
    gaussian_envelope = np.exp(-((X - x0) ** 2) / (2 * sigma_x ** 2)) \
                        * np.exp(-((Y - y0) ** 2) / (2 * sigma_y ** 2)) \
                        * np.exp(-((Z - z0) ** 2) / (2 * sigma_z ** 2))

    #(sqrt((X-X0)^2 + (Z-Z0)^2))^|m| * e^(i*m*phi)
    angular_momentum_term = (r_xz ** np.abs(m)) * np.exp(1j * m * phi)

    #kick momentum
    plane_wave = np.exp(1j * ky * Y)

    psi = angular_momentum_term * gaussian_envelope * plane_wave

    #normalization
    probability_sum = np.sum(np.abs(psi) ** 2) * dx * dy * dz
    psi /= np.sqrt(probability_sum)

    return psi

@njit
def timeDerivative3D(t, psi, dx, dy, dz, V):
    return -1j * hamiltonianOperator3D(psi, dx, dy, dz, V)

@njit
def RK4_3D(t, psi, dx, dy, dz, dt, V):
    #adding dz
    k1 = timeDerivative3D(t, psi, dx, dy, dz, V) * dt
    k2 = timeDerivative3D(t + 0.5 * dt, psi + 0.5 * k1, dx, dy, dz, V) * dt
    k3 = timeDerivative3D(t + 0.5 * dt, psi + 0.5 * k2, dx, dy, dz, V) * dt
    k4 = timeDerivative3D(t + dt, psi + k3, dx, dy, dz, V) * dt

    psi_next = psi + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return psi_next

def LJ_wall3D(Y, y_wall_pos, epsilon, sigma, Vmax):
    V = np.zeros_like(Y)
    mask = Y < y_wall_pos
    dy_wall = y_wall_pos - Y[mask]
    V[mask] = 4 * epsilon * ((sigma / dy_wall) ** 12 - (sigma / dy_wall) ** 6)
    V[~mask] = Vmax
    V = np.clip(V, -epsilon, Vmax)
    return V


def animate_dashboard3D(psi_init, dx, dy, dz, dt, V_wall, X, Y, Z):
    # Setup Figure and Grid Layout
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1])

    # הגדרת ציר תלת-ממדי אמיתי
    ax_sim = fig.add_subplot(gs[:, 0], projection='3d')
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_norm = fig.add_subplot(gs[1, 1])

    m_c, k_spring, y_eq = 0.2, 0.1, 1.0
    A_au, b_au = 200.0 / 27.2114, 0.3 / 0.52918
    X_sq = X ** 2
    Z_sq = Z ** 2  # חובה ב-3D

    state = {'psi': psi_init, 't': 0.0, 'y_c_curr': y_eq, 'y_c_prev': y_eq}

    initial_norm = np.sum(np.abs(psi_init) ** 2) * dx * dy * dz
    H_psi_init = hamiltonianOperator3D(psi_init, dx, dy, dz,
                                       V_wall + A_au * np.exp(-np.sqrt(X_sq + (Y - y_eq) ** 2 + Z_sq) / b_au))
    initial_e_quant = np.real(np.sum(np.conj(psi_init) * H_psi_init) * dx * dy * dz)
    initial_e_class = 0.0

    # --- הכנת הענן התלת-ממדי (Volumetric Heatmap) ---
    flat_X = X.flatten()
    flat_Y = Y.flatten()
    flat_Z = Z.flatten()

    density_3d = np.abs(state['psi']) ** 2
    density_flat = density_3d.flatten()

    # טריק: מציירים רק פיקסלים שההסתברות בהם גדולה מ-2%.
    # זה הופך את זה ל"כדור" ענן מוגדר ולא סתם קוביה מטושטשת!
    threshold = np.max(density_flat) * 0.02
    mask = density_flat > threshold

    # ציור ענן ההסתברות הראשוני
    cloud = [ax_sim.scatter(flat_X[mask], flat_Y[mask], flat_Z[mask],
                            c=density_flat[mask], cmap='magma',
                            alpha=0.3, s=15, edgecolors='none')]

    # ציור החלקיק הקלאסי באמצע המרחב התלת-ממדי
    scatter_marker, = ax_sim.plot([0], [y_eq], [0], 'o', markerfacecolor='none',
                                  markeredgecolor='green', markersize=14, markeredgewidth=2)

    # קיבוע גבולות וזווית המצלמה כדי שהמסך לא יקפוץ
    ax_sim.set_xlim(X.min(), X.max())
    ax_sim.set_ylim(Y.min(), Y.max())
    ax_sim.set_zlim(Z.min(), Z.max())
    ax_sim.view_init(elev=20, azim=-60)
    ax_sim.set_title("3D Volumetric Wavepacket vs Classical Oscillator")

    telemetry_text = ax_sim.text2D(0.03, 0.97, '', transform=ax_sim.transAxes, color='black', family='monospace',
                                   verticalalignment='top')

    # --- Setup Energy & Norm Axes ---
    line_eq, = ax_energy.plot([], [], label='Δ Quantum', color='blue')
    line_ec, = ax_energy.plot([], [], label='Δ Classical', color='green')
    line_etot, = ax_energy.plot([], [], label='Δ Total (Error)', color='red', linestyle='dashed')
    ax_energy.set_xlim(0, 2.2)
    ax_energy.set_ylim(-2.5, 2.5)
    ax_energy.set_title('Live Energy Exchange (ΔE)')
    ax_energy.grid(True, linestyle='--', alpha=0.6)
    ax_energy.legend(loc='upper left')

    line_norm, = ax_norm.plot([], [], label='Δ Norm', color='blue')
    ax_norm.set_xlim(0, 2.2)
    ax_norm.set_ylim(-5e-15, 5e-15)
    ax_norm.set_title('Live Normalization Drift (ΔN)')
    ax_norm.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_norm.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    time_arr, norm_arr, eq_arr, ec_arr, etot_arr = [], [], [], [], []

    def update(frame):
        for _ in range(40):
            dy_c = Y - state['y_c_curr']
            # תוקן: המרחק עכשיו כולל את ציר ה-Z!
            r_safe = np.sqrt(X_sq + dy_c ** 2 + Z_sq) + 1e-10

            V_ws = A_au * np.exp(-(r_safe - 1e-10) / b_au)
            V_current = V_wall + V_ws

            # תוקן: נוסף dz
            quantum_force = np.sum((np.abs(state['psi']) ** 2) * (-V_ws * (dy_c / (b_au * r_safe)))) * dx * dy * dz
            acceleration = (-k_spring * (state['y_c_curr'] - y_eq) + quantum_force) / m_c

            y_c_next = 2 * state['y_c_curr'] - state['y_c_prev'] + acceleration * dt ** 2
            state['y_c_prev'], state['y_c_curr'] = state['y_c_curr'], y_c_next

            # תוקן: שימוש בפונקציית RK4 התלת-ממדית
            state['psi'] = RK4_3D(state['t'], state['psi'], dx, dy, dz, dt, V_current)
            state['t'] += dt

        # תוקן: הוספת Z_sq
        dy_c_final = Y - state['y_c_curr']
        V_current_final = V_wall + A_au * np.exp(-np.sqrt(X_sq + dy_c_final ** 2 + Z_sq) / b_au)

        # תוקן: שימוש בפונקציות ה-3D והוספת dz
        current_norm = np.sum(np.abs(state['psi']) ** 2) * dx * dy * dz
        quant_E = np.real(
            np.sum(np.conj(state['psi']) * hamiltonianOperator3D(state['psi'], dx, dy, dz,
                                                                 V_current_final)) * dx * dy * dz)
        class_E = 0.5 * m_c * (((state['y_c_curr'] - state['y_c_prev']) / dt) ** 2) + 0.5 * k_spring * (
                    state['y_c_curr'] - y_eq) ** 2

        time_arr.append(state['t'])
        norm_arr.append(current_norm - initial_norm)
        eq_arr.append(quant_E - initial_e_quant)
        ec_arr.append(class_E - initial_e_class)
        etot_arr.append((quant_E + class_E) - (initial_e_quant + initial_e_class))

        # --- מחיקה וציור מחדש של כדור החום (Heatmap) ---
        cloud[0].remove()

        density_3d_curr = np.abs(state['psi']) ** 2
        density_flat_curr = density_3d_curr.flatten()
        threshold_curr = np.max(density_flat_curr) * 0.02
        mask_curr = density_flat_curr > threshold_curr

        cloud[0] = ax_sim.scatter(flat_X[mask_curr], flat_Y[mask_curr], flat_Z[mask_curr],
                                  c=density_flat_curr[mask_curr], cmap='magma',
                                  alpha=0.3, s=15, edgecolors='none')

        # עדכון מיקום החלקיק הקלאסי
        scatter_marker.set_data_3d([0], [state['y_c_curr']], [0])
        telemetry_text.set_text(f"t : {state['t']:.3f}")

        line_eq.set_data(time_arr, eq_arr)
        line_ec.set_data(time_arr, ec_arr)
        line_etot.set_data(time_arr, etot_arr)
        line_norm.set_data(time_arr, norm_arr)

        return [scatter_marker, telemetry_text, line_eq, line_ec, line_etot, line_norm]

    # חובה ש-blit יהיה False כשמציירים ב-3D
    ani = FuncAnimation(fig, update, frames=400, interval=30, blit=False)
    plt.show()


def animate_pyvista3D(psi_init, dx, dy, dz, dt, V_wall, X, Y, Z):
    m_c, k_spring, y_eq = 0.2, 1.0, 1.0
    A_au, b_au = 200.0 / 27.2114, 0.3 / 0.52918
    X_sq = X ** 2
    Z_sq = Z ** 2

    state = {'psi': psi_init, 't': 0.0, 'y_c_curr': y_eq, 'y_c_prev': y_eq}

    plotter = pv.Plotter()
    plotter.set_background('black')

    particle = pv.Sphere(radius=0.15, center=(0, y_eq, 0))
    plotter.add_mesh(particle, color='green', smooth_shading=True, specular=0.5)

    grid = pv.ImageData()
    grid.dimensions = np.array(state['psi'].shape)
    grid.spacing = (dx, dy, dz)
    grid.origin = (X.min(), Y.min(), Z.min())

    density = np.abs(state['psi']) ** 2
    grid.point_data["Density"] = density.flatten(order="F")

    # Volume Rendering
    plotter.add_volume(grid, scalars="Density", cmap="magma", opacity="linear", show_scalar_bar=False)

    plotter.add_bounding_box(color='white', line_width=1.0)
    plotter.add_axes()
    plotter.camera_position = 'iso'

    plotter.add_text("Time: 0.000", position="upper_left", font_size=14, name="time_label")

    # הפונקציה שרצה כל פריים
    def update_physics(step):
        # שומרים את מיקום החלקיק לפני שהפיזיקה מתחילה לרוץ
        y_start_frame = state['y_c_curr']

        for _ in range(40):
            dy_c = Y - state['y_c_curr']
            r_safe = np.sqrt(X_sq + dy_c ** 2 + Z_sq) + 1e-10

            V_ws = A_au * np.exp(-(r_safe - 1e-10) / b_au)
            V_current = V_wall + V_ws

            quantum_force = np.sum((np.abs(state['psi']) ** 2) * (-V_ws * (dy_c / (b_au * r_safe)))) * dx * dy * dz
            acceleration = (-k_spring * (state['y_c_curr'] - y_eq) + quantum_force) / m_c

            y_c_next = 2 * state['y_c_curr'] - state['y_c_prev'] + acceleration * dt ** 2
            state['y_c_prev'], state['y_c_curr'] = state['y_c_curr'], y_c_next
            state['psi'] = RK4_3D(state['t'], state['psi'], dx, dy, dz, dt, V_current)
            state['t'] += dt

        # 1. עדכון הענן הקוונטי אל תוך הזיכרון הקיים (In-place) כדי שה-GPU יראה את זה!
        new_density = np.abs(state['psi']) ** 2
        grid["Density"][:] = new_density.flatten(order="F")

        # 2. עדכון תנועת החלקיק על בסיס כל ה-40 צעדים!
        dy_frame = state['y_c_curr'] - y_start_frame
        particle.translate([0, dy_frame, 0], inplace=True)

        # 3. עדכון השעון
        plotter.add_text(f"Time: {state['t']:.3f}", position="upper_left", font_size=14, name="time_label")

        # 4. הכרחה של כרטיס המסך לצייר מחדש
        plotter.render()

    # טיימר שמריץ את update_physics
    plotter.add_timer_event(max_steps=10000, duration=30, callback=update_physics)

    print("Starting GPU Render Loop...")
    plotter.show()


def hamiltonianOperator2D(psi, dx, dy, V):
    laplacian_x = (np.roll(psi, -1, axis=1) - 2 * psi + np.roll(psi, 1, axis=0)) / dx ** 2
    laplacian_y = (np.roll(psi, -1, axis=0) - 2 * psi + np.roll(psi, 1, axis=1)) / dy ** 2
    laplacian = laplacian_x + laplacian_y
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

    psi_projection = np.sum(np.abs(state['psi']) ** 2, axis=2) * dz

    # Draw the projection
    img = ax_sim.imshow(psi_projection.T,
                        extent=[X[:, 0, 0].min(), X[:, 0, 0].max(), Y[0, :, 0].min(), Y[0, :, 0].max()],
                        origin='lower', cmap='magma', animated=True,
                        vmax=np.max(psi_projection) * 0.8, interpolation='bilinear')

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
        for _ in range(40):
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
        psi_projection_current = np.sum(np.abs(state['psi']) ** 2, axis=2) * dz
        img.set_data(psi_projection_current.T)
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


def save_cube_file(filename, density, dx, dy, dz, x_start, y_start, z_start, y_c_curr):
    nx, ny, nz = density.shape

    with open(filename, 'w') as f:
        # Header comments
        f.write("Quantum Wavepacket Simulation\n")
        f.write("Volumetric Probability Density\n")

        # Number of atoms (1 for the classical particle) and origin coordinates
        # Format: natoms x_origin y_origin z_origin
        f.write(f"    1 {x_start:.6f} {y_start:.6f} {z_start:.6f}\n")

        # Number of grid points and step sizes along X, Y, Z
        f.write(f"{nx:>5} {dx:.6f} 0.000000 0.000000\n")
        f.write(f"{ny:>5} 0.000000 {dy:.6f} 0.000000\n")
        f.write(f"{nz:>5} 0.000000 0.000000 {dz:.6f}\n")

        # Atom data: atomic_number, charge, x, y, z
        # We put a "dummy" atom to represent your classical particle's position
        f.write(f"    1 1.000000 0.000000 {y_c_curr:.6f} 0.000000\n")

        # Write the 3D density data
        count = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    f.write(f"{density[i, j, k]:.5E} ")
                    count += 1
                    if count % 6 == 0:
                        f.write("\n")
                if count % 6 != 0:
                    f.write("\n")
                    count = 0

def run_and_export_simulation(psi_init, dx, dy, dz, dt, V_wall, X, Y, Z, frames=100, steps_per_frame=40):
    m_c, k_spring, y_eq = 0.2, 1.0, 1.0
    A_au, b_au = 200.0 / 27.2114, 0.3 / 0.52918
    X_sq = X ** 2
    Z_sq = Z ** 2

    state = {'psi': psi_init, 't': 0.0, 'y_c_curr': y_eq, 'y_c_prev': y_eq}

    x_start, y_start, z_start = X.min(), Y.min(), Z.min()

    print("Starting simulation and exporting .cube files...")

    for frame in range(frames):
        # Calculate Density
        density = np.abs(state['psi']) ** 2

        # Generate filename (e.g., frame_000.cube, frame_001.cube)
        filename = f"wavepacket_frame_{frame:03d}.cube"

        # Export to file
        save_cube_file(filename, density, dx, dy, dz, x_start, y_start, z_start, state['y_c_curr'])
        print(f"Exported {filename} (t = {state['t']:.4f})")

        # Run Physics for the next frame
        for _ in range(steps_per_frame):
            dy_c = Y - state['y_c_curr']
            r_safe = np.sqrt(X_sq + dy_c ** 2 + Z_sq) + 1e-10

            V_ws = A_au * np.exp(-(r_safe - 1e-10) / b_au)
            V_current = V_wall + V_ws

            quantum_force = np.sum((np.abs(state['psi']) ** 2) * (-V_ws * (dy_c / (b_au * r_safe)))) * dx * dy * dz
            acceleration = (-k_spring * (state['y_c_curr'] - y_eq) + quantum_force) / m_c

            y_c_next = 2 * state['y_c_curr'] - state['y_c_prev'] + acceleration * dt ** 2
            state['y_c_prev'], state['y_c_curr'] = state['y_c_curr'], y_c_next
            state['psi'] = RK4_3D(state['t'], state['psi'], dx, dy, dz, dt, V_current)
            state['t'] += dt

    print("Export complete! You can now load these files into VESTA or VMD.")


# --- Spring visualization helpers ---------------------------------------
# Kept cheap on purpose: a small helical point-template is built ONCE, and
# every frame we just re-stretch/rotate/translate that same small array of
# points with plain numpy (no re-triangulation math beyond pv.Spline+tube,
# which stays fast as long as n_points is modest). This is the piece that
# will need to become instanced/batched once step 3 puts many springs on
# the wall at once — flagged there when we get to it.

def make_spring_template(n_coils=6, n_points=60, coil_radius=0.06):
    """Unit-length helical spring template. Local frame: axis = +Y, spans y in [0, 1]."""
    t = np.linspace(0.0, 1.0, n_points)
    theta = t * n_coils * 2.0 * np.pi
    x = coil_radius * np.cos(theta)
    z = coil_radius * np.sin(theta)
    y = t
    return np.column_stack([x, y, z])


def spring_mesh_along_axis(template_pts, anchor, axis, length,
                           tube_radius=0.02, min_length=0.02):
    """Build the spring from `anchor` along a FIXED `axis` direction, with a
    clamped `length`.

    Using a fixed axis (instead of deriving the direction from anchor->tip) is
    what keeps the spring from flipping to the other side when the electron
    overshoots past the anchor: the spring can only compress toward `min_length`,
    never invert.
    """
    anchor = np.asarray(anchor, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    length = max(float(length), min_length)  # compress toward ~0, never flip

    world_y = np.array([0.0, 1.0, 0.0])
    dot = float(np.dot(axis, world_y))
    if abs(dot) > 0.999:
        rot = np.eye(3) if dot > 0 else np.diag([1.0, -1.0, 1.0])
    else:
        v = np.cross(world_y, axis)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + dot))

    stretched = template_pts * np.array([1.0, length, 1.0])  # stretch along local Y only
    world_pts = stretched @ rot.T + anchor

    spline = pv.Spline(world_pts, template_pts.shape[0])
    return spline.tube(radius=tube_radius)


def render_mp4_simulation(psi_init, dx, dy, dz, dt, V_wall, X, Y, Z, frames=1250, steps_per_frame=40,
                          y_wall_pos=2.5,
                          nucleus_offset=0.05, r0=None,
                          nucleus_pos=None, A_nuc=None):
    """
    GEOMETRY (one site = anchored nucleus + electron on a spring):

        ... wavepacket ...   e-  ~~~~~~~spring~~~~~~~  (+)      | wall |
                             |<------- r0 = 1 A ------>|
                          y_eq                       y_nuc = y_wall_pos + nucleus_offset

    nucleus_offset : how far PAST the wall face the nucleus sits. Default 0.05 Bohr, i.e. the
                  nucleus hugs the wall surface (a monolayer sitting ON the wall, as on the
                  whiteboard where the nuclei lie on the backbone line). The electron dangles
                  out in front of it on the spring, on the wavepacket side.
    r0          : nucleus<->electron rest separation. Default 1 Angstrom (whiteboard value),
                  which SETS the electron's equilibrium: y_eq = y_nuc - r0.

    WHY THE SPRING *IS* THE BOND (and there is no separate nucleus<->electron force):
                  The spring is not decoration sitting alongside an electrostatic attraction --
                  it *is* the harmonic model of that attraction, with r0 its rest length. Adding
                  a bare attractive exponential on top would double-count the binding, and with
                  these parameters it does not even have a solution: the attraction prefactor is
                  A/b = 12.96 Ha/Bohr against a spring stiffness of only k = 0.2, so the net
                  force is positive everywhere (no equilibrium) and dF/dy = +0.94 at y = 1.0
                  (any balance point would be unstable). The electron would collapse onto the
                  nucleus. If you want a genuine two-term bond, the attraction needs a repulsive
                  core (Morse / Lennard-Jones with its minimum at r0), not a bare exponential.

    So the three-body coupling is: nucleus -> electron via the spring (this function),
    electron -> wavepacket via +A_au*exp(-r/b) (repulsive, in the loop),
    nucleus  -> wavepacket via -A_nuc*exp(-r/b) (attractive, folded into V_static).

    nucleus_pos : optional explicit (x, y, z) override for the ANCHORED +|e| nucleus.
                  The nucleus never moves, so it has no equation of motion.
    A_nuc       : strength of the nucleus <-> quantum-electron ATTRACTION. Default = A_au,
                  i.e. equal and opposite to the classical electron's repulsion.

                  CAVEAT: equal-and-opposite does NOT make the site neutral at range here.
                  The two charges sit 1.5 Bohr apart while the coupling decay length is only
                  b_au = 0.567 Bohr, so the exponentials barely overlap and they cancel
                  exactly only on the perpendicular bisector plane (y = 1.75). Anywhere else
                  the nearer charge dominates -- e.g. at the nucleus the sum is -6.83 Ha, at
                  the electron +6.83 Ha. The site acts as two separate localized wells, not a
                  neutral dipole. To get real cancellation you would need the separation to be
                  <= b_au, or a longer-ranged (Coulomb-like) form instead of an exponential.
    """
    m_c, k_spring = 0.2, 0.2
    A_au, b_au = 200.0 / 27.2114, 0.3 / 0.52918
    BOHR_PER_ANGSTROM = 1.0 / 0.52917721
    X_sq = X ** 2
    Z_sq = Z ** 2

    if r0 is None:
        r0 = 1.0 * BOHR_PER_ANGSTROM          # whiteboard: r0 = 1 Angstrom = 1.890 Bohr
    if nucleus_pos is None:
        # הגרעין יושב מעט *אחרי* הקיר (בתוך חומר הקיר), לא לפניו
        nucleus_pos = (0.0, y_wall_pos + nucleus_offset, 0.0)
    if A_nuc is None:
        A_nuc = A_au

    y_nuc = nucleus_pos[1]
    # שיווי המשקל של האלקטרון נקבע ע"י הגרעין: בדיוק r0 לפניו
    y_eq = y_nuc - r0
    print(f"  nucleus anchored at y = {y_nuc:.3f} (wall at {y_wall_pos}, offset {nucleus_offset})")
    print(f"  electron equilibrium  y_eq = y_nuc - r0 = {y_eq:.3f}  (r0 = {r0:.3f} Bohr = "
          f"{r0 * 0.52917721:.2f} Angstrom)")

    state = {'psi': psi_init, 't': 0.0, 'y_c_curr': y_eq, 'y_c_prev': y_eq}

    # --- הגרעין החיובי: מקובע במקום, ולכן הפוטנציאל שלו סטטי לחלוטין ---
    # מחשבים אותו פעם אחת בלבד ומאחדים אותו לתוך פוטנציאל הקיר. כך התוספת
    # הזו *לא* מוסיפה שום עלות חישובית בלולאה הפנימית.
    # הסימן שלילי: הגרעין (+|e|) מושך את האלקטרון הקוונטי (-|e|).
    xn, yn, zn = nucleus_pos
    r_nuc = np.sqrt((X - xn) ** 2 + (Y - yn) ** 2 + (Z - zn) ** 2) + 1e-10
    V_nuc = -A_nuc * np.exp(-r_nuc / b_au)
    V_static = V_wall + V_nuc

    print("Setting up Off-Screen Renderer...")

    # 1. מפעילים את PyVista במצב שקט (ללא חלון) וברזולוציה גבוהה
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.set_background('black')

    # 2. הוספת החלקיק הקלאסי (האלקטרון על הקפיץ)
    particle = pv.Sphere(radius=0.15, center=(0, y_eq, 0))
    plotter.add_mesh(particle, color='green', smooth_shading=True, specular=0.5)

    # 2b. הקפיץ (ויזואלי בלבד בשלב הזה — לא משפיע על הפיזיקה)
    # העוגן מקובע בקיר עצמו, והקפיץ מתוח ממנו כלפי מטה (-Y) אל האלקטרון.
    # הכיוון קבוע, ולכן הקפיץ יכול רק להתקצר לכיוון אפס — הוא לעולם לא "יתהפך"
    # לצד השני של האלקטרון גם אם האלקטרון יעבור את נקודת העיגון.
    # הקפיץ מעוגן *בגרעין* (לא בקיר) — הוא מייצג את הקשר גרעין-אלקטרון
    spring_anchor = nucleus_pos
    spring_axis = (0.0, -1.0, 0.0)
    spring_template = make_spring_template(n_coils=6, n_points=60, coil_radius=0.06)
    spring_mesh = spring_mesh_along_axis(spring_template, spring_anchor, spring_axis,
                                         y_nuc - y_eq)
    plotter.add_mesh(spring_mesh, color='silver', name='spring', specular=0.6)

    # 2c. הגרעין החיובי (+|e|) — מקובע, ולכן מצויר פעם אחת ולא מתעדכן בלולאה.
    # הוא *אמור* לא לזוז: זה כל הרעיון של "anchored".
    nucleus = pv.Sphere(radius=0.16, center=nucleus_pos)
    plotter.add_mesh(nucleus, color='red', smooth_shading=True, specular=0.5)

    # 2d. הקיר עצמו — עד עכשיו הוא היה *בלתי נראה* (רק פוטנציאל ב-y=y_wall_pos),
    # ולכן היה בלתי אפשרי לשפוט אם הגרעין לפני או אחרי הקיר. עכשיו הוא מצויר.
    wall_plane = pv.Plane(center=(0.0, y_wall_pos, 0.0), direction=(0.0, 1.0, 0.0),
                          i_size=float(X.max() - X.min()), j_size=float(Z.max() - Z.min()))
    # opacity הוא הכפתור לכוונון: העלה אם הקיר חיוור מדי, הורד אם הוא מסתיר את הגל
    plotter.add_mesh(wall_plane, color='deepskyblue', opacity=0.35, show_edges=True,
                     edge_color='deepskyblue', name='wall')

    # 3. הכנת הרשת התלת-ממדית לחבילת הגל
    grid = pv.ImageData()
    grid.dimensions = np.array(state['psi'].shape)
    grid.spacing = (dx, dy, dz)
    grid.origin = (X.min(), Y.min(), Z.min())

    density = np.abs(state['psi']) ** 2
    grid.point_data["Density"] = density.flatten(order="F")

    # רינדור ענן (Volume) עם צבע וסף שקיפות כדי שייראה כמו מפת חום
    plotter.add_volume(grid, scalars="Density", cmap="magma", opacity="linear", show_scalar_bar=False)

    plotter.add_bounding_box(color='white', line_width=1.0)
    plotter.add_axes()
    plotter.camera_position = 'yz'
    #plotter.camera.azimuth += 15  # Rotates the camera sideways (try numbers between 45 and 90)
    plotter.camera.elevation += 15  # Drops the camera angle down for a more straight-on view

    # --- Baselines for Δ tracking (mirrors animate_dashboard3D) ---
    initial_norm = np.sum(np.abs(psi_init) ** 2) * dx * dy * dz
    V_initial = V_static + A_au * np.exp(-np.sqrt(X_sq + (Y - y_eq) ** 2 + Z_sq) / b_au)
    H_psi_init = hamiltonianOperator3D(psi_init, dx, dy, dz, V_initial)
    initial_e_quant = np.real(np.sum(np.conj(psi_init) * H_psi_init) * dx * dy * dz)
    initial_e_class = 0.0

    # מונה זמן קטן בפינה השמאלית העליונה
    # color='white' חובה! ברירת המחדל של PyVista לצבע טקסט היא שחור, ועל רקע שחור
    # הטקסט פשוט בלתי נראה (זו הסיבה ש"Time:" מעולם לא הופיע בסרטונים).
    # הכל בפינה השמאלית-עליונה: שם המסך ריק, ולכן הטקסט לא מתנגש בסימולציה או בגרפים.
    hud_seed = (f"Time : 0.000\n"
                f"dN   : {0.0:.2e}\n"
                f"dEq  : {0.0:.2e}\n"
                f"dEc  : {0.0:.2e}\n"
                f"dEt  : {0.0:.2e}")
    text_actor = plotter.add_text(hud_seed, position="upper_left", font_size=14,
                                  color='white', name="hud")

    # --- גרפים חיים של ΔN ו-ΔE, "אפויים" ישירות לתוך הפריים (פינה ימנית עליונה) ---
    time_arr = [0.0]
    norm_arr = [0.0]
    eq_arr = [0.0]
    ec_arr = [0.0]
    etot_arr = [0.0]

    def order_of_magnitude(value):
        """Returns the exponent x such that |value| ~ 10^x (0 for zero/garbage input)."""
        value = abs(value)
        if value == 0 or not np.isfinite(value):
            return 0
        return int(np.floor(np.log10(value)))

    # רקע בהיר לגרפים! מספרי הצירים ב-PyVista נצבעים שחור כברירת מחדל, ולכן על
    # רקע שחור הם היו בלתי נראים לחלוטין. רקע בהיר פותר את זה בלי להסתמך על API
    # לא ודאי לשינוי צבע הטקסט של הצירים.
    panel_bg = (235, 235, 235, 235)

    chart_norm = pv.Chart2D(size=(0.32, 0.22), loc=(0.65, 0.75), x_label="t", y_label="dN (~1e+0)")
    chart_norm.background_color = panel_bg
    chart_norm.border_color = 'white'
    line_norm = chart_norm.line(time_arr, norm_arr, color='teal', width=2.0, label='dNorm')

    chart_energy = pv.Chart2D(size=(0.32, 0.22), loc=(0.65, 0.50), x_label="t", y_label="dE (~1e+0)")
    chart_energy.background_color = panel_bg
    chart_energy.border_color = 'white'
    line_eq = chart_energy.line(time_arr, eq_arr, color='blue', width=2.0, label='dE quantum')
    line_ec = chart_energy.line(time_arr, ec_arr, color='green', width=2.0, label='dE classical')
    # NOTE: not 100% sure `style='--'` is accepted here — if this line errors on your
    # machine, just drop the style kwarg (or set style='-') and rerun.
    line_etot = chart_energy.line(time_arr, etot_arr, color='red', width=2.0, style='--', label='dE total')
    chart_energy.legend_visible = True

    plotter.add_chart(chart_norm)
    plotter.add_chart(chart_energy)

    # (המספרים המדויקים עברו ל-HUD בפינה השמאלית-עליונה, יחד עם השעון —
    #  קודם הם היו ב-(0.65, 0.44) והתנגשו בגרעין ובקפיץ באמצע הסצנה.)

    # 4. פתיחת קובץ הוידאו
    video_filename = "wavepacket_collision.mp4"
    print(f"Opening {video_filename} for writing...")
    plotter.open_movie(video_filename, framerate=30)

    # 5. לולאת הרינדור (הפיזיקה והכתיבה לוידאו)
    for frame in range(frames):
        y_start_frame = state['y_c_curr']

        # הרצת הפיזיקה
        for _ in range(steps_per_frame):
            dy_c = Y - state['y_c_curr']
            r_safe = np.sqrt(X_sq + dy_c ** 2 + Z_sq) + 1e-10

            V_ws = A_au * np.exp(-(r_safe - 1e-10) / b_au)
            # V_static כבר כולל את הקיר ואת הגרעין המקובע (מחושב פעם אחת מחוץ ללולאה)
            V_current = V_static + V_ws

            quantum_force = np.sum((np.abs(state['psi']) ** 2) * (-V_ws * (dy_c / (b_au * r_safe)))) * dx * dy * dz
            acceleration = (-k_spring * (state['y_c_curr'] - y_eq) + quantum_force) / m_c

            y_c_next = 2 * state['y_c_curr'] - state['y_c_prev'] + acceleration * dt ** 2
            state['y_c_prev'], state['y_c_curr'] = state['y_c_curr'], y_c_next
            state['psi'] = RK4_3D(state['t'], state['psi'], dx, dy, dz, dt, V_current)
            state['t'] += dt

        # עדכון הגרפיקה בזיכרון
        new_density = np.abs(state['psi']) ** 2
        grid["Density"][:] = new_density.flatten(order="F")

        dy_frame = state['y_c_curr'] - y_start_frame
        particle.translate([0, dy_frame, 0], inplace=True)

        # עדכון הקפיץ: העוגן והכיוון קבועים, רק האורך משתנה (מתקצר לכיוון אפס)
        spring_mesh = spring_mesh_along_axis(spring_template, spring_anchor, spring_axis,
                                             y_nuc - state['y_c_curr'])
        plotter.add_mesh(spring_mesh, color='silver', name='spring', specular=0.6)

        # --- חישוב הדלתות (אנרגיה קוונטית/מכנית ונורמליזציה) לפריים הנוכחי ---
        dy_c_final = Y - state['y_c_curr']
        V_current_final = V_static + A_au * np.exp(-np.sqrt(X_sq + dy_c_final ** 2 + Z_sq) / b_au)

        current_norm = np.sum(np.abs(state['psi']) ** 2) * dx * dy * dz
        quant_E = np.real(
            np.sum(np.conj(state['psi']) * hamiltonianOperator3D(state['psi'], dx, dy, dz,
                                                                 V_current_final)) * dx * dy * dz)
        class_E = 0.5 * m_c * (((state['y_c_curr'] - state['y_c_prev']) / dt) ** 2) + 0.5 * k_spring * (
                    state['y_c_curr'] - y_eq) ** 2

        delta_norm = current_norm - initial_norm
        delta_e_quant = quant_E - initial_e_quant
        delta_e_class = class_E - initial_e_class
        delta_e_total = (quant_E + class_E) - (initial_e_quant + initial_e_class)

        time_arr.append(state['t'])
        norm_arr.append(delta_norm)
        eq_arr.append(delta_e_quant)
        ec_arr.append(delta_e_class)
        etot_arr.append(delta_e_total)

        # עדכון קווי הגרף בתוך הפריים
        line_norm.update(time_arr, norm_arr)
        line_eq.update(time_arr, eq_arr)
        line_ec.update(time_arr, ec_arr)
        line_etot.update(time_arr, etot_arr)

        # עדכון תווית סדר הגודל (order of magnitude) על כל גרף, לפי הערך המקסימלי שנצפה עד כה
        order_n = order_of_magnitude(np.max(np.abs(norm_arr)))
        chart_norm.y_label = f"dN (~1e{order_n:+d})"

        order_e = order_of_magnitude(max(np.max(np.abs(eq_arr)), np.max(np.abs(ec_arr)), np.max(np.abs(etot_arr))))
        chart_energy.y_label = f"dE (~1e{order_e:+d})"

        # HUD אחד בפינה השמאלית-עליונה: שעון + הערכים המדויקים, מחוץ לסצנה
        hud_text = (
            f"Time : {state['t']:.3f}\n"
            f"dN   : {delta_norm:.2e}\n"
            f"dEq  : {delta_e_quant:.2e}\n"
            f"dEc  : {delta_e_class:.2e}\n"
            f"dEt  : {delta_e_total:.2e}"
        )
        plotter.add_text(hud_text, position="upper_left", font_size=14,
                         color='white', name="hud")

        # שמירת הפריים לתוך קובץ ה-MP4
        plotter.write_frame()
        print(f"Rendered frame {frame + 1}/{frames} (t = {state['t']:.3f}, "
              f"dE_q={delta_e_quant:+.3e}, dE_c={delta_e_class:+.3e}, dN={delta_norm:+.3e})")

    # סגירה ושמירה של הקובץ
    plotter.close()
    print(f"\nSuccess! Video saved as: {video_filename}")


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


if __name__ == "__main__":
    # Shrink the physical box, but double the resolution!
    Resolution = 8
    L_limit = 3.0

    # Adding +1 ensures the grid lands exactly on 0.0 (restores perfect symmetry)
    numPoints = int(2 * L_limit * Resolution) + 1

    x_arr = np.linspace(-L_limit, L_limit, numPoints)
    y_arr = np.linspace(-L_limit, L_limit, numPoints)
    z_arr = np.linspace(-L_limit, L_limit, numPoints)

    X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]
    dz = z_arr[1] - z_arr[0]

    dt = 0.00002
    # Shifted start position slightly up since the box is smaller
    psi = createWavePacket3D(X, Y, Z, 0.0, -1, 0.0, 0.75, 0.75, 0.75, 8.0, 0, dx, dy, dz)

    # Moved the wall slightly closer
    V_wall = LJ_wall3D(Y, 2.5, 2.0, 0.5, 150.0)

    # Run the 3D Dashboard
   # animate_dashboard3D(psi, dx, dy, dz, dt, V_wall, X, Y, Z)

    render_mp4_simulation(psi, dx, dy, dz, dt, V_wall, X, Y, Z)

    # Run the PyVista GPU Engine
    #animate_pyvista3D(psi, dx, dy, dz, dt, V_wall, X, Y, Z)