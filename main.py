import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from numba import njit, prange
import pyvista as pv

# 2D Quantum Functions

@njit(parallel=True, cache=True)
def hamiltonianOperator3D(psi, dx, dy, dz, V):
    nx, ny, nz = psi.shape
    result = np.empty_like(psi)

    # Pre-calculate squared denominators for speed
    dx2 = dx ** 2
    dy2 = dy ** 2
    dz2 = dz ** 2

    # Parallel over i: iteration i only ever writes result[i, :, :], so there's
    # no cross-iteration write race (same reasoning as lattice_potential_and_forces).
    for i in prange(nx):
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

def createWavePacketSlab3D(X, Y, Z, y0, sigma_y, ky, dx, dy, dz, m=0):
    """Extended wavepacket: a SLAB, not a blob.

    Gaussian in y (the propagation direction) with width sigma_y, and UNIFORM across x and z.
    Physically this is a plane wave incident on a periodic surface -- the standard setup for
    surface scattering -- rather than a localized 3D ball.

    Two problems dissolve at once:

    1. COST. A localized sigma = 5 A ball needs a box of +/-3sigma = +/-28 Bohr in EVERY
       direction, and since the coupling range b = 0.3 A independently pins dx <= 0.19 Bohr,
       that is ~27 million grid points. As a slab, x and z only need a few lattice periods,
       which is ~35x cheaper for the same sigma_y = 5 A and the same 1 A lattice spacing.

    2. COVERAGE. A blob narrower than the lattice spacing drives only the site it happens to
       sit over (measured: centre force 0.65 vs corner 0.026). Being uniform in x,z, a slab
       drives every site in the wall equally -- which is the whole point of tiling the wall.

    Because the packet is uniform in x,z it is periodic in those directions by construction,
    so it is exactly compatible with the periodic boundary conditions: no wrap-around leak.

    m != 0 still attaches the exp(i*m*phi) vortex phase about the y axis (the CISS chirality
    knob). Note the r^|m| radial factor of the localized version is deliberately dropped here:
    it grows without bound away from the axis, which is meaningless for an extended state.
    """
    gaussian_y = np.exp(-((Y - y0) ** 2) / (2 * sigma_y ** 2))
    plane_wave = np.exp(1j * ky * Y)

    psi = gaussian_y * plane_wave
    if m != 0:
        phi = np.arctan2(Z, X)
        psi = psi * np.exp(1j * m * phi)

    probability_sum = np.sum(np.abs(psi) ** 2) * dx * dy * dz
    psi = psi / np.sqrt(probability_sum)
    return psi.astype(np.complex128)


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


def make_wall_lattice(n_per_axis, x_arr, z_arr):
    """(x, z) positions of the sites tiling the wall.

    The spacing is chosen COMMENSURATE with the box (spacing = box_width / n_per_axis) so the
    lattice tiles seamlessly under the periodic boundary conditions. Forcing an exact 1.000 A
    spacing into a 6 Bohr box would leave a 0.175-site remainder and put a visible seam at the
    x/z faces. With n_per_axis = 3 the spacing is 2.0 Bohr = 1.058 A, i.e. within 6% of the
    whiteboard's 1 A and periodic to machine precision.
    """
    if n_per_axis <= 1:
        return np.zeros((1, 2)), 0.0
    Lx = float(x_arr.max() - x_arr.min())
    spacing = Lx / n_per_axis
    c = (np.arange(n_per_axis) - (n_per_axis - 1) / 2.0) * spacing
    return np.array([[a, b] for a in c for b in c], dtype=np.float64), spacing


@njit(parallel=True, cache=True)
def lattice_potential_and_forces(psi, x_arr, y_arr, z_arr,
                                 sites_x, sites_z, y_c, A, b, dV,
                                 cutoff, V_out, F_part, period_x, period_z):
    """ONE fused, PARALLEL pass: the classical-electron potential on the grid AND the
    Ehrenfest force on every site, together.

    This is the function that makes step 3 affordable. Four things are going on:

    1. FUSED. The obvious implementation -- a Python loop over sites, each building full-grid
       numpy temporaries (r, V, force integrand) -- allocates ~4 arrays of nx*ny*nz per site
       per step and evaluates the exponential twice. Here the grid is walked once with zero
       temporaries and the force reduction accumulates in the same sweep as the potential.

    2. PARALLEL over the x index. Iteration i touches only V_out[i, :, :] and F_part[i, :],
       so there is no write race and no need for atomics. The per-site force is reduced
       afterwards by summing F_part over i.

    3. PREALLOCATED. V_out and F_part are supplied by the caller and reused every step,
       instead of allocating a fresh nx*ny*nz array 50,000 times.

    4. CUTOFF. The coupling decays as exp(-r/b) with b = 0.567 Bohr, so past r = cutoff it is
       numerically irrelevant (at 12b the factor is 6e-6). Sites are confined to the wall
       plane, so in a LARGE box most of the volume is out of range of every site and gets
       skipped entirely -- this is what keeps the cost from scaling as N_sites x full_grid
       once the box grows. In the current 6 Bohr box nothing is far enough away to skip, so
       it costs a branch and saves nothing yet.

    5. MINIMUM IMAGE in x and z. The Laplacian is periodic in every direction, so the
       potential has to be too, or sites near the x/z faces silently lose the part of their
       interaction that should wrap around. Measured without it: the 36 sites of a periodic
       lattice under a perfectly uniform slab packet felt forces spread over 25%, when by
       symmetry they must all be identical. With minimum image the spread drops to roundoff.
       y is deliberately NOT wrapped: the wall lives there and the packet never reaches that
       face, so wrapping it would fold the wall barrier back onto the incoming packet.

    psi is passed complex and |psi|^2 is formed inline, saving one full-grid allocation/step.
    """
    nx = x_arr.size
    ny = y_arr.size
    nz = z_arr.size
    N = sites_x.size
    cut2 = cutoff * cutoff

    V_out[:, :, :] = 0.0
    F_part[:, :] = 0.0

    for i in prange(nx):
        for s in range(N):
            ddx = x_arr[i] - sites_x[s]
            # minimum image: fold into [-period_x/2, +period_x/2]
            ddx = ddx - period_x * np.round(ddx / period_x)
            if ddx < -cutoff or ddx > cutoff:
                continue
            ddx2 = ddx * ddx
            sy = y_c[s]
            sz = sites_z[s]
            acc = 0.0
            for j in range(ny):
                ddy = y_arr[j] - sy
                ddxy2 = ddx2 + ddy * ddy
                if ddxy2 > cut2:
                    continue
                for k in range(nz):
                    ddz = z_arr[k] - sz
                    ddz = ddz - period_z * np.round(ddz / period_z)
                    r2 = ddxy2 + ddz * ddz
                    if r2 > cut2:
                        continue
                    r = np.sqrt(r2) + 1e-10
                    v = A * np.exp(-r / b)
                    V_out[i, j, k] += v
                    pr = psi[i, j, k].real
                    pi = psi[i, j, k].imag
                    acc += (pr * pr + pi * pi) * (-v * (ddy / (b * r)))
            F_part[i, s] = acc

    F = np.zeros(N)
    for s in range(N):
        tot = 0.0
        for i in range(nx):
            tot += F_part[i, s]
        F[s] = tot * dV
    return F


def build_springs_mesh(template_pts, sites_x, sites_z, y_nuc, y_c,
                       tube_radius=0.02, min_length=0.02):
    """All N springs as a SINGLE mesh: one PolyData holding N polylines, tubed once.

    Tubing each spring separately would mean N VTK filter calls per frame; this is one.
    """
    npts = template_pts.shape[0]
    N = sites_x.size
    all_pts = np.empty((N * npts, 3))
    lines = np.empty(N * (npts + 1), dtype=np.int64)
    for s in range(N):
        length = max(float(y_nuc - y_c[s]), min_length)
        block = template_pts * np.array([1.0, length, 1.0])
        # local +Y maps to world -Y (nucleus -> electron), so flip y
        block = block * np.array([1.0, -1.0, 1.0])
        block = block + np.array([sites_x[s], y_nuc, sites_z[s]])
        all_pts[s * npts:(s + 1) * npts] = block
        off = s * (npts + 1)
        lines[off] = npts
        lines[off + 1:off + 1 + npts] = np.arange(s * npts, (s + 1) * npts)
    poly = pv.PolyData(all_pts, lines=lines)
    return poly.tube(radius=tube_radius)


def render_mp4_simulation(psi_init, dx, dy, dz, dt, V_wall, X, Y, Z, frames=1250, steps_per_frame=40,
                          y_wall_pos=2.5,
                          nucleus_offset=0.05, r0=None,
                          n_sites_per_axis=3, render_stride=1,
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

    # 1D axes (the fused kernel indexes these instead of the big 3D meshgrids)
    x_arr = np.ascontiguousarray(X[:, 0, 0])
    y_arr = np.ascontiguousarray(Y[0, :, 0])
    z_arr = np.ascontiguousarray(Z[0, 0, :])
    dV = dx * dy * dz

    if r0 is None:
        r0 = 1.0 * BOHR_PER_ANGSTROM          # whiteboard: r0 = 1 Angstrom = 1.890 Bohr
    if A_nuc is None:
        A_nuc = A_au

    if nucleus_pos is not None:
        y_nuc = nucleus_pos[1]
        sites = np.array([[nucleus_pos[0], nucleus_pos[2]]], dtype=np.float64)
        spacing = 0.0
    else:
        # הגרעין יושב מעט *אחרי* הקיר (בתוך חומר הקיר), לא לפניו
        y_nuc = y_wall_pos + nucleus_offset
        sites, spacing = make_wall_lattice(n_sites_per_axis, x_arr, z_arr)

    sites_x = np.ascontiguousarray(sites[:, 0])
    sites_z = np.ascontiguousarray(sites[:, 1])
    N_sites = sites_x.size

    # שיווי המשקל של האלקטרון נקבע ע"י הגרעין: בדיוק r0 לפניו
    y_eq = y_nuc - r0
    print(f"  wall lattice: {N_sites} sites ({n_sites_per_axis}x{n_sites_per_axis}), "
          f"spacing {spacing:.4f} Bohr = {spacing * 0.52917721:.3f} Angstrom")
    print(f"  nuclei anchored at y = {y_nuc:.3f} (wall at {y_wall_pos}, offset {nucleus_offset})")
    print(f"  electron equilibrium  y_eq = y_nuc - r0 = {y_eq:.3f}  (r0 = {r0:.3f} Bohr = "
          f"{r0 * 0.52917721:.2f} Angstrom)")

    # כל אלקטרון קלאסי הוא דרגת חופש נפרדת -> מערכים באורך N.
    # שומרים מהירות (ולא y_prev) כי המשלב עבר ל-velocity-Verlet עם הערכת אמצע.
    state = {'psi': psi_init, 't': 0.0,
             'y_c_curr': np.full(N_sites, y_eq),
             'v_c': np.zeros(N_sites)}

    # --- הגרעינים החיוביים: מקובעים, ולכן הפוטנציאל שלהם סטטי לחלוטין ---
    # מסכמים על כל האתרים פעם אחת בלבד ומאחדים לתוך פוטנציאל הקיר. כך כל
    # שורת הגרעינים *לא* מוסיפה שום עלות חישובית בלולאה הפנימית.
    # הסימן שלילי: הגרעין (+|e|) מושך את האלקטרון הקוונטי (-|e|).
    # periods of the transverse (genuinely periodic) directions, for minimum image.
    # +dx / +dz because linspace endpoints are inclusive: the period is one cell MORE
    # than (max - min).
    period_x = float(x_arr.max() - x_arr.min()) + dx
    period_z = float(z_arr.max() - z_arr.min()) + dz

    V_nuc = np.zeros_like(V_wall)
    for sx, sz in sites:
        ddx = X - sx
        ddx -= period_x * np.round(ddx / period_x)      # minimum image, as in the kernel
        ddz = Z - sz
        ddz -= period_z * np.round(ddz / period_z)
        r_nuc = np.sqrt(ddx ** 2 + (Y - y_nuc) ** 2 + ddz ** 2) + 1e-10
        V_nuc += -A_nuc * np.exp(-r_nuc / b_au)
    V_static = V_wall + V_nuc

    print("Setting up Off-Screen Renderer...")

    # 1. מפעילים את PyVista במצב שקט (ללא חלון) וברזולוציה גבוהה
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.set_background('black')

    # --- גדלים ויזואליים ביחס למרווח הסריג, לא בערכים מוחלטים ---
    # קודם הרדיוסים היו קבועים (0.15 Bohr) — בקופסה באורך 70 Bohr זה 0.2% מהתמונה,
    # כלומר בלתי נראה. עכשיו הכל מתכייל לפי spacing, אז זה נראה נכון בכל גודל קופסה.
    vis = spacing if spacing > 0 else r0
    R_ELECTRON = 0.22 * vis
    R_NUCLEUS = 0.24 * vis
    COIL_R = 0.10 * vis
    TUBE_R = 0.035 * vis

    # 2. האלקטרונים הקלאסיים — כל N הכדורים כאקטור *אחד* דרך glyph,
    #    במקום N קריאות add_mesh נפרדות
    electron_pts = np.column_stack([sites_x, state['y_c_curr'], sites_z])
    electrons_poly = pv.PolyData(electron_pts)
    plotter.add_mesh(electrons_poly.glyph(geom=pv.Sphere(radius=R_ELECTRON), scale=False, orient=False),
                     color='green', smooth_shading=True, specular=0.5, name='electrons')

    # 2b. הקפיצים — כל N הקפיצים כרשת אחת (polyline אחת לכל אתר, tube בקריאה אחת).
    # הכיוון קבוע (מהגרעין כלפי -Y), ולכן קפיץ יכול רק להתקצר לכיוון אפס ולעולם
    # לא "יתהפך" לצד השני של האלקטרון.
    spring_template = make_spring_template(n_coils=6, n_points=60, coil_radius=COIL_R)
    springs_mesh = build_springs_mesh(spring_template, sites_x, sites_z, y_nuc,
                                     state['y_c_curr'], tube_radius=TUBE_R)
    plotter.add_mesh(springs_mesh, color='silver', name='springs', specular=0.6)

    # 2c. הגרעינים החיוביים (+|e|) — מקובעים, ולכן מצוירים פעם אחת ולא מתעדכנים בלולאה.
    # הם *אמורים* לא לזוז: זה כל הרעיון של "anchored".
    nuclei_pts = np.column_stack([sites_x, np.full(N_sites, y_nuc), sites_z])
    nuclei_poly = pv.PolyData(nuclei_pts)
    plotter.add_mesh(nuclei_poly.glyph(geom=pv.Sphere(radius=R_NUCLEUS), scale=False, orient=False),
                     color='red', smooth_shading=True, specular=0.5, name='nuclei')

    # 2d. הקיר עצמו — עד עכשיו הוא היה *בלתי נראה* (רק פוטנציאל ב-y=y_wall_pos),
    # ולכן היה בלתי אפשרי לשפוט אם הגרעין לפני או אחרי הקיר. עכשיו הוא מצויר.
    wall_plane = pv.Plane(center=(0.0, y_wall_pos, 0.0), direction=(0.0, 1.0, 0.0),
                          i_size=float(X.max() - X.min()), j_size=float(Z.max() - Z.min()))
    # opacity הוא הכפתור לכוונון: העלה אם הקיר חיוור מדי, הורד אם הוא מסתיר את הגל
    plotter.add_mesh(wall_plane, color='deepskyblue', opacity=0.35, show_edges=True,
                     edge_color='deepskyblue', name='wall')

    # 3. הכנת הרשת התלת-ממדית לחבילת הגל.
    # render_stride: הפיזיקה תמיד ברזולוציה מלאה, אבל *הרינדור* מדלל.
    # ב-27M ווקסלים add_volume נחנק לגמרי; stride=3 מוריד ל-1M בלי לגעת בפיזיקה.
    rs = max(1, int(render_stride))
    dens_view = (np.abs(state['psi']) ** 2)[::rs, ::rs, ::rs]
    grid = pv.ImageData()
    grid.dimensions = np.array(dens_view.shape)
    grid.spacing = (dx * rs, dy * rs, dz * rs)
    grid.origin = (X.min(), Y.min(), Z.min())
    grid.point_data["Density"] = dens_view.flatten(order="F")
    if rs > 1:
        print(f"  render_stride={rs}: volume shows {dens_view.size/1e6:.2f} M voxels "
              f"(physics stays at {state['psi'].size/1e6:.2f} M)")

    # רינדור ענן (Volume) עם צבע וסף שקיפות כדי שייראה כמו מפת חום
    plotter.add_volume(grid, scalars="Density", cmap="magma", opacity="linear", show_scalar_bar=False)

    plotter.add_bounding_box(color='white', line_width=1.0)
    plotter.add_axes()
    plotter.camera_position = 'yz'
    #plotter.camera.azimuth += 15  # Rotates the camera sideways (try numbers between 45 and 90)
    plotter.camera.elevation += 15  # Drops the camera angle down for a more straight-on view

    # --- מאגרים שמוקצים פעם אחת ומשומשים מחדש בכל צעד (במקום 50,000 הקצאות) ---
    V_buf = np.zeros((x_arr.size, y_arr.size, z_arr.size))
    F_part_buf = np.zeros((x_arr.size, N_sites))
    # מעבר ל-12*b: exp(-12) ~ 6e-6, כלומר תרומה זניחה מעבר לרדיוס הזה
    cutoff = 12.0 * b_au

    # --- Baselines for Δ tracking (mirrors animate_dashboard3D) ---
    initial_norm = np.sum(np.abs(psi_init) ** 2) * dx * dy * dz
    # also seeds accel_force, the opening force of the velocity-Verlet recursion
    accel_force = lattice_potential_and_forces(psi_init, x_arr, y_arr, z_arr,
                                               sites_x, sites_z, state['y_c_curr'],
                                               A_au, b_au, dV, cutoff, V_buf, F_part_buf,
                                               period_x, period_z)
    V_initial = V_static + V_buf
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
        # הרצת הפיזיקה
        for _ in range(steps_per_frame):
            # --- velocity-Verlet with a MIDPOINT potential for the quantum step ---
            # The old ordering advanced psi with the potential from the START of the interval,
            # which is only 1st order in dt: measured error ratio exactly 2.00 per halving.
            # Evaluating the quantum step at the midpoint classical position makes the whole
            # coupling 2nd order (measured ratio 4.00), and at dt=4e-4 it is ~1160x more
            # accurate. Since the splitting error goes as dt^p * T, that is what makes a long
            # run affordable at all.
            acceleration = (-k_spring * (state['y_c_curr'] - y_eq) + accel_force) / m_c
            v_half = state['v_c'] + 0.5 * acceleration * dt
            y_mid = state['y_c_curr'] + 0.5 * v_half * dt

            # potential at the MIDPOINT classical position
            lattice_potential_and_forces(state['psi'], x_arr, y_arr, z_arr,
                                         sites_x, sites_z, y_mid, A_au, b_au, dV,
                                         cutoff, V_buf, F_part_buf, period_x, period_z)
            # V_static כבר כולל את הקיר ואת *כל* הגרעינים המקובעים (מחושב פעם אחת מחוץ ללולאה)
            V_current = V_static + V_buf

            state['psi'] = RK4_3D(state['t'], state['psi'], dx, dy, dz, dt, V_current)
            state['y_c_curr'] = state['y_c_curr'] + v_half * dt

            # forces at the NEW position; reused as the opening force of the next step,
            # so this costs one kernel call per step, not two
            accel_force = lattice_potential_and_forces(
                state['psi'], x_arr, y_arr, z_arr,
                sites_x, sites_z, state['y_c_curr'], A_au, b_au, dV,
                cutoff, V_buf, F_part_buf, period_x, period_z)
            acceleration = (-k_spring * (state['y_c_curr'] - y_eq) + accel_force) / m_c
            state['v_c'] = v_half + 0.5 * acceleration * dt
            state['t'] += dt

        # עדכון הגרפיקה בזיכרון (מדולל לפי render_stride)
        new_density = np.abs(state['psi']) ** 2
        grid["Density"][:] = new_density[::rs, ::rs, ::rs].flatten(order="F")

        # עדכון כל האלקטרונים כאקטור אחד
        electron_pts = np.column_stack([sites_x, state['y_c_curr'], sites_z])
        electrons_poly = pv.PolyData(electron_pts)
        plotter.add_mesh(electrons_poly.glyph(geom=pv.Sphere(radius=R_ELECTRON), scale=False, orient=False),
                         color='green', smooth_shading=True, specular=0.5, name='electrons')

        # עדכון כל הקפיצים כרשת אחת
        springs_mesh = build_springs_mesh(spring_template, sites_x, sites_z, y_nuc,
                                          state['y_c_curr'], tube_radius=TUBE_R)
        plotter.add_mesh(springs_mesh, color='silver', name='springs', specular=0.6)

        # --- חישוב הדלתות (אנרגיה קוונטית/מכנית ונורמליזציה) לפריים הנוכחי ---
        lattice_potential_and_forces(state['psi'], x_arr, y_arr, z_arr,
                                     sites_x, sites_z, state['y_c_curr'],
                                     A_au, b_au, dV, cutoff, V_buf, F_part_buf,
                                               period_x, period_z)
        V_current_final = V_static + V_buf

        current_norm = np.sum(np.abs(state['psi']) ** 2) * dx * dy * dz
        quant_E = np.real(
            np.sum(np.conj(state['psi']) * hamiltonianOperator3D(state['psi'], dx, dy, dz,
                                                                 V_current_final)) * dx * dy * dz)
        # אנרגיה קלאסית = סכום על *כל* האתרים.
        # משתמשים במהירות האמיתית (v_c) ולא בהפרש אחורי (y_curr - y_prev)/dt, שסבל
        # מפיגור של חצי צעד ולכן הכניס רעש מדומה לאנרגיה הקינטית.
        class_E = np.sum(0.5 * m_c * state['v_c'] ** 2
                         + 0.5 * k_spring * (state['y_c_curr'] - y_eq) ** 2)

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
    # ----------------------------------------------------------------------------------
    # WHY ONLY ONE SPRING RESPONDS, and the two ways out.
    #
    # sigma = 0.75 -> FWHM 1.77 Bohr, which is SMALLER than the 2.0 Bohr lattice spacing,
    # so the packet slips between sites and only the centre one feels it (measured: centre
    # force 0.65 vs corner 0.026, a factor of 25).
    #
    #   A) widen the packet in this box -- but +/-3 Bohr is too tight. To get appreciable
    #      density at the neighbouring site you need sigma ~1.5, and that leaks 4.6% of
    #      |psi|^2 per axis straight through the periodic faces. Not worth it.
    #
    #   B) grow the box and spend the speed reclaimed below. L_limit = 6.0 with
    #      n_sites_per_axis = 6 keeps the SAME 1.058 A spacing, gives 36 sites, and lets
    #      sigma_xz = 1.5 drive ~9 of them with a wrap leak of only 6e-5 % per axis.
    #      Cost: 31x more kernel work per step, but 10x fewer steps and then spread over
    #      cores -- roughly break-even against the run you just did.
    #
    # To switch to B: L_limit = 6.0, n_sites_per_axis = 6, and sigma_x/sigma_z = 1.5 below.
    # ----------------------------------------------------------------------------------
    ANGSTROM = 1.0 / 0.52917721        # 1 A in Bohr = 1.8897

    # ==================================================================================
    # PERFECT CUBE.
    #
    # The cube side is NOT free: sigma_y = 5 A means the packet alone spans 6 sigma = 30 A,
    # so a cube has to be at least 30 A on EVERY side -- including x and z, which physically
    # only needed a few lattice periods. That is what makes this configuration expensive:
    #
    #     24 A cube -> too small, the packet wraps into itself
    #     30 A cube -> 301^3 = 27.3 M points, 900 sites, ~2.6 GB, ~19x the previous run
    #     36 A cube -> 361^3 = 47.0 M points, 1296 sites, ~4.5 GB, ~29x
    #
    # 30 A is the minimum cube that holds a 5 A packet, so that is what is set below.
    #
    # IF THIS IS TOO SLOW, in order of preference:
    #   1. FRAMES = 400        (linear: 3x faster, no physics change at all)
    #   2. render_stride = 4   (rendering only, physics untouched)
    #   3. SIGMA_A = 3.0       -> 18 A cube, ~5x cheaper. Changes the physics you specified.
    # ==================================================================================
    SPACING_A = 1.0
    SIGMA_A = 5.0
    N_SITES_PER_AXIS = 30                       # 30 x 1.0 A = 30 A cube side
    L_HALF = 0.5 * N_SITES_PER_AXIS * SPACING_A * ANGSTROM      # +/-28.35 Bohr
    SIGMA_Y = SIGMA_A * ANGSTROM

    # dx is pinned by the COUPLING RANGE b = 0.3 A = 0.567 Bohr (~3 points across it),
    # NOT by the packet or the box. This is the constraint that makes a big cube costly.
    D = 0.567 / 3.0
    n_side = int(round(2 * L_HALF / D)) + 1

    x_arr = np.linspace(-L_HALF, L_HALF, n_side)
    y_arr = np.linspace(-L_HALF, L_HALF, n_side)
    z_arr = np.linspace(-L_HALF, L_HALF, n_side)

    X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]
    dz = z_arr[1] - z_arr[0]

    # the packet fills the cube (6 sigma == the side), so it starts centred and the wall
    # sits near the +y face; the interaction begins essentially at t=0
    Y_WALL = 0.70 * L_HALF
    Y0_PACKET = -0.30 * L_HALF

    KY = 2.0                     # capped by dx: ~10 points per de Broglie wavelength
    dt = 0.0015                  # ~17% of the RK4 stability limit at this dx
    STEPS_PER_FRAME = 6
    FRAMES = 1250
    RENDER_STRIDE = 3            # 27 M voxels would choke add_volume; physics stays full-res

    print(f"CUBE {n_side}^3 = {X.size/1e6:.2f} M points, side "
          f"{2*L_HALF*0.52917721:.1f} A, dx={dx:.4f} Bohr")
    print(f"  memory ~{X.size*16*6/1e9:.1f} GB for psi + RK4 stages")
    print(f"  {N_SITES_PER_AXIS}x{N_SITES_PER_AXIS} = {N_SITES_PER_AXIS**2} sites "
          f"at {SPACING_A:.3f} A spacing")
    print(f"  simulated time {FRAMES*STEPS_PER_FRAME*dt:.2f} a.u. "
          f"({FRAMES*STEPS_PER_FRAME} steps)")

    # SLAB packet: Gaussian in y, uniform in x,z -> every site is driven equally
    psi = createWavePacketSlab3D(X, Y, Z, Y0_PACKET, SIGMA_Y, KY, dx, dy, dz, m=0)

    V_wall = LJ_wall3D(Y, Y_WALL, 2.0, 0.5, 150.0)

    render_mp4_simulation(psi, dx, dy, dz, dt, V_wall, X, Y, Z,
                          frames=FRAMES,
                          steps_per_frame=STEPS_PER_FRAME,
                          y_wall_pos=Y_WALL,
                          n_sites_per_axis=N_SITES_PER_AXIS,
                          render_stride=RENDER_STRIDE)

    # Run the PyVista GPU Engine
    #animate_pyvista3D(psi, dx, dy, dz, dt, V_wall, X, Y, Z)