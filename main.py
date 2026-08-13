import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

epsilon = 0.1

t_max = 20.0
t_steps = 100
t_eval = np.linspace(0, t_max, t_steps)

N = 256
L = 100.0 # -5 ->5
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

def compute_uxx(u, dx): # laplacian numerical method
    u_right = np.roll(u, -1)
    u_left = np.roll(u, 1)
    return (u_right - 2*u + u_left) / (dx**2)

def pde_derivs(t, y_real):
    u = y_real[:N] + 1j * y_real[N:]
    u_xx = compute_uxx(u, dx)
    u_sq = np.abs(u)**2


    # isolating
    # du_dt = i/2 * u_xx + i*|u|^2*u - eps*|u|^2*u
    du_dt = 0.5j * u_xx + 1j * u_sq * u - epsilon * u_sq * u

    return np.concatenate([du_dt.real, du_dt.imag])

# Initial condition: bright soliton sech ansatz
a0 = 1.0
w0 = 1.0
c0 = 0.1

xi0 = -5.0 # initial coordination
b0 = 0.0
phi0 = 0.0

u_init = a0 / np.cosh(w0 * (x - xi0)) * np.exp(1j * (b0 * (x - xi0)**2 + c0 * (x - xi0) + phi0))
y_init = np.concatenate([u_init.real, u_init.imag])

print("log: Running PDE Simulation...")
sol_pde = solve_ivp(pde_derivs, (0, t_max), y_init, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)
u_history = sol_pde.y[:N, :].T + 1j * sol_pde.y[N:, :].T
print("log: PDE Simulation completed.")






# --- Step 2: Decoupled Parameter Fitting ---

# x_grid: mảng không gian, a: biên độ, w: độ rộng, xi: tọa độ tâm sóng

# ansatz, sử dụng mật độ để triệt tiêu phần ảo
def sech_density(x_grid, a, w, xi):
    return (a**2) / (np.cosh(w * (x_grid - xi))**2)


a_fit_history = []
w_history_fit = []
xi_history_fit = []
b_history_fit = []
c_history_fit = []
phi_history_fit = []

print("Fitting PDE parameters...")

for i, t_val in enumerate(t_eval):
    u_current = u_history[i]
    density = np.abs(u_current)**2

    # init
    idx_max = np.argmax(density)
    xi_guess = x[idx_max]             # top of the hill have better gradient

    a_guess = np.sqrt(np.max(density)) # related to what we chooose of xi
    w_guess = w0



    # --- solve a w xi ---
    popt, _ = curve_fit(sech_density, x, density, p0=[a_guess, w_guess, xi_guess], maxfev=10000)

    a_f, w_f, xi_f = popt

    a_f = np.abs(a_f)
    w_f = np.abs(w_f)

    a_fit_history.append(a_f)
    w_history_fit.append(w_f)
    xi_history_fit.append(xi_f)



    # chain saw -> smooth function, ensure it will be a parabol
    phase = np.unwrap(np.angle(u_current))

    mask = density > 0.01 * np.max(density)

    # Kiểm tra xem vùng được lọc có đủ ít nhất 3 điểm dữ liệu để fit đa thức bậc 2 hay không
    if np.sum(mask) >= 3:
        x_win = x[mask] - xi_f        # chuyển hệ tọa độ về gốc là tâm sóng mới tìm được (xi_f)
        phase_win = phase[mask]

        # b*x^2 + c*x + phi
        poly = np.polyfit(x_win, phase_win, 2)
        b_f, c_f, phi_f = poly
    else:
        b_f, c_f, phi_f = 0.0, 0.0, 0.0

    b_history_fit.append(b_f)
    c_history_fit.append(c_f)
    phi_history_fit.append(phi_f)

print("Parameter Fitting completed.")



# --- Step 3: Solve NCVA ODEs ---
def ode_derivs(t, y):
    a, b, c, xi, w, phi = y
    da = a * ( -(2.0/3.0 + 2.0/np.pi**2) * epsilon * a**2 - b )
    db = (2.0/np.pi**2) * w**4 - (2.0/np.pi**2) * a**2 * w**2 - 2 * b**2
    dc = 0.0
    dxi = c
    dw = -2 * b * w - (4.0/np.pi**2) * epsilon * a**2 * w
    dphi = (5.0/6.0)*a**2 - (1.0/3.0)*w**2 + 0.5*c**2
    return [da, db, dc, dxi, dw, dphi]

y_ode_init = [a0, b0, c0, xi0, w0, phi0]
print("Solving ODE equations...")
sol_ode = solve_ivp(ode_derivs, (0, t_max), y_ode_init, t_eval=t_eval, method='RK45')
print("ODE equations solved.")




# --- Step 4: Plotting and Saving ---
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("Model 2: Soliton Parameters (PDE vs NCVA ODE)", fontsize=14, fontweight='bold')

params_to_plot = [
    ("Amplitude (a)", a_fit_history, sol_ode.y[0], axs[0, 0]),
    ("Chirp (b)", b_history_fit, sol_ode.y[1], axs[0, 1]),
    ("Width Parameter (w)", w_history_fit, sol_ode.y[4], axs[1, 0]),
    ("Phase (phi)", phi_history_fit, sol_ode.y[5], axs[1, 1]),
    ("Speed (c)", c_history_fit, sol_ode.y[2], axs[2, 0]),
    ("Position (xi)", xi_history_fit, sol_ode.y[3], axs[2, 1])
]

for name, pde_val, ode_val, ax in params_to_plot:
    ax.plot(t_eval, pde_val, 'ro', label='PDE Fit', markersize=3)
    ax.plot(t_eval, ode_val, 'b-', label='NCVA ODE', linewidth=1.5)
    ax.set_title(name)
    ax.set_xlabel("Time (t)")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("./nls_model2_comparison.png", dpi=150)
print("Plot saved as nls_model2_comparison.png")
