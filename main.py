import numpy as np
from scipy.sparse import csr_matrix, identity, kron, diags # use Compressed Sparse Row matrices
from scipy.sparse.linalg import expm_multiply, eigs
import matplotlib.pyplot as plt


##### helper functions #####
color1 = "#4e9773"  # green
color3 = "#d2561a"  # orange



def random_mixed_state(n_qubits, seed=None):

    dim = 2**n_qubits
    rng = np.random.default_rng(seed)

    A = (
        rng.normal(size=(dim, dim))
        + 1j * rng.normal(size=(dim, dim))
    )

    A = A / np.linalg.norm(A)

    rho = A @ A.conj().T
    rho = rho / np.trace(rho)

    return rho

def local_operator(operator, site, n_qubits):
    """
    Embed a single-qubit operator into an n-qubit Hilbert space.

    Example for n=3 and site=1:
        I ⊗ operator ⊗ I
    """

    result = csr_matrix([[1.0 + 0.0j]])

    for j in range(n_qubits):
        factor = operator if j == site else I2
        result = kron(result, factor, format="csr")

    return result

def format_float_for_filename(value, digits=4):
    """
    Convert a float into a filename-safe string.

    Examples:
        -0.2605 -> m0p2605
         0.25   -> 0p25
    """
    text = f"{value:.{digits}g}"
    return text.replace("-", "m").replace(".", "p")


###### operators ###### 

Sx = csr_matrix(
    0.5 * np.array([
        [0, 1],
        [1, 0]
    ], dtype=complex)
)

Sy = csr_matrix(
    0.5 * np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=complex)
)

Sz = csr_matrix(
    0.5 * np.array([
        [1, 0],
        [0, -1]
    ], dtype=complex)
)

# Lowering operator |0><1|
sigma_minus = csr_matrix(
    np.array([
        [0, 1],
        [0, 0]
    ], dtype=complex)
)

I2 = identity(2, format="csr", dtype=complex)

###### Heisenberg chain model ######

def heisenberg_hamiltonian(n_qubits):
    """
    H = sum_i (
          Sx_i Sx_{i+1}
        + Sy_i Sy_{i+1}
        + Sz_i Sz_{i+1}
    )
    """

    dim = 2**n_qubits
    H = csr_matrix((dim, dim), dtype=complex)

    for i in range(n_qubits - 1):
        Sx_i = local_operator(Sx, i, n_qubits)
        Sx_ip1 = local_operator(Sx, i + 1, n_qubits)

        Sy_i = local_operator(Sy, i, n_qubits)
        Sy_ip1 = local_operator(Sy, i + 1, n_qubits)

        Sz_i = local_operator(Sz, i, n_qubits)
        Sz_ip1 = local_operator(Sz, i + 1, n_qubits)

        H += (
            Sx_i @ Sx_ip1
            + Sy_i @ Sy_ip1
            + Sz_i @ Sz_ip1
        )

    return H

def cooling_jump(n_qubits):
    """
    B = I^(n-1) ⊗ |0><1|
    """

    return local_operator(
        sigma_minus,
        site=n_qubits - 1,
        n_qubits=n_qubits
    )

###### time evolution ###### 

def liouvillian_matrix(n_qubits, gamma_1=1.0):
    """
    Construct the matrix representation of the GKLS generator
    using column-wise vectorization.
    """

    H = heisenberg_hamiltonian(n_qubits)
    B = cooling_jump(n_qubits)

    dim = 2**n_qubits
    I = identity(dim, format="csr", dtype=complex)

    BdagB = B.getH() @ B

    # We use column-wise vectorization:
    #
    #     vec(A rho C) = (C^T ⊗ A) vec(rho).
    #
    # Therefore,
    #
    #     vec(H rho)          = (I ⊗ H) vec(rho),
    #     vec(rho H)          = (H^T ⊗ I) vec(rho),
    #     vec(B rho B^\dagger)= (B^* ⊗ B) vec(rho).
    #
    # Hence the Liouvillian matrix is
    #
    #     L_tilde =
    #         -i (I ⊗ H - H^T ⊗ I)
    #         + gamma_1 [
    #             B^* ⊗ B
    #             - 1/2 (I ⊗ B^\dagger B)
    #             - 1/2 ((B^\dagger B)^T ⊗ I)
    #         ].

    L_hamiltonian = -1j * (
        kron(I, H, format="csr")
        - kron(H.T, I, format="csr")
    )

    L_dissipator = gamma_1 * (
        kron(B.conjugate(), B, format="csr")
        - 0.5 * kron(I, BdagB, format="csr")
        - 0.5 * kron(BdagB.T, I, format="csr")
    )

    L_tilde = L_hamiltonian + L_dissipator

    return L_tilde

def liouvillian_gap_dense(L_tilde, zero_tol=1e-10):
    """
    Calculate the Liouvillian gap by diagonalizing the full dense matrix.

    Use only for small Liouvillians.
    """

    eigenvalues = np.linalg.eigvals(L_tilde.toarray())

    # Remove the stationary eigenvalue lambda = 0
    nonzero_eigenvalues = eigenvalues[
        np.abs(eigenvalues) > zero_tol
    ]

    # Nonzero eigenvalue with real part closest to zero
    lambda_slow = nonzero_eigenvalues[
        np.argmax(np.real(nonzero_eigenvalues))
    ]

    Delta = -np.real(lambda_slow)

    return Delta, lambda_slow


def liouvillian_gap_sparse(
    L_tilde,
    k=30,
    zero_tol=1e-10
):
    """
    Calculate the Liouvillian gap using a sparse eigensolver.

    We ask for the k eigenvalues with largest real part,
    because the gap is determined by the nonzero eigenvalue
    whose real part is closest to zero.

    This avoids converting the large Liouvillian to a dense matrix.
    """

    eigenvalues = eigs(
        L_tilde,
        k=k,
        which="LR",       # asks for eigenvalues with the largest (=closest to 0) real part
        return_eigenvectors=False,
        tol=1e-10,
        maxiter=100000
    )

    # Remove the stationary eigenvalue lambda = 0
    nonzero_eigenvalues = eigenvalues[
        np.abs(eigenvalues) > zero_tol
    ]

    lambda_slow = nonzero_eigenvalues[
        np.argmax(np.real(nonzero_eigenvalues))
    ]

    Delta = -np.real(lambda_slow)

    return Delta, lambda_slow

def asymptotic_map(n_qubits):
    """
    Matrix representation of the asymptotic cooling map

        Phi_inf(X) = rho_ss * Tr(X),

    where rho_ss = |00...0><00...0|.

    We use the same column-wise vectorization as for L_tilde.
    """

    dim = 2**n_qubits

    # Stationary state |00...0><00...0|
    rho_ss = np.zeros((dim, dim), dtype=complex)
    rho_ss[0, 0] = 1.0

    # Column-wise vectorization of rho_ss
    vec_rho_ss = rho_ss.reshape(-1, order="F")

    # vec(I) represents the trace functional:
    #
    #     vec(I)^\dagger vec(X) = Tr(X)
    #
    vec_I = np.eye(dim, dtype=complex).reshape(
        -1,
        order="F"
    )

    # Outer product:
    #
    #     |rho_ss)) ((I|
    #
    Phi_inf = np.outer(
        vec_rho_ss,
        vec_I.conj()
    )

    return Phi_inf

def superoperator_trace_norm_difference(L_tilde, Phi_inf, t):
    """
    Calculate

        || exp(L_tilde * t) - Phi_inf ||_1,

    where ||.||_1 is the Schatten trace norm,
    i.e. the sum of the singular values.
    """

    # For Figure 3 we need the full time-evolution superoperator,
    # not just its action on one initial state.
    evolution_map = expm_multiply( # entire evolution matrix, but without converting the Liouvillian to dense first and feeding it through scipy.linalg.expm
        L_tilde * t,
        np.eye(L_tilde.shape[0], dtype=complex)
    )

    difference = evolution_map - Phi_inf

    singular_values = np.linalg.svd(
        difference,
        compute_uv=False
    )

    trace_norm = np.sum(singular_values)

    return trace_norm

###### plots ###### 

def plot_infidelity_time_evolution(n, steps=120, seed=1):
    """
    Plot the infidelity during the cooling dynamics for a random
    mixed initial state.

    The infidelity is

        I(t) = 1 - <0...0|rho(t)|0...0>.

    For the dynamics considered here, its asymptotic decay is expected
    to scale as exp(-2 Delta t), where Delta is the Liouvillian gap.
    """

    L_tilde = liouvillian_matrix(n)

    # Calculate the Liouvillian gap.
    if n <= 100:
        Delta, _ = liouvillian_gap_dense(L_tilde)
    else:
        Delta, _ = liouvillian_gap_sparse(
            L_tilde,
            k=30
        )

    # Random mixed initial state.
    rho0 = random_mixed_state(
        n,
        seed=seed
    )

    # Column-wise vectorization.
    vec_rho0 = rho0.reshape(
        -1,
        order="F"
    )

    # Simulate several relaxation times.
    t_max = 6 / Delta

    times = np.linspace(
        0,
        t_max,
        steps + 1
    )

    trajectory_vec = expm_multiply(
        L_tilde,
        vec_rho0,
        start=0,
        stop=t_max,
        num=steps + 1
    )

    # Since |00...0> is the first computational basis state,
    # the first component of vec(rho) is rho_00.
    infidelity = (
        1.0
        - np.real(trajectory_vec[:, 0])
    )

    # Anchor the asymptotic reference curve at a late-time point.
    t_ref = 3 / Delta

    ref_index = np.argmin(
        np.abs(times - t_ref)
    )

    I_ref = infidelity[ref_index]

    reference_decay = I_ref * np.exp(
        -2 * Delta
        * (times - times[ref_index])
    )

    # Fit the late-time infidelity decay:
    #
    #     I(t) ~ C exp(-r t)
    #
    # so
    #
    #     log I(t) ~ log C - r t.
    fit_mask = (
        (times >= 2.5 / Delta)
        & (times <= 5 / Delta)
        & (infidelity > 0)
    )

    slope, _ = np.polyfit(
        times[fit_mask],
        np.log(infidelity[fit_mask]),
        1
    )
    
    print(f"N = {n} | Infidelity time evolution")
    print("Liouvillian gap Delta:", Delta)
    print("Fitted asymptotic slope:", slope)
    print("Expected slope -2 Delta:", -2 * Delta)
    print("\n")

    # Plot.
    plt.figure()

    plt.plot(
        times,
        infidelity,
        color=color1,
        label="Infidelity $I(t)$"
    )

    plt.plot(
        times,
        reference_decay,
        "--",
        color=color3,
        label=r"$e^{-2\Delta t}$"
    )

    plt.yscale("log")
    plt.xlabel("t")
    #plt.title(f"N = {n}")
    plt.legend()
    plt.tight_layout()
    fit_name = format_float_for_filename(slope)
    expected_name = format_float_for_filename(-2 * Delta)

    plt.savefig(
        f"infidelity_time_evolution_N{n}_fit{fit_name}_expected{expected_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_superoperator_convergence(n, steps=50):
    """
    Plot the convergence of the full dynamical map to its asymptotic map.

    The plotted quantity is

        || exp(L t) - Phi_inf ||_1,

    where Phi_inf(X) = rho_ss Tr(X).

    Its asymptotic decay is expected to scale as exp(-Delta t),
    where Delta is the Liouvillian gap.

    This calculation uses the full Liouvillian superoperator and is
    therefore intended only for small system sizes.
    """

    L_tilde = liouvillian_matrix(n)
    Phi_inf = asymptotic_map(n)

    # Calculate the Liouvillian gap.
    if n <= 5:
        Delta, lambda_slow = liouvillian_gap_dense(L_tilde)
    else:
        Delta, lambda_slow = liouvillian_gap_sparse(
            L_tilde,
            k=30
        )

    # Simulate several relaxation times.
    t_max = 6 / Delta

    times = np.linspace(
        0,
        t_max,
        steps + 1
    )

    # Distance between the time-evolution map and its asymptotic limit.
    trace_norms = np.array([
        superoperator_trace_norm_difference(
            L_tilde,
            Phi_inf,
            t
        )
        for t in times
    ])

    # Fit the late-time decay:
    #
    #     ||D(t)||_1 ~ C exp(-r t)
    #
    # so
    #
    #     log ||D(t)||_1 ~ log C - r t.
    fit_mask = (
        (times >= 2.5 / Delta)
        & (times <= 5 / Delta)
        & (trace_norms > 0)
    )

    slope, _ = np.polyfit(
        times[fit_mask],
        np.log(trace_norms[fit_mask]),
        1
    )
    print("\n")
    print(f"N = {n} | Superoperator convergence")
    print("Slow Liouvillian eigenvalue:", lambda_slow)
    print("Liouvillian gap Delta:", Delta)
    print("Fitted asymptotic slope:", slope)
    print("Expected slope -Delta:", -Delta)
    print("\n")

    # Reference decay exp(-Delta t), anchored at a late-time point
    # so that the comparison tests the slope rather than the prefactor.
    t_ref = 3 / Delta

    ref_index = np.argmin(
        np.abs(times - t_ref)
    )

    norm_ref = trace_norms[ref_index]

    reference_decay = norm_ref * np.exp(
        -Delta * (times - times[ref_index])
    )

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        times,
        trace_norms,
        color=color1,
        linewidth=2,
        label=r"$\|e^{\widetilde{\mathcal{L}}t}-\widetilde{\Phi}_{\infty}\|_1$"    )

    ax.plot(
        times,
        reference_decay,
        "--",
        color=color3,
        linewidth=2,
        label=r"$e^{-\Delta t}$"
    )

    ax.set_yscale("log")

    ax.set_xlabel(r"$t$", fontsize=16)

    ax.tick_params(
        axis="both",
        labelsize=14
    )

    ax.legend(
        fontsize=13
    )

    fig.tight_layout()

    fit_name = format_float_for_filename(slope)
    expected_name = format_float_for_filename(-Delta)

    fig.savefig(
        f"superoperator_convergence_N{n}_fit{fit_name}_expected{expected_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

###### Single-excitation reduction ######

DENSE_CUTOFF = 100  # Computational cutoff only, not a physical parameter


def single_excitation_effective_hamiltonian(n, gamma_1=1.0):
    """
    Construct the effective non-Hermitian Hamiltonian
    in the single-excitation sector.

    Basis:
        |1>, |2>, ..., |n>

    H_eff = H1 - i * gamma_1 / 2 * |n><n|

    A dense matrix is used for small systems and a sparse
    tridiagonal matrix for larger systems.
    """

    edge_energy = (n - 3) / 4
    bulk_energy = (n - 5) / 4

    # Diagonal terms from Sz_i Sz_{i+1}
    diagonal = np.full(n, bulk_energy, dtype=complex)
    diagonal[0] = edge_energy
    diagonal[-1] = edge_energy - 1j * gamma_1 / 2

    # Nearest-neighbour hopping from SxSx + SySy
    off_diagonal = np.full(n - 1, 0.5, dtype=complex)

    if n <= DENSE_CUTOFF:
        H_eff = np.diag(diagonal)
        H_eff += np.diag(off_diagonal, k=1)
        H_eff += np.diag(off_diagonal, k=-1)

    else:
        H_eff = diags(
            [off_diagonal, diagonal, off_diagonal],
            offsets=[-1, 0, 1],
            format="csr"
        )

    return H_eff


def reduced_single_excitation_gap(n, gamma_1=1.0):
    """
    Calculate the smallest single-excitation decay rate Gamma_min
    and the Liouvillian gap inferred from it.
    """

    H_eff = single_excitation_effective_hamiltonian(n, gamma_1)

    if n <= DENSE_CUTOFF:
        # Full spectrum for small systems
        eigenvalues = np.linalg.eigvals(H_eff)

    else:
        # Only the eigenvalue with imaginary part closest to zero
        eigenvalues = eigs(
            H_eff,
            k=1,
            which="LI",
            return_eigenvectors=False,
            tol=1e-10,
            maxiter=100000
        )

    # lambda_a = epsilon_a - i Gamma_a / 2
    decay_rates = -2 * np.imag(eigenvalues)
    Gamma_min = np.min(decay_rates)

    # Liouvillian gap inferred from the single-excitation sector
    Delta_reduced = Gamma_min / 2

    return Gamma_min, Delta_reduced


def plot_gap_scaling():
    """
    Calculate the reduced gap up to N = 1000 and fit its
    large-N power-law scaling on a log-log plot.
    """

    n_values = np.concatenate([
        np.arange(2, 21),          # every N from 2 to 20
        np.arange(25, 101, 5),     # every 5 up to 100
        np.arange(120, 1001, 20)   # every 20 up to 1000
    ])

    Delta_values = []

    for n in n_values:
        _, Delta_reduced = reduced_single_excitation_gap(n)
        Delta_values.append(Delta_reduced)

    Delta_values = np.array(Delta_values)

    # Fit Delta(N) ~ N^slope in the large-N regime
    fit_mask = n_values >= 20

    slope, intercept = np.polyfit(
        np.log(n_values[fit_mask]),
        np.log(Delta_values[fit_mask]),
        1
    )

    fit_curve = np.exp(intercept) * n_values**slope

    print("Fitted power-law exponent:", slope)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))

    # Numerical data
    ax.plot(
        n_values,
        Delta_values,
        "o",
        color=color1,
        markersize=4,
        label=r"$\Delta_{\mathrm{red}}(N)$"
    )

    # Power-law fit
    ax.plot(
        n_values,
        fit_curve,
        "--",
        color=color3,
        linewidth=2,
        label=fr"fit: $N^{{{slope:.2f}}}$"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$N$", fontsize=16)
    #ax.set_ylabel(r"$\Delta_{\mathrm{red}}$", fontsize=16)

    ax.tick_params(axis="both", labelsize=14)

    ax.legend(
        fontsize=12
    )

    fig.tight_layout()

    slope_name = format_float_for_filename(slope)

    fig.savefig(
        f"gap_scaling_fit{slope_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


###### Main ######

if __name__ == "__main__":

    # Infidelity time evolution
    # for n in [2, 3, 4, 5]:
    #     plot_infidelity_time_evolution(n)

    # Superoperator convergence
    # for n in [2, 3, 4]:
    #     plot_superoperator_convergence(n)

    # Large-system gap scaling
    plot_gap_scaling()