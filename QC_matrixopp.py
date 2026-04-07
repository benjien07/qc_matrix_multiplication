import time
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from braket.tracking import Tracker
from qiskit_braket_provider import BraketLocalBackend, BraketProvider


# ============================================================
# CONFIG (same pattern as your adder script)
# ============================================================
RUN_MODE = "LOCAL"  # "LOCAL" or "REMOTE"
REMOTE_BACKEND_NAME = "SV1"  # used only when RUN_MODE="REMOTE"

# shots per matrix entry
SHOTS = 2000

# If True, removes barriers before sending to REMOTE (optional)
STRIP_BARRIERS_FOR_REMOTE = True

# Maximum supported INNER dimension for this script
MAX_INNER_DIM = 3


# ============================================================
# Utilities
# ============================================================
def strip_barriers(qc: QuantumCircuit) -> QuantumCircuit:
    pm = PassManager(RemoveBarriers())
    return pm.run(qc)


def pad_to_len_4(v: np.ndarray) -> np.ndarray:
    """
    Pads a real vector of length 1, 2, or 3 to length 4.
    """
    v = np.asarray(v, dtype=float).flatten()
    if len(v) < 1 or len(v) > 3:
        raise ValueError("Only vector lengths 1, 2, or 3 are supported.")
    out = np.zeros(4, dtype=float)
    out[:len(v)] = v
    return out


def normalize_with_norm(v: np.ndarray):
    """
    Normalizes the ORIGINAL vector and also returns its original norm.

    Returns:
      v_unit_orig : normalized original vector
      norm_v      : original norm before normalization
    """
    v = np.asarray(v, dtype=float).flatten()
    n = np.linalg.norm(v)
    if np.isclose(n, 0.0):
        return v.copy(), 0.0
    return v / n, float(n)


def normalize_pad_4(v: np.ndarray):
    """
    Pads to length 4, then normalizes.
    Returns:
      v_unit_4 : normalized padded vector (length 4)
      norm_v   : original vector norm before normalization
    """
    v4 = pad_to_len_4(v)
    n = np.linalg.norm(v4)
    if np.isclose(n, 0.0):
        return v4, 0.0
    return v4 / n, float(n)


def prepare_normalized_padded_state(v: np.ndarray):
    """
    New helper:
      1) normalize the original vector first
      2) pad to length 4
      3) renormalize padded state for clean StatePreparation

    Returns:
      v_state : normalized length-4 state used for amplitude encoding
      norm_v  : original norm of the input vector (used for rescaling later)
    """
    v_unit_orig, norm_v = normalize_with_norm(v)

    if np.isclose(norm_v, 0.0):
        return np.zeros(4, dtype=float), 0.0

    v_pad = pad_to_len_4(v_unit_orig)
    pad_norm = np.linalg.norm(v_pad)

    if np.isclose(pad_norm, 0.0):
        return np.zeros(4, dtype=float), 0.0

    v_state = v_pad / pad_norm
    return v_state, norm_v


def estimate_sign_classically(v: np.ndarray, w: np.ndarray) -> float:
    """
    Swap test gives |<v|w>|^2, so for real matrix multiplication we recover
    the sign classically to form a usable hybrid estimator.
    """
    raw = float(np.dot(np.asarray(v, dtype=float), np.asarray(w, dtype=float)))
    if raw > 0:
        return 1.0
    elif raw < 0:
        return -1.0
    return 0.0


# ============================================================
# Swap-test overlap estimator
# ============================================================
def build_swap_test_circuit(v: np.ndarray, w: np.ndarray) -> QuantumCircuit:
    """
    Build a swap-test circuit for two real vectors v and w of length 1, 2, or 3.

    Encoding:
      - normalize each original vector first
      - pad each vector to length 4
      - amplitude encode onto 2 qubits

    Qubits:
      q0 = ancilla
      q1,q2 = register for |v>
      q3,q4 = register for |w>

    Measurement:
      measure ancilla only

    For normalized states:
      p(ancilla=0) = (1 + |<v|w>|^2) / 2
    """
    v_unit, nv = prepare_normalized_padded_state(v)
    w_unit, nw = prepare_normalized_padded_state(w)

    qc = QuantumCircuit(5, 1)

    # If either vector is zero, no meaningful state prep is possible.
    if np.isclose(nv, 0.0) or np.isclose(nw, 0.0):
        qc.measure(0, 0)
        return qc

    # Prepare the two encoded states
    qc.append(StatePreparation(v_unit), [1, 2])
    qc.append(StatePreparation(w_unit), [3, 4])

    # Swap test
    qc.h(0)
    qc.cswap(0, 1, 3)
    qc.cswap(0, 2, 4)
    qc.h(0)

    qc.measure(0, 0)
    return qc


def estimate_dot_product_swap_test(v: np.ndarray, w: np.ndarray, backend, shots: int, show_circuit: bool = False):
    """
    Hybrid estimator for real dot(v,w) using:
      1) swap test for |<v_unit|w_unit>|^2
      2) classical sign from np.dot(v, w)

    Improvement added here:
      - normalize original vectors BEFORE quantum state prep
      - rescale by original norms AFTER the swap-test estimate

    Returns:
      dot_est
      overlap_abs_est
      overlap_sq_est
      counts
      circuit
    """
    v = np.asarray(v, dtype=float).flatten()
    w = np.asarray(w, dtype=float).flatten()

    _, nv = normalize_with_norm(v)
    _, nw = normalize_with_norm(w)

    # If either vector is zero, the dot product is exactly zero
    if np.isclose(nv, 0.0) or np.isclose(nw, 0.0):
        qc = QuantumCircuit(5, 1)
        qc.measure(0, 0)
        return 0.0, 0.0, 0.0, {"0": shots}, qc

    qc = build_swap_test_circuit(v, w)

    if show_circuit:
        print(qc.draw(output="text"))

    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    p0 = counts.get("0", 0) / shots

    # swap test relation:
    # p0 = (1 + |<v|w>|^2)/2
    overlap_sq_est = max(0.0, 2.0 * p0 - 1.0)
    overlap_abs_est = np.sqrt(overlap_sq_est)

    sign_est = estimate_sign_classically(v, w)

    # rescale by the ORIGINAL norms
    dot_est = sign_est * overlap_abs_est * nv * nw

    return float(dot_est), float(overlap_abs_est), float(overlap_sq_est), counts, qc


# ============================================================
# Matrix multiplication via swap-test-based inner products
# ============================================================
def quantum_matmul_innerdim_upto_3(A, B, backend, shots: int = 2000, show_entry_circuits: bool = False):
    """
    Computes C = A @ B for real matrices with compatible dimensions:
        A has shape (m, n)
        B has shape (n, p)
        C has shape (m, p)

    Restriction:
        inner dimension n must satisfy n <= 3

    For each entry:
        C_ij = row_i(A) dot col_j(B)

    Quantum part:
        swap test estimates |<row_i_unit | col_j_unit>|

    Hybrid reconstruction:
        signed dot product = sign(row_i dot col_j)_classical * magnitude_quantum
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D matrices.")

    m, nA = A.shape
    nB, p = B.shape

    if nA != nB:
        raise ValueError("Inner dimensions must match: A.shape[1] must equal B.shape[0].")

    if nA < 1 or nA > MAX_INNER_DIM:
        raise ValueError("Only inner dimensions 1, 2, or 3 are supported.")

    C = np.zeros((m, p), dtype=float)
    debug = {}

    for i in range(m):
        row = A[i, :]
        for j in range(p):
            col = B[:, j]

            dot_est, overlap_abs_est, overlap_sq_est, counts, qc = estimate_dot_product_swap_test(
                row,
                col,
                backend=backend,
                shots=shots,
                show_circuit=show_entry_circuits
            )

            C[i, j] = dot_est
            debug[(i, j)] = {
                "overlap_abs_est": overlap_abs_est,
                "overlap_sq_est": overlap_sq_est,
                "counts": counts,
                "circuit": qc,
                "classical_sign": estimate_sign_classically(row, col),
                "row_norm": float(np.linalg.norm(row)),
                "col_norm": float(np.linalg.norm(col))
            }

    return C, debug


# ============================================================
# Backend selection (same pattern as your script)
# ============================================================
def get_backend():
    if RUN_MODE == "LOCAL":
        # Local Braket simulators ("braket_sv" or "braket_dm")
        return BraketLocalBackend("braket_sv")
    elif RUN_MODE == "REMOTE":
        provider = BraketProvider()
        return provider.get_backend(REMOTE_BACKEND_NAME)
    else:
        raise ValueError("RUN_MODE must be 'LOCAL' or 'REMOTE'")


# ============================================================
# Main
# ============================================================
def main():
    print("RUN_MODE =", RUN_MODE)
    print("REMOTE_BACKEND_NAME =", REMOTE_BACKEND_NAME)
    print("SHOTS per entry =", SHOTS)

    # -------- USER INPUT (edit these) --------
    # Example: A is 2x3 and B is 3x2
    # This works because the inner dimension is 3.
    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

    B = np.array([[7.0, 8.0],
                  [9.0, 10.0],
                  [11.0, 12.0]])
    # ----------------------------------------

    backend = get_backend()

    tracker = Tracker().start()
    print("Tracker started (tracks QPU tasks when REMOTE on QPU; local sim usually $0).")

    t0 = time.perf_counter()
    C_q, debug = quantum_matmul_innerdim_upto_3(A, B, backend, shots=SHOTS, show_entry_circuits=False)
    t1 = time.perf_counter()

    C_classical = A @ B
    abs_err = np.abs(C_q - C_classical)

    print("\n--- Results ---")
    print("A shape:", A.shape)
    print("B shape:", B.shape)
    print("C shape:", C_q.shape)

    print("\nA:\n", A)
    print("B:\n", B)
    print("\nQuantum-estimated C (swap-test hybrid):\n", np.round(C_q, 6))
    print("\nClassical C:\n", C_classical)
    print("\nAbsolute error |C_q - C_classical|:\n", np.round(abs_err, 6))
    print(f"\nWall time: {(t1 - t0):.3f} s  ({C_q.shape[0] * C_q.shape[1]} circuits total, each with {SHOTS} shots)")

    # Show one debug entry
    key = (0, 0)
    if key in debug:
        print(
            f"\nDebug for C{key}: "
            f"overlap_abs_est={debug[key]['overlap_abs_est']:.6f}, "
            f"overlap_sq_est={debug[key]['overlap_sq_est']:.6f}, "
            f"classical_sign={debug[key]['classical_sign']:.0f}, "
            f"row_norm={debug[key]['row_norm']:.6f}, "
            f"col_norm={debug[key]['col_norm']:.6f}, "
            f"counts={debug[key]['counts']}"
        )

    tracker.stop()
    try:
        print("Estimated QPU cost (USD):", float(tracker.qpu_tasks_cost()))
    except Exception as e:
        print("No QPU cost recorded (likely LOCAL or simulator).")
        print("Tracker info:", e)


if __name__ == "__main__":
    main()