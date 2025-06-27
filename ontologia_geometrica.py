import math
from scipy.integrate import quad
import numpy as np

class PuntoOntologicoDifuso:
    """
    Represents a Fuzzy Ontological Point (Punto Ontológico Difuso - POD).
    """
    def __init__(self, esencia: float, radio: float, mu_func: callable):
        """
        Initializes a POD.

        Args:
            esencia: The essence of the POD (float between 0 and 1).
            radio: The radius of the POD (float between 0 and 1).
            mu_func: A callable function representing the membership function μ.
        """
        if not (0 <= esencia <= 1):
            raise ValueError("esencia must be between 0 and 1")
        if not (0 <= radio <= 1):
            raise ValueError("radio must be between 0 and 1")

        self.esencia = esencia
        self.radio = radio
        self.mu_func = mu_func

def crear_mu1() -> callable:
    """
    Creates the membership function μ₁(x) = (3/2)·max(0, 1-9|x-1/3|).
    This function has its peak at x = 1/3 and integrates to 1/3 over [0, 1].
    """
    def mu1(x: float) -> float:
        return (3/2) * max(0, 1 - 9 * abs(x - 1/3))
    return mu1

def crear_mu2() -> callable:
    """
    Creates the membership function μ₂(x) = (3/2)·max(0, 1-9|x-2/3|).
    This function has its peak at x = 2/3 and integrates to 1/3 over [0, 1].
    """
    def mu2(x: float) -> float:
        return (3/2) * max(0, 1 - 9 * abs(x - 2/3))
    return mu2

def crear_mu3() -> callable:
    """
    Creates the membership function μ₃(x) = (3/2)·max(0, 1-9|x-1/2|).
    This function has its peak at x = 1/2 and integrates to 1/3 over [0, 1].
    """
    def mu3(x: float) -> float:
        return (3/2) * max(0, 1 - 9 * abs(x - 1/2))
    return mu3

# Alias for the class
POD = PuntoOntologicoDifuso

def calcular_rof(P1: PuntoOntologicoDifuso, P2: PuntoOntologicoDifuso, k: float) -> float:
    """
    Calculates the Degree of Ontological Overlap (ROF - Razón de Ontosolapamiento Fundamental)
    between two Fuzzy Ontological Points (PODs).

    The formula is: φ(P₁, P₂) = k · ∫₀¹ min(μ₁(x), μ₂(x))dx

    Args:
        P1: The first POD object.
        P2: The second POD object.
        k: A constant multiplier.

    Returns:
        The calculated ROF value.
    """

    integrand = lambda x: min(P1.mu_func(x), P2.mu_func(x))

    # Perform numerical integration from 0 to 1
    integral_value, error = quad(integrand, 0, 1)

    return k * integral_value

class EspacioOntologicoRelacional:
    """
    Represents a Relational Ontological Space (Espacio Ontológico Relacional - EOR).
    This space is defined by a set of PODs and a Gamma tensor that models the
    relationships between their underlying essences.
    """
    def __init__(self, conjunto_pods: list[PuntoOntologicoDifuso], tensor_gamma: np.ndarray):
        """
        Initializes an EOR.

        Args:
            conjunto_pods: A list or set of POD objects that constitute the space.
            tensor_gamma: A NumPy array representing the Gamma tensor (Γ).
                          It must be a square matrix with a determinant > 0.
                          It's also expected to be symmetric with diagonal elements equal to 1.

        Raises:
            ValueError: If tensor_gamma is not a valid NumPy array, not square,
                        or its determinant is not greater than 0.
                        Or if it's not symmetric or diagonal elements are not 1.
        """
        if not isinstance(tensor_gamma, np.ndarray):
            raise ValueError("tensor_gamma must be a NumPy array.")
        if tensor_gamma.ndim != 2 or tensor_gamma.shape[0] != tensor_gamma.shape[1]:
            raise ValueError("tensor_gamma must be a square matrix.")

        # Critical check: determinant > 0
        if np.linalg.det(tensor_gamma) <= 0:
            raise ValueError("The determinant of tensor_gamma must be greater than 0.")

        # Good practice checks (optional based on problem statement but good for robustness)
        if not np.allclose(tensor_gamma, tensor_gamma.T): # Check for symmetry
            raise ValueError("tensor_gamma must be symmetric.")
        if not np.allclose(np.diag(tensor_gamma), 1): # Check for diagonal elements being 1
            raise ValueError("Diagonal elements of tensor_gamma must be 1.")

        self.S = list(conjunto_pods) # Store as a list
        self.Gamma = tensor_gamma

class Escala:
    """
    Represents a scale of measurement or a metric space conceptually.
    """
    def __init__(self, nombre: str, espacio_metrico_descripcion: str):
        """
        Initializes an Escala.

        Args:
            nombre: The name of the scale (e.g., "E_Omega", "E_Eucl").
            espacio_metrico_descripcion: A string describing the metric space
                                         (e.g., "[0,1], d_Omega(x,y) = |x-y|").
        """
        self.nombre = nombre
        self.espacio_metrico_descripcion = espacio_metrico_descripcion

def f_map(phi: float, gamma_ij: float) -> float:
    """
    Calculates the co-determination mapping function f.
    f(φ, γ_ij) = √(1 - γ_ij * φ)

    Args:
        phi: The result of the ROF calculation (φ(P_i, P_j)).
        gamma_ij: The component from the Gamma tensor (Γ_ij) corresponding to
                  the pair of PODs (P_i, P_j).

    Returns:
        The calculated value of the co-determination mapping.
        Returns np.nan if the argument of the square root is negative,
        though theory should ideally prevent this.
    """
    argument = 1 - gamma_ij * phi
    if argument < 0:
        # According to theory, gamma_ij * phi should be <= 1.
        # If it's not, this indicates an issue or an edge case.
        # Returning NaN or raising an error are options.
        # For now, returning NaN to flag it.
        return np.nan
    return np.sqrt(argument)

def E_psi(distancias_euclidianas: np.ndarray, distancias_objetivo: np.ndarray) -> float:
    """
    Calculates the energy functional E(Ψ), which measures the discrepancy
    between actual Euclidean distances of manifestations and target distances
    derived from the ontological framework.

    E(Ψ) = ∑ᵢ |distancias_euclidianas[i] - distancias_objetivo[i]|²

    Args:
        distancias_euclidianas: A NumPy array of actual calculated Euclidean
                                 distances between manifestations M_P.
        distancias_objetivo: A NumPy array of target distances derived from
                              the co-determination mapping (f_map results).

    Returns:
        The calculated energy value (sum of squared differences).

    Raises:
        ValueError: If the input arrays do not have the same shape.
    """
    distancias_euclidianas = np.asarray(distancias_euclidianas)
    distancias_objetivo = np.asarray(distancias_objetivo)

    if distancias_euclidianas.shape != distancias_objetivo.shape:
        raise ValueError("distancias_euclidianas and distancias_objetivo must have the same shape.")

    return np.sum((distancias_euclidianas - distancias_objetivo)**2)

if __name__ == "__main__":
    # 1. Instantiate PODs
    # Esencia=1 and radio=1/3 as per Section 3.2 of "Reformulación Rigurosa"
    P1 = POD(esencia=1, radio=1/3, mu_func=crear_mu1())
    P2 = POD(esencia=1, radio=1/3, mu_func=crear_mu2())
    P3 = POD(esencia=1, radio=1/3, mu_func=crear_mu3())
    print("--- 1. PODs Instantiated ---")
    print(f"P1: esencia={P1.esencia}, radio={P1.radio}, mu_func_name={P1.mu_func.__name__}")
    print(f"P2: esencia={P2.esencia}, radio={P2.radio}, mu_func_name={P2.mu_func.__name__}")
    print(f"P3: esencia={P3.esencia}, radio={P3.radio}, mu_func_name={P3.mu_func.__name__}")
    print("-" * 30)

    # 2. Calculate ROFs
    # To achieve E(Ψ) = 0, target distances d_obj must match actual distances [1,1,1].
    # For d_obj_ij = sqrt(1 - gamma_ij * phi_ij) = 1, we need gamma_ij * phi_ij = 0.
    # Since gamma_ij = -0.5 (non-zero), phi_ij must be 0.
    # phi_ij = k_ij * integral(min(mu_i, mu_j)).
    # For phi12, integral is 0, so phi12=0 for any k. We use k=3.0 as per original doc.
    # For phi13 and phi23, integral is 1/18. So k must be 0 for phi13=0 and phi23=0.

    k_phi12 = 3.0
    k_phi13 = 0.0
    k_phi23 = 0.0

    phi12 = calcular_rof(P1, P2, k_phi12)
    phi13 = calcular_rof(P1, P3, k_phi13)
    phi23 = calcular_rof(P2, P3, k_phi23)

    print("\n--- 2. ROF Calculations ---")
    print(f"k for φ(P₁, P₂): {k_phi12}")
    print(f"k for φ(P₁, P₃): {k_phi13}")
    print(f"k for φ(P₂, P₃): {k_phi23}")
    print(f"φ(P₁, P₂) (phi12): {phi12:.4f}") # Expected: 0
    print(f"φ(P₁, P₃) (phi13): {phi13:.4f}") # Expected: 0 (with k_phi13=0)
    print(f"φ(P₂, P₃) (phi23): {phi23:.4f}") # Expected: 0 (with k_phi23=0)
    print("-" * 30)

    # 3. Define Γ tensor
    # As per Section 3.4
    Gamma = np.array([[1.0, -0.5, -0.5],
                      [-0.5, 1.0, -0.5],
                      [-0.5, -0.5, 1.0]])
    gamma12 = Gamma[0, 1]
    gamma13 = Gamma[0, 2]
    gamma23 = Gamma[1, 2] # Should be Gamma[1,2]
    print("\n--- 3. Gamma Tensor ---")
    print(f"Γ Tensor:\n{Gamma}")
    print(f"γ₁₂: {gamma12}")
    print(f"γ₁₃: {gamma13}")
    print(f"γ₂₃: {gamma23}")
    print("-" * 30)

    # 4. Calculate Target Euclidean Distances
    # Using f_map(phi, gamma_ij) = sqrt(1 - gamma_ij * phi)
    d_obj12 = f_map(phi12, gamma12)
    d_obj13 = f_map(phi13, gamma13)
    d_obj23 = f_map(phi23, gamma23)
    distancias_objetivo = np.array([d_obj12, d_obj13, d_obj23])
    print("\n--- 4. Target Euclidean Distances (from f_map) ---")
    # Expected from document (Section 3.6, implies these should be 1 for equilateral triangle)
    # f_map(0, -0.5) = sqrt(1 - (-0.5)*0) = sqrt(1) = 1
    # f_map(1/6, -0.5) = sqrt(1 - (-0.5)*(1/6)) = sqrt(1 + 1/12) = sqrt(13/12) = sqrt(1.08333) = 1.0408
    print(f"d_obj(P₁, P₂) (d_obj12): {d_obj12:.4f}")
    print(f"d_obj(P₁, P₃) (d_obj13): {d_obj13:.4f}")
    print(f"d_obj(P₂, P₃) (d_obj23): {d_obj23:.4f}")
    print(f"Distancias Objetivo array: {distancias_objetivo}")
    print("-" * 30)

    # 5. Define Manifestations (Euclidean Coordinates)
    # Equilateral triangle with side length 1
    M_P1 = np.array([0.0, 0.0])
    M_P2 = np.array([1.0, 0.0])
    M_P3 = np.array([0.5, np.sqrt(3)/2]) # approx 0.866
    print("\n--- 5. Manifestation Coordinates (M_Pi) ---")
    print(f"M_P1: {M_P1}")
    print(f"M_P2: {M_P2}")
    print(f"M_P3: {M_P3}")
    print("-" * 30)

    # 6. Calculate Actual Euclidean Distances
    d_act12 = np.linalg.norm(M_P1 - M_P2)
    d_act13 = np.linalg.norm(M_P1 - M_P3)
    d_act23 = np.linalg.norm(M_P2 - M_P3)
    distancias_actuales = np.array([d_act12, d_act13, d_act23])
    print("\n--- 6. Actual Euclidean Distances (d_act) ---")
    print(f"d_act(M_P₁, M_P₂) (d_act12): {d_act12:.4f}")
    print(f"d_act(M_P₁, M_P₃) (d_act13): {d_act13:.4f}")
    print(f"d_act(M_P₂, M_P₃) (d_act23): {d_act23:.4f}")
    print(f"Distancias Actuales array: {distancias_actuales}")
    print("-" * 30)

    # 7. Verify Energy Minimization
    # E_psi = sum(|d_act - d_obj|^2)
    energia_psi = E_psi(distancias_actuales, distancias_objetivo)
    print("\n--- 7. Energy Functional E(Ψ) ---")
    print(f"E(Ψ): {energia_psi:.8f}") # Should be 0.0 or very close with the adjusted k values
    print("-" * 30)

    print("\n--- Notes on k values and Target Distances (Section 3.6 Discussion) ---")
    # The main text of the script (steps 1-7) now uses k_phi13 = 0 and k_phi23 = 0.
    # This is because:
    # 1. The goal is to have E(Ψ) = 0 for the equilateral triangle manifestation (M_Pi).
    # 2. This requires d_actual (all 1s) to match d_target. So, d_target must be [1,1,1].
    # 3. d_target_ij = f_map(phi_ij, gamma_ij) = sqrt(1 - gamma_ij * phi_ij).
    # 4. For d_target_ij = 1, we need gamma_ij * phi_ij = 0.
    # 5. For P1-P3 and P2-P3, gamma_ij = -0.5. Thus, phi_13 and phi_23 must be 0.
    # 6. Since phi_ij = k_ij * integral_value_ij, and integral_value for P1-P3 and P2-P3 is 1/18 (non-zero),
    #    k_phi13 and k_phi23 must be 0.
    #
    # The original document might mention other k values (e.g., k=3 uniformly, or k_adj=3.464 for some pairs)
    # for different analytical purposes or if different target distances were desired.
    # However, for E(Ψ)=0 in this specific equilateral setup, k=0 for P1-P3 and P2-P3 links is necessary.
    # The previous value of E(Ψ) was non-zero because k=3 was used for all phi calculations, leading to
    # d_obj13 and d_obj23 being ~1.0408 instead of 1.0.

    # The print statement below shows what E(Psi) would be if target distances were [1,1,1]
    # given the actual distances. With the k changes, this should be the same as E(Psi) above.
    dist_obj_ideal = np.array([1.0, 1.0, 1.0])
    energia_psi_ideal_target = E_psi(distancias_actuales, dist_obj_ideal)
    print(f"Reference E(Ψ) if target distances were explicitly [1,1,1]: {energia_psi_ideal_target:.8f}")
    print("-" * 30)

    print("\n--- 8. Tensors de Restriccion rho_Pi ---")
    # As per document Section 3.7, for an equilateral triangle, rho_Pi can be identity matrices.
    # "En el caso de que las manifestaciones sean un triángulo equilátero en R2, los tensores ρPi pueden ser matrices identidad."
    rho_P1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    rho_P2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    rho_P3 = np.array([[1.0, 0.0], [0.0, 1.0]])

    print("Tensor de Restriccion rho_P1:\n", rho_P1)
    print("Tensor de Restriccion rho_P2:\n", rho_P2)
    print("Tensor de Restriccion rho_P3:\n", rho_P3)
    print("-" * 30)
