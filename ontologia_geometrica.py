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
    # Using k=3 as per Section 3.3 calculations
    k_rof = 3.0
    phi12 = calcular_rof(P1, P2, k_rof)
    phi13 = calcular_rof(P1, P3, k_rof)
    phi23 = calcular_rof(P2, P3, k_rof)
    print("\n--- 2. ROF Calculations (k_rof = 3) ---")
    print(f"φ(P₁, P₂) (phi12): {phi12:.4f}") # Expected: 3 * 0 = 0
    # Note: Integral of min(mu1, mu3) is 1/18 as per doc. So 3 * 1/18 = 1/6 = 0.1667
    print(f"φ(P₁, P₃) (phi13): {phi13:.4f}") # Expected: 3 * 1/18 = 1/6
    print(f"φ(P₂, P₃) (phi23): {phi23:.4f}") # Expected: 3 * 1/18 = 1/6
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
    print(f"E(Ψ): {energia_psi:.8f}")
    print("-" * 30)

    print("\n--- Verification based on Document Section 3.6 ---")
    # Document states: "To obtain exactly an equilateral triangle of side 1, we must adjust k in ROF to 3.464 for P1-P3 and P2-P3"
    # Let's recalculate phi13 and phi23 with k_adj = sqrt(12)/3 = 2/sqrt(3) * 3/2 = sqrt(3) -- no, this is not k.
    # The document states: "To obtain f(φ,γ)=1, then γφ must be 0. For γ=-0.5, φ must be 0."
    # "But if φ=1/6, and γ=-0.5, then f(1/6, -0.5)=√(1 - (-0.5)(1/6)) = √(1+1/12) = √13/12 = 1.0408"
    # "To make f(φ,γ)=1, where φ = k_adj * (1/18) and γ=-0.5, we need 1 - (-0.5) * k_adj * (1/18) = 1"
    # This means k_adj * (1/36) = 0, so k_adj = 0. This makes phi = 0.
    # This part of the document is confusing or I am misinterpreting how k is adjusted to get d_obj=1.
    # The document's statement "k_adj = 3.464" for phi13 and phi23 seems to be what they used to get d_obj=1.
    # Let's test that:
    k_adj = 3.464 # approx 2*sqrt(3)
    phi13_adj = calcular_rof(P1, P3, k_adj) # k_adj * (1/18)
    phi23_adj = calcular_rof(P2, P3, k_adj) # k_adj * (1/18)
    d_obj13_adj = f_map(phi13_adj, gamma13)
    d_obj23_adj = f_map(phi23_adj, gamma23)
    print(f"Using k_adj = {k_adj} for φ(P1,P3) and φ(P2,P3):")
    print(f"  φ_adj(P₁, P₃): {phi13_adj:.4f} (Expected: {k_adj/18:.4f})")
    print(f"  φ_adj(P₂, P₃): {phi23_adj:.4f} (Expected: {k_adj/18:.4f})")
    print(f"  d_obj_adj(P₁, P₃) with k_adj: {d_obj13_adj:.4f}") # Should be close to 1
    print(f"  d_obj_adj(P₂, P₃) with k_adj: {d_obj23_adj:.4f}") # Should be close to 1

    # If d_obj is 1, then 1 - gamma_ij * phi_adj = 1 => gamma_ij * phi_adj = 0
    # Since gamma_ij = -0.5, this means phi_adj must be 0.
    # So, k_adj must be 0 for phi_adj to be 0.
    # The document text: "Para obtener f(φ,γ)=1, entonces γφ debe ser 0. Como γ=-0.5, entonces φ debe ser 0" (for P1-P3, P2-P3)
    # This contradicts "k_adj = 3.464".
    # The only way for d_obj to be 1 for P1-P3 and P2-P3 (given gamma13=gamma23=-0.5) is if phi13 and phi23 are 0.
    # This would mean k=0 for those specific ROF calculations, which is different from phi12 where k=3 yields phi12=0.

    # Let's assume the target distances [1,1,1] are what the optimization aims for.
    # The initial setup with k=3 for all ROFs gives:
    # d_obj12 = 1.0
    # d_obj13 = 1.0408
    # d_obj23 = 1.0408
    # The energy E_psi is calculated with these.
    # The document's section 3.6 seems to imply that the *goal* is an equilateral triangle (distances are 1),
    # and then it discusses how to adjust 'k' in the ROF to make the *target distances from f_map* also 1.
    # This suggests that the f_map and ROF definitions are primary, and k is a tuning parameter.
    # The exercise as requested is to implement the formulas and then verify. The current output of E_psi
    # will show the discrepancy when k=3 is used uniformly.
    print("\nFinal E(Ψ) with k_rof=3 (as calculated initially): {:.8f}".format(energia_psi))
    # Energy if target distances were all 1.0:
    dist_obj_ideal = np.array([1.0, 1.0, 1.0])
    energia_psi_ideal_target = E_psi(distancias_actuales, dist_obj_ideal)
    print(f"E(Ψ) if target distances were [1,1,1]: {energia_psi_ideal_target:.8f}")
    # This is what the optimization process (not implemented here) would try to achieve by adjusting M_Pi.
    # The current test is about how well the fixed M_Pi (equilateral triangle) match the f_map targets for a fixed k_rof=3.

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
