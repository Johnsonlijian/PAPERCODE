"""
Elastic Buckling Analysis of Prefabricated H-Section Steel Composite Wall-Columns

This module implements the theoretical framework presented in:
"Elastic Buckling Theoretical Model and Validation of Prefabricated 
H-Section Steel Composite Wall-Columns"

Author: Tsinghua University
Date: October 2025
License: CC BY-NC 4.0 (Academic use only)

Requirements:
    numpy >= 1.21.0
    scipy >= 1.7.0
    matplotlib >= 3.4.0 (for visualization)
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs, spsolve
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


# ==================== SECTION 1: SHAPE FUNCTIONS ====================

def ni(x: float, xi: float, xj: float) -> float:
    """Linear Lagrange shape function at node i"""
    return (x - xj) / (xi - xj)

def nj(x: float, xi: float, xj: float) -> float:
    """Linear Lagrange shape function at node j"""
    return (x - xi) / (xj - xi)

def h1(x: float, xi: float, xj: float) -> float:
    """Cubic Hermite shape function H1 (Equation 40)"""
    return -(x - xj)**2 * (2*x - 3*xi + xj) / (xi - xj)**3

def h2(x: float, xi: float, xj: float) -> float:
    """Cubic Hermite shape function H2 (Equation 41)"""
    return (x - xi) * (x - xj)**2 / (xi - xj)**2

def h3(x: float, xi: float, xj: float) -> float:
    """Cubic Hermite shape function H3 (Equation 42)"""
    return (x - xi)**2 * (2*x + xi - 3*xj) / (xi - xj)**3

def h4(x: float, xi: float, xj: float) -> float:
    """Cubic Hermite shape function H4 (Equation 43)"""
    return (x - xi)**2 * (x - xj) / (xi - xj)**2

# First derivatives
def h1p(x: float, xi: float, xj: float) -> float:
    """First derivative of H1"""
    L = xi - xj
    return (-2*(x - xj)**2)/L**3 - (2*(x - xj)*(2*x - 3*xi + xj))/L**3

def h2p(x: float, xi: float, xj: float) -> float:
    """First derivative of H2"""
    L = xi - xj
    return (2*(x - xi)*(x - xj))/L**2 + (x - xj)**2/L**2

def h3p(x: float, xi: float, xj: float) -> float:
    """First derivative of H3"""
    L = xi - xj
    return (2*(x - xi)**2)/L**3 + (2*(x - xi)*(2*x + xi - 3*xj))/L**3

def h4p(x: float, xi: float, xj: float) -> float:
    """First derivative of H4"""
    L = xi - xj
    return (x - xi)**2/L**2 + (2*(x - xi)*(x - xj))/L**2

# Second derivatives
def h1pp(x: float, xi: float, xj: float) -> float:
    """Second derivative of H1"""
    L = xi - xj
    return (-8*(x - xj))/L**3 - (2*(2*x - 3*xi + xj))/L**3

def h2pp(x: float, xi: float, xj: float) -> float:
    """Second derivative of H2"""
    L = xi - xj
    return (2*(x - xi))/L**2 + (4*(x - xj))/L**2

def h3pp(x: float, xi: float, xj: float) -> float:
    """Second derivative of H3"""
    L = xi - xj
    return (8*(x - xi))/L**3 + (2*(2*x + xi - 3*xj))/L**3

def h4pp(x: float, xi: float, xj: float) -> float:
    """Second derivative of H4"""
    L = xi - xj
    return (4*(x - xi))/L**2 + (2*(x - xj))/L**2


# ==================== SECTION 2: SHAPE FUNCTION VECTORS ====================

def Nu0(x: float, xi: float, xj: float) -> np.ndarray:
    """Shape function vector for axial displacement u0"""
    return np.array([ni(x, xi, xj), 0, 0, 0, nj(x, xi, xj), 0, 0, 0])

def Ntheta(x: float, xi: float, xj: float) -> np.ndarray:
    """Shape function vector for rotation θ"""
    return np.array([0, 0, 0, ni(x, xi, xj), 0, 0, 0, nj(x, xi, xj)])

def Nv0(x: float, xi: float, xj: float) -> np.ndarray:
    """Shape function vector for lateral displacement v0"""
    return np.array([0, h1(x, xi, xj), h2(x, xi, xj), 0, 
                     0, h3(x, xi, xj), h4(x, xi, xj), 0])

def Nu0p(x: float, xi: float, xj: float) -> np.ndarray:
    """First derivative of Nu0"""
    L = xi - xj
    return np.array([1/L, 0, 0, 0, -1/L, 0, 0, 0])

def Nthetap(x: float, xi: float, xj: float) -> np.ndarray:
    """First derivative of Ntheta"""
    L = xi - xj
    return np.array([0, 0, 0, 1/L, 0, 0, 0, -1/L])

def Nv0p(x: float, xi: float, xj: float) -> np.ndarray:
    """First derivative of Nv0"""
    return np.array([0, h1p(x, xi, xj), h2p(x, xi, xj), 0,
                     0, h3p(x, xi, xj), h4p(x, xi, xj), 0])

def Nv0pp(x: float, xi: float, xj: float) -> np.ndarray:
    """Second derivative of Nv0"""
    return np.array([0, h1pp(x, xi, xj), h2pp(x, xi, xj), 0,
                     0, h3pp(x, xi, xj), h4pp(x, xi, xj), 0])


# ==================== SECTION 3: ELEMENT STIFFNESS MATRICES ====================

def Ke(xi: float, xj: float, EA: float, EIy: float, EIz: float, 
       GIy: float, GIz: float) -> np.ndarray:
    """
    Compute elastic stiffness matrix using 2-point Gauss integration
    
    Parameters:
        xi, xj: Element node coordinates
        EA: Axial stiffness
        EIy, EIz: Flexural stiffnesses
        GIy, GIz: Torsional stiffnesses
    
    Returns:
        8x8 element stiffness matrix (Equation 44)
    """
    # Gauss integration points and weights
    Xi = np.array([-0.5773502691896258, 0.5773502691896258])
    Wi = np.array([1.0, 1.0])
    
    res = np.zeros((8, 8))
    
    for i in range(2):
        # Transform to physical coordinates
        x = 0.5 * (xj - xi) * Xi[i] + 0.5 * (xi + xj)
        w = 0.5 * (xj - xi) * Wi[i]
        
        # Shape function derivatives
        nu0p = Nu0p(x, xi, xj)
        nv0pp = Nv0pp(x, xi, xj)
        nthetap = Nthetap(x, xi, xj)
        
        # Compute integrand (Equation 29, 33, 35)
        k0 = EA * np.outer(nu0p, nu0p) + \
             EIz * np.outer(nv0pp, nv0pp) + \
             (GIy + GIz) * np.outer(nthetap, nthetap)
        
        res += w * k0
    
    return res


def KG(xi: float, xj: float, Fni: float, Fnj: float, 
       Ji: float, Jj: float) -> np.ndarray:
    """
    Compute geometric stiffness matrix using 2-point Gauss integration
    
    Parameters:
        xi, xj: Element node coordinates
        Fni, Fnj: Axial forces at nodes i, j
        Ji, Jj: Torsional moments at nodes i, j
    
    Returns:
        8x8 geometric stiffness matrix (Equation 25)
    """
    Xi = np.array([-0.5773502691896258, 0.5773502691896258])
    Wi = np.array([1.0, 1.0])
    
    res = np.zeros((8, 8))
    
    for i in range(2):
        x = 0.5 * (xj - xi) * Xi[i] + 0.5 * (xi + xj)
        w = 0.5 * (xj - xi) * Wi[i]
        
        nv0p = Nv0p(x, xi, xj)
        nthetap = Nthetap(x, xi, xj)
        
        # Interpolate internal forces
        Fn = ni(x, xi, xj) * Fni + nj(x, xi, xj) * Fnj
        J = ni(x, xi, xj) * Ji + nj(x, xi, xj) * Jj
        
        k0 = Fn * np.outer(nv0p, nv0p) + J * np.outer(nthetap, nthetap)
        res += w * k0
    
    return res


# ==================== SECTION 4: ASSEMBLY FUNCTIONS ====================

def formnf(nf0: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Form nodal freedom numbering array
    
    Parameters:
        nf0: Initial array with 0 for suppressed DOFs, 1 for active DOFs
    
    Returns:
        nf: Numbered DOF array
        neq: Total number of equations
    """
    nf = nf0.copy()
    nodof, nn = nf.shape
    m = 0
    
    for j in range(nn):
        for i in range(nodof):
            if nf[i, j] != 0:
                m += 1
                nf[i, j] = m
    
    return nf, m


def numtog(num: np.ndarray, nf: np.ndarray) -> np.ndarray:
    """
    Convert element node numbers to global DOF numbers
    
    Parameters:
        num: Element node numbers (length nod)
        nf: Global nodal freedom array (nodof × nn)
    
    Returns:
        g: Global DOF numbers for element (length nod*nodof)
    """
    nod = len(num)
    nodof = nf.shape[0]
    g = np.zeros(nod * nodof, dtype=int)
    
    for i in range(nod):
        k = (i + 1) * nodof
        g[k - nodof:k] = nf[:, num[i]]
    
    return g


class PHSWCModel:
    """
    Prefabricated H-Section Steel Composite Wall-Column Model
    """
    
    def __init__(self, config: Dict):
        """
        Initialize model with configuration parameters
        
        config should contain:
            - numBeam: Number of H-sections
            - H, B: Section dimensions
            - E0: Elastic modulus
            - nu: Poisson's ratio
            - gcoord0: Nodal coordinates for single beam
            - boltNodes: Node numbers where bolts are located
            - kx, ky, kz: Bolt stiffnesses
            - beamSections: List of {H, B} for each beam
        """
        self.config = config
        self._initialize_geometry()
        self._initialize_material()
        
    def _initialize_geometry(self):
        """Setup geometry and connectivity"""
        cfg = self.config
        self.numBeam = cfg['numBeam']
        self.gcoord0 = np.array(cfg['gcoord0'])
        self.nn0 = len(self.gcoord0)
        
        # Replicate coordinates for multiple beams
        self.gcoord = np.tile(self.gcoord0, self.numBeam)
        self.nn = len(self.gcoord)
        
        # Element connectivity
        gnum0 = np.array([[i, i+1] for i in range(self.nn0 - 1)])
        self.gnum = []
        for ibeam in range(self.numBeam):
            self.gnum.append(gnum0 + ibeam * self.nn0)
        self.gnum = np.vstack(self.gnum)
        self.nels = len(self.gnum)
        
        # DOF configuration
        self.nodof = 4  # u0, v0, v0', θ per node
        self.nod = 2    # nodes per element
        self.ndof = self.nod * self.nodof
        
    def _initialize_material(self):
        """Setup material properties"""
        cfg = self.config
        H, B = cfg['H'], cfg['B']
        E0 = cfg['E0']
        nu = cfg.get('nu', 0.3)
        G = E0 / (2 * (1 + nu))
        
        # Section properties
        Iy = (1/12) * B * H**3
        Iz = (1/12) * H * B**3
        A = B * H
        
        EA = E0 * A
        EIy = E0 * Iy
        EIz = E0 * Iz
        GIy = G * Iy
        GIz = G * Iz
        
        # Store as lists for each beam
        self.EAlist = [EA] * self.numBeam
        self.EIylist = [EIy] * self.numBeam
        self.EIzlist = [EIz] * self.numBeam
        self.GIylist = [GIy] * self.numBeam
        self.GIzlist = [GIz] * self.numBeam
        
    def setup_boundary_conditions(self, fixed_nodes: List[int]):
        """
        Setup boundary conditions
        
        Parameters:
            fixed_nodes: List of node numbers to fix all DOFs
        """
        # Initialize with all DOFs active
        nf0 = np.ones((self.nodof, self.nn), dtype=int)
        
        # Suppress DOFs at fixed nodes
        for node in fixed_nodes:
            nf0[:, node] = 0
        
        self.nf, self.neq = formnf(nf0)
        
        # Create global DOF mapping for each element
        self.gg = np.zeros((self.ndof, self.nels), dtype=int)
        for iel in range(self.nels):
            num = self.gnum[iel]
            self.gg[:, iel] = numtog(num, self.nf)
    
    def assemble_stiffness(self):
        """Assemble global elastic stiffness matrix"""
        self.gK = lil_matrix((self.neq, self.neq))
        
        for iel in range(self.nels):
            # Find which beam this element belongs to
            ibeam = iel // (self.nn0 - 1)
            
            # Get material properties
            EA = self.EAlist[ibeam]
            EIy = self.EIylist[ibeam]
            EIz = self.EIzlist[ibeam]
            GIy = self.GIylist[ibeam]
            GIz = self.GIzlist[ibeam]
            
            # Element coordinates
            num = self.gnum[iel]
            xi, xj = self.gcoord[num]
            
            # Compute element matrix
            kma = Ke(xi, xj, EA, EIy, EIz, GIy, GIz)
            
            # Global DOF numbers
            g = self.gg[:, iel]
            
            # Assemble
            for i in range(self.ndof):
                row = g[i]
                if row == 0:
                    continue
                for j in range(self.ndof):
                    col = g[j]
                    if col == 0:
                        continue
                    self.gK[row-1, col-1] += kma[i, j]
        
        self.gK = self.gK.tocsr()
    
    def add_bolt_connections(self):
        """Add bolt spring stiffness contributions"""
        cfg = self.config
        boltNodes = cfg['boltNodes']
        kx = cfg['kx']
        ky = cfg['ky']
        kz = cfg['kz']
        beamSections = cfg['beamSections']
        
        nc = len(boltNodes)
        
        # Get connected nodes for each bolt
        connectedNodes = []
        for ic in range(nc):
            nodes = [boltNodes[ic] + ibeam * self.nn0 
                    for ibeam in range(self.numBeam)]
            connectedNodes.append(nodes)
        
        # Add spring stiffness between adjacent beams
        for ic in range(nc):
            for pair in range(self.numBeam - 1):
                nodei = connectedNodes[ic][pair]
                nodej = connectedNodes[ic][pair + 1]
                
                # Get DOF numbers
                dofu0i, dofv0i, dofv0pi, dofthetai = self.nf[:, nodei]
                dofu0j, dofv0j, dofv0pj, dofthetaj = self.nf[:, nodej]
                
                # X-direction connection
                nvector = np.zeros(self.neq)
                if dofu0i > 0:
                    nvector[dofu0i - 1] = 1
                if dofu0j > 0:
                    nvector[dofu0j - 1] = -1
                if dofv0pi > 0:
                    nvector[dofv0pi - 1] = -0.5 * beamSections[pair][1]
                if dofv0pj > 0:
                    nvector[dofv0pj - 1] = -0.5 * beamSections[pair+1][1]
                
                self.gK += kx[pair] * np.outer(nvector, nvector)
                
                # Y-direction connection
                nvector = np.zeros(self.neq)
                if dofv0i > 0:
                    nvector[dofv0i - 1] = 1
                if dofv0j > 0:
                    nvector[dofv0j - 1] = -1
                if dofthetai > 0:
                    nvector[dofthetai - 1] = -0.5 * beamSections[pair][0]
                if dofthetaj > 0:
                    nvector[dofthetaj - 1] = -0.5 * beamSections[pair+1][0]
                
                self.gK += ky[pair] * np.outer(nvector, nvector)
                
                # Z-direction connection
                nvector = np.zeros(self.neq)
                if dofthetai > 0:
                    nvector[dofthetai - 1] = 1
                if dofthetaj > 0:
                    nvector[dofthetaj - 1] = -1
                if dofv0pi > 0:
                    nvector[dofv0pi - 1] = -0.5 * beamSections[pair][1]
                if dofv0pj > 0:
                    nvector[dofv0pj - 1] = -0.5 * beamSections[pair+1][1]
                
                self.gK += kz[pair] * np.outer(nvector, nvector)
    
    def solve_linear(self, loads: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Solve linear system K·u = R
        
        Parameters:
            loads: Dictionary {node_number: [Fx, Fy, Mz, Mθ]}
        
        Returns:
            sol: Displacement vector
        """
        gR = np.zeros(self.neq)
        
        for node, load in loads.items():
            dofs = self.nf[:, node]
            for i, dof in enumerate(dofs):
                if dof > 0:
                    gR[dof - 1] = load[i]
        
        sol = spsolve(self.gK.tocsr(), gR)
        return sol
    
    def compute_internal_forces(self, sol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute internal forces from displacement solution
        
        Returns:
            FnSol: Axial forces at element ends [nels × 2]
            JSol: Torsional moments at element ends [nels × 2]
        """
        FnSol = np.zeros((self.nels, 2))
        JSol = np.zeros((self.nels, 2))
        
        for iel in range(self.nels):
            ibeam = iel // (self.nn0 - 1)
            EA = self.EAlist[ibeam]
            EIy = self.EIylist[ibeam]
            
            num = self.gnum[iel]
            xi, xj = self.gcoord[num]
            
            # Extract element DOF values
            g = self.gg[:, iel]
            dofVal = np.array([sol[dof-1] if dof > 0 else 0.0 for dof in g])
            
            # Compute forces
            FnSol[iel, 0] = EA * Nu0p(xi, xi, xj) @ dofVal
            FnSol[iel, 1] = EA * Nu0p(xj, xi, xj) @ dofVal
            JSol[iel, 0] = EIy * Nu0p(xi, xi, xj) @ dofVal
            JSol[iel, 1] = EIy * Nu0p(xj, xi, xj) @ dofVal
        
        return FnSol, JSol
    
    def assemble_geometric_stiffness(self, sol: np.ndarray):
        """Assemble geometric stiffness matrix from current solution"""
        FnSol, JSol = self.compute_internal_forces(sol)
        
        self.gKG = lil_matrix((self.neq, self.neq))
        
        for iel in range(self.nels):
            num = self.gnum[iel]
            xi, xj = self.gcoord[num]
            
            Fni, Fnj = FnSol[iel]
            Ji, Jj = JSol[iel]
            
            kma = KG(xi, xj, Fni, Fnj, Ji, Jj)
            
            g = self.gg[:, iel]
            
            for i in range(self.ndof):
                row = g[i]
                if row == 0:
                    continue
                for j in range(self.ndof):
                    col = g[j]
                    if col == 0:
                        continue
                    self.gKG[row-1, col-1] += kma[i, j]
        
        self.gKG = self.gKG.tocsr()
    
    def solve_buckling(self, sol: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Solve buckling eigenvalue problem
        
        Parameters:
            sol: Linear displacement solution
        
        Returns:
            lambda_cr: Critical buckling load factor
            mode: Buckling mode shape
        """
        self.assemble_geometric_stiffness(sol)
        
        # Solve (K - λ·KG)·η = 0
        eigvals, eigvecs = eigs(self.gK, k=1, M=-self.gKG, 
                                which='SM', sigma=0)
        
        lambda_cr = eigvals[0].real
        mode = eigvecs[:, 0].real
        
        return lambda_cr, mode
    
    def extract_displacement(self, sol: np.ndarray, whichBeam: int, 
                            component: int) -> np.ndarray:
        """
        Extract displacement component for visualization
        
        Parameters:
            sol: Solution vector
            whichBeam: Beam number (1-indexed)
            component: 1=u0, 2=v0, 3=v0', 4=θ
        
        Returns:
            data: [x, displacement] pairs
        """
        start_node = (whichBeam - 1) * self.nn0
        end_node = whichBeam * self.nn0
        
        data = []
        for inode in range(start_node, end_node):
            x = self.gcoord[inode]
            dof = self.nf[component - 1, inode]
            val = sol[dof - 1] if dof > 0 else 0.0
            data.append([x, val])
        
        return np.array(data)


# ==================== SECTION 5: EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Configuration
    config = {
        'numBeam': 3,
        'H': 0.1,        # Section height (m)
        'B': 0.2,        # Section width (m)
        'E0': 206e9,     # Elastic modulus (Pa)
        'nu': 0.3,       # Poisson's ratio
        'gcoord0': np.linspace(0, 8.0, 33),  # Node coordinates
        'boltNodes': [4, 9, 14, 19, 24, 29, 32],  # Bolt locations (0-indexed)
        'kx': [1e8, 1e8],  # Axial bolt stiffness (N/m)
        'ky': [1e8, 1e8],  # Lateral bolt stiffness (N/m)
        'kz': [1e8, 1e8],  # Vertical bolt stiffness (N/m)
        'beamSections': [[0.1, 0.05], [0.1, 0.05], [0.1, 0.05]]  # [H, B]
    }
    
    # Initialize model
    model = PHSWCModel(config)
    
    # Setup boundary conditions (fix bottom nodes)
    fixed_nodes = [i * model.nn0 for i in range(model.numBeam)]
    model.setup_boundary_conditions(fixed_nodes)
    
    # Assemble stiffness
    model.assemble_stiffness()
    model.add_bolt_connections()
    
    # Apply loads (axial compression at top)
    F0 = -1000.0  # N
    loads = {}
    for i in range(model.numBeam):
        top_node = (i + 1) * model.nn0 - 1
        loads[top_node] = [F0, 0, 0, 0]
    
    # Solve linear problem
    sol = model.solve_linear(loads)
    
    # Solve buckling
    lambda_cr, mode = model.solve_buckling(sol)
    
    print(f"Critical buckling load factor: {lambda_cr:.2f}")
    print(f"Critical buckling load: {lambda_cr * abs(F0):.2f} N")
    
    # Visualize buckling mode
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, beam in enumerate([1, 2, 3]):
        data = model.extract_displacement(mode, beam, 2)  # v0 displacement
        axes[i].plot(data[:, 0], data[:, 1], 'r-o')
        axes[i].set_title(f'Beam {beam}')
        axes[i].set_xlabel('x (m)')
        axes[i].set_ylabel('v₀ (m)')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('buckling_mode.png', dpi=300)
    print("Buckling mode saved to buckling_mode.png")