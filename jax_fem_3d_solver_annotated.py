#!/usr/bin/env python3
"""
EN: Annotated version of the 3D FEM Solver using JAX-FEM with CPU vs GPU timing.
KR: CPU와 GPU 시간 비교를 포함한 JAX-FEM 기반 3D FEM 솔버의 주석 버전.
"""

# EN: Import core libraries (JAX for accelerated arrays/computation, NumPy for host ops)
# KR: 핵심 라이브러리 임포트 (JAX: 가속 연산/배열, NumPy: 호스트 연산)
import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')  # EN: Hide warnings; KR: 경고 숨김

# EN: Enable 64-bit computations in JAX for higher numerical fidelity
# KR: 수치 정밀도를 위해 JAX 64비트 연산 활성화
jax.config.update('jax_enable_x64', True)

try:
    # EN: JAX-FEM (high-level FEM utilities on top of JAX)
    # KR: JAX 위의 FEM 유틸리티 라이브러리
    import jax_fem
    from jax_fem import *
    print("✓ JAX-FEM successfully imported")
except ImportError:
    print("❌ JAX-FEM not found. Please install with: pip install jax-fem")
    exit(1)


class FEM3DSolver:
    """
    EN: 3D FEM solver class. Owns mesh/material, assembles, applies BCs, solves, and times CPU/GPU.
    KR: 3D FEM 솔버 클래스. 메쉬/재료 저장, 조립, 경계조건, 풀기, CPU/GPU 시간 측정 수행.
    """

    def __init__(self, mesh_size: Tuple[int, int, int] = (10, 10, 10),
                 domain_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        EN: Initialize solver with mesh and domain sizes.
        KR: 메쉬 및 도메인 크기로 솔버를 초기화.
        """
        self.mesh_size = mesh_size      # EN: elements per axis; KR: 축별 요소 개수
        self.domain_size = domain_size  # EN: physical extents; KR: 물리적 도메인 크기
        self.nx, self.ny, self.nz = mesh_size
        self.Lx, self.Ly, self.Lz = domain_size

        # EN: Build mesh immediately so downstream methods can use it
        # KR: 이후 메서드에서 사용 가능하도록 즉시 메쉬 생성
        self.mesh = self._create_mesh()

        # EN: Material properties (linear elasticity)
        # KR: 재료 물성 (선형 탄성)
        self.E = 1e6     # EN: Young's modulus; KR: 영률
        self.nu = 0.3     # EN: Poisson ratio; KR: 포아송비
        self.rho = 1.0    # EN: Density; KR: 밀도

        # EN: Lamé parameters derived from (E, nu)
        # KR: (E, nu)로부터 라메 상수 계산
        self.lambda_lame = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu_lame = self.E / (2 * (1 + self.nu))

        print(f"✓ FEM Solver initialized")
        print(f"  Mesh: {self.nx}×{self.ny}×{self.nz} elements")
        print(f"  Domain: {self.Lx}×{self.Ly}×{self.Lz}")
        print(f"  Material: E={self.E}, ν={self.nu}")

    def _create_mesh(self):
        """
        EN: Create structured 3D hexahedral mesh (nodes + connectivity).
        KR: 구조적 3D 헥사 요소 메쉬 생성 (노드 + 연결성).
        """
        print("Creating 3D mesh...")

        # EN: Build 1D grids with NumPy, then convert to JAX (compatibility-friendly)
        # KR: NumPy로 1D 격자 생성 후 JAX 배열로 변환 (호환성 우회)
        x = jnp.array(np.linspace(0, self.Lx, self.nx + 1))
        y = jnp.array(np.linspace(0, self.Ly, self.ny + 1))
        z = jnp.array(np.linspace(0, self.Lz, self.nz + 1))

        # EN: Create 3D grid of node coordinates
        # KR: 3D 노드 좌표 격자 생성
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

        # EN: Connectivity for 8-node hexahedral elements
        # KR: 8노드 헥사 요소 연결성 생성
        elements = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    # EN: Compute node indices for the current cell
                    # KR: 현재 셀의 노드 인덱스 계산
                    n1 = i + j * (self.nx + 1) + k * (self.nx + 1) * (self.ny + 1)
                    n2 = (i + 1) + j * (self.nx + 1) + k * (self.nx + 1) * (self.ny + 1)
                    n3 = (i + 1) + (j + 1) * (self.nx + 1) + k * (self.nx + 1) * (self.ny + 1)
                    n4 = i + (j + 1) * (self.nx + 1) + k * (self.nx + 1) * (self.ny + 1)
                    n5 = i + j * (self.nx + 1) + (k + 1) * (self.nx + 1) * (self.ny + 1)
                    n6 = (i + 1) + j * (self.nx + 1) + (k + 1) * (self.nx + 1) * (self.ny + 1)
                    n7 = (i + 1) + (j + 1) * (self.nx + 1) + (k + 1) * (self.nx + 1) * (self.ny + 1)
                    n8 = i + (j + 1) * (self.nx + 1) + (k + 1) * (self.nx + 1) * (self.ny + 1)
                    elements.append([n1, n2, n3, n4, n5, n6, n7, n8])

        elements = jnp.array(elements)

        print(f"  Nodes: {len(points)}")
        print(f"  Elements: {len(elements)}")

        return {
            'points': points,
            'elements': elements,
            'num_nodes': len(points),
            'num_elements': len(elements)
        }

    def compute_stiffness_matrix_cpu(self) -> jnp.ndarray:
        """
        EN: Assemble global stiffness matrix K on CPU (dense, demo-style).
        KR: CPU에서 전역 강성행렬 K 조립 (밀집, 데모용 구현).
        """
        print("Computing stiffness matrix on CPU...")
        start_time = time.time()

        num_nodes = self.mesh['num_nodes']     # EN: total nodes; KR: 전체 노드 수
        num_dofs = 3 * num_nodes               # EN: 3 DOFs per node; KR: 노드당 3자유도
        K = jnp.zeros((num_dofs, num_dofs))    # EN/KR: 전역 강성행렬 초기화

        # EN: Loop over elements → element stiffness → assemble into K
        # KR: 요소 반복 → 요소 강성 계산 → 전역 K에 조립
        for elem_idx in range(self.mesh['num_elements']):
            elem_nodes = self.mesh['elements'][elem_idx]
            elem_coords = self.mesh['points'][elem_nodes]
            Ke = self._compute_element_stiffness(elem_coords)

            # EN: Add 24x24 Ke blocks into K with DOF mapping
            # KR: 24x24 Ke 블록을 DOF 매핑으로 K에 더함
            for i in range(8):
                for j in range(8):
                    for dof_i in range(3):
                        for dof_j in range(3):
                            global_i = 3 * elem_nodes[i] + dof_i
                            global_j = 3 * elem_nodes[j] + dof_j
                            K = K.at[global_i, global_j].add(Ke[3*i + dof_i, 3*j + dof_j])

        cpu_time = time.time() - start_time
        print(f"  CPU time: {cpu_time:.4f} seconds")
        return K, cpu_time

    def compute_stiffness_matrix_gpu(self) -> jnp.ndarray:
        """
        EN: Assemble global stiffness matrix K on GPU using JIT and vmap.
        KR: JIT 및 vmap을 이용해 GPU에서 전역 강성행렬 K 조립.
        """
        print("Computing stiffness matrix on GPU (CUDA)...")
        start_time = time.time()

        @jax.jit  # EN: JIT-compile for GPU; KR: GPU 실행을 위한 JIT 컴파일
        def compute_k_gpu():
            num_nodes = self.mesh['num_nodes']
            num_dofs = 3 * num_nodes
            K = jnp.zeros((num_dofs, num_dofs))

            # EN: Vectorize per-element Ke computation
            # KR: 요소별 Ke 계산을 벡터화
            def process_element(elem_idx):
                elem_nodes = self.mesh['elements'][elem_idx]
                elem_coords = self.mesh['points'][elem_nodes]
                Ke = self._compute_element_stiffness(elem_coords)
                return Ke, elem_nodes

            Ke_all, elem_nodes_all = jax.vmap(process_element)(jnp.arange(self.mesh['num_elements']))

            # EN: Assemble Ke blocks to K; still loops but JITed
            # KR: Ke 블록을 K에 조립; 루프지만 JIT로 가속
            for elem_idx in range(self.mesh['num_elements']):
                Ke = Ke_all[elem_idx]
                elem_nodes = elem_nodes_all[elem_idx]
                for i in range(8):
                    for j in range(8):
                        for dof_i in range(3):
                            for dof_j in range(3):
                                global_i = 3 * elem_nodes[i] + dof_i
                                global_j = 3 * elem_nodes[j] + dof_j
                                K = K.at[global_i, global_j].add(Ke[3*i + dof_i, 3*j + dof_j])
            return K

        K = compute_k_gpu()  # EN/KR: GPU 실행
        gpu_time = time.time() - start_time
        print(f"  GPU time: {gpu_time:.4f} seconds")
        return K, gpu_time

    def _compute_element_stiffness(self, coords: jnp.ndarray) -> jnp.ndarray:
        """
        EN: Toy 8-node hexahedron element stiffness (no integration, for demo).
        KR: 데모용 8노드 헥사 요소 강성 (수치적분 없음).
        """
        # EN: Approximate element size and volume from corner nodes
        # KR: 코너 노드로 요소 크기/부피 근사
        hx = coords[1, 0] - coords[0, 0]
        hy = coords[3, 1] - coords[0, 1]
        hz = coords[4, 2] - coords[0, 2]
        volume = hx * hy * hz

        Ke = jnp.zeros((24, 24))  # EN/KR: 8 nodes × 3 DOFs = 24

        # EN: Diagonal terms with mu, weak coupling with lambda (illustrative)
        # KR: 대각(mu)과 약한 결합(lambda)을 단순 반영 (예시용)
        for i in range(24):
            for j in range(24):
                if i == j:
                    Ke = Ke.at[i, j].set(self.mu_lame * volume)
                elif abs(i - j) == 3:
                    Ke = Ke.at[i, j].set(self.lambda_lame * volume * 0.1)
        return Ke

    def solve_linear_system_cpu(self, K: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        """
        EN: Dense direct solve on CPU for Ku=f.
        KR: CPU에서 밀집 직접해법으로 Ku=f 풀기.
        """
        print("Solving linear system on CPU...")
        start_time = time.time()
        u = jnp.linalg.solve(K, f)
        cpu_time = time.time() - start_time
        print(f"  CPU solve time: {cpu_time:.4f} seconds")
        return u, cpu_time

    def solve_linear_system_gpu(self, K: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        """
        EN: Dense direct solve on GPU (JIT-compiled) for Ku=f.
        KR: GPU에서 JIT로 Ku=f를 밀집 직접해법으로 풀기.
        """
        print("Solving linear system on GPU (CUDA)...")
        start_time = time.time()

        @jax.jit
        def solve_gpu(K, f):
            return jnp.linalg.solve(K, f)

        u = solve_gpu(K, f)
        gpu_time = time.time() - start_time
        print(f"  GPU solve time: {gpu_time:.4f} seconds")
        return u, gpu_time

    def create_load_vector(self) -> jnp.ndarray:
        """
        EN: Build RHS load vector with a vertical point load at top center.
        KR: 상단 중앙에 수직 점하중을 적용한 우변 벡터 생성.
        """
        num_dofs = 3 * self.mesh['num_nodes']
        f = jnp.zeros(num_dofs)

        center_x, center_y, center_z = self.Lx / 2, self.Ly / 2, self.Lz
        points = self.mesh['points']
        distances = jnp.sqrt((points[:, 0] - center_x) ** 2 +
                             (points[:, 1] - center_y) ** 2 +
                             (points[:, 2] - center_z) ** 2)
        closest_node = jnp.argmin(distances)  # EN/KR: 가장 가까운 노드

        f = f.at[3 * closest_node + 2].set(1000.0)  # EN: Fz; KR: 수직 하중
        return f

    def apply_boundary_conditions(self, K: jnp.ndarray, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        EN: Fix all DOFs on the bottom (z=0) by row/column zeroing and unit diagonal.
        KR: 바닥(z=0) 모든 자유도를 고정(행/열 0, 대각 1)하여 경계조건 적용.
        """
        points = self.mesh['points']
        bottom_nodes = jnp.where(points[:, 2] < 1e-6)[0]

        for node in bottom_nodes:
            for dof in range(3):
                dof_idx = 3 * node + dof
                K = K.at[dof_idx, :].set(0.0)
                K = K.at[:, dof_idx].set(0.0)
                K = K.at[dof_idx, dof_idx].set(1.0)
                f = f.at[dof_idx].set(0.0)
        return K, f

    def run_benchmark(self) -> Dict[str, Any]:
        """
        EN: Orchestrate CPU vs GPU assembly+solve and compute speedups.
        KR: CPU/GPU 조립+해석을 실행하고 스피드업을 계산.
        """
        print("\n" + "=" * 60)
        print("3D FEM SOLVER BENCHMARK: CPU vs GPU (CUDA)")
        print("=" * 60)

        print("\n1. Creating load vector...")
        f = self.create_load_vector()

        print("\n2. Matrix Assembly:")
        K_cpu, assembly_cpu_time = self.compute_stiffness_matrix_cpu()
        K_gpu, assembly_gpu_time = self.compute_stiffness_matrix_gpu()

        print("\n3. Applying boundary conditions...")
        K_cpu, f_cpu = self.apply_boundary_conditions(K_cpu, f)
        K_gpu, f_gpu = self.apply_boundary_conditions(K_gpu, f)

        print("\n4. Linear System Solution:")
        u_cpu, solve_cpu_time = self.solve_linear_system_cpu(K_cpu, f_cpu)
        u_gpu, solve_gpu_time = self.solve_linear_system_gpu(K_gpu, f_gpu)

        total_cpu_time = assembly_cpu_time + solve_cpu_time
        total_gpu_time = assembly_gpu_time + solve_gpu_time

        results = {
            'assembly_cpu_time': assembly_cpu_time,
            'assembly_gpu_time': assembly_gpu_time,
            'solve_cpu_time': solve_cpu_time,
            'solve_gpu_time': solve_gpu_time,
            'total_cpu_time': total_cpu_time,
            'total_gpu_time': total_gpu_time,
            'speedup_assembly': assembly_cpu_time / assembly_gpu_time if assembly_gpu_time > 0 else float('inf'),
            'speedup_solve': solve_cpu_time / solve_gpu_time if solve_gpu_time > 0 else float('inf'),
            'speedup_total': total_cpu_time / total_gpu_time if total_gpu_time > 0 else float('inf'),
            'mesh_size': self.mesh_size,
            'num_nodes': self.mesh['num_nodes'],
            'num_elements': self.mesh['num_elements']
        }

        print("\n" + "=" * 60)
        print("PERFORMANCE RESULTS")
        print("=" * 60)
        print(f"Mesh: {self.nx}×{self.ny}×{self.nz} elements")
        print(f"Nodes: {self.mesh['num_nodes']:,}")
        print(f"Elements: {self.mesh['num_elements']:,}")
        print(f"DOFs: {3 * self.mesh['num_nodes']:,}")
        print()
        print("Assembly Time:")
        print(f"  CPU:  {assembly_cpu_time:.4f} seconds")
        print(f"  GPU:  {assembly_gpu_time:.4f} seconds")
        print(f"  Speedup: {results['speedup_assembly']:.2f}x")
        print()
        print("Solve Time:")
        print(f"  CPU:  {solve_cpu_time:.4f} seconds")
        print(f"  GPU:  {solve_gpu_time:.4f} seconds")
        print(f"  Speedup: {results['speedup_solve']:.2f}x")
        print()
        print("Total Time:")
        print(f"  CPU:  {total_cpu_time:.4f} seconds")
        print(f"  GPU:  {total_gpu_time:.4f} seconds")
        print(f"  Speedup: {results['speedup_total']:.2f}x")

        return results

    def plot_results(self, results: Dict[str, Any]):
        """
        EN: Plot CPU vs GPU timings as a bar chart.
        KR: CPU/GPU 시간을 막대그래프로 시각화.
        """
        categories = ['Assembly', 'Solve', 'Total']
        cpu_times = [results['assembly_cpu_time'], results['solve_cpu_time'], results['total_cpu_time']]
        gpu_times = [results['assembly_gpu_time'], results['solve_gpu_time'], results['total_gpu_time']]

        x = jnp.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, cpu_times, width, label='CPU', alpha=0.8)
        bars2 = ax.bar(x + width/2, gpu_times, width, label='GPU (CUDA)', alpha=0.8)

        ax.set_xlabel('Operation')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('3D FEM Solver: CPU vs GPU Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}s', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('/Users/hyunjilee/Desktop/LAB Prof.Jinhui/MHIT36-main/살려줘/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✓ Performance plot saved to: performance_comparison.png")


def check_cuda_status():
    """
    EN: Detect CUDA via nvidia-smi (if available) and list JAX devices.
    KR: nvidia-smi(가능 시)로 CUDA 확인, JAX 디바이스 목록 확인.
    """
    print("Checking CUDA and GPU status...")
    cuda_version = None
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            cuda_version = result.stdout.strip()
    except:
        pass

    devices = jax.devices()
    gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]

    print(f"  CUDA Version: {cuda_version if cuda_version else 'Not detected'}")
    print(f"  JAX Backend: {jax.default_backend()}")
    print(f"  JAX Devices: {devices}")
    print(f"  GPU Devices: {len(gpu_devices)} detected")

    return len(gpu_devices) > 0, cuda_version


def main():
    """
    EN: Run benchmarks on multiple mesh sizes and plot final results.
    KR: 여러 메쉬 크기에 대해 벤치마크를 수행하고 최종 결과를 시각화.
    """
    print("3D FEM Solver using JAX-FEM")
    print("CPU vs GPU (CUDA) Performance Comparison")
    print("="*50)

    has_gpu, cuda_version = check_cuda_status()
    if not has_gpu:
        print("\n⚠️  No GPU detected. Running CPU-only benchmark.")
        print("   For GPU acceleration, install CUDA and JAX with CUDA support.")
        print("   See CUDA_INSTALLATION_GUIDE.md for details.")
    else:
        print(f"\n✓ GPU detected with CUDA {cuda_version}")
        print("   GPU acceleration will be used for comparison.")

    mesh_sizes = [(5, 5, 5), (8, 8, 8), (10, 10, 10)]
    all_results = []
    for mesh_size in mesh_sizes:
        print(f"\n{'='*60}")
        print(f"Testing mesh size: {mesh_size}")
        print(f"{'='*60}")
        solver = FEM3DSolver(mesh_size=mesh_size)
        results = solver.run_benchmark()
        all_results.append(results)
        if mesh_size == mesh_sizes[-1]:
            solver.plot_results(results)

    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Mesh Size':<15} {'Nodes':<8} {'Elements':<10} {'CPU Total':<12} {'GPU Total':<12} {'Speedup':<10}")
    print("-" * 80)
    for i, results in enumerate(all_results):
        mesh_str = f"{mesh_sizes[i][0]}×{mesh_sizes[i][1]}×{mesh_sizes[i][2]}"
        print(f"{mesh_str:<15} {results['num_nodes']:<8} {results['num_elements']:<10} "
              f"{results['total_cpu_time']:<12.4f} {results['total_gpu_time']:<12.4f} "
              f"{results['speedup_total']:<10.2f}x")

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETED!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


