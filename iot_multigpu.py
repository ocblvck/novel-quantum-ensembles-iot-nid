#!/usr/bin/env python3
"""
Quantum ML for IoT - COMPLETE GPU-OPTIMIZED VERSION WITH ALL MODELS + ENSEMBLE METHODS
Including QSVC, QNN, Quantum Random Forest, and Quantum Ensemble
Multi-GPU Parallel Processing with Automatic Hardware Detection
NO CPU FALLBACKS - Pure GPU Acceleration for Quantum Simulations

Author: @ocblvck
Date: 2025-01-28 18:32:44 UTC
"""

# Standard library imports
from typing import Optional, List, Dict, Tuple, Any, TYPE_CHECKING
import copy
import os
import sys
import pickle
import threading
import time
import io
import json
import uuid
import logging
import queue
import inspect
from collections import defaultdict, OrderedDict, deque
from datetime import datetime
from pathlib import Path
import argparse
import multiprocessing as mp
from multiprocessing import Queue as MPQueue, Process
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED

# TYPE_CHECKING imports for type hints only
if TYPE_CHECKING:
    from qiskit import QuantumCircuit

# Third-party imports
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Optional imports
try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

try:
    import gc
except ImportError:
    gc = None  # type: ignore

try:
    from multiprocessing import shared_memory  # type: ignore[attr-defined]
    SHARED_MEMORY_AVAILABLE = True
except ImportError:
    shared_memory = None  # type: ignore
    SHARED_MEMORY_AVAILABLE = False

try:
    import dill  # type: ignore
    DILL_AVAILABLE = True
except ImportError:
    dill = None  # type: ignore
    DILL_AVAILABLE = False

FAILED_QPY_VALIDATIONS: set[str] = set()

# Initialize logger
logger = logging.getLogger(__name__)

# Module name for Ray actors
_MODULE_IMPORT_NAME = __name__ if __name__ != '__main__' else None

# Setup paths
GPU_LOGS_DIR = Path("gpu_logs")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

# Shared registries populated by the main process and read by workers
_CENTRAL_QPY_REGISTRY: Optional[Dict[str, Any]] = None
_RESULT_QUEUE: Optional[Any] = None

# Check for pynvml availability
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    pynvml = None  # type: ignore
    NVML_AVAILABLE = False

# Check if Ray is available
try:
    import ray  # type: ignore
    RAY_AVAILABLE = True
except Exception:
    ray = None  # type: ignore
    RAY_AVAILABLE = False

# Always attempt to import core Qiskit symbols so downstream classes compile
try:
    from qiskit import QuantumCircuit, transpile  # type: ignore
    from qiskit.circuit import ParameterVector, Parameter  # type: ignore
    from qiskit.circuit.library import (  # type: ignore
        ZZFeatureMap,
        PauliFeatureMap,
        ZFeatureMap,
        RealAmplitudes,
        TwoLocal,
        EfficientSU2,
    )
    from qiskit_aer import AerSimulator  # type: ignore

    try:
        from qiskit_machine_learning.algorithms import QSVC, VQC  # type: ignore
        from qiskit_machine_learning.kernels import FidelityQuantumKernel  # type: ignore
    except Exception:
        QSVC = VQC = FidelityQuantumKernel = None

    try:
        from qiskit.primitives import Sampler as PrimitiveSampler, Estimator  # type: ignore
        from qiskit_machine_learning.kernels.utils import get_fidelity_circuit  # type: ignore
    except Exception:
        PrimitiveSampler = None
        get_fidelity_circuit = None

    try:
        from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, PegasosQSVC  # type: ignore
        NN_CLASSIFIER_AVAILABLE = True
        PEGASOS_AVAILABLE = PegasosQSVC is not None
    except Exception:
        NeuralNetworkClassifier = None
        PegasosQSVC = None
        NN_CLASSIFIER_AVAILABLE = False
        PEGASOS_AVAILABLE = False

    try:
        from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN  # type: ignore
        QNN_AVAILABLE = True
    except Exception:
        EstimatorQNN = SamplerQNN = None
        QNN_AVAILABLE = False

    try:
        from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM  # type: ignore
    except Exception:
        COBYLA = SPSA = ADAM = None

    try:
        from qiskit.primitives import Sampler  # type: ignore
        from qiskit_aer.primitives import Sampler as AerSampler, Estimator as AerEstimator  # type: ignore
    except Exception:
        Sampler = Estimator = AerSampler = AerEstimator = None
except Exception:
    QuantumCircuit = None  # type: ignore
    transpile = None  # type: ignore


def _primitive_accepts_backend_kwarg(primitive_cls) -> bool:
    """Return True if the primitive constructor exposes a `backend` kwarg."""
    if primitive_cls is None:
        return False
    try:
        sig = inspect.signature(primitive_cls.__init__)
    except (TypeError, ValueError):
        return False
    return 'backend' in sig.parameters


AER_ESTIMATOR_ACCEPTS_BACKEND = _primitive_accepts_backend_kwarg(globals().get('AerEstimator'))
AER_SAMPLER_ACCEPTS_BACKEND = _primitive_accepts_backend_kwarg(globals().get('AerSampler'))

# ============================================================================
# RAY DISTRIBUTED COMPUTING HELPERS
# ============================================================================

def _ensure_ray_initialized(address: Optional[str] = None):
    """Initialize Ray cluster connection for multi-node distributed computing.
    
    If address is provided, connects to existing Ray cluster.
    Otherwise, initializes a new local Ray instance.
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not available. Install with: pip install 'ray[default]'")
    
    try:
        # Check if Ray is already initialized
        ray.get_runtime_context()
        logger.info("Ray is already initialized")
        return
    except Exception:
        pass
    
    # Initialize Ray
    init_kwargs = {}
    
    if address:
        # Connect to existing cluster (multi-node setup)
        logger.info(f"Connecting to Ray cluster at: {address}")
        init_kwargs['address'] = address
        init_kwargs['ignore_reinit_error'] = True
    else:
        # Start local Ray instance
        logger.info("Initializing local Ray instance")
        init_kwargs['ignore_reinit_error'] = True
        # Auto-detect GPUs
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                num_gpus = len([l for l in result.stdout.split('\n') if l.strip()])
                init_kwargs['num_gpus'] = num_gpus
                logger.info(f"Detected {num_gpus} GPUs for local Ray instance")
        except Exception as e:
            logger.warning(f"Could not auto-detect GPUs: {e}")
    
    try:
        ray.init(**init_kwargs)
        logger.info("✅ Ray initialized successfully")
        
        # Log cluster information
        try:
            resources = ray.cluster_resources()
            logger.info(f"Ray cluster resources: {resources}")
            nodes = ray.nodes()
            logger.info(f"Ray cluster nodes: {len(nodes)}")
            for i, node in enumerate(nodes):
                alive = node.get('Alive', False)
                node_id = node.get('NodeID', 'unknown')
                resources = node.get('Resources', {})
                gpus = int(resources.get('GPU', 0))
                logger.info(f"  Node {i}: ID={node_id}, Alive={alive}, GPUs={gpus}")
        except Exception as e:
            logger.warning(f"Could not fetch Ray cluster info: {e}")
            
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        raise


def _ray_list_alive_gpu_nodes() -> List[Dict[str, Any]]:
    """List all alive Ray nodes that have GPUs.
    
    Returns a list of node info dictionaries containing NodeID, Resources, etc.
    This is used to determine if we have a true multi-node cluster.
    """
    if not RAY_AVAILABLE:
        return []
    
    try:
        nodes = ray.nodes()
        gpu_nodes = []
        
        for node in nodes:
            # Check if node is alive
            if not node.get('Alive', False):
                continue
            
            # Check if node has GPUs
            resources = node.get('Resources', {})
            gpu_count = int(resources.get('GPU', 0))
            
            if gpu_count > 0:
                gpu_nodes.append(node)
        
        logger.debug(f"Found {len(gpu_nodes)} alive GPU nodes in Ray cluster")
        return gpu_nodes
        
    except Exception as e:
        logger.warning(f"Failed to list Ray GPU nodes: {e}")
        return []


def _ray_get_module_attr(attr_name: str, default=None):
    """Helper to get module attribute for Ray remote functions."""
    try:
        return globals().get(attr_name, default)
    except Exception:
        return default


def _partition_indices(total: int, num_partitions: int) -> List[Tuple[int, int]]:
    """Partition a range from 0 to total into num_partitions balanced chunks.
    
    Returns list of (start, end) tuples for each partition.
    """
    if num_partitions <= 0 or total <= 0:
        return []
    
    chunk_size = max(1, total // num_partitions)
    partitions = []
    
    for i in range(num_partitions):
        start = i * chunk_size
        # Last partition gets any remainder
        end = total if i == num_partitions - 1 else (i + 1) * chunk_size
        if start < total:
            partitions.append((start, end))
    
    return partitions


# ============================================================================
# OPTIMIZED CONFIGURATION FOR LARGE-SCALE EXPERIMENTS
# ============================================================================

# Model-specific Nyström configuration (preserves model differentiation)
NYSTROM_CONFIG = {
    'QSVC_Standard': {'use_nystrom': False, 'landmarks': 2000},  # Use full kernel by default (Nyström only for large datasets)
    'QSVC_Precomputed': {'use_nystrom': False, 'landmarks': None},  # Full kernel (no Nyström)
    'QSVC_Callable': {'use_nystrom': False, 'landmarks': None},  # Computes on-demand
    'PegasosQSVC': {'use_nystrom': False, 'landmarks': None},  # Stochastic (no Nyström)
}

# GPU optimization parameters
MAX_PARALLEL_GPUS = 8  # Maximum GPUs for parallel kernel chunks
KERNEL_CHUNK_SIZE = 1024  # Rows per chunk for kernel computation (increased to reduce transfers)

# Fidelity kernel batching safeguards
FIDELITY_COMPLEX_BYTES = 8  # bytes per amplitude after downcasting statevectors to complex64
FIDELITY_MIN_CIRCUITS = 1  # allow single-circuit fallback when GPU memory is tight
FIDELITY_MAX_CIRCUITS = 2  # CRITICAL: baseline cap for 22+ qubit runs to avoid segfaults
FIDELITY_TARGET_BATCH_MEMORY_MB = 128  # aim for sub-128MB of statevector memory per batch
FIDELITY_DEVICE_RESERVE_MB = 2048  # leave at least 2GB free on the device
FIDELITY_MEMORY_SAFETY = 1.8  # safety multiplier for transient allocations (increased)
FIDELITY_MIN_TARGET_MB = 128  # minimal effective memory budget when estimates are low
GPU_MEMORY_RESERVE_GB = 2  # Reserve GPU memory to prevent OOM
AUTOTUNE_GPU_MEMORY_FRACTION = 0.6  # Fraction of free GPU memory for autotune sizing
AUTOTUNE_MIN_BUDGET_GB = 6  # Lower bound when NVML info unavailable
AUTOTUNE_MAX_BUDGET_GB = 32  # Upper bound to avoid overcommitting very large GPUs
RESULT_QUEUE_POLL_SECONDS = 30  # Interval to poll for worker results when running in parallel
KERNEL_MAX_IDLE_SECONDS = 1200  # Max idle time without progress before considering the run stalled
# Cache sizing limits (keeps memory usage bounded)
# Reduced defaults for high-qubit production runs; these can be tuned via CLI or environment
MAX_SIMULATOR_CACHE_SIZE = 2
MAX_FIDELITY_KERNEL_CACHE_SIZE = 4
MAX_MODEL_KERNEL_CACHE_ENTRIES = 8
# Safety thresholds for very large experiments
LARGE_QUBIT_THRESHOLD = 20  # Above this, kernel objects are extremely large on-device
FULL_KERNEL_MAX_SAMPLES = 8000  # Above this, avoid computing/storing full NxN kernel

# Transpile optimization level used during central precompile (can be overridden by CLI)
TRANSPILE_OPT_LEVEL = 2

# Known circuits that trigger QPY ParameterVector bugs (skip serialization)
KNOWN_QPY_UNSTABLE_KEYS: set[str] = {
    'ZZ_reps1',
    'ZZ_reps2',
    'Pauli_reps1',
    'Pauli_reps2',
}
_WARNED_QPY_SKIPS: set[str] = set()

# Approximation / distributed runtime configuration (overridden via CLI or env)
APPROXIMATION_MODE = (os.environ.get('IOT_APPROX_MODE', 'gpu_strict') or 'gpu_strict').strip().lower()
CIRCUIT_CUT_PARTITIONS = max(0, int(os.environ.get('IOT_CIRCUIT_CUTS', '1') or 1))
MIN_DISTRIBUTED_GPU_NODES = max(1, int(os.environ.get('IOT_MIN_GPU_NODES', '2') or 2))
DISTRIBUTED_STRATEGY = os.environ.get('IOT_DISTRIBUTED_MODE', 'off').strip().lower()
DISTRIBUTED_ADDRESS = (os.environ.get('IOT_DISTRIBUTED_ADDRESS') or '').strip() or None
if not DISTRIBUTED_ADDRESS:
    env_ray_addr = os.environ.get('RAY_ADDRESS')
    if env_ray_addr:
        DISTRIBUTED_ADDRESS = env_ray_addr.strip() or None

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {'1', 'true', 'yes', 'on'}

ALLOW_RETRANSPILATION_FALLBACK = _env_flag('IOT_ALLOW_RETRANSPILATION_FALLBACK', False)


def _fidelity_cap_for_qubits(num_qubits: int) -> int:
    """Return a safe max-circuits-per-eval cap given the circuit width.

    Small-qubit circuits fit many evaluations in a batch, while 20+ qubit
    circuits must be kept extremely small to avoid GPU Aer segfaults. The
    helper preserves those safeguards without penalising lightweight runs.
    """

    try:
        qubits = int(num_qubits)
    except (TypeError, ValueError):
        qubits = 0

    if qubits >= 24:
        return max(FIDELITY_MIN_CIRCUITS, 1)
    if qubits >= 22:
        return max(FIDELITY_MIN_CIRCUITS, 2)
    if qubits >= 18:
        return max(FIDELITY_MIN_CIRCUITS, 4)
    if qubits >= 14:
        return max(FIDELITY_MIN_CIRCUITS, 8)
    if qubits >= 10:
        return max(FIDELITY_MIN_CIRCUITS, 16)
    if qubits >= 6:
        return max(FIDELITY_MIN_CIRCUITS, 64)
    return max(FIDELITY_MIN_CIRCUITS, 256)

def _validate_qpy_bytes(qpy_bytes: bytes, label: str) -> bool:
    """Ensure serialized qpy bytes can be reloaded before sharing.
    Returns True if valid, False if there's a known QPY bug we can work around."""
    if not qpy_bytes:
        raise ValueError(f"{label} produced empty qpy serialization")
    global FAILED_QPY_VALIDATIONS
    try:
        import io
        from qiskit.qpy import load  # type: ignore
        buf = io.BytesIO(qpy_bytes)
        circuits = load(buf)
        if not isinstance(circuits, (list, tuple)) or len(circuits) == 0:
            raise RuntimeError(f"qpy reload for {label} returned no circuits")
        return True
    except (IndexError, KeyError) as exc:
        # Known QPY bug with ParameterVector - log warning but don't fail (once per label)
        if label not in FAILED_QPY_VALIDATIONS:
            try:
                logger.warning(f"QPY validation failed for {label} (known ParameterVector bug): {exc}")
            except Exception:
                pass
            FAILED_QPY_VALIDATIONS.add(label)
        return False
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to reload qpy bytes for {label}: {exc}") from exc

def _handle_missing_precompiled_circuit(circuit_key: str, num_qubits: int, reason: str):
    message = f"Pre-transpiled circuit '{circuit_key}' unavailable ({reason})"
    fallback_forced = circuit_key in KNOWN_QPY_UNSTABLE_KEYS
    if not (ALLOW_RETRANSPILATION_FALLBACK or fallback_forced):
        raise RuntimeError(
            message +
            " — re-transpilation fallback disabled. Set IOT_ALLOW_RETRANSPILATION_FALLBACK=1 to allow on-the-fly compilation."
        )
    try:
        if fallback_forced and not ALLOW_RETRANSPILATION_FALLBACK:
            logger.warning("%s; auto re-transpiling due to known QPY instability", message)
        else:
            logger.warning("%s; re-transpiling because fallback flag is enabled", message)
    except Exception:
        pass
    return _build_feature_map_from_key(circuit_key, num_qubits)


def _persist_bad_qpy_bytes(label: str, qpy_bytes: Optional[bytes]) -> None:
    """Write invalid qpy payloads to disk for post-mortem inspection."""
    if not qpy_bytes:
        return
    try:
        GPU_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace('::', '_')
        path = GPU_LOGS_DIR / f"bad_qpy_{safe_label}_{os.getpid()}_{int(time.time())}.qpy"
        with open(path, 'wb') as fh:
            fh.write(qpy_bytes)
        try:
            logger.warning("Persisted invalid qpy bytes for %s to %s", label, path)
        except Exception:
            pass
    except Exception:
        pass

# Ray placement group imports
placement_group = None
remove_placement_group = None
PlacementGroupSchedulingStrategy = None
if RAY_AVAILABLE:
    try:
        from ray.util import placement_group, remove_placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    except Exception:
        pass


def _ensure_cupy():
    """Try to import cupy and return the module if available, else None."""
    try:
        import cupy
        return cupy
    except Exception:
        return None


def _assert_picklable(obj, name: str = ''):
    """Assert that `obj` is picklable. Raises TypeError if not."""
    try:
        pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise TypeError(f"Object '{name}' is not picklable: {e}")


def _get_gpu_free_memory_mb(gpu_id: int) -> Optional[float]:
    """Return free GPU memory in MiB for given gpu_id using pynvml if available."""
    try:
        if NVML_AVAILABLE:
            h = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            return mem.free / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def _compute_safe_chunk_size(num_qubits: int, gpu_id: int, base_chunk: int = KERNEL_CHUNK_SIZE) -> int:
    """Heuristic to choose a safe chunk size given GPU free memory and qubit count.

    Conservative estimate: per-sample statevector footprint approximately 2**num_qubits * 16 bytes.
    We add a safety factor to avoid OOMs and bound the chunk size.
    """
    try:
        per_state_bytes = (2 ** max(0, num_qubits)) * 16
    except OverflowError:
        per_state_bytes = float('inf')
    per_sample_mb = per_state_bytes / (1024 ** 2) if per_state_bytes != float('inf') else float('inf')
    free_mb = _get_gpu_free_memory_mb(gpu_id)
    reserve_mb = GPU_MEMORY_RESERVE_GB * 1024
    if free_mb is None or per_sample_mb <= 0 or per_sample_mb == float('inf'):
        return max(64, min(base_chunk, 512))
    avail_mb = max(0.0, free_mb - reserve_mb)
    # Leave room for intermediate allocations (safety factor)
    est_per_row_mb = per_sample_mb * 1.5 + 1.0
    if est_per_row_mb <= 0:
        return max(64, min(base_chunk, 512))
    candidate = int(max(32, min(base_chunk, avail_mb / est_per_row_mb)))
    # Bound chunk size
    candidate = max(32, min(candidate, 4096))
    return candidate


def _validate_registry_value(val) -> bool:
    """Validate that a value is safe to put into the central Manager registry.

    Allowed values: None, bytes, or a dict with keys qpy_bytes or qpy_file and optional meta.
    This prevents accidentally placing GPU-bound objects into the shared registry.
    """
    try:
        if val is None:
            return True
        if isinstance(val, (bytes, bytearray)):
            return True
        if isinstance(val, dict):
            # Accept dict with qpy_bytes, qpy_file, dill_file, and meta keys
            if 'qpy_bytes' in val or 'qpy_file' in val or 'dill_file' in val:
                return True
        return False
    except Exception:
        return False

# Global circuit manager cache for worker processes
_CIRCUIT_MANAGER_PARAMS: Optional[Dict[str, Any]] = None
_CIRCUIT_MANAGER_CACHE: Dict[int, Any] = {}

# NOTE: Do NOT import Qiskit here. Worker processes must set
# CUDA_VISIBLE_DEVICES before importing qiskit/aer so that the
# Aer primitives detect the correct local visible GPU (index 0).
# Importing Qiskit in the initializer would happen before any
# per-worker CUDA_VISIBLE_DEVICES is set and can cause Aer to
# select CPU backends unexpectedly. Workers should call
# _lazy_qiskit_imports() after they set their environment.

def _lazy_qiskit_imports():
    """Lazy import of Qiskit modules to avoid early GPU binding."""
    # This function is called when needed to import Qiskit symbols
    # The actual imports are already done conditionally at module level
    # This is a no-op placeholder for compatibility
    pass

def _set_central_registry(key: str, value: Any) -> None:
    """Store a serialized circuit payload in the central shared registry."""
    global _CENTRAL_QPY_REGISTRY

    try:
        if _CENTRAL_QPY_REGISTRY is None:
            return

        if not _validate_registry_value(value):
            try:
                logger.warning("Central registry rejected %s: value not serializable", key)
            except Exception:
                pass
            _CENTRAL_QPY_REGISTRY[key] = None
            return

        # Manager proxies require picklable objects; ensure we push plain builtins
        if isinstance(value, dict):
            sanitized = {k: value.get(k) for k in ("qpy_bytes", "qpy_file", "dill_file", "meta")}
        else:
            sanitized = value

        _CENTRAL_QPY_REGISTRY[key] = sanitized
    except Exception as exc:
        try:
            logger.debug("Central registry store failed for %s: %s", key, exc)
        except Exception:
            pass

def _write_qpy_file(qpy_bytes: bytes, key: str) -> Optional[str]:
    """Write QPY bytes to a file and return the path."""
    try:
        GPU_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        safe_key = key.replace('::', '_').replace('/', '_')
        path = GPU_LOGS_DIR / f"qpy_{safe_key}_{os.getpid()}.qpy"
        with open(path, 'wb') as f:
            f.write(qpy_bytes)
        return str(path)
    except Exception:
        return None

def _write_dill_file(circuit, key: str) -> Optional[str]:
    """Write circuit to dill file and return the path."""
    try:
        import dill
        GPU_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        safe_key = key.replace('::', '_').replace('/', '_')
        path = GPU_LOGS_DIR / f"circuit_{safe_key}_{os.getpid()}.dill"
        with open(path, 'wb') as f:
            dill.dump(circuit, f)
        return str(path)
    except Exception:
        return None

def _circuit_to_qpy_bytes(circuit) -> Optional[bytes]:
    """Convert a circuit to QPY bytes."""
    try:
        from qiskit.qpy import dump
        import io
        buf = io.BytesIO()
        dump([circuit], buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

def _init_circuit_manager_worker(num_qubits: int, num_samples: int, registry: Optional[Any] = None, result_queue: Optional[Any] = None):
    """Initialize worker process with circuit manager parameters."""
    global _CIRCUIT_MANAGER_PARAMS, _CIRCUIT_MANAGER_CACHE, _CENTRAL_QPY_REGISTRY, _RESULT_QUEUE
    _CIRCUIT_MANAGER_PARAMS = {
        'num_qubits': num_qubits,
        'num_samples': num_samples,
        'registry': registry,
        'result_queue': result_queue
    }
    _CIRCUIT_MANAGER_CACHE = {}

    # Ensure worker processes have access to central registries/result queue
    _CENTRAL_QPY_REGISTRY = registry
    _RESULT_QUEUE = result_queue

def _autotune_kernel_chunk_size(num_qubits: int, num_samples: int, base_chunk: int) -> int:
    """Auto-tune kernel chunk size based on GPU memory and qubit count."""
    # Conservative estimate for large qubit counts
    if num_qubits >= 20:
        return min(base_chunk, 512)
    elif num_qubits >= 18:
        return min(base_chunk, 1024)
    else:
        return base_chunk

def _autotune_nystrom_landmarks(num_qubits: int, num_samples: int, base_landmarks: int, chunk_size: int = 1024) -> int:
    """Auto-tune Nyström landmark count based on available resources."""
    # For very large qubit counts, reduce landmarks
    if num_qubits >= 22:
        tuned = min(base_landmarks, 1000)
    elif num_qubits >= 20:
        tuned = min(base_landmarks, 1500)
    else:
        tuned = base_landmarks
    
    # Don't exceed sample count
    return min(tuned, num_samples)

def adjust_cache_sizes_for_qubits(num_qubits: int):
    """Adjust cache sizes based on qubit count."""
    global MAX_SIMULATOR_CACHE_SIZE, MAX_FIDELITY_KERNEL_CACHE_SIZE
    if num_qubits >= 22:
        MAX_SIMULATOR_CACHE_SIZE = 1
        MAX_FIDELITY_KERNEL_CACHE_SIZE = 2
    elif num_qubits >= 20:
        MAX_SIMULATOR_CACHE_SIZE = 2
        MAX_FIDELITY_KERNEL_CACHE_SIZE = 3

def _get_circuit_manager(gpu_id=0):
    """Get or create circuit manager for current subprocess"""
    global _CIRCUIT_MANAGER_CACHE, _CIRCUIT_MANAGER_PARAMS
    
    if gpu_id not in _CIRCUIT_MANAGER_CACHE:
        if _CIRCUIT_MANAGER_PARAMS is None:
            raise RuntimeError("Circuit manager parameters not initialized!")
        
        logger.info(f"[GPU:{gpu_id}] Creating circuit manager in subprocess...")
        # CentralizedGPUCircuitManager will decide whether to hydrate from the
        # shared qpy registry (worker) or perform a local precompile (main).
        _CIRCUIT_MANAGER_CACHE[gpu_id] = CentralizedGPUCircuitManager(
            num_qubits=_CIRCUIT_MANAGER_PARAMS['num_qubits'],
            num_samples=_CIRCUIT_MANAGER_PARAMS['num_samples'],
            gpu_id=gpu_id
        )
    
    return _CIRCUIT_MANAGER_CACHE[gpu_id]

# ============================================================================
# STEP 1: CENTRALIZED GPU CIRCUIT MANAGER (PRESERVES COMPLEXITY)
# ============================================================================

class CentralizedGPUCircuitManager:
    """Centralized circuit management for ALL quantum models
    
    CRITICAL: Maintains full circuit complexity for fair comparison
    - Pre-transpiles all circuit templates on GPU
    - Preserves model-specific characteristics
    - Caches compiled circuits for reuse
    """
    
    def __init__(self, num_qubits, num_samples, gpu_id=0):
        self.num_qubits = num_qubits
        self.num_samples = num_samples
        self.gpu_id = gpu_id
        
        # Centralized caches
        self.transpiled_circuits = {}
        self.kernel_cache = {}
        self.feature_maps = {}
        self.ansatzes = {}
        
        # If a central qpy registry exists (set by the master process) and we
        # are in a worker process, hydrate circuits from qpy bytes instead of
        # re-transpiling. This avoids redundant, expensive transpilation in
        # worker subprocesses.
        try:
            in_worker = mp.current_process().name != 'MainProcess'
        except Exception:
            in_worker = False

        if in_worker and _CENTRAL_QPY_REGISTRY is not None:
            logger.info(f"[GPU:{self.gpu_id}] 🔁 Hydrating circuits from central qpy registry (no transpile)")
            try:
                self._hydrate_from_qpy_registry()
            except Exception as e:
                logger.warning(f"[GPU:{self.gpu_id}] Hydration failed, falling back to local precompile: {e}")
                self._precompile_all_circuits()
        else:
            # Pre-compile all standard circuit types
            self._precompile_all_circuits()

        # If we are the main process and we have a central registry proxy, ensure
        # it is populated (this is a no-op in workers)
        try:
                if not in_worker and _CENTRAL_QPY_REGISTRY is not None:
                    # populate registry with qpy bytes so child workers can hydrate
                    for k, circ in list(self.feature_maps.items()):
                        try:
                            entry = {'qpy_bytes': getattr(circ, 'qpy_bytes', None), 'meta': getattr(circ, 'qpy_meta', None)}
                            _set_central_registry(k, entry)
                        except Exception:
                            _set_central_registry(k, {'qpy_bytes': None, 'meta': None})

                    for k, circ in list(self.ansatzes.items()):
                        try:
                            entry = {'qpy_bytes': getattr(circ, 'qpy_bytes', None), 'meta': getattr(circ, 'qpy_meta', None)}
                            _set_central_registry(f"ansatz::{k}", entry)
                        except Exception:
                            _set_central_registry(f"ansatz::{k}", {'qpy_bytes': None, 'meta': None})
        except Exception:
            pass
    
    def _store_circuit(self, container: Dict[str, Any], key: str, circuit: 'QuantumCircuit') -> None:
        """Attach serialization metadata to a transpiled circuit and store it."""
        container[key] = circuit
        qpy_bytes = None
        qpy_valid = False

        global KNOWN_QPY_UNSTABLE_KEYS

        # Skip QPY serialization for known unstable circuits or to speed up initialization
        if key in KNOWN_QPY_UNSTABLE_KEYS:
            circuit.qpy_bytes = None
            if key not in _WARNED_QPY_SKIPS:
                try:
                    logger.debug(f"Skipping QPY serialization for {key} due to known ParameterVector bug")
                except Exception:
                    pass
                _WARNED_QPY_SKIPS.add(key)
        else:
            try:
                from qiskit.qpy import dump  # type: ignore
                import io
                buf = io.BytesIO()
                dump([circuit], buf)
                buf.seek(0)
                qpy_bytes = buf.read()
                # SKIP VALIDATION for speed - trust that dump succeeded
                qpy_valid = True
            except Exception as exc:
                # Log the error but don't fail - we can still use the circuit object
                try:
                    logger.debug(f"QPY serialization failed for {key}: {exc}. Circuit will be stored as object only.")
                except Exception:
                    pass
                qpy_bytes = None
                qpy_valid = False

            # Store QPY bytes only if serialization succeeded
            if qpy_valid and qpy_bytes:
                circuit.qpy_bytes = qpy_bytes
            else:
                circuit.qpy_bytes = None
                if key not in KNOWN_QPY_UNSTABLE_KEYS:
                    KNOWN_QPY_UNSTABLE_KEYS.add(key)
            
        try:
            import qiskit
            try:
                import qiskit_machine_learning as qml
                qml_ver = getattr(qml, '__version__', None)
            except Exception:
                qml_ver = None
            circuit.qpy_meta = {
                'qiskit_version': getattr(qiskit, '__version__', None),
                'qml_version': qml_ver,
                'transpile_options': {
                    'optimization_level': TRANSPILE_OPT_LEVEL,
                    'basis_gates': ['u', 'cx', 'rz', 'sx', 'x', 'ry']
                },
                'timestamp': time.time(),
                'qpy_valid': qpy_valid
            }
        except Exception:
            circuit.qpy_meta = None

        try:
            dill_path = _write_dill_file(circuit, key)
            if dill_path:
                circuit.dill_file = dill_path
        except Exception:
            pass

    def _precompile_all_circuits(self):
        """Pre-compile ALL circuit types with FULL complexity preserved"""
        # Ensure Qiskit symbols are available (lazy import)
        try:
            _lazy_qiskit_imports()
        except Exception:
            pass

        logger.info(f"[GPU:{self.gpu_id}] 🔧 Pre-compiling circuit templates (preserving complexity)...")
        start_time = time.time()
        
        # Feature maps with PRESERVED complexity for comparison
        total_circuits = 0
        for map_type in ['Z', 'ZZ', 'Pauli']:
            for reps in [1, 2]:  # Maintain different repetition counts
                key = f"{map_type}_reps{reps}"
                logger.info(f"[GPU:{self.gpu_id}]   Compiling feature map: {key}")
                transpiled = self._compile_feature_map(map_type, reps)
                self._store_circuit(self.feature_maps, key, transpiled)
                total_circuits += 1
        
        # Ansatzes for QNN/VQC models
        for ansatz_type in ['RealAmplitudes', 'EfficientSU2', 'TwoLocal']:
            for reps in [1, 2]:
                key = f"{ansatz_type}_reps{reps}"
                logger.info(f"[GPU:{self.gpu_id}]   Compiling ansatz: {key}")
                transpiled = self._compile_ansatz(ansatz_type, reps)
                self._store_circuit(self.ansatzes, key, transpiled)
                total_circuits += 1
        
        elapsed = time.time() - start_time
        logger.info(f"[GPU:{self.gpu_id}] ✅ Pre-compiled {len(self.feature_maps)} feature maps, {len(self.ansatzes)} ansatzes in {elapsed:.1f}s")
    
    def _compile_feature_map(self, map_type, reps):
        """Compile feature map with FULL complexity"""
        try:
            _lazy_qiskit_imports()
        except Exception:
            pass

        if map_type == 'Z':
            feature_map = ZFeatureMap(self.num_qubits, reps=reps)
        elif map_type == 'ZZ':
            # Preserve full entanglement for proper comparison
            feature_map = ZZFeatureMap(self.num_qubits, reps=reps, entanglement='full')
        elif map_type == 'Pauli':
            feature_map = PauliFeatureMap(self.num_qubits, reps=reps, paulis=['Z', 'ZZ'])
        else:
            feature_map = ZZFeatureMap(self.num_qubits, reps=reps)
        
        # Optimize transpilation for GPU execution
        logger.debug(f"[GPU:{self.gpu_id}] Transpiling {map_type}_reps{reps}...")
        transpiled = transpile(
            feature_map,
            optimization_level=TRANSPILE_OPT_LEVEL,  # Balanced optimization (configurable)
            basis_gates=['u', 'cx', 'rz', 'sx', 'x', 'ry']
        )
        
        logger.debug(f"[GPU:{self.gpu_id}] Compiled {map_type}_reps{reps}: depth={transpiled.depth()}, gates={sum(transpiled.count_ops().values())}")
        
        return transpiled
    
    def _compile_ansatz(self, ansatz_type, reps):
        """Compile ansatz with FULL complexity"""
        try:
            _lazy_qiskit_imports()
        except Exception:
            pass

        if ansatz_type == 'RealAmplitudes':
            ansatz = RealAmplitudes(self.num_qubits, reps=reps, entanglement='full')
        elif ansatz_type == 'EfficientSU2':
            ansatz = EfficientSU2(self.num_qubits, reps=reps, entanglement='full')
        else:  # TwoLocal
            ansatz = TwoLocal(self.num_qubits, 'ry', 'cz', reps=reps, entanglement='full')
        
        transpiled = transpile(
            ansatz,
            optimization_level=TRANSPILE_OPT_LEVEL,
            basis_gates=['u', 'cx', 'rz', 'sx', 'x', 'ry']
        )
        
        return transpiled

    def _hydrate_from_qpy_registry(self):
        """Hydrate feature maps and ansatzes from the shared registry without silent re-transpilation."""
        global _CENTRAL_QPY_REGISTRY
        self.feature_maps = {}
        self.ansatzes = {}

        def _registry_entry(reg_key: str):
            try:
                return _CENTRAL_QPY_REGISTRY.get(reg_key)
            except Exception:
                return None

        def _materialize(entry: Any, label: str) -> Tuple[Optional[QuantumCircuit], Optional[dict], List[str]]:
            if entry is None:
                return None, None, [f"registry entry for {label} missing"]

            meta = None
            qpy_bytes = None
            qpy_file = None
            dill_file = None
            if isinstance(entry, dict):
                meta = entry.get('meta')
                qpy_bytes = entry.get('qpy_bytes')
                qpy_file = entry.get('qpy_file')
                dill_file = entry.get('dill_file')
            else:
                qpy_bytes = entry

            errors: List[str] = []
            circuit: Optional[QuantumCircuit] = None

            if dill_file:
                if DILL_AVAILABLE:
                    try:
                        with open(dill_file, 'rb') as fh:
                            circuit = dill.load(fh)
                    except Exception as exc:
                        errors.append(f'dill load failed: {exc}')
                else:
                    errors.append('dill payload provided but dill is unavailable')

            if circuit is None and qpy_file:
                try:
                    from qiskit.qpy import load  # type: ignore
                    with open(qpy_file, 'rb') as fh:
                        circuits = load(fh)
                    if isinstance(circuits, (list, tuple)) and circuits:
                        circuit = circuits[0]
                    else:
                        errors.append('qpy file load returned no circuits')
                except Exception as exc:
                    errors.append(f'qpy file load failed: {exc}')

            if circuit is None and qpy_bytes:
                try:
                    import io
                    from qiskit.qpy import load  # type: ignore
                    buf = io.BytesIO(qpy_bytes)
                    circuits = load(buf)
                    if isinstance(circuits, (list, tuple)) and circuits:
                        circuit = circuits[0]
                    else:
                        errors.append('qpy bytes load returned no circuits')
                except Exception as exc:
                    errors.append(f'qpy bytes load failed: {exc}')
                    _persist_bad_qpy_bytes(label, qpy_bytes)

            return circuit, meta, errors

        # Hydrate feature maps
        for map_type in ['Z', 'ZZ', 'Pauli']:
            for reps in [1, 2]:
                key = f"{map_type}_reps{reps}"
                entry = _registry_entry(key)
                circuit, meta, errors = _materialize(entry, key)

                if circuit is not None:
                    self._store_circuit(self.feature_maps, key, circuit)
                    if meta is not None:
                        self.feature_maps[key].qpy_meta = meta
                    try:
                        logger.info(f"[GPU:{self.gpu_id}] Hydrated circuit {key} from central registry")
                    except Exception:
                        pass
                    continue

                reason = '; '.join(errors) if errors else 'missing serialized payload'
                circuit = _handle_missing_precompiled_circuit(key, self.num_qubits, reason)
                self._store_circuit(self.feature_maps, key, circuit)

        # Hydrate ansatzes
        for ansatz_type in ['RealAmplitudes', 'EfficientSU2', 'TwoLocal']:
            for reps in [1, 2]:
                key = f"{ansatz_type}_reps{reps}"
                reg_key = f"ansatz::{key}"
                entry = _registry_entry(reg_key)
                circuit, meta, errors = _materialize(entry, reg_key)

                if circuit is not None:
                    self._store_circuit(self.ansatzes, key, circuit)
                    if meta is not None:
                        self.ansatzes[key].qpy_meta = meta
                    try:
                        logger.info(f"[GPU:{self.gpu_id}] Hydrated ansatz {reg_key} from central registry")
                    except Exception:
                        pass
                    continue

                reason = '; '.join(errors) if errors else 'missing serialized payload'
                if not ALLOW_RETRANSPILATION_FALLBACK:
                    raise RuntimeError(
                        f"Pre-transpiled ansatz '{reg_key}' unavailable ({reason}) — re-transpilation fallback disabled. "
                        "Set IOT_ALLOW_RETRANSPILATION_FALLBACK=1 to allow on-the-fly compilation."
                    )
                try:
                    logger.warning(
                        "Pre-transpiled ansatz %s unavailable (%s); re-transpiling because fallback flag is enabled",
                        reg_key,
                        reason
                    )
                except Exception:
                    pass
                circuit = self._compile_ansatz(ansatz_type, reps)
                self._store_circuit(self.ansatzes, key, circuit)
    
    def get_circuit_for_model(self, model_type, config=None):
        """Get pre-compiled circuit for model type"""
        if config is None:
            config = {}
        
        if 'QSVC' in model_type:
            map_type = config.get('feature_map', 'ZZ')
            reps = config.get('reps', 2)
            key = f"{map_type}_reps{reps}"
            return self.feature_maps.get(key)
        
        elif 'VQC' in model_type or 'QNN' in model_type:
            feature_key = f"Z_reps1"
            ansatz_key = f"RealAmplitudes_reps2"
            return self.feature_maps.get(feature_key), self.ansatzes.get(ansatz_key)
        
        return None

# ============================================================================
# STEP 2: OPTIMIZED MULTI-GPU KERNEL WORKER
# ============================================================================

def _process_kernel_chunk_worker_optimized(args):
    """Worker process that evaluates a kernel submatrix on a dedicated GPU.

    The worker keeps simulator creation to a minimum and leverages Qiskit
    fidelity primitives so we avoid per-pair circuit transpilation as well as
    repeated shot-based measurements. Fidelity values are computed directly on
    the statevector simulator running on the requested GPU."""

    import logging

    worker_logger = logging.getLogger("GPU_Worker")

    X1_chunk = np.asarray(args['X1_chunk'], dtype=np.float32)
    X2 = np.asarray(args['X2'], dtype=np.float32)
    circuit_key = args['circuit_key']
    gpu_id = int(args['gpu_id'])
    chunk_idx = int(args['chunk_idx'])
    chunk_end = int(args['chunk_end'])
    chunk_num = int(args['chunk_num'])
    total_chunks = int(args['total_chunks'])
    num_qubits = int(args['num_qubits'])
    max_circuits = args.get('max_circuits')
    if max_circuits is not None:
        try:
            max_circuits = int(max_circuits)
        except Exception:
            max_circuits = None

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'

    # After remapping visibility, GPU libraries see the selected physical
    # GPU as local device index 0. Use local_cuda_device when calling GPU APIs.
    local_cuda_device = 0

    if X1_chunk.ndim == 1:
        X1_chunk = X1_chunk.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)

    worker_logger.info(
        f"[GPU:{gpu_id}] Evaluating kernel chunk {chunk_num}/{total_chunks} "
        f"(rows {chunk_idx}:{chunk_end})"
    )


    # For very large qubit counts, create a transient kernel object in this
    # worker and immediately clean it up after computing the chunk to avoid
    # holding large device-resident kernel objects in a process-local cache.
    if num_qubits >= LARGE_QUBIT_THRESHOLD:
        kernel = _create_transient_fidelity_kernel(
            circuit_key=circuit_key,
            num_qubits=num_qubits,
            gpu_id=local_cuda_device,
            max_circuits_per_eval=max_circuits
        )
        try:
            kernel_chunk = kernel.evaluate(X1_chunk, X2)
            kernel_chunk = np.asarray(np.real(kernel_chunk), dtype=np.float32)
        finally:
            try:
                _safe_cleanup(kernel, gpu_hint=gpu_id)
            except Exception:
                pass
            try:
                del kernel
            except Exception:
                pass
            gc.collect()
    else:
        kernel = _get_shared_fidelity_kernel(
            circuit_key=circuit_key,
            num_qubits=num_qubits,
            gpu_id=local_cuda_device,
            max_circuits=max_circuits
        )

        kernel_chunk = kernel.evaluate(X1_chunk, X2)
        kernel_chunk = np.asarray(np.real(kernel_chunk), dtype=np.float32)

    # Attempt to release any temporary GPU allocations
    try:
        # Clear local visible device memory
        clear_gpu_memory(local_cuda_device)
    except Exception:
        pass

    worker_logger.info(f"[GPU:{gpu_id}] ✅ Chunk {chunk_num}/{total_chunks} complete")
    return kernel_chunk


def _result_writer_process(result_queue, session_id: str):
    """Dedicated single-writer process that consumes result dicts and writes
    them atomically to the CSV and checkpoint. This avoids concurrent writes
    from worker processes.
    Expected queue items: None (sentinel) or dict with keys: 'result', 'gpu_id'
    """
    try:
        results_manager = ThreadSafeResultsManager(session_id)
    except Exception:
        # If ThreadSafeResultsManager cannot be created, we still attempt to write
        results_manager = None

    while True:
        try:
            item = result_queue.get()
        except Exception:
            break

        if item is None:
            break

        # Support either direct result dict or wrapper
        if isinstance(item, dict) and 'result' in item:
            res = item['result']
            gpu_id = item.get('gpu_id', -1)
        else:
            res = item
            gpu_id = res.get('gpu_id', -1) if isinstance(res, dict) else -1

        try:
            if results_manager is not None:
                results_manager.save_result(res, gpu_id=gpu_id)
            else:
                # Fallback: append to CSV manually
                try:
                    results_file = RESULTS_DIR / f'gpu_quantum_results_{session_id}.csv'
                    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    import fcntl
                    row_df = pd.DataFrame([res])
                    write_header = not results_file.exists()
                    with open(results_file, 'a', newline='') as fh:
                        try:
                            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                            row_df.to_csv(fh, header=write_header, index=False)
                        finally:
                            try:
                                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            # We don't want the writer to crash; log and continue
            try:
                logger.exception("ResultWriter failed to save result")
            except Exception:
                pass

    # Finalize: ensure checkpoint saved
    try:
        if results_manager is not None:
            results_manager._save_checkpoint()
    except Exception:
        pass

# ============================================================================
# STEP 3: SHARED GPU KERNEL COMPUTER (PRESERVES MODEL VARIANTS)
# ============================================================================

class SharedGPUKernelComputer:
    """Shared kernel computation engine for ALL models
    
    CRITICAL: Each model variant uses its specific computation method
    - QSVC_Standard: Uses Nyström approximation for efficiency
    - QSVC_Precomputed: Computes FULL kernel matrix
    - QSVC_Callable: Computes kernel on-demand
    - PegasosQSVC: Uses stochastic approach
    """
    
    def __init__(self, circuit_manager, gpu_id=0, batch_size=1024):
        self.circuit_manager = circuit_manager
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        
        # Kernel cache (LRU bounded)
        self.kernel_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create GPU simulator
        self.simulator = self._create_gpu_simulator()
        self.max_circuits_per_eval = self._estimate_max_circuits_per_eval()
        self.approx_mode = (APPROXIMATION_MODE or 'auto').strip().lower()
        self.circuit_cuts = max(0, CIRCUIT_CUT_PARTITIONS)
        self._cut_kernel_cache: Dict[Tuple[str, int], Any] = {}
        self._fallback_activated = False
        self._parallel_retry_serialized = False
        try:
            logger.info(
                f"[GPU:{self.gpu_id}] Max fidelity circuits per batch set to {self.max_circuits_per_eval}"
            )
        except Exception:
            pass
    
    def _create_gpu_simulator(self):
        """Create optimized GPU simulator"""
        return _get_shared_aer_simulator(self.gpu_id)

    def _estimate_max_circuits_per_eval(self) -> int:
        """Heuristic to bound fidelity batch size based on GPU memory and qubit count."""
        num_qubits = int(getattr(self.circuit_manager, 'num_qubits', 0) or 0)
        cap = _fidelity_cap_for_qubits(num_qubits)

        try:
            # EMERGENCY OVERRIDE: For 22+ qubits, NEVER exceed 1 circuit per eval
            if num_qubits >= 22:
                return max(FIDELITY_MIN_CIRCUITS, 1)

            per_state_mb = float((2 ** max(0, num_qubits)) * FIDELITY_COMPLEX_BYTES) / (1024 ** 2)
            if not np.isfinite(per_state_mb) or per_state_mb <= 0:
                return max(FIDELITY_MIN_CIRCUITS, min(cap, 8))

            free_mb = _get_gpu_free_memory_mb(self.gpu_id)
            if free_mb is None or free_mb <= 0:
                target_mb = FIDELITY_TARGET_BATCH_MEMORY_MB
            else:
                target_mb = max(FIDELITY_MIN_TARGET_MB, free_mb - FIDELITY_DEVICE_RESERVE_MB)

            denom = max(per_state_mb * FIDELITY_MEMORY_SAFETY, 1.0)
            approx = int(target_mb / denom)
            approx = max(FIDELITY_MIN_CIRCUITS, min(cap, approx))
            return approx
        except Exception:
            return max(FIDELITY_MIN_CIRCUITS, min(cap, 8))
    
    def compute_kernel_for_model(self, X1, X2, circuit_key, model_type, available_gpus=None):
        """Compute kernel based on model type to preserve differentiation"""
        
        # Get model-specific configuration
        config = NYSTROM_CONFIG.get(model_type, {'use_nystrom': False})
        
        logger.info(f"[GPU:{self.gpu_id}] Computing kernel for {model_type}")
        logger.info(f"[GPU:{self.gpu_id}] Configuration: {config}")
        
        n_train = len(X1)
        # If user attempts a full kernel on very large sample sets, automatically
        # fall back to Nyström (if configured) or to a stochastic Pegasos-style
        # approach. This prevents attempting to allocate an N x N matrix when
        # N is large (e.g., 50k) which would OOM both host and device memory.
        if not config.get('use_nystrom', False) and n_train > FULL_KERNEL_MAX_SAMPLES:
            # Attempt to fall back to Nyström where possible
            logger.warning(f"[GPU:{self.gpu_id}] Requested FULL kernel for {n_train} samples exceeds safe threshold ({FULL_KERNEL_MAX_SAMPLES}). Falling back to Nyström/pegasos automatically.")
            # If landmarks configured, use them; otherwise use a safe default
            landmarks = config.get('landmarks') or min(2000, n_train)
            logger.info(f"[GPU:{self.gpu_id}] Switching to Nyström with m={landmarks} landmarks")
            return self.compute_kernel_nystrom(X1, X2, circuit_key,
                                              n_landmarks=landmarks,
                                              available_gpus=available_gpus)

        if config.get('use_nystrom', False) and n_train > 5000:
            logger.info(f"[GPU:{self.gpu_id}] Using Nyström approximation with {config.get('landmarks')} landmarks")
            return self.compute_kernel_nystrom(X1, X2, circuit_key,
                                              n_landmarks=config.get('landmarks'),
                                              available_gpus=available_gpus)

        # Other variants compute full kernel (safe for small N)
        logger.info(f"[GPU:{self.gpu_id}] Computing FULL kernel matrix")
        return self.compute_kernel_batch(X1, X2, circuit_key, available_gpus)
    
    def compute_kernel_batch(self, X1, X2, circuit_key, available_gpus=None):
        """Compute full kernel matrix with optimized chunking"""

        # Accept X2==None as shorthand for computing kernel of X1 vs X1
        X1 = np.asarray(X1, dtype=np.float32)
        if X2 is None:
            logger.debug(f"[GPU:{self.gpu_id}] compute_kernel_batch: X2 is None, using X1 for symmetric kernel")
            X2 = X1
        else:
            X2 = np.asarray(X2, dtype=np.float32)
        
        # Check cache
        cache_key = self._get_cache_key(X1, X2, circuit_key)
        if cache_key in self.kernel_cache:
            self.cache_hits += 1
            try:
                # move to end (most-recent)
                self.kernel_cache.move_to_end(cache_key)
            except Exception:
                pass
            logger.debug(f"[GPU:{self.gpu_id}] Cache HIT ({self.cache_hits} hits)")
            return self.kernel_cache[cache_key]
        self.cache_misses += 1
        
        # Get pre-compiled circuit
        circuit = self.circuit_manager.feature_maps.get(circuit_key)
        if circuit is None:
            raise ValueError(f"Circuit {circuit_key} not found")
        
        n1, n2 = len(X1), len(X2)
        total_circuits = n1 * n2
        
        logger.info(f"[GPU:{self.gpu_id}] Computing kernel: {n1}×{n2} = {total_circuits:,} circuits")
        
        # Use chunking for large matrices
        if total_circuits > 50000:
            kernel = self._compute_kernel_chunked(X1, X2, circuit, circuit_key, available_gpus)
        else:
            kernel = self._compute_kernel_direct(X1, X2, circuit, circuit_key)
        
        # Cache result with bounded LRU behavior
        try:
            self.kernel_cache[cache_key] = kernel
            try:
                self.kernel_cache.move_to_end(cache_key)
            except Exception:
                pass
            while len(self.kernel_cache) > MAX_MODEL_KERNEL_CACHE_ENTRIES:
                try:
                    ev_key, ev_val = self.kernel_cache.popitem(last=False)
                    try:
                        _log_eviction(f'kernel_cache_gpu{self.gpu_id}', ev_key, ev_val, 'max_entries_exceeded')
                    except Exception:
                        pass
                except Exception:
                    break
        except Exception:
            pass

        try:
            _trim_caches_if_needed()
        except Exception:
            pass
        
        return kernel
    
    def compute_kernel_nystrom(self, X_train, X_test, circuit_key, n_landmarks=2000, available_gpus=None):
        """Nyström approximation for QSVC_Standard only"""
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32) if X_test is not None else None

        n_train = len(X_train)
        n_test = len(X_test) if X_test is not None else 0
        
        logger.info(f"[GPU:{self.gpu_id}] 🚀 Nyström approximation")
        logger.info(f"[GPU:{self.gpu_id}] Dataset: {n_train} train, {n_test} test")
        logger.info(f"[GPU:{self.gpu_id}] Landmarks: {n_landmarks}")
        
        # Select landmark points
        np.random.seed(42)
        landmark_indices = np.random.choice(n_train, size=min(n_landmarks, n_train), replace=False)
        X_landmarks = X_train[landmark_indices]
        
        # Get circuit
        circuit = self.circuit_manager.feature_maps.get(circuit_key)
        if circuit is None:
            raise ValueError(f"Circuit {circuit_key} not found")
        
        # Compute kernels
        logger.info(f"[GPU:{self.gpu_id}] Computing landmark kernel...")
        K_mm = self._compute_kernel_chunked(X_landmarks, X_landmarks, circuit, circuit_key, available_gpus)
        
        logger.info(f"[GPU:{self.gpu_id}] Computing train-landmark kernel...")
        K_nm = self._compute_kernel_chunked(X_train, X_landmarks, circuit, circuit_key, available_gpus)
        
        if X_test is not None:
            logger.info(f"[GPU:{self.gpu_id}] Computing test-landmark kernel...")
            K_tm = self._compute_kernel_chunked(X_test, X_landmarks, circuit, circuit_key, available_gpus)
        
        # Nyström approximation
        from scipy.linalg import pinv
        K_mm_reg = K_mm + 1e-6 * np.eye(len(K_mm))
        K_mm_inv = pinv(K_mm_reg)
        
        K_train_approx = K_nm @ K_mm_inv @ K_nm.T
        K_train_approx = (K_train_approx + K_train_approx.T) / 2
        
        if X_test is not None:
            K_test_approx = K_tm @ K_mm_inv @ K_nm.T
            return K_train_approx, K_test_approx
        
        return K_train_approx
    
    def _compute_kernel_direct(self, X1, X2, circuit, circuit_key):
        """Direct kernel computation for small matrices"""
        return self._evaluate_kernel_block(X1, X2, circuit_key, circuit)
    
    def _compute_kernel_chunked(self, X1, X2, circuit, circuit_key, available_gpus=None):
        """Chunked computation with multi-GPU support"""
        n1, n2 = len(X1), len(X2)
        
        # Determine chunk size dynamically based on free GPU memory and matrix width
        chunk_size = self._determine_chunk_size(n2)
        num_chunks = (n1 + chunk_size - 1) // chunk_size
        
        num_qubits = int(getattr(self.circuit_manager, 'num_qubits', 0))
        logger.info(f"[GPU:{self.gpu_id}] CHUNK SIZE DETERMINED: {chunk_size} rows (qubits={num_qubits}, n_cols={n2}, total_chunks={num_chunks})")
        logger.info(f"[GPU:{self.gpu_id}] Processing {num_chunks} chunks")
        
        # Distributed execution via Ray (optional)
        if DISTRIBUTED_STRATEGY == 'ray':
            try:
                return self._compute_kernel_distributed_ray(
                    X1, X2, circuit, circuit_key, chunk_size, available_gpus
                )
            except Exception as dist_err:
                try:
                    logger.warning(
                        "[GPU:%s] Ray distributed execution failed (%s); falling back to local GPUs",
                        self.gpu_id,
                        dist_err
                    )
                except Exception:
                    pass

        # Multi-GPU parallel processing
        if available_gpus and len(available_gpus) > 1 and num_chunks > 4:
            return self._compute_kernel_parallel(X1, X2, circuit, circuit_key, chunk_size, available_gpus)
        
        return self._compute_kernel_sequential(X1, X2, circuit, circuit_key, chunk_size)
    
    def _compute_kernel_parallel(self, X1, X2, circuit, circuit_key, chunk_size, available_gpus):
        """Multi-GPU kernel computation using dedicated worker processes per GPU."""
        if not available_gpus:
            raise ValueError("available_gpus must contain at least one GPU id")

        self._parallel_retry_serialized = False
        n1, n2 = len(X1), len(X2)
        num_chunks = (n1 + chunk_size - 1) // chunk_size
        gpu_ring = available_gpus[:MAX_PARALLEL_GPUS]
        logger.info(f"[PARALLEL] Using {len(gpu_ring)} GPUs for {num_chunks} chunks")

        kernel = np.zeros((n1, n2), dtype=np.float32)

        # Shared memory to avoid copying large arrays for each task
        X1_info = create_shared_memory_for_array(X1)
        X2_info = create_shared_memory_for_array(X2)
        shm_registry = {'X1': X1_info, 'X2': X2_info}

        circuit_qpy = getattr(circuit, 'qpy_bytes', None)
        if not circuit_qpy:
            circuit_qpy = _circuit_to_qpy_bytes(circuit)

        manager = GPUWorkerManager(gpu_ring)
        fallback_exc: Optional[Exception] = None
        try:
            manager.start(shm_registry)

            if manager.result_queue is None:
                raise RuntimeError("GPU worker manager failed to initialise result queue")

            submitted = 0
            for chunk_num, chunk_idx in enumerate(range(0, n1, chunk_size)):
                chunk_end = min(chunk_idx + chunk_size, n1)
                gpu_id = gpu_ring[chunk_num % len(gpu_ring)]

                task = {
                    'action': 'eval_block',
                    'circuit_key': circuit_key,
                    'circuit_qpy': circuit_qpy,
                    'X1_shm': X1_info,
                    'X2_shm': X2_info,
                    'slice1': (chunk_idx, chunk_end),
                    'slice2': (0, n2),
                    'chunk_idx': chunk_idx,
                    'chunk_end': chunk_end,
                    'chunk_num': chunk_num,
                    'num_qubits': self.circuit_manager.num_qubits,
                    'gpu_id': gpu_id,
                    'max_circuits': int(self.max_circuits_per_eval or FIDELITY_MAX_CIRCUITS)
                }
                manager.submit(gpu_id, task)
                submitted += 1

            completed = 0
            result_queue = manager.result_queue
            last_progress = time.time()
            idle_poll_count = 0
            poll_timeout = max(5, min(RESULT_QUEUE_POLL_SECONDS, 300))
            while completed < submitted:
                try:
                    item = result_queue.get(timeout=poll_timeout)
                except queue.Empty:
                    manager.reap_terminated()
                    worker_status = manager.worker_status()
                    
                    # CRITICAL FIX: Immediately detect crashed workers
                    crashed_workers = [
                        gid for gid, status in worker_status.items() 
                        if 'exit' in status or status == 'stopped'
                    ]
                    if crashed_workers:
                        raise RuntimeError(
                            f"GPU workers {crashed_workers} crashed before completing kernel evaluation. "
                            f"Status: {worker_status}. Check GPU memory and logs for segfault details."
                        )
                    
                    if not manager.any_alive():
                        raise RuntimeError(
                            f"All GPU workers exited before completing kernel chunks (status={worker_status})"
                        )
                    
                    idle_duration = time.time() - last_progress
                    if idle_duration >= KERNEL_MAX_IDLE_SECONDS:
                        raise TimeoutError(
                            f"No kernel results received for {int(idle_duration)}s (status={worker_status})"
                        )
                    if idle_poll_count % 4 == 0:
                        try:
                            logger.info(f"[PARALLEL] Waiting for GPU chunks... idle={idle_duration:.0f}s status={worker_status}")
                        except Exception:
                            pass
                    idle_poll_count += 1
                    continue

                last_progress = time.time()
                idle_poll_count = 0
                if 'error' in item and item['error']:
                    raise RuntimeError(f"GPU worker {item.get('gpu_id')} failed: {item['error']}")
                start, end = item['chunk_idx'], item['chunk_end']
                kernel[start:end, :] = item['kernel']
                completed += 1
        except Exception as exc:
            fallback_exc = exc
        finally:
            manager.stop()
            try:
                free_shared_memory(X1_info)
            except Exception:
                pass
            try:
                free_shared_memory(X2_info)
            except Exception:
                pass

        if fallback_exc is not None:
            if not self._parallel_retry_serialized:
                self._parallel_retry_serialized = True
                try:
                    logger.warning("[GPU:%s] Parallel path failed (%s); retrying with serialized GPU execution", self.gpu_id, fallback_exc)
                except Exception:
                    pass
                try:
                    result = self._compute_kernel_sequential(
                        X1, X2, circuit, circuit_key, max(1, (chunk_size // 2) or chunk_size)
                    )
                    self._parallel_retry_serialized = False
                    return result
                except Exception as serial_exc:
                    fallback_exc = serial_exc

            if self._should_activate_fallback(fallback_exc):
                try:
                    logger.warning("[GPU:%s] Falling back to approximate kernel due to parallel failure: %s", self.gpu_id, fallback_exc)
                except Exception:
                    pass
                return self._compute_kernel_fallback(X1, X2, circuit, circuit_key, fallback_exc)
            raise fallback_exc

        return kernel

    def _compute_kernel_sequential(self, X1, X2, circuit, circuit_key, chunk_size):
        """Compute kernel chunk-by-chunk on a single GPU without worker processes."""
        X1_arr = np.asarray(X1, dtype=np.float32)
        X2_arr = np.asarray(X2, dtype=np.float32)

        n1 = len(X1_arr)
        n2 = len(X2_arr)
        kernel = np.zeros((n1, n2), dtype=np.float32)

        serial_chunk = max(1, int(chunk_size) if chunk_size else 1)
        last_log = time.time()
        for chunk_idx in range(0, n1, serial_chunk):
            chunk_end = min(chunk_idx + serial_chunk, n1)
            X1_chunk = X1_arr[chunk_idx:chunk_end]
            chunk_kernel = self._evaluate_kernel_block(X1_chunk, X2_arr, circuit_key, circuit)
            kernel[chunk_idx:chunk_end, :] = chunk_kernel

            progress = (chunk_end / max(1, n1)) * 100.0
            now = time.time()
            if (chunk_idx == 0) or (now - last_log >= 10) or (chunk_end == n1):
                try:
                    logger.info("[GPU:%s] Serialized GPU progress: %.1f%%", self.gpu_id, progress)
                except Exception:
                    pass
                last_log = now

        return kernel
    
    def _should_activate_fallback(self, exc: Exception) -> bool:
        # Only allow fallback when circuit cutting is explicitly enabled.
        return self.circuit_cuts > 1

    def _compute_kernel_fallback(self, X1, X2, circuit, circuit_key, original_error: Optional[Exception] = None):
        num_qubits = self.circuit_manager.num_qubits
        if self.circuit_cuts > 1 and num_qubits >= max(2, self.circuit_cuts):
            try:
                logger.info("[GPU:%s] Applying circuit cutting fallback (cuts=%s)", self.gpu_id, self.circuit_cuts)
            except Exception:
                pass
            return self._evaluate_kernel_with_circuit_cuts(X1, X2, circuit_key)
        error_msg = "GPU distributed execution failed and circuit cutting is disabled"
        if original_error is not None:
            error_msg = f"{error_msg}: {original_error}"
        raise RuntimeError(error_msg)

    def _evaluate_kernel_with_circuit_cuts(self, X_left, X_right, circuit_key):
        partitions = _partition_indices(self.circuit_manager.num_qubits, self.circuit_cuts)
        result = np.ones((len(X_left), len(X_right)), dtype=np.float32)
        for idx, (start, end) in enumerate(partitions):
            if end <= start:
                continue
            sub_dim = end - start
            X_left_slice = np.asarray(X_left[:, start:end], dtype=np.float32)
            X_right_slice = np.asarray(X_right[:, start:end], dtype=np.float32)
            cache_key = (circuit_key, sub_dim)
            kernel_obj = self._cut_kernel_cache.get(cache_key)
            if kernel_obj is None:
                sub_feature_map = _build_feature_map_from_key(circuit_key, sub_dim)
                kernel_obj = create_quantum_kernel(
                    feature_map=sub_feature_map,
                    cuda_device=self.gpu_id,
                    max_circuits_per_eval=self.max_circuits_per_eval
                )
                self._cut_kernel_cache[cache_key] = kernel_obj
            block = kernel_obj.evaluate(X_left_slice, X_right_slice)
            result *= np.asarray(block, dtype=np.float32)
        return result

    def _compute_kernel_distributed_ray(self, X1, X2, circuit, circuit_key, chunk_size, available_gpus=None):
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray distributed execution requested but Ray is unavailable")

        chunk_rows = max(1, int(chunk_size) if chunk_size else 1)
        X1_array = np.asarray(X1, dtype=np.float32)
        X2_array = np.asarray(X2, dtype=np.float32)

        _ensure_ray_initialized(DISTRIBUTED_ADDRESS)

        x2_ref = ray.put(X2_array)  # type: ignore[attr-defined]
        try:
            if self._ray_should_use_gpu_workers() and _RayGPUKernelActor is not None:
                return self._compute_kernel_distributed_ray_gpu(
                    X1_array,
                    X2_array,
                    circuit,
                    circuit_key,
                    chunk_rows,
                    x2_ref,
                    available_gpus
                )
            raise RuntimeError("Ray GPU workers are required but unavailable")
        except Exception as gpu_err:
            raise RuntimeError(
                f"Ray GPU distributed execution failed; CPU fallback is disabled: {gpu_err}"
            ) from gpu_err
        finally:
            try:
                del x2_ref
            except Exception:
                pass

    def _ray_should_use_gpu_workers(self) -> bool:
        mode = (self.approx_mode or 'auto').strip().lower()
        if mode in ('gpu', 'gpu-only', 'gpu_strict', 'auto'):
            return True
        return False

    def _determine_ray_worker_count(self, available_gpus, num_chunks: int) -> int:
        worker_count = 0
        try:
            if available_gpus:
                worker_count = max(worker_count, len(available_gpus))
        except Exception:
            pass
        try:
            gpu_nodes = _ray_list_alive_gpu_nodes()
            if gpu_nodes:
                node_gpu_total = 0
                for node in gpu_nodes:
                    try:
                        node_gpu_total += int(max(0, int(node.get('Resources', {}).get('GPU', 0))))
                    except Exception:
                        node_gpu_total += 1
                if node_gpu_total > 0:
                    worker_count = max(worker_count, node_gpu_total)
        except Exception:
            pass
        try:
            cluster_total = ray.cluster_resources().get('GPU', 0.0)  # type: ignore[attr-defined]
            if cluster_total:
                worker_count = max(worker_count, int(cluster_total))
            cluster_available = ray.available_resources().get('GPU', 0.0)  # type: ignore[attr-defined]
            if cluster_available:
                worker_count = max(worker_count, int(cluster_available))
        except Exception:
            pass
        if worker_count <= 0:
            worker_count = 1
        worker_count = max(1, min(int(worker_count), num_chunks, MAX_PARALLEL_GPUS))
        return worker_count

    def _compute_kernel_distributed_ray_gpu(self, X1_array, X2_array, circuit, circuit_key, chunk_size, x2_ref, available_gpus):
        if _RayGPUKernelActor is None:
            raise RuntimeError("Ray GPU kernel actor unavailable")

        num_rows = len(X1_array)
        chunk_size = max(1, int(chunk_size))
        num_chunks = (num_rows + chunk_size - 1) // chunk_size
        worker_count = self._determine_ray_worker_count(available_gpus, num_chunks)

        gpu_nodes = _ray_list_alive_gpu_nodes()
        if not gpu_nodes:
            raise RuntimeError("No alive Ray GPU nodes detected for distributed execution")

        total_slots = 0
        for node in gpu_nodes:
            try:
                total_slots += int(max(0, node.get('Resources', {}).get('GPU', 0)))
            except Exception:
                total_slots += 1

        if total_slots <= 0:
            raise RuntimeError("Ray cluster reports zero GPU slots across all nodes")

        worker_count = max(1, min(worker_count, total_slots))

        pg = None
        if (
            placement_group is not None and
            PlacementGroupSchedulingStrategy is not None and
            worker_count > 1
        ):
            try:
                pg = placement_group([{'GPU': 1}] * worker_count, strategy='SPREAD')  # type: ignore[misc]
                ray.get(pg.ready())  # type: ignore[attr-defined]
            except Exception as pg_exc:
                pg = None
                try:
                    logger.warning(
                        "Ray placement group spread strategy failed (%s); continuing without strict node pinning",
                        pg_exc
                    )
                except Exception:
                    pass

        circuit_qpy = getattr(circuit, 'qpy_bytes', None)
        if not circuit_qpy:
            circuit_qpy = _circuit_to_qpy_bytes(circuit)

        actors: List[Any] = []
        try:
            for idx in range(worker_count):
                options_kwargs: Dict[str, Any] = {}
                if pg is not None and PlacementGroupSchedulingStrategy is not None:
                    try:
                        options_kwargs['scheduling_strategy'] = PlacementGroupSchedulingStrategy(
                            pg,
                            bundle_index=idx,
                            capture_child_tasks=True
                        )
                    except Exception:
                        options_kwargs.pop('scheduling_strategy', None)
                else:
                    options_kwargs['scheduling_strategy'] = 'SPREAD'

                try:
                    actor_handle = _RayGPUKernelActor.options(**options_kwargs).remote(  # type: ignore[call-arg]
                        self.approx_mode,
                        int(self.max_circuits_per_eval or 1)
                    )
                except Exception:
                    actor_handle = _RayGPUKernelActor.remote(  # type: ignore[call-arg]
                        self.approx_mode,
                        int(self.max_circuits_per_eval or 1)
                    )
                actors.append(actor_handle)

            actor_locations: List[str] = []
            for idx, actor in enumerate(actors):
                try:
                    info = ray.get(actor.runtime_info.remote())  # type: ignore[attr-defined]
                    node_ref = info.get('node_ip') or info.get('node_id') or 'unknown'
                    actor_locations.append(f"actor-{idx}@{node_ref}")
                except Exception:
                    pass
            if actor_locations:
                try:
                    logger.info(
                        "[GPU:%s] Ray GPU actors placed on %s",
                        self.gpu_id,
                        ', '.join(actor_locations)
                    )
                except Exception:
                    pass

            kernel = np.zeros((num_rows, len(X2_array)), dtype=np.float32)
            futures = []
            for chunk_idx, start in enumerate(range(0, num_rows, chunk_size)):
                end = min(start + chunk_size, num_rows)
                actor = actors[chunk_idx % len(actors)]
                payload = {
                    'X1_chunk': np.asarray(X1_array[start:end], dtype=np.float32),
                    'X2_ref': x2_ref,
                    'circuit_key': circuit_key,
                    'num_qubits': self.circuit_manager.num_qubits,
                    'circuit_qpy': circuit_qpy,
                    'chunk_idx': start,
                    'chunk_end': end,
                    'max_circuits': int(self.max_circuits_per_eval or 1),
                    'approx_mode': self.approx_mode
                }
                futures.append(actor.evaluate.remote(payload))  # type: ignore[operator]

            for item in ray.get(futures):  # type: ignore[attr-defined]
                if 'error' in item and item['error']:
                    raise RuntimeError(item['error'])
                kernel[item['chunk_idx']:item['chunk_end'], :] = np.asarray(item['kernel'], dtype=np.float32)

            return kernel
        finally:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
            if pg is not None and remove_placement_group is not None:
                try:
                    remove_placement_group(pg)  # type: ignore[misc]
                except Exception:
                    pass

    def _evaluate_kernel_block(self, X_left, X_right, circuit_key, circuit=None):
        """Evaluate a kernel block using the shared fidelity kernel."""

        X_left = np.asarray(X_left, dtype=np.float32)
        X_right = np.asarray(X_right, dtype=np.float32)

        if X_left.ndim == 1:
            X_left = X_left.reshape(1, -1)
        if X_right.ndim == 1:
            X_right = X_right.reshape(1, -1)

        feature_map = circuit if circuit is not None else self.circuit_manager.feature_maps.get(circuit_key)
        if feature_map is None:
            feature_map = _handle_missing_precompiled_circuit(
                circuit_key,
                self.circuit_manager.num_qubits,
                'feature map not present in cache'
            )

        if (
            self.circuit_cuts > 1
            and self.circuit_manager.num_qubits >= max(2, self.circuit_cuts)
            and self.circuit_manager.num_qubits >= self.circuit_cuts * 2
        ):
            return self._evaluate_kernel_with_circuit_cuts(X_left, X_right, circuit_key)

        # EMERGENCY: For 22+ qubits, evaluate row-by-row AND chunk X_right to prevent creating
        # thousands of circuit objects at once. 1 row × 2000 cols = 2000 circuits is too much.
        num_qubits = self.circuit_manager.num_qubits
        if num_qubits >= 22:
            kernel_block = np.zeros((X_left.shape[0], X_right.shape[0]), dtype=np.float32)
            # ULTRA-AGGRESSIVE chunking for 22+ qubits: only 5 circuits per evaluate call
            # This minimizes GPU memory pressure per kernel.evaluate() invocation
            max_right_cols = max(1, 5)  # VERY small to avoid crashes
            for i in range(X_left.shape[0]):
                row_results = []
                for j_start in range(0, X_right.shape[0], max_right_cols):
                    j_end = min(j_start + max_right_cols, X_right.shape[0])
                    
                    # CRITICAL: Recreate fidelity kernel EVERY batch to ensure fresh GPU state
                    # This prevents AER internal GPU buffer accumulation
                    fidelity_kernel = _get_shared_fidelity_kernel(
                        circuit_key=circuit_key,
                        num_qubits=self.circuit_manager.num_qubits,
                        gpu_id=self.gpu_id,
                        feature_map=feature_map,
                        max_circuits=1  # Always 1 for ultra-safety
                    )
                    
                    chunk_result = fidelity_kernel.evaluate(
                        X_left[i:i+1], 
                        X_right[j_start:j_end]
                    )
                    row_results.append(np.asarray(np.real(chunk_result), dtype=np.float32))
                    # Clear GPU memory after EACH tiny batch
                    clear_gpu_memory(self.gpu_id)
                # Concatenate column chunks
                kernel_block[i:i+1, :] = np.concatenate(row_results, axis=1)
        else:
            fidelity_kernel = _get_shared_fidelity_kernel(
                circuit_key=circuit_key,
                num_qubits=self.circuit_manager.num_qubits,
                gpu_id=self.gpu_id,
                feature_map=feature_map,
                max_circuits=int(self.max_circuits_per_eval or FIDELITY_MAX_CIRCUITS)
            )
            kernel_block = fidelity_kernel.evaluate(X_left, X_right)
            kernel_block = np.asarray(np.real(kernel_block), dtype=np.float32)
            clear_gpu_memory(self.gpu_id)

        return kernel_block

    def _determine_chunk_size(self, n_columns: int) -> int:
        """Determine an adaptive chunk size based on GPU memory availability."""
        
        num_qubits = int(getattr(self.circuit_manager, 'num_qubits', 0))
        
        # EMERGENCY OVERRIDE: For 22+ qubits, the total circuit count (rows × cols)
        # must stay extremely low to avoid worker crashes. Force tiny chunk size.
        if num_qubits >= 22:
            # CRITICAL: At 22+ qubits, each circuit pair costs ~32MB GPU memory.
            # Total circuits per batch = rows × cols. Cap at 32 total circuits max.
            emergency_cap = max(1, 32 // max(1, n_columns))
            result = max(1, min(emergency_cap, 4))
            try:
                logger.warning(
                    f"[GPU:{self.gpu_id}] EMERGENCY chunk sizing for {num_qubits} qubits: "
                    f"chunk_rows={result}, n_columns={n_columns}, "
                    f"total_circuits_per_batch={result * n_columns} (cap: 32)"
                )
            except Exception:
                pass
            return result

        default_rows = KERNEL_CHUNK_SIZE
        # Prefer NVML-aware safe chunk heuristic if available
        try:
            safe = _compute_safe_chunk_size(num_qubits, self.gpu_id, base_chunk=default_rows)
            # further bound by default_rows extremes
            safe = max(1, min(safe, max(default_rows * 8, default_rows)))
        except Exception:
            # Fallback to original NVML logic if heuristic fails
            if not NVML_AVAILABLE or (gpu_manager is None) or self.gpu_id >= gpu_manager.gpu_count:
                safe = default_rows
            else:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    free_gb = max(0.0, (mem_info.free / (1024 ** 3)) - GPU_MEMORY_RESERVE_GB)
                    if free_gb <= 0:
                        safe = max(1, default_rows // 2)
                    else:
                        # Each fidelity evaluation produces n_columns circuits; cap total circuits per chunk.
                        max_circuits = max(32, int(free_gb * 512))
                        adaptive_rows = max(1, min(default_rows * 4, max_circuits // max(1, n_columns)))
                        safe = adaptive_rows
                except Exception:
                    safe = default_rows

        # Bound row-size by fidelity evaluation limits to avoid GPU worker OOM/segfaults
        row_bound = self.max_circuits_per_eval or FIDELITY_MAX_CIRCUITS
        row_cap = max(1, row_bound)
        safe = max(1, min(int(safe), row_cap))
        return int(safe)

        
    
    def _get_cache_key(self, X1, X2, circuit_key):
        """Generate cache key for kernel matrix"""
        try:
            key = (hash(X1.tobytes()), hash(X2.tobytes()), circuit_key)
            return key
        except:
            return None

# ============================================================================
# STEP 4: BATCH PROCESSING PIPELINE (TRUE PARALLEL EXECUTION)
# ============================================================================

class BatchProcessingPipeline:
    """Process all quantum models with true GPU parallelization"""
    
    def __init__(self, num_gpus=3, session_id=None, num_qubits=22, num_samples=50000, gpus_per_model=None):
        self.num_gpus = num_gpus
        self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S_BATCH')
        self.num_qubits = num_qubits
        self.num_samples = num_samples
        requested = gpus_per_model or num_gpus
        capped_request = min(self.num_gpus, MAX_PARALLEL_GPUS, requested)
        self.gpus_per_model = max(1, capped_request)
        self._gpu_groups = self._build_gpu_groups()
        # Auto-tune a few parameters for very large experiments (22 qubits / 50k+ samples)
        try:
            if self.num_samples >= 20000:
                # For very large sample sizes prefer Nyström for QSVC_Standard
                logger.info(f"Auto-enabling Nyström approximation for large dataset ({self.num_samples} samples)")
                base_landmarks = min(5000, max(2000, int(self.num_samples // 10)))
                NYSTROM_CONFIG['QSVC_Standard']['use_nystrom'] = True
            else:
                # Small/medium datasets use full kernel for best accuracy
                logger.info(f"Using full kernel computation for dataset ({self.num_samples} samples < 20000 threshold)")
                NYSTROM_CONFIG['QSVC_Standard']['use_nystrom'] = False
                global KERNEL_CHUNK_SIZE
                tuned_chunk = _autotune_kernel_chunk_size(self.num_qubits, self.num_samples, KERNEL_CHUNK_SIZE)
                if tuned_chunk != KERNEL_CHUNK_SIZE:
                    logger.info(f"Auto-tuning kernel chunk size for large run: {KERNEL_CHUNK_SIZE} -> {tuned_chunk}")
                    KERNEL_CHUNK_SIZE = tuned_chunk

                tuned_landmarks = _autotune_nystrom_landmarks(
                    self.num_qubits,
                    self.num_samples,
                    base_landmarks,
                    KERNEL_CHUNK_SIZE
                )
                if tuned_landmarks != base_landmarks:
                    logger.info(f"Auto-tuning Nyström landmarks: {base_landmarks} -> {tuned_landmarks}")
                NYSTROM_CONFIG['QSVC_Standard']['landmarks'] = tuned_landmarks
                # Propagate landmark cap to fallback paths for other QSVC variants
                for variant in ('QSVC_Precomputed', 'QSVC_Callable'):
                    if variant in NYSTROM_CONFIG:
                        NYSTROM_CONFIG[variant]['landmarks'] = tuned_landmarks

                # Increase kernel chunk size to reduce host-device roundtrips
                global GPU_MEMORY_RESERVE_GB
                GPU_MEMORY_RESERVE_GB = max(GPU_MEMORY_RESERVE_GB, 4)

                logger.info(
                    f"Auto-tuning for large dataset: Nyström landmarks={NYSTROM_CONFIG['QSVC_Standard']['landmarks']}, "
                    f"KERNEL_CHUNK_SIZE={KERNEL_CHUNK_SIZE}, GPU_MEMORY_RESERVE_GB={GPU_MEMORY_RESERVE_GB}"
                )
        except Exception:
            pass
        
        # Lazily initialize global GPU manager here in the main process so
        # worker processes do not re-run GPU detection at import time.
        global gpu_manager
        if gpu_manager is None:
            try:
                gpu_manager = GPUManager()
            except Exception as e:
                logger.warning(f"GPU manager initialization failed: {e}")
                gpu_manager = None

        # Ensure heavy Qiskit imports happen in main process (lazy)
        try:
            _lazy_qiskit_imports()
        except Exception:
            pass

        # Create centralized circuit manager
        logger.info("=" * 80)
        logger.info("🔧 INITIALIZING CENTRALIZED GPU CIRCUIT MANAGER")
        logger.info("=" * 80)
        
        self.circuit_manager = CentralizedGPUCircuitManager(
            num_qubits=num_qubits,
            num_samples=num_samples,
            gpu_id=0
        )

        # Create a Manager-backed registry containing qpy bytes for all
        # precompiled circuits so worker processes can hydrate without
        # re-transpiling. This proxy will be passed to ProcessPoolExecutor
        # worker initializers.
        try:
            self._mp_manager = mp.Manager()
            self.central_qpy_registry = self._mp_manager.dict()
            # publish proxy globally so helper _set_central_registry can use it
            try:
                global _CENTRAL_QPY_REGISTRY
                _CENTRAL_QPY_REGISTRY = self.central_qpy_registry
            except Exception:
                pass

            # CRITICAL: Populate registry BEFORE creating any worker pools
            # so workers can hydrate qpy bytes rather than retranspile.
            try:
                self._populate_central_registry()
            except Exception:
                # Fallback to inline population if helper fails
                for k, circ in list(self.circuit_manager.feature_maps.items()):
                    try:
                        qpy_bytes = getattr(circ, 'qpy_bytes', None)
                        qpy_file = None
                        if qpy_bytes:
                            qpy_file = _write_qpy_file(qpy_bytes, k)
                        val = {
                            'qpy_bytes': None if qpy_file else qpy_bytes,
                            'qpy_file': qpy_file,
                            'dill_file': getattr(circ, 'dill_file', None),
                            'meta': getattr(circ, 'qpy_meta', None)
                        }
                        try:
                            _set_central_registry(k, val)
                        except Exception:
                            logger.warning(f"Central registry: failed to set {k}, storing None")
                            _set_central_registry(k, {'qpy_bytes': None, 'qpy_file': None, 'meta': None})
                    except Exception:
                        _set_central_registry(k, {'qpy_bytes': None, 'qpy_file': None, 'meta': None})

                for k, circ in list(self.circuit_manager.ansatzes.items()):
                    try:
                        reg_key = f"ansatz::{k}"
                        qpy_bytes = getattr(circ, 'qpy_bytes', None)
                        qpy_file = None
                        if qpy_bytes:
                            qpy_file = _write_qpy_file(qpy_bytes, reg_key)
                        val = {
                            'qpy_bytes': None if qpy_file else qpy_bytes,
                            'qpy_file': qpy_file,
                            'dill_file': getattr(circ, 'dill_file', None),
                            'meta': getattr(circ, 'qpy_meta', None)
                        }
                        try:
                            _set_central_registry(reg_key, val)
                        except Exception:
                            logger.warning(f"Central registry: failed to set {reg_key}, storing None")
                            _set_central_registry(reg_key, {'qpy_bytes': None, 'qpy_file': None, 'meta': None})
                    except Exception:
                        _set_central_registry(f"ansatz::{k}", {'qpy_bytes': None, 'qpy_file': None, 'meta': None})
        except Exception:
            self.central_qpy_registry = None
        # Central result queue and writer process
        try:
            self.result_queue = self._mp_manager.Queue()
        except Exception:
            self.result_queue = mp.Queue()

        try:
            global _RESULT_QUEUE
            _RESULT_QUEUE = self.result_queue
        except Exception:
            pass

        # Start a dedicated ResultWriter process (single writer)
        try:
            self._result_writer_proc = Process(target=_result_writer_process, args=(self.result_queue, self.session_id), daemon=True)
            self._result_writer_proc.start()
            logger.info("📝 ResultWriter process started")
        except Exception as e:
            logger.warning(f"Failed to start ResultWriter process: {e}")
        
        # Create kernel computers for each GPU
        self.kernel_computers = []
        # Pre-warm per-GPU simulators/cache to reduce first-call overhead
        for gpu_id in range(num_gpus):
            try:
                logger.info(f"⏱️ Pre-warming simulator/cache for GPU {gpu_id}")
                _get_shared_aer_simulator(gpu_id)
            except Exception:
                logger.warning(f"Pre-warm simulator failed for GPU {gpu_id}")

        for gpu_id in range(num_gpus):
            logger.info(f"🚀 Creating SharedGPUKernelComputer for GPU {gpu_id}")
            kernel_computer = SharedGPUKernelComputer(
                self.circuit_manager,
                gpu_id=gpu_id,
                batch_size=1024
            )
            self.kernel_computers.append(kernel_computer)
        
        try:
            logger.info(f"GPU groups per model run: {self._gpu_groups}")
        except Exception:
            pass

        logger.info("✅ CENTRALIZED SYSTEM INITIALIZED")
        logger.info("=" * 80)

    def _build_gpu_groups(self) -> List[List[int]]:
        """Create disjoint GPU groups that will be dedicated to each model run."""
        groups: List[List[int]] = []
        step = max(1, self.gpus_per_model)
        gpu_ids = list(range(self.num_gpus))
        for idx in range(0, len(gpu_ids), step):
            group = gpu_ids[idx:idx + step]
            if group:
                groups.append(group)
        if not groups:
            groups = [gpu_ids]
        return groups
    
    def process_all_models(self, models, X_train, y_train, X_test, y_test):
        """Process models in TRUE parallel across GPUs"""
        
        logger.info("=" * 80)
        logger.info("🎯 STARTING PARALLEL BATCH PROCESSING")
        logger.info(f"📊 Models: {len(models)}")
        logger.info(f"🖥️  GPUs: {self.num_gpus}")
        logger.info(f"📐 Data: {len(X_train)} train, {len(X_test)} test")
        logger.info("=" * 80)
        
        gpu_groups = self._gpu_groups or [list(range(self.num_gpus))]
        if not gpu_groups:
            gpu_groups = [[0]]

        all_results: List[Dict[str, Any]] = []
        max_workers = max(1, len(gpu_groups))

        if not models:
            logger.info("No models requested; skipping execution")
            return all_results

        # Use initializer so each worker process receives the central qpy registry and result queue
        if getattr(self, 'central_qpy_registry', None) is not None:
            initargs = (self.num_qubits, self.num_samples, self.central_qpy_registry, self.result_queue)
        else:
            initargs = (self.num_qubits, self.num_samples, None, self.result_queue)

        available_groups: deque[List[int]] = deque(gpu_groups)

        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_circuit_manager_worker, initargs=initargs) as executor:
            active: Dict[Any, Tuple[str, List[int]]] = {}
            next_model_idx = 0

            def submit_next() -> bool:
                nonlocal next_model_idx
                if next_model_idx >= len(models):
                    return False
                if not available_groups:
                    return False

                model_name, model_type, config = models[next_model_idx]
                gpu_group = available_groups.popleft()
                primary_gpu = gpu_group[0]
                gpu_label = ','.join(str(g) for g in gpu_group)

                logger.info(f"[GPU:{gpu_label}] Starting {model_name}")
                future = executor.submit(
                    train_quantum_model_centralized,
                    model_name, model_type, config,
                    X_train, y_train, X_test, y_test,
                    self.session_id, primary_gpu,
                    self.num_qubits, self.num_samples,
                    gpu_group
                )
                active[future] = (model_name, gpu_group)
                next_model_idx += 1
                return True

            # Prime the executor with as many tasks as we have worker slots
            while len(active) < max_workers and submit_next():
                pass

            while active:
                done, _ = wait(set(active.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    model_name, gpu_group = active.pop(future)
                    gpu_label = ','.join(str(g) for g in gpu_group)
                    try:
                        result = future.result(timeout=7200)
                        all_results.append(result)
                        logger.info(f"✅ [GPU:{gpu_label}] {model_name} complete")
                    except Exception as e:
                        logger.error(f"❌ [GPU:{gpu_label}] {model_name} failed: {e}")
                        all_results.append({
                            'model': model_name,
                            'status': 'failed',
                            'error': str(e)[:200]
                        })
                    finally:
                        available_groups.append(gpu_group)

                # Refill executor slots with the next pending models
                while len(active) < max_workers and submit_next():
                    pass
        
        logger.info("=" * 80)
        logger.info(f"✅ BATCH PROCESSING COMPLETE: {len(all_results)}/{len(models)}")
        logger.info("=" * 80)
        
        return all_results

    def shutdown(self):
        """Shutdown any persistent GPU worker manager and free resources."""
        global _GLOBAL_GPU_WORKER_MANAGER
        try:
            if _GLOBAL_GPU_WORKER_MANAGER is not None:
                _GLOBAL_GPU_WORKER_MANAGER.stop()
                _GLOBAL_GPU_WORKER_MANAGER = None
                logger.info("✅ GPU worker manager stopped")
        except NameError:
            pass
        # Shutdown result writer
        try:
            if hasattr(self, '_result_writer_proc') and self.result_queue is not None:
                try:
                    # send sentinel and join
                    self.result_queue.put(None)
                except Exception:
                    pass
                try:
                    self._result_writer_proc.join(timeout=30)
                except Exception:
                    pass
                logger.info("📝 ResultWriter process stopped")
        except Exception:
            pass

    def _populate_central_registry(self):
        """Populate the Manager-backed central qpy registry from the
        circuit manager precompiled circuits and ansatzes.

        This is separated into its own method so it can be called before
        any worker pools are created (prevents retranspilation in workers).
        """
        try:
            if getattr(self, 'central_qpy_registry', None) is None:
                return
            for k, circ in list(self.circuit_manager.feature_maps.items()):
                try:
                    qpy_bytes = getattr(circ, 'qpy_bytes', None)
                    qpy_file = None
                    if qpy_bytes:
                        try:
                            qpy_file = _write_qpy_file(qpy_bytes, k)
                        except Exception:
                            qpy_file = None

                    val = {
                        'qpy_bytes': None if qpy_file else qpy_bytes,
                        'qpy_file': qpy_file,
                        'dill_file': getattr(circ, 'dill_file', None),
                        'meta': getattr(circ, 'qpy_meta', None)
                    }
                    try:
                        _set_central_registry(k, val)
                    except Exception:
                        logger.warning(f"Central registry: failed to set {k}, storing None")
                        _set_central_registry(k, {'qpy_bytes': None, 'qpy_file': None, 'meta': None})
                except Exception:
                    _set_central_registry(k, {'qpy_bytes': None, 'qpy_file': None, 'meta': None})

            for k, circ in list(self.circuit_manager.ansatzes.items()):
                try:
                    reg_key = f"ansatz::{k}"
                    qpy_bytes = getattr(circ, 'qpy_bytes', None)
                    qpy_file = None
                    if qpy_bytes:
                        try:
                            qpy_file = _write_qpy_file(qpy_bytes, reg_key)
                        except Exception:
                            qpy_file = None

                    val = {
                        'qpy_bytes': None if qpy_file else qpy_bytes,
                        'qpy_file': qpy_file,
                        'dill_file': getattr(circ, 'dill_file', None),
                        'meta': getattr(circ, 'qpy_meta', None)
                    }
                    try:
                        _set_central_registry(reg_key, val)
                    except Exception:
                        logger.warning(f"Central registry: failed to set {reg_key}, storing None")
                        _set_central_registry(reg_key, {'qpy_bytes': None, 'qpy_file': None, 'meta': None})
                except Exception:
                    _set_central_registry(f"ansatz::{k}", {'qpy_bytes': None, 'qpy_file': None, 'meta': None})
        except Exception:
            pass

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_all_metrics(y_true, y_pred, y_pred_proba=None, train_time=0.0):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['sensitivity'] = metrics['recall']
    
    # Specificity
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape[0] == 2:
            tn = cm[0, 0]
            fp = cm[0, 1]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['specificity'] = specificity
        else:
            specificities = []
            for i in range(cm.shape[0]):
                tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
                fp = cm[:, i].sum() - cm[i, i]
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                specificities.append(spec)
            metrics['specificity'] = np.mean(specificities)
    except:
        metrics['specificity'] = 0.0
    
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # ROC-AUC and PR-AUC
    metrics['roc_auc'] = 0.0
    metrics['pr_auc'] = 0.0
    
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    y_proba_pos = y_pred_proba[:, 1]
                else:
                    y_proba_pos = y_pred_proba
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba_pos)
                metrics['pr_auc'] = auc(recall_curve, precision_curve)
            else:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                except:
                    metrics['roc_auc'] = 0.0
        except:
            pass
    
    metrics['training_time'] = train_time
    metrics['train_time'] = train_time  # Alias
    
    return metrics

# ============================================================================
# GPU MANAGEMENT
# ============================================================================

class GPUManager:
    """GPU detection and management"""
    
    def __init__(self):
        self.gpu_count = 0
        self.gpu_info = []
        self.gpu_locks = {}
        self.gpu_usage = {}
        self._detect_gpus()
    
    def _detect_gpus(self):
        """Detect available GPUs"""
        if NVML_AVAILABLE:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                    has_tensor_cores = float(compute_capability) >= 7.0
                    
                    gpu_data = {
                        'id': i,
                        'name': name,
                        'memory_total': mem_info.total / (1024**3),
                        'memory_free': mem_info.free / (1024**3),
                        'memory_used': mem_info.used / (1024**3),
                        'utilization': util.gpu,
                        'compute_capability': compute_capability,
                        'has_tensor_cores': has_tensor_cores,
                        'is_available': True
                    }
                    self.gpu_info.append(gpu_data)
                    self.gpu_locks[i] = threading.Lock()
                    self.gpu_usage[i] = 0
                    
                logger.info(f"🖥️ Detected {self.gpu_count} GPU(s):")
                for gpu in self.gpu_info:
                    logger.info(f"   GPU {gpu['id']}: {gpu['name']} "
                              f"({gpu['memory_free']:.1f}/{gpu['memory_total']:.1f} GB free) "
                              f"CC: {gpu['compute_capability']}")
            except Exception as e:
                logger.error(f"GPU detection failed: {e}")
                self.gpu_count = 0
        
        if self.gpu_count == 0:
            raise RuntimeError("❌ No GPUs detected! This script requires GPU support.")
    
    def allocate_gpu(self, preferred_gpu=None):
        """Allocate a GPU for a task"""
        if preferred_gpu is not None and preferred_gpu < self.gpu_count:
            with self.gpu_locks[preferred_gpu]:
                if self.gpu_usage[preferred_gpu] < 3:
                    self.gpu_usage[preferred_gpu] += 1
                    return preferred_gpu
        
        for gpu_id in range(self.gpu_count):
            with self.gpu_locks[gpu_id]:
                if self.gpu_usage[gpu_id] < 3:
                    self.gpu_usage[gpu_id] += 1
                    return gpu_id
        
        return 0
    
    def release_gpu(self, gpu_id):
        """Release a GPU"""
        if gpu_id in self.gpu_locks:
            with self.gpu_locks[gpu_id]:
                self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)
    
    def get_gpu_memory_status(self, gpu_id):
        """Get GPU memory status"""
        if NVML_AVAILABLE and gpu_id < self.gpu_count:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    'free': mem_info.free / (1024**3),
                    'used': mem_info.used / (1024**3),
                    'total': mem_info.total / (1024**3)
                }
            except:
                pass
        return None

# Global GPU Manager (lazily initialized in main process)
# Avoid creating GPUManager at module import time to prevent worker
# processes from re-running GPU detection and emitting repeated logs.
gpu_manager = None

def get_optimal_thread_count(gpu_id=0):
    """Get optimal thread count for GPU"""
    if NVML_AVAILABLE and (gpu_manager is not None) and gpu_id < gpu_manager.gpu_count:
        gpu_info = gpu_manager.gpu_info[gpu_id]
        cc = float(gpu_info.get('compute_capability', '7.5'))
        if cc >= 8.6:
            return 384
        elif cc >= 8.0:
            return 384
        elif cc >= 7.5:
            return 256
    return 256

def clear_gpu_memory(gpu_id=0):
    """Clear GPU memory caches"""
    cupy_mod = _ensure_cupy()
    if cupy_mod is not None:
        try:
            with cupy_mod.cuda.Device(gpu_id):
                cupy_mod.get_default_memory_pool().free_all_blocks()
        except Exception:
            try:
                with cupy_mod.cuda.Device(0):
                    cupy_mod.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
    gc.collect()


def monitor_gpu_usage(interval: float = 0.1, out_file: Optional[str] = None, stop_event: Optional[threading.Event] = None):
    """Background GPU monitor using pynvml writing CSV lines for correlation with logs.

    Writes lines: timestamp, gpu_index, memory_used_mib, memory_free_mib, utilization_percent
    """
    if not NVML_AVAILABLE:
        logger.warning("pynvml not available: GPU monitor disabled")
        return

    out_path = out_file or str(GPU_LOGS_DIR / f'nvidia_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    try:
        with open(out_path, 'w') as fh:
            fh.write('timestamp,gpu_index,memory_used_mib,memory_free_mib,utilization_percent\n')
            fh.flush()
            while stop_event is None or not stop_event.is_set():
                ts = time.time()
                for i in range(pynvml.nvmlDeviceGetCount()):
                    try:
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        util = pynvml.nvmlDeviceGetUtilizationRates(h)
                        line = f"{ts},{i},{int(mem.used/1024/1024)},{int(mem.free/1024/1024)},{int(util.gpu)}\n"
                        fh.write(line)
                    except Exception:
                        pass
                try:
                    fh.flush()
                except Exception:
                    pass
                time.sleep(interval)
    except Exception as e:
        try:
            logger.warning(f"GPU monitor failed: {e}")
        except Exception:
            pass

# ============================================================================
# GPU ACCELERATED PRIMITIVES
# ============================================================================

def create_gpu_simulator(gpu_id=0, method='statevector', precision='single'):
    """Create GPU-accelerated quantum simulator"""
    try:
        try:
            _lazy_qiskit_imports()
        except Exception:
            pass

        backend_options = {
            'method': method,
            'device': 'GPU',
            'precision': precision,
        }
        
        simulator = AerSimulator(**backend_options)
        config = simulator.configuration()
        backend_name = config.backend_name if hasattr(config, 'backend_name') else str(config)
        
        if 'gpu' not in backend_name.lower():
            raise RuntimeError(f"GPU simulator not available - got {backend_name} instead")
        
        logger.info(f"✅ GPU Simulator created with {method} method")
        return simulator
    
    except Exception as e:
        logger.error(f"❌ Failed to create GPU simulator: {e}")
        raise RuntimeError(f"GPU simulator required but not available: {e}") from e

def create_gpu_estimator(cuda_device=0):
    """Create GPU-accelerated Estimator primitive"""
    import warnings
    try:
        _lazy_qiskit_imports()
    except Exception:
        pass

    optimal_threads = get_optimal_thread_count(cuda_device)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # Create or reuse a GPU-backed AerSimulator and bind the estimator to it
        try:
            simulator = _get_shared_aer_simulator(cuda_device, method='statevector')
        except Exception as e:
            raise RuntimeError(f"Failed to obtain GPU AerSimulator for Estimator: {e}") from e

        base_backend_options = {
            'device': 'GPU',
            'method': 'statevector',
            'precision': 'single',
            'max_parallel_threads': optimal_threads,
            'max_parallel_experiments': 1,
            'batched_shots_gpu': False,
            'blocking_enable': True,
            'blocking_qubits': 5,
        }

        backend_options = dict(base_backend_options)
        backend_options['executor'] = ThreadPoolExecutor(max_workers=4)

        try:
            if AER_ESTIMATOR_ACCEPTS_BACKEND:
                estimator = AerEstimator(backend=simulator, backend_options=backend_options)
            else:
                estimator = AerEstimator(
                    backend_options=base_backend_options,
                    run_options={'shots': 4096}
                )
        except Exception as e:
            raise RuntimeError(f"Failed to create AerEstimator bound to GPU simulator: {e}") from e

        # Validate the estimator's underlying backend/simulator (best-effort)
        try:
            cfg = simulator.configuration()
            backend_name = cfg.backend_name if hasattr(cfg, 'backend_name') else str(cfg)
            if 'gpu' not in backend_name.lower():
                raise RuntimeError(f"AerEstimator backend is not GPU-enabled: {backend_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to validate GPU-backed AerEstimator: {e}") from e

    logger.info(f"✅ GPU Estimator created with {optimal_threads} threads")
    return estimator

def create_gpu_sampler(cuda_device=0, shots=1024):
    """Create GPU-accelerated Sampler primitive"""
    import warnings
    try:
        _lazy_qiskit_imports()
    except Exception:
        pass

    optimal_threads = get_optimal_thread_count(cuda_device)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # Create or reuse a GPU-backed AerSimulator and bind the sampler to it
        try:
            simulator = _get_shared_aer_simulator(cuda_device, method='statevector')
        except Exception as e:
            raise RuntimeError(f"Failed to obtain GPU AerSimulator for Sampler: {e}") from e

        base_backend_options = {
            'device': 'GPU',
            'method': 'statevector',
            'precision': 'single',
            'max_parallel_threads': optimal_threads,
            'max_parallel_experiments': 1,
            'batched_shots_gpu': False,
            'blocking_enable': True,
            'blocking_qubits': 5,
        }

        backend_options = dict(base_backend_options)
        backend_options['executor'] = ThreadPoolExecutor(max_workers=4)

        try:
            if AER_SAMPLER_ACCEPTS_BACKEND:
                sampler = AerSampler(
                    backend=simulator,
                    backend_options=backend_options,
                    run_options={'shots': shots}
                )
            else:
                sampler = AerSampler(
                    backend_options=base_backend_options,
                    run_options={'shots': shots}
                )
        except Exception as e:
            raise RuntimeError(f"Failed to create AerSampler bound to GPU simulator: {e}") from e

        # Validate against the known GPU-backed simulator
        try:
            cfg = simulator.configuration()
            backend_name = cfg.backend_name if hasattr(cfg, 'backend_name') else str(cfg)
            if 'gpu' not in backend_name.lower():
                raise RuntimeError(f"AerSampler backend is not GPU-enabled: {backend_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to validate AerSampler GPU backend: {e}") from e

    logger.info(f"✅ GPU Sampler created with {optimal_threads} threads")
    return sampler


class GPUFidelityKernel:
    """Minimal GPU-backed fidelity kernel using AerSimulator statevectors.

    This class provides an evaluate(X1, X2) -> kernel_matrix interface similar
    to FidelityQuantumKernel but is implemented directly on top of a
    GPU-enabled AerSimulator to avoid primitive binding issues and CPU
    fallbacks.
    """
    def __init__(self, feature_map: 'QuantumCircuit', gpu_id: int = 0,
                 max_circuits_per_eval: Optional[int] = None):
        try:
            _lazy_qiskit_imports()
        except Exception:
            pass

        # Keep a reference to the feature map template (already transpiled)
        self.gpu_id = int(gpu_id)
        self.simulator = _get_shared_aer_simulator(self.gpu_id, method='statevector')
        self.num_qubits = int(getattr(feature_map, 'num_qubits', len(getattr(feature_map, 'qubits', []))) or 0)
        self.max_circuits_per_eval = self._normalize_batch_size(max_circuits_per_eval)

        # Create a copy we can safely mutate (add save_statevector, bind params)
        circuit = copy.deepcopy(feature_map)

        # Resolve blueprint instructions so the GPU backend does not see high-level ops
        try:
            circuit = circuit.decompose()
        except Exception:
            pass

        transpile_fn = globals().get('transpile')
        if transpile_fn is None:
            try:
                from qiskit import transpile as _transpile  # type: ignore
                transpile_fn = _transpile
            except Exception:
                transpile_fn = None

        if transpile_fn is not None:
            try:
                circuit = transpile_fn(
                    circuit,
                    optimization_level=TRANSPILE_OPT_LEVEL,
                    basis_gates=['u', 'cx', 'rz', 'sx', 'x', 'ry']
                )
            except Exception:
                pass

        self._base_circuit = circuit

        # Ensure the circuit saves the statevector exactly once
        try:
            if not any(getattr(inst.operation, 'name', '').lower() == 'save_statevector' for inst in getattr(self._base_circuit, 'data', [])):
                self._base_circuit.save_statevector()
        except Exception:
            # Best-effort: if save_statevector not available, allow simulator to infer
            pass

        # Cache parameter ordering for fast binding
        try:
            self._param_order = list(self._base_circuit.parameters)
        except Exception:
            self._param_order = []

    def set_max_circuits_per_eval(self, value: Optional[int]) -> None:
        """Update batch size bound for simulator runs."""
        self.max_circuits_per_eval = self._normalize_batch_size(value)

    def _normalize_batch_size(self, value: Optional[int]) -> int:
        cap = _fidelity_cap_for_qubits(self.num_qubits)
        try:
            if value is None:
                return cap
            normalized = int(value)
            if normalized <= 0:
                return cap
            return max(FIDELITY_MIN_CIRCUITS, min(cap, normalized))
        except Exception:
            return cap

    def _simulate_batch(self, circuits: List['QuantumCircuit']) -> np.ndarray:
        """Simulate a batch of circuits sequentially and return statevectors as complex64."""
        if not circuits:
            return np.empty((0, 0), dtype=np.complex64)

        batch_count = len(circuits)
        statevecs_np: Optional[np.ndarray] = None

        try:
            job = self.simulator.run(circuits, shots=0)
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"AerSimulator batch run failed in GPUFidelityKernel: {exc}") from exc

        for idx, circuit in enumerate(circuits):
            try:
                sv = result.get_statevector(idx)
            except Exception:
                try:
                    data = result.data(idx)
                except Exception:
                    data = None

                sv = None
                if isinstance(data, dict):
                    sv = data.get('statevector')
                    if sv is None:
                        for key, value in data.items():
                            if isinstance(key, str) and 'state' in key.lower():
                                sv = value
                                break
                if sv is None:
                    raise RuntimeError(f"No statevector returned for circuit index {idx}")

            sv_array = np.asarray(sv, dtype=np.complex128)
            if statevecs_np is None:
                state_dim = sv_array.size
                statevecs_np = np.empty((batch_count, state_dim), dtype=np.complex64)
            statevecs_np[idx, :] = sv_array.astype(np.complex64, copy=False)

        try:
            clear_gpu_memory(self.gpu_id)
        except Exception:
            pass

        if statevecs_np is None:
            return np.empty((0, 0), dtype=np.complex64)

        return statevecs_np

    def evaluate(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate the fidelity kernel between X1 and X2.

        Returns a float32 numpy array of shape (len(X1), len(X2)).
        """
        try:
            with np.errstate(invalid='ignore'):
                X1 = np.asarray(X1, dtype=float)
                if X2 is None:
                    X2 = X1
                else:
                    X2 = np.asarray(X2, dtype=float)
        except Exception:
            raise ValueError("Input arrays must be convertible to numpy arrays of floats")

        same_inputs = X2 is X1

        def _ensure_finite(label: str, array: np.ndarray) -> np.ndarray:
            if np.isfinite(array).all():
                return array
            invalid = int(array.size - np.isfinite(array).sum())
            try:
                logger.warning(
                    "Detected %d non-finite entries in %s kernel inputs; applying safe replacements",
                    invalid,
                    label
                )
            except Exception:
                pass
            return np.nan_to_num(array, nan=0.0, posinf=np.pi, neginf=0.0)

        X1 = _ensure_finite('left', X1)
        if same_inputs:
            X2 = X1
        else:
            X2 = _ensure_finite('right', X2)

        n1 = len(X1)
        n2 = len(X2)

        param_order = self._param_order or []
        if not param_order:
            param_order = list(self._base_circuit.parameters)

        binds_left: List[Dict[Any, float]] = []
        binds_right: List[Dict[Any, float]] = []
        param_len = len(param_order)
        if param_len == 0:
            raise RuntimeError("Feature map contains no parameters to bind for fidelity evaluation")

        for row in X1:
            if len(row) < param_len:
                raise ValueError(f"Input vector length {len(row)} is smaller than parameter count {param_len}")
            binds_left.append({p: float(v) for p, v in zip(param_order, row[:param_len])})

        if same_inputs:
            binds_right = binds_left
        else:
            for row in X2:
                if len(row) < param_len:
                    raise ValueError(f"Input vector length {len(row)} is smaller than parameter count {param_len}")
                binds_right.append({p: float(v) for p, v in zip(param_order, row[:param_len])})

        # Bind parameters per row to avoid Aer parameter_binds restrictions
        circuits_left: List['QuantumCircuit'] = []
        for bind in binds_left:
            try:
                circuits_left.append(self._base_circuit.assign_parameters(bind, inplace=False))
            except Exception as exc:
                raise RuntimeError(f"Failed to assign parameters for fidelity evaluation: {exc}") from exc

        if same_inputs:
            circuits_right = circuits_left
        else:
            circuits_right: List['QuantumCircuit'] = []
            for bind in binds_right:
                try:
                    circuits_right.append(self._base_circuit.assign_parameters(bind, inplace=False))
                except Exception as exc:
                    raise RuntimeError(f"Failed to assign parameters for fidelity evaluation: {exc}") from exc

        cap = _fidelity_cap_for_qubits(self.num_qubits)
        batch_limit = max(FIDELITY_MIN_CIRCUITS, self.max_circuits_per_eval or cap)

        def _batch_slices(total: int) -> List[slice]:
            return [slice(start, min(start + batch_limit, total)) for start in range(0, total, batch_limit)]

        left_batches: List[Tuple[slice, np.ndarray]] = []
        slice_state_map: Dict[Tuple[int, int], np.ndarray] = {}
        for sl in _batch_slices(n1):
            batch_circuits = circuits_left[sl]
            batch_states = self._simulate_batch(batch_circuits)
            left_batches.append((sl, batch_states))
            slice_state_map[(sl.start, sl.stop)] = batch_states

        cupy_mod = _ensure_cupy()
        if cupy_mod is None:
            raise RuntimeError('CuPy became unavailable in GPU fidelity kernel')

        result = np.zeros((n1, n2), dtype=np.float32)
        right_slices = _batch_slices(n2)

        for r_slice in right_slices:
            if same_inputs:
                right_states_np = slice_state_map[(r_slice.start, r_slice.stop)]
            else:
                batch_circuits = circuits_right[r_slice]
                right_states_np = self._simulate_batch(batch_circuits)

            with cupy_mod.cuda.Device(self.gpu_id):
                right_gpu = cupy_mod.asarray(right_states_np, dtype=cupy_mod.complex64)

            for l_slice, left_states_np in left_batches:
                with cupy_mod.cuda.Device(self.gpu_id):
                    left_gpu = cupy_mod.asarray(left_states_np, dtype=cupy_mod.complex64)
                    block_gpu = cupy_mod.abs(left_gpu @ right_gpu.conj().T) ** 2
                    block = cupy_mod.asnumpy(block_gpu).astype(np.float32, copy=False)
                result[l_slice.start:l_slice.stop, r_slice.start:r_slice.stop] = block
                try:
                    del left_gpu
                    cupy_mod.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            try:
                del right_gpu
                if not same_inputs:
                    del right_states_np
                cupy_mod.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

        if same_inputs:
            result = (result + result.T) / 2.0

        try:
            gc.collect()
        except Exception:
            pass

        try:
            clear_gpu_memory(self.gpu_id)
        except Exception:
            pass

        return np.asarray(result, dtype=np.float32)


class CPUApproximateFidelityKernel:
    """Approximate fidelity kernel using CPU/tensor-network simulators.

    This kernel is used when GPU execution is not possible. It mirrors the
    GPUFidelityKernel interface but relies on CPU simulators such as
    tensor-network or matrix-product-state backends.
    """

    def __init__(self, feature_map: 'QuantumCircuit', method: str = 'tensor_network',
                 max_circuits_per_eval: Optional[int] = None):
        try:
            _lazy_qiskit_imports()
        except Exception:
            pass

        requested_method = (method or 'tensor_network').strip().lower()
        method_candidates: List[str]
        if requested_method in ('tensor_network', 'tensor'):
            method_candidates = ['matrix_product_state', 'statevector']
        elif requested_method in ('matrix_product_state', 'mps'):
            method_candidates = ['matrix_product_state', 'statevector']
        else:
            method_candidates = [requested_method]
            if requested_method != 'statevector':
                method_candidates.append('statevector')

        self.simulator = None
        self.method = requested_method
        last_error: Optional[BaseException] = None

        for candidate in method_candidates:
            try:
                simulator = AerSimulator(method=candidate, precision='single')
                try:
                    simulator.set_options(device='CPU')
                except Exception:
                    pass
                self.simulator = simulator
                self.method = candidate
                if candidate != requested_method:
                    try:
                        logger.warning(
                            "CPU fallback switched simulator method to '%s' (requested '%s')",
                            candidate,
                            requested_method,
                        )
                    except Exception:
                        pass
                break
            except Exception as exc:
                last_error = exc

        if self.simulator is None:
            raise RuntimeError(
                f"Unable to initialise CPU AerSimulator for methods {method_candidates}: {last_error}"
            )

        self._base_circuit = copy.deepcopy(feature_map)
        try:
            if not any(getattr(inst.operation, 'name', '').lower() == 'save_statevector' for inst in getattr(self._base_circuit, 'data', [])):
                self._base_circuit.save_statevector()
        except Exception:
            pass

        try:
            self._param_order = list(self._base_circuit.parameters)
        except Exception:
            self._param_order = []

        self.max_circuits_per_eval = self._normalize_batch_size(max_circuits_per_eval)

    def set_max_circuits_per_eval(self, value: Optional[int]) -> None:
        self.max_circuits_per_eval = self._normalize_batch_size(value)

    def _normalize_batch_size(self, value: Optional[int]) -> int:
        cap = max(1, _fidelity_cap_for_qubits(self.num_qubits))
        try:
            if value is None:
                return cap
            normalized = int(value)
            if normalized <= 0:
                return cap
            return max(1, min(cap, normalized))
        except Exception:
            return cap

    def _simulate_batch(self, circuits: List['QuantumCircuit']) -> np.ndarray:
        if not circuits:
            return np.empty((0, 0), dtype=np.complex64)

        batch_count = len(circuits)
        statevecs_np: Optional[np.ndarray] = None

        try:
            job = self.simulator.run(circuits, shots=0)
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"CPU simulator batch run failed: {exc}") from exc

        for idx, circuit in enumerate(circuits):
            try:
                sv = result.get_statevector(idx)
            except Exception:
                try:
                    data = result.data(idx)
                except Exception:
                    data = None

                sv = None
                if isinstance(data, dict):
                    sv = data.get('statevector')
                    if sv is None:
                        for key, value in data.items():
                            if isinstance(key, str) and 'state' in key.lower():
                                sv = value
                                break
                if sv is None:
                    raise RuntimeError(f"No statevector returned for circuit index {idx}")

            sv_array = np.asarray(sv, dtype=np.complex128)
            if statevecs_np is None:
                state_dim = sv_array.size
                statevecs_np = np.empty((batch_count, state_dim), dtype=np.complex64)
            statevecs_np[idx, :] = sv_array.astype(np.complex64, copy=False)

        if statevecs_np is None:
            return np.empty((0, 0), dtype=np.complex64)

        return statevecs_np

    def evaluate(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        try:
            X1 = np.asarray(X1, dtype=float)
            if X2 is None:
                X2 = X1
            else:
                X2 = np.asarray(X2, dtype=float)
        except Exception:
            raise ValueError("Input arrays must be convertible to numpy arrays of floats")

        n1 = len(X1)
        n2 = len(X2)
        same_inputs = X2 is X1

        param_order = self._param_order or list(self._base_circuit.parameters)
        if not param_order:
            raise RuntimeError("Feature map contains no parameters to bind for fidelity evaluation")

        binds_left: List[Dict[Any, float]] = []
        binds_right: List[Dict[Any, float]] = []
        param_len = len(param_order)

        for row in X1:
            if len(row) < param_len:
                raise ValueError(f"Input vector length {len(row)} is smaller than parameter count {param_len}")
            binds_left.append({p: float(v) for p, v in zip(param_order, row[:param_len])})

        if same_inputs:
            binds_right = binds_left
        else:
            for row in X2:
                if len(row) < param_len:
                    raise ValueError(f"Input vector length {len(row)} is smaller than parameter count {param_len}")
                binds_right.append({p: float(v) for p, v in zip(param_order, row[:param_len])})

        def _parameterize_circuit(bind_map: Dict[Any, float]):
            circuit = copy.deepcopy(self._base_circuit)
            if hasattr(circuit, "bind_parameters"):
                return circuit.bind_parameters(bind_map)
            if hasattr(circuit, "assign_parameters"):
                return circuit.assign_parameters(bind_map, inplace=False)
            raise RuntimeError("Feature map does not support parameter assignment on this Qiskit version")

        circuits_left = [_parameterize_circuit(bind) for bind in binds_left]

        if same_inputs:
            circuits_right = circuits_left
        else:
            circuits_right = [_parameterize_circuit(bind) for bind in binds_right]

        cap = max(1, _fidelity_cap_for_qubits(self.num_qubits))
        batch_limit = max(1, self.max_circuits_per_eval or cap)

        def _batch_slices(total: int) -> List[slice]:
            return [slice(start, min(start + batch_limit, total)) for start in range(0, total, batch_limit)]

        left_batches: List[Tuple[slice, np.ndarray]] = []
        slice_state_map: Dict[Tuple[int, int], np.ndarray] = {}
        for sl in _batch_slices(n1):
            batch_circuits = circuits_left[sl]
            batch_states = self._simulate_batch(batch_circuits)
            left_batches.append((sl, batch_states))
            slice_state_map[(sl.start, sl.stop)] = batch_states

        result = np.zeros((n1, n2), dtype=np.float32)
        right_slices = _batch_slices(n2)

        for r_slice in right_slices:
            if same_inputs:
                right_states_np = slice_state_map[(r_slice.start, r_slice.stop)]
            else:
                batch_circuits = circuits_right[r_slice]
                right_states_np = self._simulate_batch(batch_circuits)

            right_states_conj = right_states_np.conj().T
            for l_slice, left_states_np in left_batches:
                block = np.abs(left_states_np @ right_states_conj) ** 2
                result[l_slice.start:l_slice.stop, r_slice.start:r_slice.stop] = block.astype(np.float32, copy=False)

        if same_inputs:
            result = (result + result.T) / 2.0

        try:
            gc.collect()
        except Exception:
            pass

        return np.asarray(result, dtype=np.float32)


if RAY_AVAILABLE:

    @ray.remote(num_gpus=1)
    class _RayGPUKernelActor:
        def __init__(self, approx_mode: str = 'auto', max_circuits: Optional[int] = None):
            self.approx_mode = (approx_mode or 'auto').strip().lower()
            self.max_circuits_default = max_circuits
            self._kernel_cache: Dict[str, GPUFidelityKernel] = {}
            self._kernel_max: Dict[str, Optional[int]] = {}
            self._lazy_imports = _ray_get_module_attr('_lazy_qiskit_imports')
            if callable(self._lazy_imports):
                try:
                    self._lazy_imports()
                except Exception:
                    pass
            self._build_feature_map = _ray_get_module_attr('_build_feature_map_from_key', _build_feature_map_from_key)
            self._gpu_kernel_cls = _ray_get_module_attr('GPUFidelityKernel', GPUFidelityKernel)
            self._clear_gpu_memory = _ray_get_module_attr('clear_gpu_memory', clear_gpu_memory)

        def _resolve_feature_map(self, circuit_key: str, circuit_qpy: Optional[bytes], num_qubits: int) -> QuantumCircuit:
            feature_map = None
            if circuit_qpy:
                try:
                    import io
                    from qiskit.qpy import load
                    buf = io.BytesIO(circuit_qpy)
                    circuits = load(buf)
                    if isinstance(circuits, (list, tuple)) and circuits:
                        feature_map = circuits[0]
                except Exception:
                    feature_map = None
            if feature_map is None:
                feature_map = _handle_missing_precompiled_circuit(
                    circuit_key,
                    num_qubits,
                    'feature map missing in Ray actor hydration'
                )
            return feature_map

        def _get_kernel(self, circuit_key: str, circuit_qpy: Optional[bytes], num_qubits: int, max_circuits: Optional[int]) -> GPUFidelityKernel:
            kernel = self._kernel_cache.get(circuit_key)
            target_max = max_circuits or self._kernel_max.get(circuit_key) or self.max_circuits_default
            if target_max is not None and target_max <= 0:
                target_max = 1

            if kernel is None:
                feature_map = self._resolve_feature_map(circuit_key, circuit_qpy, num_qubits)
                kernel_cls = self._gpu_kernel_cls or GPUFidelityKernel
                kernel = kernel_cls(
                    feature_map=feature_map,
                    gpu_id=0,
                    max_circuits_per_eval=target_max
                )
                self._kernel_cache[circuit_key] = kernel
            elif target_max is not None:
                kernel.set_max_circuits_per_eval(target_max)

            self._kernel_max[circuit_key] = target_max
            return kernel

        def runtime_info(self) -> Dict[str, Any]:
            info: Dict[str, Any] = {'node_id': None, 'node_ip': None}
            try:
                ctx = ray.get_runtime_context()  # type: ignore[attr-defined]
                node_id = None
                if hasattr(ctx, 'get_node_id') and callable(ctx.get_node_id):  # type: ignore[attr-defined]
                    node_id = ctx.get_node_id()  # type: ignore[attr-defined]
                else:
                    node_id = getattr(ctx, 'node_id', None)
                if node_id is not None:
                    info['node_id'] = str(node_id)
            except Exception:
                pass
            try:
                from ray.util import get_node_ip_address  # type: ignore
                info['node_ip'] = get_node_ip_address()
            except Exception:
                try:
                    info['node_ip'] = socket.gethostbyname(socket.gethostname())
                except Exception:
                    info['node_ip'] = None
            return info

        def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            X1_chunk = np.asarray(payload['X1_chunk'], dtype=np.float32)
            if 'X2_ref' in payload:
                X2 = ray.get(payload['X2_ref'])  # type: ignore[attr-defined]
            else:
                X2 = payload['X2']
            X2_arr = np.asarray(X2, dtype=np.float32)

            num_qubits = int(payload.get('num_qubits', 0) or 0)
            if num_qubits <= 0:
                if X1_chunk.ndim > 1:
                    num_qubits = X1_chunk.shape[1]
                elif X2_arr.ndim > 1:
                    num_qubits = X2_arr.shape[1]

            kernel = self._get_kernel(
                payload['circuit_key'],
                payload.get('circuit_qpy'),
                num_qubits,
                payload.get('max_circuits')
            )

            block = kernel.evaluate(X1_chunk, X2_arr)
            block = np.asarray(np.real(block), dtype=np.float32)
            try:
                if callable(self._clear_gpu_memory):
                    self._clear_gpu_memory(0)
                else:
                    clear_gpu_memory(0)
            except Exception:
                pass

            return {
                'chunk_idx': payload['chunk_idx'],
                'chunk_end': payload['chunk_end'],
                'kernel': block
            }

    if _MODULE_IMPORT_NAME:
        try:
            _RayGPUKernelActor.__module__ = _MODULE_IMPORT_NAME
        except Exception:
            pass

else:

    _RayGPUKernelActor = None

def create_quantum_kernel(feature_map, cuda_device=0, max_circuits_per_eval: Optional[int] = None):
    """Create a quantum kernel using GPU or approximate simulators."""
    try:
        try:
            _lazy_qiskit_imports()
        except Exception:
            pass

        mode = (APPROXIMATION_MODE or 'gpu_strict').strip().lower()
        gpu_only_modes = {'', 'auto', 'gpu', 'gpu-only', 'gpu_strict'}
        if mode not in gpu_only_modes:
            raise RuntimeError(
                f"CPU-based approximation modes are disabled (received '{mode}'). Set IOT_APPROX_MODE to a GPU mode instead."
            )

        return GPUFidelityKernel(
            feature_map=feature_map,
            gpu_id=cuda_device,
            max_circuits_per_eval=max_circuits_per_eval
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create a GPU fidelity kernel while CPU fallbacks are disabled: {e}"
        ) from e


def _create_transient_fidelity_kernel(circuit_key: str, num_qubits: int, gpu_id: int = 0,
                                      feature_map: Optional[QuantumCircuit] = None,
                                      max_circuits_per_eval: Optional[int] = None):
    """Create a FidelityQuantumKernel without inserting it into any cache.

    Use this for very large-qubit runs where keeping kernel objects in a cache
    can consume excessive device memory. Callers are responsible for cleanup
    (preferably via _safe_cleanup) after use.
    """
    if feature_map is None:
        feature_map = _handle_missing_precompiled_circuit(
            circuit_key,
            num_qubits,
            'feature map not present when creating transient kernel'
        )
    # Create a GPU-backed kernel or raise; do not return a CPU fallback.
    k = create_quantum_kernel(
        feature_map=feature_map,
        cuda_device=gpu_id,
        max_circuits_per_eval=max_circuits_per_eval
    )
    try:
        logger.debug(f"[GPU:{gpu_id}] Created transient FidelityQuantumKernel for key={circuit_key} qubits={num_qubits}")
    except Exception:
        pass
    return k


# =========================================================================
# SHARED GPU SIMULATOR / KERNEL HELPERS
# =========================================================================

# IMPORTANT: process-local caches only. Each process (main or worker) will keep
# its own small cache of simulators/kernels. We avoid storing GPU-bound objects
# in a central sharable structure to prevent cross-process reuse and OOM.
#
# _PROCESS_LOCAL_SIMULATORS and _PROCESS_LOCAL_FIDELITY_KERNELS are plain
# dicts keyed by (gpu_id, method) and (gpu_id, circuit_key) respectively.
# They are process-global (module-level) but are not safe to share across
# processes — which is desirable here.
_PROCESS_LOCAL_SIMULATORS: Dict[Tuple[int, str], Any] = {}
_PROCESS_LOCAL_SIMULATORS_LOCK = threading.Lock()
_PROCESS_LOCAL_FIDELITY_KERNELS: Dict[Tuple[int, str], Any] = {}
_PROCESS_LOCAL_FIDELITY_KERNELS_LOCK = threading.Lock()
_PROCESS_LOCAL_KERNEL_COMPUTERS: Dict[int, 'SharedGPUKernelComputer'] = {}
_PROCESS_LOCAL_KERNEL_COMPUTERS_LOCK = threading.Lock()

# Global counters and light-weight metadata caches (picklable) can remain if
# needed, but must not hold heavy GPU objects. We keep light OrderedDicts for
# bookkeeping only.
_SIMULATOR_META: Dict[Tuple[int, str], Dict[str, Any]] = OrderedDict()
_FIDELITY_KERNEL_META: Dict[Tuple[int, str], Dict[str, Any]] = OrderedDict()
_SIMULATOR_LOCK = threading.Lock()
_FIDELITY_KERNEL_LOCK = threading.Lock()

# Cache eviction/debugging controls
_CACHE_DEBUG = False
_CACHE_EVICTION_COUNTERS = defaultdict(int)
_CACHE_EVICTED_BYTES = defaultdict(int)
_CACHE_EVICTION_LOCK = threading.Lock()


def _approx_size_bytes(obj) -> int:
    """Cheap size estimate for eviction logging."""
    try:
        if hasattr(obj, 'nbytes'):
            return int(getattr(obj, 'nbytes'))
        if isinstance(obj, (bytes, bytearray)):
            return len(obj)
        # Fallback: try numpy if it's an array-like
        if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
            try:
                return int(np.prod(obj.shape) * np.dtype(obj.dtype).itemsize)
            except Exception:
                pass
    except Exception:
        pass
    return 0


def _log_eviction(cache_name: str, key: Any, evicted_obj: Any, reason: str = 'evicted'):
    """Log and increment counters for a cache eviction in a lightweight way."""
    size = _approx_size_bytes(evicted_obj)
    with _CACHE_EVICTION_LOCK:
        _CACHE_EVICTION_COUNTERS[cache_name] += 1
        _CACHE_EVICTED_BYTES[cache_name] += size

    if _CACHE_DEBUG:
        try:
            rss_mb = None
            try:
                rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            except Exception:
                pass

            logger.debug(
                "evict: cache=%s key=%s size=%sB reason=%s rss_mb=%s",
                cache_name,
                str(key)[:120],
                size,
                reason,
                f"{rss_mb:.1f}" if rss_mb is not None else "?"
            )
        except Exception:
            pass


def _safe_cleanup(evicted_obj: Any, gpu_hint: Optional[int] = None):
    """Attempt best-effort cleanup of objects holding GPU resources.

    We avoid raising errors here; the goal is to release references and call
    any available close/shutdown methods so device memory/pools are freed.
    """
    try:
        # Common close/shutdown hooks
        if hasattr(evicted_obj, 'close') and callable(getattr(evicted_obj, 'close')):
            try:
                evicted_obj.close()
            except Exception:
                pass
    except Exception:
        pass
    try:
        if hasattr(evicted_obj, 'shutdown') and callable(getattr(evicted_obj, 'shutdown')):
            try:
                evicted_obj.shutdown()
            except Exception:
                pass
    except Exception:
        pass

    # If object contains a sampler or backend, try to close it as well
    try:
        if hasattr(evicted_obj, 'sampler'):
            s = getattr(evicted_obj, 'sampler')
            if hasattr(s, 'close') and callable(getattr(s, 'close')):
                try:
                    s.close()
                except Exception:
                    pass
    except Exception:
        pass

    # Delete reference and force a GC cycle
    try:
        del evicted_obj
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        # Best-effort free process-local caches if RSS is high
        pass
    except Exception:
        pass

    # Best-effort free GPU memory hint
    try:
        if gpu_hint is not None:
            clear_gpu_memory(gpu_hint)
    except Exception:
        pass


def _get_or_create_shared_kernel_computer(
    circuit_manager: 'CentralizedGPUCircuitManager',
    gpu_id: int = 0,
    batch_size: int = 1024,
):
    """Return a process-local SharedGPUKernelComputer for the given GPU.

    Reusing the kernel computer lets us persist kernel caches across model
    variants that operate on the same dataset, avoiding redundant 𝑁×𝑁 kernel
    builds on small runs.
    """

    with _PROCESS_LOCAL_KERNEL_COMPUTERS_LOCK:
        existing = _PROCESS_LOCAL_KERNEL_COMPUTERS.get(gpu_id)
        if existing is not None and getattr(existing, 'circuit_manager', None) is circuit_manager:
            if batch_size and getattr(existing, 'batch_size', batch_size) != batch_size:
                existing.batch_size = batch_size
            return existing

        if existing is not None:
            try:
                _safe_cleanup(existing, gpu_hint=gpu_id)
            except Exception:
                pass

        kernel_computer = SharedGPUKernelComputer(
            circuit_manager,
            gpu_id=gpu_id,
            batch_size=batch_size
        )
        _PROCESS_LOCAL_KERNEL_COMPUTERS[gpu_id] = kernel_computer
        return kernel_computer



# -----------------------------
# Shared memory helpers
# -----------------------------
def create_shared_memory_for_array(arr: np.ndarray, name: Optional[str] = None) -> Dict[str, Any]:
    """Create a multiprocessing.SharedMemory block for a numpy array and return metadata."""
    if not SHARED_MEMORY_AVAILABLE or shared_memory is None:
        raise RuntimeError("Shared memory support is unavailable on this platform")
    # Ensure float32 for compactness
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        try:
            invalid = int(arr.size - np.isfinite(arr).sum())
            logger.warning("Shared memory staging found %d non-finite entries; sanitizing before publish", invalid)
        except Exception:
            pass
        arr = np.nan_to_num(arr, nan=0.0, posinf=np.pi, neginf=0.0)
    # Create shared memory region sized for the array
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    # Use a memoryview assignment to avoid intermediate copies when possible
    try:
        shm_arr[:] = arr
    except Exception:
        # Fallback to element-wise copy if necessary
        for i in range(arr.shape[0]):
            shm_arr[i] = arr[i]
    return {'name': shm.name, 'shape': arr.shape, 'dtype': str(arr.dtype), 'nbytes': arr.nbytes}


def free_shared_memory(info: Dict[str, Any]):
    if not SHARED_MEMORY_AVAILABLE or shared_memory is None:
        return
    try:
        shm = shared_memory.SharedMemory(name=info['name'])
        shm.close()
        shm.unlink()
    except Exception:
        pass


def attach_shared_memory(info: Dict[str, Any], *, copy_local: bool = False) -> Tuple[np.ndarray, Optional[Any]]:
    """Attach to an existing shared memory block.

    Returns a tuple of (numpy_array, shared_memory_handle). When copy_local=True the
    data is copied into process-local memory and the handle is closed immediately,
    so the caller receives (array, None).
    """
    if not SHARED_MEMORY_AVAILABLE or shared_memory is None:
        raise RuntimeError("Shared memory support is unavailable on this platform")
    shm_obj = shared_memory.SharedMemory(name=info['name'])
    dtype = np.dtype(info['dtype'])
    arr = np.ndarray(tuple(info['shape']), dtype=dtype, buffer=shm_obj.buf)
    if copy_local:
        local = np.array(arr, copy=True)
        try:
            shm_obj.close()
        except Exception:
            pass
        return local, None
    return arr, shm_obj


# -----------------------------
# GPU Worker process
# -----------------------------
def _gpu_worker_loop(task_queue: MPQueue, result_queue: MPQueue, shm_registry: Dict[str, Any], gpu_id: int):
    """Worker loop that runs in a separate process pinned to a single GPU."""
    import sys
    import traceback
    
    try:
        # ENTIRE WORKER LOGIC IN MASSIVE TRY-CATCH
        _gpu_worker_loop_impl(task_queue, result_queue, shm_registry, gpu_id)
    except Exception as fatal_err:
        # Write to stderr (will be captured by parent)
        print(f"[FATAL WORKER CRASH pid:{os.getpid()} GPU:{gpu_id}] {fatal_err}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        
        # Also try to put error in queue
        try:
            result_queue.put({'error': f'FATAL: {fatal_err}', 'gpu_id': gpu_id, 'chunk_idx': -9999})
        except Exception:
            pass
        sys.exit(1)


def _gpu_worker_loop_impl(task_queue: MPQueue, result_queue: MPQueue, shm_registry: Dict[str, Any], gpu_id: int):
    """Actual worker loop implementation."""
    # Set device visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Quiet noisy INFO logs in GPU worker processes (they are separate processes)
    try:
        import logging as _logging
        if mp.current_process().name != 'MainProcess':
            _logging.getLogger().setLevel(_logging.WARNING)
            _logging.getLogger('qiskit').setLevel(_logging.WARNING)
            _logging.getLogger('qiskit.transpiler').setLevel(_logging.WARNING)
    except Exception:
        pass
    try:
        # Initialize NVML or CuPy if needed inside worker
        _ensure_cupy()
    except Exception:
        pass

    # Worker-local logger
    try:
        import logging as _logging
        worker_logger = _logging.getLogger(f"gpu_worker_{gpu_id}")
        # Ensure worker INFO logs are emitted even if root logger level was lowered
        try:
            worker_logger.setLevel(_logging.INFO)
            if not worker_logger.handlers:
                ch = _logging.StreamHandler()
                ch.setLevel(_logging.INFO)
                fmt = _logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
                ch.setFormatter(fmt)
                worker_logger.addHandler(ch)
            # Prevent double emission through the root logger
            worker_logger.propagate = False
        except Exception:
            pass
    except Exception:
        worker_logger = logger

    # Lazy-import Qiskit primitives inside worker (after we set logging levels)
    try:
        _lazy_qiskit_imports()
    except Exception:
        pass

    # Map physical GPU -> local visible device index after setting CUDA_VISIBLE_DEVICES.
    # When CUDA_VISIBLE_DEVICES is set to the physical id (e.g. '1'), CUDA runtimes
    # will remap visible devices so the first visible device is index 0. Therefore
    # GPU libraries (CuPy, Qiskit Aer) should be given local index 0 inside the
    # worker process. We keep `gpu_id` (physical id) for logging only.
    local_cuda_device = 0

    # Create shared simulator and kernel caches local to this worker (use local index)
    local_sim = _get_shared_aer_simulator(local_cuda_device)
    # Log simulator backend in the worker for diagnostics
    try:
        try:
            cfg = local_sim.configuration()
            backend_name = getattr(cfg, 'backend_name', None)
            worker_logger.info(f"[worker pid:{os.getpid()}][GPU:{gpu_id}] AerSimulator backend_name={backend_name}")
        except Exception as e:
            worker_logger.warning(f"[worker pid:{os.getpid()}][GPU:{gpu_id}] AerSimulator config read failed: {e}")
    except Exception:
        pass
    local_kernel_cache: Dict[str, FidelityQuantumKernel] = {}
    try:
        worker_logger.info(f"[GPU:{gpu_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} -> using local_device={local_cuda_device} for GPU APIs")
    except Exception:
        pass

    # Attach shared memory arrays once and optionally stage to device
    local_host_arrays: Dict[str, np.ndarray] = {}
    local_shm_handles: Dict[str, Any] = {}
    local_device_arrays: Dict[str, Any] = {}
    staged_to_device = False
    try:
        if 'X1' in shm_registry and 'X2' in shm_registry:
            x1_arr, x1_handle = attach_shared_memory(shm_registry['X1'])
            x2_arr, x2_handle = attach_shared_memory(shm_registry['X2'])
            local_host_arrays['X1'] = x1_arr
            local_host_arrays['X2'] = x2_arr
            if x1_handle is not None:
                local_shm_handles['X1'] = x1_handle
            if x2_handle is not None:
                local_shm_handles['X2'] = x2_handle
            # DISABLED: Skip staging to device memory - too risky for 22+ qubits
            # The overhead of staging arrays to GPU and CuPy operations is causing crashes
            staged_to_device = False
            try:
                worker_logger.info(f"[GPU:{gpu_id}] Attached shared memory arrays: X1={local_host_arrays['X1'].shape}, X2={local_host_arrays['X2'].shape}")
            except Exception:
                pass
    except Exception as shm_err:
        worker_logger.error(f"[GPU:{gpu_id}] FAILED to attach shared memory: {shm_err}")
        # If shared memory not provided or attach fails, we'll attach on demand per task
        local_host_arrays = {}
        local_device_arrays = {}
        staged_to_device = False

    while True:
        try:
            task = task_queue.get()
            if task is None:
                worker_logger.info(f"[GPU:{gpu_id}] Received None task (shutdown signal), exiting worker")
                break
            worker_logger.info(f"[GPU:{gpu_id}] Received task with action={task.get('action')}")
        except Exception as task_err:
            worker_logger.error(f"[GPU:{gpu_id}] FAILED to receive task: {task_err}")
            import traceback
            worker_logger.error(traceback.format_exc())
            break

        try:
            action = task.get('action')
            if action == 'eval_block':
                circuit_key = task['circuit_key']
                circuit_qpy = task.get('circuit_qpy')
                X1_info = task['X1_shm']
                X2_info = task['X2_shm']
                s1, e1 = task['slice1']
                s2, e2 = task['slice2']
                max_circuits = task.get('max_circuits')

                # Use cached attached arrays if present
                if staged_to_device:
                    cupy_mod = _ensure_cupy()
                    if cupy_mod is None:
                        raise RuntimeError('CuPy became unavailable in GPU worker')
                    X1_chunk = cupy_mod.asnumpy(local_device_arrays['X1'][s1:e1, :])
                    X2_chunk = cupy_mod.asnumpy(local_device_arrays['X2'][s2:e2, :])
                else:
                    if 'X1' in local_host_arrays and 'X2' in local_host_arrays:
                        # CRITICAL FIX: Extract ROWS from shared memory arrays
                        # X1 shape: (n1, n_features), we want rows s1:e1
                        # X2 shape: (n2, n_features), we want rows s2:e2
                        X1_chunk = local_host_arrays['X1'][s1:e1, :]
                        X2_chunk = local_host_arrays['X2'][s2:e2, :]
                    else:
                        X1_shm_arr, X1_handle = attach_shared_memory(X1_info, copy_local=True)
                        X2_shm_arr, X2_handle = attach_shared_memory(X2_info, copy_local=True)
                        X1_chunk = X1_shm_arr[s1:e1, :]
                        X2_chunk = X2_shm_arr[s2:e2, :]
                        if X1_handle is not None:
                            try:
                                X1_handle.close()
                            except Exception:
                                pass
                        if X2_handle is not None:
                            try:
                                X2_handle.close()
                            except Exception:
                                pass

                # EMERGENCY: For 22+ qubits, force max_circuits to 1 to prevent segfaults
                num_qubits = task.get('num_qubits', 0)
                chunk_rows = e1 - s1
                chunk_cols = e2 - s2
                if num_qubits >= 22:
                    max_circuits = 1
                    try:
                        worker_logger.warning(f"[GPU:{gpu_id}] EMERGENCY: qubits={num_qubits}, chunk_size={chunk_rows}×{chunk_cols}, forcing max_circuits=1")
                    except Exception:
                        pass
                else:
                    try:
                        worker_logger.info(f"[GPU:{gpu_id}] Processing chunk {chunk_rows}×{chunk_cols} (qubits={num_qubits}, max_circuits={max_circuits})")
                    except Exception:
                        pass

                # Hydrate fidelity kernel for circuit_key if needed
                if circuit_key in local_kernel_cache:
                    kernel = local_kernel_cache[circuit_key]
                    if max_circuits is not None and hasattr(kernel, 'set_max_circuits_per_eval'):
                        try:
                            kernel.set_max_circuits_per_eval(max_circuits)
                        except Exception:
                            pass
                else:
                    if circuit_qpy:
                        try:
                            # load transpiled circuit from qpy bytes
                            import io
                            from qiskit.qpy import load
                            buf = io.BytesIO(circuit_qpy)
                            circuits = load(buf)
                            if not isinstance(circuits, (list, tuple)) or len(circuits) == 0:
                                try:
                                    worker_logger.warning(
                                        f"[worker pid:{os.getpid()}][GPU:{gpu_id}] Invalid qpy load for {circuit_key}: returned {type(circuits)} len={len(circuits) if hasattr(circuits, '__len__') else 'n/a'} size={len(circuit_qpy)}"
                                    )
                                except Exception:
                                    pass
                                raise ValueError("qpy.load did not return a non-empty list of circuits")

                            feature_map = circuits[0]
                        except Exception:
                            feature_map = _handle_missing_precompiled_circuit(
                                circuit_key,
                                num_qubits,
                                'feature map qpy hydration failed in GPU worker'
                            )
                    else:
                        feature_map = _handle_missing_precompiled_circuit(
                            circuit_key,
                            num_qubits,
                            'feature map not provided to GPU worker'
                        )

                    # Create kernel bound to the local visible device index (0) because
                    # CUDA_VISIBLE_DEVICES has been set to the physical GPU id above.
                    kernel = create_quantum_kernel(
                        feature_map,
                        cuda_device=local_cuda_device,
                        max_circuits_per_eval=max_circuits
                    )
                    local_kernel_cache[circuit_key] = kernel

                # Evaluate kernel block
                # Ensure numpy arrays for kernel evaluation
                if not isinstance(X1_chunk, np.ndarray):
                    X1_chunk = np.asarray(X1_chunk, dtype=np.float32)
                if not isinstance(X2_chunk, np.ndarray):
                    X2_chunk = np.asarray(X2_chunk, dtype=np.float32)

                try:
                    worker_logger.info(f"[GPU:{gpu_id}] About to evaluate kernel: X1={X1_chunk.shape}, X2={X2_chunk.shape}, qubits={num_qubits}")
                    kb = kernel.evaluate(X1_chunk.astype(np.float32), X2_chunk.astype(np.float32))
                    kb = np.asarray(np.real(kb), dtype=np.float32)
                    worker_logger.info(f"[GPU:{gpu_id}] Kernel evaluation SUCCESS: shape={kb.shape}")
                except Exception as eval_err:
                    worker_logger.error(f"[GPU:{gpu_id}] KERNEL EVALUATE CRASHED: {type(eval_err).__name__}: {eval_err}")
                    import traceback
                    worker_logger.error(traceback.format_exc())
                    raise

                result_queue.put({'chunk_idx': task['chunk_idx'], 'chunk_end': task['chunk_end'], 'kernel': kb, 'gpu_id': gpu_id})

        except Exception as e:
            result_queue.put({'error': str(e), 'chunk_idx': task.get('chunk_idx', -1), 'gpu_id': gpu_id})
    # cleanup
    try:
        if staged_to_device:
            cupy_mod = _ensure_cupy()
            if cupy_mod is not None:
                del local_device_arrays['X1']
                del local_device_arrays['X2']
                cupy_mod.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        for handle in local_shm_handles.values():
            try:
                handle.close()
            except Exception:
                pass
    except Exception:
        pass
    local_host_arrays.clear()


class GPUWorkerManager:
    """Manager for GPU worker processes (one dedicated process per GPU id)."""

    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = list(gpu_ids)
        self.task_queues: Dict[int, MPQueue] = {}
        self.result_queue: Optional[MPQueue] = None
        self.procs: Dict[int, Process] = {}

    def start(self, shm_registry: Dict[str, Any]):
        self.task_queues.clear()
        self.procs.clear()
        self.result_queue = mp.Queue()
        for physical_id in self.gpu_ids:
            tq = mp.Queue()
            p = Process(target=_gpu_worker_loop, args=(tq, self.result_queue, shm_registry, physical_id), daemon=True)
            p.start()
            self.task_queues[physical_id] = tq
            self.procs[physical_id] = p

    def stop(self):
        for q in self.task_queues.values():
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs.values():
            p.join(timeout=5)
        self.task_queues.clear()
        self.procs.clear()
        if self.result_queue is not None:
            try:
                self.result_queue.close()
            except Exception:
                pass
            self.result_queue = None

    def submit(self, gpu_id: int, task: Dict[str, Any]):
        queue = self.task_queues.get(gpu_id)
        if queue is None:
            raise ValueError(f"GPU id {gpu_id} is not managed by this worker pool")
        queue.put(task)

    def worker_status(self) -> Dict[int, str]:
        """Return simple status information for managed worker processes."""
        status = {}
        for gid, proc in self.procs.items():
            if proc.is_alive():
                status[gid] = 'alive'
            else:
                exit_code = proc.exitcode
                status[gid] = f'exit:{exit_code}' if exit_code is not None else 'stopped'
        return status

    def any_alive(self) -> bool:
        return any(proc.is_alive() for proc in self.procs.values())

    def reap_terminated(self):
        for proc in self.procs.values():
            if not proc.is_alive():
                try:
                    proc.join(timeout=0.1)
                except Exception:
                    pass



def _build_feature_map_from_key(circuit_key: str, num_qubits: int) -> QuantumCircuit:
    """Reconstruct a feature map from its cache key."""
    try:
        _lazy_qiskit_imports()
    except Exception:
        pass

    if '_reps' not in circuit_key:
        raise ValueError(f"Invalid circuit_key format: {circuit_key}")

    feature_map_type, reps_str = circuit_key.split('_reps')
    reps = int(reps_str)

    if feature_map_type == 'Z':
        feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=reps)
    elif feature_map_type == 'ZZ':
        feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement='full')
    elif feature_map_type == 'Pauli':
        feature_map = PauliFeatureMap(feature_dimension=num_qubits, reps=reps, paulis=['Z', 'ZZ'])
    else:
        raise ValueError(f"Unknown feature map type: {feature_map_type}")

    try:
        # Resolve blueprint instructions before transpile when supported
        feature_map = feature_map.decompose()
    except Exception:
        pass

    try:
        compiled = transpile(
            feature_map,
            optimization_level=TRANSPILE_OPT_LEVEL,
            basis_gates=['u', 'cx', 'rz', 'sx', 'x', 'ry']
        )
        return compiled
    except Exception:
        return feature_map


def _get_shared_aer_simulator(gpu_id: int = 0, method: str = 'statevector') -> AerSimulator:
    """Return a cached AerSimulator configured for the requested GPU."""
    key = (gpu_id, method)
    # Use a process-local simulator cache to avoid cross-process sharing of
    # GPU-bound objects. Each process keeps its own small cache in
    # _PROCESS_LOCAL_SIMULATORS.
    try:
        with _PROCESS_LOCAL_SIMULATORS_LOCK:
            sim = _PROCESS_LOCAL_SIMULATORS.get(key)
            if sim is not None:
                return sim

            try:
                _lazy_qiskit_imports()
            except Exception:
                pass

            simulator = AerSimulator(method=method, device='GPU', precision='single')
            # Validate the backend is actually GPU-enabled. If Aer was built
            # without CUDA support it may return a CPU backend despite the
            # request; fail loudly in that case.
            try:
                cfg = simulator.configuration()
                backend_name = cfg.backend_name if hasattr(cfg, 'backend_name') else str(cfg)
                if 'gpu' not in backend_name.lower():
                    raise RuntimeError(f"AerSimulator backend is not GPU-enabled: {backend_name}")
            except Exception as e:
                raise RuntimeError(f"Failed to validate AerSimulator GPU backend: {e}") from e

            simulator.set_options(
                max_parallel_threads=get_optimal_thread_count(gpu_id),
                max_parallel_experiments=1,
                batched_shots_gpu=False,
                blocking_enable=True,
                blocking_qubits=5,
                precision='single',
                max_memory_mb=16384
            )

            # Store in process-local cache and register lightweight metadata
            _PROCESS_LOCAL_SIMULATORS[key] = simulator
            try:
                logger.debug(f"[GPU:{gpu_id}] Stored AerSimulator in process-local cache key={key}")
            except Exception:
                pass
            try:
                with _SIMULATOR_LOCK:
                    _SIMULATOR_META[key] = {'created_at': time.time(), 'gpu_id': gpu_id}
                    _SIMULATOR_META.move_to_end(key)
            except Exception:
                pass

            # Trim metadata if it grows too large (does not free simulator objects)
            try:
                with _SIMULATOR_LOCK:
                    while len(_SIMULATOR_META) > MAX_SIMULATOR_CACHE_SIZE:
                        _SIMULATOR_META.popitem(last=False)
            except Exception:
                pass

            # Best-effort trim of caches under memory pressure
            try:
                _trim_caches_if_needed()
            except Exception:
                pass

            return simulator
    except Exception as e:
        # Do not silently fall back to a CPU simulator; the user requested
        # GPU-only operation. Raise an explicit error so the caller can stop.
        raise RuntimeError(f"Unable to create a GPU-backed AerSimulator for gpu_id={gpu_id}: {e}") from e


def _get_shared_fidelity_kernel(circuit_key: str, num_qubits: int, gpu_id: int = 0,
                                feature_map: Optional[QuantumCircuit] = None,
                                max_circuits: Optional[int] = None) -> FidelityQuantumKernel:
    """Return a cached FidelityQuantumKernel bound to the GPU simulator."""
    key = (gpu_id, circuit_key)

    # Prefer process-local kernel cache (each process owns its GPU objects)
    try:
        with _PROCESS_LOCAL_FIDELITY_KERNELS_LOCK:
            kernel = _PROCESS_LOCAL_FIDELITY_KERNELS.get(key)
            if kernel is not None:
                if max_circuits is not None and hasattr(kernel, 'set_max_circuits_per_eval'):
                    try:
                        kernel.set_max_circuits_per_eval(max_circuits)
                    except Exception:
                        pass
                return kernel

        # If this is a very-large-qubit kernel, create a transient kernel and
        # do not store it in any long-lived cache to avoid device memory buildup.
        if num_qubits >= LARGE_QUBIT_THRESHOLD:
            if feature_map is None:
                feature_map = _handle_missing_precompiled_circuit(circuit_key, num_qubits, 'feature map not present in cache')
            kernel = _create_transient_fidelity_kernel(
                circuit_key=circuit_key,
                num_qubits=num_qubits,
                gpu_id=gpu_id,
                feature_map=feature_map,
                max_circuits_per_eval=max_circuits
            )
            return kernel

        # Normal case: create kernel and store in process-local cache for reuse
        if feature_map is None:
            feature_map = _handle_missing_precompiled_circuit(circuit_key, num_qubits, 'feature map not present in cache')
        kernel = create_quantum_kernel(
            feature_map=feature_map,
            cuda_device=gpu_id,
            max_circuits_per_eval=max_circuits
        )

        try:
            with _PROCESS_LOCAL_FIDELITY_KERNELS_LOCK:
                _PROCESS_LOCAL_FIDELITY_KERNELS[key] = kernel
        except Exception:
            pass

        # Maintain light-weight metadata map for bookkeeping (picklable)
        try:
            with _FIDELITY_KERNEL_LOCK:
                _FIDELITY_KERNEL_META[key] = {'created_at': time.time(), 'gpu_id': gpu_id}
                try:
                    _FIDELITY_KERNEL_META.move_to_end(key)
                except Exception:
                    pass
                # Trim metadata entries if too many
                while len(_FIDELITY_KERNEL_META) > MAX_FIDELITY_KERNEL_CACHE_SIZE:
                    _FIDELITY_KERNEL_META.popitem(last=False)
        except Exception:
            pass

        # Best-effort trim if process memory is high
        try:
            _trim_caches_if_needed()
        except Exception:
            pass

        return kernel
    except Exception:
        # Last-resort: create transient kernel
        if feature_map is None:
            feature_map = _handle_missing_precompiled_circuit(circuit_key, num_qubits, 'feature map not present in cache (transient path)')
        try:
            return _create_transient_fidelity_kernel(
                circuit_key=circuit_key,
                num_qubits=num_qubits,
                gpu_id=gpu_id,
                feature_map=feature_map,
                max_circuits_per_eval=max_circuits
            )
        except Exception:
            raise

# ============================================================================
# RESULTS MANAGER
# ============================================================================

class ThreadSafeResultsManager:
    """Thread-safe results manager"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.results_file = RESULTS_DIR / f'gpu_quantum_results_{session_id}.csv'
        self.checkpoint_file = CHECKPOINT_DIR / f'gpu_checkpoint_{session_id}.json'
        self.completed_models = set()
        self.results = []
        self.lock = threading.Lock()
        self.gpu_timings = {}
        
        self._load_existing_results()
    
    def _load_existing_results(self):
        """Load existing results if resuming"""
        with self.lock:
            if self.checkpoint_file.exists():
                try:
                    with open(self.checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)
                        self.completed_models = set(checkpoint_data.get('completed_models', []))
                        self.gpu_timings = checkpoint_data.get('gpu_timings', {})
                        logger.info(f"Loaded checkpoint: {len(self.completed_models)} models completed")
                except:
                    pass
            
            if self.results_file.exists():
                try:
                    existing_df = pd.read_csv(self.results_file)
                    self.results = existing_df.to_dict('records')
                    for result in self.results:
                        if result.get('status') == 'success':
                            model_key = self._get_model_key(result['model'])
                            self.completed_models.add(model_key)
                except:
                    pass
    
    def _get_model_key(self, model_name: str) -> str:
        """Get unique key for model"""
        return model_name.lower().strip()
    
    def is_model_completed(self, model_name: str) -> bool:
        """Check if model is completed"""
        with self.lock:
            return self._get_model_key(model_name) in self.completed_models
    
    def save_result(self, result: dict, gpu_id: int = -1):
        """Save result"""
        with self.lock:
            result['gpu_id'] = gpu_id
            result['timestamp'] = datetime.utcnow().isoformat()
            
            self.results.append(result)
            
            if result.get('status') == 'success':
                model_key = self._get_model_key(result['model'])
                self.completed_models.add(model_key)
                
                if gpu_id >= 0:
                    if str(gpu_id) not in self.gpu_timings:
                        self.gpu_timings[str(gpu_id)] = []
                    self.gpu_timings[str(gpu_id)].append(result.get('train_time', 0))

            # Append to CSV atomically using file lock to avoid concurrent overwrite from multiple processes
            try:
                import fcntl
                row_df = pd.DataFrame([result])
                # Ensure directory exists
                self.results_file.parent.mkdir(parents=True, exist_ok=True)

                # Open file for append and lock while writing header/row
                write_header = not self.results_file.exists()
                with open(self.results_file, 'a', newline='') as fh:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                        row_df.to_csv(fh, header=write_header, index=False)
                    finally:
                        try:
                            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                        except Exception:
                            pass
            except Exception:
                # Fallback: overwrite entire file (less safe) if locking not available
                try:
                    df = pd.DataFrame(self.results)
                    df.to_csv(self.results_file, index=False)
                except Exception:
                    pass

            # Update checkpoint (still done under thread lock)
            self._save_checkpoint()

            gpu_str = f"GPU:{gpu_id}" if gpu_id >= 0 else "CPU"
            logger.info(f"💾 [{gpu_str}] Saved result for {result['model']}")


    
    
    def _save_checkpoint(self):
        """Save checkpoint"""
        checkpoint_data = {
            'session_id': self.session_id,
            'completed_models': list(self.completed_models),
            'gpu_timings': self.gpu_timings,
            'timestamp': datetime.utcnow().isoformat(),
            'total_models_completed': len(self.completed_models)
        }
        
        try:
            # Atomic write: write to temp file then replace
            tmp_path = str(self.checkpoint_file) + '.tmp'
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            os.replace(tmp_path, str(self.checkpoint_file))
        except Exception:
            # Best-effort fallback
            try:
                with open(self.checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
            except Exception:
                pass
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame"""
        with self.lock:
            if self.results:
                return pd.DataFrame(self.results)
            return pd.DataFrame()


# Helper: enqueue results to central writer or fallback to direct save
def _enqueue_or_save_result(result: dict, gpu_id: int, session_id: str):
    """Put result into the central result queue when available, else fallback to direct save.

    This helper is used by worker processes to avoid concurrent file writes.
    """
    try:
        if _RESULT_QUEUE is not None:
            try:
                _RESULT_QUEUE.put({'result': result, 'gpu_id': gpu_id})
                return
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: write directly using ThreadSafeResultsManager
    try:
        ThreadSafeResultsManager(session_id).save_result(result, gpu_id)
    except Exception:
        try:
            logger.exception("Failed to persist result via fallback")
        except Exception:
            pass

# ============================================================================
# CIRCUIT CREATION
# ============================================================================

def create_unique_feature_map(num_qubits: int, feature_map_type: str, reps: int = 2):
    """Create feature map with unique parameters - PRESERVES COMPLEXITY"""
    try:
        _lazy_qiskit_imports()
    except Exception:
        pass

    unique_id = str(uuid.uuid4())[:8]
    
    if feature_map_type == 'ZZ':
        feature_map = ZZFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            parameter_prefix=f'x_{unique_id}',
            entanglement='full'  # Preserve full entanglement
        )
    elif feature_map_type == 'Pauli':
        feature_map = PauliFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            parameter_prefix=f'x_{unique_id}',
            paulis=['Z', 'ZZ']  # Full Pauli terms
        )
    elif feature_map_type == 'Z':
        feature_map = ZFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            parameter_prefix=f'x_{unique_id}'
        )
    else:
        # Default circuit
        feature_map = QuantumCircuit(num_qubits)
        params = ParameterVector(f'theta_{unique_id}', num_qubits)
        for _ in range(reps):
            for i in range(num_qubits):
                feature_map.ry(params[i], i)
            if num_qubits > 1:
                for i in range(num_qubits - 1):
                    feature_map.cx(i, i + 1)
    
    return feature_map

def create_adaptive_ansatz(num_qubits: int, ansatz_type: str = 'real_amplitudes', 
                           param_prefix: str = 'theta'):
    """Create ansatz with preserved complexity"""
    try:
        _lazy_qiskit_imports()
    except Exception:
        pass
    
    if ansatz_type == 'real_amplitudes' or ansatz_type == 'RealAmplitudes':
        ansatz = RealAmplitudes(
            num_qubits, 
            reps=2,
            entanglement='full',
            parameter_prefix=param_prefix
        )
    elif ansatz_type == 'efficient_su2' or ansatz_type == 'EfficientSU2':
        ansatz = EfficientSU2(
            num_qubits,
            reps=2,
            entanglement='full',
            parameter_prefix=param_prefix
        )
    else:  # TwoLocal
        ansatz = TwoLocal(
            num_qubits,
            rotation_blocks=['ry', 'rz'],
            entanglement_blocks='cz',
            entanglement='full',
            reps=2,
            parameter_prefix=param_prefix
        )
    
    return ansatz

# ============================================================================
# QUANTUM ENSEMBLE MODELS
# ============================================================================

class QuantumRandomForest(BaseEstimator, ClassifierMixin):
    """Quantum Random Forest using multiple QSVC estimators"""
    
    def __init__(self, n_estimators=5, num_qubits=4, max_features='sqrt', random_state=42, gpu_id=0):
        self.n_estimators = n_estimators
        self.num_qubits = num_qubits
        self.max_features = max_features
        self.random_state = random_state
        self.gpu_id = gpu_id
        self.estimators_ = []
        self.feature_indices_ = []
    
    def _get_n_features(self, n_total_features):
        """Calculate number of features for each tree"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_total_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        else:
            return n_total_features
    
    def fit(self, X, y):
        """Fit the Quantum Random Forest"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        n_selected_features = min(self._get_n_features(n_features), self.num_qubits)
        
        logger.info(f"[GPU:{self.gpu_id}] Training Quantum Random Forest with {self.n_estimators} trees...")
        
        for i in range(self.n_estimators):
            # Random feature selection
            feature_indices = np.random.choice(n_features, n_selected_features, replace=False)
            self.feature_indices_.append(feature_indices)
            
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices][:, feature_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Create QSVC for this tree
            feature_map = create_unique_feature_map(n_selected_features, 'Z', 1)
            # Worker processes set CUDA_VISIBLE_DEVICES to the physical GPU id
            # and therefore should use local visible device index 0 when creating
            # GPU-bound objects.
            quantum_kernel = create_quantum_kernel(feature_map, 0)
            
            qsvc = QSVC(quantum_kernel=quantum_kernel)
            logger.info(f"[GPU:{self.gpu_id}]   Training tree {i+1}/{self.n_estimators}...")
            qsvc.fit(X_bootstrap, y_bootstrap)
            
            self.estimators_.append(qsvc)
            # Clear local device (index 0) memory pool
            clear_gpu_memory(0)
        
        return self
    
    def predict(self, X):
        """Predict using majority voting"""
        predictions = []
        
        for estimator, feature_indices in zip(self.estimators_, self.feature_indices_):
            X_subset = X[:, feature_indices]
            pred = estimator.predict(X_subset)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions).T
        final_predictions = []
        for row in predictions:
            counts = Counter(row)
            final_predictions.append(counts.most_common(1)[0][0])
        
        return np.array(final_predictions)

class QuantumVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Quantum Ensemble using voting"""
    
    def __init__(self, estimators, voting='hard', weights=None, gpu_id=0):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.gpu_id = gpu_id
        self.fitted_estimators_ = []
    
    def fit(self, X, y):
        """Fit all quantum estimators"""
        logger.info(f"[GPU:{self.gpu_id}] Training Quantum Voting Ensemble ({self.voting} voting)...")
        
        for name, estimator in self.estimators:
            logger.info(f"[GPU:{self.gpu_id}]   Training {name}...")
            fitted_estimator = copy.deepcopy(estimator)
            fitted_estimator.fit(X, y)
            self.fitted_estimators_.append((name, fitted_estimator))
            # Use local visible device index 0 for memory operations
            clear_gpu_memory(0)
        
        return self
    
    def predict(self, X):
        """Predict using voting"""
        predictions = []
        for name, estimator in self.fitted_estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        if self.weights is not None:
            weighted_predictions = []
            for pred, weight in zip(predictions, self.weights):
                for _ in range(int(weight * 10)):  # Weight scaling
                    weighted_predictions.append(pred)
            predictions = weighted_predictions
        
        # Majority voting
        predictions = np.array(predictions).T
        final_predictions = []
        for row in predictions:
            counts = Counter(row)
            final_predictions.append(counts.most_common(1)[0][0])
        
        return np.array(final_predictions)

class QuantumWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Quantum Ensemble with adaptive weights based on validation performance"""
    
    def __init__(self, estimators, validation_split=0.2, gpu_id=0):
        self.estimators = estimators
        self.validation_split = validation_split
        self.gpu_id = gpu_id
        self.fitted_estimators_ = []
        self.weights_ = []
    
    def fit(self, X, y):
        """Fit estimators and compute weights based on validation performance"""
        logger.info(f"[GPU:{self.gpu_id}] Training Quantum Weighted Ensemble...")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, stratify=y, random_state=42
        )
        
        performances = []
        
        for name, estimator in self.estimators:
            logger.info(f"[GPU:{self.gpu_id}]   Training {name}...")
            fitted_estimator = copy.deepcopy(estimator)
            fitted_estimator.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = fitted_estimator.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            performances.append(accuracy)
            
            self.fitted_estimators_.append((name, fitted_estimator))
            logger.info(f"[GPU:{self.gpu_id}]     Validation accuracy: {accuracy:.4f}")
            # Clear device 0 memory (local visible device)
            clear_gpu_memory(0)
        
        # Calculate weights based on performance
        performances = np.array(performances)
        self.weights_ = performances / performances.sum()
        logger.info(f"[GPU:{self.gpu_id}]   Computed weights: {self.weights_}")
        
        return self
    
    def predict(self, X):
        """Predict using weighted voting"""
        predictions = []
        
        for (name, estimator), weight in zip(self.fitted_estimators_, self.weights_):
            pred = estimator.predict(X)
            for _ in range(int(weight * 100)):
                predictions.append(pred)
        
        # Weighted majority voting
        predictions = np.array(predictions).T
        final_predictions = []
        for row in predictions:
            counts = Counter(row)
            final_predictions.append(counts.most_common(1)[0][0])
        
        return np.array(final_predictions)

# ============================================================================
# DATA PROCESSING
# ============================================================================

class DataProcessor:
    """Data preprocessing for quantum ML"""
    
    def __init__(self, num_qubits: int, random_seed: int = 42, use_gpu: bool = True):
        self.num_qubits = num_qubits
        self.random_seed = random_seed
        self.use_gpu = use_gpu and (_ensure_cupy() is not None)
        self.scaler = None
        self.pca = None
        self.selector = None
        self.label_encoder = None

    def _sanitize_features(self, data: np.ndarray, stage: str, *, nan_value: float = 0.0,
                           posinf_value: Optional[float] = None, neginf_value: Optional[float] = None) -> np.ndarray:
        """Replace non-finite entries to keep downstream GPU kernels stable."""
        if not isinstance(data, np.ndarray):
            return data

        finite_mask = np.isfinite(data)
        if finite_mask.all():
            return data

        total_invalid = data.size - int(finite_mask.sum())
        try:
            logger.warning(
                "Detected %d non-finite feature entries after %s; applying safe replacements",
                total_invalid,
                stage
            )
        except Exception:
            pass

        sanitized = np.nan_to_num(
            data,
            nan=nan_value,
            posinf=posinf_value if posinf_value is not None else nan_value,
            neginf=neginf_value if neginf_value is not None else nan_value
        )
        return sanitized
    
    def prepare_data(self, df: pd.DataFrame, sample_size: int = None):
        """Prepare data from dataframe"""
        if 'Label' in df.columns:
            X = df.drop('Label', axis=1)
            y = df['Label']
        elif 'label' in df.columns:
            X = df.drop('label', axis=1)
            y = df['label']
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        X = X.values.astype(np.float32)
        X = self._sanitize_features(X, 'raw feature extraction')
        y = y.values
        
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        if sample_size and sample_size < len(X):
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]
            y = y[indices]
        
        logger.info(f"Original data shape: {X.shape}")
        return X, y
    
    def fit_transform(self, X, y):
        """Transform data to quantum-ready format"""
        original_features = X.shape[1]
        
        # Feature selection if needed
        if X.shape[1] > self.num_qubits * 2:
            self.selector = SelectKBest(mutual_info_classif, k=min(self.num_qubits * 2, X.shape[1]))
            X = self.selector.fit_transform(X, y)
            logger.info(f"Feature selection: {original_features} → {X.shape[1]} features")
            X = self._sanitize_features(X, 'feature selection output')
        
        # Scale features
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        X = self.scaler.fit_transform(X)
        X = self._sanitize_features(X, 'min-max scaling', nan_value=0.0, posinf_value=np.pi, neginf_value=0.0)
        
        # Ensure exactly num_qubits features
        if X.shape[1] != self.num_qubits:
            if X.shape[1] > self.num_qubits:
                # Use PCA to reduce
                n_components = min(self.num_qubits, X.shape[0] - 1, X.shape[1])
                self.pca = PCA(n_components=n_components, random_state=self.random_seed)
                X = self.pca.fit_transform(X)
                
                if hasattr(self.pca, 'explained_variance_ratio_'):
                    logger.info(f"PCA variance retained: {sum(self.pca.explained_variance_ratio_):.2%}")
                X = self._sanitize_features(X, 'pca reduction', nan_value=0.0)
            
            # Pad if needed
            if X.shape[1] < self.num_qubits:
                padding_size = self.num_qubits - X.shape[1]
                padding = np.zeros((X.shape[0], padding_size))
                X = np.hstack([X, padding])
                logger.info(f"Padded {padding_size} zero features to reach {self.num_qubits} features")
        
        logger.info(f"Final features: {X.shape[1]} (normalized to [0, π])")

        X = self._sanitize_features(X, 'final feature check', nan_value=0.0, posinf_value=np.pi, neginf_value=0.0)
        return X.astype(np.float32)

# ============================================================================
# CENTRALIZED TRAINING FUNCTION
# ============================================================================

def train_quantum_model_centralized(model_name: str, model_type: str, config: dict,
                                    X_train, y_train, X_test, y_test,
                                    session_id: str, gpu_id: int,
                                    num_qubits: int, num_samples: int,
                                    available_gpus: Optional[List[int]] = None):
    """Train quantum model using centralized GPU management
    
    PRESERVES MODEL DIFFERENTIATION:
    - Each QSVC variant uses its specific kernel computation method
    - Circuit complexity is maintained for fair comparison
    """
    
    # Normalize GPU assignment
    assigned_gpus: List[int] = []
    if available_gpus:
        for gpu in available_gpus if isinstance(available_gpus, (list, tuple, set)) else [available_gpus]:
            try:
                gpu_index = int(gpu)
            except (TypeError, ValueError):
                continue
            if gpu_index not in assigned_gpus:
                assigned_gpus.append(gpu_index)
    if not assigned_gpus:
        assigned_gpus = [int(gpu_id)]

    primary_gpu = assigned_gpus[0]
    gpu_id = primary_gpu
    gpu_label = ','.join(str(g) for g in assigned_gpus)

    # CRITICAL FIX: Set CUDA visibility to PRIMARY GPU only for this worker
    # The worker will use local device 0, which maps to the primary physical GPU
    # Other GPUs in assigned_gpus are used for parallel kernel chunk distribution
    os.environ['CUDA_VISIBLE_DEVICES'] = str(primary_gpu)
    os.environ['NUMBA_CUDA_DEVICE'] = '0'
    local_cuda_device = 0
    try:
        logger.info(f"[GPU:{gpu_label}] Worker using primary GPU:{primary_gpu} (local_device=0), available_gpus={assigned_gpus}")
    except Exception:
        pass

    # Store full GPU list for kernel computer to use for parallel chunking
    available_gpus = list(assigned_gpus)

    # After setting per-process CUDA visibility, import Qiskit/Aer symbols
    # so that Aer picks up the correct local visible GPU (index 0). This
    # must happen before any CentralizedGPUCircuitManager hydration or
    # kernel/sampler/estimator creation.
    try:
        _lazy_qiskit_imports()
    except Exception:
        # Let downstream code raise a clear error if imports fail.
        pass
    
    # Initialize circuit manager parameters in-process only if not already
    # initialized by the ProcessPoolExecutor initializer.
    if _CIRCUIT_MANAGER_PARAMS is None:
        try:
            _init_circuit_manager_worker(num_qubits, num_samples, None)
        except Exception:
            _init_circuit_manager_worker(num_qubits, num_samples)

    # Get circuit manager (hydrate from central qpy registry if provided).
    # NOTE: after setting CUDA_VISIBLE_DEVICES above we must pass the local
    # visible device index (0) to GPU libraries. Keep `gpu_id` for logging only.
    circuit_manager = _get_circuit_manager(local_cuda_device)

    # Reuse a shared kernel computer so model variants can share cached kernels
    kernel_computer = _get_or_create_shared_kernel_computer(
        circuit_manager,
        gpu_id=local_cuda_device,
        batch_size=1024
    )
    
    # Create results manager
    results_manager = ThreadSafeResultsManager(session_id)
    
    if results_manager.is_model_completed(model_name):
        logger.info(f"[GPU:{gpu_id}] ⏩ Skipping {model_name} (already completed)")
        return
    
    num_features = X_train.shape[1]
    result = {
        'model': model_name,
        'type': 'quantum',
        'num_features': num_features,
        'status': 'failed',
        'gpu_id': gpu_id
    }
    
    try:
        logger.info(f"[GPU:{gpu_id}] 🔥 Training {model_name} with {num_features} features")
        start_time = time.time()
        
        # Get circuit configuration
        feature_map_type = config.get('feature_map', 'ZZ')
        reps = config.get('reps', 2)
        circuit_key = f"{feature_map_type}_reps{reps}"
        
        if model_type in ['QSVC_Standard', 'QSVC_Callable', 'QSVC_Precomputed']:
            logger.info(f"[GPU:{gpu_id}] Using circuit: {circuit_key}")
            logger.info(f"[GPU:{gpu_id}] Model variant: {model_type}")

            # Merge repo-level NYSTROM_CONFIG defaults with per-model config
            repo_cfg = NYSTROM_CONFIG.get(model_type, {})
            effective_cfg = {**repo_cfg, **(config or {})}

            # Explicitly enforce per-variant computation method to avoid ambiguity
            if model_type == 'QSVC_Standard':
                # QSVC_Standard should use Nyström when configured to do so.
                if effective_cfg.get('use_nystrom', False) or effective_cfg.get('force_nystrom', False):
                    logger.info(f"[GPU:{gpu_id}] QSVC_Standard: Using Nyström approximation (landmarks={effective_cfg.get('landmarks')})")
                    landmarks = effective_cfg.get('landmarks')
                    if landmarks is None:
                        landmarks = min(len(X_train), 2000)
                    # compute_kernel_nystrom returns (K_train, K_test) when X_test is provided
                    nystrom_result = kernel_computer.compute_kernel_nystrom(
                        X_train,
                        X_test if X_test is not None and len(X_test) > 0 else None,
                        circuit_key,
                        n_landmarks=landmarks,
                        available_gpus=available_gpus
                    )

                    if isinstance(nystrom_result, tuple):
                        K_train, K_test = nystrom_result
                    else:
                        K_train = nystrom_result
                        if X_test is not None and len(X_test) > 0:
                            test_result = kernel_computer.compute_kernel_nystrom(
                                X_train,
                                X_test,
                                circuit_key,
                                n_landmarks=landmarks,
                                available_gpus=available_gpus
                            )
                            if isinstance(test_result, tuple):
                                _, K_test = test_result
                            else:
                                K_test = test_result
                        else:
                            K_test = K_train

                else:
                    logger.info(f"[GPU:{gpu_id}] QSVC_Standard: Computing FULL kernel (Nyström not enabled for this run)")
                    K_train = kernel_computer.compute_kernel_for_model(X_train, X_train, circuit_key, 'QSVC_Precomputed', available_gpus)
                    K_test = kernel_computer.compute_kernel_for_model(X_test, X_train, circuit_key, 'QSVC_Precomputed', available_gpus)

                # Train SVC on precomputed kernels
                svc = SVC(kernel='precomputed')
                svc.fit(K_train, y_train)
                y_pred = svc.predict(K_test)

            elif model_type == 'QSVC_Precomputed':
                # Explicitly compute full kernel matrices (precomputed approach)
                logger.info(f"[GPU:{gpu_id}] QSVC_Precomputed: Computing FULL kernel matrices")
                K_train = kernel_computer.compute_kernel_for_model(X_train, X_train, circuit_key, 'QSVC_Precomputed', available_gpus)
                K_test = kernel_computer.compute_kernel_for_model(X_test, X_train, circuit_key, 'QSVC_Precomputed', available_gpus)

                svc = SVC(kernel='precomputed')
                svc.fit(K_train, y_train)
                y_pred = svc.predict(K_test)

            elif model_type == 'QSVC_Callable':
                # Callable variant: kernel function provided to SVC, computes kernels on-demand
                logger.info(f"[GPU:{gpu_id}] QSVC_Callable: On-demand kernel computation (callable)")

                def quantum_kernel_callable(X1, X2):
                    """Callable kernel function evaluating full kernel on demand."""
                    # Force callable to compute full kernel (no Nyström for callable by default)
                    return kernel_computer.compute_kernel_for_model(X1, X2, circuit_key, 'QSVC_Callable', available_gpus)

                svc = SVC(kernel=quantum_kernel_callable)
                svc.fit(X_train, y_train)
                y_pred = svc.predict(X_test)
        
        elif model_type == 'PegasosQSVC' and PEGASOS_AVAILABLE:
            logger.info(f"[GPU:{gpu_id}] PegasosQSVC: Stochastic gradient approach")
            
            # Custom GPU kernel for Pegasos
            class GPUQuantumKernel:
                def __init__(self, kernel_computer, circuit_key, available_gpus):
                    self.kernel_computer = kernel_computer
                    self.circuit_key = circuit_key
                    self.available_gpus = available_gpus
                
                def evaluate(self, X1, X2=None):
                    if X2 is None:
                        X2 = X1
                    return self.kernel_computer.compute_kernel_for_model(
                        X1, X2, self.circuit_key, 'PegasosQSVC', self.available_gpus
                    )
            
            gpu_kernel = GPUQuantumKernel(kernel_computer, circuit_key, available_gpus)
            
            pegasos_qsvc = PegasosQSVC(quantum_kernel=gpu_kernel, num_steps=100, C=1.0)
            pegasos_qsvc.fit(X_train, y_train)
            y_pred = pegasos_qsvc.predict(X_test)
        
        elif model_type == 'VQC':
            optimizer_name = config.get('optimizer', 'COBYLA')
            logger.info(f"[GPU:{gpu_id}] 🎛️ VQC with {optimizer_name} optimizer")
            logger.info(f"[GPU:{gpu_id}] Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
            
            # Get pre-compiled circuits
            feature_map = circuit_manager.feature_maps.get(f"Z_reps1")
            ansatz = circuit_manager.ansatzes.get(f"RealAmplitudes_reps2")
            
            if feature_map is None or ansatz is None:
                logger.error(f"[GPU:{gpu_id}] ❌ Missing pre-compiled circuits!")
                raise RuntimeError("Pre-compiled circuits not found in cache")
            
            logger.info(f"[GPU:{gpu_id}] ✅ Using pre-transpiled feature_map: {feature_map.num_qubits}q, {feature_map.num_parameters} params")
            logger.info(f"[GPU:{gpu_id}] ✅ Using pre-transpiled ansatz: {ansatz.num_qubits}q, {ansatz.num_parameters} params")
            
            # Adaptive iteration count based on dataset size
            base_maxiter = 100 if num_samples <= 1000 else (50 if num_samples <= 5000 else 30)
            
            # Select optimizer with fallback for large runs
            if optimizer_name == 'ADAM' and (num_qubits >= 8 or num_samples >= 5000):
                logger.warning(
                    f"[GPU:{gpu_id}] VQC_ADAM requires large param-shift sampler jobs for {num_qubits} qubits/{num_samples} samples; "
                    "falling back to SPSA to avoid cudaErrorInvalidResourceHandle."
                )
                optimizer_name = 'SPSA'
            
            # Progress callback for VQC
            iteration_state = {'count': 0, 'start_time': time.time()}
            def vqc_progress_callback(weights, loss):
                iteration_state['count'] += 1
                elapsed = time.time() - iteration_state['start_time']
                logger.info(
                    f"[GPU:{gpu_id}] 📈 VQC_{optimizer_name} iter {iteration_state['count']}: "
                    f"loss={loss:.6f}, elapsed={elapsed:.1f}s"
                )
            
            if optimizer_name == 'COBYLA':
                optimizer = COBYLA(maxiter=base_maxiter)
            elif optimizer_name == 'SPSA':
                optimizer = SPSA(maxiter=base_maxiter)
            else:
                optimizer = ADAM(maxiter=base_maxiter)
            
            logger.info(f"[GPU:{gpu_id}] Creating GPU Sampler primitive (shots=2048)...")
            sampler = create_gpu_sampler(local_cuda_device, shots=2048)
            logger.info(f"[GPU:{gpu_id}] ✅ GPU Sampler created successfully")
            
            logger.info(f"[GPU:{gpu_id}] 🚀 Starting VQC training with {optimizer_name} (maxiter={base_maxiter})...")
            
            # Train VQC with callback
            vqc = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer,
                sampler=sampler,
                callback=vqc_progress_callback
            )
            
            logger.info(f"[GPU:{gpu_id}] Calling vqc.fit() on {len(X_train)} samples...")
            vqc.fit(X_train, y_train)
            logger.info(f"[GPU:{gpu_id}] ✅ VQC Training complete! Running predictions on {len(X_test)} samples...")
            y_pred = vqc.predict(X_test)
            logger.info(f"[GPU:{gpu_id}] ✅ VQC Predictions complete!")
        
        elif model_type.startswith('QNN'):
            logger.info(f"[GPU:{gpu_id}] 🧠 QNN model: {model_type}")
            logger.info(f"[GPU:{gpu_id}] Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
            
            # Get pre-transpiled circuits from cache
            feature_map = circuit_manager.feature_maps.get(f"Z_reps1")
            ansatz = circuit_manager.ansatzes.get(f"RealAmplitudes_reps2")
            
            if feature_map is None or ansatz is None:
                logger.error(f"[GPU:{gpu_id}] ❌ Missing pre-compiled circuits! feature_map={feature_map is not None}, ansatz={ansatz is not None}")
                raise RuntimeError("Pre-compiled circuits not found in cache")
            
            logger.info(f"[GPU:{gpu_id}] ✅ Using pre-transpiled feature_map with {feature_map.num_parameters} params")
            logger.info(f"[GPU:{gpu_id}] ✅ Using pre-transpiled ansatz with {ansatz.num_parameters} params")
            
            # Combine circuits
            qc = QuantumCircuit(num_qubits)
            qc.compose(feature_map, inplace=True)
            qc.compose(ansatz, inplace=True)
            logger.info(f"[GPU:{gpu_id}] Combined circuit: {qc.num_qubits} qubits, depth={qc.depth()}, gates={len(qc.data)}")
            
            # Adaptive iteration count based on dataset size to prevent multi-day runs
            base_maxiter = 50 if num_samples <= 1000 else (30 if num_samples <= 5000 else 20)
            
            # Progress callback for logging
            iteration_state = {'count': 0, 'last_log': time.time(), 'start_time': time.time()}
            def qnn_progress_callback(weights, loss):
                iteration_state['count'] += 1
                now = time.time()
                elapsed = now - iteration_state['start_time']
                # Log every iteration for visibility
                logger.info(
                    f"[GPU:{gpu_id}] 📈 {model_name} iter {iteration_state['count']}: "
                    f"loss={loss:.6f}, elapsed={elapsed:.1f}s"
                )
                iteration_state['last_log'] = now
            
            if model_type == 'QNN_Estimator':
                from qiskit_machine_learning.neural_networks import EstimatorQNN
                from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
                from qiskit.quantum_info import SparsePauliOp
                
                logger.info(f"[GPU:{gpu_id}] Creating GPU Estimator primitive...")
                estimator = create_gpu_estimator(local_cuda_device)
                logger.info(f"[GPU:{gpu_id}] ✅ GPU Estimator created successfully")
                
                # Create observables for multi-class output
                observables: List[SparsePauliOp] = []
                max_outputs = 2 if num_qubits >= 2 else 1
                for idx in range(num_qubits):
                    label = ['I'] * num_qubits
                    label[idx] = 'Z'
                    observables.append(SparsePauliOp.from_list([(''.join(label), 1.0)]))
                    if len(observables) >= max_outputs:
                        break
                if not observables:
                    observables = [SparsePauliOp.from_list([('Z', 1.0)])]
                
                logger.info(f"[GPU:{gpu_id}] Using {len(observables)} observable(s) for output")
                
                qnn = EstimatorQNN(
                    circuit=qc,
                    input_params=feature_map.parameters,
                    weight_params=ansatz.parameters,
                    observables=observables if len(observables) > 1 else observables[0],
                    estimator=estimator
                )
                logger.info(f"[GPU:{gpu_id}] ✅ EstimatorQNN created: input_params={qnn.num_inputs}, weight_params={qnn.num_weights}")
                
                # Use SPSA with adaptive iterations and progress callback
                maxiter = base_maxiter
                logger.info(f"[GPU:{gpu_id}] 🚀 Starting EstimatorQNN training with SPSA (maxiter={maxiter})...")
                
                classifier = NeuralNetworkClassifier(
                    qnn,
                    optimizer=SPSA(maxiter=maxiter, learning_rate=0.1, perturbation=0.05),
                    one_hot=bool(len(observables) > 1),
                    callback=qnn_progress_callback
                )
                
                logger.info(f"[GPU:{gpu_id}] Calling classifier.fit() on {len(X_train)} samples...")
                classifier.fit(X_train, y_train)
                logger.info(f"[GPU:{gpu_id}] ✅ Training complete! Running predictions on {len(X_test)} samples...")
                y_pred = classifier.predict(X_test)
                logger.info(f"[GPU:{gpu_id}] ✅ Predictions complete!")
            
            elif model_type == 'QNN_Sampler':
                from qiskit_machine_learning.neural_networks import SamplerQNN
                from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
                
                logger.info(f"[GPU:{gpu_id}] Creating GPU Sampler primitive (shots=2048)...")
                sampler = create_gpu_sampler(local_cuda_device, shots=2048)
                logger.info(f"[GPU:{gpu_id}] ✅ GPU Sampler created successfully")
                
                def parity(x):
                    return f"{x:b}".count('1') % 2
                
                qnn = SamplerQNN(
                    circuit=qc,
                    input_params=feature_map.parameters,
                    weight_params=ansatz.parameters,
                    interpret=parity,
                    output_shape=2,
                    sampler=sampler
                )
                logger.info(f"[GPU:{gpu_id}] ✅ SamplerQNN created: input_params={qnn.num_inputs}, weight_params={qnn.num_weights}")
                
                # Use SPSA with adaptive iterations and progress callback
                maxiter = base_maxiter
                logger.info(f"[GPU:{gpu_id}] 🚀 Starting SamplerQNN training with SPSA (maxiter={maxiter})...")
                
                classifier = NeuralNetworkClassifier(
                    qnn,
                    optimizer=SPSA(maxiter=maxiter),
                    callback=qnn_progress_callback
                )
                
                logger.info(f"[GPU:{gpu_id}] Calling classifier.fit() on {len(X_train)} samples...")
                classifier.fit(X_train, y_train)
                logger.info(f"[GPU:{gpu_id}] ✅ Training complete! Running predictions on {len(X_test)} samples...")
                y_pred = classifier.predict(X_test)
                logger.info(f"[GPU:{gpu_id}] ✅ Predictions complete!")
            else:
                raise ValueError(f"Unknown QNN type: {model_type}")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Calculate metrics
        train_time = time.time() - start_time
        metrics = calculate_all_metrics(y_test, y_pred, train_time=train_time)
        
        result.update({
            'status': 'success',
            **metrics
        })
        
        logger.info(f"[GPU:{gpu_id}] ✅ {model_name}: Acc={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}, Time={train_time:.2f}s")

        # Save results (enqueue to central writer when available)
        _enqueue_or_save_result(result, gpu_id, session_id)

        return result

    except Exception as e:
        logger.error(f"[GPU:{gpu_id}] ❌ {model_name} failed: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)[:200]
        _enqueue_or_save_result(result, gpu_id, session_id)
        return result

# ============================================================================
# INDIVIDUAL MODEL TRAINERS (For ensemble and special models)
# ============================================================================

def train_quantum_random_forest_gpu(X_train, y_train, X_test, y_test, num_qubits,
                                    session_id: str, gpu_id):
    """Train QuantumRandomForest model on specific GPU"""
    # CRITICAL FIX: After setting CUDA_VISIBLE_DEVICES, use local device 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['NUMBA_CUDA_DEVICE'] = '0'  # Always 0 after visibility restriction
    
    clear_gpu_memory(0)
    
    results_manager = ThreadSafeResultsManager(session_id)
    
    model_name = 'QuantumRandomForest'
    if not results_manager.is_model_completed(model_name):
        logger.info(f"[GPU:{gpu_id}] Training {model_name}...")
        try:
            start_time = time.time()
            
            qrf = QuantumRandomForest(
                n_estimators=3,
                num_qubits=min(num_qubits, 4),
                max_features='sqrt',
                random_state=42,
                gpu_id=gpu_id
            )
            
            qrf.fit(X_train, y_train)
            y_pred = qrf.predict(X_test)
            
            train_time = time.time() - start_time
            
            metrics = calculate_all_metrics(y_test, y_pred, train_time=train_time)
            
            logger.info(f"[GPU:{gpu_id}] ✅ {model_name}: Acc={metrics['accuracy']:.4f}, "
                       f"F1={metrics['f1_score']:.4f}, Time={train_time:.1f}s")
            
            result = {
                'model': model_name,
                'type': 'quantum',
                'num_features': num_qubits,
                'status': 'success',
                **metrics
            }
            _enqueue_or_save_result(result, gpu_id, session_id)
            
        except Exception as e:
            logger.error(f"[GPU:{gpu_id}] ❌ {model_name} failed: {e}")
            result = {'model': model_name, 'type': 'quantum', 'status': 'failed', 'error': str(e)[:200]}
            _enqueue_or_save_result(result, gpu_id, session_id)
        finally:
            clear_gpu_memory(0)

def train_quantum_voting_ensemble_gpu(X_train, y_train, X_test, y_test, num_qubits,
                                      session_id: str, gpu_id):
    """Train QuantumVotingEnsemble model on specific GPU"""
    # CRITICAL FIX: After setting CUDA_VISIBLE_DEVICES, use local device 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['NUMBA_CUDA_DEVICE'] = '0'  # Always 0 after visibility restriction
    results_manager = ThreadSafeResultsManager(session_id)
    
    model_name = 'QuantumVotingEnsemble_Hard'
    if not results_manager.is_model_completed(model_name):
        logger.info(f"[GPU:{gpu_id}] Training {model_name}...")
        try:
            start_time = time.time()
            
            # Create diverse estimators
            estimators = []
            
            for feature_map_type in ['Z', 'ZZ']:
                feature_map = create_unique_feature_map(num_qubits, feature_map_type, 1)
                kernel = create_quantum_kernel(feature_map, 0)
                qsvc = QSVC(quantum_kernel=kernel)
                estimators.append((f'QSVC_{feature_map_type}', qsvc))
            
            qve_hard = QuantumVotingEnsemble(estimators=estimators, voting='hard', gpu_id=gpu_id)
            qve_hard.fit(X_train, y_train)
            y_pred = qve_hard.predict(X_test)
            
            train_time = time.time() - start_time
            
            metrics = calculate_all_metrics(y_test, y_pred, train_time=train_time)
            
            logger.info(f"[GPU:{gpu_id}] ✅ {model_name}: Acc={metrics['accuracy']:.4f}, "
                       f"F1={metrics['f1_score']:.4f}, Time={train_time:.1f}s")
            
            result = {
                'model': model_name,
                'type': 'quantum',
                'num_features': num_qubits,
                'status': 'success',
                **metrics
            }
            _enqueue_or_save_result(result, gpu_id, session_id)
            
        except Exception as e:
            logger.error(f"[GPU:{gpu_id}] ❌ {model_name} failed: {e}")
            result = {'model': model_name, 'type': 'quantum', 'status': 'failed', 'error': str(e)[:200]}
            _enqueue_or_save_result(result, gpu_id, session_id)
        finally:
            clear_gpu_memory(0)

def train_quantum_weighted_ensemble_gpu(X_train, y_train, X_test, y_test, num_qubits,
                                        session_id: str, gpu_id):
    """Train QuantumWeightedEnsemble model on specific GPU"""
    # CRITICAL FIX: After setting CUDA_VISIBLE_DEVICES, use local device 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['NUMBA_CUDA_DEVICE'] = '0'  # Always 0 after visibility restriction
    results_manager = ThreadSafeResultsManager(session_id)
    
    model_name = 'QuantumWeightedEnsemble'
    if not results_manager.is_model_completed(model_name):
        logger.info(f"[GPU:{gpu_id}] Training {model_name}...")
        try:
            start_time = time.time()
            
            estimators = []
            
            feature_map_z = create_unique_feature_map(num_qubits, 'Z', 1)
            kernel_z = create_quantum_kernel(feature_map_z, 0)
            qsvc_z = QSVC(quantum_kernel=kernel_z)
            estimators.append(('QSVC_Z', qsvc_z))
            
            feature_map_pauli = create_unique_feature_map(num_qubits, 'Pauli', 1)
            kernel_pauli = create_quantum_kernel(feature_map_pauli, 0)
            qsvc_pauli = QSVC(quantum_kernel=kernel_pauli)
            estimators.append(('QSVC_Pauli', qsvc_pauli))
            
            qwe = QuantumWeightedEnsemble(estimators=estimators, validation_split=0.2, gpu_id=gpu_id)
            qwe.fit(X_train, y_train)
            y_pred = qwe.predict(X_test)
            
            train_time = time.time() - start_time
            
            metrics = calculate_all_metrics(y_test, y_pred, train_time=train_time)
            
            logger.info(f"[GPU:{gpu_id}] ✅ {model_name}: Acc={metrics['accuracy']:.4f}, "
                       f"F1={metrics['f1_score']:.4f}, Time={train_time:.1f}s")
            
            result = {
                'model': model_name,
                'type': 'quantum',
                'num_features': num_qubits,
                'status': 'success',
                **metrics
            }
            _enqueue_or_save_result(result, gpu_id, session_id)
            
        except Exception as e:
            logger.error(f"[GPU:{gpu_id}] ❌ {model_name} failed: {e}")
            result = {'model': model_name, 'type': 'quantum', 'status': 'failed', 'error': str(e)[:200]}
            _enqueue_or_save_result(result, gpu_id, session_id)
        finally:
            clear_gpu_memory(0)

# ============================================================================
# CLASSICAL MODEL TRAINERS
# ============================================================================

def train_classical_models(X_train, y_train, X_test, y_test, num_features, session_id: str):
    """Train classical ML models for comparison"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    
    results_manager = ThreadSafeResultsManager(session_id)
    
    # Define classical models
    classical_models = {
        'Classical_SVM_Linear': SVC(kernel='linear', random_state=42),
        'Classical_SVM_RBF': SVC(kernel='rbf', random_state=42),
        'Classical_SVM_Poly': SVC(kernel='poly', degree=3, random_state=42),
        'Classical_RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Classical_GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Classical_LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Classical_KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Classical_DecisionTree': DecisionTreeClassifier(random_state=42),
        'Classical_NaiveBayes': GaussianNB(),
    }
    
    logger.info("Training classical models for comparison...")
    
    for model_name, model in classical_models.items():
        if results_manager.is_model_completed(model_name):
            logger.info(f"Skipping {model_name} (already completed)")
            continue
            
        logger.info(f"Training {model_name}...")
        try:
            start_time = time.time()
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass
            
            train_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba, train_time)
            
            logger.info(f"✅ {model_name}: Acc={metrics['accuracy']:.4f}, "
                       f"F1={metrics['f1_score']:.4f}, Time={train_time:.2f}s")
            
            result = {
                'model': model_name,
                'type': 'classical',
                'num_features': num_features,
                'status': 'success',
                **metrics
            }
            _enqueue_or_save_result(result, -1, session_id)
            
        except Exception as e:
            logger.error(f"❌ {model_name} failed: {e}")
            result = {
                'model': model_name,
                'type': 'classical',
                'status': 'failed',
                'error': str(e)[:200]
            }
            _enqueue_or_save_result(result, -1, session_id)

# ============================================================================
# MAIN CENTRALIZED PIPELINE
# ============================================================================

def main_centralized():
    """Main pipeline using centralized GPU management with PRESERVED complexity"""
    global APPROXIMATION_MODE, CIRCUIT_CUT_PARTITIONS, DISTRIBUTED_STRATEGY, DISTRIBUTED_ADDRESS, MIN_DISTRIBUTED_GPU_NODES
    
    mp.set_start_method('spawn', force=True)
    logger.info("[CENTRALIZED] Multi-GPU quantum ML with preserved circuit complexity")
    
    parser = argparse.ArgumentParser(description='Quantum ML CENTRALIZED GPU Pipeline')
    parser.add_argument('--num_qubits', type=int, required=True, help='Number of qubits/features')
    parser.add_argument('--sample_size', type=int, required=True, help='Sample size')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--model_group', type=str, default='qsvc', 
                       choices=['all', 'quantum', 'qsvc', 'vqc', 'qnn', 'ensemble', 'classical'])
    parser.add_argument('--max_gpus', type=int, default=None, help='Maximum GPUs to use')
    parser.add_argument('--session_id', type=str, default=None, help='Session ID')
    parser.add_argument('--nystrom-landmarks', type=int, default=None, help='Override Nyström landmark count')
    parser.add_argument('--prefer-pegasos', action='store_true', help='Prefer PegasosQSVC (stochastic) for large datasets')
    parser.add_argument('--transpile-opt-level', type=int, default=2, choices=[0,1,2,3], help='Optimization level for transpile during precompile')
    parser.add_argument('--cache-debug', action='store_true', help='Enable cache eviction debug logs and counters')
    parser.add_argument('--gpus-per-model', type=int, default=None, help='Number of GPUs to dedicate to each model evaluation')
    parser.add_argument('--approx-mode', type=str, choices=['auto', 'gpu', 'gpu-only', 'gpu_strict', 'tensor_network', 'tensor', 'mps', 'matrix_product_state', 'statevector', 'classical'], default=None, help='Select approximation strategy when GPUs fail')
    parser.add_argument('--circuit-cuts', type=int, default=None, help='Number of partitions for circuit cutting fallback')
    parser.add_argument('--distributed-ray', dest='distributed_ray', action='store_true', default=None,
                        help='Enable Ray-based distributed kernel execution')
    parser.add_argument('--no-distributed-ray', dest='distributed_ray', action='store_false', help='Disable Ray-based distributed execution')
    parser.add_argument('--ray-address', type=str, default=None, help='Optional Ray cluster address (auto-init if omitted)')
    parser.add_argument('--min-gpu-nodes', type=int, default=MIN_DISTRIBUTED_GPU_NODES,
                        help='Minimum distinct Ray nodes with GPUs required for distributed runs')
    parser.add_argument('--only-models', type=str, default=None,
                        help='Comma-separated list of model names to run (case-insensitive)')
    
    args = parser.parse_args()
    
    session_id = args.session_id or datetime.now().strftime('%Y%m%d_%H%M%S_CENTRALIZED')

    if args.approx_mode:
        APPROXIMATION_MODE = args.approx_mode.strip().lower()
    if args.circuit_cuts is not None:
        CIRCUIT_CUT_PARTITIONS = max(0, args.circuit_cuts)
    if args.distributed_ray is True:
        DISTRIBUTED_STRATEGY = 'ray'
    elif args.distributed_ray is False:
        DISTRIBUTED_STRATEGY = 'off'
    if args.ray_address:
        DISTRIBUTED_ADDRESS = args.ray_address
    MIN_DISTRIBUTED_GPU_NODES = max(1, int(args.min_gpu_nodes))

    if APPROXIMATION_MODE not in ('', 'auto', 'gpu', 'gpu-only', 'gpu_strict'):
        raise ValueError(
            "CPU-based approximation modes are disabled; use one of ['auto','gpu','gpu-only','gpu_strict']."
        )

    if DISTRIBUTED_STRATEGY == 'ray' and not RAY_AVAILABLE:
        logger.warning("Ray requested but unavailable; falling back to local multi-GPU execution")
        DISTRIBUTED_STRATEGY = 'off'
        DISTRIBUTED_ADDRESS = None

    if CIRCUIT_CUT_PARTITIONS and CIRCUIT_CUT_PARTITIONS > args.num_qubits:
        CIRCUIT_CUT_PARTITIONS = args.num_qubits
    if CIRCUIT_CUT_PARTITIONS > 1 and args.num_qubits < CIRCUIT_CUT_PARTITIONS * 2:
        logger.info(
            "Circuit cutting reduced from %s to 1 to keep at least two qubits per partition",
            CIRCUIT_CUT_PARTITIONS
        )
        CIRCUIT_CUT_PARTITIONS = 1

    # Adjust cache sizes heuristically for large-qubit runs before precompile
    try:
        adjust_cache_sizes_for_qubits(args.num_qubits)
    except Exception:
        pass

    # Skip GPU initialization if running classical models only
    if args.model_group == 'classical':
        print("="*80)
        print("🔷 CLASSICAL ML PIPELINE")
        print("="*80)
        print(f"Session ID: {session_id}")
        print(f"Configuration: {args.num_qubits} features, {args.sample_size} samples")
        print(f"Model Group: classical")
        print("="*80)
        
        # Load and prepare data
        try:
            df = pd.read_csv(args.dataset)
            logger.info(f"Dataset loaded: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
        
        processor = DataProcessor(num_qubits=args.num_qubits, use_gpu=False)
        X, y = processor.prepare_data(df, sample_size=args.sample_size)
        X = processor.fit_transform(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        logger.info(f"Data split: Train={X_train.shape}, Test={X_test.shape}")
        
        # Train classical models
        print("\n" + "="*80)
        print("🔷 TRAINING CLASSICAL MODELS")
        print("="*80)
        
        try:
            train_classical_models(X_train, y_train, X_test, y_test, args.num_qubits, session_id)
        except Exception as e:
            logger.error(f"Classical models training failed: {e}")
        
        # Print summary
        print("\n" + "="*80)
        print("📊 TRAINING SUMMARY")
        print("="*80)
        
        results_manager = ThreadSafeResultsManager(session_id)
        results_df = results_manager.get_results_dataframe()
        
        if not results_df.empty:
            success_df = results_df[results_df['status'] == 'success']
            
            if not success_df.empty:
                print(f"\n✅ Successfully trained: {len(success_df)} models")
                print("\n🏆 TOP MODELS BY ACCURACY:")
                print("-"*60)
                
                for _, row in success_df.nlargest(min(10, len(success_df)), 'accuracy').iterrows():
                    print(f"🔷 {row['model']:<30} | Acc: {row['accuracy']:.4f} | "
                          f"F1: {row.get('f1_score', 0):.4f} | Time: {row.get('train_time', 0):.2f}s")
        
        print("="*80)
        logger.info("🎊 Classical model training complete!")
        return

    # Ensure GPU manager is initialized in main process before we query GPUs
    global gpu_manager
    try:
        if gpu_manager is None:
            gpu_manager = GPUManager()
    except Exception as e:
        logger.error(f"Failed to initialize GPU manager: {e}")
        return
    
    print("="*80)
    print("🔥 QUANTUM ML CENTRALIZED GPU PIPELINE 🔥")
    print("="*80)
    print(f"Session ID: {session_id}")
    print(f"Configuration: {args.num_qubits} qubits, {args.sample_size} samples")
    print(f"Model Group: {args.model_group}")
    print(f"GPUs detected: {gpu_manager.gpu_count}")
    print("="*80)
    
    # Display GPU info
    for gpu in gpu_manager.gpu_info:
        print(f"GPU {gpu['id']}: {gpu['name']} - {gpu['memory_total']:.1f} GB")
    
    num_gpus = min(args.max_gpus or gpu_manager.gpu_count, gpu_manager.gpu_count)
    if args.gpus_per_model is not None:
        capped_request = max(1, min(num_gpus, args.gpus_per_model))
        if capped_request != args.gpus_per_model:
            try:
                logger.warning(
                    "Requested gpus-per-model=%s exceeds available GPUs (%s); capping to %s",
                    args.gpus_per_model,
                    num_gpus,
                    capped_request
                )
            except Exception:
                pass
        gpus_per_model = capped_request
    else:
        if num_gpus >= 2 and args.sample_size >= 10000:
            gpus_per_model = min(num_gpus, MAX_PARALLEL_GPUS)
        else:
            gpus_per_model = 1
        try:
            logger.info(
                "Auto-selected gpus_per_model=%s (sample_size=%s, num_gpus=%s)",
                gpus_per_model,
                args.sample_size,
                num_gpus
            )
        except Exception:
            pass

    print(f"\n🚀 Using {num_gpus} GPU(s) with CENTRALIZED management")
    print(f"🧮 GPUs per model run: {gpus_per_model}")
    print("="*80)
    print(f"Approximation mode: {APPROXIMATION_MODE}")
    if CIRCUIT_CUT_PARTITIONS > 1:
        print(f"Circuit cuts enabled: {CIRCUIT_CUT_PARTITIONS} partitions")
    if DISTRIBUTED_STRATEGY == 'ray':
        print("Distributed execution: Ray ({} )".format(DISTRIBUTED_ADDRESS or 'auto-init'))
        print(f"Ray cluster minimum GPU nodes required: {MIN_DISTRIBUTED_GPU_NODES}")
    else:
        print("Distributed execution: Local multi-GPU only")
    # Start background GPU monitor (writes CSV) to correlate with per-worker logs.
    monitor_stop_event = threading.Event()
    monitor_thread = None
    if NVML_AVAILABLE:
        try:
            monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(0.1, None, monitor_stop_event), daemon=True, name='GPU_Monitor')
            monitor_thread.start()
            logger.info("Started background GPU monitor (pynvml)")
        except Exception as e:
            logger.warning(f"Failed to start GPU monitor: {e}")
    
    # Load and prepare data
    try:
        df = pd.read_csv(args.dataset)
        logger.info(f"Dataset loaded: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    processor = DataProcessor(num_qubits=args.num_qubits, use_gpu=True)
    X, y = processor.prepare_data(df, sample_size=args.sample_size)
    X = processor.fit_transform(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    logger.info(f"Data split: Train={X_train.shape}, Test={X_test.shape}")

    # Apply CLI overrides for Nyström / Pegasos
    if args.nystrom_landmarks is not None:
        NYSTROM_CONFIG['QSVC_Standard']['landmarks'] = int(args.nystrom_landmarks)
        logger.info(f"Override: Nyström landmarks set to {args.nystrom_landmarks}")

    if args.prefer_pegasos:
        # Prefer Pegasos for large dataset sizes
        NYSTROM_CONFIG['QSVC_Standard']['use_nystrom'] = False
        NYSTROM_CONFIG['PegasosQSVC']['use_nystrom'] = False
        logger.info("Override: Preferring PegasosQSVC (stochastic) for large datasets")

    # Update transpile optimization level used during central precompile
    # Update global transpile level
    try:
        global TRANSPILE_OPT_LEVEL
        TRANSPILE_OPT_LEVEL = int(args.transpile_opt_level)
    except Exception:
        TRANSPILE_OPT_LEVEL = 2

    # Cache debug flag
    try:
        global _CACHE_DEBUG
        _CACHE_DEBUG = bool(args.cache_debug)
        if _CACHE_DEBUG:
            logger.info("Cache debug enabled: eviction logs will be emitted at DEBUG level")
            try:
                logger.setLevel(logging.DEBUG)
            except Exception:
                pass
        try:
            # start eviction monitor thread so we can observe evictions live
            if _CACHE_DEBUG:
                t = threading.Thread(target=_eviction_monitor, args=(30.0,), daemon=True, name='EvictionMonitor')
                t.start()
        except Exception:
            pass
    except Exception:
        pass

    # Decide strategy for estimation and build the model list. This uses runtime
    # context (args, X_train, X_test) so it must run inside main_centralized.
    chosen = 'full'
    if NYSTROM_CONFIG.get('QSVC_Standard', {}).get('use_nystrom', False):
        chosen = 'nystrom'
        landmarks = NYSTROM_CONFIG['QSVC_Standard'].get('landmarks')
    elif args.prefer_pegasos:
        chosen = 'pegasos'
        landmarks = None
    else:
        landmarks = None

    # Run a lightweight size estimate and warn user if needed
    try:
        _estimate_run_cost(len(X_train), len(X_test), method=chosen, n_landmarks=landmarks)
    except Exception:
        pass

    # Ensure Qiskit and Pegasos availability are detected before building the model list
    try:
        _lazy_qiskit_imports()
    except Exception:
        pass
    try:
        logger.info(f"PEGASOS_AVAILABLE={PEGASOS_AVAILABLE}")
    except Exception:
        pass

    models = []
    
    if args.model_group in ['all', 'quantum', 'qsvc']:
        # QSVC models with different characteristics
        models.extend([
            ('QSVC_Z_Simple', 'QSVC_Standard', {'feature_map': 'Z', 'reps': 1}),
            ('QSVC_ZZ_Standard', 'QSVC_Standard', {'feature_map': 'ZZ', 'reps': 2}),
            ('QSVC_Pauli_Standard', 'QSVC_Standard', {'feature_map': 'Pauli', 'reps': 2}),
            ('QSVC_Z_Precomputed', 'QSVC_Precomputed', {'feature_map': 'Z', 'reps': 1}),
            ('QSVC_ZZ_Precomputed', 'QSVC_Precomputed', {'feature_map': 'ZZ', 'reps': 2}),
            ('QSVC_Z_Callable', 'QSVC_Callable', {'feature_map': 'Z', 'reps': 1}),
            ('QSVC_ZZ_Callable', 'QSVC_Callable', {'feature_map': 'ZZ', 'reps': 2}),
        ])
        
        if PEGASOS_AVAILABLE:
            models.append(('QSVC_Pegasos', 'PegasosQSVC', {'feature_map': 'Z', 'reps': 1}))
    
    if args.model_group in ['all', 'quantum', 'vqc', 'qnn']:
        models.extend([
            ('VQC_COBYLA', 'VQC', {'optimizer': 'COBYLA'}),
            ('VQC_SPSA', 'VQC', {'optimizer': 'SPSA'}),
            ('VQC_ADAM', 'VQC', {'optimizer': 'ADAM'}),
        ])
    
    if args.model_group in ['all', 'quantum', 'qnn'] and QNN_AVAILABLE:
        models.extend([
            ('EstimatorQNN', 'QNN_Estimator', {}),
            ('SamplerQNN', 'QNN_Sampler', {}),
        ])
    
    if args.model_group in ['all', 'quantum', 'ensemble']:
        # Process ensemble models separately
        logger.info("Note: Ensemble models will be processed separately")

    requested_models: set[str] = set()
    if args.only_models:
        requested_models = {
            model.strip().lower() for model in args.only_models.split(',') if model.strip()
        }
        if requested_models:
            before_filter = len(models)
            models = [entry for entry in models if entry[0].lower() in requested_models]
            missing = requested_models.difference({entry[0].lower() for entry in models})
            if missing:
                logger.warning("Requested models not found or unavailable: %s", ', '.join(sorted(missing)))
            logger.info("Filtered model list based on --only-models; %s -> %s entries", before_filter, len(models))

    # Check if we have models to process OR if ensemble-only run
    is_ensemble_only = args.model_group == 'ensemble'
    
    if not models and not is_ensemble_only:
        logger.info("No models left to run after applying filters; exiting")
        print("No models selected for execution after applying --only-models filter. Exiting.")
        return

    # Initialize batch processing pipeline (skip if ensemble-only)
    if models:
        print("\n" + "="*80)
        print("🔥 INITIALIZING CENTRALIZED BATCH PROCESSING PIPELINE")
        print(f"📊 Processing {len(models)} models")
        print("="*80)

        pipeline = BatchProcessingPipeline(
            num_gpus=num_gpus,
            session_id=session_id,
            num_qubits=args.num_qubits,
            num_samples=args.sample_size,
            gpus_per_model=gpus_per_model
        )

        results = pipeline.process_all_models(models, X_train, y_train, X_test, y_test)
    else:
        results = []

    # Process ensemble models if requested
    if args.model_group in ['all', 'quantum', 'ensemble']:
        print("\n" + "="*80)
        print("🔥 PROCESSING ENSEMBLE MODELS")
        print("="*80)

        ensemble_models = [
            'QuantumRandomForest',
            'QuantumVotingEnsemble_Hard',
            'QuantumWeightedEnsemble'
        ]

        with ProcessPoolExecutor(max_workers=min(3, num_gpus)) as executor:
            futures = []

            for i, model_name in enumerate(ensemble_models):
                gpu_id = i % num_gpus

                if model_name == 'QuantumRandomForest':
                    future = executor.submit(
                        train_quantum_random_forest_gpu,
                        X_train, y_train, X_test, y_test,
                        args.num_qubits, session_id, gpu_id
                    )
                elif model_name == 'QuantumVotingEnsemble_Hard':
                    future = executor.submit(
                        train_quantum_voting_ensemble_gpu,
                        X_train, y_train, X_test, y_test,
                        args.num_qubits, session_id, gpu_id
                    )
                else:  # QuantumWeightedEnsemble
                    future = executor.submit(
                        train_quantum_weighted_ensemble_gpu,
                        X_train, y_train, X_test, y_test,
                        args.num_qubits, session_id, gpu_id
                    )

                futures.append((model_name, future))

            for model_name, future in futures:
                try:
                    future.result()
                    logger.info(f"✅ {model_name} complete")
                except Exception as e:
                    logger.error(f"❌ {model_name} failed: {e}")

    # Process classical models if requested
    if args.model_group in ['all', 'classical']:
        print("\n" + "="*80)
        print("🔷 PROCESSING CLASSICAL MODELS")
        print("="*80)
        
        try:
            train_classical_models(X_train, y_train, X_test, y_test, args.num_qubits, session_id)
        except Exception as e:
            logger.error(f"Classical models training failed: {e}")

    # Print summary
    print("\n" + "="*80)
    print("📊 TRAINING SUMMARY")
    print("="*80)

    results_manager = ThreadSafeResultsManager(session_id)
    results_df = results_manager.get_results_dataframe()

    if not results_df.empty:
        success_df = results_df[results_df['status'] == 'success']

        if not success_df.empty:
            print(f"\n✅ Successfully trained: {len(success_df)} models")
            print("\n🏆 TOP MODELS BY ACCURACY:")
            print("-"*60)

            for _, row in success_df.nlargest(min(10, len(success_df)), 'accuracy').iterrows():
                print(f"⚛️ {row['model']:<30} | Acc: {row['accuracy']:.4f} | "
                      f"F1: {row.get('f1_score', 0):.4f} | Time: {row.get('train_time', 0):.1f}s")

    print("="*80)
    # Stop GPU monitor if it was started
    try:
        if 'monitor_stop_event' in globals() and monitor_stop_event is not None:
            monitor_stop_event.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=2)
    except Exception:
        pass

    logger.info("🎊 Centralized training pipeline complete!")
    # If cache-debug enabled, print eviction counters summary
    try:
        if _CACHE_DEBUG:
            print("\nCache eviction summary:")
            for cname, cnt in _CACHE_EVICTION_COUNTERS.items():
                bytes_evicted = _CACHE_EVICTED_BYTES.get(cname, 0)
                mb = bytes_evicted / (1024**2) if bytes_evicted else 0
                print(f" - {cname}: evictions={cnt}, evicted={mb:.3f} MiB")
                logger.debug("cache_summary: %s evictions=%s evicted_bytes=%s", cname, cnt, bytes_evicted)
    except Exception:
        pass


def _trim_caches_if_needed(threshold_fraction: float = 0.75, max_evictions: int = 8):
    """Trim global caches when process RSS exceeds a fraction of system memory.

    This is a lightweight, best-effort protection: it evicts up to `max_evictions`
    items from the largest caches to relieve memory pressure and logs the evictions.
    """
    try:
        vm = psutil.virtual_memory()
        total = vm.total
        rss = psutil.Process(os.getpid()).memory_info().rss
        threshold = total * float(threshold_fraction)
        if rss <= threshold:
            return

        evicted = 0
        # Evict from process-local fidelity kernels first (likely largest), then simulators
        try:
            with _PROCESS_LOCAL_FIDELITY_KERNELS_LOCK:
                while evicted < max_evictions and rss > threshold and len(_PROCESS_LOCAL_FIDELITY_KERNELS) > 0:
                    k, v = _PROCESS_LOCAL_FIDELITY_KERNELS.popitem(last=False)
                    try:
                        _log_eviction('_PROCESS_LOCAL_FIDELITY_KERNELS', k, v, 'memory_trim')
                    except Exception:
                        pass
                    try:
                        _safe_cleanup(v, gpu_hint=k[0])
                    except Exception:
                        pass
                    try:
                        del v
                    except Exception:
                        pass
                    evicted += 1
                    rss = psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            pass

        try:
            with _PROCESS_LOCAL_SIMULATORS_LOCK:
                while evicted < max_evictions and rss > threshold and len(_PROCESS_LOCAL_SIMULATORS) > 0:
                    k, v = _PROCESS_LOCAL_SIMULATORS.popitem(last=False)
                    try:
                        _log_eviction('_PROCESS_LOCAL_SIMULATORS', k, v, 'memory_trim')
                    except Exception:
                        pass
                    try:
                        _safe_cleanup(v, gpu_hint=k[0])
                    except Exception:
                        pass
                    try:
                        del v
                    except Exception:
                        pass
                    evicted += 1
                    rss = psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            pass

        if _CACHE_DEBUG and evicted > 0:
            logger.debug("_trim_caches_if_needed: evicted %d items to reduce RSS", evicted)
    except Exception:
        pass


def _eviction_monitor(interval: float = 30.0):
    """Background thread that periodically logs eviction counters when cache-debug is enabled."""
    try:
        while True:
            time.sleep(interval)
            if not _CACHE_DEBUG:
                continue
            try:
                with _CACHE_EVICTION_LOCK:
                    if not _CACHE_EVICTION_COUNTERS:
                        logger.debug("eviction_monitor: no evictions yet")
                        continue
                    for cname, cnt in list(_CACHE_EVICTION_COUNTERS.items()):
                        bytes_evicted = _CACHE_EVICTED_BYTES.get(cname, 0)
                        mb = bytes_evicted / (1024 ** 2) if bytes_evicted else 0
                        logger.debug("eviction_monitor: %s evictions=%s evicted=%.3fMiB", cname, cnt, mb)
            except Exception:
                pass
    except Exception:
        pass


def _start_eviction_monitor_thread():
    t = threading.Thread(target=_eviction_monitor, args=(30.0,), daemon=True, name='EvictionMonitor')
    t.start()

# Run a small pre-run estimator to warn if chosen strategy will be infeasible
def _estimate_run_cost(n_train, n_test, method='full', n_landmarks=None):
    """Estimate memory and kernel sizes and log warnings for the user."""
    float_bytes = 4  # float32
    if method == 'full':
        entries = n_train * n_train
        bytes_needed = entries * float_bytes
        gb = bytes_needed / (1024**3)
        logger.info(f"ESTIMATE: Full kernel matrix for {n_train} samples ~ {entries:,} entries (~{gb:.2f} GiB)")
        try:
            if gb > max(1, min([g['memory_total'] for g in gpu_manager.gpu_info])) * 0.8:
                logger.warning("ESTIMATE WARNING: Full kernel may be large relative to GPU memory - consider Nyström or Pegasos")
        except Exception:
            pass
    elif method == 'nystrom' and n_landmarks is not None:
        m = min(n_landmarks, n_train)
        entries_mm = m * m
        entries_nm = n_train * m
        bytes_mm = entries_mm * float_bytes
        bytes_nm = entries_nm * float_bytes
        gb_mm = bytes_mm / (1024**3)
        gb_nm = bytes_nm / (1024**3)
        logger.info(f"ESTIMATE: Nyström m={m}: K_mm={entries_mm:,} (~{gb_mm:.2f} GiB), K_nm={entries_nm:,} (~{gb_nm:.2f} GiB)")
        try:
            if gb_mm + gb_nm > max(1, min([g['memory_total'] for g in gpu_manager.gpu_info])) * 0.9:
                logger.warning("ESTIMATE WARNING: Nyström memory footprint may still be large; reduce landmarks or use Pegasos")
        except Exception:
            pass
    else:
        logger.info("ESTIMATE: Using Pegasos/stochastic training (low memory footprint)")

    # This function only performs a sizing estimate and returns; decision-making
    # about which strategy to use (Nyström/Pegasos/full) is handled in
    # `main_centralized()` where full runtime context (args, X_train, etc.) is
    # available.
    return

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Set multiprocessing start method for CUDA
    mp.set_start_method('spawn', force=True)
    
    print("🔥 Using CENTRALIZED GPU management pipeline")
    print("💡 Key features:")
    print("   - Pre-compiled circuits (compile once, use everywhere)")
    print("   - Model-specific kernel computation methods")
    print("   - True multi-GPU parallelization")
    print("   - Preserved circuit complexity for fair comparison")
    main_centralized()