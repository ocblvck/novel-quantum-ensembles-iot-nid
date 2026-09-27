# Quantum ML for IoT Intrusion Detection

This repository contains the GPU-focused experiment script used for quantum machine learning experiments on IoT intrusion detection data. The main entry point is `iot_multigpu.py`, which runs quantum models, ensemble variants, and classical baselines through a centralized multi-GPU pipeline.

## Repository contents

- `iot_multigpu.py`: main experiment script
- `requirements.txt`: Python dependencies used for this repo
- `download_dataset.sh`: dataset download instructions

## Dataset

The dataset is not stored in this repository. Download the IoTID20 CSV separately and place it in the repository root as `IoT_Original_Distribution.csv`.

Sources:

- Kaggle: https://www.kaggle.com/datasets/subhajournal/iotid20-iot-botnet-dataset
- IEEE DataPort: https://ieee-dataport.org/open-access/iot-network-intrusion-dataset

## Environment

- Python 3.10 or newer
- NVIDIA GPU with CUDA support
- Qiskit Aer GPU build compatible with the installed CUDA version

Install the base dependencies with:

```bash
pip install -r requirements.txt
```

If you are using GPU simulation, install the appropriate `qiskit-aer-gpu` build for your CUDA environment.

## Running the script

Example run:

```bash
python iot_multigpu.py \
    --num_qubits 10 \
    --sample_size 5000 \
    --dataset IoT_Original_Distribution.csv \
    --model_group qsvc
```

Useful model groups:

- `qsvc`
- `vqc`
- `qnn`
- `ensemble`
- `classical`
- `all`

Run `python iot_multigpu.py --help` for the full CLI.

## Outputs

The script writes generated artifacts to directories such as:

- `results/`
- `checkpoints/`
- `gpu_logs/`
- `kernel_cache/`

These outputs are excluded from version control.

## Notes

- Large runs can require substantial GPU memory.
- The heavy kernel path is designed for GPU execution rather than CPU fallback.
- Start with a smaller configuration before launching long multi-GPU runs.

## Authors and citation

This code was written by Chibuike C. Okekeogbu (GitHub handle `ocblvck`), a doctoral researcher at North Carolina A&T State University, with Sayed A. Mostafa and Ahmad Patooghy as co-authors of the published study it implements.

If you use this code or build on it, please cite the paper:

C. C. Okekeogbu, S. A. Mostafa, and A. Patooghy, "Novel Quantum Ensemble Machine Learning Models for IoT Intrusion Detection," in *2026 IEEE 19th Dallas Circuits and Systems Conference (DCAS)*, 2026, pp. 1-6, doi: 10.1109/DCAS69364.2026.11544356.

```bibtex
@inproceedings{okekeogbu2026quantumensemble,
  author    = {Okekeogbu, Chibuike C. and Mostafa, Sayed A. and Patooghy, Ahmad},
  title     = {Novel Quantum Ensemble Machine Learning Models for {IoT} Intrusion Detection},
  booktitle = {2026 IEEE 19th Dallas Circuits and Systems Conference (DCAS)},
  year      = {2026},
  pages     = {1--6},
  doi       = {10.1109/DCAS69364.2026.11544356}
}
```

Reuse under the MIT License below is welcome. Please keep the copyright notice and, where practical, the citation above.

## License

This project is released under the MIT License. See `LICENSE` for details.
