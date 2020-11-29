import numpy as np
import torch
from ase import Atoms
# from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.vasp.vasp2 import Vasp2
from ase.calculators.singlepoint import SinglePointCalculator

from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from ase.io.trajectory import Trajectory

images_test = Trajectory('data/test.traj')
images_train = Trajectory('data/train.traj')
'''
distances = np.linspace(2, 5, 100)
for x in range(0,1):
    print(mol)
    mol.set_cell([10, 10, 10])
    mol.wrap(pbc=True)
    vec = mol[2].position - mol[0].position
    for d in distances:
        mol[2].position = mol[0].position * vec * d
        mol[2].position = mol[0].position * vec * d
        images_train.append(mol)
        images_test.append(mol)
'''

Gs = {
    "default": {
        "G2": {
            "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4),
            "rs_s": [0],
        },
        "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
        "cutoff": 6,
    },
}

config = {
    "model": {
        "get_forces": True,
        "num_layers": 3,
        "num_nodes": 5,
        "batchnorm": False,
    },
    "optim": {
        "force_coefficient": 0.04,
        "lr": 1e-2,
        "batch_size": 32,
        "epochs": 100,
        "loss": "mse",
        "metric": "mae",
        "gpus": 0,
    },
    "dataset": {
        "raw_data": images_train,
        "val_split": 0.1,
        "fp_params": Gs,
        "save_fps": True,
        # feature scaling to be used - normalize or standardize
        # normalize requires a range to be specified
        "scaling": {"type": "normalize", "range": (0, 1)},
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        # Weights and Biases used for logging - an account(free) is required
        "logger": False,
    },
}

torch.set_num_threads(1)
trainer = AtomsTrainer(config)
trainer.train()

predictions = trainer.predict(images_test)

true_energies = np.array([image.get_potential_energy() for image in images_test])
pred_energies = np.array(predictions["energy"])

print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))
print("Energy MAE:", np.mean(np.abs(true_energies - pred_energies)))