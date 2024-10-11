from sys import argv

if len(argv) != 3:
    print(
        "Please pass the number of modes and run id e.g. python train_dynemo.py 6 1"
    )
    exit()

n_modes = int(argv[1])
run = int(argv[2])


from osl_dynamics.data import Data
from osl_dynamics.models.dynemo import Config
from osl_dynamics.models.dynemo import Model

BASE_DIR = "/run/user/1001/gvfs/smb-share:server=sum-lpnc-nas.u-ga.fr,share=securevault/LPNC-SecureVault/MEGAGING/Processed/osl_processing"
import os
model_dir = f"/home/clement/Bureau/train_dynemo_1_90/{n_modes:02d}_modes/run{run:02d}/model"
os.makedirs(f"{model_dir}", exist_ok=True)

data = Data(f"{BASE_DIR}/hmm_dataprep_1_90", n_jobs=16)

# Create model
n_epochs = 50

config = Config(
    n_modes=n_modes,
    n_channels=data.n_channels,
    sequence_length=100,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=5,
    n_kl_annealing_epochs=n_epochs/2,
    batch_size=32,
    learning_rate=1e-4,
    n_epochs=n_epochs,
)

model = Model(config)
model.summary()

# Full training
init_history = model.random_subset_initialization(data, n_epochs=1, n_init=3, take=0.2)
history = model.fit(data)

# Save trained model
model.save(model_dir)

# Calculate the free energy
free_energy = model.free_energy(data)
history["free_energy"] = free_energy

## Save training history
import pickle
with open(f"{model_dir}/history.pkl", "wb") as file:
    pickle.dump(history, file)

with open(f"{model_dir}/loss.dat", "w") as file:
    file.write(f"ll_loss = {history['loss'][-1]}\n")
    file.write(f"free_energy = {free_energy}\n")
