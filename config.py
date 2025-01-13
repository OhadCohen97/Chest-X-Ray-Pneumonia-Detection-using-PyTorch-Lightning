imges_paths = "/home/dsi/ohadico97/chest-XRay/data_copy/"
n_classes = 2
classes= {
    "Normal": 0,
  "Pneumonia": 1
}

checkpoint = True
checkpoint_dir= "/home/dsi/ohadico97/chest-XRay/checkpoints"
verbose = True
optim = "adam"
lr = 0.0001
epochs = 80
batch_size= 32
num_workers= 4
# lr_sched: step
# step:
#   step_size: 2
#   gamma: 0.98
# plateau:
#   factor: 0.5
#   patience: 5
