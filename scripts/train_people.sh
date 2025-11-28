

cfgs=(
  "./config/people_m3c.toml"
  # "./config/people_m4c.toml"
  # "./config/people_f3c.toml"
  # "./config/people_f4c.toml"
)

exp_name="baseline"
device=0
epoch=20

for cfg in "${cfgs[@]}"; do
    echo "Training with config: $cfg"
    python main.py --train --cfg "$cfg" --device "$device"  \
          exp_name=$exp_name \
          trainer.max_epochs=$epoch \
          smpl.subdivide=1 \
          model.learnable_scale=True \
          model.use_vertex_gaussians=False \
          train.edge_equal_loss_weight=0.01 \
          --animate --test
done