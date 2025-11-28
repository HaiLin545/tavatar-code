$cfgs = @(
    ".\config\people_m3c.toml"
    ".\config\people_m4c.toml"
    ".\config\people_f3c.toml"
    ".\config\people_f4c.toml"
)

$exp_name = "baseline-lite"
$device = 0
$epoch = 20

foreach ($cfg in $cfgs) {
    Write-Host "Training with config: $cfg"
    python main.py --train --cfg $cfg --device $device `
          exp_name=$exp_name `
          trainer.max_epochs=$epoch `
          smpl.subdivide=1 `
          model.learnable_scale=True `
          model.use_vertex_gaussians=False `
          train.edge_equal_loss_weight=0.00 `
          --animate
}

$exp_name = "tavatar-lite"
foreach ($cfg in $cfgs) {
    Write-Host "Training with config: $cfg"
    python main.py --train --cfg $cfg --device $device `
          exp_name=$exp_name `
          trainer.max_epochs=$epoch `
          smpl.subdivide=0 `
          model.learnable_scale=False `
          model.use_vertex_gaussians=True `
          train.edge_equal_loss_weight=0.01 `
          --animate
}
