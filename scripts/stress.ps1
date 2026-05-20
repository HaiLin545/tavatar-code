$stress = @(
    "0.00"
)

$subj = "m3c"
foreach ($stress_level in $stress) {
    $exp_name = "tavatar-stress-$stress_level"
    python main.py --cfg ".\config\people_$subj.toml" --train --test --animate dataset.noise=$stress_level --exp_name=$exp_name
}

$stress = @(
    "0.00"
    "0.005"
    "0.01"
    "0.02"
)


foreach ($stress_level in $stress) {
    $exp_name = "baseline-stress-$stress_level"
    python main.py --cfg ".\config\people_${subj}_baseline.toml" --train --test --animate dataset.noise=$stress_level --exp_name=$exp_name
}

