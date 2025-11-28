
$subjects = @(
    "male-3-casual"
    "male-4-casual"
    "female-3-casual"
    "female-4-casual"
)

$exp_name = "tavatar-lite"

foreach ($subj in $subjects) {
    Write-Host "Evaluating subject: $subj"
    python main.py --resume_dir ".\output\$exp_name\people_snapshot\$subj" --animate
}


$exp_name = "baseline-lite"
foreach ($subj in $subjects) {
    Write-Host "Evaluating subject: $subj"
    python main.py --resume_dir ".\output\$exp_name\people_snapshot\$subj" --animate
}

