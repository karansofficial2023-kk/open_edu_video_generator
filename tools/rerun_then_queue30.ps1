# 1) re-render the 3 videos that failed the final safety check (queue10), 2) then continue queue30 (older 6 lessons).
# Both steps use the current safety rules (prompts, negative prompts, vision checks, motion check, final-frame QA).
$root = "D:\Python\claude version\open_edu_video_generator"
$py = "D:/AI/ComfyUI_windows_portable/python_embeded/python.exe"
Set-Location $root
$src = "C:\Users\User\output"
$inputs = @("Female Reproductive Anatomy", "Male Reproductive System in Humans", "Male Sex Hormones") | ForEach-Object { "$src\$($_)_contract.json" }
& $py -u scripts/local_entry.py --config config.realistic.12gb.yaml --input $inputs --output "outputs\queue10" *> "outputs\rerun3.log"
& $py -u tools/run_queue5.py 10 25 queue30 *> "outputs\queue30_runner.log"
