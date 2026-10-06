# Run a Decision Index request list through jjava (one request at a time), from the harness checkout.
#   .\run_pass.ps1 -Rows compat-86.jsonl.gz -Out runs\jjava-compat86 [-Model jjava-latest]
# Reads the account key from the stack's key file into DECISION_INDEX_API_KEY (never printed). The runner is sequential.
param([Parameter(Mandatory)][string]$Rows, [Parameter(Mandatory)][string]$Out, [string]$Model = "jjava-latest",
      [string]$Harness = "C:\Users\jwals\octo\decision-index", [string]$KeyFile = "C:\Users\jwals\llama-stack\.keys\live-test.key",
      [string]$Base = "http://127.0.0.1:1234/jev")
Set-Location $Harness
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "C:\Users\jwals\llama-stack\bench\decider\decision_index"
$env:DECISION_INDEX_API_KEY = (Get-Content $KeyFile -Raw).Trim()
& .venv\Scripts\python.exe -m decision_index run --engine jjava_engine:JjavaEngine --option "base_url=$Base" --option "model=$Model" --rows $Rows --out $Out
