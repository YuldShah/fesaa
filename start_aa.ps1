# start_aa.ps1
# Starts the AA assessment API services (Redis, Celery worker, uvicorn) and a
# Cloudflare quick tunnel.  Prints the tunnel URL so you can set it on the bot
# via /setapi in Telegram.
#
# Usage:
#   .\start_aa.ps1          — start everything
#   .\start_aa.ps1 -stop    — stop everything

param([switch]$stop)

$aaDir    = "C:\Users\Based\Desktop\Project\AA"
$python   = "$aaDir\venv\Scripts\python.exe"
$celery   = "$aaDir\venv\Scripts\celery.exe"
$cfExe    = "C:\Users\Based\scoop\shims\cloudflared.exe"
$pidDir   = "$aaDir\.pids"

function Get-EnvValue {
    param(
        [string]$Content,
        [string]$Key
    )

    if (-not $Content) { return '' }
    $match = [regex]::Match($Content, "(?m)^$([regex]::Escape($Key))=(.*)$")
    if ($match.Success) {
        return $match.Groups[1].Value.Trim()
    }
    return ''
}

# ─── Helper: stop everything ─────────────────────────────────────────────────

function Stop-AAServices {
    Write-Host '--- Stopping AA services ---'

    # Stop by PID files
    foreach ($name in @('uvicorn', 'celery', 'cloudflared', 'redis')) {
        $pidFile = "$pidDir\$name.pid"
        if (Test-Path $pidFile) {
            $procId = [int](Get-Content $pidFile -Raw).Trim()
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Remove-Item $pidFile -ErrorAction SilentlyContinue
            Write-Host "  Stopped $name (PID $procId)"
        }
    }

    # Also kill any stray processes on port 8000
    $lines = netstat -ano | Select-String ':8000'
    foreach ($line in $lines) {
        if ($line -match '\s+(\d+)\s*$') {
            $p = [int]$Matches[1]
            if ($p -gt 0) {
                Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Write-Host 'All AA services stopped.'
}

if ($stop) {
    Stop-AAServices
    exit 0
}

# ─── Pre-flight checks ──────────────────────────────────────────────────────

if (-not (Test-Path $python)) {
    Write-Error "Python venv not found at $python"
    exit 1
}
if (-not (Test-Path $cfExe)) {
    Write-Error "cloudflared not found at $cfExe"
    exit 1
}

# Create PID directory
if (-not (Test-Path $pidDir)) {
    New-Item -ItemType Directory -Path $pidDir | Out-Null
}

$envPath = "$aaDir\.env"
$envContent = if (Test-Path $envPath) { Get-Content $envPath -Raw } else { '' }
$namedTunnelToken = Get-EnvValue $envContent 'CLOUDFLARE_TUNNEL_TOKEN'
$publicApiBaseUrl = (Get-EnvValue $envContent 'PUBLIC_API_BASE_URL').TrimEnd('/')

# ─── Stop any existing services first ────────────────────────────────────────
Stop-AAServices
Start-Sleep -Seconds 2

# ─── 1. Ensure Redis is running ──────────────────────────────────────────────
Write-Host '--- Checking Redis ---'
$redisRunning = $false
try {
    $redisCheck = redis-cli ping 2>&1
    if ($redisCheck -eq 'PONG') { $redisRunning = $true }
} catch {}

if (-not $redisRunning) {
    Write-Host '  Redis not running — starting it...'
    $redisProc = Start-Process `
        -FilePath 'redis-server' `
        -WorkingDirectory $aaDir `
        -RedirectStandardOutput "$aaDir\redis_stdout.log" `
        -RedirectStandardError "$aaDir\redis_stderr.log" `
        -WindowStyle Hidden `
        -PassThru

    $redisProc.Id | Out-File "$pidDir\redis.pid" -Encoding ascii
    Write-Host "  redis-server started (PID $($redisProc.Id))"
    Start-Sleep -Seconds 2

    # Verify it came up
    try {
        $redisCheck = redis-cli ping 2>&1
        if ($redisCheck -ne 'PONG') {
            Write-Error "Redis failed to start. Check redis_stderr.log"
            exit 1
        }
    } catch {
        Write-Error "Redis failed to start. Check redis_stderr.log"
        exit 1
    }
}
Write-Host '  Redis is running (PONG)'

# ─── 2. Start Celery worker ─────────────────────────────────────────────────
Write-Host '--- Starting Celery worker ---'
$celeryProc = Start-Process `
    -FilePath $celery `
    -ArgumentList '-A', 'worker.celery_app', 'worker', '--loglevel=info', '--pool=solo', '--concurrency=1' `
    -WorkingDirectory $aaDir `
    -RedirectStandardOutput "$aaDir\celery_stdout.log" `
    -RedirectStandardError "$aaDir\celery_stderr.log" `
    -WindowStyle Hidden `
    -PassThru

$celeryProc.Id | Out-File "$pidDir\celery.pid" -Encoding ascii
Write-Host "  Celery worker started (PID $($celeryProc.Id))"

Start-Sleep -Seconds 3

# ─── 3. Start uvicorn (AA API on port 8000) ─────────────────────────────────
Write-Host '--- Starting AA API (uvicorn on port 8000) ---'
$uvicornProc = Start-Process `
    -FilePath $python `
    -ArgumentList '-m', 'uvicorn', 'api.main:app', '--host', '0.0.0.0', '--port', '8000' `
    -WorkingDirectory $aaDir `
    -RedirectStandardOutput "$aaDir\uvicorn_stdout.log" `
    -RedirectStandardError "$aaDir\uvicorn_stderr.log" `
    -WindowStyle Hidden `
    -PassThru

$uvicornProc.Id | Out-File "$pidDir\uvicorn.pid" -Encoding ascii
Write-Host "  Uvicorn started (PID $($uvicornProc.Id))"

Start-Sleep -Seconds 4

# Confirm port 8000 is listening
$listening = netstat -ano | Select-String ':8000.*LISTENING'
if (-not $listening) {
    Write-Error 'ERROR: uvicorn is NOT listening on port 8000. Check uvicorn_stderr.log'
    Get-Content "$aaDir\uvicorn_stderr.log" -Tail 20
    exit 1
}
Write-Host '  Confirmed: port 8000 is listening'

# ─── 4. Start Cloudflare tunnel ──────────────────────────────────────────────
$tunnelStderr = "$aaDir\tunnel_stderr.log"
'' | Set-Content -Path $tunnelStderr -Encoding UTF8

if ($namedTunnelToken) {
    if (-not $publicApiBaseUrl) {
        Write-Error 'ERROR: CLOUDFLARE_TUNNEL_TOKEN is set, but PUBLIC_API_BASE_URL is empty in .env'
        exit 1
    }

    Write-Host '--- Starting Cloudflare named tunnel ---'
    $cfProc = Start-Process `
        -FilePath $cfExe `
        -ArgumentList 'tunnel', 'run', '--token', $namedTunnelToken `
        -WorkingDirectory $aaDir `
        -RedirectStandardOutput "$aaDir\tunnel_stdout.log" `
        -RedirectStandardError $tunnelStderr `
        -WindowStyle Hidden `
        -PassThru

    $tunnelUrl = $publicApiBaseUrl
    Start-Sleep -Seconds 3
} else {
    Write-Host '--- Starting Cloudflare quick tunnel ---'
    $cfProc = Start-Process `
        -FilePath $cfExe `
        -ArgumentList 'tunnel', '--url', 'http://localhost:8000' `
        -WorkingDirectory $aaDir `
        -RedirectStandardOutput "$aaDir\tunnel_stdout.log" `
        -RedirectStandardError $tunnelStderr `
        -WindowStyle Hidden `
        -PassThru

    Write-Host '--- Waiting for tunnel URL (up to 30s) ---'
    $tunnelUrl = $null
    $attempts = 0
    while (-not $tunnelUrl -and $attempts -lt 30) {
        Start-Sleep -Seconds 1
        $attempts++
        if (Test-Path $tunnelStderr) {
            $logContent = Get-Content $tunnelStderr -Raw -ErrorAction SilentlyContinue
            if ($logContent -match 'https://[a-z0-9\-]+\.trycloudflare\.com') {
                $tunnelUrl = $Matches[0]
            }
        }
    }

    if (-not $tunnelUrl) {
        Write-Error 'ERROR: Could not capture tunnel URL after 30s. Check tunnel_stderr.log'
        exit 1
    }
}

$cfProc.Id | Out-File "$pidDir\cloudflared.pid" -Encoding ascii
Write-Host "  cloudflared started (PID $($cfProc.Id))"

# ─── 6. Summary ──────────────────────────────────────────────────────────────
Write-Host ''
Write-Host '============================================='
Write-Host '  AA Assessment API is live!'
Write-Host '============================================='
Write-Host ''
Write-Host "  Local:   http://localhost:8000"
Write-Host "  Tunnel:  $tunnelUrl"
if ($namedTunnelToken) {
    Write-Host '  Mode:    named tunnel'
} else {
    Write-Host '  Mode:    quick tunnel'
}
Write-Host ''
Write-Host "  Celery PID:      $($celeryProc.Id)"
Write-Host "  Uvicorn PID:     $($uvicornProc.Id)"
Write-Host "  Cloudflared PID: $($cfProc.Id)"
Write-Host ''
Write-Host '  Now send this command to the bot in Telegram:'
Write-Host ''
Write-Host "  /setapi $tunnelUrl"
Write-Host ''
Write-Host '  To stop all services:  .\start_aa.ps1 -stop'
Write-Host '============================================='

# Copy tunnel URL to clipboard if possible
try {
    $tunnelUrl | Set-Clipboard
    Write-Host '  (Tunnel URL copied to clipboard)'
} catch {}
