<#
.SYNOPSIS
    Забирает с серверов новые файлы collect_capture.sh (захват и журнал SSH).

.DESCRIPTION
    Для каждого сервера из pull_captures.json:
      1. по SSH получает список файлов в каталоге сбора (размер, время);
      2. пропускает файл захвата, который ещё пишется (самый новый в сессии,
         изменён меньше 3 минут назад);
      3. скачивает новые и недокачанные файлы через .part с проверкой размера.
    Кольцо на сервере перезаписывает старые файлы, поэтому запускать нужно
    чаще, чем оно заполняется (см. README); -Install ставит задачу в
    Планировщик Windows.

    Нужен вход по SSH-ключу: скрипт работает без пароля (BatchMode).

.EXAMPLE
    .\pull_captures.ps1 -DryRun          # что будет скачано, без скачивания
    .\pull_captures.ps1                  # скачать
    .\pull_captures.ps1 -Install -EveryHours 2
    .\pull_captures.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot 'pull_captures.json'),
    [switch]$DryRun,
    [switch]$Install,
    [switch]$Uninstall,
    [ValidateRange(1, 24)][int]$EveryHours = 2
)

$ErrorActionPreference = 'Stop'
$TaskName = 'cic-pull-captures'
# Файл кольца, изменённый недавно, ещё пишется dumpcap — его не трогаем
$ActiveSeconds = 180
$Patterns = @('capture_*.pcapng', 'ssh_*.log', 'session.json')

function Write-Log([string]$Message) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Host $line
    if ($script:LogPath) { Add-Content -Path $script:LogPath -Value $line -Encoding UTF8 }
}

function ConvertTo-ShQuoted([string]$Text) {
    return "'" + ($Text -replace "'", "'\''") + "'"
}

function Invoke-Native([scriptblock]$Command) {
    # В Windows PowerShell 5.1 при ErrorActionPreference=Stop любая строка
    # stderr внешней программы (например, «Warning: Permanently added ...»
    # от ssh) — фатальная ошибка. Код возврата проверяем сами.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Command 2>&1 | ForEach-Object { "$_" }
        return [pscustomobject]@{ Output = @($output); ExitCode = $LASTEXITCODE }
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Invoke-Remote([string]$Target, [string]$Script) {
    # Команду передаём в base64: Windows PowerShell 5.1 портит кавычки
    # в аргументах внешних программ, а base64 состоит только из [A-Za-z0-9+/=]
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Script))
    $result = Invoke-Native { ssh -o BatchMode=yes -o ConnectTimeout=20 $Target "echo $encoded | base64 -d | bash" }
    if ($result.ExitCode -ne 0) {
        throw "ssh $Target завершился с кодом $($result.ExitCode): $($result.Output -join ' ')"
    }
    return $result.Output
}

function Get-RemoteListing($Server) {
    $names = ($Patterns | ForEach-Object { "-name $(ConvertTo-ShQuoted $_)" }) -join ' -o '
    $script = @"
printf 'NOW\t%s\n' "`$(date +%s)"
cd $(ConvertTo-ShQuoted $Server.remote) || exit 3
find . -type f \( $names \) -printf '%P\t%s\t%T@\n'
"@
    return Invoke-Remote $Server.target $script
}

function Get-PullPlan([string[]]$Listing, [string]$LocalRoot) {
    <# Строки «NOW\t<epoch>» и «<путь>\t<размер>\t<mtime>» -> что качать. #>
    $now = $null
    $files = @()
    foreach ($line in $Listing) {
        $parts = "$line".Split("`t")
        if ($parts[0] -eq 'NOW') { $now = [double]$parts[1]; continue }
        if ($parts.Count -ne 3) { continue }
        $files += [pscustomobject]@{
            Path  = $parts[0]
            Size  = [int64]$parts[1]
            MTime = [double]::Parse($parts[2], [Globalization.CultureInfo]::InvariantCulture)
        }
    }
    if ($null -eq $now) { throw 'сервер не вернул текущее время (NOW)' }

    # Самый новый файл захвата в каждой сессии, если он свежий, ещё пишется
    $active = @{}
    $files | Where-Object { (Split-Path $_.Path -Leaf) -like 'capture_*' } |
        Group-Object { Split-Path $_.Path -Parent } | ForEach-Object {
            $newest = $_.Group | Sort-Object MTime | Select-Object -Last 1
            if ($now - $newest.MTime -lt $ActiveSeconds) { $active[$newest.Path] = $true }
        }

    foreach ($file in $files) {
        $local = Join-Path $LocalRoot ($file.Path -replace '/', '\')
        $leaf = Split-Path $file.Path -Leaf
        $reason = $null
        if ($active.ContainsKey($file.Path)) {
            $reason = 'skip-active'
        } elseif ($leaf -eq 'session.json') {
            $reason = 'refresh'  # маленький и меняется — берём всегда
        } elseif (-not (Test-Path -LiteralPath $local)) {
            $reason = 'new'
        } elseif ((Get-Item -LiteralPath $local).Length -ne $file.Size) {
            $reason = 'size-changed'
        }
        if ($reason) {
            [pscustomobject]@{ Path = $file.Path; Size = $file.Size; Local = $local; Reason = $reason }
        }
    }
}

function Receive-File($Server, $Item) {
    $directory = Split-Path $Item.Local -Parent
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $part = "$($Item.Local).part"
    $remote = '{0}:{1}/{2}' -f $Server.target, $Server.remote.TrimEnd('/'), $Item.Path
    $result = Invoke-Native { scp -q -o BatchMode=yes -o ConnectTimeout=20 $remote $part }
    if ($result.ExitCode -ne 0) {
        Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        throw "scp завершился с кодом $($result.ExitCode): $($result.Output -join ' ')"
    }
    $got = (Get-Item -LiteralPath $part).Length
    if ($got -ne $Item.Size) {
        Remove-Item -LiteralPath $part -Force
        throw "размер не совпал ($got вместо $($Item.Size)) — файл, видимо, ещё меняется"
    }
    Move-Item -LiteralPath $part -Destination $Item.Local -Force
}

function Sync-Server($Server) {
    Write-Log "== $($Server.name) ($($Server.target))"
    $plan = @(Get-PullPlan (Get-RemoteListing $Server) $Server.local)
    $todo = @($plan | Where-Object Reason -ne 'skip-active')
    $active = @($plan | Where-Object Reason -eq 'skip-active')
    foreach ($item in $active) { Write-Log "  пропуск (ещё пишется): $($item.Path)" }
    $megabytes = [math]::Round(($todo | Measure-Object Size -Sum).Sum / 1MB, 1)
    Write-Log "  к скачиванию: $($todo.Count) файлов, $megabytes МБ"
    $failed = 0
    foreach ($item in $todo) {
        if ($DryRun) { Write-Log "  [dry-run] $($item.Reason): $($item.Path)"; continue }
        try {
            Receive-File $Server $item
            Write-Log "  OK $($item.Reason): $($item.Path) ($([math]::Round($item.Size / 1MB, 1)) МБ)"
        } catch {
            $failed++
            Write-Log "  ОШИБКА $($item.Path): $($_.Exception.Message)"
        }
    }
    return $failed
}

function Install-Task {
    $arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Config "{1}"' -f $PSCommandPath, (Resolve-Path $Config)
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $PSScriptRoot
    # Без -RepetitionDuration повторяется бессрочно (Windows 10/11)
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Hours $EveryHours)
    # StartWhenAvailable: если компьютер спал в момент запуска — выполнить при пробуждении
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 3)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Description 'Скачивание захватов collect_capture.sh с серверов' -Force | Out-Null
    Write-Host "Задача '$TaskName' установлена: каждые $EveryHours ч, первый запуск через минуту."
    Write-Host "Журнал: $(Join-Path $PSScriptRoot 'pull.log'). Удалить: .\pull_captures.ps1 -Uninstall"
}

function Main {
    if ($Uninstall) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Задача '$TaskName' удалена."
        return
    }
    if (-not (Test-Path -LiteralPath $Config)) {
        throw "Нет конфигурации $Config. Скопируйте pull_captures.example.json в pull_captures.json и впишите серверы."
    }
    foreach ($tool in 'ssh', 'scp') {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "Нет $tool. Установите клиент OpenSSH: Параметры -> Приложения -> Дополнительные компоненты."
        }
    }
    if ($Install) { Install-Task; return }

    $servers = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
    $script:LogPath = Join-Path $PSScriptRoot 'pull.log'
    $failed = 0
    foreach ($server in $servers) {
        try {
            $failed += Sync-Server $server
        } catch {
            $failed++
            Write-Log "  ОШИБКА $($server.name): $($_.Exception.Message)"
        }
    }
    if ($failed) { Write-Log "Готово с ошибками: $failed"; exit 1 }
    Write-Log 'Готово'
}

# При dot-source (тесты) — только функции, без запуска
if ($MyInvocation.InvocationName -ne '.') { Main }
