$ErrorActionPreference = 'Stop'

# Pick the interface used for the default IPv4 route.
$defaultRoute = Get-NetRoute -DestinationPrefix '0.0.0.0/0' |
  Sort-Object RouteMetric, InterfaceMetric |
  Select-Object -First 1

if (-not $defaultRoute) {
  throw 'Could not determine default IPv4 route.'
}

$ip = Get-NetIPAddress -InterfaceIndex $defaultRoute.InterfaceIndex -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike '169.254*' -and $_.IPAddress -notlike '127*' } |
  Select-Object -First 1 -ExpandProperty IPAddress

if (-not $ip) {
  throw "Could not determine IPv4 address for interface index $($defaultRoute.InterfaceIndex)."
}

$env:REACT_NATIVE_PACKAGER_HOSTNAME = $ip
Write-Host "Using REACT_NATIVE_PACKAGER_HOSTNAME=$ip"

npx expo start --go --lan --port 8081 -c
