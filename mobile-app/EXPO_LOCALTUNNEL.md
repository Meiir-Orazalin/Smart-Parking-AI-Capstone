# Expo Go via LocalTunnel

Use this fallback when Expo Go times out on `exp://<LAN-IP>:8081` (common on restricted Wi-Fi networks).

## Start (2 terminals)

### 1) Choose one subdomain name

Example: `smartpark-260223-0205`

Keep this same value in both terminals below.

### 2) Terminal A: start LocalTunnel

```powershell
cd "C:\Users\Алмат\Desktop\Capstone Project\mobile-app"
npx localtunnel --port 8081 --subdomain <SUBDOMAIN>
```

Expected output:

```text
your url is: https://<SUBDOMAIN>.loca.lt
```

### 3) Terminal B: start Expo with proxy URL

```powershell
cd "C:\Users\Алмат\Desktop\Capstone Project\mobile-app"
$env:EXPO_PACKAGER_PROXY_URL = "https://<SUBDOMAIN>.loca.lt"
npx expo start --go --lan --port 8081 -c
```

### 4) Open in Expo Go

```text
exp://<SUBDOMAIN>.loca.lt
```

## Quick check

Open this in phone browser:

```text
https://<SUBDOMAIN>.loca.lt/status
```

It should show `packager-status:running`.

## Stop

Press `Ctrl+C` in both terminals.

## Troubleshooting

- `failed to start tunnel` from Expo tunnel mode: use this LocalTunnel method instead.
- Subdomain already taken: choose a new `<SUBDOMAIN>`.
- Still not loading in Expo Go: verify `https://<SUBDOMAIN>.loca.lt/status` opens on phone first.
