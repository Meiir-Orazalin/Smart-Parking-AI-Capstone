# SmartParkAI Mobile App Run Guide

This guide explains how to install and run the SmartParkAI mobile app from a fresh copy of the submitted project files.

## 1. Requirements

Install these before running the app:

- Node.js 20 or newer
- npm
- Expo Go app on an iOS or Android phone
- A terminal or command prompt

Optional:

- Android Studio if you want to run on an Android emulator
- Xcode if you want to run on an iOS simulator on macOS

## 2. Project Folder

After extracting the submission zip, open a terminal in the extracted folder and go to the mobile app directory:

```bash
cd Moving
```

The app source code is inside `Moving/src`.

## 3. Install Dependencies

Run:

```bash
npm install
```

This installs Expo, React Native, navigation libraries, AsyncStorage, and the other packages listed in `package.json`.

## 4. Configure Environment

The submission includes `.env.example`.

For the default demo mode, create a `.env` file from it:

```bash
cp .env.example .env
```

Default values:

```bash
EXPO_PUBLIC_SMARTPARK_DATA_MODE=hybrid
EXPO_PUBLIC_SMARTPARK_API_URL=http://localhost:5001
```

Data modes:

- `mock`: uses only local demo data
- `hybrid`: tries the backend first, then falls back to local demo data
- `api`: uses backend data only

If you are running without a backend, use:

```bash
EXPO_PUBLIC_SMARTPARK_DATA_MODE=mock
```

If you are connecting to a backend from a physical phone, do not use `localhost` unless the backend is running on the phone. Use your computer LAN IP address or a tunnel URL, for example:

```bash
EXPO_PUBLIC_SMARTPARK_API_URL=http://192.168.1.20:5001
```

## 5. Run the App

Start Expo:

```bash
npm start
```

Or use the included local startup helper on macOS/Linux:

```bash
npm run start:app
```

When Expo starts, scan the QR code with Expo Go.

## 6. Run on Emulator or Web

Android emulator:

```bash
npm run android
```

iOS simulator on macOS:

```bash
npm run ios
```

Web preview:

```bash
npm run web
```

## 7. Backend Endpoints Used by the App

When `EXPO_PUBLIC_SMARTPARK_DATA_MODE` is `hybrid` or `api`, the app expects these endpoints from `EXPO_PUBLIC_SMARTPARK_API_URL`:

- `GET /status`
- `GET /lots`
- `GET /dashboard`
- `GET /alerts`
- `GET /cameras`
- `POST /cameras`
- `GET /profile`

Typed request and response contracts are documented in:

```text
src/types/apiContracts.ts
```

## 8. LocalTunnel Fallback

If Expo Go cannot connect on the same Wi-Fi network, use the LocalTunnel instructions in:

```text
EXPO_LOCALTUNNEL.md
```

That file explains how to expose the Expo dev server through a temporary tunnel.

## 9. Troubleshooting

If dependencies fail to install:

```bash
npm cache verify
npm install
```

If Expo opens but the app shows backend warnings:

- Use `EXPO_PUBLIC_SMARTPARK_DATA_MODE=mock` for a standalone demo.
- Check that the backend URL is reachable from the phone.
- Restart Expo after changing `.env`.

If the QR code does not load on a phone:

- Make sure the phone and computer are on the same network.
- Try `npm run start:app`.
- Use the LocalTunnel fallback in `EXPO_LOCALTUNNEL.md`.

## 10. Main App Features

The submitted SmartParkAI mobile app includes:

- Driver mode and operations mode
- Parking lot list and lot detail screens
- Occupancy and availability status
- Operations dashboard
- Camera management workflow
- Mock, hybrid, and backend-only data modes
- Local persistence for demo camera and operations state

