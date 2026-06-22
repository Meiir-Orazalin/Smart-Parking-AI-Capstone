# SmartPark AI Mobile Prototype

For full setup instructions, see `../SMARTPARKAI_RUN_GUIDE.md` in the submission root.

## Run

```bash
cd Moving
npm install
npx expo start
```

## Environment

Create a `.env` in `Moving/` when needed:

```bash
EXPO_PUBLIC_SMARTPARK_DATA_MODE=hybrid
EXPO_PUBLIC_SMARTPARK_API_URL=http://localhost:5001
```

`EXPO_PUBLIC_SMARTPARK_DATA_MODE` options:
- `mock`: fully local demo mode
- `hybrid`: try backend endpoints, fallback to local mocks
- `api`: backend-only mode (no mock fallback)

Endpoints used by the app are aligned with `../docs/API_ENDPOINTS.md`:
- `GET /status`
- `GET /lots`
- `GET /dashboard`
- `GET /alerts`
- `GET /cameras`
- `POST /cameras`
- `GET /profile`

## Contracts

Typed API contracts and sample payloads are defined in:

- `src/types/apiContracts.ts`

## Notes

- Built with Expo + TypeScript + React Navigation.
- Camera list and operator alert workflow states persist locally via AsyncStorage.
- App remains backend-ready: replace mock endpoints progressively without UI rewrites.
- If Expo Go times out on LAN, use `EXPO_LOCALTUNNEL.md`.
- In `mock`/`hybrid` mode, app shows a local-data notice and supports "Reset Demo Data" from Profile.
