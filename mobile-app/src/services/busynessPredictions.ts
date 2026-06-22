import { ParkingLot } from "../types/models";

export type BusynessPoint = {
  hour: number;
  busynessPct: number;
};

export type BusynessDay = "today" | "tomorrow" | "weekend";

const explicitPredictions: Record<string, BusynessPoint[]> = {
  "north-lot": [
    { hour: 8, busynessPct: 18 },
    { hour: 9, busynessPct: 24 },
    { hour: 10, busynessPct: 31 },
    { hour: 11, busynessPct: 39 },
    { hour: 12, busynessPct: 48 },
    { hour: 13, busynessPct: 55 },
    { hour: 14, busynessPct: 61 },
    { hour: 15, busynessPct: 66 },
    { hour: 16, busynessPct: 70 },
    { hour: 17, busynessPct: 74 },
    { hour: 18, busynessPct: 78 },
    { hour: 19, busynessPct: 80 },
    { hour: 20, busynessPct: 77 },
    { hour: 21, busynessPct: 70 }
  ],
  "central-garage": [
    { hour: 8, busynessPct: 38 },
    { hour: 9, busynessPct: 52 },
    { hour: 10, busynessPct: 64 },
    { hour: 11, busynessPct: 73 },
    { hour: 12, busynessPct: 82 },
    { hour: 13, busynessPct: 88 },
    { hour: 14, busynessPct: 91 },
    { hour: 15, busynessPct: 89 },
    { hour: 16, busynessPct: 84 },
    { hour: 17, busynessPct: 79 },
    { hour: 18, busynessPct: 74 },
    { hour: 19, busynessPct: 67 },
    { hour: 20, busynessPct: 60 },
    { hour: 21, busynessPct: 54 }
  ],
  "west-plaza": [
    { hour: 8, busynessPct: 55 },
    { hour: 9, busynessPct: 66 },
    { hour: 10, busynessPct: 75 },
    { hour: 11, busynessPct: 83 },
    { hour: 12, busynessPct: 90 },
    { hour: 13, busynessPct: 94 },
    { hour: 14, busynessPct: 96 },
    { hour: 15, busynessPct: 97 },
    { hour: 16, busynessPct: 95 },
    { hour: 17, busynessPct: 92 },
    { hour: 18, busynessPct: 88 },
    { hour: 19, busynessPct: 81 },
    { hour: 20, busynessPct: 74 },
    { hour: 21, busynessPct: 68 }
  ],
  "south-deck": [
    { hour: 8, busynessPct: 21 },
    { hour: 9, busynessPct: 28 },
    { hour: 10, busynessPct: 34 },
    { hour: 11, busynessPct: 42 },
    { hour: 12, busynessPct: 50 },
    { hour: 13, busynessPct: 57 },
    { hour: 14, busynessPct: 62 },
    { hour: 15, busynessPct: 65 },
    { hour: 16, busynessPct: 68 },
    { hour: 17, busynessPct: 72 },
    { hour: 18, busynessPct: 76 },
    { hour: 19, busynessPct: 73 },
    { hour: 20, busynessPct: 66 },
    { hour: 21, busynessPct: 58 }
  ],
  "visitor-garage-l2": [
    { hour: 8, busynessPct: 34 },
    { hour: 9, busynessPct: 46 },
    { hour: 10, busynessPct: 58 },
    { hour: 11, busynessPct: 69 },
    { hour: 12, busynessPct: 79 },
    { hour: 13, busynessPct: 84 },
    { hour: 14, busynessPct: 87 },
    { hour: 15, busynessPct: 83 },
    { hour: 16, busynessPct: 78 },
    { hour: 17, busynessPct: 73 },
    { hour: 18, busynessPct: 70 },
    { hour: 19, busynessPct: 64 },
    { hour: 20, busynessPct: 57 },
    { hour: 21, busynessPct: 49 }
  ],
  "west-wing": [
    { hour: 8, busynessPct: 48 },
    { hour: 9, busynessPct: 60 },
    { hour: 10, busynessPct: 72 },
    { hour: 11, busynessPct: 81 },
    { hour: 12, busynessPct: 89 },
    { hour: 13, busynessPct: 93 },
    { hour: 14, busynessPct: 96 },
    { hour: 15, busynessPct: 95 },
    { hour: 16, busynessPct: 91 },
    { hour: 17, busynessPct: 87 },
    { hour: 18, busynessPct: 82 },
    { hour: 19, busynessPct: 76 },
    { hour: 20, busynessPct: 70 },
    { hour: 21, busynessPct: 62 }
  ]
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function dayAdjustment(day: BusynessDay): number {
  if (day === "tomorrow") return -4;
  if (day === "weekend") return 9;
  return 0;
}

function buildFallbackPrediction(lot: ParkingLot): BusynessPoint[] {
  const occupancyRatio = lot.capacity === 0 ? 0.35 : clamp(lot.occupied / lot.capacity, 0, 1);
  const baseline = 18 + occupancyRatio * 54;
  const curve = [0, 6, 12, 19, 26, 31, 35, 38, 40, 43, 45, 42, 35, 27];

  return curve.map((lift, index) => ({
    hour: index + 8,
    busynessPct: Math.round(clamp(baseline + lift - index * 1.2, 12, 98))
  }));
}

function expandToFullDay(points: BusynessPoint[]): BusynessPoint[] {
  const byHour = new Map(points.map((point) => [point.hour, point.busynessPct]));
  const first = points[0]?.busynessPct ?? 25;
  const last = points[points.length - 1]?.busynessPct ?? first;

  return Array.from({ length: 24 }, (_, hour) => {
    if (byHour.has(hour)) {
      return { hour, busynessPct: byHour.get(hour) ?? first };
    }

    if (hour < 8) {
      return { hour, busynessPct: Math.round(clamp(first - (8 - hour) * 3.5, 8, 100)) };
    }

    return { hour, busynessPct: Math.round(clamp(last - (hour - 21) * 5, 8, 100)) };
  });
}

function applyOccupancyAnchor(points: BusynessPoint[], lot: ParkingLot, day: BusynessDay): BusynessPoint[] {
  const actualOccupancyPct = lot.capacity === 0 ? 0 : Math.round((lot.occupied / lot.capacity) * 100);
  const nowHour = new Date().getHours();
  const adjustment = dayAdjustment(day);

  return points.map((point) => {
    const distanceFromNow = Math.abs(point.hour - nowHour);
    const currentWeight = day === "today" ? Math.max(0.35, 1 - distanceFromNow * 0.08) : 0.45;
    const occupancyFloor = actualOccupancyPct * currentWeight;
    const adjustedPrediction = point.busynessPct + adjustment;

    return {
      hour: point.hour,
      busynessPct: Math.round(clamp(Math.max(adjustedPrediction, occupancyFloor), 8, 100))
    };
  });
}

export function getBusynessPrediction(lot: ParkingLot, day: BusynessDay = "today"): BusynessPoint[] {
  const base = expandToFullDay(explicitPredictions[lot.id] ?? buildFallbackPrediction(lot));
  return applyOccupancyAnchor(base, lot, day);
}
