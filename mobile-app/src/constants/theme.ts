export const theme = {
  colors: {
    backgroundTop: "#070B16",
    backgroundBottom: "#111A2E",
    card: "#131D31",
    cardAlt: "#182540",
    border: "#2A3857",
    textPrimary: "#EAF1FF",
    textSecondary: "#9DB0CF",
    textMuted: "#7C8EAE",
    accent: "#4FA4FF",
    success: "#2ECC71",
    warning: "#F7B731",
    danger: "#FF5D5D",
    overlay: "rgba(8, 12, 22, 0.72)",
    white: "#FFFFFF",
    black: "#000000"
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 12,
    lg: 16,
    xl: 20,
    xxl: 24
  },
  radii: {
    sm: 10,
    md: 14,
    lg: 16,
    xl: 20,
    pill: 999
  },
  typography: {
    h1: {
      fontSize: 30,
      lineHeight: 36,
      fontWeight: "700" as const
    },
    h2: {
      fontSize: 22,
      lineHeight: 28,
      fontWeight: "700" as const
    },
    h3: {
      fontSize: 18,
      lineHeight: 24,
      fontWeight: "600" as const
    },
    body: {
      fontSize: 15,
      lineHeight: 22,
      fontWeight: "400" as const
    },
    bodyBold: {
      fontSize: 15,
      lineHeight: 22,
      fontWeight: "600" as const
    },
    caption: {
      fontSize: 13,
      lineHeight: 18,
      fontWeight: "400" as const
    }
  }
} as const;
